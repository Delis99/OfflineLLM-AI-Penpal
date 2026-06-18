"""
web_api.py — AI Penpal Web API
Handles: send message, file attachments, history polling, clear history

Google Sign-In dependency:
    pip install google-auth
"""

import json
import base64
import logging
import os
import re
import secrets
import smtplib
import sqlite3
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from functools import wraps
from flask import Flask, request, jsonify, session, send_from_directory, abort
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash

try:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token
except ImportError:
    google_requests = None
    id_token = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
app.permanent_session_lifetime = timedelta(days=7)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_COOKIE_SECURE", "false").lower() == "true",
)

OUR_DOMAIN = os.getenv("EMAIL_DOMAIN", "offlinellm.me")
OUR_EMAIL = os.getenv("OUR_EMAIL", f"ask@{OUR_DOMAIN}")

# Google OAuth 2.0 Web Client ID from Google Cloud Console.
# Configure authorized JavaScript origins there:
#   http://localhost:5050
#   http://127.0.0.1:5050
#   https://offlinellm.me
# Set on the server, for example:
#   export GOOGLE_CLIENT_ID="YOUR_WEB_CLIENT_ID.apps.googleusercontent.com"
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()

DEFAULT_DB_PATH = (
    "/root/ai-penpal/ai_penpal.db"
    if os.path.isdir("/root/ai-penpal")
    else os.path.join(BASE_DIR, "ai_penpal.db")
)
DB_PATH = os.environ.get("AI_PENPAL_DB_PATH", DEFAULT_DB_PATH)
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
MIN_PASSWORD_LENGTH = 8
VERIFICATION_CODE_TTL_MINUTES = 10
MAX_VERIFICATION_ATTEMPTS = 5
SMTP_REPLY_HOST = os.getenv("SMTP_REPLY_HOST", "localhost")
SMTP_REPLY_PORT = int(os.getenv("SMTP_REPLY_PORT", "25"))
SMTP_REPLY_USER = os.getenv("SMTP_REPLY_USER", "")
SMTP_REPLY_PASS = os.getenv("SMTP_REPLY_PASS", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", "offlinellmaipenpal@gmail.com")
DISABLE_OUTBOUND_EMAIL = os.getenv("DISABLE_OUTBOUND_EMAIL", "false").lower() in {"1", "true", "yes", "on"}


def _utc_now_isoformat():
    return datetime.utcnow().isoformat()


def _parse_utc_isoformat(value):
    return datetime.fromisoformat(value)


def generate_verification_code():
    return f"{secrets.randbelow(1_000_000):06d}"


def get_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _create_users_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            name TEXT,
            picture TEXT,
            google_sub TEXT UNIQUE,
            password_hash TEXT,
            created_at TEXT NOT NULL,
            last_login_at TEXT NOT NULL
        )
        """
    )


def _create_email_verifications_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS email_verifications (
            email TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            code_hash TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def _ensure_users_table(conn):
    columns = conn.execute("PRAGMA table_info(users)").fetchall()
    if not columns:
        _create_users_table(conn)
        return

    column_names = {column["name"] for column in columns}
    google_sub_column = next((column for column in columns if column["name"] == "google_sub"), None)
    google_sub_is_required = bool(google_sub_column and google_sub_column["notnull"])
    needs_rebuild = google_sub_is_required or "password_hash" not in column_names

    if not needs_rebuild:
        return

    rows = [dict(row) for row in conn.execute("SELECT * FROM users").fetchall()]
    conn.execute("ALTER TABLE users RENAME TO users_legacy")
    _create_users_table(conn)

    for row in rows:
        conn.execute(
            """
            INSERT INTO users (
                id, email, name, picture, google_sub, password_hash, created_at, last_login_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.get("id"),
                row.get("email"),
                row.get("name", ""),
                row.get("picture", ""),
                row.get("google_sub"),
                row.get("password_hash"),
                row.get("created_at") or _utc_now_isoformat(),
                row.get("last_login_at") or _utc_now_isoformat(),
            ),
        )

    conn.execute("DROP TABLE users_legacy")


def _ensure_email_verifications_table(conn):
    _create_email_verifications_table(conn)


def init_db():
    conn = get_db()
    _ensure_users_table(conn)
    _ensure_email_verifications_table(conn)
    conn.commit()
    conn.close()


def normalize_email(email):
    return (email or "").strip().lower()


def validate_email(email):
    return bool(EMAIL_RE.match(email))


def session_payload(user):
    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "picture": user["picture"],
        "google_sub": user["google_sub"],
    }


def start_user_session(user):
    session.clear()
    session.permanent = True
    session["user"] = session_payload(user)


def serialize_user(row):
    return {
        "id": row["id"],
        "email": row["email"],
        "name": row["name"],
        "picture": row["picture"],
    }


def current_user():
    return session.get("user")


def current_user_email():
    user = current_user()
    return user.get("email") if user else None


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user_email():
            return jsonify({"success": False, "error": "Authentication required"}), 401
        return fn(*args, **kwargs)

    return wrapper


def verify_google_credential(credential):
    if not GOOGLE_CLIENT_ID:
        raise RuntimeError("GOOGLE_CLIENT_ID is not configured")
    if id_token is None or google_requests is None:
        raise RuntimeError("google-auth is not installed. Run: pip install google-auth")

    token_info = id_token.verify_oauth2_token(
        credential,
        google_requests.Request(),
        GOOGLE_CLIENT_ID,
        clock_skew_in_seconds=10,
    )

    if token_info.get("aud") != GOOGLE_CLIENT_ID:
        raise ValueError("Token audience does not match GOOGLE_CLIENT_ID")

    email = (token_info.get("email") or "").strip().lower()
    if not email:
        raise ValueError("Google token did not include an email address")

    email_verified = token_info.get("email_verified")
    if email_verified not in (True, "true", "True", "1", 1):
        raise ValueError("Google email is not verified")

    google_sub = token_info.get("sub")
    if not google_sub:
        raise ValueError("Google token did not include a subject")

    return {
        "email": email,
        "name": token_info.get("name", ""),
        "picture": token_info.get("picture", ""),
        "google_sub": google_sub,
    }


def upsert_user(profile):
    now = _utc_now_isoformat()
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM users WHERE google_sub = ?",
            (profile["google_sub"],),
        ).fetchone()

        if row:
            conn.execute(
                """
                UPDATE users
                SET email = ?, name = ?, picture = ?, last_login_at = ?
                WHERE id = ?
                """,
                (profile["email"], profile["name"], profile["picture"], now, row["id"]),
            )
        else:
            email_row = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                (profile["email"],),
            ).fetchone()
            if email_row and email_row["google_sub"] != profile["google_sub"]:
                if not email_row["google_sub"]:
                    conn.execute(
                        """
                        UPDATE users
                        SET name = ?, picture = ?, google_sub = ?, last_login_at = ?
                        WHERE id = ?
                        """,
                        (
                            profile["name"],
                            profile["picture"],
                            profile["google_sub"],
                            now,
                            email_row["id"],
                        ),
                    )
                    conn.commit()
                    user = conn.execute(
                        "SELECT * FROM users WHERE id = ?",
                        (email_row["id"],),
                    ).fetchone()
                    return user
                raise ValueError("This email is already linked to another Google account")

            if email_row:
                conn.execute(
                    """
                    UPDATE users
                    SET name = ?, picture = ?, google_sub = ?, last_login_at = ?
                    WHERE id = ?
                    """,
                    (
                        profile["name"],
                        profile["picture"],
                        profile["google_sub"],
                        now,
                        email_row["id"],
                    ),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO users (email, name, picture, google_sub, created_at, last_login_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        profile["email"],
                        profile["name"],
                        profile["picture"],
                        profile["google_sub"],
                        now,
                        now,
                    ),
                )

        conn.commit()
        user = conn.execute(
            "SELECT * FROM users WHERE google_sub = ?",
            (profile["google_sub"],),
        ).fetchone()
        return user
    finally:
        conn.close()


def create_password_user_with_hash(email, password_hash):
    init_db()
    now = _utc_now_isoformat()
    name = email.split("@", 1)[0]
    conn = get_db()
    try:
        existing = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            if existing["password_hash"]:
                raise ValueError("An account already exists for this email. Sign in instead.")
            raise ValueError("This email already uses Google sign-in.")

        conn.execute(
            """
            INSERT INTO users (email, name, picture, google_sub, password_hash, created_at, last_login_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                email,
                name,
                "",
                None,
                password_hash,
                now,
                now,
            ),
        )
        conn.commit()
        return conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    finally:
        conn.close()


def create_password_user(email, password):
    return create_password_user_with_hash(email, generate_password_hash(password))


def authenticate_password_user(email, password):
    init_db()
    conn = get_db()
    try:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if not user or not user["password_hash"]:
            return None
        if not check_password_hash(user["password_hash"], password):
            return None
        conn.execute(
            "UPDATE users SET last_login_at = ? WHERE id = ?",
            (_utc_now_isoformat(), user["id"]),
        )
        conn.commit()
        return conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
    finally:
        conn.close()


def _verification_email_body(code):
    return (
        "Your AI Penpal verification code is:\n\n"
        f"{code}\n\n"
        f"This code expires in {VERIFICATION_CODE_TTL_MINUTES} minutes. "
        "If you did not request this account, you can ignore this email."
    )


def send_verification_email(email, code):
    msg = MIMEText(_verification_email_body(code), "plain", "utf-8")
    msg["From"] = SMTP_FROM_EMAIL
    msg["To"] = email
    msg["Subject"] = "Your AI Penpal verification code"

    if DISABLE_OUTBOUND_EMAIL:
        logger.info(f"[AUTH] Email verification code for {email}: {code}")
        return

    with smtplib.SMTP(SMTP_REPLY_HOST, SMTP_REPLY_PORT, timeout=15) as smtp:
        if SMTP_REPLY_USER:
            smtp.starttls()
            smtp.login(SMTP_REPLY_USER, SMTP_REPLY_PASS)
        smtp.sendmail(SMTP_FROM_EMAIL, [email], msg.as_string())


def request_signup_verification(email, password):
    init_db()
    code = generate_verification_code()
    now = datetime.utcnow()
    expires_at = now + timedelta(minutes=VERIFICATION_CODE_TTL_MINUTES)
    password_hash = generate_password_hash(password)
    code_hash = generate_password_hash(code)

    conn = get_db()
    try:
        existing = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            if existing["password_hash"]:
                raise ValueError("An account already exists for this email. Sign in instead.")
            raise ValueError("This email already uses Google sign-in.")

        conn.execute(
            """
            INSERT INTO email_verifications (
                email, password_hash, code_hash, attempts, expires_at, created_at, updated_at
            )
            VALUES (?, ?, ?, 0, ?, ?, ?)
            ON CONFLICT(email) DO UPDATE SET
                password_hash = excluded.password_hash,
                code_hash = excluded.code_hash,
                attempts = 0,
                expires_at = excluded.expires_at,
                updated_at = excluded.updated_at
            """,
            (
                email,
                password_hash,
                code_hash,
                expires_at.isoformat(),
                now.isoformat(),
                now.isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    send_verification_email(email, code)


def verify_signup_code(email, code):
    init_db()
    code = (code or "").strip()
    conn = get_db()
    try:
        row = conn.execute(
            "SELECT * FROM email_verifications WHERE email = ?",
            (email,),
        ).fetchone()
        if not row:
            raise ValueError("No verification code was requested for this email.")

        if row["attempts"] >= MAX_VERIFICATION_ATTEMPTS:
            conn.execute("DELETE FROM email_verifications WHERE email = ?", (email,))
            conn.commit()
            raise ValueError("Too many incorrect codes. Request a new code.")

        if datetime.utcnow() > _parse_utc_isoformat(row["expires_at"]):
            conn.execute("DELETE FROM email_verifications WHERE email = ?", (email,))
            conn.commit()
            raise ValueError("Verification code expired. Request a new code.")

        if not check_password_hash(row["code_hash"], code):
            conn.execute(
                "UPDATE email_verifications SET attempts = attempts + 1, updated_at = ? WHERE email = ?",
                (_utc_now_isoformat(), email),
            )
            conn.commit()
            raise ValueError("Incorrect verification code.")

        password_hash = row["password_hash"]
        conn.execute("DELETE FROM email_verifications WHERE email = ?", (email,))
        conn.commit()
    finally:
        conn.close()

    return create_password_user_with_hash(email, password_hash)


@app.route("/api/config", methods=["GET"])
def config():
    return jsonify({"success": True, "googleClientId": GOOGLE_CLIENT_ID})


@app.route("/api/auth/google", methods=["POST"])
def google_auth():
    try:
        data = request.get_json() or {}
        credential = data.get("credential", "").strip()
        if not credential:
            return jsonify({"success": False, "error": "Missing Google credential"}), 400

        init_db()
        profile = verify_google_credential(credential)
        user = upsert_user(profile)

        start_user_session(user)

        logger.info(f"[AUTH] Google login success for {user['email']}")
        return jsonify({"success": True, "user": serialize_user(user)})
    except RuntimeError as e:
        logger.error(f"[AUTH] Configuration error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.warning(f"[AUTH] Google login failed: {e}")
        return jsonify({"success": False, "error": "Google sign-in failed"}), 401


@app.route("/api/auth/password/signup", methods=["POST"])
def password_signup():
    try:
        data = request.get_json() or {}
        email = normalize_email(data.get("email", ""))
        password = data.get("password", "")
        confirm_password = data.get("confirmPassword", "")

        if not validate_email(email):
            return jsonify({"success": False, "error": "Enter a valid email address."}), 400
        if len(password) < MIN_PASSWORD_LENGTH:
            return jsonify(
                {
                    "success": False,
                    "error": f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
                }
            ), 400
        if password != confirm_password:
            return jsonify({"success": False, "error": "Passwords do not match."}), 400

        request_signup_verification(email, password)

        logger.info(f"[AUTH] Verification code sent to {email}")
        return jsonify(
            {
                "success": True,
                "needsVerification": True,
                "email": email,
                "message": "Verification code sent.",
            }
        )
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 409
    except smtplib.SMTPException as e:
        logger.error(f"[AUTH] Verification email failed for {email}: {e}")
        return jsonify({"success": False, "error": "Could not send verification email."}), 502
    except Exception as e:
        logger.error(f"[AUTH] Password signup failed: {e}")
        return jsonify({"success": False, "error": "Could not create account."}), 500


@app.route("/api/auth/password/verify", methods=["POST"])
def password_verify():
    try:
        data = request.get_json() or {}
        email = normalize_email(data.get("email", ""))
        code = data.get("code", "")

        if not validate_email(email):
            return jsonify({"success": False, "error": "Enter a valid email address."}), 400
        if not re.fullmatch(r"\d{6}", code.strip()):
            return jsonify({"success": False, "error": "Enter the 6-digit verification code."}), 400

        user = verify_signup_code(email, code)
        start_user_session(user)

        logger.info(f"[AUTH] Password account verified for {user['email']}")
        return jsonify({"success": True, "user": serialize_user(user)})
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    except Exception as e:
        logger.error(f"[AUTH] Password verification failed: {e}")
        return jsonify({"success": False, "error": "Could not verify account."}), 500


@app.route("/api/auth/password/login", methods=["POST"])
def password_login():
    try:
        data = request.get_json() or {}
        email = normalize_email(data.get("email", ""))
        password = data.get("password", "")

        if not validate_email(email) or not password:
            return jsonify({"success": False, "error": "Enter your email and password."}), 400

        user = authenticate_password_user(email, password)
        if not user:
            return jsonify({"success": False, "error": "Invalid email or password."}), 401

        start_user_session(user)

        logger.info(f"[AUTH] Password login success for {user['email']}")
        return jsonify({"success": True, "user": serialize_user(user)})
    except Exception as e:
        logger.error(f"[AUTH] Password login failed: {e}")
        return jsonify({"success": False, "error": "Could not sign in."}), 500


@app.route("/api/me", methods=["GET"])
def me():
    user = current_user()
    if not user:
        return jsonify({"success": False, "user": None}), 401
    return jsonify(
        {
            "success": True,
            "user": {
                "id": user.get("id"),
                "email": user.get("email"),
                "name": user.get("name"),
                "picture": user.get("picture"),
            },
        }
    )


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})


@app.route("/api/send", methods=["POST"])
@require_auth
def send_message():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "Invalid request"}), 400

        from_email = current_user_email()
        subject    = data.get("subject", "Message from AI Penpal website").strip()
        body       = data.get("body", "").strip()
        files      = data.get("files", [])

        if not body:
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        if files:
            # Send as multipart with attachments
            msg = MIMEMultipart()
            msg["From"]     = from_email
            msg["To"]       = OUR_EMAIL
            msg["Subject"]  = subject
            msg["Reply-To"] = from_email
            msg.attach(MIMEText(body, "plain", "utf-8"))

            for f in files:
                fname = f.get("name", "file")
                ftype = f.get("type", "txt")
                fdata = f.get("data", "")

                try:
                    if ftype == "pdf" and fdata.startswith("data:"):
                        # Strip data URL prefix
                        raw = base64.b64decode(fdata.split(",", 1)[1])
                    elif ftype == "txt":
                        raw = fdata.encode("utf-8")
                    else:
                        raw = fdata.encode("utf-8")

                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(raw)
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
                    msg.attach(part)
                    logger.info(f"[API] Attached file: {fname}")
                except Exception as e:
                    logger.error(f"[API] Failed to attach {fname}: {e}")
        else:
            # Simple text email
            msg = MIMEText(body, "plain", "utf-8")
            msg["From"]     = from_email
            msg["To"]       = OUR_EMAIL
            msg["Subject"]  = subject
            msg["Reply-To"] = from_email

        with smtplib.SMTP("localhost", 25) as smtp:
            smtp.sendmail(from_email, [OUR_EMAIL], msg.as_string())

        logger.info(f"[API] Message from {from_email} sent with {len(files)} attachment(s)")
        return jsonify({"success": True})

    except smtplib.SMTPException as e:
        logger.error(f"[API] SMTP error: {e}")
        return jsonify({"success": False, "error": "Mail server error. Please try again."}), 500
    except Exception as e:
        logger.error(f"[API] Error: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/history", methods=["GET"])
@require_auth
def get_history():
    requested_email = request.args.get("email", "").strip().lower()
    email = current_user_email()
    if requested_email and requested_email != email:
        return jsonify({"success": False, "error": "Forbidden"}), 403
    try:
        conn = get_db()
        row = conn.execute(
            "SELECT history, updated_at FROM sessions WHERE transport_id = ?",
            (email,)
        ).fetchone()
        conn.close()
        if not row:
            return jsonify({"success": True, "messages": [], "updatedAt": None})
        history = json.loads(row["history"])
        return jsonify({"success": True, "messages": history, "updatedAt": row["updated_at"]})
    except Exception as e:
        logger.error(f"[API] History error: {e}")
        return jsonify({"success": False, "error": "Could not fetch history"}), 500


@app.route("/api/clear-history", methods=["POST"])
@require_auth
def clear_history():
    try:
        data = request.get_json() or {}
        requested_email = data.get("email", "").strip().lower()
        email = current_user_email()
        if requested_email and requested_email != email:
            return jsonify({"success": False, "error": "Forbidden"}), 403
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE transport_id = ?", (email,))
        conn.commit()
        conn.close()
        logger.info(f"[API] History cleared for {email}")
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"[API] Clear history error: {e}")
        return jsonify({"success": False, "error": "Could not clear history"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "domain": OUR_DOMAIN})


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def frontend_file(filename):
    if filename.startswith("api/") or filename == "health":
        abort(404)
    path = os.path.join(BASE_DIR, filename)
    if not os.path.isfile(path):
        abort(404)
    return send_from_directory(BASE_DIR, filename)


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", "5050"))
    app.run(host="0.0.0.0", port=port, debug=False)
