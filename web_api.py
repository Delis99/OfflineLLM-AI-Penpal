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

OUR_DOMAIN = "offlinellm.me"
OUR_EMAIL  = f"ask@{OUR_DOMAIN}"

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


def get_db():
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            name TEXT,
            picture TEXT,
            google_sub TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            last_login_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


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
    now = datetime.utcnow().isoformat()
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

        session.clear()
        session.permanent = True
        session["user"] = {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "picture": user["picture"],
            "google_sub": user["google_sub"],
        }

        logger.info(f"[AUTH] Google login success for {user['email']}")
        return jsonify({"success": True, "user": serialize_user(user)})
    except RuntimeError as e:
        logger.error(f"[AUTH] Configuration error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.warning(f"[AUTH] Google login failed: {e}")
        return jsonify({"success": False, "error": "Google sign-in failed"}), 401


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
