"""
web_api.py — AI Penpal Web API
Handles: Google auth, send message, file attachments, history polling,
clear history, health checks, and optional frontend serving.
"""

import base64
import json
import logging
import os
import smtplib
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import wraps
from pathlib import Path

from flask import abort, jsonify, request, send_from_directory, session, Flask
from flask_cors import CORS

from config import DB_PATH as CONFIG_DB_PATH, DEMO_MODE
from llm import get_active_model, is_ollama_available

try:
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token
except ImportError as google_auth_import_error:
    google_requests = None
    id_token = None
    GOOGLE_AUTH_IMPORT_ERROR = google_auth_import_error
else:
    GOOGLE_AUTH_IMPORT_ERROR = None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
app.permanent_session_lifetime = timedelta(days=7)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.environ.get("FLASK_COOKIE_SECURE", "false").lower() == "true",
)

APP_START_TIME = time.time()

OUR_DOMAIN = "offlinellm.me"
OUR_EMAIL = f"ask@{OUR_DOMAIN}"
DB_PATH = CONFIG_DB_PATH
API_PORT = int(os.getenv("API_PORT", os.getenv("PORT", "5050")))

# Google OAuth 2.0 Web Client ID from Google Cloud Console.
# Authorized JavaScript origins should include:
#   http://localhost:5050
#   http://127.0.0.1:5050
#   https://offlinellm.me
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
STATIC_FILE_EXTENSIONS = {
    ".css",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".map",
    ".png",
    ".svg",
    ".ttf",
    ".webp",
    ".woff",
    ".woff2",
}
BLOCKED_STATIC_PATH_PARTS = {
    ".git",
    "__pycache__",
    "ai-penpal",
    "offline-LLM",
    "venv",
    ".venv",
}

_db_initialized = False


def _candidate_frontend_dirs() -> list[Path]:
    configured = os.environ.get("FRONTEND_DIR", "").strip()
    candidates = []
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.extend(
        [
            BASE_DIR / "ai-penpal-website",
            BASE_DIR.parent / "ai-penpal-website",
            BASE_DIR.parent,
            BASE_DIR,
        ]
    )
    return candidates


def _resolve_frontend_dir() -> Path | None:
    for candidate in _candidate_frontend_dirs():
        resolved = candidate.resolve()
        if (resolved / "index.html").is_file():
            return resolved
    return None


FRONTEND_DIR = _resolve_frontend_dir()


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_db():
    if DB_PATH != ":memory:":
        db_dir = os.path.dirname(DB_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize conversation tables and the Google-auth user table."""
    global _db_initialized
    if _db_initialized:
        return

    import database

    database.DB_PATH = DB_PATH
    database.init_db()

    conn = get_db()
    try:
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
    finally:
        conn.close()

    _db_initialized = True


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
    email = user.get("email") if user else ""
    return email.strip().lower()


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
        raise RuntimeError(
            "Google auth dependencies are not installed. "
            "Run: pip install google-auth requests"
        ) from GOOGLE_AUTH_IMPORT_ERROR

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
    init_db()
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


def _build_message(from_email: str, subject: str, body: str, files: list):
    if files:
        msg = MIMEMultipart()
        msg.attach(MIMEText(body, "plain", "utf-8"))

        for f in files:
            fname = f.get("name", "file")
            ftype = f.get("type", "txt")
            fdata = f.get("data", "")

            try:
                if isinstance(fdata, str) and fdata.startswith("data:"):
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
                logger.info("[API] Attached file: %s", fname)
            except Exception as e:
                logger.error("[API] Failed to attach %s: %s", fname, e)
    else:
        msg = MIMEText(body, "plain", "utf-8")

    msg["From"] = from_email
    msg["To"] = OUR_EMAIL
    msg["Subject"] = subject
    msg["Reply-To"] = from_email
    return msg


@app.route("/api/config", methods=["GET"])
def api_config():
    return jsonify({"success": True, "googleClientId": GOOGLE_CLIENT_ID})


@app.route("/api/auth/google", methods=["GET", "POST"])
def google_auth():
    if request.method == "GET":
        return jsonify({"success": False, "error": "Use POST with a Google credential"}), 405

    try:
        data = request.get_json() or {}
        credential = data.get("credential", "").strip()
        if not credential:
            return jsonify({"success": False, "error": "Missing Google credential"}), 400

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

        logger.info("[AUTH] Google login success for %s", user["email"])
        return jsonify({"success": True, "user": serialize_user(user)})
    except RuntimeError as e:
        logger.error("[AUTH] Configuration error: %s", e)
        return jsonify({"success": False, "error": str(e)}), 500
    except Exception as e:
        logger.warning("[AUTH] Google login failed: %s", e)
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
        subject = data.get("subject", "Message from AI Penpal website").strip()
        body = data.get("body", "").strip()
        files = data.get("files", [])

        if not body:
            return jsonify({"success": False, "error": "Missing required fields"}), 400

        msg = _build_message(from_email, subject, body, files)

        with smtplib.SMTP("localhost", 25) as smtp:
            smtp.sendmail(from_email, [OUR_EMAIL], msg.as_string())

        logger.info("[API] Message from %s sent with %s attachment(s)", from_email, len(files))
        return jsonify({"success": True})

    except smtplib.SMTPException as e:
        logger.error("[API] SMTP error: %s", e)
        return jsonify({"success": False, "error": "Mail server error. Please try again."}), 500
    except Exception as e:
        logger.error("[API] Error: %s", e)
        return jsonify({"success": False, "error": "Internal server error"}), 500


@app.route("/api/history", methods=["GET"])
@require_auth
def get_history():
    requested_email = request.args.get("email", "").strip().lower()
    email = current_user_email()
    if requested_email and requested_email != email:
        return jsonify({"success": False, "error": "Forbidden"}), 403

    try:
        init_db()
        conn = get_db()
        row = conn.execute(
            "SELECT history, updated_at FROM sessions WHERE transport_id = ?",
            (email,),
        ).fetchone()
        conn.close()
        if not row:
            return jsonify({"success": True, "messages": [], "updatedAt": None})
        history = json.loads(row["history"])
        return jsonify({"success": True, "messages": history, "updatedAt": row["updated_at"]})
    except Exception as e:
        logger.error("[API] History error: %s", e)
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

        init_db()
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE transport_id = ?", (email,))
        conn.commit()
        conn.close()
        logger.info("[API] History cleared for %s", email)
        return jsonify({"success": True})
    except Exception as e:
        logger.error("[API] Clear history error: %s", e)
        return jsonify({"success": False, "error": "Could not clear history"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "domain": OUR_DOMAIN})


@app.route("/api/health", methods=["GET"])
def api_health():
    ollama_available = is_ollama_available()
    return jsonify(
        {
            "status": "ok",
            "model": get_active_model(),
            "demo_mode": DEMO_MODE,
            "uptime_seconds": int(max(0, time.time() - APP_START_TIME)),
            "ollama_status": "ok" if ollama_available else "unavailable",
        }
    )


@app.route("/", methods=["GET"])
def index():
    if FRONTEND_DIR:
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({"status": "ok", "service": "AI Penpal API"})


@app.route("/<path:filename>", methods=["GET"])
def frontend_file(filename):
    if filename.startswith("api/") or filename == "health":
        abort(404)
    if not FRONTEND_DIR:
        abort(404)

    requested = Path(filename)
    if (
        requested.is_absolute()
        or ".." in requested.parts
        or any(part.startswith(".") for part in requested.parts)
        or any(part in BLOCKED_STATIC_PATH_PARTS for part in requested.parts)
        or requested.suffix.lower() not in STATIC_FILE_EXTENSIONS
    ):
        abort(404)

    frontend_root = FRONTEND_DIR.resolve()
    path = (frontend_root / requested).resolve()
    try:
        path.relative_to(frontend_root)
    except ValueError:
        abort(404)

    if not path.is_file():
        abort(404)
    return send_from_directory(frontend_root, str(requested))


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=API_PORT, debug=False)
