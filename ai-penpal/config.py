"""
config.py
---------
All configuration for AI Penpal.
Edit this file to configure your deployment.
Never commit secrets — use environment variables in production.
"""

import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent


def _env_flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}

# ── OLLAMA ────────────────────────────────────────────────────────────────────
OLLAMA_HOST    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL   = os.getenv("OLLAMA_MODEL", "llama3.1:8b")   # Change to any installed model
DEFAULT_MODEL  = OLLAMA_MODEL
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava")
DEMO_FAST_MODEL = os.getenv("DEMO_FAST_MODEL", "").strip()
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
USE_ANTHROPIC_API = os.getenv("USE_ANTHROPIC_API", "false").lower() == "true"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600")) # seconds
DEFAULT_NUM_PREDICT = int(os.getenv("DEFAULT_NUM_PREDICT", "700"))
MATH_NUM_PREDICT = int(os.getenv("MATH_NUM_PREDICT", "250"))
DOCUMENT_NUM_PREDICT = int(os.getenv("DOCUMENT_NUM_PREDICT", "700"))
SUMMARY_NUM_PREDICT = int(os.getenv("SUMMARY_NUM_PREDICT", "500"))

# ── OUR DOMAIN ────────────────────────────────────────────────────────────────
OUR_EMAIL_DOMAIN = os.getenv("EMAIL_DOMAIN", "offlinellm.me")

# ── SMTP (195A) ───────────────────────────────────────────────────────────────
SMTP_HOST        = os.getenv("SMTP_HOST", "0.0.0.0")
SMTP_PORT        = int(os.getenv("SMTP_PORT", "8025"))   # 25 requires root; use 8025 in dev

# Outbound reply SMTP (Postfix relay or direct)
SMTP_REPLY_HOST  = os.getenv("SMTP_REPLY_HOST", "localhost")
SMTP_REPLY_PORT  = int(os.getenv("SMTP_REPLY_PORT", "25"))
SMTP_REPLY_USER  = os.getenv("SMTP_REPLY_USER", "")
SMTP_REPLY_PASS  = os.getenv("SMTP_REPLY_PASS", "")
SMTP_FROM_EMAIL  = os.getenv("SMTP_FROM_EMAIL", "offlinellmaipenpal@gmail.com")
LOCAL_TEST_MODE  = _env_flag("LOCAL_TEST_MODE", "false")
DISABLE_OUTBOUND_EMAIL = _env_flag("DISABLE_OUTBOUND_EMAIL", "false") or LOCAL_TEST_MODE

# ── DDD gRPC (195B) ───────────────────────────────────────────────────────────
DDD_GRPC_HOST    = os.getenv("DDD_GRPC_HOST", "localhost")
DDD_GRPC_PORT    = int(os.getenv("DDD_GRPC_PORT", "50051"))
DDD_APP_NAME     = os.getenv("DDD_APP_NAME", "ai-penpal")
DDD_OUR_URL      = os.getenv("DDD_OUR_URL", "http://localhost:8080")
DDD_RECEIVE_PORT = int(os.getenv("DDD_RECEIVE_PORT", "8080"))

# ── DATABASE ──────────────────────────────────────────────────────────────────
DB_PATH = os.getenv("AI_PENPAL_DB_PATH", str(BASE_DIR / "ai_penpal.db"))
MAX_HISTORY_EXCHANGES = int(os.getenv("MAX_HISTORY_EXCHANGES", "3"))
MAX_ATTACHMENT_HISTORY_EXCHANGES = int(os.getenv("MAX_ATTACHMENT_HISTORY_EXCHANGES", "1"))

# ── ATTACHMENTS ───────────────────────────────────────────────────────────────
SUPPORTED_ATTACHMENT_EXTENSIONS = {".pdf", ".docx", ".png", ".jpg", ".jpeg"}
MAX_ATTACHMENT_SIZE_BYTES = 10 * 1024 * 1024
MAX_EXTRACTED_CHARS_PER_ATTACHMENT = 15000
MAX_TOTAL_EXTRACTED_CHARS = 30000
MAX_ATTACHMENT_TEXT_CHARS = int(os.getenv("MAX_ATTACHMENT_TEXT_CHARS", "3000"))
MAX_RESUME_TEXT_CHARS = int(os.getenv("MAX_RESUME_TEXT_CHARS", "3000"))
ENABLE_IMAGE_OCR = True

# ── LOGGING ───────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ── DDD gRPC UPDATED (195B) ───────────────────────────────────────────────────
# Your gRPC server port (DDD connects TO you on this port)
DDD_OUR_GRPC_PORT = int(os.getenv("DDD_OUR_GRPC_PORT", "50052"))
# Your public gRPC URL that DDD will use to connect to you
# Example: "myserver.aipenpal.me:50052"
DDD_OUR_GRPC_URL  = os.getenv("DDD_OUR_GRPC_URL", "localhost:50052")
