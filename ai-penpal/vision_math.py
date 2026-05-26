"""
vision_math.py
--------------
Vision-based math extraction for image attachments using Ollama + LLaVA.
"""

import base64
import logging
from pathlib import Path
import re
import time

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    OLLAMA_HOST,
    OLLAMA_VISION_MODEL,
    USE_ANTHROPIC_API,
)
from math_validator import detect_math_expression, is_expression_suspicious

logger = logging.getLogger(__name__)

VISION_PROMPT = (
    "Look at this image and extract the exact mathematical expression.\n"
    "Return ONLY the expression, no explanation.\n"
    "Example output:\n"
    "6 ÷ 2(1+2)"
)


def extract_math_from_image(image_path: str) -> dict:
    resolved_path = str(Path(image_path))
    if not resolved_path:
        return {"expression": "", "confidence": "low"}

    logger.info(
        "[VISION] Attempting Anthropic vision (USE_ANTHROPIC_API=%s, key_set=%s, ANTHROPIC_AVAILABLE=%s)",
        USE_ANTHROPIC_API,
        bool(ANTHROPIC_API_KEY),
        ANTHROPIC_AVAILABLE,
    )
    try:
        anthropic_response = _extract_math_with_anthropic(resolved_path)
        if anthropic_response is not None:
            expression = _sanitize_expression(anthropic_response)
            if detect_math_expression(expression):
                confidence = "low" if is_expression_suspicious(expression) else "high"
                return {"expression": expression, "confidence": confidence}
            raise RuntimeError("invalid expression")
    except Exception as exc:
        logger.warning("[VISION] Anthropic vision failed: %s — falling back to Ollama", exc)

    response = _extract_math_with_ollama(resolved_path)
    expression = _sanitize_expression(_extract_response_text(response))
    if not detect_math_expression(expression):
        return {"expression": "", "confidence": "low"}

    confidence = "low" if is_expression_suspicious(expression) else "high"
    return {"expression": expression, "confidence": confidence}


def _extract_math_with_anthropic(image_path: str) -> str | None:
    if not (USE_ANTHROPIC_API and ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY):
        return None

    image_bytes = Path(image_path).read_bytes()
    media_type = _detect_media_type(image_bytes)
    if media_type not in {"image/png", "image/jpeg"}:
        raise RuntimeError(f"Unsupported image media type: {media_type}")
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    started_at = time.time()
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=128,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract any math expression from this image. "
                            "Return only the expression, nothing else."
                        ),
                    },
                ],
            }
        ],
    )
    elapsed = time.time() - started_at
    logger.info("[VISION] Anthropic vision in %.1fs", elapsed)
    return _extract_anthropic_response_text(message)


def _extract_math_with_ollama(resolved_path: str):
    try:
        from ollama import Client
    except ImportError:
        return None

    try:
        client = Client(host=OLLAMA_HOST)
        return client.chat(
            model=OLLAMA_VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": VISION_PROMPT,
                    "images": [resolved_path],
                }
            ],
            options={"temperature": 0},
        )
    except Exception:
        return None


def _extract_response_text(response) -> str:
    if response is None:
        return ""
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(response.get("response") or "")

    message = getattr(response, "message", None)
    if message is not None:
        return str(getattr(message, "content", "") or "")

    return str(getattr(response, "response", "") or "")


def _extract_anthropic_response_text(message) -> str:
    content_blocks = getattr(message, "content", None) or []
    parts = []
    for block in content_blocks:
        text = getattr(block, "text", None)
        if text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def _detect_media_type(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if image_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"
    return "image/png"


def _sanitize_expression(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    cleaned = cleaned.replace("```", "").strip().strip("`")
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

    for line in lines:
        candidate = re.sub(r"^(?:expression|math expression)\s*:\s*", "", line, flags=re.IGNORECASE)
        candidate = candidate.strip().strip("\"'")
        if detect_math_expression(candidate):
            return candidate.rstrip(".")

    condensed = re.sub(r"\s+", " ", cleaned)
    match = re.search(r"[\d(][0-9+\-*/=().xX×÷\[\]{}\s]+", condensed)
    if match:
        return match.group(0).strip().strip("\"'").rstrip(".")

    return ""
