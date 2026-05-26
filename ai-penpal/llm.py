"""
llm.py
------
Handles all communication with Ollama (local LLM).

Ollama runs locally on the server — no internet required.
Default model: llama3.2 (good balance of speed vs quality for this use case)
Swap model name in config.py to use any other Ollama-supported model.

API: http://localhost:11434/api/chat  (Ollama's OpenAI-compatible endpoint)
"""

import json
import time
import urllib.request
import urllib.error
from typing import Optional
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    anthropic = None
    ANTHROPIC_AVAILABLE = False

import config
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    DEFAULT_NUM_PREDICT,
    DEMO_FAST_MODEL,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    USE_ANTHROPIC_API,
)

import logging
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are OfflineLLM, an AI assistant that runs on a private server and
communicates entirely through email. You were built by Juan, Chloe, Barak,
and Akif as a senior design project at San Jose State University.

Your actual architecture:
- User sends an email to ask@offlinellm.me
- Postfix receives it and forwards to a custom Python SMTP server (port 8025)
- smtp_server.py parses the email, extracts attachments (PDF, images, TXT)
- processor.py handles conversation history, math detection, and routing
- For math expressions in images, vision_math.py extracts and solves them
  directly without calling the LLM
- For everything else, the LLM (you) generates the response
- database.py stores conversation history in SQLite keyed by email+subject
- The reply is sent back to the user via Postfix/SMTP
- The website at offlinellm.me shows a live conversation UI

When answering questions:
- Be concise and direct — you are replying via email
- Do not use excessive markdown headers or ASCII diagrams
- Write in plain paragraphs unless a list genuinely helps
- If asked how you work, describe YOUR actual pipeline above
- If asked about your team, mention Juan, Chloe, Barak, and Akif at SJSU"""


def get_active_model() -> str:
    return DEMO_FAST_MODEL or OLLAMA_MODEL


def _query_ollama_backend(
    prompt: str,
    history: list,
    num_predict: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
) -> Optional[str]:
    """
    Send a prompt to Ollama with conversation history.
    
    Args:
        prompt:  The user's current message
        history: List of prior exchanges [{"role": "user/assistant", "content": "..."}]
    
    Returns:
        LLM response string, or None if inference failed
    """
    model_name = get_active_model()
    selected_num_predict = num_predict if num_predict is not None else DEFAULT_NUM_PREDICT
    selected_timeout = timeout_seconds if timeout_seconds is not None else OLLAMA_TIMEOUT

    # Build message list: system prompt + history + new user message
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    logger.info("[LLM] Using model '%s' with num_predict=%s", model_name, selected_num_predict)

    payload = json.dumps({
        "model": model_name,
        "messages": messages,
        "stream": False,        # We want a single complete response
        "options": {
            "temperature": 0.7,
            "num_predict": selected_num_predict
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    started_at = time.time()
    try:
        with urllib.request.urlopen(req, timeout=selected_timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            response_text = result["message"]["content"]
            elapsed = time.time() - started_at
            logger.info("[LLM] Response generated in %.1fs (%s chars)", elapsed, len(response_text))
            return response_text

    except urllib.error.URLError as e:
        logger.error(f"[LLM] Ollama connection failed: {e}")
        return None
    except KeyError as e:
        logger.error(f"[LLM] Unexpected Ollama response format: {e}")
        return None
    except Exception as e:
        logger.error(f"[LLM] Unexpected error: {e}")
        return None


def _build_anthropic_prompt(prompt: str, history: list) -> str:
    if not history:
        return prompt

    parts = []
    for item in history:
        role = item.get("role", "user").capitalize()
        content = item.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append(f"User: {prompt}")
    return "\n\n".join(parts)


def _resolve_anthropic_enabled() -> bool:
    return bool(USE_ANTHROPIC_API or getattr(config, "USE_ANTHROPIC_API", False))


def _resolve_anthropic_api_key() -> str:
    return ANTHROPIC_API_KEY or getattr(config, "ANTHROPIC_API_KEY", "")


def _resolve_anthropic_model() -> str:
    return ANTHROPIC_MODEL or getattr(config, "ANTHROPIC_MODEL", "")


def generate_response(
    prompt: str,
    history: list,
    num_predict: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    force_ollama: bool = False,
) -> Optional[str]:
    anthropic_api_key = _resolve_anthropic_api_key()
    anthropic_model = _resolve_anthropic_model()

    use_anthropic = (
        not force_ollama
        and _resolve_anthropic_enabled()
        and ANTHROPIC_AVAILABLE
        and bool(anthropic_api_key)
    )

    if use_anthropic:
        anthropic_prompt = _build_anthropic_prompt(prompt, history)
        started_at = time.time()
        try:
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            message = client.messages.create(
                model=anthropic_model,
                max_tokens=num_predict or config.DEFAULT_NUM_PREDICT,
                messages=[{"role": "user", "content": anthropic_prompt}],
            )
            response_text = ""
            if getattr(message, "content", None):
                response_text = getattr(message.content[0], "text", "") or ""
            elapsed = time.time() - started_at
            logger.info("[LLM] Anthropic response in %.1fs (%s chars)", elapsed, len(response_text))
            return response_text
        except Exception as exc:
            logger.warning("[LLM] Anthropic API failed: %s — falling back to Ollama", exc)

    return _query_ollama_backend(prompt, history, num_predict=num_predict, timeout_seconds=timeout_seconds)


def query_ollama(
    prompt: str,
    history: list,
    num_predict: Optional[int] = None,
    timeout_seconds: Optional[int] = None,
    force_ollama: bool = False,
) -> Optional[str]:
    return generate_response(
        prompt,
        history,
        num_predict=num_predict,
        timeout_seconds=timeout_seconds,
        force_ollama=force_ollama,
    )


def is_ollama_available() -> bool:
    """Health check — verify Ollama is running before accepting messages."""
    try:
        model_name = get_active_model()
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/tags",
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if model_name not in models and not any(model_name in m for m in models):
                logger.warning(f"[LLM] Model '{model_name}' not found. Available: {models}")
                return False
            logger.info(f"[LLM] Ollama ready. Model '{model_name}' available.")
            return True
    except Exception as e:
        logger.error(f"[LLM] Ollama health check failed: {e}")
        return False


def prewarm_ollama_model() -> bool:
    """Send a short request to reduce first-token latency after startup."""
    return query_ollama("hi", [], num_predict=16, timeout_seconds=10, force_ollama=True) is not None
