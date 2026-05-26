"""
main.py
-------
Entry point for AI Penpal server.

Usage:
  # 195A mode (SMTP transport)
  python main.py --mode smtp

  # 195B mode (DDD gRPC transport)  
  python main.py --mode ddd

  # Test mode (process a single message from command line)
  python main.py --mode test --from user@example.com --subject "Hello" --body "What is Python?"
"""

import argparse
import logging
import signal
import sys
import time

import config
from database import init_db
from llm import is_ollama_available, prewarm_ollama_model, query_ollama
from config import DEMO_MODE, LOG_LEVEL

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def _log_graceful_shutdown():
    logger.info("[MAIN] Shutting down gracefully...")


def _verify_ollama_prewarm() -> bool:
    started_at = time.time()
    response = query_ollama("ping", [], num_predict=16, timeout_seconds=10, force_ollama=True)
    elapsed = time.time() - started_at
    if response and response.strip():
        logger.info("[MAIN] Ollama pre-warm verification complete in %.1fs", elapsed)
        return True
    logger.warning("[MAIN] Ollama pre-warm verification failed in %.1fs", elapsed)
    return False


def run_smtp_mode():
    """195A: Run with SMTP transport (aiosmtpd)."""
    from smtp_server import run_smtp_server
    controller = run_smtp_server()
    logger.info("[MAIN] AI Penpal running in SMTP mode. Press Ctrl+C to stop.")
    shutdown_requested = False

    def _handle_shutdown(signum, frame):
        del signum, frame
        nonlocal shutdown_requested
        shutdown_requested = True
        _log_graceful_shutdown()

    previous_sigint = signal.getsignal(signal.SIGINT)
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)
    try:
        while not shutdown_requested:
            time.sleep(1)
    except KeyboardInterrupt:
        _log_graceful_shutdown()
    finally:
        controller.stop()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def run_ddd_mode():
    """195B: Run with DDD gRPC transport."""
    from ddd_adapter import run_ddd_adapter
    logger.info("[MAIN] AI Penpal running in DDD mode.")
    run_ddd_adapter()


def run_test_mode(from_addr: str, subject: str, body: str):
    """Dev/test: Process a single message directly."""
    from processor import process_message
    logger.info(f"[TEST] Processing message from {from_addr}")
    result = process_message(from_addr, subject, body)
    print("\n" + "="*60)
    print(f"SUCCESS: {result['success']}")
    print(f"SUBJECT: {result['reply_subject']}")
    print(f"BODY:\n{result['reply_body']}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="AI Penpal Server")
    parser.add_argument(
        "--mode",
        choices=["smtp", "ddd", "test"],
        default="smtp",
        help="Transport mode: smtp (195A) or ddd (195B) or test"
    )
    parser.add_argument("--from", dest="from_addr", default="test@example.com")
    parser.add_argument("--subject", default="Test message")
    parser.add_argument("--body", default="Hello, can you help me?")
    args = parser.parse_args()

    # Initialize database
    init_db()
    logger.info("[MAIN] Demo mode: %s", "enabled" if DEMO_MODE else "disabled")

    # Check Ollama is available
    logger.info("[MAIN] Checking Ollama availability...")
    ollama_available = is_ollama_available()
    if not ollama_available:
        logger.warning("[MAIN] Ollama not available — LLM responses will fail until it starts")
        if not (
            config.USE_ANTHROPIC_API
            and config.ANTHROPIC_API_KEY
            and args.mode != "test"
        ):
            pass
        elif args.mode != "test":
            logger.warning("[MAIN] Anthropic is enabled; continuing without Ollama fallback")
        if args.mode != "test" and not (config.USE_ANTHROPIC_API and config.ANTHROPIC_API_KEY):
            logger.error("[MAIN] Please start Ollama before running the server: ollama serve")
            sys.exit(1)
    else:
        if config.USE_ANTHROPIC_API:
            if not config.ANTHROPIC_API_KEY:
                logger.warning("[MAIN] USE_ANTHROPIC_API=true but ANTHROPIC_API_KEY is not set — will use Ollama")
            else:
                logger.info(f"[MAIN] LLM backend: Anthropic {config.ANTHROPIC_MODEL}")
        else:
            logger.info(f"[MAIN] LLM backend: Ollama {config.DEFAULT_MODEL}")
        if not config.USE_ANTHROPIC_API:
            logger.info("[MAIN] Pre-warming Ollama model...")
            try:
                if prewarm_ollama_model():
                    logger.info("[MAIN] Ollama pre-warm complete")
                    try:
                        _verify_ollama_prewarm()
                    except Exception as exc:
                        logger.warning("[MAIN] Ollama pre-warm verification failed: %s", exc)
                else:
                    logger.warning("[MAIN] Ollama pre-warm failed")
            except Exception as exc:
                logger.warning("[MAIN] Ollama pre-warm failed: %s", exc)
        else:
            logger.info("[MAIN] Skipping Ollama pre-warm (Anthropic API is primary backend)")
    if not ollama_available:
        if config.USE_ANTHROPIC_API:
            if not config.ANTHROPIC_API_KEY:
                logger.warning("[MAIN] USE_ANTHROPIC_API=true but ANTHROPIC_API_KEY is not set — will use Ollama")
            else:
                logger.info(f"[MAIN] LLM backend: Anthropic {config.ANTHROPIC_MODEL}")
        else:
            logger.info(f"[MAIN] LLM backend: Ollama {config.DEFAULT_MODEL}")

    # Start in selected mode
    if args.mode == "smtp":
        run_smtp_mode()
    elif args.mode == "ddd":
        run_ddd_mode()
    elif args.mode == "test":
        run_test_mode(args.from_addr, args.subject, args.body)


if __name__ == "__main__":
    main()
