#!/bin/bash
# Run this on your server: bash fix_server.sh
# Fixes: history isolation per conversation + file reading

echo "Installing pypdf2..."
pip install pypdf2 --break-system-packages --quiet

echo "Patching smtp_server.py..."
cat > /root/ai-penpal/smtp_server.py << 'PYEOF'
import asyncio
import email
import logging
import smtplib
import io
from email.mime.text import MIMEText
from aiosmtpd.controller import Controller

try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

from processor import process_message
from database import queue_incoming
from config import (
    SMTP_HOST, SMTP_PORT,
    SMTP_REPLY_HOST, SMTP_REPLY_PORT,
    SMTP_REPLY_USER, SMTP_REPLY_PASS,
    OUR_EMAIL_DOMAIN
)

logger = logging.getLogger(__name__)


class EmailHandler:

    async def handle_RCPT(self, server, session, envelope, address, rcpt_options):
        envelope.rcpt_tos.append(address)
        return "250 OK"

    async def handle_DATA(self, server, session, envelope):
        try:
            msg = email.message_from_bytes(envelope.content)
            sender = envelope.mail_from

            if not sender or sender == '<>' or sender.startswith('MAILER-DAEMON'):
                return "250 OK"

            subject = msg.get("Subject", "(no subject)")
            body = _extract_body(msg)

            # Extract text from PDF/TXT attachments
            attachment_text = _extract_attachments(msg)
            if attachment_text:
                body = body + "\n\n" + attachment_text
                logger.info(f"[SMTP] Extracted attachment content ({len(attachment_text)} chars)")

            logger.info(f"[SMTP] Received email from {sender} | Subject: {subject}")

            queue_incoming(sender, subject, body)
            result = process_message(sender, subject, body)

            if result["success"]:
                _send_reply(sender, result["reply_subject"], result["reply_body"])
                logger.info(f"[SMTP] Reply sent to {sender}")
            else:
                logger.error(f"[SMTP] Processing failed for {sender}: {result.get('error')}")
                _send_reply(sender, result["reply_subject"], result["reply_body"])

        except Exception as e:
            logger.exception(f"[SMTP] Unhandled error: {e}")

        return "250 Message accepted for delivery"


def _extract_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", ""))
            if content_type == "text/plain" and "attachment" not in disposition:
                charset = part.get_content_charset() or "utf-8"
                return part.get_payload(decode=True).decode(charset, errors="replace")
    else:
        charset = msg.get_content_charset() or "utf-8"
        return msg.get_payload(decode=True).decode(charset, errors="replace")
    return ""


def _extract_attachments(msg) -> str:
    extracted = []
    for part in msg.walk():
        disposition = str(part.get("Content-Disposition", ""))
        if "attachment" not in disposition:
            continue
        filename = part.get_filename() or ""
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        if filename.lower().endswith('.txt'):
            try:
                text = payload.decode('utf-8', errors='replace')
                extracted.append(f"[Attached file: {filename}]\n{text[:8000]}")
                logger.info(f"[SMTP] Read TXT: {filename}")
            except Exception as e:
                logger.warning(f"Could not read TXT {filename}: {e}")
        elif filename.lower().endswith('.pdf'):
            if PDF_SUPPORT:
                try:
                    reader = PyPDF2.PdfReader(io.BytesIO(payload))
                    pages_text = []
                    for page in reader.pages[:10]:
                        pages_text.append(page.extract_text() or "")
                    text = "\n".join(pages_text)[:8000]
                    extracted.append(f"[Attached PDF: {filename}]\n{text}")
                    logger.info(f"[SMTP] Read PDF: {filename} ({len(text)} chars)")
                except Exception as e:
                    logger.warning(f"Could not read PDF {filename}: {e}")
            else:
                extracted.append(f"[PDF attached: {filename} — install pypdf2 to extract content]")
    return "\n\n".join(extracted)


def _send_reply(to: str, subject: str, body: str):
    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = f"AI Penpal <ask@{OUR_EMAIL_DOMAIN}>"
    msg["To"] = to
    msg["Subject"] = subject
    try:
        with smtplib.SMTP(SMTP_REPLY_HOST, SMTP_REPLY_PORT) as smtp:
            if SMTP_REPLY_USER and SMTP_REPLY_PASS:
                smtp.starttls()
                smtp.login(SMTP_REPLY_USER, SMTP_REPLY_PASS)
            smtp.sendmail(f"ask@{OUR_EMAIL_DOMAIN}", [to], msg.as_string())
    except Exception as e:
        logger.error(f"[SMTP] Failed to send reply to {to}: {e}")


def run_smtp_server():
    handler = EmailHandler()
    controller = Controller(handler, hostname=SMTP_HOST, port=SMTP_PORT)
    controller.start()
    logger.info(f"[SMTP] Server listening on {SMTP_HOST}:{SMTP_PORT}")
    return controller
PYEOF

echo "Patching processor.py to isolate conversation history..."
cat > /root/ai-penpal/processor.py << 'PYEOF'
"""
processor.py — Core pipeline
History isolation: only the messages passed in the email thread are used.
The subject line acts as a conversation identifier to keep contexts separate.
"""

import logging
from database import (
    get_history,
    append_to_history,
    clear_history,
    queue_outgoing
)
from llm import query_ollama

logger = logging.getLogger(__name__)

RESET_KEYWORDS = {"reset", "start over", "clear history", "new conversation"}


def process_message(transport_id: str, subject: str, body: str) -> dict:
    logger.info(f"[PROCESSOR] Processing message from {transport_id}")

    prompt = _clean_body(body)

    if prompt.lower().strip() in RESET_KEYWORDS:
        clear_history(transport_id)
        reply = _format_reply(
            "Your conversation history has been cleared. You can start a fresh conversation now.",
            transport_id
        )
        reply_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject
        queue_outgoing(transport_id, reply_subject, reply)
        return {"success": True, "reply_subject": reply_subject, "reply_body": reply}

    # Use subject as conversation key to isolate history per conversation
    # Strip Re: prefix to normalize
    convo_key = subject.replace("Re: ", "").replace("Re:", "").strip()
    session_key = f"{transport_id}::{convo_key}"

    history = get_history(session_key)
    has_history = len(history) > 0

    logger.info(
        f"[PROCESSOR] {transport_id} convo='{convo_key}' — "
        f"{'continuing' if has_history else 'new'} ({len(history)//2} exchanges)"
    )

    llm_response = query_ollama(prompt, history)

    if llm_response is None:
        error_reply = _format_error_reply()
        reply_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject
        queue_outgoing(transport_id, reply_subject, error_reply)
        return {"success": False, "reply_subject": reply_subject, "reply_body": error_reply, "error": "LLM inference failed"}

    # Save history under the conversation-specific key
    append_to_history(session_key, "user", prompt)
    append_to_history(session_key, "assistant", llm_response)

    # Also save under the plain email key so /api/history still works
    append_to_history(transport_id, "user", prompt)
    append_to_history(transport_id, "assistant", llm_response)

    reply = _format_reply(llm_response, transport_id)
    reply_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject
    queue_outgoing(transport_id, reply_subject, reply)

    logger.info(f"[PROCESSOR] Reply queued for {transport_id}")

    return {"success": True, "reply_subject": reply_subject, "reply_body": reply}


def _clean_body(body: str) -> str:
    lines = body.splitlines()
    cleaned = []
    for line in lines:
        if line.startswith(">"):
            break
        if line.strip().startswith("On ") and "wrote:" in line:
            break
        if line.strip() == "-- ":
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _format_reply(content: str, transport_id: str) -> str:
    return (
        f"{content}\n\n"
        f"---\n"
        f"AI Penpal | Reply to continue your conversation\n"
        f"Send 'reset' to start a new conversation"
    )


def _format_error_reply() -> str:
    return (
        "Sorry, I was unable to process your message at this time. "
        "Please try sending your message again.\n\n"
        "---\nAI Penpal | Automated Response"
    )
PYEOF

echo "Restarting services..."
systemctl restart ai-penpal
systemctl restart ai-penpal-api

echo ""
echo "Done! Both fixes applied:"
echo "1. History is now isolated per conversation (by subject)"
echo "2. PDF and TXT files are extracted and sent to Ollama"
