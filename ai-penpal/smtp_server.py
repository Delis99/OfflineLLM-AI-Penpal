import email
import logging
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from aiosmtpd.controller import Controller

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from attachment_extractor import extract_text_from_attachment
from processor import process_message
from database import queue_incoming
from config import (
    SMTP_HOST, SMTP_PORT,
    SMTP_REPLY_HOST, SMTP_REPLY_PORT,
    SMTP_REPLY_USER, SMTP_REPLY_PASS,
    OUR_EMAIL_DOMAIN,
    MAX_ATTACHMENT_SIZE_BYTES,
    SUPPORTED_ATTACHMENT_EXTENSIONS,
    DISABLE_OUTBOUND_EMAIL,
)

logger = logging.getLogger(__name__)
LATEST_REPLY_PATH = Path(__file__).resolve().parent / "latest_reply.txt"


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
            attachment_results = _extract_attachments(msg)

            logger.info(f"[SMTP] Received email from {sender} | Subject: {subject}")

            queue_incoming(sender, subject, body)
            result = process_message(sender, subject, body, attachment_results=attachment_results)

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
    html_body = None

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition", "")).lower()
            if content_type == "text/plain" and "attachment" not in disposition:
                return _decode_part(part)
            if content_type == "text/html" and "attachment" not in disposition and html_body is None:
                html_body = _decode_part(part)
        if html_body is not None:
            return _html_to_text(html_body)
        return ""

    body = _decode_part(msg)
    if msg.get_content_type() == "text/html":
        return _html_to_text(body)
    return body


def _extract_attachments(msg) -> list[dict]:
    extracted = []
    for part in msg.walk():
        disposition = str(part.get("Content-Disposition", "")).lower()
        if "attachment" not in disposition:
            continue
        filename = part.get_filename() or "attachment"
        content_type = part.get_content_type() or "application/octet-stream"
        payload = part.get_payload(decode=True)
        payload = payload or b""
        payload_size = len(payload)

        if payload_size == 0:
            result = {
                "filename": filename,
                "content_type": content_type,
                "supported": _is_supported_attachment(filename, content_type),
                "text": "",
                "error": "Attachment was empty.",
            }
            extracted.append(result)
            logger.warning("[SMTP] Attachment failed filename=%s size=%d error=%s", filename, payload_size, result["error"])
            continue

        if payload_size > MAX_ATTACHMENT_SIZE_BYTES:
            result = {
                "filename": filename,
                "content_type": content_type,
                "supported": _is_supported_attachment(filename, content_type),
                "text": "",
                "error": f"Attachment exceeds max size of {MAX_ATTACHMENT_SIZE_BYTES} bytes.",
            }
            extracted.append(result)
            logger.warning("[SMTP] Attachment skipped filename=%s size=%d error=%s", filename, payload_size, result["error"])
            continue

        result = extract_text_from_attachment(filename, content_type, payload)
        extracted.append(result)

        if result["text"]:
            logger.info(
                "[SMTP] Attachment processed filename=%s size=%d status=ok",
                filename,
                payload_size
            )
        elif result["supported"]:
            logger.warning(
                "[SMTP] Attachment failed filename=%s size=%d error=%s",
                filename,
                payload_size,
                result.get("error") or "No readable text extracted."
            )
        else:
            logger.info(
                "[SMTP] Attachment skipped filename=%s size=%d status=unsupported",
                filename,
                payload_size
            )

    return extracted


def _decode_part(part) -> str:
    charset = part.get_content_charset() or "utf-8"
    payload = part.get_payload(decode=True)
    if payload is None:
        raw_payload = part.get_payload()
        return raw_payload if isinstance(raw_payload, str) else ""
    return payload.decode(charset, errors="replace")


def _html_to_text(html: str) -> str:
    if BeautifulSoup is None:
        return html
    return BeautifulSoup(html, "html.parser").get_text("\n", strip=True)


def _is_supported_attachment(filename: str, content_type: str) -> bool:
    if any(filename.lower().endswith(ext) for ext in SUPPORTED_ATTACHMENT_EXTENSIONS):
        return True
    if content_type == "application/pdf":
        return True
    if content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return True
    if content_type in {"image/png", "image/jpeg"}:
        return True
    return False


def _send_reply(to: str, subject: str, body: str):
    if DISABLE_OUTBOUND_EMAIL:
        logger.info(
            "[SMTP] Outbound email disabled; latest reply follows\nTo: %s\nSubject: %s\n\n%s",
            to,
            subject,
            body
        )
        _write_latest_reply_file(to, subject, body)
        return

    msg = MIMEText(body, "plain", "utf-8")
    msg["From"] = f"OfflineLLM <ask@{OUR_EMAIL_DOMAIN}>"
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


def _write_latest_reply_file(to: str, subject: str, body: str):
    try:
        content = (
            f"To: {to}\n"
            f"Subject: {subject}\n"
            f"\n"
            f"{body}\n"
        )
        LATEST_REPLY_PATH.write_text(content, encoding="utf-8")
        logger.info("[SMTP] Saved latest reply to %s", LATEST_REPLY_PATH)
    except Exception as exc:
        logger.error("[SMTP] Failed to write %s: %s", LATEST_REPLY_PATH, exc)


def run_smtp_server():
    handler = EmailHandler()
    controller = Controller(handler, hostname=SMTP_HOST, port=SMTP_PORT)
    controller.start()
    logger.info(f"[SMTP] Server listening on {SMTP_HOST}:{SMTP_PORT}")
    return controller
