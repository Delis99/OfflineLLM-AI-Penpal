"""
smtp_patch.py
Run this on the server to patch smtp_server.py with file extraction support.
Usage: python3 smtp_patch.py
"""
import re

path = '/root/ai-penpal/smtp_server.py'
content = open(path).read()

# Add PDF extraction imports at the top
old_imports = 'import asyncio\nimport email\nimport logging\nimport smtplib'
new_imports = '''import asyncio
import email
import logging
import smtplib
import io

# PDF text extraction (install: pip install pypdf2 --break-system-packages)
try:
    import PyPDF2
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.getLogger(__name__).warning("PyPDF2 not installed — PDF support disabled. Run: pip install pypdf2 --break-system-packages")'''

content = content.replace(old_imports, new_imports, 1)

# Add file extraction helper function before _send_reply
old_fn = 'def _send_reply('
new_fn = '''def _extract_attachments(msg) -> str:
    """Extract text content from PDF and TXT attachments."""
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
                extracted.append(f"[Attached file: {filename}]\\n{text[:8000]}")
            except Exception as e:
                logger.warning(f"Could not read TXT {filename}: {e}")
        elif filename.lower().endswith('.pdf'):
            if PDF_SUPPORT:
                try:
                    reader = PyPDF2.PdfReader(io.BytesIO(payload))
                    pages_text = []
                    for page in reader.pages[:10]:  # Max 10 pages
                        pages_text.append(page.extract_text() or "")
                    text = "\\n".join(pages_text)[:8000]
                    extracted.append(f"[Attached PDF: {filename}]\\n{text}")
                except Exception as e:
                    logger.warning(f"Could not read PDF {filename}: {e}")
            else:
                extracted.append(f"[Attached PDF: {filename} — could not extract text, PyPDF2 not installed]")
    return "\\n\\n".join(extracted)


def _send_reply('''

content = content.replace(old_fn, new_fn, 1)

# Update handle_DATA to extract attachments and append to body
old_extract = "            subject = msg.get(\"Subject\", \"(no subject)\")\n            body = _extract_body(msg)"
new_extract = """            subject = msg.get("Subject", "(no subject)")
            body = _extract_body(msg)

            # Extract text from attachments and append to body
            attachment_text = _extract_attachments(msg)
            if attachment_text:
                body = body + "\\n\\n" + attachment_text
                logger.info(f"[SMTP] Extracted attachment content ({len(attachment_text)} chars)")"""

content = content.replace(old_extract, new_extract, 1)

open(path, 'w').write(content)
print("smtp_server.py patched successfully!")
