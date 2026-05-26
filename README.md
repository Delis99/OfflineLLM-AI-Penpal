# OfflineLLM (AI Penpal)

OfflineLLM is a local AI email assistant that combines Ollama-based inference, SMTP automation, OCR/document processing, SQLite persistence, and Linux VPS deployment into a production-style workflow.

## Highlights

- Processes email-based AI requests through a Python SMTP pipeline
- Runs local LLM inference with Ollama, with optional Anthropic API fallback
- Extracts text from PDF, DOCX, and image attachments
- Supports OCR-based document understanding
- Persists conversations and queued messages with SQLite
- Includes Flask web API support and a static demo frontend
- Supports Linux VPS deployment with nginx and systemd
- Includes local test mode and pytest coverage for safe development

## Tech Stack

- Python
- Ollama
- Flask
- SQLite
- SMTP / aiosmtpd
- OCR / pytesseract
- PDF and DOCX parsing
- Linux / nginx / systemd

## Project Structure

```text
.
├── ai-penpal/          # Core backend pipeline, SMTP server, LLM logic, tests
├── index.html          # Static web demo UI
├── web_api.py          # Flask API entry point
├── deploy.sh           # VPS deployment helper
├── nginx.conf          # nginx site configuration
└── README.md
```

## Local Setup

```bash
cd ai-penpal
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install the Tesseract OCR binary separately:

```bash
# macOS
brew install tesseract

# Ubuntu
sudo apt install tesseract-ocr
```

## Run Locally

Start Ollama first:

```bash
ollama serve
ollama pull llama3.1:8b
```

Run the app in test mode:

```bash
cd ai-penpal
python main.py --mode test --from user@example.com --subject "Hello" --body "Summarize this project."
```

Run the SMTP pipeline:

```bash
python main.py --mode smtp
```

Run tests:

```bash
python -m pytest test_pipeline.py -v
```

## Environment Variables

The app is designed to read secrets from environment variables. Do not commit local `.env` files.

Common variables:

```text
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
USE_ANTHROPIC_API=false
ANTHROPIC_API_KEY=
EMAIL_DOMAIN=offlinellm.me
SMTP_HOST=0.0.0.0
SMTP_PORT=8025
SMTP_REPLY_HOST=localhost
SMTP_REPLY_PORT=25
AI_PENPAL_DB_PATH=ai_penpal.db
FLASK_SECRET_KEY=
GOOGLE_CLIENT_ID=
```

## My Role

I designed and implemented the backend workflow, SMTP processing pipeline, local LLM integration, attachment parsing, SQLite persistence layer, Flask API surface, and VPS deployment flow.
