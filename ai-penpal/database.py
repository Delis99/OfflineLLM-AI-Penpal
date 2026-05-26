"""
database.py
-----------
Handles all SQLite operations for:
  - Conversation history (per user session)
  - Message queue (incoming + outgoing)

Schema mirrors DDD's TransportMessage structure:
  transportId = user's email address (primary session key)
  subject     = email subject
  body        = email content / LLM prompt or response
"""

import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Optional
from config import DB_PATH as CONFIG_DB_PATH

DB_PATH = CONFIG_DB_PATH

# Thread-safe connection pool
_local = threading.local()


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_conn() -> sqlite3.Connection:
    """Return a thread-local SQLite connection."""
    if not hasattr(_local, "conn"):
        if DB_PATH != ":memory:":
            db_dir = os.path.dirname(os.path.abspath(DB_PATH))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
    return _local.conn


def init_db():
    """Create all tables if they don't exist."""
    conn = get_conn()
    conn.executescript("""
        -- One row per user email. Stores full conversation as JSON list.
        CREATE TABLE IF NOT EXISTS sessions (
            transport_id    TEXT PRIMARY KEY,
            history         TEXT NOT NULL DEFAULT '[]',
            created_at      TEXT NOT NULL,
            updated_at      TEXT NOT NULL
        );

        -- Incoming messages from DDD (or aiosmtpd in 195A)
        CREATE TABLE IF NOT EXISTS incoming_messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            transport_id    TEXT NOT NULL,
            subject         TEXT NOT NULL,
            body            TEXT NOT NULL,
            received_at     TEXT NOT NULL,
            processed       INTEGER NOT NULL DEFAULT 0
        );

        -- Outgoing replies queued to send back via DDD
        CREATE TABLE IF NOT EXISTS outgoing_messages (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            transport_id    TEXT NOT NULL,
            subject         TEXT NOT NULL,
            body            TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            sent            INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS conversation_summaries (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            sender_email        TEXT NOT NULL,
            conversation_key    TEXT NOT NULL,
            summary             TEXT NOT NULL,
            updated_at          TEXT NOT NULL,
            UNIQUE(sender_email, conversation_key)
        );

        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            email           TEXT NOT NULL UNIQUE,
            name            TEXT,
            picture         TEXT,
            google_sub      TEXT NOT NULL UNIQUE,
            created_at      TEXT NOT NULL,
            last_login_at   TEXT NOT NULL
        );
    """)
    conn.commit()
    print("[DB] Tables initialized.")


# ── SESSION MANAGEMENT ────────────────────────────────────────────────────────

def get_history(transport_id: str) -> list:
    """
    Return conversation history for a user as a list of dicts:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    import json
    conn = get_conn()
    row = conn.execute(
        "SELECT history FROM sessions WHERE transport_id = ?",
        (transport_id,)
    ).fetchone()
    if row is None:
        return []
    return json.loads(row["history"])


def save_history(transport_id: str, history: list):
    """Upsert conversation history for a user."""
    import json
    conn = get_conn()
    now = _utc_now_isoformat()
    conn.execute("""
        INSERT INTO sessions (transport_id, history, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(transport_id) DO UPDATE SET
            history    = excluded.history,
            updated_at = excluded.updated_at
    """, (transport_id, json.dumps(history), now, now))
    conn.commit()


def append_to_history(transport_id: str, role: str, content: str):
    """Append a single message to a user's conversation history."""
    history = get_history(transport_id)
    history.append({"role": role, "content": content})

    # Context window management: keep last 20 exchanges (40 messages)
    # to avoid hitting Ollama's context limit on long conversations
    if len(history) > 40:
        history = history[-40:]

    save_history(transport_id, history)


def clear_history(transport_id: str):
    """Clear conversation history for a user (e.g. if they send 'reset')."""
    save_history(transport_id, [])


def get_conversation_summary(sender_email: str, conversation_key: str) -> str:
    """Return the saved compact summary for a sender/conversation pair."""
    conn = get_conn()
    row = conn.execute(
        """
        SELECT summary
        FROM conversation_summaries
        WHERE sender_email = ? AND conversation_key = ?
        """,
        (sender_email, conversation_key),
    ).fetchone()
    if row is None:
        return ""
    return row["summary"]


def save_conversation_summary(sender_email: str, conversation_key: str, summary: str) -> None:
    """Upsert the compact summary for a sender/conversation pair."""
    conn = get_conn()
    now = _utc_now_isoformat()
    conn.execute(
        """
        INSERT INTO conversation_summaries (sender_email, conversation_key, summary, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(sender_email, conversation_key) DO UPDATE SET
            summary = excluded.summary,
            updated_at = excluded.updated_at
        """,
        (sender_email, conversation_key, summary, now),
    )
    conn.commit()


def clear_conversation_summary(sender_email: str, conversation_key: str) -> None:
    """Delete the compact summary for a sender/conversation pair."""
    conn = get_conn()
    conn.execute(
        """
        DELETE FROM conversation_summaries
        WHERE sender_email = ? AND conversation_key = ?
        """,
        (sender_email, conversation_key),
    )
    conn.commit()


# ── MESSAGE QUEUE ─────────────────────────────────────────────────────────────

def queue_incoming(transport_id: str, subject: str, body: str) -> int:
    """Store an incoming message. Returns the new row ID."""
    conn = get_conn()
    cursor = conn.execute("""
        INSERT INTO incoming_messages (transport_id, subject, body, received_at, processed)
        VALUES (?, ?, ?, ?, 0)
    """, (transport_id, subject, body, _utc_now_isoformat()))
    conn.commit()
    return cursor.lastrowid


def get_unprocessed_messages() -> list:
    """Return all incoming messages not yet processed."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM incoming_messages
        WHERE processed = 0
        ORDER BY received_at ASC
    """).fetchall()
    return [dict(r) for r in rows]


def mark_processed(message_id: int):
    """Mark an incoming message as processed."""
    conn = get_conn()
    conn.execute(
        "UPDATE incoming_messages SET processed = 1 WHERE id = ?",
        (message_id,)
    )
    conn.commit()


def queue_outgoing(transport_id: str, subject: str, body: str) -> int:
    """Queue an outgoing reply. Returns the new row ID."""
    conn = get_conn()
    cursor = conn.execute("""
        INSERT INTO outgoing_messages (transport_id, subject, body, created_at, sent)
        VALUES (?, ?, ?, ?, 0)
    """, (transport_id, subject, body, _utc_now_isoformat()))
    conn.commit()
    return cursor.lastrowid


def get_unsent_messages() -> list:
    """Return all outgoing messages not yet sent."""
    conn = get_conn()
    rows = conn.execute("""
        SELECT * FROM outgoing_messages
        WHERE sent = 0
        ORDER BY created_at ASC
    """).fetchall()
    return [dict(r) for r in rows]


def mark_sent(message_id: int):
    """Mark an outgoing message as sent."""
    conn = get_conn()
    conn.execute(
        "UPDATE outgoing_messages SET sent = 1 WHERE id = ?",
        (message_id,)
    )
    conn.commit()
