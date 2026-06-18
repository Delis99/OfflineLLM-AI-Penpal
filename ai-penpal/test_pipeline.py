"""
tests/test_pipeline.py
----------------------
Unit tests for the AI Penpal pipeline.
Tests processor, database, and email parsing — without needing
a live Ollama instance (LLM calls are mocked).

Run: python -m pytest tests/ -v
"""

import sys
import os
import json
import logging
from pathlib import Path
import types
sys.path.insert(0, os.path.dirname(__file__))

import pytest
from unittest.mock import patch, MagicMock

PROJECT_DIR = Path(__file__).resolve().parent


# ── RUNTIME CONFIGURATION TESTS ───────────────────────────────────────────────

class TestRuntimeConfiguration:

    def test_requirements_include_runtime_dependencies(self):
        deps = {
            line.strip().lower()
            for line in (PROJECT_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

        assert "flask" in deps
        assert "aiosmtpd" in deps
        assert "anthropic" in deps
        assert "google-auth" in deps
        assert "requests" in deps

    def test_config_does_not_contain_hardcoded_anthropic_key(self):
        source = (PROJECT_DIR / "config.py").read_text(encoding="utf-8")

        assert "sk-ant-" not in source
        assert 'ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")' in source

    def test_database_path_is_consistent(self):
        import config
        import database
        import web_api

        expected = os.getenv("AI_PENPAL_DB_PATH", str(PROJECT_DIR / "ai_penpal.db"))

        assert config.DB_PATH == expected
        assert database.DB_PATH == config.DB_PATH
        assert web_api.DB_PATH == config.DB_PATH


# ── DATABASE TESTS ────────────────────────────────────────────────────────────

class TestDatabase:

    def setup_method(self):
        """Use in-memory DB for each test."""
        import database
        database.DB_PATH = ":memory:"
        # Force new connection
        if hasattr(database._local, "conn"):
            del database._local.conn
        database.init_db()

    def test_history_empty_for_new_user(self):
        from database import get_history
        history = get_history("newuser@example.com")
        assert history == []

    def test_save_and_retrieve_history(self):
        from database import save_history, get_history
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        save_history("user@example.com", history)
        retrieved = get_history("user@example.com")
        assert retrieved == history

    def test_append_to_history(self):
        from database import append_to_history, get_history
        append_to_history("user@example.com", "user", "What is Python?")
        append_to_history("user@example.com", "assistant", "Python is a programming language.")
        history = get_history("user@example.com")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_clear_history(self):
        from database import append_to_history, clear_history, get_history
        append_to_history("user@example.com", "user", "Hello")
        clear_history("user@example.com")
        assert get_history("user@example.com") == []

    def test_history_context_window_limit(self):
        """History should be capped at 40 messages (20 exchanges)."""
        from database import append_to_history, get_history
        for i in range(25):
            append_to_history("user@example.com", "user", f"Message {i}")
            append_to_history("user@example.com", "assistant", f"Reply {i}")
        history = get_history("user@example.com")
        assert len(history) <= 40

    def test_queue_incoming_message(self):
        from database import queue_incoming, get_unprocessed_messages
        msg_id = queue_incoming("user@example.com", "Test Subject", "Test body")
        assert msg_id > 0
        messages = get_unprocessed_messages()
        assert len(messages) == 1
        assert messages[0]["transport_id"] == "user@example.com"

    def test_mark_processed(self):
        from database import queue_incoming, mark_processed, get_unprocessed_messages
        msg_id = queue_incoming("user@example.com", "Subject", "Body")
        mark_processed(msg_id)
        assert get_unprocessed_messages() == []

    def test_separate_sessions_per_user(self):
        from database import append_to_history, get_history
        append_to_history("alice@example.com", "user", "Alice's message")
        append_to_history("bob@example.com", "user", "Bob's message")
        alice_history = get_history("alice@example.com")
        bob_history = get_history("bob@example.com")
        assert len(alice_history) == 1
        assert len(bob_history) == 1
        assert alice_history[0]["content"] == "Alice's message"
        assert bob_history[0]["content"] == "Bob's message"

    def test_summary_table_is_created(self):
        import database
        conn = database.get_conn()
        row = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'conversation_summaries'
            """
        ).fetchone()
        assert row is not None
        assert row["name"] == "conversation_summaries"

    def test_conversation_summary_can_be_saved_and_retrieved(self):
        from database import get_conversation_summary, save_conversation_summary

        save_conversation_summary(
            "user@example.com",
            "Project chat",
            "User is building an email backend with document extraction.",
        )

        summary = get_conversation_summary("user@example.com", "Project chat")
        assert "document extraction" in summary


# ── PROCESSOR TESTS ───────────────────────────────────────────────────────────

class TestProcessor:

    def setup_method(self):
        import database
        database.DB_PATH = ":memory:"
        if hasattr(database._local, "conn"):
            del database._local.conn
        database.init_db()

    @patch("processor.query_ollama")
    def test_process_new_user_message(self, mock_llm):
        mock_llm.return_value = "Python is a high-level programming language."
        from processor import process_message
        result = process_message("user@example.com", "Python question", "What is Python?")
        assert result["success"] is True
        assert "Re:" in result["reply_subject"]
        assert "Python" in result["reply_body"]
        assert mock_llm.call_count == 1

    @patch("processor.query_ollama")
    def test_summary_updates_without_second_llm_call(self, mock_llm):
        mock_llm.return_value = "Main model response."
        from processor import process_message
        from database import get_conversation_summary

        process_message("user@example.com", "Summary update", "Remember this context.")

        summary = get_conversation_summary("user@example.com", "Summary update")
        assert "User: Current user request: Remember this context." in summary
        assert "Assistant: Main model response." in summary
        assert mock_llm.call_count == 1

    @patch("processor.query_ollama")
    def test_history_is_passed_to_llm(self, mock_llm):
        """LLM should receive prior conversation history."""
        mock_llm.return_value = "Sure!"
        from processor import process_message
        from database import append_to_history

        # Simulate existing history
        append_to_history("user@example.com::Follow up", "user", "My name is Juan")
        append_to_history("user@example.com::Follow up", "assistant", "Nice to meet you Juan!")

        process_message("user@example.com", "Follow up", "What is my name?")

        # Check that history was passed to Ollama
        call_args = mock_llm.call_args
        history = call_args[0][1]  # second argument is history
        assert any(m["content"] == "My name is Juan" for m in history)
        assert mock_llm.call_count == 1

    @patch("processor.query_ollama")
    def test_text_email_uses_summary_and_last_one_exchange(self, mock_llm):
        mock_llm.return_value = "Trimmed history response."
        from processor import process_message
        from database import append_to_history, get_history, save_conversation_summary

        session_key = "user@example.com::Long chat"
        for index in range(9):
            append_to_history(session_key, "user", f"User message {index}")
            append_to_history(session_key, "assistant", f"Assistant reply {index}")
        save_conversation_summary(
            "user@example.com",
            "Long chat",
            "The user previously asked about a long-running project discussion.",
        )

        process_message("user@example.com", "Long chat", "What did we discuss?")

        prompt = mock_llm.call_args[0][0]
        history = mock_llm.call_args[0][1]
        assert "Conversation memory summary:" in prompt
        assert "long-running project discussion" in prompt
        assert "Current user request:\nWhat did we discuss?" in prompt
        assert len(history) == 2
        assert history[0]["content"] == "User message 8"
        assert history[-1]["content"] == "Assistant reply 8"
        assert len(get_history(session_key)) == 20
        assert mock_llm.call_count == 1

    @patch("processor.query_ollama")
    def test_attachment_email_uses_summary_and_no_raw_history(self, mock_llm):
        mock_llm.return_value = "Attachment response."
        from processor import process_message
        from database import append_to_history, save_conversation_summary

        session_key = "user@example.com::Project doc"
        for index in range(9):
            append_to_history(session_key, "user", f"User message {index}")
            append_to_history(session_key, "assistant", f"Assistant reply {index}")
        save_conversation_summary(
            "user@example.com",
            "Project doc",
            "The user is reviewing project documents and wants concise factual answers.",
        )

        process_message(
            "user@example.com",
            "Project doc",
            "Please review the file.",
            attachment_results=[
                {
                    "filename": "notes.pdf",
                    "content_type": "application/pdf",
                    "supported": True,
                    "text": "Short attached notes.",
                    "error": None,
                }
            ],
        )

        prompt = mock_llm.call_args[0][0]
        history = mock_llm.call_args[0][1]
        assert "Conversation memory summary:" in prompt
        assert "reviewing project documents" in prompt
        assert "Attached document content:" in prompt
        assert history == []
        assert mock_llm.call_count == 1

    @patch("processor.query_ollama")
    def test_large_attachment_uses_zero_history_exchanges(self, mock_llm):
        mock_llm.return_value = "Large attachment response."
        from processor import process_message
        from database import append_to_history, save_conversation_summary

        session_key = "user@example.com::Large attachment"
        for index in range(9):
            append_to_history(session_key, "user", f"User message {index}")
            append_to_history(session_key, "assistant", f"Assistant reply {index}")
        save_conversation_summary(
            "user@example.com",
            "Large attachment",
            "The user has been sending large attached reports for summarization.",
        )

        process_message(
            "user@example.com",
            "Large attachment",
            "Please summarize the attachment.",
            attachment_results=[
                {
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                    "supported": True,
                    "text": "A" * 10001,
                    "error": None,
                }
            ],
        )

        prompt = mock_llm.call_args[0][0]
        history = mock_llm.call_args[0][1]
        assert "Conversation memory summary:" in prompt
        assert history == []
        assert mock_llm.call_count == 1

    @patch("processor.query_ollama")
    def test_reset_command_clears_history_and_summary(self, mock_llm):
        from processor import process_message
        from database import append_to_history, get_conversation_summary, get_history, save_conversation_summary

        append_to_history("user@example.com", "user", "Hello")
        append_to_history("user@example.com::Reset", "user", "Hello")
        save_conversation_summary("user@example.com", "Reset", "Stored summary to clear.")

        result = process_message("user@example.com", "Reset", "reset")

        assert result["success"] is True
        assert get_history("user@example.com") == []
        assert get_history("user@example.com::Reset") == []
        assert get_conversation_summary("user@example.com", "Reset") == ""
        mock_llm.assert_not_called()

    @patch("processor.query_ollama")
    def test_llm_failure_returns_error_reply(self, mock_llm):
        mock_llm.return_value = None  # Simulate Ollama failure
        from processor import process_message
        result = process_message("user@example.com", "Test", "Hello")
        assert result["success"] is False
        assert "unable to process" in result["reply_body"]

    @patch("processor.query_ollama")
    def test_reply_subject_prefixed(self, mock_llm):
        mock_llm.return_value = "Hello!"
        from processor import process_message
        result = process_message("user@example.com", "My question", "Hello")
        assert result["reply_subject"] == "Re: My question"

    @patch("processor.query_ollama")
    def test_reply_subject_not_double_prefixed(self, mock_llm):
        mock_llm.return_value = "Hello!"
        from processor import process_message
        result = process_message("user@example.com", "Re: My question", "Hello")
        assert result["reply_subject"] == "Re: My question"  # Not "Re: Re: My question"

    @patch("processor.query_ollama")
    def test_attachment_results_are_composed_into_prompt(self, mock_llm):
        mock_llm.return_value = "Here is the summary."
        from processor import process_message

        attachment_results = [
            {
                "filename": "notes.pdf",
                "content_type": "application/pdf",
                "supported": True,
                "text": "Project deadline is May 10.",
                "error": None,
            }
        ]

        process_message(
            "user@example.com",
            "Document question",
            "When is the deadline?",
            attachment_results=attachment_results,
        )

        prompt = mock_llm.call_args[0][0]
        assert "Current user request:" in prompt
        assert "When is the deadline?" in prompt
        assert "Attached document content:" in prompt
        assert "File: notes.pdf" in prompt
        assert "Project deadline is May 10." in prompt
        assert "Do not invent information." in prompt

    def test_attachment_text_cap_is_applied_to_prompt(self):
        from config import MAX_ATTACHMENT_TEXT_CHARS
        from processor import _build_prompt

        long_text = "A" * (MAX_ATTACHMENT_TEXT_CHARS + 250)
        prompt, _, _, _ = _build_prompt(
            "Summarize this document.",
            [
                {
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                    "supported": True,
                    "text": long_text,
                    "error": None,
                }
            ],
        )

        assert f"File: report.pdf\n{'A' * MAX_ATTACHMENT_TEXT_CHARS}" in prompt
        assert ("A" * (MAX_ATTACHMENT_TEXT_CHARS + 1)) not in prompt

    @patch("processor.query_ollama")
    def test_processor_includes_summary_in_prompt(self, mock_llm):
        mock_llm.return_value = "Summary-aware response."
        from processor import process_message
        from database import save_conversation_summary

        save_conversation_summary(
            "user@example.com",
            "Project summary",
            "The user prefers short answers and is building an email backend.",
        )

        process_message("user@example.com", "Project summary", "What should I do next?")

        prompt = mock_llm.call_args[0][0]
        assert "Conversation memory summary:" in prompt
        assert "prefers short answers" in prompt
        assert "Current user request:\nWhat should I do next?" in prompt

    @patch("processor.query_ollama")
    def test_processor_truncates_long_summary_for_llm(self, mock_llm, caplog):
        mock_llm.return_value = "Summary-aware response."
        from processor import process_message
        from database import save_conversation_summary

        long_summary = "BEGIN" + ("X" * 2050) + "END"
        save_conversation_summary("user@example.com", "Long summary", long_summary)

        with caplog.at_level(logging.INFO):
            process_message("user@example.com", "Long summary", "What should I do next?")

        prompt = mock_llm.call_args[0][0]
        assert "Conversation memory summary:" in prompt
        assert "BEGIN" not in prompt
        assert "END" in prompt
        assert "Truncated conversation summary from" in caplog.text

    @patch("processor.query_ollama")
    def test_query_ollama_receives_num_predict_for_summary_request(self, mock_llm):
        mock_llm.return_value = "Summary response."
        from config import SUMMARY_NUM_PREDICT
        from processor import process_message

        process_message("user@example.com", "Summary request", "Summarize this.")

        assert mock_llm.call_args.kwargs["num_predict"] == SUMMARY_NUM_PREDICT

    @patch("processor.query_ollama")
    def test_math_expression_note_is_added_to_prompt_when_uncertain(self, mock_llm):
        mock_llm.return_value = "Please confirm the expression."
        from processor import _build_prompt, process_message

        attachment_results = [
            {
                "filename": "equation.png",
                "content_type": "image/png",
                "supported": True,
                "text": (
                    "OCR detected math expression:\n"
                    "6+2*(14+2)\n"
                    "Computed result:\n"
                    "Unavailable\n"
                    "Note: OCR extraction may be inaccurate. Please verify the expression from the image."
                ),
                "math_expression": "6+2*(14+2)",
                "math_expression_uncertain": True,
                "math_expression_note": "OCR extraction may be inaccurate. Please verify the expression from the image.",
                "computed_result": None,
                "math_confirmation_expression": "6 ÷ 2(1+2)",
                "error": None,
            }
        ]

        prompt, _, _, _ = _build_prompt("What is the solution?", attachment_results)

        result = process_message(
            "user@example.com",
            "Math OCR",
            "What is the solution?",
            attachment_results=attachment_results,
        )

        assert "OCR math extraction is uncertain. Ask the user to confirm the expression before solving, or solve only if the expression is clearly readable." in prompt
        assert "Do not present any computed result as final while the OCR math extraction is uncertain." in prompt
        assert "6+2*(14+2)" in prompt
        assert "OCR extraction may be inaccurate. Please verify the expression from the image." in prompt
        assert "Please confirm the correct expression. For example, is it:" in result["reply_body"]
        assert "6 ÷ 2(1+2) ?" in result["reply_body"]
        mock_llm.assert_not_called()

    @patch("processor.query_ollama")
    def test_vision_math_reply_uses_deterministic_format(self, mock_llm):
        from processor import process_message

        result = process_message(
            "user@example.com",
            "Math image",
            "Solve this image.",
            attachment_results=[
                {
                    "filename": "equation.png",
                    "content_type": "image/png",
                    "supported": True,
                    "text": (
                        "Interpreted expression from the image:\n"
                        "6 ÷ 2(1+2)\n\n"
                        "Normalized:\n"
                        "6/2*(1+2)\n\n"
                        "Computed result:\n"
                        "9"
                    ),
                    "math_expression": "6 ÷ 2(1+2)",
                    "normalized_math_expression": "6/2*(1+2)",
                    "math_source": "vision",
                    "math_expression_uncertain": False,
                    "math_expression_note": None,
                    "computed_result": "9",
                    "error": None,
                }
            ],
        )

        assert "Interpreted expression from the image:\n6 ÷ 2(1+2)" in result["reply_body"]
        assert "Normalized:\n6/2*(1+2)" in result["reply_body"]
        assert "Step-by-step:" in result["reply_body"]
        assert "Step 1: Start with the normalized expression 6/2*(1+2)." in result["reply_body"]
        assert "Final Answer:\n9" in result["reply_body"]
        mock_llm.assert_not_called()

    @patch("processor.query_ollama")
    def test_math_prompt_uses_computed_result_as_authoritative(self, mock_llm):
        mock_llm.return_value = "The answer is 7."
        from processor import process_message

        process_message(
            "user@example.com",
            "Math OCR",
            "What is the answer?",
            attachment_results=[
                {
                    "filename": "equation.png",
                    "content_type": "image/png",
                    "supported": True,
                    "text": (
                        "OCR detected math expression:\n"
                        "6/2*(1+2)\n"
                        "Computed result:\n"
                        "9"
                    ),
                    "math_expression": "6/2*(1+2)",
                    "math_expression_uncertain": False,
                    "math_expression_note": None,
                    "computed_result": "9",
                    "error": None,
                }
            ],
        )

        prompt = mock_llm.call_args[0][0]
        assert "Use the computed result as the authoritative answer." in prompt
        assert "Do not invent, replace, or simplify to a different expression" in prompt
        assert "6/2*(1+2)" in prompt

    @patch("processor.query_ollama")
    def test_document_solution_requests_get_step_by_step_instruction(self, mock_llm):
        mock_llm.return_value = "Here is the solution."
        from processor import process_message

        process_message(
            "user@example.com",
            "Math worksheet",
            "Explain step by step how to solve question 3.",
            attachment_results=[
                {
                    "filename": "worksheet.pdf",
                    "content_type": "application/pdf",
                    "supported": True,
                    "text": "Question 3 asks the student to solve a system of equations.",
                    "error": None,
                }
            ],
        )

        prompt = mock_llm.call_args[0][0]
        assert "Provide a structured explanation with clear step-by-step guidance." in prompt
        assert "Use Step 1, Step 2, Step 3 labels." in prompt
        assert "Each step should represent a clear reasoning or explanation stage." in prompt
        assert "Do not use unnumbered section headers for step-by-step requests" in prompt
        assert "Below is a step-by-step explanation of the attached document." in prompt
        assert "Explicitly connect each major point or step to the relevant document content." in prompt
        assert "summarize the attached document in a clear, concise way" not in prompt.lower()

    @patch("processor.query_ollama")
    def test_step_by_step_request_for_section_based_document_aligns_wording(self, mock_llm):
        mock_llm.return_value = "Here is the explanation."
        from processor import process_message

        process_message(
            "user@example.com",
            "Resume walkthrough",
            "Explain step by step what this says.",
            attachment_results=[
                {
                    "filename": "resume.pdf",
                    "content_type": "application/pdf",
                    "supported": True,
                    "text": (
                        "Resume\n"
                        "Professional Experience\n"
                        "Education\n"
                        "Skills\n"
                    ),
                    "error": None,
                }
            ],
        )

        prompt = mock_llm.call_args[0][0]
        assert "Use Step 1, Step 2, Step 3 labels." in prompt
        assert "Do not use unnumbered section headers for step-by-step requests" in prompt
        assert "Step 1: Contact Information" in prompt
        assert "section-by-section explanation" not in prompt

    @patch("processor.query_ollama")
    def test_document_summary_requests_keep_summary_instruction(self, mock_llm):
        mock_llm.return_value = "Here is the summary."
        from processor import process_message

        process_message(
            "user@example.com",
            "Mystery document",
            "What is this?",
            attachment_results=[
                {
                    "filename": "mystery.docx",
                    "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "supported": True,
                    "text": "This document describes the onboarding process for new volunteers.",
                    "error": None,
                }
            ],
        )

        prompt = mock_llm.call_args[0][0]
        assert "Summarize the attached document in a clear, concise way" in prompt
        assert "Provide a structured explanation with clear step-by-step guidance." not in prompt

    @patch("processor.query_ollama")
    def test_clear_resume_context_uses_confident_document_type_instruction(self, mock_llm):
        mock_llm.return_value = "This document is a resume."
        from processor import process_message

        process_message(
            "user@example.com",
            "Candidate document",
            "What is this?",
            attachment_results=[
                {
                    "filename": "candidate.pdf",
                    "content_type": "application/pdf",
                    "supported": True,
                    "text": (
                        "Resume\n"
                        "Professional Experience\n"
                        "Education\n"
                        "Skills\n"
                    ),
                    "error": None,
                }
            ],
        )

        prompt = mock_llm.call_args[0][0]
        assert "The attached document is a resume." in prompt
        assert "instead of saying it appears to be a resume" in prompt

    @patch("processor.query_ollama")
    def test_full_raw_history_is_still_stored(self, mock_llm):
        mock_llm.return_value = "Stored history response."
        from processor import process_message
        from database import get_history

        process_message("user@example.com", "Stored history", "Keep this in history.")

        history = get_history("user@example.com::Stored history")
        assert len(history) == 2
        assert history[0]["content"] == "Keep this in history."
        assert "Conversation memory summary:" not in history[0]["content"]

    @patch("processor.query_ollama")
    def test_resume_expected_graduation_does_not_become_completed_degree(self, mock_llm):
        mock_llm.return_value = "He has completed his Bachelor's degree."
        from processor import process_message

        result = process_message(
            "user@example.com",
            "Resume question",
            "Summarize this resume.",
            attachment_results=[
                {
                    "filename": "resume.pdf",
                    "content_type": "application/pdf",
                    "supported": True,
                    "text": (
                        "Resume\n"
                        "Bachelor's degree in Computer Science\n"
                        "Expected graduation December 2026\n"
                        "Skills\n"
                    ),
                    "error": None,
                }
            ],
        )

        prompt = mock_llm.call_args[0][0]
        assert "Do not say the degree is completed, has been completed, or already awarded." in prompt
        assert "completed his Bachelor's degree" not in result["reply_body"]
        assert "currently pursuing" in result["reply_body"].lower() or "expected" in result["reply_body"].lower()

    @patch("processor.query_ollama")
    def test_clear_technical_report_context_uses_confident_document_type_instruction(self, mock_llm):
        mock_llm.return_value = "This document is a technical report."
        from processor import process_message

        process_message(
            "user@example.com",
            "Report question",
            "Summarize this.",
            attachment_results=[
                {
                    "filename": "report.pdf",
                    "content_type": "application/pdf",
                    "supported": True,
                    "text": (
                        "Technical Report\n"
                        "Abstract\n"
                        "Methodology\n"
                        "Results\n"
                        "Conclusion\n"
                    ),
                    "error": None,
                }
            ],
        )

        prompt = mock_llm.call_args[0][0]
        assert "The attached document is a technical report." in prompt
        assert "instead of saying it appears to be a technical report" in prompt

    @patch("processor.query_ollama")
    def test_computed_math_result_overrides_mismatched_llm_answer(self, mock_llm):
        mock_llm.return_value = "The answer is 9."
        from processor import process_message

        result = process_message(
            "user@example.com",
            "Math check",
            "What is the solution?",
            attachment_results=[
                {
                    "filename": "equation.png",
                    "content_type": "image/png",
                    "supported": True,
                    "text": (
                        "OCR detected math expression:\n"
                        "8/2+3\n"
                        "Computed result:\n"
                        "7"
                    ),
                    "math_expression": "8/2+3",
                    "math_expression_uncertain": False,
                    "math_expression_note": None,
                    "computed_result": "7",
                    "error": None,
                }
            ],
        )

        assert "Expression: 8/2+3" in result["reply_body"]
        assert "Answer: 7" in result["reply_body"]
        assert "did not match the computed result" in result["reply_body"]

    @patch("processor.query_ollama")
    def test_uncertain_math_expression_is_not_treated_as_authoritative(self, mock_llm):
        mock_llm.return_value = "The answer is 38."
        from processor import process_message

        result = process_message(
            "user@example.com",
            "Math check",
            "Solve this image.",
            attachment_results=[
                {
                    "filename": "equation.png",
                    "content_type": "image/png",
                    "supported": True,
                    "text": (
                        "OCR detected math expression:\n"
                        "6+2*(14+2)\n"
                        "Computed result:\n"
                        "Unavailable\n"
                        "Note: OCR extraction may be inaccurate. Please verify the expression from the image."
                    ),
                    "math_expression": "6+2*(14+2)",
                    "math_expression_uncertain": True,
                    "math_expression_note": "OCR extraction may be inaccurate. Please verify the expression from the image.",
                    "computed_result": None,
                    "math_confirmation_expression": "6 ÷ 2(1+2)",
                    "error": None,
                }
            ],
        )

        assert "Extracted expression:\n6+2*(14+2)" in result["reply_body"]
        assert "Please confirm the correct expression. For example, is it:" in result["reply_body"]
        assert "6 ÷ 2(1+2) ?" in result["reply_body"]
        assert "38" not in result["reply_body"]
        assert "final answer" not in result["reply_body"].lower()
        mock_llm.assert_not_called()

    @patch("processor.query_ollama")
    def test_confident_ocr_expression_returns_nine(self, mock_llm):
        mock_llm.return_value = "The answer is 3."
        from processor import process_message

        result = process_message(
            "user@example.com",
            "Math check",
            "Solve this image.",
            attachment_results=[
                {
                    "filename": "equation.png",
                    "content_type": "image/png",
                    "supported": True,
                    "text": (
                        "OCR detected math expression:\n"
                        "6/2*(1+2)\n"
                        "Computed result:\n"
                        "9"
                    ),
                    "math_expression": "6/2*(1+2)",
                    "math_expression_uncertain": False,
                    "math_expression_note": None,
                    "computed_result": "9",
                    "math_confirmation_expression": "6 ÷ 2(1+2)",
                    "error": None,
                }
            ],
        )

        assert "Expression: 6/2*(1+2)" in result["reply_body"]
        assert "Answer: 9" in result["reply_body"]

    @patch("processor.query_ollama")
    def test_unreadable_attachment_returns_polite_reply(self, mock_llm):
        from processor import process_message

        result = process_message(
            "user@example.com",
            "Unreadable attachment",
            "Please review the attachment.",
            attachment_results=[
                {
                    "filename": "scan.png",
                    "content_type": "image/png",
                    "supported": True,
                    "text": "",
                    "error": "No readable text extracted.",
                }
            ],
        )

        assert result["success"] is True
        assert "could not extract readable text" in result["reply_body"]
        mock_llm.assert_not_called()

    @patch("processor.query_ollama")
    def test_reply_body_strips_duplicate_subject_line(self, mock_llm):
        mock_llm.return_value = "Subject: Re: Summary of the Attached Document\n\nHere is the actual reply."
        from processor import process_message

        result = process_message("user@example.com", "Test PDF", "Please summarize this.")

        assert result["reply_subject"] == "Re: Test PDF"
        assert "Subject: Re: Summary of the Attached Document" not in result["reply_body"]
        assert "Here is the actual reply." in result["reply_body"]


# ── EMAIL PARSING TESTS ───────────────────────────────────────────────────────

class TestEmailParsing:

    def test_clean_body_removes_quoted_reply(self):
        from processor import _clean_body
        body = """This is my actual question.

> On March 14, 2026, AI Penpal wrote:
> Hello, how can I help you?"""
        cleaned = _clean_body(body)
        assert "actual question" in cleaned
        assert "AI Penpal wrote" not in cleaned

    def test_clean_body_removes_signature(self):
        from processor import _clean_body
        body = """My question here.

-- 
John Doe
john@example.com"""
        cleaned = _clean_body(body)
        assert "My question here" in cleaned
        assert "John Doe" not in cleaned


class TestSmtpReplyHandling:

    def test_send_reply_writes_local_file_when_outbound_disabled(self, tmp_path, monkeypatch):
        import smtp_server

        reply_path = tmp_path / "latest_reply.txt"
        monkeypatch.setattr(smtp_server, "DISABLE_OUTBOUND_EMAIL", True)
        monkeypatch.setattr(smtp_server, "LATEST_REPLY_PATH", reply_path)

        sendmail_called = False

        class FailingSMTP:
            def __init__(self, *args, **kwargs):
                nonlocal sendmail_called
                sendmail_called = True
                raise AssertionError("SMTP should not be opened when outbound email is disabled")

        monkeypatch.setattr(smtp_server.smtplib, "SMTP", FailingSMTP)

        smtp_server._send_reply("user@example.com", "Re: Test", "Local reply body")

        assert sendmail_called is False
        assert reply_path.exists()
        saved = reply_path.read_text(encoding="utf-8")
        assert "To: user@example.com" in saved
        assert "Subject: Re: Test" in saved
        assert "Local reply body" in saved


class TestAttachmentExtractor:

    def test_vision_math_expression_is_normalized_and_computed_without_ocr(self, monkeypatch):
        import attachment_extractor

        monkeypatch.setattr(
            attachment_extractor,
            "extract_math_from_image",
            lambda _: {"expression": "6 ÷ 2(1+2)", "confidence": "high"},
        )
        monkeypatch.setattr(
            attachment_extractor,
            "_extract_image_text",
            lambda _: (_ for _ in ()).throw(AssertionError("OCR should not run when vision math succeeds")),
        )

        result = attachment_extractor.extract_text_from_attachment(
            "equation.png",
            "image/png",
            b"fake-image-bytes",
        )

        assert "Interpreted expression from the image:" in result["text"]
        assert "6 ÷ 2(1+2)" in result["text"]
        assert "Normalized:\n6/2*(1+2)" in result["text"]
        assert "Computed result:" in result["text"]
        assert "9" in result["text"]
        assert result["math_expression"] == "6 ÷ 2(1+2)"
        assert result["normalized_math_expression"] == "6/2*(1+2)"
        assert result["computed_result"] == "9"
        assert result["math_expression_uncertain"] is False
        assert result["math_source"] == "vision"

    def test_ocr_fallback_keeps_uncertain_math_guardrail_when_vision_fails(self, monkeypatch):
        import attachment_extractor

        monkeypatch.setattr(
            attachment_extractor,
            "extract_math_from_image",
            lambda _: {"expression": "", "confidence": "low"},
        )
        monkeypatch.setattr(attachment_extractor, "_extract_image_text", lambda _: "6+2*(14+2)")

        result = attachment_extractor.extract_text_from_attachment(
            "equation.png",
            "image/png",
            b"fake-image-bytes",
        )

        assert "OCR detected math expression:" in result["text"]
        assert "6+2*(14+2)" in result["text"]
        assert "Computed result:\nUnavailable" in result["text"]
        assert "OCR extraction may be inaccurate. Please verify the expression from the image." in result["text"]
        assert result["math_expression_uncertain"] is True
        assert result["computed_result"] is None
        assert result["math_confirmation_expression"] == "6 ÷ 2(1+2)"
        assert result["math_source"] == "ocr"


class TestVisionMath:

    def test_extract_math_from_image_returns_expression_from_llava(self, monkeypatch):
        import vision_math

        class FakeClient:
            def __init__(self, host=None):
                self.host = host

            def chat(self, model, messages, options=None):
                return {"message": {"content": "6 ÷ 2(1+2)"}}

        monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(Client=FakeClient))

        result = vision_math.extract_math_from_image("/tmp/equation.png")

        assert result["expression"] == "6 ÷ 2(1+2)"
        assert result["confidence"] == "high"

    def test_extract_math_from_image_uses_anthropic_before_ollama(self, monkeypatch, tmp_path, caplog):
        import vision_math

        image_path = tmp_path / "equation.png"
        image_path.write_bytes(b"fake-image-bytes")

        class FakeBlock:
            text = "6 ÷ 2(1+2)"

        class FakeMessage:
            content = [FakeBlock()]

        class FakeClient:
            def __init__(self, api_key=None):
                assert api_key == "test-key"
                self.messages = types.SimpleNamespace(create=lambda **kwargs: FakeMessage())

        monkeypatch.setattr(vision_math, "USE_ANTHROPIC_API", True)
        monkeypatch.setattr(vision_math, "ANTHROPIC_AVAILABLE", True)
        monkeypatch.setattr(vision_math, "ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(vision_math, "anthropic", types.SimpleNamespace(Anthropic=FakeClient))
        monkeypatch.setitem(
            sys.modules,
            "ollama",
            types.SimpleNamespace(
                Client=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Ollama should not run"))
            ),
        )

        with caplog.at_level(logging.INFO):
            result = vision_math.extract_math_from_image(str(image_path))

        assert result["expression"] == "6 ÷ 2(1+2)"
        assert result["confidence"] == "high"
        assert "[VISION] Anthropic vision in" in caplog.text


class TestLLM:

    def setup_method(self):
        import llm

        llm.USE_ANTHROPIC_API = False
        llm.ANTHROPIC_API_KEY = ""
        llm.ANTHROPIC_AVAILABLE = False
        llm.anthropic = None

    def test_query_ollama_uses_passed_num_predict(self, monkeypatch):
        import llm

        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"message": {"content": "ok"}}).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)

        response = llm.query_ollama("hello", [], num_predict=321)

        assert response == "ok"
        assert captured["payload"]["options"]["num_predict"] == 321

    def test_query_ollama_logs_latency(self, monkeypatch):
        import llm

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"message": {"content": "ok"}}).encode("utf-8")

        monkeypatch.setattr(llm.urllib.request, "urlopen", lambda request, timeout: FakeResponse())
        timeline = [10.0, 12.3]

        def fake_time():
            if timeline:
                return timeline.pop(0)
            return 12.3

        monkeypatch.setattr(llm.time, "time", fake_time)
        logged_messages = []

        def fake_info(message, *args):
            logged_messages.append(message % args if args else message)

        monkeypatch.setattr(llm.logger, "info", fake_info)

        response = llm.query_ollama("hello", [])

        assert response == "ok"
        assert "[LLM] Response generated in 2.3s (2 chars)" in logged_messages

    def test_demo_model_override_is_used(self, monkeypatch):
        import llm

        captured = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps({"message": {"content": "ok"}}).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.setattr(llm, "DEMO_FAST_MODEL", "llama3.2:3b")
        monkeypatch.setattr(llm, "OLLAMA_MODEL", "llama3.1:8b")

        llm.query_ollama("hello", [])

        assert captured["payload"]["model"] == "llama3.2:3b"

    def test_anthropic_success(self, monkeypatch, caplog):
        import llm

        class FakeContent:
            text = "Anthropic reply"

        class FakeMessage:
            content = [FakeContent()]

        class FakeClient:
            def __init__(self, api_key=None):
                self.api_key = api_key
                assert api_key == "test-key"

            class messages:
                @staticmethod
                def create(**kwargs):
                    return FakeMessage()

        monkeypatch.setattr(llm, "USE_ANTHROPIC_API", True)
        monkeypatch.setattr(llm, "ANTHROPIC_AVAILABLE", True)
        monkeypatch.setattr(llm, "ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(llm, "anthropic", types.SimpleNamespace(Anthropic=FakeClient))

        with caplog.at_level(logging.INFO):
            response = llm.generate_response("hello", [])

        assert response == "Anthropic reply"
        assert "Anthropic response in" in caplog.text

    def test_anthropic_fallback_to_ollama(self, monkeypatch, caplog):
        import llm

        class FailingClient:
            def __init__(self, api_key=None):
                self.api_key = api_key

            class messages:
                @staticmethod
                def create(**kwargs):
                    raise Exception("API error")

        monkeypatch.setattr(llm, "USE_ANTHROPIC_API", True)
        monkeypatch.setattr(llm, "ANTHROPIC_AVAILABLE", True)
        monkeypatch.setattr(llm, "ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(llm, "anthropic", types.SimpleNamespace(Anthropic=FailingClient))
        monkeypatch.setattr(llm, "_query_ollama_backend", lambda *args, **kwargs: "Ollama fallback reply")

        with caplog.at_level(logging.INFO):
            response = llm.generate_response("hello", [])

        assert response == "Ollama fallback reply"
        assert "falling back to Ollama" in caplog.text

    def test_anthropic_missing_key(self, monkeypatch):
        import llm

        monkeypatch.setattr(llm, "USE_ANTHROPIC_API", True)
        monkeypatch.setattr(llm, "ANTHROPIC_AVAILABLE", True)
        monkeypatch.setattr(llm, "ANTHROPIC_API_KEY", "")
        monkeypatch.setattr(llm, "_query_ollama_backend", lambda *args, **kwargs: "Ollama fallback reply")

        response = llm.generate_response("hello", [])

        assert response == "Ollama fallback reply"

    def test_anthropic_import_unavailable(self, monkeypatch):
        import llm

        monkeypatch.setattr(llm, "USE_ANTHROPIC_API", True)
        monkeypatch.setattr(llm, "ANTHROPIC_AVAILABLE", False)
        monkeypatch.setattr(llm, "ANTHROPIC_API_KEY", "test-key")
        monkeypatch.setattr(llm, "_query_ollama_backend", lambda *args, **kwargs: "Ollama fallback reply")

        response = llm.generate_response("hello", [])

        assert response == "Ollama fallback reply"


class TestMain:

    def test_prewarm_does_not_crash_startup_if_ollama_fails(self, monkeypatch):
        import main

        ran = {"test_mode": False}

        monkeypatch.setattr(main, "init_db", lambda: None)
        monkeypatch.setattr(main, "is_ollama_available", lambda: True)

        def fail_prewarm():
            raise RuntimeError("prewarm failed")

        monkeypatch.setattr(main, "prewarm_ollama_model", fail_prewarm)
        monkeypatch.setattr(main, "run_test_mode", lambda *args: ran.__setitem__("test_mode", True))
        monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "test"])

        main.main()

        assert ran["test_mode"] is True

    def test_main_logs_demo_mode_status(self, monkeypatch, caplog):
        import main

        monkeypatch.setattr(main, "init_db", lambda: None)
        monkeypatch.setattr(main, "is_ollama_available", lambda: False)
        monkeypatch.setattr(main, "DEMO_MODE", True)
        monkeypatch.setattr(main, "run_test_mode", lambda *args: None)
        monkeypatch.setattr(sys, "argv", ["main.py", "--mode", "test"])

        with caplog.at_level(logging.INFO):
            main.main()

        assert "[MAIN] Demo mode: enabled" in caplog.text


class TestWebAPI:

    def _client_with_tmp_db(self, monkeypatch, tmp_path):
        import database
        import web_api

        db_path = str(tmp_path / "api-test.db")
        conn = getattr(database._local, "conn", None)
        if conn is not None:
            conn.close()
            del database._local.conn

        monkeypatch.setattr(database, "DB_PATH", db_path)
        monkeypatch.setattr(web_api, "DB_PATH", db_path)
        monkeypatch.setattr(web_api, "_db_initialized", False)
        web_api.app.config.update(TESTING=True)
        web_api.init_db()
        return web_api, web_api.app.test_client()

    def _login(self, client, email="verified@example.com"):
        with client.session_transaction() as sess:
            sess["user"] = {
                "id": 1,
                "email": email,
                "name": "Verified User",
                "picture": "",
                "google_sub": "google-sub-123",
            }

    def test_api_health_returns_runtime_status(self, monkeypatch):
        import web_api

        monkeypatch.setattr(web_api, "DEMO_MODE", True)
        monkeypatch.setattr(web_api, "APP_START_TIME", 100.0)
        monkeypatch.setattr(web_api, "get_active_model", lambda: "llama3.2:3b")
        monkeypatch.setattr(web_api, "is_ollama_available", lambda: True)
        monkeypatch.setattr(web_api.time, "time", lambda: 112.8)

        client = web_api.app.test_client()
        response = client.get("/api/health")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["status"] == "ok"
        assert payload["model"] == "llama3.2:3b"
        assert payload["demo_mode"] is True
        assert payload["uptime_seconds"] == 12
        assert payload["ollama_status"] == "ok"

    def test_send_rejects_unauthenticated_request(self, monkeypatch, tmp_path):
        web_api, client = self._client_with_tmp_db(monkeypatch, tmp_path)

        response = client.post(
            "/api/send",
            json={
                "from": "attacker@example.com",
                "subject": "Should fail",
                "body": "This should not send.",
            },
        )

        assert response.status_code == 401
        assert response.get_json()["error"] == "Authentication required"

    def test_send_uses_session_email_not_frontend_from_email(self, monkeypatch, tmp_path):
        web_api, client = self._client_with_tmp_db(monkeypatch, tmp_path)
        self._login(client, email="verified@example.com")
        captured = {}

        class FakeSMTP:
            def __init__(self, host, port):
                captured["host"] = host
                captured["port"] = port

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def sendmail(self, from_email, recipients, raw_message):
                captured["from_email"] = from_email
                captured["recipients"] = recipients
                captured["raw_message"] = raw_message

        monkeypatch.setattr(web_api.smtplib, "SMTP", FakeSMTP)

        response = client.post(
            "/api/send",
            json={
                "from": "attacker@example.com",
                "subject": "Session identity",
                "body": "Use the verified account.",
            },
        )

        assert response.status_code == 200
        assert response.get_json()["success"] is True
        assert captured["from_email"] == "verified@example.com"
        assert captured["recipients"] == ["ask@offlinellm.me"]
        assert "From: verified@example.com" in captured["raw_message"]
        assert "Reply-To: verified@example.com" in captured["raw_message"]
        assert "attacker@example.com" not in captured["raw_message"]

    def test_password_signup_verifies_email_before_creating_user(self, monkeypatch, tmp_path):
        web_api, client = self._client_with_tmp_db(monkeypatch, tmp_path)
        sent = {}

        monkeypatch.setattr(web_api, "generate_verification_code", lambda: "123456")
        monkeypatch.setattr(
            web_api,
            "send_verification_email",
            lambda email, code: sent.update({"email": email, "code": code}),
        )
        picture = "data:image/png;base64,aGVsbG8="

        signup_response = client.post(
            "/api/auth/password/signup",
            json={
                "email": "NewUser@Example.com",
                "password": "TestPass123!",
                "confirmPassword": "TestPass123!",
                "name": "New User",
                "picture": picture,
            },
        )

        assert signup_response.status_code == 200
        signup_payload = signup_response.get_json()
        assert signup_payload["success"] is True
        assert signup_payload["needsVerification"] is True
        assert signup_payload["email"] == "newuser@example.com"
        assert sent == {"email": "newuser@example.com", "code": "123456"}

        conn = web_api.get_db()
        try:
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                ("newuser@example.com",),
            ).fetchone()
            pending = conn.execute(
                "SELECT * FROM email_verifications WHERE email = ?",
                ("newuser@example.com",),
            ).fetchone()
        finally:
            conn.close()

        assert user is None
        assert pending is not None
        assert pending["attempts"] == 0
        assert pending["name"] == "New User"
        assert pending["picture"] == picture

        wrong_response = client.post(
            "/api/auth/password/verify",
            json={"email": "newuser@example.com", "code": "000000"},
        )
        assert wrong_response.status_code == 400
        assert wrong_response.get_json()["error"] == "Incorrect verification code."

        conn = web_api.get_db()
        try:
            attempts = conn.execute(
                "SELECT attempts FROM email_verifications WHERE email = ?",
                ("newuser@example.com",),
            ).fetchone()["attempts"]
            user = conn.execute(
                "SELECT * FROM users WHERE email = ?",
                ("newuser@example.com",),
            ).fetchone()
        finally:
            conn.close()

        assert attempts == 1
        assert user is None

        verify_response = client.post(
            "/api/auth/password/verify",
            json={"email": "newuser@example.com", "code": "123456"},
        )

        assert verify_response.status_code == 200
        verify_payload = verify_response.get_json()
        assert verify_payload["success"] is True
        assert verify_payload["user"]["email"] == "newuser@example.com"
        assert verify_payload["user"]["name"] == "New User"
        assert verify_payload["user"]["picture"] == picture

        with client.session_transaction() as sess:
            assert sess["user"]["email"] == "newuser@example.com"
            assert "picture" not in sess["user"]

        me_response = client.get("/api/me")
        assert me_response.status_code == 200
        me_payload = me_response.get_json()
        assert me_payload["user"]["email"] == "newuser@example.com"
        assert me_payload["user"]["name"] == "New User"
        assert me_payload["user"]["picture"] == picture

        conn = web_api.get_db()
        try:
            pending = conn.execute(
                "SELECT * FROM email_verifications WHERE email = ?",
                ("newuser@example.com",),
            ).fetchone()
        finally:
            conn.close()

        assert pending is None

    def test_google_sign_in_links_existing_password_account_by_email(self, monkeypatch, tmp_path):
        web_api, _ = self._client_with_tmp_db(monkeypatch, tmp_path)
        user = web_api.create_password_user_with_hash(
            "linked@example.com",
            "hashed-password",
            "Manual Name",
            "data:image/png;base64,bWFudWFs",
        )

        linked = web_api.upsert_user(
            {
                "email": "linked@example.com",
                "name": "Google Name",
                "picture": "https://example.com/google.png",
                "google_sub": "google-sub-linked",
            }
        )

        assert linked["id"] == user["id"]
        assert linked["email"] == "linked@example.com"
        assert linked["name"] == "Manual Name"
        assert linked["picture"] == "data:image/png;base64,bWFudWFs"
        assert linked["google_sub"] == "google-sub-linked"
        assert linked["password_hash"] == "hashed-password"

    def test_linked_google_sign_in_preserves_password_account_profile(self, monkeypatch, tmp_path):
        web_api, _ = self._client_with_tmp_db(monkeypatch, tmp_path)
        user = web_api.create_password_user_with_hash(
            "manual@example.com",
            "hashed-password",
            "Manual User",
            "data:image/png;base64,bWFudWFs",
        )

        conn = web_api.get_db()
        try:
            conn.execute(
                "UPDATE users SET google_sub = ? WHERE id = ?",
                ("google-sub-manual", user["id"]),
            )
            conn.commit()
        finally:
            conn.close()

        linked = web_api.upsert_user(
            {
                "email": "google-alias@example.com",
                "name": "Google Alias",
                "picture": "https://example.com/google.png",
                "google_sub": "google-sub-manual",
            }
        )

        assert linked["id"] == user["id"]
        assert linked["email"] == "manual@example.com"
        assert linked["name"] == "Manual User"
        assert linked["picture"] == "data:image/png;base64,bWFudWFs"
        assert linked["password_hash"] == "hashed-password"

    def test_me_returns_current_user_when_logged_in(self, monkeypatch, tmp_path):
        web_api, client = self._client_with_tmp_db(monkeypatch, tmp_path)
        self._login(client, email="verified@example.com")

        response = client.get("/api/me")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["success"] is True
        assert payload["user"]["email"] == "verified@example.com"

    def test_logout_clears_session(self, monkeypatch, tmp_path):
        web_api, client = self._client_with_tmp_db(monkeypatch, tmp_path)
        self._login(client, email="verified@example.com")

        response = client.post("/api/logout")
        assert response.status_code == 200
        assert response.get_json()["success"] is True

        me_response = client.get("/api/me")
        assert me_response.status_code == 401

    def test_frontend_static_route_does_not_serve_backend_source(self, monkeypatch, tmp_path):
        web_api, client = self._client_with_tmp_db(monkeypatch, tmp_path)

        response = client.get("/ai-penpal/config.py")

        assert response.status_code == 404


class TestMathValidator:

    def test_detect_math_expression_handles_division_and_implicit_multiplication(self):
        from math_validator import detect_math_expression
        assert detect_math_expression("6 ÷ 2(1+2)") is True

    def test_normalize_math_expression_handles_division_and_implicit_multiplication(self):
        from math_validator import normalize_math_expression
        assert normalize_math_expression("6 ÷ 2(1+2)") == "6/2*(1+2)"

    def test_safe_evaluate_expression_returns_nine(self):
        from math_validator import safe_evaluate_expression
        result = safe_evaluate_expression("6 ÷ 2(1+2)")
        assert result["detected"] is True
        assert result["expression"] == "6/2*(1+2)"
        assert result["result"] == "9"
        assert result["error"] is None

    def test_is_expression_suspicious_flags_bad_ocr_math(self):
        from math_validator import is_expression_suspicious
        assert is_expression_suspicious("6+2*(14+2)") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
