"""
processor.py — Core pipeline
History isolation: only the messages passed in the email thread are used.
The subject line acts as a conversation identifier to keep contexts separate.
"""

import logging
import re
from database import (
    clear_conversation_summary,
    get_history,
    get_conversation_summary,
    append_to_history,
    clear_history,
    queue_outgoing,
    save_conversation_summary,
)
from llm import query_ollama
from config import (
    DEFAULT_NUM_PREDICT,
    DOCUMENT_NUM_PREDICT,
    MAX_HISTORY_EXCHANGES,
    MAX_ATTACHMENT_TEXT_CHARS,
    MAX_RESUME_TEXT_CHARS,
    MAX_TOTAL_EXTRACTED_CHARS,
    MATH_NUM_PREDICT,
    SUMMARY_NUM_PREDICT,
)
from math_validator import detect_math_expression, safe_evaluate_expression

logger = logging.getLogger(__name__)

RESET_KEYWORDS = {"reset", "start over", "clear history", "new conversation"}
STEP_BY_STEP_REQUEST_KEYWORDS = (
    "step by step",
)
DOCUMENT_SOLUTION_KEYWORDS = (
    "how to solve",
    "explain step by step",
    "explain this step by step",
    "what is the solution",
)
DOCUMENT_SUMMARY_KEYWORDS = (
    "summarize",
    "summary",
    "what is this",
)
DOCUMENT_TYPE_PATTERNS = {
    "resume": (
        "resume",
        "curriculum vitae",
        "work experience",
        "professional experience",
        "education",
        "skills",
    ),
    "technical report": (
        "technical report",
        "abstract",
        "methodology",
        "results",
        "conclusion",
        "references",
    ),
    "assignment": (
        "assignment",
        "homework",
        "student name",
        "course",
        "question 1",
        "problem 1",
    ),
}
SECTION_BASED_DOCUMENT_TYPES = {"resume", "technical report"}
EXPECTED_GRADUATION_PATTERNS = (
    "expected graduation",
    "expected date",
    "expected completion",
)
RESUME_COMPLETION_CLAIM_PATTERNS = (
    re.compile(r"\b(?:has\s+)?completed\b[^.?!\n]*\bdegree\b", re.IGNORECASE),
    re.compile(r"\bgraduated with\b[^.?!\n]*\bdegree\b", re.IGNORECASE),
)
RESUME_DEGREE_PATTERNS = (
    re.compile(r"\b(Bachelor(?:'s)?(?:\s+(?:degree|of|in)\b[^,\n.;:]*)?)", re.IGNORECASE),
    re.compile(r"\b(Master(?:'s)?(?:\s+(?:degree|of|in)\b[^,\n.;:]*)?)", re.IGNORECASE),
    re.compile(r"\b(Associate(?:'s)?(?:\s+(?:degree|of|in)\b[^,\n.;:]*)?)", re.IGNORECASE),
    re.compile(r"\b(Ph\.?D\.?(?:\s+in\b[^,\n.;:]*)?)", re.IGNORECASE),
    re.compile(r"\b(Doctor(?:ate)?(?:\s+(?:degree|of|in)\b[^,\n.;:]*)?)", re.IGNORECASE),
)
EXPECTED_DATE_REGEX = re.compile(
    r"expected\s+(?:graduation|date|completion)\s*[:\-]?\s*([A-Za-z]+\s+\d{4}|\d{4})",
    re.IGNORECASE,
)
SUMMARY_CHAR_LIMIT = 1500
MEMORY_ATTACHMENT_CHAR_LIMIT = 2000
MAX_SUMMARY_CHARS_FOR_LLM = 2000


def process_message(
    transport_id: str,
    subject: str,
    body: str,
    attachment_results=None,
    *,
    save_history: bool = True,
    queue_reply: bool = True,
) -> dict:
    logger.info(f"[PROCESSOR] Processing message from {transport_id}")
    llm_call_count = 0

    def _call_llm_once(prompt_text: str, history_messages: list[dict], num_predict: int):
        nonlocal llm_call_count
        llm_call_count += 1
        assert llm_call_count == 1, "query_ollama called more than once for a single request"
        return query_ollama(prompt_text, history_messages, num_predict=num_predict)

    cleaned_body = _clean_body(body)
    convo_key = subject.replace("Re: ", "").replace("Re:", "").strip()
    session_key = f"{transport_id}::{convo_key}"

    if cleaned_body.lower().strip() in RESET_KEYWORDS:
        reset_message = "Your conversation history has been cleared. You can start a fresh conversation now."
        if save_history:
            clear_history(session_key)
            clear_history(transport_id)
            clear_conversation_summary(transport_id, convo_key)
        reply = _format_reply(reset_message, transport_id) if queue_reply else reset_message
        reply_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject
        if queue_reply:
            queue_outgoing(transport_id, reply_subject, reply)
        return {"success": True, "reply_subject": reply_subject, "reply_body": reply}

    history = get_history(session_key) if save_history else []
    has_history = len(history) > 0
    conversation_summary = get_conversation_summary(transport_id, convo_key) if save_history else ""

    logger.info(
        f"[PROCESSOR] {transport_id} convo='{convo_key}' — "
        f"{'continuing' if has_history else 'new'} ({len(history)//2} exchanges)"
    )
    logger.info("[PROCESSOR] Loaded %s stored exchanges", len(history) // 2)
    logger.info("[PROCESSOR] Loaded conversation summary chars=%s", len(conversation_summary))

    readable_texts = _collect_readable_attachment_texts(attachment_results or [])
    attachment_text_length = sum(len(text) for text in readable_texts)
    trimmed_history = trim_history_for_llm(
        history,
        has_attachments=bool(readable_texts),
        attachment_text_length=attachment_text_length,
    )
    logger.info("[PROCESSOR] Using %s history exchanges for LLM", len(trimmed_history) // 2)

    math_contexts = _collect_math_contexts(attachment_results or [])
    _log_math_contexts(math_contexts)
    inferred_document_type = _infer_document_type(readable_texts)
    resume_context = _extract_resume_expected_context(readable_texts, inferred_document_type)
    stored_user_message = _build_stored_user_message(cleaned_body, attachment_results or [])
    num_predict = _select_num_predict(cleaned_body, attachment_results or [], math_contexts)

    prompt, has_attachments, readable_attachment_count, used_summary = _build_prompt(
        cleaned_body,
        attachment_results or [],
        conversation_summary=conversation_summary,
    )
    if used_summary:
        logger.info("[PROCESSOR] Using memory summary for LLM")

    if has_attachments and readable_attachment_count == 0:
        reply = _format_attachment_unreadable_reply()
        reply_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject
        if queue_reply:
            queue_outgoing(transport_id, reply_subject, reply)
        return {"success": True, "reply_subject": reply_subject, "reply_body": reply}

    if _has_vision_math(math_contexts):
        llm_response = _build_vision_math_reply(math_contexts)
    elif _has_uncertain_math(math_contexts):
        llm_response = _build_uncertain_math_reply(math_contexts)
    else:
        llm_response = _call_llm_once(prompt, trimmed_history, num_predict)

        if llm_response is None:
            error_reply = _format_error_reply()
            reply_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject
            if queue_reply:
                queue_outgoing(transport_id, reply_subject, error_reply)
            return {"success": False, "reply_subject": reply_subject, "reply_body": error_reply, "error": "LLM inference failed"}

        llm_response = _apply_math_verification(llm_response, math_contexts)
        llm_response = _apply_resume_expected_graduation_guard(llm_response, resume_context)

    if save_history:
        # Save history under the conversation-specific key
        append_to_history(session_key, "user", stored_user_message)
        append_to_history(session_key, "assistant", llm_response)

        # Also save under the plain email key so /api/history still works
        append_to_history(transport_id, "user", stored_user_message)
        append_to_history(transport_id, "assistant", llm_response)

        update_conversation_summary(
            transport_id,
            convo_key,
            _build_memory_user_message(cleaned_body, attachment_results or []),
            llm_response,
        )
    logger.info("[PROCESSOR] LLM call count = %s", llm_call_count)

    reply = _format_reply(llm_response, transport_id) if queue_reply else llm_response
    reply_subject = f"Re: {subject}" if not subject.startswith("Re:") else subject
    if queue_reply:
        queue_outgoing(transport_id, reply_subject, reply)

    if queue_reply:
        logger.info(f"[PROCESSOR] Reply queued for {transport_id}")
    else:
        logger.info(f"[PROCESSOR] Reply generated without queue for {transport_id}")

    return {"success": True, "reply_subject": reply_subject, "reply_body": reply}


def _build_prompt(
    cleaned_body: str,
    attachment_results: list,
    conversation_summary: str = "",
    apply_attachment_caps: bool = True,
) -> tuple[str, bool, int, bool]:
    conversation_summary = _guard_summary_for_llm(conversation_summary)
    if not attachment_results:
        prompt_sections = []
        selected_summary = _select_memory_summary(conversation_summary, cleaned_body, has_attachments=False)
        use_summary = bool(selected_summary)
        if use_summary:
            prompt_sections.append(f"Conversation memory summary:\n{selected_summary}")
        prompt_sections.append(
            f"Current user request:\n{cleaned_body or '(no email body provided)'}"
        )
        return "\n\n".join(prompt_sections), False, 0, use_summary

    attachment_sections = []
    total_chars = 0
    readable_texts = _collect_readable_attachment_texts(attachment_results)
    math_contexts = _collect_math_contexts(attachment_results)

    for result in attachment_results:
        extracted_text = (result.get("text") or "").strip()
        if not extracted_text:
            continue

        prompt_text = (
            _cap_attachment_text_for_prompt(result, extracted_text)
            if apply_attachment_caps
            else extracted_text
        )
        if not prompt_text:
            continue

        remaining = MAX_TOTAL_EXTRACTED_CHARS - total_chars
        if remaining <= 0:
            break

        trimmed_text = prompt_text[:remaining].rstrip()
        attachment_sections.append(_format_attachment_section(result, trimmed_text))
        total_chars += len(trimmed_text)

    if not attachment_sections:
        prompt_sections = []
        use_summary = bool(_select_memory_summary(conversation_summary, cleaned_body, has_attachments=True))
        selected_summary = _select_memory_summary(conversation_summary, cleaned_body, has_attachments=True)
        if use_summary:
            prompt_sections.append(f"Conversation memory summary:\n{selected_summary}")
        prompt_sections.append(
            f"Current user request:\n{cleaned_body or '(no email body provided)'}"
        )
        return "\n\n".join(prompt_sections), True, 0, use_summary

    user_request = cleaned_body or "(no email body provided)"
    attachment_block = "\n\n".join(attachment_sections)
    inferred_document_type = _infer_document_type(readable_texts)
    resume_context = _extract_resume_expected_context(readable_texts, inferred_document_type)
    if math_contexts:
        instruction = _math_instruction_for_request(cleaned_body, math_contexts)
    else:
        instruction = _document_instruction_for_request(cleaned_body, inferred_document_type, resume_context)
    selected_summary = _select_memory_summary(conversation_summary, cleaned_body, has_attachments=True)
    prompt_sections = []
    use_summary = bool(selected_summary)
    if use_summary:
        prompt_sections.append(f"Conversation memory summary:\n{selected_summary}")
    prompt_sections.append(f"Current user request:\n{user_request}")
    prompt_sections.append(f"Attached document content:\n{attachment_block}")
    prompt_sections.append(f"Instruction:\n{instruction}")
    prompt = "\n\n".join(prompt_sections)
    return prompt, True, len(attachment_sections), use_summary


def _clean_body(body: str) -> str:
    lines = body.splitlines()
    cleaned = []
    for line in lines:
        if line.startswith(">"):
            break
        if line.strip().startswith("On ") and "wrote:" in line:
            break
        if line.strip() in {"--", "-- "}:
            break
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def _format_attachment_section(result: dict, trimmed_text: str) -> str:
    filename = result.get("filename", "attachment")
    return f"File: {filename}\n{trimmed_text}"


def _format_reply(content: str, transport_id: str) -> str:
    normalized_content = _markdown_to_plain_text(_strip_subject_lines(content))
    return (
        f"{normalized_content}\n\n"
        f"---\n"
        f"OfflineLLM | Reply to continue your conversation\n"
        f"Send 'reset' to start a new conversation"
    )


def _format_error_reply() -> str:
    return (
        "Sorry, I was unable to process your message at this time. "
        "Please try sending your message again.\n\n"
        "---\nOfflineLLM | Automated Response"
    )


def _format_attachment_unreadable_reply() -> str:
    return (
        "I received your attachment, but I could not extract readable text from it. "
        "Please resend the file as a readable PDF, DOCX, PNG, or JPEG, or paste the text into the email body.\n\n"
        "---\nOfflineLLM | Automated Response"
    )


def _strip_subject_lines(content: str) -> str:
    lines = content.splitlines()
    index = 0

    while index < len(lines) and not lines[index].strip():
        index += 1

    while index < len(lines) and lines[index].strip().lower().startswith("subject:"):
        index += 1
        while index < len(lines) and not lines[index].strip():
            index += 1

    stripped = "\n".join(lines[index:]).strip()
    return stripped or content.strip()


def _markdown_to_plain_text(content: str) -> str:
    plain_text = content or ""

    def replace_heading(match: re.Match) -> str:
        heading_text = match.group(1).strip().upper()
        underline = "-" * max(11, len(heading_text))
        return f"{heading_text}\n{underline}"

    plain_text = re.sub(r"(?m)^\s*#{1,2}\s*(.+?)\s*$", replace_heading, plain_text)
    plain_text = re.sub(r"\*\*(.+?)\*\*", lambda match: match.group(1).upper(), plain_text)
    plain_text = re.sub(r"(?m)^\s*-\s+(.*)$", r"• \1", plain_text)
    plain_text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", lambda match: match.group(1), plain_text)
    plain_text = re.sub(r"(?<!_)_([^_\n]+)_(?!_)", lambda match: match.group(1), plain_text)
    plain_text = re.sub(r"(?<![\d)])\*(?![\d(])", "", plain_text)
    plain_text = re.sub(r"(?<!\w)_|_(?!\w)", "", plain_text)
    return plain_text


def trim_history_for_llm(history: list[dict], has_attachments: bool = False, attachment_text_length: int = 0) -> list[dict]:
    if attachment_text_length > 10000:
        return []

    if has_attachments:
        return []

    max_exchanges = min(MAX_HISTORY_EXCHANGES, 1)
    if max_exchanges <= 0:
        return []

    max_messages = max_exchanges * 2
    return history[-max_messages:]


def update_conversation_summary(
    sender_email: str,
    conversation_key: str,
    user_message: str,
    assistant_reply: str,
) -> str:
    previous_summary = get_conversation_summary(sender_email, conversation_key)
    summary_parts = []
    if previous_summary:
        summary_parts.append(previous_summary.strip())

    normalized_user = _normalize_summary_text(user_message)
    if normalized_user:
        summary_parts.append(f"User: {normalized_user}")

    normalized_assistant = _normalize_summary_text(assistant_reply)
    if normalized_assistant:
        summary_parts.append(f"Assistant: {normalized_assistant}")

    final_summary = _truncate_summary("\n".join(summary_parts))
    save_conversation_summary(sender_email, conversation_key, final_summary)
    logger.info("[PROCESSOR] Updated conversation summary chars=%s", len(final_summary))
    return final_summary


def _truncate_summary(summary: str) -> str:
    normalized = (summary or "").strip()
    if len(normalized) <= SUMMARY_CHAR_LIMIT:
        return normalized
    trimmed = normalized[-SUMMARY_CHAR_LIMIT:].lstrip()
    if len(trimmed) < len(normalized):
        return f"...{trimmed[3:]}" if len(trimmed) > 3 else trimmed
    return trimmed


def _normalize_summary_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _guard_summary_for_llm(summary: str) -> str:
    normalized_summary = (summary or "").strip()
    if len(normalized_summary) <= MAX_SUMMARY_CHARS_FOR_LLM:
        return normalized_summary

    logger.info(
        "[PROCESSOR] Truncated conversation summary from %s chars to %s chars for LLM",
        len(normalized_summary),
        MAX_SUMMARY_CHARS_FOR_LLM,
    )
    return normalized_summary[-MAX_SUMMARY_CHARS_FOR_LLM:]


def _select_memory_summary(conversation_summary: str, cleaned_body: str, has_attachments: bool = False) -> str:
    del cleaned_body
    normalized_summary = _guard_summary_for_llm(conversation_summary)
    if not normalized_summary:
        return ""
    if has_attachments and len(normalized_summary) > SUMMARY_CHAR_LIMIT:
        return ""
    return normalized_summary


def _build_stored_user_message(cleaned_body: str, attachment_results: list[dict]) -> str:
    if not attachment_results:
        return cleaned_body

    prompt, _, _, _ = _build_prompt(
        cleaned_body,
        attachment_results,
        conversation_summary="",
        apply_attachment_caps=False,
    )
    return prompt


def _build_memory_user_message(cleaned_body: str, attachment_results: list[dict]) -> str:
    base_sections = [f"Current user request:\n{cleaned_body or '(no email body provided)'}"]

    if not attachment_results:
        return "\n\n".join(base_sections)

    attachment_sections = []
    total_chars = 0

    for result in attachment_results:
        extracted_text = (result.get("text") or "").strip()
        if not extracted_text:
            continue

        remaining = MEMORY_ATTACHMENT_CHAR_LIMIT - total_chars
        if remaining <= 0:
            break

        trimmed_text = extracted_text[:remaining].rstrip()
        attachment_sections.append(_format_attachment_section(result, trimmed_text))
        total_chars += len(trimmed_text)

    if attachment_sections:
        base_sections.append("Attached document content:\n" + "\n\n".join(attachment_sections))

    return "\n\n".join(base_sections)


def _select_num_predict(cleaned_body: str, attachment_results: list[dict], math_contexts: list[dict]) -> int:
    normalized_request = cleaned_body.lower().strip()

    if math_contexts or _looks_like_math_text_request(cleaned_body):
        return MATH_NUM_PREDICT

    if any(keyword in normalized_request for keyword in DOCUMENT_SUMMARY_KEYWORDS):
        return SUMMARY_NUM_PREDICT

    if attachment_results:
        return DOCUMENT_NUM_PREDICT

    return DEFAULT_NUM_PREDICT


def _looks_like_math_text_request(cleaned_body: str) -> bool:
    normalized_request = cleaned_body.lower()
    return detect_math_expression(cleaned_body) or any(
        keyword in normalized_request for keyword in ("equation", "calculate", "solve", "math")
    )


def _cap_attachment_text_for_prompt(result: dict, extracted_text: str) -> str:
    if not extracted_text:
        return ""

    inferred_type = _infer_document_type([extracted_text])
    cap = MAX_RESUME_TEXT_CHARS if inferred_type == "resume" else MAX_ATTACHMENT_TEXT_CHARS
    if len(extracted_text) <= cap:
        return extracted_text

    logger.info("[PROCESSOR] Attachment text capped from %s chars to %s chars", len(extracted_text), cap)
    return extracted_text[:cap].rstrip()


def _document_instruction_for_request(
    cleaned_body: str,
    inferred_document_type: str | None,
    resume_context: dict | None = None,
) -> str:
    normalized_request = cleaned_body.lower().strip()
    confidence_instruction = _document_confidence_instruction(inferred_document_type)
    resume_instruction = _resume_expected_completion_instruction(resume_context)

    if any(keyword in normalized_request for keyword in STEP_BY_STEP_REQUEST_KEYWORDS):
        return (
            "Answer the user's request using the attached document content. "
            "Provide a structured explanation with clear step-by-step guidance. "
            "Start with the sentence: Below is a step-by-step explanation of the attached document. "
            "Use Step 1, Step 2, Step 3 labels. "
            "Each step should represent a clear reasoning or explanation stage. "
            f"{_strict_step_structure_instruction(inferred_document_type)}"
            "Explain how the answer follows from the document. "
            "Explicitly connect each major point or step to the relevant document content. "
            "Do not use filler such as 'indeed' or 'I'll break it down step by step for you'. "
            f"{confidence_instruction}"
            f"{resume_instruction}"
            "If the answer is not in the document, say that clearly. "
            "Do not invent information."
        )

    if any(keyword in normalized_request for keyword in DOCUMENT_SOLUTION_KEYWORDS):
        return (
            "Answer the user's request using the attached document content. "
            "Provide a structured explanation with clear step-by-step guidance. "
            "Use explicit labels such as Step 1, Step 2, and Step 3. "
            "Each step should represent a clear reasoning or explanation stage. "
            f"{_step_structure_alignment_instruction(inferred_document_type)}"
            "Explain how the solution follows from the document. "
            "Explicitly connect each major point or step to the relevant document content. "
            f"{confidence_instruction}"
            f"{resume_instruction}"
            "If the answer is not in the document, say that clearly. "
            "Do not invent information."
        )

    if any(keyword in normalized_request for keyword in DOCUMENT_SUMMARY_KEYWORDS):
        return (
            "Summarize the attached document in a clear, concise way based on the user's request. "
            "Focus on what the document is, its main idea, and the most important details. "
            f"{confidence_instruction}"
            f"{resume_instruction}"
            "If the document does not contain enough information, say that clearly. "
            "Do not invent information."
        )

    return (
        "Answer the user's request using the attached document content. "
        f"{confidence_instruction}"
        f"{resume_instruction}"
        "If the user did not ask a clear question, summarize the document. "
        "If the answer is not in the document, say that clearly. "
        "Do not invent information."
    )


def _math_instruction_for_request(cleaned_body: str, math_contexts: list[dict]) -> str:
    uncertain_math = any(context["uncertain"] for context in math_contexts)
    if any(context["source"] == "vision" for context in math_contexts):
        return (
            "Answer the user's request using the attached vision-extracted math content. "
            "Use the computed result as the authoritative answer. "
            "Explain the solution step by step using the normalized expression exactly as shown after 'Normalized'. "
            "Use Step 1, Step 2, and Step 3 labels when walking through the calculation. "
            "Do not invent, replace, or simplify to a different expression than the interpreted expression from the image. "
            "Do not invent information."
        )
    if uncertain_math:
        return (
            "Answer the user's request using the attached OCR math content. "
            "OCR math extraction is uncertain. Ask the user to confirm the expression before solving, or solve only if the expression is clearly readable. "
            "Do not present any computed result as final while the OCR math extraction is uncertain. "
            "If a likely expression is available, ask the user whether that exact expression is correct. "
            "Do not give a final answer or say 'therefore the result is X' while the OCR math extraction is uncertain. "
            "Do not invent, replace, or simplify to a different expression than the OCR content unless you are explicitly asking the user to confirm it. "
            "Do not invent information."
        )

    return (
        "Answer the user's request using the attached OCR math content. "
        "Use the computed result as the authoritative answer. "
        "Explain the solution step by step using the normalized expression exactly as shown after 'OCR detected math expression'. "
        "Use Step 1, Step 2, and Step 3 labels when walking through the calculation. "
        "Do not invent, replace, or simplify to a different expression than the normalized OCR expression. "
        "If the image content is not enough to answer confidently, say so clearly. "
        "Do not invent information."
    )


def _collect_math_contexts(attachment_results: list[dict]) -> list[dict]:
    contexts = []

    for result in attachment_results:
        expression = result.get("math_expression")
        if not expression:
            continue

        contexts.append(
            {
                "filename": result.get("filename", "attachment"),
                "expression": expression,
                "normalized_expression": result.get("normalized_math_expression") or expression,
                "source": result.get("math_source", "ocr"),
                "uncertain": bool(result.get("math_expression_uncertain")),
                "note": result.get("math_expression_note"),
                "computed_result": result.get("computed_result"),
                "confirmation_expression": result.get("math_confirmation_expression"),
            }
        )

    return contexts


def _log_math_contexts(math_contexts: list[dict]):
    for context in math_contexts:
        logger.info(
            "[PROCESSOR] math expression source=%s filename=%s expression=%s normalized=%s computed_result=%s uncertain=%s",
            context["source"],
            context["filename"],
            context["expression"],
            context.get("normalized_expression"),
            context.get("computed_result") or "n/a",
            context["uncertain"],
        )


def _apply_math_verification(llm_response: str, math_contexts: list[dict]) -> str:
    verified_contexts = [context for context in math_contexts if context.get("computed_result")]
    if len(verified_contexts) != 1:
        return _append_math_uncertainty_note(llm_response, math_contexts)

    context = verified_contexts[0]
    computed_result = context["computed_result"]
    llm_result = _extract_numeric_answer(llm_response)

    if llm_result is not None and _numeric_strings_match(llm_result, computed_result):
        return _append_math_uncertainty_note(llm_response, math_contexts)

    if llm_result is None:
        verified_response = (
            f"Below is the verified result for the extracted expression.\n\n"
            f"Expression: {context['expression']}\n"
            f"Answer: {computed_result}"
        )
        return _append_math_uncertainty_note(verified_response, math_contexts)

    if llm_result is not None and not _numeric_strings_match(llm_result, computed_result):
        logger.warning(
            "[PROCESSOR] LLM arithmetic mismatch for %s: llm=%s computed=%s",
            context["filename"],
            llm_result,
            computed_result,
        )
        verified_response = (
            f"Below is the verified result for the extracted expression.\n\n"
            f"Expression: {context['expression']}\n"
            f"Answer: {computed_result}\n\n"
            f"I used a direct arithmetic check because the language-model answer did not match the computed result."
        )
        return _append_math_uncertainty_note(verified_response, math_contexts)

    return _append_math_uncertainty_note(llm_response, math_contexts)


def _append_math_uncertainty_note(response: str, math_contexts: list[dict]) -> str:
    if not any(context["uncertain"] for context in math_contexts):
        return response
    if "OCR extraction may be inaccurate. Please verify the expression from the image." in response:
        return response

    return (
        f"{response}\n\n"
        f"Note: OCR extraction may be inaccurate. Please verify the expression from the image."
    )


def _extract_numeric_answer(text: str) -> str | None:
    matches = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1]


def _numeric_strings_match(left: str, right: str) -> bool:
    try:
        return abs(float(left) - float(right)) < 1e-9
    except ValueError:
        return left.strip() == right.strip()


def _has_uncertain_math(math_contexts: list[dict]) -> bool:
    return any(context["uncertain"] for context in math_contexts)


def _has_vision_math(math_contexts: list[dict]) -> bool:
    return any(
        context["source"] == "vision"
        and context.get("normalized_expression")
        and context.get("computed_result")
        for context in math_contexts
    )


def _build_vision_math_reply(math_contexts: list[dict]) -> str:
    if not math_contexts:
        return _format_attachment_unreadable_reply()

    context = next(
        (
            candidate
            for candidate in math_contexts
            if candidate["source"] == "vision" and candidate.get("computed_result")
        ),
        math_contexts[0],
    )
    original_expression = context["expression"]
    normalized_expression = context.get("normalized_expression") or context["expression"]
    computed_result = context.get("computed_result") or "Unavailable"
    step_block = _build_math_step_explanation(normalized_expression, computed_result)

    return (
        f"Interpreted expression from the image:\n"
        f"{original_expression}\n\n"
        f"Normalized:\n"
        f"{normalized_expression}\n\n"
        f"Step-by-step:\n"
        f"{step_block}\n\n"
        f"Final Answer:\n"
        f"{computed_result}"
    )


def _build_uncertain_math_reply(math_contexts: list[dict]) -> str:
    if not math_contexts:
        return (
            "I could not confidently read the expression from the image. "
            "Please resend the image more clearly or type the expression in plain text."
        )

    context = math_contexts[0]
    confirmation_expression = context.get("confirmation_expression") or context["expression"]
    extracted_expression = context["expression"]
    return (
        "I attempted to extract the math expression from the image, but the OCR result may be incorrect.\n\n"
        f"Extracted expression:\n"
        f"{extracted_expression}\n\n"
        f"This does not look fully reliable.\n\n"
        f"Please confirm the correct expression. For example, is it:\n"
        f"{confirmation_expression} ?\n\n"
        f"Once confirmed, I will solve it step by step."
    )


def _document_confidence_instruction(inferred_document_type: str | None) -> str:
    if not inferred_document_type:
        return ""

    return (
        f"The attached document is a {inferred_document_type}. "
        f"When describing it, state that directly instead of saying it appears to be a {inferred_document_type}. "
    )


def _resume_expected_completion_instruction(resume_context: dict | None) -> str:
    if not resume_context:
        return ""

    status_phrase = _resume_expected_status_phrase(resume_context)
    return (
        f"The resume shows the candidate {status_phrase}. "
        "Do not say the degree is completed, has been completed, or already awarded. "
    )


def _collect_readable_attachment_texts(attachment_results: list[dict]) -> list[str]:
    readable_texts = []
    total_chars = 0

    for result in attachment_results:
        extracted_text = (result.get("text") or "").strip()
        if not extracted_text:
            continue

        remaining = MAX_TOTAL_EXTRACTED_CHARS - total_chars
        if remaining <= 0:
            break

        trimmed_text = extracted_text[:remaining].rstrip()
        readable_texts.append(trimmed_text)
        total_chars += len(trimmed_text)

    return readable_texts


def _extract_resume_expected_context(
    readable_texts: list[str],
    inferred_document_type: str | None,
) -> dict | None:
    if inferred_document_type != "resume":
        return None

    combined_text = "\n".join(readable_texts)
    lower_text = combined_text.lower()
    if not any(pattern in lower_text for pattern in EXPECTED_GRADUATION_PATTERNS):
        return None

    degree = None
    for pattern in RESUME_DEGREE_PATTERNS:
        match = pattern.search(combined_text)
        if match:
            degree = match.group(1).strip()
            break

    expected_date_match = EXPECTED_DATE_REGEX.search(combined_text)
    expected_date = expected_date_match.group(1).strip() if expected_date_match else None

    return {
        "degree": degree,
        "expected_date": expected_date,
    }


def _resume_expected_status_phrase(resume_context: dict) -> str:
    degree = resume_context.get("degree")
    expected_date = resume_context.get("expected_date")

    if degree:
        phrase = f"is currently pursuing {_with_indefinite_article(degree)}"
    else:
        phrase = "is currently pursuing the listed degree"

    if expected_date:
        phrase = f"{phrase}, expected graduation {expected_date}"
    else:
        phrase = f"{phrase}, with an expected graduation date"

    return phrase


def _with_indefinite_article(phrase: str) -> str:
    normalized = phrase.strip()
    if not normalized:
        return normalized
    if normalized.lower().startswith(("a ", "an ", "the ")):
        return normalized
    article = "an" if normalized[0].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {normalized}"


def _apply_resume_expected_graduation_guard(response: str, resume_context: dict | None) -> str:
    if not resume_context:
        return response

    if not any(pattern.search(response) for pattern in RESUME_COMPLETION_CLAIM_PATTERNS):
        return response

    factual_phrase = _resume_expected_status_phrase(resume_context)
    updated_response = response

    for pattern in RESUME_COMPLETION_CLAIM_PATTERNS:
        updated_response = pattern.sub(factual_phrase, updated_response)

    if updated_response and updated_response[0].islower():
        updated_response = updated_response[0].upper() + updated_response[1:]

    return updated_response


def _build_math_step_explanation(normalized_expression: str, computed_result: str) -> str:
    steps = [f"Step 1: Start with the normalized expression {normalized_expression}."]
    step_number = 2
    current = normalized_expression

    current, step_number = _resolve_parentheses_steps(current, steps, step_number)
    current, step_number = _resolve_operator_steps(
        current,
        "*/",
        "Apply multiplication and division from left to right",
        steps,
        step_number,
    )
    current, step_number = _resolve_operator_steps(
        current,
        "+-",
        "Apply addition and subtraction from left to right",
        steps,
        step_number,
    )

    if not steps[-1].endswith(f"{computed_result}."):
        steps.append(f"Step {step_number}: The final result is {computed_result}.")

    return "\n".join(steps)


def _resolve_parentheses_steps(expression: str, steps: list[str], step_number: int) -> tuple[str, int]:
    current = expression

    while True:
        match = re.search(r"\([^()]+\)", current)
        if not match:
            return current, step_number

        inner_expression = match.group(0)[1:-1]
        evaluation = safe_evaluate_expression(inner_expression)
        if evaluation["result"] is None:
            return current, step_number

        replacement = evaluation["result"]
        updated = current[:match.start()] + replacement + current[match.end():]
        steps.append(
            f"Step {step_number}: Evaluate {match.group(0)} = {replacement}, so the expression becomes {updated}."
        )
        current = updated
        step_number += 1


def _resolve_operator_steps(
    expression: str,
    operators: str,
    description: str,
    steps: list[str],
    step_number: int,
) -> tuple[str, int]:
    current = expression
    pattern = re.compile(rf"(-?\d+(?:\.\d+)?)([{re.escape(operators)}])(-?\d+(?:\.\d+)?)")

    while True:
        match = pattern.search(current)
        if not match:
            return current, step_number

        snippet = match.group(0)
        evaluation = safe_evaluate_expression(snippet)
        if evaluation["result"] is None:
            return current, step_number

        replacement = evaluation["result"]
        updated = current[:match.start()] + replacement + current[match.end():]
        steps.append(
            f"Step {step_number}: {description}: {snippet} = {replacement}, so the expression becomes {updated}."
        )
        current = updated
        step_number += 1


def _strict_step_structure_instruction(inferred_document_type: str | None) -> str:
    if inferred_document_type in SECTION_BASED_DOCUMENT_TYPES:
        return (
            "For section-based documents such as resumes or reports, convert the sections into numbered steps like "
            "'Step 1: Contact Information', 'Step 2: Professional Summary', and 'Step 3: Education'. "
            "Do not use unnumbered section headers for step-by-step requests. "
        )

    return (
        "Use numbered Step labels for the full explanation. "
        "Do not use unnumbered section headers for step-by-step requests. "
    )


def _step_structure_alignment_instruction(inferred_document_type: str | None) -> str:
    if inferred_document_type in SECTION_BASED_DOCUMENT_TYPES:
        return (
            "Because this document is organized by sections, either convert those sections into numbered steps "
            "or clearly say you are giving a section-by-section explanation. "
            "Make sure the wording matches the structure you actually use. "
        )

    return (
        "If the document is better explained by sections rather than procedural steps, either convert those sections "
        "into numbered steps or clearly say you are giving a section-by-section explanation. "
        "Make sure the wording matches the structure you actually use. "
    )


def _infer_document_type(readable_texts: list[str]) -> str | None:
    combined_text = "\n".join(readable_texts).lower()
    if not combined_text:
        return None

    best_type = None
    best_score = 0

    for document_type, patterns in DOCUMENT_TYPE_PATTERNS.items():
        score = sum(1 for pattern in patterns if pattern in combined_text)
        direct_title_bonus = 2 if document_type in combined_text else 0
        total_score = score + direct_title_bonus

        if total_score > best_score:
            best_type = document_type
            best_score = total_score

    if best_score >= 3:
        return best_type

    return None
