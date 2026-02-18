"""DeepSeek client construction and context-rollover operations."""

from __future__ import annotations

import logging
from typing import Any, cast

from openai import OpenAI

from .chat_logs import DeepSeekChatLogWriter
from .constants import (
    AgentToolMode,
    CONTEXT_SUMMARY_MAX_CHARS,
    CONTEXT_SUMMARY_MAX_TOKENS,
    DEEPSEEK_CHAT_MODEL,
    EndpointMode,
)
from .parsing import _normalize_assistant_content
from .prompting import _build_rollover_continuation_message
from .runtime_helpers import _error_digest, _extract_total_tokens_from_usage

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def _build_deepseek_client(
    api_key: str,
    base_url: str,
    request_timeout_seconds: float,
    http_retries: int,
) -> OpenAI:
    """Create a DeepSeek client with request timeout and transport retries."""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=request_timeout_seconds,
        max_retries=http_retries,
    )

def _clone_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Clone chat messages without mutating the source state on failed attempts."""
    return [dict(message) for message in messages]

def _summarize_conversation_context_via_deepseek(
    *,
    client: OpenAI,
    conversation_messages: list[dict[str, Any]],
    tokens_limit: int,
    endpoint_mode: EndpointMode,
    chat_log_writer: DeepSeekChatLogWriter | None = None,
) -> str:
    """Generate a compact summary of prior conversation for context rollover."""
    if not conversation_messages:
        return "No prior context."

    summary_messages = _clone_messages(conversation_messages)
    summary_messages.extend(
        [
            {
                "role": "system",
                "content": (
                    "You compress an existing assistant conversation for continuation. "
                    "Preserve task objective, schema constraints, tool state, unresolved "
                    "errors, validation/budget targets, and key factual transcript evidence."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize this conversation for a fresh context window. "
                    "Keep it concise and execution-focused."
                ),
            },
        ]
    )
    call_log_id: str | None = None
    response: Any | None = None
    if chat_log_writer is not None:
        call_log_id = chat_log_writer.start_call(
            call_type="context_rollover_summary",
            endpoint_mode=endpoint_mode,
            model=DEEPSEEK_CHAT_MODEL,
            metadata={
                "messages_count": len(summary_messages),
                "max_tokens": min(tokens_limit, CONTEXT_SUMMARY_MAX_TOKENS),
            },
        )
    try:
        response = client.chat.completions.create(
            **cast(
                Any,
                {
                    "model": DEEPSEEK_CHAT_MODEL,
                    "messages": summary_messages,
                    "max_tokens": min(tokens_limit, CONTEXT_SUMMARY_MAX_TOKENS),
                    "temperature": 0.0,
                },
            )
        )
        summary_choice = response.choices[0]
        summary_text = _normalize_assistant_content(summary_choice.message.content).strip()
        if not summary_text:
            raise ValueError("DeepSeek context summary was empty.")
        if call_log_id is not None and chat_log_writer is not None:
            chat_log_writer.write_call_event(
                call_log_id,
                {
                    "event": "message",
                    "phase": "answer",
                    "text": summary_text,
                },
            )
        if len(summary_text) > CONTEXT_SUMMARY_MAX_CHARS:
            return summary_text[:CONTEXT_SUMMARY_MAX_CHARS]
        logger.info(
            "Generated context rollover summary endpoint=%s chars=%d",
            endpoint_mode,
            len(summary_text),
        )
        return summary_text
    except Exception as error:  # noqa: BLE001
        if call_log_id is not None and chat_log_writer is not None:
            chat_log_writer.finish_call(
                call_log_id,
                status="error",
                error=_error_digest(error),
            )
            call_log_id = None
        raise
    finally:
        if call_log_id is not None and chat_log_writer is not None:
            total_tokens = _extract_total_tokens_from_usage(
                getattr(response, "usage", None)
            )
            chat_log_writer.finish_call(
                call_log_id,
                status="ok",
                finish_reason="stop",
                usage_total_tokens=total_tokens,
            )

def _reset_conversation_with_summary(
    *,
    messages: list[dict[str, Any]],
    system_prompt: str,
    summary_text: str,
    tool_mode: AgentToolMode,
) -> None:
    """Replace message history with summarized context for a fresh window."""
    messages.clear()
    messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "system",
            "content": (
                "Summarized context from previous conversation window:\n"
                f"{summary_text}"
            ),
        }
    )
    messages.append(
        {
            "role": "user",
            "content": _build_rollover_continuation_message(tool_mode),
        }
    )
