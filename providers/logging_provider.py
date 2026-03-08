"""
Logging Decorator for LLM Providers
====================================
Wraps an existing LLMProvider to capture its inputs and outputs.
"""

import json
import time
from collections.abc import Sequence
from typing import Any

from llm_provider import (
    LLMProvider,
    MessageInput,
    ToolChoice,
    ToolInput,
    normalize_messages,
    normalize_tools,
)
from logger_utils import logger
from events import event_bus


def _json_default(obj: object) -> object:
    """Serialize non-JSON-native payloads for diagnostic logging."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return str(obj)


def _truncate_preview(text: str, *, max_chars: int = 120) -> str:
    """Compress one text field into a terminal-friendly single-line preview."""
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def _extract_tool_name(tool_payload: dict[str, object]) -> str:
    """Return the function name from one normalized OpenAI-style tool schema."""
    function_payload = tool_payload.get("function")
    function = function_payload if isinstance(function_payload, dict) else {}
    return str(function.get("name") or "unnamed_tool")


def _summarize_messages(messages: Sequence[MessageInput]) -> tuple[list[dict[str, object]], str]:
    """Return normalized messages plus a concise human-readable summary."""
    normalized_messages = normalize_messages(messages)
    roles = [str(message.get("role") or "unknown") for message in normalized_messages]
    last_preview = ""
    if normalized_messages:
        last_preview = _truncate_preview(str(normalized_messages[-1].get("content") or ""))
    summary = (
        f"count={len(normalized_messages)} roles={roles}"
        + (f" last_preview={last_preview!r}" if last_preview else "")
    )
    return normalized_messages, summary


def _summarize_tools(tools: Sequence[ToolInput]) -> tuple[list[dict[str, object]], str]:
    """Return normalized tools plus a concise terminal-friendly summary."""
    normalized_tools = normalize_tools(tools) or []
    tool_names = [_extract_tool_name(tool_payload) for tool_payload in normalized_tools]
    return normalized_tools, f"count={len(tool_names)} names={tool_names}"


def _extract_response_tool_names(tool_calls: Sequence[object] | None) -> list[str]:
    """Return tool-call names from a response message for concise logging."""
    resolved_names: list[str] = []
    for tool_call in tool_calls or []:
        if isinstance(tool_call, dict):
            function_payload = tool_call.get("function")
            function = function_payload if isinstance(function_payload, dict) else {}
            resolved_names.append(str(function.get("name") or "unknown_tool"))
            continue
        function = getattr(tool_call, "function", None)
        resolved_names.append(str(getattr(function, "name", "unknown_tool")))
    return resolved_names


def _serialize_response_message(response_message: Any) -> dict[str, Any]:
    """Build a stable JSON-serializable response payload for debug logging."""
    res_log: dict[str, Any] = {
        "role": getattr(response_message, "role", "assistant"),
        "content": getattr(response_message, "content", ""),
    }
    reasoning_content = getattr(response_message, "reasoning_content", None)
    if reasoning_content is None:
        reasoning_content = getattr(response_message, "reasoning", None)
    if reasoning_content:
        res_log["reasoning_content"] = reasoning_content
    tool_calls = getattr(response_message, "tool_calls", None)
    if tool_calls:
        serialized_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if isinstance(tool_call, dict):
                function_payload = tool_call.get("function")
                function = function_payload if isinstance(function_payload, dict) else {}
                serialized_tool_calls.append(
                    {
                        "id": tool_call.get("id"),
                        "function": {
                            "name": function.get("name"),
                            "arguments": function.get("arguments"),
                        },
                    }
                )
                continue
            function = getattr(tool_call, "function", None)
            serialized_tool_calls.append(
                {
                    "id": getattr(tool_call, "id", None),
                    "function": {
                        "name": getattr(function, "name", None),
                        "arguments": getattr(function, "arguments", None),
                    },
                }
            )
        res_log["tool_calls"] = serialized_tool_calls
    return res_log


def _summarize_response_message(response_message: Any) -> str:
    """Return a concise response summary suitable for terminal logging."""
    content = str(getattr(response_message, "content", "") or "")
    reasoning_content = getattr(response_message, "reasoning_content", None)
    if reasoning_content is None:
        reasoning_content = getattr(response_message, "reasoning", None)
    reasoning_text = str(reasoning_content or "")
    tool_calls = getattr(response_message, "tool_calls", None)
    tool_names = _extract_response_tool_names(tool_calls)
    summary_parts = [
        f"role={getattr(response_message, 'role', 'assistant')}",
        f"content_chars={len(content)}",
        f"reasoning_chars={len(reasoning_text)}",
        f"tool_calls={len(tool_names)}",
    ]
    if tool_names:
        summary_parts.append(f"tool_names={tool_names}")
    content_preview = _truncate_preview(content)
    if content_preview:
        summary_parts.append(f"content_preview={content_preview!r}")
    return " ".join(summary_parts)


class LoggingLLMProvider(LLMProvider):
    """
    Decorator for LLMProvider that logs all generation requests and responses.
    """

    def __init__(self, inner_provider: LLMProvider) -> None:
        self.inner_provider = inner_provider
        logger.info(
            f"LoggingLLMProvider initialized wrapping {type(inner_provider).__name__}"
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        started = time.perf_counter()
        logger.info("-" * 40)
        logger.info(f"LLM REQUEST - {type(self.inner_provider).__name__}")
        logger.info(
            "Prompt Summary: "
            f"system_chars={len(system_prompt)} user_chars={len(user_prompt)} "
            f"user_preview={_truncate_preview(user_prompt)!r}"
        )
        logger.debug(f"System Prompt:\n{system_prompt}")
        logger.debug(f"User Prompt:\n{user_prompt}")
        if max_tokens:
            logger.info(f"Max Tokens: {max_tokens}")

        try:
            response = self.inner_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "LLM RESPONSE SUMMARY: "
                f"chars={len(response)} preview={_truncate_preview(response)!r}"
            )
            logger.debug(f"LLM RESPONSE:\n{response}")
            event_bus.publish(
                "llm_call_completed",
                operation="generate",
                provider=type(self.inner_provider).__name__,
                latency_ms=latency_ms,
                input_messages=2,
                tool_count=0,
            )
            logger.info("-" * 40)
            return response
        except Exception as e:
            logger.error(f"LLM ERROR: {str(e)}")
            logger.info("-" * 40)
            raise

    def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
    ):
        started = time.perf_counter()
        logger.info("-" * 40)
        logger.info(f"LLM CHAT REQUEST - {type(self.inner_provider).__name__}")
        try:
            normalized_messages, message_summary = _summarize_messages(messages)
            logger.info(f"Message Summary: {message_summary}")
            logger.debug(
                "Messages: "
                f"{json.dumps(normalized_messages, indent=2, default=_json_default)}"
            )
        except Exception as e:
            logger.warning(f"Failed to log messages: {e}")

        if tools:
            normalized_tools, tool_summary = _summarize_tools(tools)
            logger.info(f"Tool Summary: {tool_summary}")
            logger.debug(f"Tools: {json.dumps(normalized_tools, indent=2)}")
        if tool_choice:
            logger.info(f"Tool Choice: {tool_choice}")

        try:
            response_message = self.inner_provider.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
            )
            logger.info(
                "LLM CHAT RESPONSE SUMMARY: "
                f"{_summarize_response_message(response_message)}"
            )
            logger.debug(
                "LLM CHAT RESPONSE:\n"
                f"{json.dumps(_serialize_response_message(response_message), indent=2, default=_json_default)}"
            )
            latency_ms = int((time.perf_counter() - started) * 1000)
            event_bus.publish(
                "llm_call_completed",
                operation="chat",
                provider=type(self.inner_provider).__name__,
                latency_ms=latency_ms,
                input_messages=len(messages),
                tool_count=len(tools) if tools else 0,
            )
            logger.info("-" * 40)
            return response_message
        except Exception as e:
            logger.error(f"LLM CHAT ERROR: {str(e)}")
            logger.info("-" * 40)
            raise
