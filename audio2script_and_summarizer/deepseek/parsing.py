"""DeepSeek parsing, model policy, and error classification helpers."""

from __future__ import annotations

import json
import math
from json import JSONDecodeError
from typing import Any, cast

from openai import APIError as OpenAIAPIError
from pydantic import ValidationError

from .constants import (
    AgentToolMode,
    DEEPSEEK_CHAT_MAX_COMPLETION_TOKENS,
    DEEPSEEK_CHAT_MODEL,
    DEEPSEEK_REASONER_MAX_COMPLETION_TOKENS,
    DEEPSEEK_REASONER_MODEL,
    DEFAULT_MAX_COMPLETION_TOKENS,
    TOKEN_BUDGET_SAFETY_BUFFER,
    TOKEN_BUDGET_WORD_FACTOR,
    TOKEN_BUDGET_MIN,
    ErrorType,
)
from .models import (
    DeepSeekRequestSettings,
    OutputTruncatedError,
    PodcastScript,
    ToolLoopExhaustedError,
)

def _classify_json_error(error: JSONDecodeError) -> ErrorType:
    """Classify JSON decode failures as malformed vs likely truncated."""
    lowered_message = error.msg.lower()
    if "unterminated" in lowered_message or "expecting value" in lowered_message:
        return "truncated_json"
    if error.pos >= max(0, len(error.doc) - 2):
        return "truncated_json"
    return "malformed_json"

def _normalized_model_name(model: str) -> str:
    """Normalize model name for routing and policy checks."""
    return model.strip().lower()

def _is_reasoner_model(model: str) -> bool:
    """Return True when model is DeepSeek Reasoner."""
    return _normalized_model_name(model) == DEEPSEEK_REASONER_MODEL

def _model_completion_token_ceiling(model: str) -> int:
    """Return model-specific max completion token ceiling."""
    normalized_model = _normalized_model_name(model)
    if normalized_model == DEEPSEEK_REASONER_MODEL:
        return DEEPSEEK_REASONER_MAX_COMPLETION_TOKENS
    if normalized_model == DEEPSEEK_CHAT_MODEL:
        return DEEPSEEK_CHAT_MAX_COMPLETION_TOKENS
    return DEFAULT_MAX_COMPLETION_TOKENS

def _clamp_completion_tokens_for_model(
    configured_max_tokens: int,
    model: str,
) -> int:
    """Clamp completion tokens to the selected model ceiling.

    Args:
        configured_max_tokens: User/configured output-token limit.
        model: Target DeepSeek model.

    Returns:
        Token cap bounded by model-specific limits.
    """
    return max(1, min(configured_max_tokens, _model_completion_token_ceiling(model)))

def _effective_agent_tool_mode(settings: DeepSeekRequestSettings) -> AgentToolMode:
    """Resolve the active tool mode from loop toggle plus mode selection."""
    if not settings.agent_tool_loop:
        return "off"
    return settings.agent_tool_mode

def _should_persist_reasoning_for_replay(
    settings: DeepSeekRequestSettings,
    *,
    tool_mode: AgentToolMode,
) -> bool:
    """Return whether assistant reasoning should be replayed in history.

    DeepSeek reasoning-tool loops require replaying assistant ``reasoning_content``
    within the same question turn. We therefore force-enable replay for
    ``deepseek-reasoner`` when tool mode is active.

    Args:
        settings: Runtime request settings.
        tool_mode: Effective tool mode for the current request.

    Returns:
        True when reasoning content should be included in replayed assistant
        messages for subsequent sub-requests.
    """
    if settings.agent_persist_reasoning_content:
        return True
    return _is_reasoner_model(settings.model) and tool_mode != "off"

def _derive_completion_token_cap(
    word_budget: int | None,
    configured_max_tokens: int,
    *,
    min_tokens: int = TOKEN_BUDGET_MIN,
    max_tokens_ceiling: int = DEFAULT_MAX_COMPLETION_TOKENS,
) -> int:
    """Derive completion token cap from word budget with safety headroom.

    Args:
        word_budget: Target summary word budget, if configured.
        configured_max_tokens: User-configured upper bound.
        min_tokens: Lower bound when dynamic cap is applied.
        max_tokens_ceiling: Absolute ceiling for completion tokens.

    Returns:
        Effective completion token cap.
    """
    bounded_configured = max(1, min(configured_max_tokens, max_tokens_ceiling))
    if word_budget is None:
        return bounded_configured

    estimated_tokens = (
        int(math.ceil(max(1, word_budget) * TOKEN_BUDGET_WORD_FACTOR))
        + TOKEN_BUDGET_SAFETY_BUFFER
    )
    dynamic_cap = max(min_tokens, estimated_tokens)
    return min(dynamic_cap, bounded_configured, max_tokens_ceiling)

def _strip_markdown_fence(content: str) -> str:
    """Strip a top-level Markdown code fence around a JSON blob when present."""
    stripped = content.strip()
    if not stripped.startswith("```"):
        return content

    lines = stripped.splitlines()
    if len(lines) < 3:
        return content

    opening = lines[0].strip().lower()
    closing = lines[-1].strip()
    if not opening.startswith("```") or closing != "```":
        return content
    return "\n".join(lines[1:-1]).strip()

def _extract_first_json_object(content: str) -> str | None:
    """Extract the first balanced JSON object from arbitrary text."""
    start = content.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(content)):
        char = content[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return content[start : index + 1]
    return None

def _decode_podcast_script(raw_content: str) -> tuple[PodcastScript, bool]:
    """Decode and validate LLM output with one balanced repair pass.

    Args:
        raw_content: Raw content returned from DeepSeek.

    Returns:
        Tuple of validated `PodcastScript` and whether local repair was used.

    Raises:
        JSONDecodeError: If JSON parsing fails after recovery attempt.
        ValidationError: If parsed payload fails schema validation.
        ValueError: If no parseable JSON object exists in content.
    """
    direct_payload = json.loads(raw_content)
    try:
        return PodcastScript(**direct_payload), False
    except ValidationError:
        # Schema errors are surfaced to caller as-is.
        raise

def _decode_podcast_script_with_fallback(
    raw_content: str,
) -> tuple[PodcastScript, bool]:
    """Decode and validate LLM output using balanced JSON recovery."""
    try:
        return _decode_podcast_script(raw_content)
    except JSONDecodeError as original_error:
        stripped_content = _strip_markdown_fence(raw_content)
        candidate = _extract_first_json_object(stripped_content)
        if not candidate:
            raise original_error
        repaired_payload = json.loads(candidate)
        return PodcastScript(**repaired_payload), True

def _looks_like_beta_endpoint_error(error: Exception) -> bool:
    """Return True when an exception likely indicates beta endpoint incompatibility."""
    if not isinstance(error, OpenAIAPIError):
        return False

    status_code = getattr(error, "status_code", None)
    lowered = str(error).lower()
    endpoint_keywords = (
        "beta",
        "not found",
        "unsupported",
        "unknown path",
        "invalid url",
    )

    if status_code == 404:
        return True
    if status_code in {400, 405, 422} and any(
        keyword in lowered for keyword in endpoint_keywords
    ):
        return True
    return False

def _looks_like_tool_protocol_error(error: Exception) -> bool:
    """Return True when a failure likely indicates tool-calling incompatibility."""
    if not isinstance(error, OpenAIAPIError):
        return False
    lowered = str(error).lower()
    status_code = getattr(error, "status_code", None)
    protocol_markers = (
        "tool",
        "function",
        "tool_call",
        "tool_choice",
        "reasoning_content",
        "messages with role 'tool'",
    )
    return status_code in {400, 404, 405, 422} and any(
        marker in lowered for marker in protocol_markers
    )

def _normalize_assistant_content(content: object) -> str:
    """Normalize assistant content payloads into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text_piece = block.get("text")
                if isinstance(text_piece, str):
                    text_parts.append(text_piece)
        return "".join(text_parts)
    return ""

def _message_to_replay_dict(message: object) -> dict[str, Any]:
    """Convert assistant message object into API-replay dictionary payload."""
    if hasattr(message, "model_dump"):
        dumped = cast(dict[str, Any], message.model_dump(exclude_none=True))
        dumped.setdefault("role", "assistant")
        return dumped

    message_dict: dict[str, Any] = {"role": "assistant"}
    role = getattr(message, "role", None)
    if isinstance(role, str) and role:
        message_dict["role"] = role
    content = getattr(message, "content", None)
    normalized_content = _normalize_assistant_content(content)
    if normalized_content:
        message_dict["content"] = normalized_content
    reasoning_content = getattr(message, "reasoning_content", None)
    if isinstance(reasoning_content, str) and reasoning_content:
        message_dict["reasoning_content"] = reasoning_content
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        normalized_tool_calls: list[dict[str, Any]] = []
        for tool_call in tool_calls:
            if hasattr(tool_call, "model_dump"):
                normalized_tool_calls.append(
                    cast(dict[str, Any], tool_call.model_dump(exclude_none=True))
                )
                continue
            tool_call_id = getattr(tool_call, "id", None)
            function_obj = getattr(tool_call, "function", None)
            function_name = getattr(function_obj, "name", "")
            function_args = getattr(function_obj, "arguments", "{}")
            normalized_tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": str(function_name),
                        "arguments": str(function_args),
                    },
                }
            )
        message_dict["tool_calls"] = normalized_tool_calls
    return message_dict

def _classify_generation_error(error: Exception) -> ErrorType:
    """Classify generation/parsing exception into retry-guidance categories."""
    if isinstance(error, ToolLoopExhaustedError):
        return "tool_loop_exhausted"
    if isinstance(error, OutputTruncatedError):
        return "truncated_output"
    if isinstance(error, JSONDecodeError):
        return _classify_json_error(error)
    if isinstance(error, ValidationError):
        return "schema_validation"
    if isinstance(error, ValueError) and "empty response" in str(error).lower():
        return "empty_response"
    return "api_error"
