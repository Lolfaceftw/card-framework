"""DeepSeek streaming/runtime helper types and functions."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, cast

from openai import OpenAI

from .chat_logs import DeepSeekChatLogWriter

ERROR_DIGEST_MAX_CHARS = 280
TOOL_EVENT_DETAILS_MAX_CHARS = 1200
CONTEXT_WINDOW_ROLLOVER_LEFT_RATIO = 0.30
DEEPSEEK_STREAM_EVENT_PREFIX = "[DEEPSEEK_STREAM] "
DEEPSEEK_STREAM_OUTPUT_MODE_ENV = "CARD_DEEPSEEK_STREAM_OUTPUT"

StreamOutputMode = Literal["auto", "marker", "plain"]


@dataclass(slots=True)
class _PlainStreamRenderState:
    """Track current plain-stream rendering state across token events."""

    active_phase: str | None = None
    line_open: bool = False


_PLAIN_STREAM_RENDER_STATE = _PlainStreamRenderState()


@dataclass(slots=True, frozen=True)
class ContextUsageSnapshot:
    """Represent normalized context-window usage for UI reporting."""

    tokens_used: int
    tokens_limit: int
    tokens_left: int
    percent_left: float


@dataclass(slots=True, frozen=True)
class StreamedAssistantTurn:
    """Represent one streamed assistant turn with optional tool calls."""

    finish_reason: str | None
    reasoning_content: str
    content: str
    tool_calls: list[dict[str, Any]]
    usage_total_tokens: int | None = None


def _error_digest(error: Exception) -> str:
    """Create a compact, single-line error digest for logs and retry prompts."""
    digest = " ".join(str(error).split())
    return digest[:ERROR_DIGEST_MAX_CHARS]


def _emit_deepseek_stream_event(payload: dict[str, object]) -> None:
    """Emit stream event as marker JSON or plain coalesced token text.

    Marker JSON is retained for non-TTY/piped output (used by pipeline dashboard
    parsing). Plain mode is used for interactive terminals so token fragments are
    rendered as connected streaming text.
    """
    output_mode = _resolve_stream_output_mode()
    if _should_emit_marker(output_mode):
        marker = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        print(f"{DEEPSEEK_STREAM_EVENT_PREFIX}{marker}", flush=True)
        return
    _emit_plain_stream_event(payload)


def _resolve_stream_output_mode() -> StreamOutputMode:
    """Resolve stream output mode from environment and terminal capabilities."""
    raw_mode = os.getenv(DEEPSEEK_STREAM_OUTPUT_MODE_ENV, "auto").strip().lower()
    if raw_mode in {"marker", "plain", "auto"}:
        return cast(StreamOutputMode, raw_mode)
    return "auto"


def _should_emit_marker(mode: StreamOutputMode) -> bool:
    """Return True when stream events should be printed as marker JSON lines."""
    if mode == "marker":
        return True
    if mode == "plain":
        return False
    return not sys.stdout.isatty()


def _plain_write(text: str) -> None:
    """Write raw text to stdout and flush immediately for streaming UX."""
    sys.stdout.write(text)
    sys.stdout.flush()


def _plain_close_line_if_open() -> None:
    """Close any in-progress streaming line before status/event output."""
    if not _PLAIN_STREAM_RENDER_STATE.line_open:
        return
    _plain_write("\n")
    _PLAIN_STREAM_RENDER_STATE.line_open = False
    _PLAIN_STREAM_RENDER_STATE.active_phase = None


def _emit_plain_stream_event(payload: dict[str, object]) -> None:
    """Render stream events as human-readable terminal output."""
    event_name = str(payload.get("event", "")).strip().lower()
    if not event_name:
        return

    if event_name == "start":
        model_name = str(payload.get("model", "deepseek-reasoner")).strip()
        _plain_close_line_if_open()
        _plain_write(f"[DEEPSEEK STATUS] Streaming response from {model_name}\n")
        return

    if event_name == "token":
        text_value = payload.get("text")
        if not isinstance(text_value, str) or not text_value:
            return
        phase = str(payload.get("phase", "answer")).strip().lower()
        if phase in {"reasoning", "answer"}:
            if (
                not _PLAIN_STREAM_RENDER_STATE.line_open
                or _PLAIN_STREAM_RENDER_STATE.active_phase != phase
            ):
                _plain_close_line_if_open()
                phase_label = "REASONING" if phase == "reasoning" else "ANSWER"
                _plain_write(f"[DEEPSEEK {phase_label}] ")
                _PLAIN_STREAM_RENDER_STATE.line_open = True
                _PLAIN_STREAM_RENDER_STATE.active_phase = phase
            _plain_write(text_value)
            return

        _plain_close_line_if_open()
        phase_label = phase.upper() if phase else "TOKEN"
        _plain_write(f"[DEEPSEEK {phase_label}] {text_value}\n")
        return

    if event_name == "summary_json_ready":
        output_path = str(payload.get("path", "")).strip()
        _plain_close_line_if_open()
        if output_path:
            _plain_write(
                "[DEEPSEEK STATUS] Summary JSON ready; finalizing subprocess "
                f"(output={output_path}).\n"
            )
        else:
            _plain_write(
                "[DEEPSEEK STATUS] Summary JSON ready; finalizing subprocess.\n"
            )
        return

    if event_name == "done":
        _plain_close_line_if_open()
        _plain_write("[DEEPSEEK STATUS] Stream finished for current DeepSeek call.\n")
        return

    if event_name == "context_usage":
        return


def _should_persist_stream_event(payload: dict[str, object]) -> bool:
    """Return True when a stream event should be written to persistent trace logs."""
    event_value = payload.get("event")
    if event_value != "token":
        return True
    return payload.get("phase") == "status"


def _emit_stream_token_event(
    stream_event_callback: Callable[[dict[str, object]], None] | None,
    *,
    phase: str,
    text: str,
) -> None:
    """Emit a token-like stream event for dashboard visibility."""
    if stream_event_callback is None:
        return
    if not text.strip():
        return
    stream_event_callback({"event": "token", "phase": phase, "text": text})


def _coerce_non_negative_int(value: object) -> int | None:
    """Coerce object to a non-negative integer when possible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return max(0, int(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = int(stripped)
        except ValueError:
            return None
        return max(0, parsed)
    return None


def _extract_total_tokens_from_usage(usage: object) -> int | None:
    """Extract total token usage from SDK usage payloads."""
    if usage is None:
        return None

    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    if isinstance(usage, dict):
        total_tokens = _coerce_non_negative_int(usage.get("total_tokens"))
        prompt_tokens = _coerce_non_negative_int(usage.get("prompt_tokens"))
        completion_tokens = _coerce_non_negative_int(usage.get("completion_tokens"))
    else:
        total_tokens = _coerce_non_negative_int(getattr(usage, "total_tokens", None))
        prompt_tokens = _coerce_non_negative_int(getattr(usage, "prompt_tokens", None))
        completion_tokens = _coerce_non_negative_int(
            getattr(usage, "completion_tokens", None)
        )

    if total_tokens is not None:
        return total_tokens
    if prompt_tokens is None and completion_tokens is None:
        return None
    return (prompt_tokens or 0) + (completion_tokens or 0)


def _build_context_usage_snapshot(
    *,
    tokens_used: int,
    tokens_limit: int,
) -> ContextUsageSnapshot:
    """Create normalized context usage values for dashboard updates."""
    bounded_limit = max(1, tokens_limit)
    bounded_used = max(0, tokens_used)
    tokens_left = max(0, bounded_limit - bounded_used)
    percent_left = tokens_left / bounded_limit
    return ContextUsageSnapshot(
        tokens_used=bounded_used,
        tokens_limit=bounded_limit,
        tokens_left=tokens_left,
        percent_left=percent_left,
    )


def _emit_context_usage_event(
    stream_event_callback: Callable[[dict[str, object]], None] | None,
    *,
    snapshot: ContextUsageSnapshot,
    rollover_triggered: bool,
    rollover_count: int,
) -> None:
    """Emit context-window usage telemetry for the parent dashboard."""
    if stream_event_callback is None:
        return
    stream_event_callback(
        {
            "event": "context_usage",
            "tokens_used": snapshot.tokens_used,
            "tokens_limit": snapshot.tokens_limit,
            "tokens_left": snapshot.tokens_left,
            "percent_left": snapshot.percent_left,
            "rollover_triggered": rollover_triggered,
            "rollover_count": rollover_count,
        }
    )


def _should_rollover_context(snapshot: ContextUsageSnapshot) -> bool:
    """Return True when context remaining budget is at or below threshold."""
    return snapshot.percent_left <= CONTEXT_WINDOW_ROLLOVER_LEFT_RATIO


def _format_tool_event_details(payload: object) -> str:
    """Format tool args/results as compact text for output-panel status events."""
    if isinstance(payload, str):
        text = " ".join(payload.strip().split())
    else:
        try:
            text = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        except TypeError:
            text = " ".join(str(payload).strip().split())

    if len(text) <= TOOL_EVENT_DETAILS_MAX_CHARS:
        return text
    return f"{text[:TOOL_EVENT_DETAILS_MAX_CHARS]}...(truncated)"


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


def _merge_stream_fragment(existing: str, fragment: str) -> str:
    """Merge streamed text fragments while avoiding duplicate overlap."""
    if not fragment:
        return existing
    if not existing:
        return fragment
    max_overlap = min(len(existing), len(fragment))
    for overlap in range(max_overlap, 0, -1):
        if existing.endswith(fragment[:overlap]):
            return f"{existing}{fragment[overlap:]}"
    return f"{existing}{fragment}"


def _normalize_stream_text_piece(piece: object) -> str:
    """Normalize a streamed delta text field into plain text."""
    if isinstance(piece, str):
        return piece
    return _normalize_assistant_content(piece)


@dataclass(slots=True)
class _StreamCallLogBuffer:
    """Batch streamed text into interval-based call-log events."""

    write_event: Callable[[dict[str, object]], None]
    interval_seconds: float
    _reasoning_parts: list[str] = field(default_factory=list)
    _answer_parts: list[str] = field(default_factory=list)
    _last_flush_monotonic: float = field(default_factory=time.monotonic)

    def append(self, *, phase: Literal["reasoning", "answer"], text: str) -> None:
        """Buffer one streamed text fragment and flush when interval expires."""
        if not text:
            return
        if phase == "reasoning":
            self._reasoning_parts.append(text)
        else:
            self._answer_parts.append(text)
        self.flush_if_due()

    def flush_if_due(self) -> None:
        """Flush buffered text when interval has elapsed."""
        elapsed = time.monotonic() - self._last_flush_monotonic
        if elapsed < self.interval_seconds:
            return
        self.flush(force=False, trigger="interval")

    def flush(self, *, force: bool, trigger: Literal["interval", "stream_end"]) -> None:
        """Flush buffered text into one event per phase."""
        if not force:
            elapsed = time.monotonic() - self._last_flush_monotonic
            if elapsed < self.interval_seconds:
                return
        wrote = False
        if self._reasoning_parts:
            self.write_event(
                {
                    "event": "stream_flush",
                    "phase": "reasoning",
                    "text": "".join(self._reasoning_parts),
                    "trigger": trigger,
                }
            )
            self._reasoning_parts.clear()
            wrote = True
        if self._answer_parts:
            self.write_event(
                {
                    "event": "stream_flush",
                    "phase": "answer",
                    "text": "".join(self._answer_parts),
                    "trigger": trigger,
                }
            )
            self._answer_parts.clear()
            wrote = True
        if wrote:
            self._last_flush_monotonic = time.monotonic()


def _stream_assistant_turn(
    client: OpenAI,
    request_payload: dict[str, Any],
    stream_event_callback: Callable[[dict[str, object]], None] | None,
    chat_log_writer: DeepSeekChatLogWriter | None = None,
    call_log_id: str | None = None,
    on_chunk_callback: Callable[[], None] | None = None,
) -> StreamedAssistantTurn:
    """Stream one assistant turn and aggregate text and tool deltas."""

    def _write_call_event(payload: dict[str, object]) -> None:
        """Write one call-scoped event when call logging is enabled."""
        if chat_log_writer is None or call_log_id is None:
            return
        chat_log_writer.write_call_event(call_log_id, payload)

    finish_reason: str | None = None
    reasoning_parts: list[str] = []
    answer_parts: list[str] = []
    tool_call_accumulator: dict[int, dict[str, Any]] = {}
    usage_total_tokens: int | None = None
    stream_log_buffer: _StreamCallLogBuffer | None = None
    if chat_log_writer is not None and call_log_id is not None:
        stream_log_buffer = _StreamCallLogBuffer(
            write_event=_write_call_event,
            interval_seconds=chat_log_writer.stream_flush_interval_seconds,
        )
    stream_response = client.chat.completions.create(**cast(Any, request_payload))
    try:
        for chunk in stream_response:
            if on_chunk_callback is not None:
                on_chunk_callback()
            chunk_total_tokens = _extract_total_tokens_from_usage(
                getattr(chunk, "usage", None)
            )
            if chunk_total_tokens is not None:
                usage_total_tokens = chunk_total_tokens
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            choice = choices[0]
            finish_reason = getattr(choice, "finish_reason", None) or finish_reason
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue

            reasoning_piece = _normalize_stream_text_piece(
                getattr(delta, "reasoning_content", None)
            )
            if reasoning_piece:
                reasoning_parts.append(reasoning_piece)
                if stream_log_buffer is not None:
                    stream_log_buffer.append(phase="reasoning", text=reasoning_piece)
                _emit_stream_token_event(
                    stream_event_callback,
                    phase="reasoning",
                    text=reasoning_piece,
                )

            answer_piece = _normalize_stream_text_piece(getattr(delta, "content", None))
            if answer_piece:
                answer_parts.append(answer_piece)
                if stream_log_buffer is not None:
                    stream_log_buffer.append(phase="answer", text=answer_piece)
                _emit_stream_token_event(
                    stream_event_callback,
                    phase="answer",
                    text=answer_piece,
                )

            delta_tool_calls = getattr(delta, "tool_calls", None)
            if not delta_tool_calls:
                continue
            for fallback_index, delta_tool_call in enumerate(delta_tool_calls):
                tool_call_index_raw = getattr(delta_tool_call, "index", None)
                tool_call_index = (
                    int(tool_call_index_raw)
                    if isinstance(tool_call_index_raw, int)
                    else fallback_index
                )
                tool_call_entry = tool_call_accumulator.setdefault(
                    tool_call_index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                tool_call_id = getattr(delta_tool_call, "id", None)
                if isinstance(tool_call_id, str) and tool_call_id:
                    tool_call_entry["id"] = tool_call_id

                tool_call_type = getattr(delta_tool_call, "type", None)
                if isinstance(tool_call_type, str) and tool_call_type:
                    tool_call_entry["type"] = tool_call_type

                function_obj = getattr(delta_tool_call, "function", None)
                if function_obj is None:
                    continue

                function_name_piece = getattr(function_obj, "name", None)
                if isinstance(function_name_piece, str) and function_name_piece:
                    existing_function_name = str(tool_call_entry["function"]["name"])
                    tool_call_entry["function"]["name"] = _merge_stream_fragment(
                        existing_function_name,
                        function_name_piece,
                    )

                function_args_piece = getattr(function_obj, "arguments", None)
                if isinstance(function_args_piece, str) and function_args_piece:
                    existing_function_args = str(
                        tool_call_entry["function"]["arguments"]
                    )
                    tool_call_entry["function"]["arguments"] = _merge_stream_fragment(
                        existing_function_args,
                        function_args_piece,
                    )
    finally:
        if stream_log_buffer is not None:
            stream_log_buffer.flush(force=True, trigger="stream_end")

    normalized_tool_calls: list[dict[str, Any]] = []
    for tool_call_index in sorted(tool_call_accumulator):
        tool_call_entry = tool_call_accumulator[tool_call_index]
        function_payload = tool_call_entry.get("function", {})
        function_name = (
            str(function_payload.get("name", ""))
            if isinstance(function_payload, dict)
            else ""
        )
        function_arguments = (
            str(function_payload.get("arguments", ""))
            if isinstance(function_payload, dict)
            else ""
        )
        if not function_name and not function_arguments:
            continue
        normalized_tool_calls.append(
            {
                "id": str(tool_call_entry.get("id", "")).strip(),
                "type": str(tool_call_entry.get("type", "function")),
                "function": {
                    "name": function_name,
                    "arguments": function_arguments or "{}",
                },
            }
        )

    if normalized_tool_calls:
        _write_call_event(
            {
                "event": "assistant_tool_calls",
                "tool_calls": cast(object, normalized_tool_calls),
            }
        )

    return StreamedAssistantTurn(
        finish_reason=finish_reason,
        reasoning_content="".join(reasoning_parts),
        content="".join(answer_parts),
        tool_calls=normalized_tool_calls,
        usage_total_tokens=usage_total_tokens,
    )
