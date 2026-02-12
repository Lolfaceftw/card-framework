"""CARD Script Summarizer with DeepSeek and transcript-grounded speaker validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import re
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field, replace
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Literal, TextIO, TypedDict, cast

from openai import OpenAI
from openai import APIError as OpenAIAPIError
from pydantic import BaseModel, Field, ValidationError

from .logging_utils import configure_logging
from .speaker_validation import (
    DialogueLinePayload,
    TranscriptSegment,
    ValidatedDialogueLine,
    build_segment_speaker_map,
    collect_allowed_speakers,
    format_segments_for_prompt,
    load_transcript_segments,
    validate_and_repair_dialogue,
)

DEEPSEEK_STABLE_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_BETA_BASE_URL = "https://api.deepseek.com/beta"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"
DEEPSEEK_CHAT_MODEL = "deepseek-chat"
DEEPSEEK_MODEL = DEEPSEEK_REASONER_MODEL
MAX_RETRIES_DEFAULT = 2
DEEPSEEK_REASONER_MAX_COMPLETION_TOKENS = 64000
DEEPSEEK_CHAT_MAX_COMPLETION_TOKENS = 8192
DEFAULT_MAX_COMPLETION_TOKENS = DEEPSEEK_REASONER_MAX_COMPLETION_TOKENS
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_HTTP_RETRIES = 1
DEFAULT_TEMPERATURE = 0.2
SUMMARY_LINE_COUNT_MIN = 8
SUMMARY_LINE_COUNT_MAX = 60
SUMMARY_LINE_WORD_TARGET = 14
SUMMARY_LINE_WORD_FLOOR = 7.0
SUMMARY_MAX_SHORT_QUESTION_RATIO = 0.35
ERROR_DIGEST_MAX_CHARS = 280
TOOL_EVENT_DETAILS_MAX_CHARS = 1200
TOKEN_BUDGET_WORD_FACTOR = 1.8
TOKEN_BUDGET_SAFETY_BUFFER = 256
TOKEN_BUDGET_MIN = 512
DEEPSEEK_STREAM_EVENT_PREFIX = "[DEEPSEEK_STREAM] "
DEFAULT_AGENT_MAX_TOOL_ROUNDS = 10
DEFAULT_AGENT_READ_MAX_LINES = 120
DEFAULT_AGENT_WRITE_MAX_CHARS = 12000
DEFAULT_AGENT_WRITE_MAX_FILE_CHARS = 400000
DEFAULT_AGENT_LOOP_EXHAUSTION_POLICY = "auto_salvage"
DEFAULT_AGENT_MAX_REPEATED_WRITE_OVERWRITES = 2
DEFAULT_AGENT_PERSIST_REASONING_CONTENT = False
DEFAULT_AGENT_ALLOW_MODEL_FALLBACK = False
EVALUATE_SCRIPT_TOOL_NAME = "evaluate_script_constraints"
READ_TRANSCRIPT_LINES_TOOL_NAME = "read_transcript_lines"
COUNT_WORDS_TOOL_NAME = "count_words"
WRITE_OUTPUT_SEGMENT_TOOL_NAME = "write_output_segment"
STAGED_OUTPUT_READY_MARKER = "STAGED_OUTPUT_READY"
DEFAULT_BUDGET_FAILURE_POLICY = "degraded_success"
DEFAULT_AGENT_TOOL_MODE = "full_agentic"
CONTEXT_WINDOW_ROLLOVER_LEFT_RATIO = 0.30
CONTEXT_USAGE_UPDATE_INTERVAL_SECONDS = 1.0
CONTEXT_SUMMARY_MAX_TOKENS = 1024
CONTEXT_SUMMARY_MAX_CHARS = 12000
DEEPSEEK_CHAT_LOG_ROOT_DIR = "deepseek_chat_logs"
DEEPSEEK_CHAT_LOG_FILE = "trace.ndjson"
DEEPSEEK_CHAT_LOG_META_FILE = "run_meta.json"
DEEPSEEK_CHAT_LOG_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
DEEPSEEK_CHAT_LOG_STREAM_FLUSH_INTERVAL_SECONDS = 10.0
DISFLUENCY_PATTERN = re.compile(r"\b(?:um+|uh+|ah+)\b", re.IGNORECASE)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ErrorType = Literal[
    "api_error",
    "malformed_json",
    "truncated_json",
    "truncated_output",
    "schema_validation",
    "empty_response",
    "tool_loop_exhausted",
]
EndpointMode = Literal["beta", "stable"]
BudgetFailurePolicy = Literal["degraded_success", "strict_fail"]
AgentToolMode = Literal["off", "constraints_only", "full_agentic"]
AgentLoopExhaustionPolicy = Literal["auto_salvage", "fail_fast"]


@dataclass(slots=True)
class DeepSeekChatLogWriter:
    """Persist per-call and global DeepSeek trace events as line-delimited JSON."""

    run_directory: Path
    log_path: Path
    _log_handle: TextIO
    _open_call_ids: set[str]
    stream_flush_interval_seconds: float = (
        DEEPSEEK_CHAT_LOG_STREAM_FLUSH_INTERVAL_SECONDS
    )
    _next_call_index: int = 0

    @classmethod
    def create(
        cls,
        root_dir: str | Path,
        *,
        stream_flush_interval_seconds: float = (
            DEEPSEEK_CHAT_LOG_STREAM_FLUSH_INTERVAL_SECONDS
        ),
    ) -> DeepSeekChatLogWriter:
        """Create a timestamped DeepSeek chat-log run directory.

        Args:
            root_dir: Base directory where run folders are created.
            stream_flush_interval_seconds: Interval used to batch streamed
                token text into periodic call-log events.

        Returns:
            Initialized writer with open append handles.
        """
        if (
            not math.isfinite(stream_flush_interval_seconds)
            or stream_flush_interval_seconds <= 0
        ):
            raise ValueError("stream_flush_interval_seconds must be positive and finite.")
        resolved_root = Path(root_dir).resolve()
        resolved_root.mkdir(parents=True, exist_ok=True)
        run_directory = _build_unique_chat_log_run_directory(resolved_root)
        run_directory.mkdir(parents=True, exist_ok=False)
        log_path = run_directory / DEEPSEEK_CHAT_LOG_FILE
        log_handle = log_path.open("a", encoding="utf-8")
        writer = cls(
            run_directory=run_directory,
            log_path=log_path,
            _log_handle=log_handle,
            _open_call_ids=set(),
            stream_flush_interval_seconds=stream_flush_interval_seconds,
        )
        writer._write_run_metadata()
        writer.write_global_event(
            {
                "event": "run_start",
                "run_directory": str(run_directory),
            }
        )
        return writer

    def write_global_event(self, payload: dict[str, object]) -> None:
        """Append one event to the single run NDJSON trace log."""
        record = {
            "ts_utc": _utc_now_iso(),
            **payload,
        }
        self._write_json_line(self._log_handle, record)

    def start_call(
        self,
        *,
        call_type: str,
        endpoint_mode: EndpointMode,
        model: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Register one call and write a call-start event.

        Args:
            call_type: Logical call category (tool loop, single completion, rollover).
            endpoint_mode: Active endpoint mode for this request.
            model: Model name used for this request.
            metadata: Optional extra call metadata fields.

        Returns:
            Stable call identifier such as ``call_0001``.
        """
        self._next_call_index += 1
        call_id = f"call_{self._next_call_index:04d}"
        self._open_call_ids.add(call_id)
        start_payload: dict[str, object] = {
            "event": "call_start",
            "call_id": call_id,
            "call_type": call_type,
            "endpoint_mode": endpoint_mode,
            "model": model,
        }
        if metadata:
            start_payload["metadata"] = metadata
        self.write_call_event(call_id, start_payload)
        return call_id

    def write_call_event(self, call_id: str, payload: dict[str, object]) -> None:
        """Append one call-scoped event to the single run NDJSON trace log."""
        record = {
            "ts_utc": _utc_now_iso(),
            "call_id": call_id,
            **payload,
        }
        self._write_json_line(self._log_handle, record)

    def finish_call(
        self,
        call_id: str,
        *,
        status: Literal["ok", "error"],
        finish_reason: str | None = None,
        usage_total_tokens: int | None = None,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Mark one call closed with a final completion event."""
        done_payload: dict[str, object] = {
            "event": "call_done",
            "status": status,
        }
        if finish_reason is not None:
            done_payload["finish_reason"] = finish_reason
        if usage_total_tokens is not None:
            done_payload["usage_total_tokens"] = usage_total_tokens
        if error is not None:
            done_payload["error"] = error
        if metadata:
            done_payload["metadata"] = metadata
        self.write_call_event(call_id, done_payload)
        self._open_call_ids.discard(call_id)

    def close(self) -> None:
        """Flush and close the run log file handle."""
        self.write_global_event({"event": "run_done"})
        for call_id in list(self._open_call_ids):
            self.write_call_event(
                call_id,
                {
                    "event": "call_done",
                    "status": "error",
                    "error": "call closed without explicit finish_call",
                },
            )
            self._open_call_ids.discard(call_id)
        self._log_handle.close()

    def _write_run_metadata(self) -> None:
        """Persist static run metadata once per execution run."""
        metadata_path = self.run_directory / DEEPSEEK_CHAT_LOG_META_FILE
        metadata_payload = {
            "run_started_at_utc": _utc_now_iso(),
            "run_directory": str(self.run_directory),
            "log_path": str(self.log_path),
            "stream_flush_interval_seconds": self.stream_flush_interval_seconds,
        }
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata_payload, metadata_file, indent=2, ensure_ascii=True)
            metadata_file.write("\n")

    @staticmethod
    def _write_json_line(handle: TextIO, payload: dict[str, object]) -> None:
        """Write one JSON line and flush immediately for continuous visibility."""
        handle.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        handle.flush()


def _build_unique_chat_log_run_directory(root_dir: Path) -> Path:
    """Return a unique timestamped chat-log run directory path."""
    timestamp = datetime.now(timezone.utc).strftime(DEEPSEEK_CHAT_LOG_TIMESTAMP_FORMAT)
    candidate = root_dir / timestamp
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        suffixed = root_dir / f"{timestamp}_{suffix}"
        if not suffixed.exists():
            return suffixed
        suffix += 1


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ToolDialogueLine(TypedDict):
    """Represent one dialogue line payload passed to local tool handlers."""

    speaker: str
    text: str
    emo_text: str
    emo_alpha: float
    source_segment_ids: list[str]


class WordBudgetEvaluation(TypedDict):
    """Represent deterministic word-budget evaluation details."""

    enabled: bool
    total_words: int
    target_words: int | None
    lower_bound: int | None
    upper_bound: int | None
    in_range: bool
    source_shorter_than_lower_bound: bool


class ValidationEvaluation(TypedDict):
    """Represent deterministic speaker/schema validation details."""

    is_valid: bool
    issues: list[str]
    repaired_lines: int


class NaturalnessEvaluation(TypedDict):
    """Represent deterministic naturalness checks for dialogue quality."""

    is_natural: bool
    avg_words_per_line: float
    short_question_ratio: float
    disfluency_count: int
    issues: list[str]
    hints: list[str]


class ToolConstraintEvaluation(TypedDict):
    """Represent local tool output consumed by model in the tool loop."""

    status: Literal["pass", "fail"]
    word_budget: WordBudgetEvaluation
    validation: ValidationEvaluation
    naturalness: NaturalnessEvaluation
    hints: list[str]
    repaired_dialogue: list[ToolDialogueLine]
    tool_version: str


class TranscriptToolLine(TypedDict):
    """Represent one transcript line returned by read-transcript tool calls."""

    index: int
    segment_id: str
    speaker: str
    text: str


class ReadTranscriptToolResult(TypedDict):
    """Represent read-transcript tool responses."""

    status: Literal["ok", "fail"]
    requested_start_index: int
    requested_end_index: int
    returned_start_index: int | None
    returned_end_index: int | None
    returned_count: int
    total_segments: int
    lines: list[TranscriptToolLine]
    hints: list[str]


class CountWordsToolResult(TypedDict):
    """Represent deterministic word-count tool responses."""

    status: Literal["ok", "fail"]
    total_words: int
    line_word_counts: list[int]
    hints: list[str]


class WriteOutputSegmentToolResult(TypedDict):
    """Represent segmented output-write tool responses."""

    status: Literal["ok", "fail"]
    mode: Literal["overwrite", "append"] | str
    path: str
    chunk_chars: int
    total_chars: int
    hints: list[str]


@dataclass(slots=True, frozen=True)
class CandidateSelection:
    """Represent a best-effort valid candidate for degraded success mode."""

    lines: list[ValidatedDialogueLine]
    total_words: int | None
    delta_to_target: int | None
    attempt_index: int
    endpoint_mode: EndpointMode
    model: str
    tool_rounds: int


class SummaryReport(TypedDict):
    """Represent sidecar summary diagnostics persisted by the CLI."""

    summary_outcome: Literal["success", "degraded_success", "failure"]
    budget_status: Literal["in_range", "out_of_range", "not_configured", "unknown"]
    validation_status: Literal["valid", "invalid", "unknown"]
    degraded_success: bool
    attempts: int
    max_attempts: int
    endpoint_mode: str
    model: str
    tool_rounds: int
    total_words: int | None
    target_words: int | None
    lower_bound: int | None
    upper_bound: int | None
    tool_calls_by_name: dict[str, int]
    tool_loop_exhausted: bool
    tool_loop_exhaustion_reason: str
    tool_round_limit: int
    repeated_overwrite_count: int
    staged_output_present: bool
    staged_output_valid_json: bool
    last_validation_issues: list[str]
    model_path_selected: str
    naturalness_metrics: dict[str, float | int]
    issues: list[str]
    output_path: str


class ToolLoopDiagnostics(TypedDict):
    """Represent tool-loop termination and salvage diagnostics."""

    tool_loop_exhausted: bool
    tool_loop_exhaustion_reason: str
    tool_round_limit: int
    tool_rounds_used: int
    repeated_overwrite_count: int
    staged_output_present: bool
    staged_output_valid_json: bool
    last_validation_issues: list[str]


class DialogueLine(BaseModel):
    """Represent one model-generated dialogue line."""

    speaker: str = Field(
        ..., description="Speaker label from the allowed transcript speakers."
    )
    text: str = Field(
        ..., description="Natural dialogue text, about 10-15 seconds of speech."
    )
    emo_text: str = Field(..., description="Short emotion/tone description.")
    emo_alpha: float = Field(
        0.6, description="Emotion intensity, typically 0.5 to 0.9."
    )
    source_segment_ids: list[str] = Field(
        ...,
        min_length=1,
        description="Transcript segment IDs used as evidence for this line.",
    )


class PodcastScript(BaseModel):
    """Represent model response payload."""

    dialogue: list[DialogueLine]


class FinalScriptLine(TypedDict):
    """Represent saved output line consumed by the downstream pipeline."""

    speaker: str
    voice_sample: str
    use_emo_text: bool
    emo_text: str
    emo_alpha: float
    text: str
    source_segment_ids: list[str]
    validation_status: str
    repair_reason: str | None


@dataclass(slots=True, frozen=True)
class RetryContinuationState:
    """Represent compact state used to resume corrective retry attempts."""

    read_ranges: list[tuple[int, int]]
    max_read_index: int | None
    write_tool_succeeded: bool
    latest_constraints_status: Literal["pass", "fail", "unknown"]
    last_validation_issues: list[str]
    staged_output_present: bool
    staged_output_valid_json: bool


@dataclass(slots=True, frozen=True)
class RetryContext:
    """Represent context from the previous failed generation attempt."""

    attempt_index: int
    endpoint_mode: EndpointMode
    error_type: ErrorType
    error_digest: str
    continuation: RetryContinuationState | None = None


@dataclass(slots=True, frozen=True)
class DeepSeekRequestSettings:
    """Represent tunable DeepSeek request settings."""

    model: str
    max_completion_tokens: int
    request_timeout_seconds: float
    http_retries: int
    temperature: float | None
    auto_beta: bool = True
    agent_tool_loop: bool = False
    agent_tool_mode: AgentToolMode = DEFAULT_AGENT_TOOL_MODE
    agent_max_tool_rounds: int = DEFAULT_AGENT_MAX_TOOL_ROUNDS
    agent_read_max_lines: int = DEFAULT_AGENT_READ_MAX_LINES
    agent_loop_exhaustion_policy: AgentLoopExhaustionPolicy = (
        DEFAULT_AGENT_LOOP_EXHAUSTION_POLICY
    )
    agent_max_repeated_write_overwrites: int = (
        DEFAULT_AGENT_MAX_REPEATED_WRITE_OVERWRITES
    )
    agent_persist_reasoning_content: bool = DEFAULT_AGENT_PERSIST_REASONING_CONTENT
    agent_allow_model_fallback: bool = DEFAULT_AGENT_ALLOW_MODEL_FALLBACK


@dataclass(slots=True)
class ConversationState:
    """Persist DeepSeek multi-round chat context across repeated calls."""

    messages: list[dict[str, Any]]
    rollover_count: int = 0
    context_tokens_used: int | None = None
    context_tokens_limit: int | None = None
    last_tool_loop_diagnostics: ToolLoopDiagnostics | None = None


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


@dataclass(slots=True, frozen=True)
class GenerationSuccess:
    """Represent a successful structured generation result."""

    script: PodcastScript
    endpoint_mode: EndpointMode
    used_json_repair: bool
    model: str
    tool_rounds: int = 0
    tool_call_counts: dict[str, int] | None = None
    tool_loop_diagnostics: ToolLoopDiagnostics | None = None


@dataclass(slots=True, frozen=True)
class GenerationAttemptError(Exception):
    """Wrap a generation failure with endpoint attribution."""

    endpoint_mode: EndpointMode
    cause: Exception


class OutputTruncatedError(ValueError):
    """Represent model output truncation due to output token limits."""


class ToolLoopExhaustedError(RuntimeError):
    """Represent failure after hitting the configured tool-loop round limit."""

    def __init__(
        self,
        message: str,
        diagnostics: ToolLoopDiagnostics,
        *,
        continuation_state: RetryContinuationState | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics: ToolLoopDiagnostics
        self.diagnostics = diagnostics
        self.continuation_state: RetryContinuationState | None
        self.continuation_state = continuation_state


def _error_digest(error: Exception) -> str:
    """Create a compact, single-line error digest for logs and retry prompts."""
    digest = " ".join(str(error).split())
    return digest[:ERROR_DIGEST_MAX_CHARS]


def _emit_deepseek_stream_event(payload: dict[str, object]) -> None:
    """Emit a structured stream event for the parent pipeline dashboard."""
    marker = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    print(f"{DEEPSEEK_STREAM_EVENT_PREFIX}{marker}", flush=True)


def _should_persist_stream_event(payload: dict[str, object]) -> bool:
    """Return True when a stream event should be written to persistent trace logs."""
    event_value = payload.get("event")
    if event_value != "token":
        return True
    # Keep high-signal status updates but skip high-volume token streams.
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


def _build_transcript_manifest(transcript_segments: list[TranscriptSegment]) -> str:
    """Build a compact transcript manifest for tool-driven reading workflows."""
    if not transcript_segments:
        return "Transcript manifest unavailable (no segments)."

    first_segment = transcript_segments[0].segment_id
    last_segment = transcript_segments[-1].segment_id
    speaker_counts: dict[str, int] = {}
    for segment in transcript_segments:
        speaker_counts[segment.speaker] = speaker_counts.get(segment.speaker, 0) + 1
    speaker_summary = ", ".join(
        f"{speaker}={count}" for speaker, count in sorted(speaker_counts.items())
    )
    return (
        f"segment_count={len(transcript_segments)} "
        f"segment_id_range=[{first_segment}..{last_segment}] "
        f"speaker_distribution={speaker_summary}"
    )


def _build_initial_user_message(
    *,
    tool_mode: AgentToolMode,
    transcript_text: str,
    transcript_manifest: str,
) -> str:
    """Build the initial user turn used to seed a fresh conversation."""
    if tool_mode == "full_agentic":
        return (
            "Use tools to iteratively build a compliant summary.\n"
            f"Transcript manifest: {transcript_manifest}\n"
            f"Read transcript slices with {READ_TRANSCRIPT_LINES_TOOL_NAME}.\n"
            f"Stage JSON chunks with {WRITE_OUTPUT_SEGMENT_TOOL_NAME}.\n"
            "When staged JSON is complete, you may answer with STAGED_OUTPUT_READY."
        )
    return f"Here is the raw transcript:\n\n{transcript_text}"


def _build_rollover_continuation_message(tool_mode: AgentToolMode) -> str:
    """Build continuation instruction after context rollover compression."""
    if tool_mode == "full_agentic":
        return (
            "Continue the same task using the summarized prior conversation. "
            "Keep using tools until constraints pass and final JSON is valid."
        )
    return (
        "Continue the same summarization task using the summarized prior context. "
        "Return strict JSON matching the required schema."
    )


def _build_system_prompt(
    allowed_speakers: set[str],
    target_min_lines: int,
    target_max_lines: int,
    word_budget: int | None,
    target_minutes: float | None,
    avg_wpm: float | None,
    word_budget_tolerance: float,
    require_tool_call: bool,
    tool_mode: AgentToolMode,
    max_repeated_write_overwrites: int,
) -> str:
    """Build a constrained prompt with explicit objective and acceptance checks.

    Args:
        allowed_speakers: Speaker labels allowed in the response.
        target_min_lines: Lower bound of desired output dialogue line count.
        target_max_lines: Upper bound of desired output dialogue line count.
        word_budget: Optional summary word budget target.
        target_minutes: Optional target duration in minutes.
        avg_wpm: Optional calibrated words-per-minute value.
        word_budget_tolerance: Allowed relative budget tolerance.
        require_tool_call: Whether finalization must be gated by tool pass.
        tool_mode: Active tool-mode policy.

    Returns:
        Prompt string for the LLM.
    """
    speaker_list = ", ".join(sorted(allowed_speakers))
    sample_speaker = sorted(allowed_speakers)[0]
    tool_mode_note = (
        "Tool mode: full_agentic (use read/count/write/evaluate tools)."
        if tool_mode == "full_agentic"
        else "Tool mode: constraints_only (use count/write/evaluate tools)."
    )
    budget_clause = ""
    if word_budget is not None:
        tolerance_pct = int(round(word_budget_tolerance * 100))
        minutes_note = (
            f" This corresponds to about {target_minutes:.2f} minutes at {avg_wpm:.2f} WPM."
            if target_minutes is not None and avg_wpm is not None
            else ""
        )
        budget_clause = (
            f"- Target total word count: {word_budget} (+/-{tolerance_pct}%)."
            f"{minutes_note}\n"
            "- Word count applies only to dialogue[].text words (not JSON keys, "
            "field names, IDs, or metadata).\n"
        )
    tool_loop_clause = ""
    if require_tool_call:
        if tool_mode == "full_agentic":
            tool_loop_clause = (
                f"- {tool_mode_note}\n"
                f"- Read transcript slices with `{READ_TRANSCRIPT_LINES_TOOL_NAME}` before drafting.\n"
                f"- Use `{COUNT_WORDS_TOOL_NAME}` with batched `lines` payloads for counting; do not self-count.\n"
                f"- Stage output JSON with `{WRITE_OUTPUT_SEGMENT_TOOL_NAME}` in chunks (overwrite first, append next).\n"
                f"- Do not repeat identical overwrite writes more than {max_repeated_write_overwrites} times in a row; revise then re-evaluate.\n"
                f"- You must call `{EVALUATE_SCRIPT_TOOL_NAME}` before finalizing JSON.\n"
                f"- If `{EVALUATE_SCRIPT_TOOL_NAME}` returns status=fail, revise and call it again.\n"
                f"- Return final response as strict JSON or `STAGED_OUTPUT_READY` after staged output is complete.\n"
                f"- Return final JSON only after the latest `{EVALUATE_SCRIPT_TOOL_NAME}` is status=pass.\n"
            )
        else:
            tool_loop_clause = (
                f"- {tool_mode_note}\n"
                f"- Use `{COUNT_WORDS_TOOL_NAME}` with batched `lines` payloads for word-count checks; do not self-count.\n"
                f"- Stage output JSON with `{WRITE_OUTPUT_SEGMENT_TOOL_NAME}` in chunks (overwrite first, append next).\n"
                f"- You must call `{EVALUATE_SCRIPT_TOOL_NAME}` before finalizing JSON.\n"
                f"- If `{EVALUATE_SCRIPT_TOOL_NAME}` returns status=fail, revise and call it again.\n"
                f"- Return final response as strict JSON or `STAGED_OUTPUT_READY` after staged output is complete.\n"
                f"- Return final JSON only after the latest `{EVALUATE_SCRIPT_TOOL_NAME}` is status=pass.\n"
            )

    input_clause = (
        "- Use transcript tools to read lines by index range (segment_id, speaker, text).\n"
        "- A compact manifest is provided in user content; do not assume hidden transcript text.\n"
        if require_tool_call and tool_mode == "full_agentic"
        else "- Transcript lines are provided in [segment_id|speaker]: text format."
    )
    output_clause = (
        (
            "- Output STRICT JSON ONLY, or reply "
            f"`{STAGED_OUTPUT_READY_MARKER}` after staged output is complete.\n"
        )
        if require_tool_call
        else "- Output STRICT JSON ONLY: no markdown fences, no prose before/after JSON.\n"
    )
    completion_checklist_tail = (
        (
            f"  6) If replying `{STAGED_OUTPUT_READY_MARKER}`, staged file contains a full valid JSON object.\n"
            "  7) Spoken style feels conversational, not bulletized.\n"
            f"  8) Latest `{EVALUATE_SCRIPT_TOOL_NAME}` result is status=pass.\n"
            f"  9) `{WRITE_OUTPUT_SEGMENT_TOOL_NAME}` wrote staged output successfully.\n"
        )
        if require_tool_call
        else (
            "  6) Output is only JSON and is complete (not truncated).\n"
            "  7) Spoken style feels conversational, not bulletized.\n"
            "  8) If tools are enabled, the latest "
            f"`{EVALUATE_SCRIPT_TOOL_NAME}` result is status=pass.\n"
        )
    )

    return f"""Objective:
- Rewrite the transcript into concise, engaging podcast dialogue while preserving factual meaning and conversational flow.

Context:
- This is for CARD (Constraint-aware Audio Resynthesis).
- Transcript lines include evidence metadata in the form [segment_id|speaker]: text.
- Allowed speakers are exactly: {speaker_list}

Inputs:
{input_clause}
- Transcript line schema:
  - segment_id: unique identifier (example: seg_00012)
  - speaker: one allowed speaker label
  - text: source utterance

Output contract:
- Return a JSON object with shape:
  {{
    "dialogue": [
      {{
        "speaker": "ONE_ALLOWED_SPEAKER",
        "text": "spoken dialogue line",
        "emo_text": "brief emotional style",
        "emo_alpha": 0.6,
        "source_segment_ids": ["seg_00012", "seg_00013"]
      }}
    ]
  }}
{output_clause.rstrip()}
- The response must end with a fully closed JSON object.
- Each line must include non-empty source_segment_ids.
- Target {target_min_lines} to {target_max_lines} dialogue lines total.
- Keep each line concise (prefer <= 220 characters) and about 10-15 seconds of speech.
{budget_clause}
{tool_loop_clause}

Rules:
- Never invent, rename, or merge speaker labels.
- speaker must be exactly one of the allowed speakers.
- Every source_segment_id must come from the input transcript.
- Preserve segment ID formatting exactly, including zero padding (example: seg_00004).
- All source_segment_ids for a single line must belong to the same speaker.
- Avoid meta-summary phrases like "the speaker discusses".
- Preserve key claims and tone.
- Keep spoken cadence natural; occasional disfluencies like "um" or "uh" are allowed when they fit.
- Avoid robotic stubs, one-word question chains, or telegraphic phrasing.
- Never perform more than {max_repeated_write_overwrites} identical overwrite writes in a row; revise, evaluate, then write.

Examples:
- Input excerpt:
  [seg_00000|{sample_speaker}]: Welcome back to the show.
  [seg_00001|{sample_speaker}]: Today we are breaking down the market shift.
- Valid output excerpt:
  {{
    "dialogue": [
      {{
        "speaker": "{sample_speaker}",
        "text": "Welcome back. Today we're breaking down the market shift.",
        "emo_text": "Warm and focused",
        "emo_alpha": 0.65,
        "source_segment_ids": ["seg_00000", "seg_00001"]
      }}
    ]
  }}

Evaluation:
- Before answering, verify:
  1) JSON is valid and matches schema exactly.
  2) Each line has non-empty source_segment_ids.
  3) Each line uses only allowed speakers.
  4) Each line's source_segment_ids belong to one speaker only.
  5) Segment IDs match exact transcript formatting, including leading zeros.
{completion_checklist_tail.rstrip()}
"""


def _build_retry_instruction(context: RetryContext) -> str:
    """Build retry guidance that explains the previous failure."""
    message = (
        "Retry correction request:\n"
        f"- Previous attempt: {context.attempt_index}\n"
        f"- Previous endpoint: {context.endpoint_mode}\n"
        f"- Failure type: {context.error_type}\n"
        f"- Failure detail: {context.error_digest}\n"
        "- Fix the exact issue and regenerate a complete STRICT JSON object only.\n"
        "- Do not include markdown, explanations, or trailing text.\n"
    )
    continuation = context.continuation
    if continuation is None:
        return message
    read_ranges_text = _format_retry_read_ranges(continuation.read_ranges)
    max_read_index_text = (
        str(continuation.max_read_index)
        if continuation.max_read_index is not None
        else "unknown"
    )
    return (
        f"{message}"
        "- Resume from prior progress instead of restarting from scratch.\n"
        f"- Prior transcript coverage: ranges={read_ranges_text} "
        f"max_read_index={max_read_index_text}.\n"
        "- Reuse staged output or partial draft if available; revise incrementally.\n"
        "- Do not reread transcript lines from index 0 unless unresolved validation "
        "issues require it.\n"
        "- Stage JSON with write_output_segment before extensive redrafting.\n"
    )


def _format_retry_read_ranges(read_ranges: list[tuple[int, int]]) -> str:
    """Format compact read-range telemetry for retry guidance text."""
    if not read_ranges:
        return "none"
    preview = read_ranges[:4]
    rendered_preview = ", ".join(f"[{start},{end}]" for start, end in preview)
    if len(read_ranges) <= len(preview):
        return rendered_preview
    return f"{rendered_preview}, ... (+{len(read_ranges) - len(preview)} more)"


def _build_retry_resume_guard_message(continuation: RetryContinuationState) -> str:
    """Build an explicit guardrail to avoid unnecessary transcript rereads."""
    read_ranges_text = _format_retry_read_ranges(continuation.read_ranges)
    max_read_index_text = (
        str(continuation.max_read_index)
        if continuation.max_read_index is not None
        else "unknown"
    )
    return (
        "Retry resume guard:\n"
        f"- Prior read coverage: {read_ranges_text} (max_read_index={max_read_index_text}).\n"
        "- Continue from uncovered transcript ranges first.\n"
        "- Do not reread transcript lines from index 0 unless there is a specific "
        "unresolved validation issue that requires it.\n"
        "- Call write_output_segment early to preserve iterative progress.\n"
    )


def _build_word_budget_retry_digest(
    total_words: int,
    target_words: int,
    lower_bound: int,
    upper_bound: int,
) -> str:
    """Build retry digest for word-budget validation failures."""
    delta = total_words - target_words
    sign = "+" if delta >= 0 else ""
    return (
        "word_budget_out_of_range "
        f"total={total_words} target={target_words} "
        f"range=[{lower_bound},{upper_bound}] delta={sign}{delta}"
    )


def _parse_on_off_flag(raw_value: str) -> bool:
    """Parse CLI on/off values into booleans."""
    return raw_value.strip().lower() == "on"


def _coerce_tool_dialogue_lines(raw_lines: object) -> list[DialogueLinePayload]:
    """Coerce tool-call arguments into dialogue payload objects."""
    if not isinstance(raw_lines, list):
        return []
    payload_lines: list[DialogueLinePayload] = []
    for raw_line in raw_lines:
        if not isinstance(raw_line, dict):
            continue
        source_raw = raw_line.get("source_segment_ids", [])
        source_segment_ids = (
            [
                str(source_id).strip()
                for source_id in source_raw
                if str(source_id).strip()
            ]
            if isinstance(source_raw, list)
            else []
        )
        payload_lines.append(
            {
                "speaker": str(raw_line.get("speaker", "")).strip(),
                "text": str(raw_line.get("text", "")),
                "emo_text": str(raw_line.get("emo_text", "")),
                "emo_alpha": float(raw_line.get("emo_alpha", 0.6)),
                "source_segment_ids": source_segment_ids,
            }
        )
    return payload_lines


def _count_words_from_payload_lines(lines: list[DialogueLinePayload]) -> int:
    """Count dialogue words directly from pre-validation payload lines."""
    return sum(len(str(line.get("text", "")).split()) for line in lines)


def _count_words_in_text(text: str) -> int:
    """Count words from plain text content."""
    return len([part for part in text.split() if part])


def _evaluate_naturalness(
    dialogue_payload: list[DialogueLinePayload],
) -> NaturalnessEvaluation:
    """Evaluate lightweight conversational naturalness heuristics."""
    texts = [str(line.get("text", "")) for line in dialogue_payload]
    line_count = len(texts)
    if line_count == 0:
        return {
            "is_natural": False,
            "avg_words_per_line": 0.0,
            "short_question_ratio": 0.0,
            "disfluency_count": 0,
            "issues": ["Dialogue is empty."],
            "hints": ["Generate dialogue lines before finalizing output."],
        }

    word_counts = [_count_words_in_text(text) for text in texts]
    avg_words_per_line = float(sum(word_counts)) / float(line_count)
    short_question_lines = sum(
        1
        for text, words in zip(texts, word_counts)
        if text.strip().endswith("?") and words <= 5
    )
    short_question_ratio = float(short_question_lines) / float(line_count)
    disfluency_count = sum(len(DISFLUENCY_PATTERN.findall(text)) for text in texts)

    issues: list[str] = []
    hints: list[str] = []
    if avg_words_per_line < SUMMARY_LINE_WORD_FLOOR:
        issues.append(
            "Average words per line is too low for natural cadence."
        )
        hints.append(
            f"Increase line richness: average words/line should be >= {SUMMARY_LINE_WORD_FLOOR:.1f}."
        )
    if short_question_ratio > SUMMARY_MAX_SHORT_QUESTION_RATIO:
        issues.append("Too many short question-style lines.")
        hints.append(
            f"Reduce short question lines to <= {SUMMARY_MAX_SHORT_QUESTION_RATIO:.2f} of total lines."
        )
    if disfluency_count == 0:
        hints.append(
            'Optional: add sparse spoken fillers like "um" or "uh" where tone supports it.'
        )
    if not issues and not hints:
        hints.append("Naturalness checks satisfied.")

    return {
        "is_natural": len(issues) == 0,
        "avg_words_per_line": avg_words_per_line,
        "short_question_ratio": short_question_ratio,
        "disfluency_count": disfluency_count,
        "issues": issues,
        "hints": hints,
    }


def _build_constraint_hints(
    *,
    validation_issues: list[str],
    budget_pass: bool,
    total_words: int,
    target_words: int | None,
    lower_bound: int | None,
    upper_bound: int | None,
    naturalness: NaturalnessEvaluation,
) -> list[str]:
    """Build deterministic hint messages for tool-loop corrective steps."""
    hints: list[str] = []
    for issue in validation_issues[:3]:
        if "span multiple speakers" in issue:
            hints.append(
                "Do not mix source_segment_ids from different speakers in one line."
            )
        elif "unknown source_segment_ids" in issue:
            hints.append(
                "Use only source_segment_ids that exist in the provided transcript."
            )
        elif "source_segment_ids is empty" in issue:
            hints.append("Each dialogue line must include at least one source_segment_id.")
        else:
            hints.append(f"Fix validation issue: {issue}")

    if not budget_pass and target_words is not None:
        if lower_bound is not None and total_words < lower_bound:
            hints.append(
                f"Increase total dialogue words from {total_words} to at least {lower_bound}."
            )
        if upper_bound is not None and total_words > upper_bound:
            hints.append(
                f"Reduce total dialogue words from {total_words} to at most {upper_bound}."
            )
        hints.append(
            "Preserve factual claims while tightening phrasing and removing redundant text."
        )
    for issue in naturalness["issues"][:2]:
        if "Average words per line" in issue:
            hints.append(
                "Use fuller spoken sentences; avoid clipped fragments."
            )
        elif "short question-style" in issue:
            hints.append(
                "Reduce rapid-fire short questions and combine them into natural turns."
            )
        else:
            hints.append(f"Fix naturalness issue: {issue}")
    for soft_hint in naturalness["hints"]:
        if soft_hint not in hints:
            hints.append(soft_hint)

    if not hints:
        hints.append("Constraints satisfied. Return strict JSON only.")
    return hints


def _evaluate_script_constraints_tool(
    *,
    dialogue_payload: list[DialogueLinePayload],
    allowed_speakers: set[str],
    segment_speaker_map: dict[str, str],
    word_budget: int | None,
    word_budget_tolerance: float,
    source_word_count: int,
) -> ToolConstraintEvaluation:
    """Evaluate schema/speaker and budget constraints for tool-calling loops."""
    validation_report = validate_and_repair_dialogue(
        dialogue_payload,
        allowed_speakers=allowed_speakers,
        segment_speaker_map=segment_speaker_map,
    )
    total_words = _count_words_from_payload_lines(dialogue_payload)
    naturalness = _evaluate_naturalness(dialogue_payload)

    lower_bound: int | None = None
    upper_bound: int | None = None
    budget_pass = True
    source_shorter_than_lower_bound = False
    if word_budget is not None:
        lower_bound, upper_bound = _budget_bounds(word_budget, word_budget_tolerance)
        if lower_bound is not None and upper_bound is not None:
            budget_pass = lower_bound <= total_words <= upper_bound
            if (
                not budget_pass
                and total_words < lower_bound
                and source_word_count > 0
                and source_word_count < lower_bound
            ):
                budget_pass = True
                source_shorter_than_lower_bound = True

    status: Literal["pass", "fail"] = (
        "pass"
        if validation_report.is_valid and budget_pass and naturalness["is_natural"]
        else "fail"
    )
    hints = _build_constraint_hints(
        validation_issues=validation_report.issues,
        budget_pass=budget_pass,
        total_words=total_words,
        target_words=word_budget,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        naturalness=naturalness,
    )
    repaired_dialogue: list[ToolDialogueLine] = [
        {
            "speaker": line.speaker,
            "text": line.text,
            "emo_text": line.emo_text,
            "emo_alpha": line.emo_alpha,
            "source_segment_ids": line.source_segment_ids,
        }
        for line in validation_report.lines
    ]

    return {
        "status": status,
        "word_budget": {
            "enabled": word_budget is not None,
            "total_words": total_words,
            "target_words": word_budget,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "in_range": budget_pass,
            "source_shorter_than_lower_bound": source_shorter_than_lower_bound,
        },
        "validation": {
            "is_valid": validation_report.is_valid,
            "issues": validation_report.issues,
            "repaired_lines": validation_report.repaired_lines,
        },
        "naturalness": naturalness,
        "hints": hints,
        "repaired_dialogue": repaired_dialogue,
        "tool_version": "1.0",
    }


def _read_transcript_lines_tool(
    *,
    transcript_segments: list[TranscriptSegment],
    start_index: int,
    end_index: int,
    max_lines: int,
) -> ReadTranscriptToolResult:
    """Read transcript lines by inclusive segment-index range."""
    total_segments = len(transcript_segments)
    if total_segments == 0:
        return {
            "status": "fail",
            "requested_start_index": start_index,
            "requested_end_index": end_index,
            "returned_start_index": None,
            "returned_end_index": None,
            "returned_count": 0,
            "total_segments": 0,
            "lines": [],
            "hints": ["Transcript is empty."],
        }

    requested_start = int(start_index)
    requested_end = int(end_index)
    bounded_start = max(0, min(requested_start, total_segments - 1))
    bounded_end = max(0, min(requested_end, total_segments - 1))
    if bounded_end < bounded_start:
        bounded_start, bounded_end = bounded_end, bounded_start

    hints: list[str] = []
    max_allowed = max(1, int(max_lines))
    available_count = bounded_end - bounded_start + 1
    if available_count > max_allowed:
        bounded_end = bounded_start + max_allowed - 1
        hints.append(
            f"Range truncated to {max_allowed} lines due to configured max read window."
        )
    if requested_start != bounded_start or requested_end != bounded_end:
        hints.append("Requested range was clamped to available transcript bounds.")
    if not hints:
        hints.append("Read completed.")

    lines: list[TranscriptToolLine] = []
    for index in range(bounded_start, bounded_end + 1):
        segment = transcript_segments[index]
        lines.append(
            {
                "index": index,
                "segment_id": segment.segment_id,
                "speaker": segment.speaker,
                "text": segment.text,
            }
        )
    return {
        "status": "ok",
        "requested_start_index": requested_start,
        "requested_end_index": requested_end,
        "returned_start_index": bounded_start,
        "returned_end_index": bounded_end,
        "returned_count": len(lines),
        "total_segments": total_segments,
        "lines": lines,
        "hints": hints,
    }


def _count_words_tool(
    *,
    text: str | None,
    lines: list[str] | None,
) -> CountWordsToolResult:
    """Count words for candidate text or line arrays."""
    if text is not None:
        normalized_lines = [str(text)]
    elif lines is not None:
        normalized_lines = [str(line) for line in lines]
    else:
        return {
            "status": "fail",
            "total_words": 0,
            "line_word_counts": [],
            "hints": ["Provide either `text` or `lines` for counting."],
        }

    line_word_counts = [_count_words_in_text(line) for line in normalized_lines]
    total_words = sum(line_word_counts)
    return {
        "status": "ok",
        "total_words": total_words,
        "line_word_counts": line_word_counts,
        "hints": ["Word count computed from whitespace-delimited tokens."],
    }


def _resolve_tool_output_staging_path(output_path: str | None) -> str:
    """Resolve a deterministic staging path for segmented tool writes."""
    if output_path is not None and output_path.strip():
        normalized_output = Path(output_path.strip())
        return str(normalized_output.with_name(f"{normalized_output.name}.agent_buffer"))

    fallback_name = (
        f".deepseek_agent_output_{os.getpid()}_{int(time.time() * 1000)}.json"
    )
    return str(Path.cwd() / fallback_name)


def _write_output_segment_tool(
    *,
    staging_path: str,
    mode: str,
    content: str,
    max_chunk_chars: int,
    max_file_chars: int,
) -> WriteOutputSegmentToolResult:
    """Write output JSON content in chunks to a local staging file."""
    normalized_mode = mode.strip().lower()
    if normalized_mode not in {"overwrite", "append"}:
        return {
            "status": "fail",
            "mode": normalized_mode or mode,
            "path": staging_path,
            "chunk_chars": len(content),
            "total_chars": 0,
            "hints": ["mode must be either 'overwrite' or 'append'."],
        }

    bounded_chunk_limit = max(1, int(max_chunk_chars))
    bounded_file_limit = max(bounded_chunk_limit, int(max_file_chars))
    chunk_chars = len(content)
    if chunk_chars > bounded_chunk_limit:
        return {
            "status": "fail",
            "mode": normalized_mode,
            "path": staging_path,
            "chunk_chars": chunk_chars,
            "total_chars": 0,
            "hints": [
                (
                    f"Chunk exceeds max size ({chunk_chars}>{bounded_chunk_limit}). "
                    "Split content into smaller segments."
                )
            ],
        }

    staging_file = Path(staging_path)
    existing_text = ""
    if normalized_mode == "append" and staging_file.exists():
        try:
            existing_text = staging_file.read_text(encoding="utf-8")
        except OSError as error:
            return {
                "status": "fail",
                "mode": normalized_mode,
                "path": staging_path,
                "chunk_chars": chunk_chars,
                "total_chars": 0,
                "hints": [f"Failed to read existing staging file: {error}"],
            }
    projected_total = (
        chunk_chars if normalized_mode == "overwrite" else len(existing_text) + chunk_chars
    )
    if projected_total > bounded_file_limit:
        return {
            "status": "fail",
            "mode": normalized_mode,
            "path": staging_path,
            "chunk_chars": chunk_chars,
            "total_chars": len(existing_text),
            "hints": [
                (
                    f"Projected file size exceeds limit ({projected_total}>{bounded_file_limit}). "
                    "Finalize or reduce content."
                )
            ],
        }

    try:
        staging_file.parent.mkdir(parents=True, exist_ok=True)
        if normalized_mode == "overwrite":
            staging_file.write_text(content, encoding="utf-8")
            total_chars = chunk_chars
        else:
            with open(staging_file, "a", encoding="utf-8") as output_handle:
                output_handle.write(content)
            total_chars = projected_total
    except OSError as error:
        return {
            "status": "fail",
            "mode": normalized_mode,
            "path": staging_path,
            "chunk_chars": chunk_chars,
            "total_chars": 0,
            "hints": [f"Failed to write staging file: {error}"],
        }

    return {
        "status": "ok",
        "mode": cast(Literal["overwrite", "append"], normalized_mode),
        "path": staging_path,
        "chunk_chars": chunk_chars,
        "total_chars": total_chars,
        "hints": ["Segment written to staging file."],
    }


def _read_staged_output_text(staging_path: str) -> str:
    """Read staged output text written through segmented tool calls."""
    try:
        return Path(staging_path).read_text(encoding="utf-8")
    except OSError:
        return ""


def _copy_tool_loop_diagnostics(
    diagnostics: ToolLoopDiagnostics,
) -> ToolLoopDiagnostics:
    """Return a defensive copy of tool-loop diagnostics."""
    return {
        "tool_loop_exhausted": diagnostics["tool_loop_exhausted"],
        "tool_loop_exhaustion_reason": diagnostics["tool_loop_exhaustion_reason"],
        "tool_round_limit": diagnostics["tool_round_limit"],
        "tool_rounds_used": diagnostics["tool_rounds_used"],
        "repeated_overwrite_count": diagnostics["repeated_overwrite_count"],
        "staged_output_present": diagnostics["staged_output_present"],
        "staged_output_valid_json": diagnostics["staged_output_valid_json"],
        "last_validation_issues": list(diagnostics["last_validation_issues"]),
    }


def _merge_tool_loop_diagnostics(
    target: ToolLoopDiagnostics,
    source: ToolLoopDiagnostics,
) -> None:
    """Copy diagnostic values from source into target in place."""
    target["tool_loop_exhausted"] = source["tool_loop_exhausted"]
    target["tool_loop_exhaustion_reason"] = source["tool_loop_exhaustion_reason"]
    target["tool_round_limit"] = source["tool_round_limit"]
    target["tool_rounds_used"] = source["tool_rounds_used"]
    target["repeated_overwrite_count"] = source["repeated_overwrite_count"]
    target["staged_output_present"] = source["staged_output_present"]
    target["staged_output_valid_json"] = source["staged_output_valid_json"]
    target["last_validation_issues"] = source["last_validation_issues"][:]


def _new_tool_loop_diagnostics(tool_round_limit: int) -> ToolLoopDiagnostics:
    """Create empty diagnostics for one tool-loop run."""
    return {
        "tool_loop_exhausted": False,
        "tool_loop_exhaustion_reason": "",
        "tool_round_limit": max(1, int(tool_round_limit)),
        "tool_rounds_used": 0,
        "repeated_overwrite_count": 0,
        "staged_output_present": False,
        "staged_output_valid_json": False,
        "last_validation_issues": [],
    }


def _hash_text(text: str) -> str:
    """Return a deterministic digest for short overwrite-repeat detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tool_dialogue_payload_from_script(script: PodcastScript) -> list[DialogueLinePayload]:
    """Convert script model payload into tool-evaluation dialogue payload lines."""
    return [cast(DialogueLinePayload, line.model_dump()) for line in script.dialogue]


def _podcast_script_from_tool_dialogue(
    repaired_dialogue: list[ToolDialogueLine],
) -> PodcastScript:
    """Build a strict script model from tool-repaired dialogue payload."""
    dialogue_lines = [
        DialogueLine(
            speaker=line["speaker"],
            text=line["text"],
            emo_text=line["emo_text"],
            emo_alpha=float(line["emo_alpha"]),
            source_segment_ids=list(line["source_segment_ids"]),
        )
        for line in repaired_dialogue
    ]
    return PodcastScript(dialogue=dialogue_lines)


def _build_tool_loop_exhaustion_error(
    *,
    diagnostics: ToolLoopDiagnostics,
    reason: str,
    issues: list[str],
    continuation_state: RetryContinuationState | None = None,
) -> ToolLoopExhaustedError:
    """Build a typed exhaustion error with normalized diagnostics payload."""
    updated = _copy_tool_loop_diagnostics(diagnostics)
    updated["tool_loop_exhausted"] = True
    updated["tool_loop_exhaustion_reason"] = reason
    updated["last_validation_issues"] = issues[:]
    return ToolLoopExhaustedError(
        f"Tool loop exhausted: {reason}",
        diagnostics=updated,
        continuation_state=continuation_state,
    )


def _handle_tool_loop_exhaustion(
    *,
    settings: DeepSeekRequestSettings,
    tool_output_staging_path: str,
    diagnostics: ToolLoopDiagnostics,
    allowed_speakers: set[str],
    segment_speaker_map: dict[str, str],
    word_budget: int | None,
    word_budget_tolerance: float,
    source_word_count: int,
    stream_event_callback: Callable[[dict[str, object]], None] | None,
    continuation_state: RetryContinuationState | None = None,
) -> tuple[PodcastScript, bool]:
    """Attempt final salvage when the bounded tool loop reaches round limit."""
    round_limit = max(1, settings.agent_max_tool_rounds)
    rounds_used = diagnostics["tool_rounds_used"]
    status_text = (
        f"Tool loop reached max rounds ({rounds_used}/{round_limit}). "
        "Attempting local staged-output salvage."
    )
    _emit_stream_token_event(stream_event_callback, phase="status", text=status_text)

    staged_candidate = _read_staged_output_text(tool_output_staging_path)
    staged_present = bool(staged_candidate.strip())
    diagnostics["staged_output_present"] = staged_present
    diagnostics["tool_loop_exhausted"] = True
    diagnostics["tool_loop_exhaustion_reason"] = "max_tool_rounds_reached"
    continuation_for_error = (
        replace(
            continuation_state,
            staged_output_present=staged_present,
            staged_output_valid_json=False,
        )
        if continuation_state is not None
        else None
    )

    if not staged_present:
        error = _build_tool_loop_exhaustion_error(
            diagnostics=diagnostics,
            reason="max_tool_rounds_reached_no_staged_output",
            issues=[
                "No staged JSON was available after tool-loop round exhaustion.",
                (
                    "Call write_output_segment to stage final JSON before finalization."
                ),
            ],
            continuation_state=continuation_for_error,
        )
        if stream_event_callback is not None:
            stream_event_callback(
                {
                    "event": "tool_loop_exhausted",
                    "diagnostics": cast(object, error.diagnostics),
                    "next_action_hint": (
                        "Stage valid JSON with write_output_segment, then call "
                        "evaluate_script_constraints."
                    ),
                }
            )
        _merge_tool_loop_diagnostics(diagnostics, error.diagnostics)
        raise error

    try:
        parsed_script, used_repair = _decode_podcast_script_with_fallback(staged_candidate)
        diagnostics["staged_output_valid_json"] = True
        if continuation_for_error is not None:
            continuation_for_error = replace(
                continuation_for_error,
                staged_output_valid_json=True,
            )
    except (JSONDecodeError, ValidationError, ValueError) as decode_error:
        diagnostics["staged_output_valid_json"] = False
        error = _build_tool_loop_exhaustion_error(
            diagnostics=diagnostics,
            reason="max_tool_rounds_reached_staged_json_invalid",
            issues=[
                "Staged output is not valid JSON matching the script schema.",
                _error_digest(decode_error),
            ],
            continuation_state=continuation_for_error,
        )
        if stream_event_callback is not None:
            stream_event_callback(
                {
                    "event": "tool_loop_exhausted",
                    "diagnostics": cast(object, error.diagnostics),
                    "next_action_hint": (
                        "Fix JSON structure in staged output and rerun constraint "
                        "evaluation before finalizing."
                    ),
                }
            )
        _merge_tool_loop_diagnostics(diagnostics, error.diagnostics)
        raise error

    local_constraints = _evaluate_script_constraints_tool(
        dialogue_payload=_tool_dialogue_payload_from_script(parsed_script),
        allowed_speakers=allowed_speakers,
        segment_speaker_map=segment_speaker_map,
        word_budget=word_budget,
        word_budget_tolerance=word_budget_tolerance,
        source_word_count=source_word_count,
    )
    if local_constraints["status"] == "pass":
        _emit_stream_token_event(
            stream_event_callback,
            phase="status",
            text=(
                "Tool loop exhausted but local salvage passed. "
                "Accepting staged JSON output."
            ),
        )
        return (
            _podcast_script_from_tool_dialogue(local_constraints["repaired_dialogue"]),
            used_repair,
        )

    budget_info = local_constraints["word_budget"]
    budget_issue = ""
    if budget_info["enabled"] and budget_info["in_range"] is False:
        budget_issue = (
            "Word budget out of range: "
            f"total={budget_info['total_words']} "
            f"target={budget_info['target_words']} "
            f"range=[{budget_info['lower_bound']},{budget_info['upper_bound']}]."
        )
    issues: list[str] = []
    issues.extend(local_constraints["validation"]["issues"])
    issues.extend(local_constraints["naturalness"]["issues"])
    if budget_issue:
        issues.append(budget_issue)
    if not issues:
        issues.extend(local_constraints["hints"][:2])

    error = _build_tool_loop_exhaustion_error(
        diagnostics=diagnostics,
        reason="max_tool_rounds_reached_constraints_failed",
        issues=issues,
        continuation_state=(
            replace(
                continuation_for_error,
                last_validation_issues=issues[:],
            )
            if continuation_for_error is not None
            else None
        ),
    )
    if stream_event_callback is not None:
        stream_event_callback(
            {
                "event": "tool_loop_exhausted",
                "diagnostics": cast(object, error.diagnostics),
                "next_action_hint": (
                    "Reduce repeat overwrites and call evaluate_script_constraints "
                    "after each material revision."
                ),
            }
        )
    _merge_tool_loop_diagnostics(diagnostics, error.diagnostics)
    raise error


def _deepseek_tool_schemas(tool_mode: AgentToolMode) -> list[dict[str, object]]:
    """Build DeepSeek tool schema definitions for local agentic evaluation."""
    tool_schemas: list[dict[str, object]] = []
    if tool_mode == "full_agentic":
        tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": READ_TRANSCRIPT_LINES_TOOL_NAME,
                    "description": (
                        "Read transcript lines by inclusive index range "
                        "and return segment_id/speaker/text."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_index": {"type": "integer"},
                            "end_index": {"type": "integer"},
                        },
                        "required": ["start_index", "end_index"],
                    },
                },
            }
        )

    tool_schemas.append(
        {
            "type": "function",
            "function": {
                "name": COUNT_WORDS_TOOL_NAME,
                "description": (
                    "Count words from candidate summary text. "
                    "Use this instead of estimating counts manually."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "lines": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
        }
    )

    tool_schemas.append(
        {
            "type": "function",
            "function": {
                "name": WRITE_OUTPUT_SEGMENT_TOOL_NAME,
                "description": (
                    "Write output JSON in chunks to a local staging file. "
                    "Use mode='overwrite' for the first chunk, then mode='append' "
                    "for later chunks."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["overwrite", "append"],
                        },
                        "content": {"type": "string"},
                    },
                    "required": ["content"],
                },
            },
        }
    )

    tool_schemas.append(
        {
            "type": "function",
            "function": {
                "name": EVALUATE_SCRIPT_TOOL_NAME,
                "description": (
                    "Check dialogue schema/speaker provenance, naturalness, "
                    "and word-budget constraints."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dialogue": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "speaker": {"type": "string"},
                                    "text": {"type": "string"},
                                    "emo_text": {"type": "string"},
                                    "emo_alpha": {"type": "number"},
                                    "source_segment_ids": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [
                                    "speaker",
                                    "text",
                                    "emo_text",
                                    "emo_alpha",
                                    "source_segment_ids",
                                ],
                            },
                        }
                    },
                    "required": ["dialogue"],
                },
            },
        }
    )
    return tool_schemas


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
    response: object | None = None
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


def _request_deepseek_completion(
    client: OpenAI,
    settings: DeepSeekRequestSettings,
    transcript_text: str,
    transcript_segments: list[TranscriptSegment],
    transcript_manifest: str,
    system_prompt: str,
    retry_context: RetryContext | None,
    endpoint_mode: EndpointMode,
    allowed_speakers: set[str],
    segment_speaker_map: dict[str, str],
    word_budget: int | None,
    word_budget_tolerance: float,
    source_word_count: int,
    tool_output_path: str | None = None,
    stream_event_callback: Callable[[dict[str, object]], None] | None = None,
    conversation_state: ConversationState | None = None,
    chat_log_writer: DeepSeekChatLogWriter | None = None,
) -> tuple[PodcastScript, bool, int, dict[str, int]]:
    """Request completion from a specific endpoint and parse JSON output.

    Args:
        client: OpenAI-compatible DeepSeek client.
        settings: Runtime settings for request and agent loop behavior.
        transcript_text: Prompt-ready transcript text with segment provenance.
        transcript_segments: Structured transcript segments for read-line tool.
        transcript_manifest: Compact transcript summary provided to the model.
        system_prompt: Instruction prompt with schema and policy constraints.
        retry_context: Prior-attempt failure context for corrective retries.
        endpoint_mode: Endpoint mode currently in use (beta or stable).
        allowed_speakers: Allowed transcript speakers for validation checks.
        segment_speaker_map: Segment ID to speaker lookup map.
        word_budget: Optional target total word budget.
        word_budget_tolerance: Inclusive tolerance ratio around word budget.
        source_word_count: Source transcript word count for short-source handling.
        tool_output_path: Optional output path used to derive a tool-write buffer file.
        stream_event_callback: Optional callback for dashboard stream events.
        conversation_state: Mutable conversation context reused across calls.
        chat_log_writer: Optional writer used for continuous per-call trace logs.

    Returns:
        Tuple of parsed script, local JSON-repair flag, tool-loop rounds used,
        and per-tool invocation counts.
    """
    tool_mode = _effective_agent_tool_mode(settings)
    tokens_limit = _clamp_completion_tokens_for_model(
        settings.max_completion_tokens,
        settings.model,
    )
    persist_reasoning_for_replay = _should_persist_reasoning_for_replay(
        settings,
        tool_mode=tool_mode,
    )
    if (
        _is_reasoner_model(settings.model)
        and tool_mode != "off"
        and not settings.agent_persist_reasoning_content
    ):
        logger.info(
            "Enabling reasoning_content replay for DeepSeek reasoner tool loop compatibility."
        )
    tool_loop_diagnostics = _new_tool_loop_diagnostics(settings.agent_max_tool_rounds)
    if conversation_state is not None:
        conversation_state.context_tokens_limit = tokens_limit
        conversation_state.last_tool_loop_diagnostics = _copy_tool_loop_diagnostics(
            tool_loop_diagnostics
        )

    if conversation_state is not None and conversation_state.messages:
        messages = _clone_messages(conversation_state.messages)
    else:
        messages = [{"role": "system", "content": system_prompt}]
        messages.append(
            {
                "role": "user",
                "content": _build_initial_user_message(
                    tool_mode=tool_mode,
                    transcript_text=transcript_text,
                    transcript_manifest=transcript_manifest,
                ),
            }
        )

    if retry_context is not None:
        logger.info(
            "Applying retry context: attempt=%d endpoint=%s type=%s detail=%s",
            retry_context.attempt_index,
            retry_context.endpoint_mode,
            retry_context.error_type,
            retry_context.error_digest,
        )
        messages.append(
            {
                "role": "system",
                "content": _build_retry_instruction(retry_context),
            }
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Regenerate the summary JSON now. Apply the latest corrective "
                    "feedback, preserve prior accepted constraints, and return a "
                    "fully compliant result."
                ),
            }
        )
        if retry_context.continuation is not None:
            messages.append(
                {
                    "role": "system",
                    "content": _build_retry_resume_guard_message(
                        retry_context.continuation
                    ),
                }
            )

    latest_context_snapshot: ContextUsageSnapshot | None = None
    if (
        conversation_state is not None
        and conversation_state.context_tokens_used is not None
    ):
        prior_limit = conversation_state.context_tokens_limit or tokens_limit
        latest_context_snapshot = _build_context_usage_snapshot(
            tokens_used=conversation_state.context_tokens_used,
            tokens_limit=prior_limit,
        )
    else:
        latest_context_snapshot = _build_context_usage_snapshot(
            tokens_used=0,
            tokens_limit=tokens_limit,
        )
    last_context_usage_emit_monotonic = 0.0

    def _write_call_event(call_id: str | None, payload: dict[str, object]) -> None:
        """Append one call-scoped event when chat tracing is enabled."""
        if chat_log_writer is None or call_id is None:
            return
        chat_log_writer.write_call_event(call_id, payload)

    def _persist_tool_loop_diagnostics() -> None:
        """Persist latest loop diagnostics in shared conversation state."""
        if conversation_state is None:
            return
        conversation_state.last_tool_loop_diagnostics = _copy_tool_loop_diagnostics(
            tool_loop_diagnostics
        )

    def _emit_context_usage_heartbeat(*, force: bool = False) -> None:
        """Emit context usage updates at most once per heartbeat interval."""
        nonlocal last_context_usage_emit_monotonic
        if latest_context_snapshot is None:
            return
        if stream_event_callback is None:
            return
        now = time.monotonic()
        if (
            not force
            and now - last_context_usage_emit_monotonic
            < CONTEXT_USAGE_UPDATE_INTERVAL_SECONDS
        ):
            return
        rollover_count = (
            conversation_state.rollover_count if conversation_state is not None else 0
        )
        _emit_context_usage_event(
            stream_event_callback,
            snapshot=latest_context_snapshot,
            rollover_triggered=False,
            rollover_count=rollover_count,
        )
        last_context_usage_emit_monotonic = now

    def _record_context_usage(total_tokens: int | None) -> None:
        """Persist and emit usage telemetry when available."""
        nonlocal latest_context_snapshot
        if total_tokens is None:
            return
        snapshot = _build_context_usage_snapshot(
            tokens_used=total_tokens,
            tokens_limit=tokens_limit,
        )
        latest_context_snapshot = snapshot
        if conversation_state is not None:
            conversation_state.context_tokens_used = snapshot.tokens_used
            conversation_state.context_tokens_limit = snapshot.tokens_limit
        _emit_context_usage_heartbeat(force=True)

    def _rollover_conversation_if_needed() -> None:
        """Compress context and reset chat history when remaining budget is low."""
        nonlocal latest_context_snapshot
        if latest_context_snapshot is None:
            return
        if not _should_rollover_context(latest_context_snapshot):
            return

        percent_left = latest_context_snapshot.percent_left * 100.0
        _emit_stream_token_event(
            stream_event_callback,
            phase="status",
            text=(
                f"Context window low ({percent_left:.1f}% left). "
                "Summarizing prior context and starting a new conversation window."
            ),
        )
        summary_text = _summarize_conversation_context_via_deepseek(
            client=client,
            conversation_messages=messages,
            tokens_limit=tokens_limit,
            endpoint_mode=endpoint_mode,
            chat_log_writer=chat_log_writer,
        )
        _reset_conversation_with_summary(
            messages=messages,
            system_prompt=system_prompt,
            summary_text=summary_text,
            tool_mode=tool_mode,
        )
        if retry_context is not None:
            messages.append(
                {
                    "role": "system",
                    "content": _build_retry_instruction(retry_context),
                }
            )
        if conversation_state is not None:
            conversation_state.rollover_count += 1
            conversation_state.context_tokens_used = 0
            conversation_state.context_tokens_limit = tokens_limit
        latest_context_snapshot = _build_context_usage_snapshot(
            tokens_used=0,
            tokens_limit=tokens_limit,
        )
        rollover_count = (
            conversation_state.rollover_count if conversation_state is not None else 1
        )
        _emit_context_usage_event(
            stream_event_callback,
            snapshot=latest_context_snapshot,
            rollover_triggered=True,
            rollover_count=rollover_count,
        )
        last_context_usage_emit_monotonic = time.monotonic()
        _emit_stream_token_event(
            stream_event_callback,
            phase="status",
            text="Context rollover complete. Continuing in new chat window.",
        )

    _emit_context_usage_heartbeat(force=True)
    _rollover_conversation_if_needed()

    base_request_payload: dict[str, Any] = {
        "model": settings.model,
        "messages": cast(Any, messages),
        "max_tokens": tokens_limit,
    }
    if settings.temperature is not None:
        base_request_payload["temperature"] = settings.temperature

    if tool_mode != "off":
        tool_output_staging_path = _resolve_tool_output_staging_path(tool_output_path)
        cleanup_staging_output = tool_output_path is None or not tool_output_path.strip()
        staging_init = _write_output_segment_tool(
            staging_path=tool_output_staging_path,
            mode="overwrite",
            content="",
            max_chunk_chars=DEFAULT_AGENT_WRITE_MAX_CHARS,
            max_file_chars=DEFAULT_AGENT_WRITE_MAX_FILE_CHARS,
        )
        if staging_init["status"] != "ok":
            logger.warning(
                "Failed to initialize staged output file path=%s hints=%s",
                tool_output_staging_path,
                staging_init["hints"],
            )
        if stream_event_callback is not None:
            stream_event_callback(
                {
                    "event": "start",
                    "model": settings.model,
                    "endpoint_mode": endpoint_mode,
                }
            )
        _emit_stream_token_event(
            stream_event_callback,
            phase="status",
            text=(
                f"Tool loop initialized (mode={tool_mode}). "
                "Waiting for tool calls, staged writes, and validation results."
            ),
        )
        tool_payload: dict[str, Any] = dict(base_request_payload)
        tool_payload["tools"] = cast(Any, _deepseek_tool_schemas(tool_mode))
        tool_payload["tool_choice"] = "auto"
        tool_payload["stream"] = True
        tool_payload["stream_options"] = cast(Any, {"include_usage": True})
        latest_tool_status: Literal["pass", "fail"] | None = None
        tool_call_counts: dict[str, int] = {}
        write_tool_succeeded = False
        read_ranges: list[tuple[int, int]] = []
        max_read_index: int | None = None
        retry_resume_used = (
            retry_context is not None and retry_context.continuation is not None
        )
        retry_max_read_index_prior = (
            retry_context.continuation.max_read_index
            if retry_context is not None and retry_context.continuation is not None
            else None
        )
        retry_read_from_start_detected = False
        max_tool_rounds = max(1, settings.agent_max_tool_rounds)
        rounds_used = 0
        repeated_overwrite_count = 0
        last_overwrite_hash: str | None = None

        def _snapshot_retry_continuation_state(
            *,
            validation_issues: list[str] | None = None,
            staged_output_present: bool | None = None,
            staged_output_valid_json: bool | None = None,
        ) -> RetryContinuationState:
            """Capture compact retry-resume context for downstream attempts."""
            latest_constraints_status: Literal["pass", "fail", "unknown"] = (
                latest_tool_status if latest_tool_status is not None else "unknown"
            )
            normalized_ranges = [
                (int(start), int(end))
                for start, end in read_ranges
            ]
            return RetryContinuationState(
                read_ranges=normalized_ranges,
                max_read_index=max_read_index,
                write_tool_succeeded=write_tool_succeeded,
                latest_constraints_status=latest_constraints_status,
                last_validation_issues=(
                    validation_issues[:]
                    if validation_issues is not None
                    else tool_loop_diagnostics["last_validation_issues"][:]
                ),
                staged_output_present=(
                    staged_output_present
                    if staged_output_present is not None
                    else tool_loop_diagnostics["staged_output_present"]
                ),
                staged_output_valid_json=(
                    staged_output_valid_json
                    if staged_output_valid_json is not None
                    else tool_loop_diagnostics["staged_output_valid_json"]
                ),
            )

        if retry_resume_used and stream_event_callback is not None:
            stream_event_callback(
                {
                    "event": "retry_resume_telemetry",
                    "retry_resume_used": True,
                    "retry_max_read_index_prior": retry_max_read_index_prior,
                    "retry_read_from_start_detected": False,
                }
            )
        try:
            while rounds_used < max_tool_rounds:
                _rollover_conversation_if_needed()
                tool_payload["messages"] = cast(Any, messages)
                rounds_used += 1
                tool_loop_diagnostics["tool_rounds_used"] = rounds_used
                tool_loop_diagnostics["repeated_overwrite_count"] = (
                    repeated_overwrite_count
                )
                _persist_tool_loop_diagnostics()
                logger.info(
                    "Tool loop iteration endpoint=%s model=%s round=%d/%d",
                    endpoint_mode,
                    settings.model,
                    rounds_used,
                    max_tool_rounds,
                )
                call_log_id: str | None = None
                turn: StreamedAssistantTurn | None = None
                if chat_log_writer is not None:
                    call_log_id = chat_log_writer.start_call(
                        call_type="tool_loop_round",
                        endpoint_mode=endpoint_mode,
                        model=settings.model,
                        metadata={
                            "round": rounds_used,
                            "tool_mode": tool_mode,
                            "messages_count": len(messages),
                        },
                    )
                try:
                    turn = _stream_assistant_turn(
                        client=client,
                        request_payload=tool_payload,
                        stream_event_callback=stream_event_callback,
                        chat_log_writer=chat_log_writer,
                        call_log_id=call_log_id,
                        on_chunk_callback=_emit_context_usage_heartbeat,
                    )
                except Exception as call_error:  # noqa: BLE001
                    if call_log_id is not None and chat_log_writer is not None:
                        chat_log_writer.finish_call(
                            call_log_id,
                            status="error",
                            error=_error_digest(call_error),
                        )
                    raise
                _record_context_usage(turn.usage_total_tokens)
                finish_reason = turn.finish_reason
                if finish_reason == "length":
                    if call_log_id is not None and chat_log_writer is not None:
                        chat_log_writer.finish_call(
                            call_log_id,
                            status="error",
                            finish_reason=finish_reason,
                            usage_total_tokens=turn.usage_total_tokens,
                            error="DeepSeek response truncated by max_tokens.",
                        )
                    raise OutputTruncatedError(
                        "DeepSeek response truncated by max_tokens (finish_reason=length)."
                    )

                assistant_message: dict[str, Any] = {"role": "assistant"}
                if turn.content:
                    assistant_message["content"] = turn.content
                if turn.reasoning_content and persist_reasoning_for_replay:
                    assistant_message["reasoning_content"] = turn.reasoning_content
                if turn.tool_calls:
                    assistant_message["tool_calls"] = turn.tool_calls
                messages.append(assistant_message)
                tool_payload["messages"] = cast(Any, messages)

                _write_call_event(
                    call_log_id,
                    {
                        "event": "assistant_message",
                        "has_content": bool(turn.content),
                        "has_reasoning_content": bool(turn.reasoning_content),
                        "tool_call_count": len(turn.tool_calls),
                    },
                )
                tool_calls = turn.tool_calls
                if tool_calls:
                    repeated_overwrite_blocked = False
                    for raw_tool_call in tool_calls:
                        tool_call_id = str(raw_tool_call.get("id", "")).strip()
                        function_obj = raw_tool_call.get("function", {})
                        function_name = ""
                        raw_arguments = "{}"
                        if isinstance(function_obj, dict):
                            function_name = str(function_obj.get("name", "")).strip()
                            raw_arguments = str(function_obj.get("arguments", "{}")).strip()
                        if not tool_call_id:
                            logger.warning(
                                "Skipping tool call without id for function=%s.",
                                function_name or "unknown",
                            )
                            continue

                        tool_name_display = function_name or "unknown"
                        tool_call_counts[tool_name_display] = (
                            tool_call_counts.get(tool_name_display, 0) + 1
                        )
                        tool_result: dict[str, object]
                        decoded_args: object | None
                        try:
                            decoded_args = json.loads(raw_arguments)
                            args_text = _format_tool_event_details(decoded_args)
                        except JSONDecodeError as error:
                            decoded_args = None
                            args_text = _format_tool_event_details(raw_arguments)
                            tool_result = {
                                "status": "fail",
                                "error": f"Invalid tool arguments JSON: {error.msg}",
                                "hints": [
                                    "Provide valid JSON arguments for the requested tool."
                                ],
                            }
                        _write_call_event(
                            call_log_id,
                            {
                                "event": "tool_invocation",
                                "tool_name": tool_name_display,
                                "arguments": args_text,
                            },
                        )
                        _emit_stream_token_event(
                            stream_event_callback,
                            phase="tool_call",
                            text=(
                                f"Invoked tool={tool_name_display} "
                                f"arguments={args_text}"
                            ),
                        )
                        if decoded_args is None:
                            pass
                        elif function_name == READ_TRANSCRIPT_LINES_TOOL_NAME:
                            if not isinstance(decoded_args, dict):
                                tool_result = {
                                    "status": "fail",
                                    "hints": ["Provide start_index and end_index integers."],
                                }
                            else:
                                try:
                                    start_index = int(decoded_args.get("start_index", 0))
                                    end_index = int(decoded_args.get("end_index", 0))
                                except (TypeError, ValueError):
                                    tool_result = {
                                        "status": "fail",
                                        "hints": [
                                            "start_index and end_index must be integers."
                                        ],
                                    }
                                else:
                                    read_result = _read_transcript_lines_tool(
                                        transcript_segments=transcript_segments,
                                        start_index=start_index,
                                        end_index=end_index,
                                        max_lines=settings.agent_read_max_lines,
                                    )
                                    if read_result["status"] == "ok":
                                        returned_start = read_result[
                                            "returned_start_index"
                                        ]
                                        returned_end = read_result["returned_end_index"]
                                        if (
                                            returned_start is not None
                                            and returned_end is not None
                                            and read_result["returned_count"] > 0
                                        ):
                                            normalized_start = min(
                                                int(returned_start),
                                                int(returned_end),
                                            )
                                            normalized_end = max(
                                                int(returned_start),
                                                int(returned_end),
                                            )
                                            read_ranges.append(
                                                (normalized_start, normalized_end)
                                            )
                                            max_read_index = (
                                                normalized_end
                                                if max_read_index is None
                                                else max(max_read_index, normalized_end)
                                            )
                                            if (
                                                retry_resume_used
                                                and retry_max_read_index_prior is not None
                                                and retry_max_read_index_prior > 0
                                                and normalized_start <= 0
                                                and not retry_read_from_start_detected
                                            ):
                                                retry_read_from_start_detected = True
                                                _emit_stream_token_event(
                                                    stream_event_callback,
                                                    phase="status",
                                                    text=(
                                                        "Retry resumed but read_transcript_lines "
                                                        "requested index 0 despite prior coverage. "
                                                        "Continue from later ranges unless required."
                                                    ),
                                                )
                                                if stream_event_callback is not None:
                                                    stream_event_callback(
                                                        {
                                                            "event": (
                                                                "retry_resume_telemetry"
                                                            ),
                                                            "retry_resume_used": True,
                                                            "retry_max_read_index_prior": (
                                                                retry_max_read_index_prior
                                                            ),
                                                            "retry_read_from_start_detected": True,
                                                            "read_start_index": normalized_start,
                                                            "read_end_index": normalized_end,
                                                        }
                                                    )
                                    tool_result = cast(dict[str, object], read_result)
                        elif function_name == COUNT_WORDS_TOOL_NAME:
                            if not isinstance(decoded_args, dict):
                                tool_result = {
                                    "status": "fail",
                                    "hints": ["Provide text or lines for word counting."],
                                }
                            else:
                                text_arg = decoded_args.get("text")
                                text = str(text_arg) if text_arg is not None else None
                                lines_arg = decoded_args.get("lines")
                                lines = (
                                    [str(line) for line in lines_arg]
                                    if isinstance(lines_arg, list)
                                    else None
                                )
                                count_result = _count_words_tool(
                                    text=text,
                                    lines=lines,
                                )
                                tool_result = cast(dict[str, object], count_result)
                        elif function_name == WRITE_OUTPUT_SEGMENT_TOOL_NAME:
                            if not isinstance(decoded_args, dict):
                                tool_result = {
                                    "status": "fail",
                                    "hints": [
                                        "Provide mode and content for segmented output writes."
                                    ],
                                }
                            else:
                                mode_arg = decoded_args.get("mode", "append")
                                content_arg = decoded_args.get("content", "")
                                mode = str(mode_arg) if mode_arg is not None else "append"
                                content = (
                                    str(content_arg) if content_arg is not None else ""
                                )
                                normalized_mode = mode.strip().lower()
                                if normalized_mode == "overwrite":
                                    overwrite_hash = _hash_text(content)
                                    same_as_previous = overwrite_hash == last_overwrite_hash
                                    next_repeated_count = (
                                        repeated_overwrite_count + 1
                                        if same_as_previous
                                        else 1
                                    )
                                    repeated_limit = max(
                                        1,
                                        int(settings.agent_max_repeated_write_overwrites),
                                    )
                                    if (
                                        same_as_previous
                                        and repeated_overwrite_count >= repeated_limit
                                    ):
                                        repeated_overwrite_count = next_repeated_count
                                        repeated_overwrite_blocked = True
                                        tool_result = {
                                            "status": "fail",
                                            "mode": normalized_mode,
                                            "path": tool_output_staging_path,
                                            "chunk_chars": len(content),
                                            "total_chars": len(content),
                                            "hints": [
                                                (
                                                    "Repeated overwrite detected for identical JSON "
                                                    "payload. Revise content, then call "
                                                    f"{EVALUATE_SCRIPT_TOOL_NAME} before writing again."
                                                )
                                            ],
                                        }
                                    else:
                                        write_result = _write_output_segment_tool(
                                            staging_path=tool_output_staging_path,
                                            mode=mode,
                                            content=content,
                                            max_chunk_chars=DEFAULT_AGENT_WRITE_MAX_CHARS,
                                            max_file_chars=DEFAULT_AGENT_WRITE_MAX_FILE_CHARS,
                                        )
                                        if write_result["status"] == "ok":
                                            write_tool_succeeded = True
                                            repeated_overwrite_count = next_repeated_count
                                            last_overwrite_hash = overwrite_hash
                                        tool_result = cast(dict[str, object], write_result)
                                else:
                                    write_result = _write_output_segment_tool(
                                        staging_path=tool_output_staging_path,
                                        mode=mode,
                                        content=content,
                                        max_chunk_chars=DEFAULT_AGENT_WRITE_MAX_CHARS,
                                        max_file_chars=DEFAULT_AGENT_WRITE_MAX_FILE_CHARS,
                                    )
                                    if write_result["status"] == "ok":
                                        write_tool_succeeded = True
                                    tool_result = cast(dict[str, object], write_result)
                        elif function_name == EVALUATE_SCRIPT_TOOL_NAME:
                            dialogue_payload = _coerce_tool_dialogue_lines(
                                decoded_args.get("dialogue", [])
                                if isinstance(decoded_args, dict)
                                else []
                            )
                            constraint_result = _evaluate_script_constraints_tool(
                                dialogue_payload=dialogue_payload,
                                allowed_speakers=allowed_speakers,
                                segment_speaker_map=segment_speaker_map,
                                word_budget=word_budget,
                                word_budget_tolerance=word_budget_tolerance,
                                source_word_count=source_word_count,
                            )
                            latest_tool_status = cast(
                                Literal["pass", "fail"], constraint_result["status"]
                            )
                            tool_result = cast(dict[str, object], constraint_result)
                            logger.info(
                                "Tool evaluation result: status=%s total_words=%d target=%s natural=%s",
                                constraint_result["status"],
                                constraint_result["word_budget"]["total_words"],
                                constraint_result["word_budget"]["target_words"],
                                constraint_result["naturalness"]["is_natural"],
                            )
                        else:
                            tool_result = {
                                "status": "fail",
                                "error": f"Unsupported tool name: {function_name}",
                                "hints": [
                                    (
                                        "Supported tools: "
                                        f"{READ_TRANSCRIPT_LINES_TOOL_NAME}, "
                                        f"{COUNT_WORDS_TOOL_NAME}, "
                                        f"{WRITE_OUTPUT_SEGMENT_TOOL_NAME}, "
                                        f"{EVALUATE_SCRIPT_TOOL_NAME}."
                                    )
                                ],
                            }

                        _emit_stream_token_event(
                            stream_event_callback,
                            phase="tool_result",
                            text=(
                                f"Tool result: tool={tool_name_display} "
                                f"status={tool_result.get('status', 'unknown')} "
                                f"details={_format_tool_event_details(tool_result)}"
                            ),
                        )
                        _write_call_event(
                            call_log_id,
                            {
                                "event": "tool_result",
                                "tool_name": tool_name_display,
                                "status": str(tool_result.get("status", "unknown")),
                                "details": _format_tool_event_details(tool_result),
                            },
                        )

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": str(tool_call_id),
                                "content": json.dumps(
                                    tool_result,
                                    ensure_ascii=True,
                                    separators=(",", ":"),
                                ),
                            }
                        )
                        tool_payload["messages"] = cast(Any, messages)
                    if repeated_overwrite_blocked:
                        messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "Detected repeated identical overwrite writes. "
                                    f"Call {EVALUATE_SCRIPT_TOOL_NAME} on your current "
                                    "dialogue draft, revise materially, then write again."
                                ),
                            }
                        )
                        tool_payload["messages"] = cast(Any, messages)
                    tool_loop_diagnostics["repeated_overwrite_count"] = (
                        repeated_overwrite_count
                    )
                    _persist_tool_loop_diagnostics()
                    if call_log_id is not None and chat_log_writer is not None:
                        chat_log_writer.finish_call(
                            call_log_id,
                            status="ok",
                            finish_reason=finish_reason,
                            usage_total_tokens=turn.usage_total_tokens,
                            metadata={"assistant_tool_call_count": len(turn.tool_calls)},
                        )
                    continue

                if (
                    tool_mode == "full_agentic"
                    and tool_call_counts.get(READ_TRANSCRIPT_LINES_TOOL_NAME, 0) == 0
                ):
                    _emit_stream_token_event(
                        stream_event_callback,
                        phase="status",
                        text=(
                            "Final JSON blocked: read_transcript_lines was not called. "
                            "Read transcript evidence first."
                        ),
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Before final JSON, call "
                                f"{READ_TRANSCRIPT_LINES_TOOL_NAME} to inspect transcript "
                                "evidence, then call evaluate tool."
                            ),
                        }
                    )
                    tool_payload["messages"] = cast(Any, messages)
                    if call_log_id is not None and chat_log_writer is not None:
                        chat_log_writer.finish_call(
                            call_log_id,
                            status="ok",
                            finish_reason=finish_reason,
                            usage_total_tokens=turn.usage_total_tokens,
                            metadata={"assistant_tool_call_count": len(turn.tool_calls)},
                        )
                    continue

                if latest_tool_status != "pass":
                    _emit_stream_token_event(
                        stream_event_callback,
                        phase="status",
                        text=(
                            "Assistant returned JSON before a passing tool result. "
                            "Requesting another tool-validated revision."
                        ),
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                f"Latest {EVALUATE_SCRIPT_TOOL_NAME} result is not pass. "
                                "Revise, call the tool again, then return final strict JSON."
                            ),
                        }
                    )
                    tool_payload["messages"] = cast(Any, messages)
                    if call_log_id is not None and chat_log_writer is not None:
                        chat_log_writer.finish_call(
                            call_log_id,
                            status="ok",
                            finish_reason=finish_reason,
                            usage_total_tokens=turn.usage_total_tokens,
                            metadata={"assistant_tool_call_count": len(turn.tool_calls)},
                        )
                    continue

                if not write_tool_succeeded:
                    _emit_stream_token_event(
                        stream_event_callback,
                        phase="status",
                        text=(
                            "Final JSON blocked: write_output_segment has no successful "
                            "writes yet. Stage output chunks before finalizing."
                        ),
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Before final response, stage JSON through "
                                f"{WRITE_OUTPUT_SEGMENT_TOOL_NAME} "
                                "(overwrite first chunk, append remaining chunks)."
                            ),
                        }
                    )
                    tool_payload["messages"] = cast(Any, messages)
                    if call_log_id is not None and chat_log_writer is not None:
                        chat_log_writer.finish_call(
                            call_log_id,
                            status="ok",
                            finish_reason=finish_reason,
                            usage_total_tokens=turn.usage_total_tokens,
                            metadata={"assistant_tool_call_count": len(turn.tool_calls)},
                        )
                    continue

                normalized_content = turn.content.strip()
                staged_candidate = ""
                if (
                    normalized_content == STAGED_OUTPUT_READY_MARKER
                    or not normalized_content
                ):
                    staged_candidate = _read_staged_output_text(tool_output_staging_path)
                    if staged_candidate.strip():
                        _emit_stream_token_event(
                            stream_event_callback,
                            phase="status",
                            text=(
                                "Using staged output from write_output_segment "
                                "for final JSON validation."
                            ),
                        )
                candidate_content = (
                    staged_candidate if staged_candidate.strip() else normalized_content
                )
                if not candidate_content.strip():
                    _emit_stream_token_event(
                        stream_event_callback,
                        phase="status",
                        text=(
                            "Final JSON missing. Continue staging with "
                            f"{WRITE_OUTPUT_SEGMENT_TOOL_NAME}."
                        ),
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Your final output was empty. Continue writing JSON chunks "
                                f"with {WRITE_OUTPUT_SEGMENT_TOOL_NAME}, then respond with "
                                f"{STAGED_OUTPUT_READY_MARKER}."
                            ),
                        }
                    )
                    tool_payload["messages"] = cast(Any, messages)
                    if call_log_id is not None and chat_log_writer is not None:
                        chat_log_writer.finish_call(
                            call_log_id,
                            status="ok",
                            finish_reason=finish_reason,
                            usage_total_tokens=turn.usage_total_tokens,
                            metadata={"assistant_tool_call_count": len(turn.tool_calls)},
                        )
                    continue

                try:
                    parsed_script, used_repair = _decode_podcast_script_with_fallback(
                        candidate_content
                    )
                except (JSONDecodeError, ValidationError, ValueError) as decode_error:
                    _emit_stream_token_event(
                        stream_event_callback,
                        phase="status",
                        text=(
                            "Final JSON validation failed after staged write: "
                            f"{_error_digest(decode_error)}"
                        ),
                    )
                    messages.append(
                        {
                            "role": "system",
                            "content": (
                                "Staged output is not valid strict JSON matching schema. "
                                f"Revise with {WRITE_OUTPUT_SEGMENT_TOOL_NAME}, call "
                                f"{EVALUATE_SCRIPT_TOOL_NAME}, then finalize."
                            ),
                        }
                    )
                    tool_payload["messages"] = cast(Any, messages)
                    if call_log_id is not None and chat_log_writer is not None:
                        chat_log_writer.finish_call(
                            call_log_id,
                            status="ok",
                            finish_reason=finish_reason,
                            usage_total_tokens=turn.usage_total_tokens,
                            metadata={"assistant_tool_call_count": len(turn.tool_calls)},
                        )
                    continue
                if used_repair:
                    logger.warning(
                        "Used balanced local JSON recovery for endpoint=%s.",
                        endpoint_mode,
                    )

                _emit_stream_token_event(
                    stream_event_callback,
                    phase="status",
                    text="Tool loop satisfied constraints. Final JSON accepted.",
                )
                if call_log_id is not None and chat_log_writer is not None:
                    chat_log_writer.finish_call(
                        call_log_id,
                        status="ok",
                        finish_reason=finish_reason,
                        usage_total_tokens=turn.usage_total_tokens,
                        metadata={"assistant_tool_call_count": len(turn.tool_calls)},
                    )
                tool_loop_diagnostics["tool_rounds_used"] = rounds_used
                tool_loop_diagnostics["repeated_overwrite_count"] = (
                    repeated_overwrite_count
                )
                tool_loop_diagnostics["tool_loop_exhausted"] = False
                tool_loop_diagnostics["tool_loop_exhaustion_reason"] = ""
                tool_loop_diagnostics["staged_output_present"] = bool(
                    staged_candidate.strip()
                )
                tool_loop_diagnostics["staged_output_valid_json"] = True
                tool_loop_diagnostics["last_validation_issues"] = []
                _persist_tool_loop_diagnostics()
                if conversation_state is not None:
                    conversation_state.messages = _clone_messages(messages)
                return parsed_script, used_repair, rounds_used, tool_call_counts
            tool_loop_diagnostics["tool_rounds_used"] = rounds_used
            tool_loop_diagnostics["repeated_overwrite_count"] = repeated_overwrite_count
            _persist_tool_loop_diagnostics()
            continuation_state = _snapshot_retry_continuation_state()
            if settings.agent_loop_exhaustion_policy == "auto_salvage":
                try:
                    parsed_script, used_repair = _handle_tool_loop_exhaustion(
                        settings=settings,
                        tool_output_staging_path=tool_output_staging_path,
                        diagnostics=tool_loop_diagnostics,
                        allowed_speakers=allowed_speakers,
                        segment_speaker_map=segment_speaker_map,
                        word_budget=word_budget,
                        word_budget_tolerance=word_budget_tolerance,
                        source_word_count=source_word_count,
                        stream_event_callback=stream_event_callback,
                        continuation_state=continuation_state,
                    )
                finally:
                    _persist_tool_loop_diagnostics()
                    if conversation_state is not None:
                        conversation_state.messages = _clone_messages(messages)
                if conversation_state is not None:
                    conversation_state.messages = _clone_messages(messages)
                return parsed_script, used_repair, rounds_used, tool_call_counts

            exhaustion_error = _build_tool_loop_exhaustion_error(
                diagnostics=tool_loop_diagnostics,
                reason="max_tool_rounds_reached_fail_fast",
                issues=[
                    (
                        "Tool loop reached max rounds and fail-fast policy is active. "
                        "Enable auto_salvage to validate staged output locally."
                    )
                ],
                continuation_state=continuation_state,
            )
            if stream_event_callback is not None:
                stream_event_callback(
                    {
                        "event": "tool_loop_exhausted",
                        "diagnostics": cast(object, exhaustion_error.diagnostics),
                        "next_action_hint": (
                            "Switch policy to auto_salvage, or reduce loop churn by "
                            "calling evaluate_script_constraints earlier."
                        ),
                    }
                )
            _merge_tool_loop_diagnostics(
                tool_loop_diagnostics, exhaustion_error.diagnostics
            )
            _persist_tool_loop_diagnostics()
            if conversation_state is not None:
                conversation_state.messages = _clone_messages(messages)
            raise exhaustion_error
        finally:
            if cleanup_staging_output:
                try:
                    Path(tool_output_staging_path).unlink(missing_ok=True)
                except OSError:
                    logger.debug(
                        "Could not delete temporary staged output file: %s",
                        tool_output_staging_path,
                    )
            if stream_event_callback is not None:
                stream_event_callback({"event": "done"})

    request_payload: dict[str, Any] = dict(base_request_payload)
    request_payload["response_format"] = cast(Any, {"type": "json_object"})

    content: str
    if _is_reasoner_model(settings.model):
        _rollover_conversation_if_needed()
        request_payload["messages"] = cast(Any, messages)
        request_payload["stream"] = True
        request_payload["stream_options"] = cast(Any, {"include_usage": True})
        reasoner_call_log_id: str | None = None
        if chat_log_writer is not None:
            reasoner_call_log_id = chat_log_writer.start_call(
                call_type="single_stream_completion",
                endpoint_mode=endpoint_mode,
                model=settings.model,
                metadata={"messages_count": len(messages)},
            )
        if stream_event_callback is not None:
            stream_event_callback(
                {
                    "event": "start",
                    "model": settings.model,
                    "endpoint_mode": endpoint_mode,
                }
            )
        streamed_answer_parts: list[str] = []
        streamed_reasoning_parts: list[str] = []
        finish_reason: str | None = None
        usage_total_tokens: int | None = None
        stream_call_status: Literal["ok", "error"] = "ok"
        stream_call_error: str | None = None
        stream_log_buffer: _StreamCallLogBuffer | None = None
        if chat_log_writer is not None and reasoner_call_log_id is not None:
            def _write_reasoner_call_event(payload: dict[str, object]) -> None:
                """Write one call-scoped event for reasoner stream traces."""
                _write_call_event(reasoner_call_log_id, payload)

            stream_log_buffer = _StreamCallLogBuffer(
                write_event=_write_reasoner_call_event,
                interval_seconds=chat_log_writer.stream_flush_interval_seconds,
            )
        try:
            stream_response = client.chat.completions.create(**cast(Any, request_payload))
            for chunk in stream_response:
                _emit_context_usage_heartbeat()
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
                    streamed_reasoning_parts.append(reasoning_piece)
                    if stream_log_buffer is not None:
                        stream_log_buffer.append(phase="reasoning", text=reasoning_piece)
                    if stream_event_callback is not None:
                        stream_event_callback(
                            {
                                "event": "token",
                                "phase": "reasoning",
                                "text": reasoning_piece,
                            }
                        )
                answer_piece = _normalize_stream_text_piece(getattr(delta, "content", None))
                if answer_piece:
                    streamed_answer_parts.append(answer_piece)
                    if stream_log_buffer is not None:
                        stream_log_buffer.append(phase="answer", text=answer_piece)
                    if stream_event_callback is not None:
                        stream_event_callback(
                            {
                                "event": "token",
                                "phase": "answer",
                                "text": answer_piece,
                            }
                        )
        except Exception as stream_error:  # noqa: BLE001
            stream_call_status = "error"
            stream_call_error = _error_digest(stream_error)
            raise
        finally:
            if stream_event_callback is not None:
                stream_event_callback({"event": "done"})
            if stream_log_buffer is not None:
                stream_log_buffer.flush(force=True, trigger="stream_end")
            if (
                stream_call_status == "error"
                and reasoner_call_log_id is not None
                and chat_log_writer is not None
            ):
                chat_log_writer.finish_call(
                    reasoner_call_log_id,
                    status=stream_call_status,
                    finish_reason=finish_reason,
                    usage_total_tokens=usage_total_tokens,
                    error=stream_call_error,
                    metadata={"assistant_tool_call_count": 0},
                )
        _record_context_usage(usage_total_tokens)
        if finish_reason == "length":
            if reasoner_call_log_id is not None and chat_log_writer is not None:
                chat_log_writer.finish_call(
                    reasoner_call_log_id,
                    status="error",
                    finish_reason=finish_reason,
                    usage_total_tokens=usage_total_tokens,
                    error="DeepSeek response truncated by max_tokens.",
                )
            raise OutputTruncatedError(
                "DeepSeek response truncated by max_tokens (finish_reason=length)."
            )
        content = "".join(streamed_answer_parts)
        assistant_message: dict[str, Any] = {"role": "assistant"}
        if content:
            assistant_message["content"] = content
        if streamed_reasoning_parts and persist_reasoning_for_replay:
            assistant_message["reasoning_content"] = "".join(streamed_reasoning_parts)
        messages.append(assistant_message)
        _write_call_event(
            reasoner_call_log_id,
            {
                "event": "message",
                "phase": "answer",
                "text": content,
            },
        )
        if reasoner_call_log_id is not None and chat_log_writer is not None:
            chat_log_writer.finish_call(
                reasoner_call_log_id,
                status="ok",
                finish_reason=finish_reason,
                usage_total_tokens=usage_total_tokens,
                metadata={"assistant_tool_call_count": 0},
            )
    else:
        _rollover_conversation_if_needed()
        request_payload["messages"] = cast(Any, messages)
        chat_call_log_id: str | None = None
        if chat_log_writer is not None:
            chat_call_log_id = chat_log_writer.start_call(
                call_type="single_completion",
                endpoint_mode=endpoint_mode,
                model=settings.model,
                metadata={"messages_count": len(messages)},
            )
        # DeepSeek's OpenAI-compatible endpoint accepts these parameters at runtime.
        # The upstream OpenAI SDK type stubs are stricter than the provider surface,
        # so we cast request payload fields for static typing compatibility.
        try:
            response = client.chat.completions.create(**cast(Any, request_payload))
        except Exception as call_error:  # noqa: BLE001
            if chat_call_log_id is not None and chat_log_writer is not None:
                chat_log_writer.finish_call(
                    chat_call_log_id,
                    status="error",
                    error=_error_digest(call_error),
                )
            raise
        _record_context_usage(_extract_total_tokens_from_usage(getattr(response, "usage", None)))
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "length":
            if chat_call_log_id is not None and chat_log_writer is not None:
                chat_log_writer.finish_call(
                    chat_call_log_id,
                    status="error",
                    finish_reason=finish_reason,
                    usage_total_tokens=_extract_total_tokens_from_usage(
                        getattr(response, "usage", None)
                    ),
                    error="DeepSeek response truncated by max_tokens.",
                )
            raise OutputTruncatedError(
                "DeepSeek response truncated by max_tokens (finish_reason=length)."
            )
        content = _normalize_assistant_content(choice.message.content)
        messages.append({"role": "assistant", "content": content})
        _write_call_event(
            chat_call_log_id,
            {
                "event": "message",
                "phase": "answer",
                "text": content,
            },
        )
        if chat_call_log_id is not None and chat_log_writer is not None:
            chat_log_writer.finish_call(
                chat_call_log_id,
                status="ok",
                finish_reason=finish_reason,
                usage_total_tokens=_extract_total_tokens_from_usage(
                    getattr(response, "usage", None)
                ),
                metadata={"assistant_tool_call_count": 0},
            )

    if not isinstance(content, str) or not content.strip():
        raise ValueError("DeepSeek returned empty response.")

    parsed_script, used_repair = _decode_podcast_script_with_fallback(content)
    if used_repair:
        logger.warning(
            "Used balanced local JSON recovery for endpoint=%s.",
            endpoint_mode,
        )
    tool_loop_diagnostics["tool_rounds_used"] = 0
    tool_loop_diagnostics["repeated_overwrite_count"] = 0
    tool_loop_diagnostics["tool_loop_exhausted"] = False
    tool_loop_diagnostics["tool_loop_exhaustion_reason"] = ""
    tool_loop_diagnostics["staged_output_present"] = False
    tool_loop_diagnostics["staged_output_valid_json"] = False
    tool_loop_diagnostics["last_validation_issues"] = []
    _persist_tool_loop_diagnostics()
    if conversation_state is not None:
        conversation_state.messages = _clone_messages(messages)
    return parsed_script, used_repair, 0, {}


def _target_line_bounds(
    segment_count: int,
    word_budget: int | None,
) -> tuple[int, int]:
    """Estimate summary line-count bounds from segment count and word budget."""
    if word_budget is not None and word_budget > 0:
        estimated = max(1, int(round(word_budget / SUMMARY_LINE_WORD_TARGET)))
        lower_bound = max(SUMMARY_LINE_COUNT_MIN, estimated - 4)
        upper_bound = min(SUMMARY_LINE_COUNT_MAX, estimated + 5)
    else:
        estimated = max(1, segment_count // 2)
        lower_bound = max(SUMMARY_LINE_COUNT_MIN, estimated - 8)
        upper_bound = min(SUMMARY_LINE_COUNT_MAX, estimated + 8)
    if lower_bound > upper_bound:
        lower_bound = upper_bound
    return lower_bound, upper_bound


def generate_summary_deepseek(
    transcript_text: str,
    transcript_segments: list[TranscriptSegment],
    api_key: str,
    allowed_speakers: set[str],
    segment_speaker_map: dict[str, str],
    source_word_count: int,
    settings: DeepSeekRequestSettings,
    segment_count: int,
    retry_context: RetryContext | None,
    word_budget: int | None,
    target_minutes: float | None,
    avg_wpm: float | None,
    word_budget_tolerance: float,
    tool_output_path: str | None = None,
    stream_event_callback: Callable[[dict[str, object]], None] | None = None,
    conversation_state: ConversationState | None = None,
    chat_log_writer: DeepSeekChatLogWriter | None = None,
) -> GenerationSuccess:
    """Generate a structured summary from transcript text via DeepSeek.

    Args:
        transcript_text: Prompt-ready transcript with segment IDs.
        transcript_segments: Structured transcript segments used by read tools.
        api_key: DeepSeek API key.
        allowed_speakers: Speaker labels allowed in output.
        segment_speaker_map: Segment ID to canonical speaker mapping.
        source_word_count: Total source transcript words.
        settings: Request settings for DeepSeek chat completions.
        segment_count: Number of input transcript segments.
        retry_context: Previous-attempt failure context.
        word_budget: Optional target word budget.
        target_minutes: Optional summary target duration.
        avg_wpm: Optional calibrated words-per-minute value.
        word_budget_tolerance: Inclusive tolerance ratio for word budget.
        tool_output_path: Optional file path used as staging target for tool writes.
        stream_event_callback: Optional callback for dashboard stream updates.
        conversation_state: Mutable conversation state reused across attempts.
        chat_log_writer: Optional writer used for continuous per-call trace logs.

    Returns:
        Structured generation result and endpoint metadata.

    Raises:
        OpenAIAPIError: Upstream API issues that should trigger retry.
        JSONDecodeError: Response content is not parseable JSON.
        ValidationError: Parsed payload is not compatible with `PodcastScript`.
        ValueError: Empty response payload.
    """
    effective_tool_mode = _effective_agent_tool_mode(settings)
    target_min_lines, target_max_lines = _target_line_bounds(segment_count, word_budget)
    system_prompt = _build_system_prompt(
        allowed_speakers=allowed_speakers,
        target_min_lines=target_min_lines,
        target_max_lines=target_max_lines,
        word_budget=word_budget,
        target_minutes=target_minutes,
        avg_wpm=avg_wpm,
        word_budget_tolerance=word_budget_tolerance,
        require_tool_call=effective_tool_mode != "off",
        tool_mode=effective_tool_mode,
        max_repeated_write_overwrites=settings.agent_max_repeated_write_overwrites,
    )
    transcript_manifest = _build_transcript_manifest(transcript_segments)

    endpoint_plan: list[tuple[EndpointMode, str]] = []
    if settings.auto_beta:
        endpoint_plan.append(("beta", DEEPSEEK_BETA_BASE_URL))
    endpoint_plan.append(("stable", DEEPSEEK_STABLE_BASE_URL))

    last_error: Exception | None = None
    for endpoint_mode, base_url in endpoint_plan:
        endpoint_settings = settings
        client = _build_deepseek_client(
            api_key=api_key,
            base_url=base_url,
            request_timeout_seconds=endpoint_settings.request_timeout_seconds,
            http_retries=endpoint_settings.http_retries,
        )
        model_candidates: list[str] = [endpoint_settings.model]
        if (
            effective_tool_mode != "off"
            and endpoint_settings.agent_allow_model_fallback
        ):
            for candidate in (DEEPSEEK_CHAT_MODEL, DEEPSEEK_REASONER_MODEL):
                if _normalized_model_name(candidate) not in {
                    _normalized_model_name(model_name)
                    for model_name in model_candidates
                }:
                    model_candidates.append(candidate)

        for model_index, candidate_model in enumerate(model_candidates):
            candidate_temperature = endpoint_settings.temperature
            if _is_reasoner_model(candidate_model):
                candidate_temperature = None

            candidate_max_completion_tokens = _clamp_completion_tokens_for_model(
                endpoint_settings.max_completion_tokens,
                candidate_model,
            )
            if candidate_max_completion_tokens < endpoint_settings.max_completion_tokens:
                logger.info(
                    "Clamping max_tokens for model=%s from %d to %d based on model ceiling.",
                    candidate_model,
                    endpoint_settings.max_completion_tokens,
                    candidate_max_completion_tokens,
                )

            candidate_settings = DeepSeekRequestSettings(
                model=candidate_model,
                max_completion_tokens=candidate_max_completion_tokens,
                request_timeout_seconds=endpoint_settings.request_timeout_seconds,
                http_retries=endpoint_settings.http_retries,
                temperature=candidate_temperature,
                auto_beta=endpoint_settings.auto_beta,
                agent_tool_loop=endpoint_settings.agent_tool_loop,
                agent_tool_mode=endpoint_settings.agent_tool_mode,
                agent_max_tool_rounds=endpoint_settings.agent_max_tool_rounds,
                agent_read_max_lines=endpoint_settings.agent_read_max_lines,
                agent_loop_exhaustion_policy=(
                    endpoint_settings.agent_loop_exhaustion_policy
                ),
                agent_max_repeated_write_overwrites=(
                    endpoint_settings.agent_max_repeated_write_overwrites
                ),
                agent_persist_reasoning_content=(
                    endpoint_settings.agent_persist_reasoning_content
                ),
                agent_allow_model_fallback=(
                    endpoint_settings.agent_allow_model_fallback
                ),
            )
            temperature_label = (
                f"{candidate_settings.temperature:.2f}"
                if candidate_settings.temperature is not None
                else "ignored(reasoner)"
            )
            effective_reasoning_replay = _should_persist_reasoning_for_replay(
                candidate_settings,
                tool_mode=effective_tool_mode,
            )
            logger.info(
                "Sending transcript to DeepSeek endpoint=%s model=%s max_tokens=%d temperature=%s timeout=%.1fs retries=%d tool_loop=%s tool_mode=%s max_tool_rounds=%d read_max_lines=%d exhaustion_policy=%s max_repeated_overwrites=%d persist_reasoning=%s effective_reasoning_replay=%s",
                endpoint_mode,
                candidate_settings.model,
                candidate_settings.max_completion_tokens,
                temperature_label,
                candidate_settings.request_timeout_seconds,
                candidate_settings.http_retries,
                candidate_settings.agent_tool_loop,
                candidate_settings.agent_tool_mode,
                candidate_settings.agent_max_tool_rounds,
                candidate_settings.agent_read_max_lines,
                candidate_settings.agent_loop_exhaustion_policy,
                candidate_settings.agent_max_repeated_write_overwrites,
                candidate_settings.agent_persist_reasoning_content,
                effective_reasoning_replay,
            )

            try:
                script, used_repair, tool_rounds, tool_call_counts = _request_deepseek_completion(
                    client=client,
                    settings=candidate_settings,
                    transcript_text=transcript_text,
                    transcript_segments=transcript_segments,
                    transcript_manifest=transcript_manifest,
                    system_prompt=system_prompt,
                    retry_context=retry_context,
                    endpoint_mode=endpoint_mode,
                    allowed_speakers=allowed_speakers,
                    segment_speaker_map=segment_speaker_map,
                    word_budget=word_budget,
                    word_budget_tolerance=word_budget_tolerance,
                    source_word_count=source_word_count,
                    tool_output_path=tool_output_path,
                    stream_event_callback=stream_event_callback,
                    conversation_state=conversation_state,
                    chat_log_writer=chat_log_writer,
                )
                return GenerationSuccess(
                    script=script,
                    endpoint_mode=endpoint_mode,
                    used_json_repair=used_repair,
                    model=candidate_settings.model,
                    tool_rounds=tool_rounds,
                    tool_call_counts=tool_call_counts,
                    tool_loop_diagnostics=(
                        _copy_tool_loop_diagnostics(
                            conversation_state.last_tool_loop_diagnostics
                        )
                        if (
                            conversation_state is not None
                            and conversation_state.last_tool_loop_diagnostics is not None
                        )
                        else None
                    ),
                )
            except Exception as error:  # noqa: BLE001
                last_error = error
                if (
                    model_index < len(model_candidates) - 1
                    and _looks_like_tool_protocol_error(error)
                ):
                    next_model = model_candidates[model_index + 1]
                    logger.warning(
                        "DeepSeek model=%s tool-calling protocol failed (%s). Falling back to model=%s.",
                        candidate_settings.model,
                        _error_digest(error),
                        next_model,
                    )
                    _emit_stream_token_event(
                        stream_event_callback,
                        phase="status",
                        text=(
                            f"Tool protocol mismatch on {candidate_settings.model}; "
                            f"retrying with {next_model}."
                        ),
                    )
                    continue
                if endpoint_mode == "beta" and _looks_like_beta_endpoint_error(error):
                    logger.warning(
                        "DeepSeek beta endpoint failed (%s). Falling back to stable endpoint.",
                        _error_digest(error),
                    )
                    _emit_stream_token_event(
                        stream_event_callback,
                        phase="status",
                        text="DeepSeek beta endpoint failed; retrying on stable endpoint.",
                    )
                    break
                raise GenerationAttemptError(
                    endpoint_mode=endpoint_mode, cause=error
                ) from error

    if last_error is not None:
        raise GenerationAttemptError(
            endpoint_mode="stable", cause=last_error
        ) from last_error
    raise RuntimeError("No DeepSeek endpoint candidates were available.")


def post_process_script(
    lines: list[ValidatedDialogueLine], voice_sample_dir: str
) -> list[FinalScriptLine]:
    """Inject voice paths and validation metadata into final output schema.

    Args:
        lines: Validated and repaired dialogue lines.
        voice_sample_dir: Directory that stores speaker reference wav files.

    Returns:
        Final JSON-ready script lines.
    """
    final_output: list[FinalScriptLine] = []

    for line in lines:
        voice_path = os.path.join(voice_sample_dir, f"{line.speaker}.wav")
        normalized_voice_path = voice_path.replace("\\", "/")
        if not os.path.exists(voice_path):
            logger.warning(
                "Voice sample not found for speaker '%s': %s",
                line.speaker,
                voice_path,
            )

        final_output.append(
            {
                "speaker": line.speaker,
                "voice_sample": normalized_voice_path,
                "use_emo_text": True,
                "emo_text": line.emo_text,
                "emo_alpha": line.emo_alpha,
                "text": line.text,
                "source_segment_ids": line.source_segment_ids,
                "validation_status": line.validation_status,
                "repair_reason": line.repair_reason,
            }
        )

    return final_output


def _count_words(lines: list[ValidatedDialogueLine]) -> int:
    """Count dialogue words only from line text fields.

    This intentionally excludes JSON structure, keys, and metadata.
    """
    return sum(len(line.text.split()) for line in lines)


def _naturalness_metrics_from_validated_lines(
    lines: list[ValidatedDialogueLine],
) -> dict[str, float | int]:
    """Compute summary-level naturalness metrics for diagnostics reporting."""
    texts = [line.text for line in lines]
    if not texts:
        return {
            "avg_words_per_line": 0.0,
            "short_question_ratio": 0.0,
            "disfluency_count": 0,
            "line_count": 0,
        }
    word_counts = [_count_words_in_text(text) for text in texts]
    short_question_lines = sum(
        1
        for text, words in zip(texts, word_counts)
        if text.strip().endswith("?") and words <= 5
    )
    disfluency_count = sum(len(DISFLUENCY_PATTERN.findall(text)) for text in texts)
    return {
        "avg_words_per_line": float(sum(word_counts)) / float(len(word_counts)),
        "short_question_ratio": float(short_question_lines) / float(len(word_counts)),
        "disfluency_count": disfluency_count,
        "line_count": len(texts),
    }


def _count_words_from_segments(segments: list[dict]) -> int:
    """Count words across transcript segments."""
    total = 0
    for seg in segments:
        if hasattr(seg, "text"):
            text = getattr(seg, "text", "")
        elif isinstance(seg, dict):
            text = seg.get("text", "")
        else:
            text = str(seg)
        total += len(str(text).split())
    return total


def _budget_bounds(word_budget: int, tolerance: float) -> tuple[int, int]:
    """Compute inclusive word-budget bounds."""
    if word_budget <= 0:
        return 0, 0
    lower = int(word_budget * (1.0 - tolerance))
    upper = int(word_budget * (1.0 + tolerance))
    return max(1, lower), max(1, upper)


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


def _resolve_summary_report_path(output_path: str, report_path: str | None) -> str:
    """Resolve the diagnostics report path for summarizer output."""
    if report_path and report_path.strip():
        return report_path
    output = Path(output_path)
    return str(output.with_name(f"{output.name}.report.json"))


def _write_summary_report(report_path: str, report: SummaryReport) -> None:
    """Write deterministic summary diagnostics report to disk."""
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)


def main() -> int:
    """Run the DeepSeek summarizer with strict transcript-grounded speaker validation."""
    parser = argparse.ArgumentParser(
        description="CARD Script Summarizer & Emotion Annotator (Deepseek)"
    )
    parser.add_argument(
        "--transcript", required=True, help="Path to input WhisperX JSON transcript"
    )
    parser.add_argument(
        "--voice-dir",
        required=True,
        help="Directory where separated speaker audios are stored",
    )
    parser.add_argument(
        "--output",
        default="summarized_script.json",
        help="Path to save output JSON",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help="DeepSeek API Key",
    )
    parser.add_argument(
        "--model",
        default=DEEPSEEK_MODEL,
        help=f"DeepSeek model to use (default: {DEEPSEEK_MODEL})",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES_DEFAULT,
        help=(
            "Max retry attempts for LLM generation and validation "
            f"(default: {MAX_RETRIES_DEFAULT})"
        ),
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help=(
            "Maximum output token budget for DeepSeek responses; "
            f"default: {DEFAULT_MAX_COMPLETION_TOKENS}"
        ),
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=(
            "HTTP request timeout in seconds for DeepSeek requests; "
            f"default: {DEFAULT_TIMEOUT_SECONDS}"
        ),
    )
    parser.add_argument(
        "--http-retries",
        type=int,
        default=DEFAULT_HTTP_RETRIES,
        help=f"Transport retries per HTTP request; default: {DEFAULT_HTTP_RETRIES}",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=(
            "Sampling temperature for DeepSeek generation "
            f"(ignored for {DEEPSEEK_REASONER_MODEL}); default: {DEFAULT_TEMPERATURE}"
        ),
    )
    parser.add_argument(
        "--target-minutes",
        type=float,
        default=None,
        help="Target summary duration in minutes",
    )
    parser.add_argument(
        "--avg-wpm",
        type=float,
        default=None,
        help="Average calibrated WPM for word budget",
    )
    parser.add_argument(
        "--word-budget",
        type=int,
        default=None,
        help="Target total word count for summary",
    )
    parser.add_argument(
        "--word-budget-tolerance",
        type=float,
        default=0.05,
        help="Tolerance ratio for word budget (default: 0.05 = +/-5%%)",
    )
    parser.add_argument(
        "--agent-tool-loop",
        choices=["on", "off"],
        default="on",
        help=(
            "Enable DeepSeek tool-calling loop against local constraint checker "
            "(default: on)."
        ),
    )
    parser.add_argument(
        "--agent-tool-mode",
        choices=["constraints_only", "full_agentic"],
        default=DEFAULT_AGENT_TOOL_MODE,
        help=(
            "Agentic tool profile when tool loop is enabled: "
            "'constraints_only' or 'full_agentic' (default: full_agentic)."
        ),
    )
    parser.add_argument(
        "--agent-max-tool-rounds",
        type=int,
        default=DEFAULT_AGENT_MAX_TOOL_ROUNDS,
        help=(
            "Maximum tool-loop rounds before exhaustion handling is applied "
            f"(default: {DEFAULT_AGENT_MAX_TOOL_ROUNDS})."
        ),
    )
    parser.add_argument(
        "--agent-loop-exhaustion-policy",
        choices=["auto_salvage", "fail_fast"],
        default=DEFAULT_AGENT_LOOP_EXHAUSTION_POLICY,
        help=(
            "Behavior when max tool rounds are reached: attempt local staged-output "
            "salvage or fail immediately "
            f"(default: {DEFAULT_AGENT_LOOP_EXHAUSTION_POLICY})."
        ),
    )
    parser.add_argument(
        "--agent-max-repeated-write-overwrites",
        type=int,
        default=DEFAULT_AGENT_MAX_REPEATED_WRITE_OVERWRITES,
        help=(
            "Maximum identical overwrite writes allowed before forcing revision "
            f"(default: {DEFAULT_AGENT_MAX_REPEATED_WRITE_OVERWRITES})."
        ),
    )
    parser.add_argument(
        "--agent-persist-reasoning-content",
        choices=["on", "off"],
        default=(
            "on" if DEFAULT_AGENT_PERSIST_REASONING_CONTENT else "off"
        ),
        help=(
            "Persist assistant reasoning_content in replay messages "
            "(default: off)."
        ),
    )
    parser.add_argument(
        "--agent-allow-model-fallback",
        choices=["on", "off"],
        default=("on" if DEFAULT_AGENT_ALLOW_MODEL_FALLBACK else "off"),
        help=(
            "Allow fallback between deepseek-reasoner and deepseek-chat when "
            "tool-protocol errors occur (default: off)."
        ),
    )
    parser.add_argument(
        "--agent-read-max-lines",
        type=int,
        default=DEFAULT_AGENT_READ_MAX_LINES,
        help=(
            "Maximum transcript lines returned per read-transcript tool call "
            f"(default: {DEFAULT_AGENT_READ_MAX_LINES})."
        ),
    )
    parser.add_argument(
        "--budget-failure-policy",
        choices=["degraded_success", "strict_fail"],
        default=DEFAULT_BUDGET_FAILURE_POLICY,
        help=(
            "Behavior when constraints remain unmet after all retries "
            f"(default: {DEFAULT_BUDGET_FAILURE_POLICY})."
        ),
    )
    parser.add_argument(
        "--summary-report",
        default=None,
        help="Optional path for sidecar summary diagnostics JSON.",
    )
    parser.add_argument(
        "--deepseek-chat-log-root",
        default=DEEPSEEK_CHAT_LOG_ROOT_DIR,
        help=(
            "Directory where timestamped per-call DeepSeek trace logs are written "
            f"(default: {DEEPSEEK_CHAT_LOG_ROOT_DIR})."
        ),
    )
    args = parser.parse_args()
    configure_logging(
        level=os.getenv("AUDIO2SCRIPT_LOG_LEVEL", "INFO"),
        component="summarizer_deepseek",
    )

    if not args.api_key:
        logger.error("No API key provided. Set DEEPSEEK_API_KEY or pass --api-key.")
        return 1

    requested_temperature = min(2.0, max(0.0, float(args.temperature)))
    effective_temperature: float | None = requested_temperature
    if _is_reasoner_model(args.model):
        if not math.isclose(requested_temperature, DEFAULT_TEMPERATURE, abs_tol=1e-9):
            logger.warning(
                "Model %s does not support temperature; requested %.2f will be ignored.",
                args.model,
                requested_temperature,
            )
        effective_temperature = None

    requested_max_tokens = max(TOKEN_BUDGET_MIN, int(args.max_completion_tokens))
    model_token_ceiling = _model_completion_token_ceiling(args.model)
    configured_max_tokens = min(requested_max_tokens, model_token_ceiling)
    if configured_max_tokens < requested_max_tokens:
        logger.warning(
            "Requested max-completion-tokens=%d exceeds model ceiling=%d for %s; clamping.",
            requested_max_tokens,
            model_token_ceiling,
            args.model,
        )

    agent_tool_loop_enabled = _parse_on_off_flag(args.agent_tool_loop)
    agent_tool_mode: AgentToolMode = (
        cast(AgentToolMode, args.agent_tool_mode)
        if agent_tool_loop_enabled
        else "off"
    )
    agent_loop_exhaustion_policy: AgentLoopExhaustionPolicy = cast(
        AgentLoopExhaustionPolicy,
        args.agent_loop_exhaustion_policy,
    )
    agent_persist_reasoning_content = _parse_on_off_flag(
        args.agent_persist_reasoning_content
    )
    agent_allow_model_fallback = _parse_on_off_flag(args.agent_allow_model_fallback)
    budget_failure_policy: BudgetFailurePolicy = cast(
        BudgetFailurePolicy, args.budget_failure_policy
    )
    summary_report_path = _resolve_summary_report_path(args.output, args.summary_report)

    # Keep max_completion_tokens as the output ceiling to avoid accidental truncation.
    # Word-budget compliance is handled via deterministic tool feedback and retries.
    settings = DeepSeekRequestSettings(
        model=args.model,
        max_completion_tokens=configured_max_tokens,
        request_timeout_seconds=max(5.0, float(args.request_timeout_seconds)),
        http_retries=max(0, int(args.http_retries)),
        temperature=effective_temperature,
        auto_beta=True,
        agent_tool_loop=agent_tool_loop_enabled,
        agent_tool_mode=agent_tool_mode,
        agent_max_tool_rounds=max(1, int(args.agent_max_tool_rounds)),
        agent_read_max_lines=max(1, int(args.agent_read_max_lines)),
        agent_loop_exhaustion_policy=agent_loop_exhaustion_policy,
        agent_max_repeated_write_overwrites=max(
            1, int(args.agent_max_repeated_write_overwrites)
        ),
        agent_persist_reasoning_content=agent_persist_reasoning_content,
        agent_allow_model_fallback=agent_allow_model_fallback,
    )
    temperature_label = (
        f"{settings.temperature:.2f}"
        if settings.temperature is not None
        else "ignored(reasoner)"
    )
    logger.info(
        "DeepSeek settings: model=%s max_tokens=%d timeout=%.1fs http_retries=%d temperature=%s endpoint_mode=auto_beta tool_loop=%s tool_mode=%s max_tool_rounds=%d read_max_lines=%d exhaustion_policy=%s max_repeated_overwrites=%d persist_reasoning=%s allow_model_fallback=%s budget_failure_policy=%s",
        settings.model,
        settings.max_completion_tokens,
        settings.request_timeout_seconds,
        settings.http_retries,
        temperature_label,
        settings.agent_tool_loop,
        settings.agent_tool_mode,
        settings.agent_max_tool_rounds,
        settings.agent_read_max_lines,
        settings.agent_loop_exhaustion_policy,
        settings.agent_max_repeated_write_overwrites,
        settings.agent_persist_reasoning_content,
        settings.agent_allow_model_fallback,
        budget_failure_policy,
    )
    logger.info(
        "Token budget policy: model_ceiling=%d requested_max_tokens=%d configured_max_tokens=%d effective_max_tokens=%d dynamic_cap_applied=%s word_budget=%s",
        model_token_ceiling,
        requested_max_tokens,
        configured_max_tokens,
        settings.max_completion_tokens,
        False,
        args.word_budget,
    )

    logger.info("Loading transcript from %s", args.transcript)
    segments = load_transcript_segments(args.transcript)
    if not segments:
        logger.error("No transcript segments found in %s", args.transcript)
        return 1

    allowed_speakers = collect_allowed_speakers(segments)
    if not allowed_speakers:
        logger.error("No speaker labels found in transcript.")
        return 1

    segment_speaker_map = build_segment_speaker_map(segments)
    transcript_text = format_segments_for_prompt(segments)
    logger.info(
        "Loaded %d transcript segments across %d speakers.",
        len(segments),
        len(allowed_speakers),
    )

    source_word_count = _count_words_from_segments(segments)
    if args.word_budget is not None:
        lower, upper = _budget_bounds(args.word_budget, args.word_budget_tolerance)
        logger.info(
            "Word budget target: metric=dialogue_text_words_only target=%d range=[%d, %d] source_words=%d",
            args.word_budget,
            lower,
            upper,
            source_word_count,
        )

    max_attempts = max(1, args.max_retries + 1)
    logger.info(
        "Generation retry policy: max_retries=%d max_attempts=%d http_retries_per_request=%d",
        args.max_retries,
        max_attempts,
        settings.http_retries,
    )
    validated_lines: list[ValidatedDialogueLine] | None = None
    retry_context: RetryContext | None = None
    best_candidate: CandidateSelection | None = None
    selected_endpoint_mode = "unknown"
    selected_model = settings.model
    selected_tool_rounds = 0
    selected_total_words: int | None = None
    selected_lower_bound: int | None = None
    selected_upper_bound: int | None = None
    selected_tool_calls_by_name: dict[str, int] = {}
    selected_tool_loop_exhausted = False
    selected_tool_loop_exhaustion_reason = ""
    selected_tool_round_limit = settings.agent_max_tool_rounds
    selected_repeated_overwrite_count = 0
    selected_staged_output_present = False
    selected_staged_output_valid_json = False
    selected_last_validation_issues: list[str] = []
    selected_model_path = "unknown"
    selected_naturalness_metrics: dict[str, float | int] = {
        "avg_words_per_line": 0.0,
        "short_question_ratio": 0.0,
        "disfluency_count": 0,
        "line_count": 0,
    }
    selected_issues: list[str] = []
    attempts_executed = 0
    conversation_state = ConversationState(messages=[])
    budget_status: Literal["in_range", "out_of_range", "not_configured", "unknown"] = (
        "unknown"
    )
    validation_status: Literal["valid", "invalid", "unknown"] = "unknown"
    chat_log_writer = DeepSeekChatLogWriter.create(args.deepseek_chat_log_root)
    logger.info(
        "DeepSeek chat tracing enabled: run_directory=%s",
        chat_log_writer.run_directory,
    )

    def _emit_and_record_deepseek_stream_event(payload: dict[str, object]) -> None:
        """Forward stream events to UI marker output and persistent event logs."""
        _emit_deepseek_stream_event(payload)
        if _should_persist_stream_event(payload):
            chat_log_writer.write_global_event({"event": "stream_event", "payload": payload})

    for attempt in range(1, max_attempts + 1):
        attempts_executed = attempt
        logger.info("Summarization attempt %d/%d.", attempt, max_attempts)
        try:
            generation_result = generate_summary_deepseek(
                transcript_text=transcript_text,
                transcript_segments=segments,
                api_key=args.api_key,
                allowed_speakers=allowed_speakers,
                segment_speaker_map=segment_speaker_map,
                source_word_count=source_word_count,
                settings=settings,
                segment_count=len(segments),
                retry_context=retry_context,
                word_budget=args.word_budget,
                target_minutes=args.target_minutes,
                avg_wpm=args.avg_wpm,
                word_budget_tolerance=args.word_budget_tolerance,
                tool_output_path=args.output,
                stream_event_callback=_emit_and_record_deepseek_stream_event,
                conversation_state=conversation_state,
                chat_log_writer=chat_log_writer,
            )
            structured_script = generation_result.script
        except Exception as exc:  # noqa: BLE001
            failure_endpoint: EndpointMode = "stable"
            root_error: Exception = exc
            if isinstance(exc, GenerationAttemptError):
                failure_endpoint = exc.endpoint_mode
                root_error = exc.cause

            classified_error = _classify_generation_error(root_error)
            retry_context = RetryContext(
                attempt_index=attempt,
                endpoint_mode=failure_endpoint,
                error_type=classified_error,
                error_digest=_error_digest(root_error),
                continuation=(
                    root_error.continuation_state
                    if isinstance(root_error, ToolLoopExhaustedError)
                    else None
                ),
            )
            if isinstance(root_error, ToolLoopExhaustedError):
                diagnostics = root_error.diagnostics
                selected_tool_loop_exhausted = diagnostics["tool_loop_exhausted"]
                selected_tool_loop_exhaustion_reason = diagnostics[
                    "tool_loop_exhaustion_reason"
                ]
                selected_tool_round_limit = diagnostics["tool_round_limit"]
                selected_repeated_overwrite_count = diagnostics[
                    "repeated_overwrite_count"
                ]
                selected_staged_output_present = diagnostics["staged_output_present"]
                selected_staged_output_valid_json = diagnostics[
                    "staged_output_valid_json"
                ]
                selected_last_validation_issues = diagnostics[
                    "last_validation_issues"
                ][:]
            selected_issues = [retry_context.error_digest]
            if selected_last_validation_issues:
                selected_issues.extend(selected_last_validation_issues[:3])
            if attempt < max_attempts:
                logger.warning(
                    "Generation failed on attempt %d: type=%s detail=%s",
                    attempt,
                    classified_error,
                    retry_context.error_digest,
                )
                sleep_seconds = min(8.0, float(2 ** (attempt - 1)))
                logger.info("Retrying after %.1f seconds backoff.", sleep_seconds)
                time.sleep(sleep_seconds)
                continue
            logger.error(
                "Generation failed after %d attempts: type=%s detail=%s",
                max_attempts,
                classified_error,
                retry_context.error_digest,
            )
            break

        payload_lines: list[DialogueLinePayload] = [
            cast(DialogueLinePayload, dialogue_line.model_dump())
            for dialogue_line in structured_script.dialogue
        ]
        validation_report = validate_and_repair_dialogue(
            payload_lines,
            allowed_speakers=allowed_speakers,
            segment_speaker_map=segment_speaker_map,
        )

        logger.info(
            "Validation result: total_lines=%d repaired_lines=%d issues=%d endpoint=%s json_repair=%s",
            len(validation_report.lines),
            validation_report.repaired_lines,
            len(validation_report.issues),
            generation_result.endpoint_mode,
            generation_result.used_json_repair,
        )
        if validation_report.is_valid:
            validation_status = "valid"
            selected_endpoint_mode = generation_result.endpoint_mode
            selected_model = generation_result.model
            selected_model_path = (
                f"{generation_result.endpoint_mode}:{generation_result.model}"
            )
            selected_tool_rounds = generation_result.tool_rounds
            selected_tool_calls_by_name = generation_result.tool_call_counts or {}
            diagnostics = generation_result.tool_loop_diagnostics
            if diagnostics is not None:
                selected_tool_loop_exhausted = diagnostics["tool_loop_exhausted"]
                selected_tool_loop_exhaustion_reason = diagnostics[
                    "tool_loop_exhaustion_reason"
                ]
                selected_tool_round_limit = diagnostics["tool_round_limit"]
                selected_repeated_overwrite_count = diagnostics[
                    "repeated_overwrite_count"
                ]
                selected_staged_output_present = diagnostics["staged_output_present"]
                selected_staged_output_valid_json = diagnostics[
                    "staged_output_valid_json"
                ]
                selected_last_validation_issues = diagnostics[
                    "last_validation_issues"
                ][:]
            else:
                selected_tool_loop_exhausted = False
                selected_tool_loop_exhaustion_reason = ""
                selected_tool_round_limit = settings.agent_max_tool_rounds
                selected_repeated_overwrite_count = 0
                selected_staged_output_present = False
                selected_staged_output_valid_json = False
                selected_last_validation_issues = []
            if args.word_budget is not None:
                total_words = _count_words(validation_report.lines)
                lower, upper = _budget_bounds(
                    args.word_budget, args.word_budget_tolerance
                )
                logger.info(
                    "Word budget check: metric=dialogue_text_words_only total=%d target=%d range=[%d, %d]",
                    total_words,
                    args.word_budget,
                    lower,
                    upper,
                )
                selected_total_words = total_words
                selected_lower_bound = lower
                selected_upper_bound = upper
                if total_words < lower:
                    if source_word_count > 0 and source_word_count < lower:
                        logger.warning(
                            "Transcript shorter than target budget: source=%d lower_bound=%d. "
                            "Accepting shorter summary.",
                            source_word_count,
                            lower,
                        )
                        validated_lines = validation_report.lines
                        budget_status = "in_range"
                        selected_issues = []
                        selected_naturalness_metrics = (
                            _naturalness_metrics_from_validated_lines(
                                validation_report.lines
                            )
                        )
                        break
                    logger.warning(
                        "Word budget out of range: total=%d target=%d range=[%d, %d]",
                        total_words,
                        args.word_budget,
                        lower,
                        upper,
                    )
                    budget_status = "out_of_range"
                    selected_issues = [
                        _build_word_budget_retry_digest(
                            total_words=total_words,
                            target_words=args.word_budget,
                            lower_bound=lower,
                            upper_bound=upper,
                        )
                    ]
                    delta_to_target = abs(total_words - args.word_budget)
                    candidate = CandidateSelection(
                        lines=validation_report.lines,
                        total_words=total_words,
                        delta_to_target=delta_to_target,
                        attempt_index=attempt,
                        endpoint_mode=generation_result.endpoint_mode,
                        model=generation_result.model,
                        tool_rounds=generation_result.tool_rounds,
                    )
                    if (
                        best_candidate is None
                        or best_candidate.delta_to_target is None
                        or (
                            candidate.delta_to_target is not None
                            and candidate.delta_to_target
                            < best_candidate.delta_to_target
                        )
                    ):
                        best_candidate = candidate
                    retry_context = RetryContext(
                        attempt_index=attempt,
                        endpoint_mode=generation_result.endpoint_mode,
                        error_type="schema_validation",
                        error_digest=_build_word_budget_retry_digest(
                            total_words=total_words,
                            target_words=args.word_budget,
                            lower_bound=lower,
                            upper_bound=upper,
                        ),
                    )
                    if attempt < max_attempts:
                        logger.warning("Retrying generation to satisfy word budget.")
                        sleep_seconds = min(8.0, float(2 ** (attempt - 1)))
                        logger.info(
                            "Retrying after %.1f seconds backoff.", sleep_seconds
                        )
                        time.sleep(sleep_seconds)
                        continue
                    logger.error("Word budget not met after %d attempts.", max_attempts)
                    break
                if total_words > upper:
                    logger.warning(
                        "Word budget out of range: total=%d target=%d range=[%d, %d]",
                        total_words,
                        args.word_budget,
                        lower,
                        upper,
                    )
                    budget_status = "out_of_range"
                    selected_issues = [
                        _build_word_budget_retry_digest(
                            total_words=total_words,
                            target_words=args.word_budget,
                            lower_bound=lower,
                            upper_bound=upper,
                        )
                    ]
                    delta_to_target = abs(total_words - args.word_budget)
                    candidate = CandidateSelection(
                        lines=validation_report.lines,
                        total_words=total_words,
                        delta_to_target=delta_to_target,
                        attempt_index=attempt,
                        endpoint_mode=generation_result.endpoint_mode,
                        model=generation_result.model,
                        tool_rounds=generation_result.tool_rounds,
                    )
                    if (
                        best_candidate is None
                        or best_candidate.delta_to_target is None
                        or (
                            candidate.delta_to_target is not None
                            and candidate.delta_to_target
                            < best_candidate.delta_to_target
                        )
                    ):
                        best_candidate = candidate
                    retry_context = RetryContext(
                        attempt_index=attempt,
                        endpoint_mode=generation_result.endpoint_mode,
                        error_type="schema_validation",
                        error_digest=_build_word_budget_retry_digest(
                            total_words=total_words,
                            target_words=args.word_budget,
                            lower_bound=lower,
                            upper_bound=upper,
                        ),
                    )
                    if attempt < max_attempts:
                        logger.warning("Retrying generation to satisfy word budget.")
                        sleep_seconds = min(8.0, float(2 ** (attempt - 1)))
                        logger.info(
                            "Retrying after %.1f seconds backoff.", sleep_seconds
                        )
                        time.sleep(sleep_seconds)
                        continue
                    logger.error("Word budget not met after %d attempts.", max_attempts)
                    break
                budget_status = "in_range"
                selected_issues = []
            validated_lines = validation_report.lines
            if args.word_budget is None:
                budget_status = "not_configured"
                selected_total_words = _count_words(validation_report.lines)
                selected_lower_bound = None
                selected_upper_bound = None
            selected_naturalness_metrics = _naturalness_metrics_from_validated_lines(
                validation_report.lines
            )
            break

        for issue in validation_report.issues:
            logger.error("Validation issue: %s", issue)
        validation_status = "invalid"
        selected_issues = validation_report.issues[:]

        retry_context = RetryContext(
            attempt_index=attempt,
            endpoint_mode=generation_result.endpoint_mode,
            error_type="schema_validation",
            error_digest=(
                f"validation issues={len(validation_report.issues)}; "
                f"first={validation_report.issues[0] if validation_report.issues else 'none'}"
            ),
        )
        if attempt < max_attempts:
            logger.warning(
                "Validation failed on attempt %d; retrying generation with corrective feedback.",
                attempt,
            )
            sleep_seconds = min(8.0, float(2 ** (attempt - 1)))
            logger.info("Retrying after %.1f seconds backoff.", sleep_seconds)
            time.sleep(sleep_seconds)
            continue

        logger.error("Validation failed after %d attempts.", max_attempts)
        break

    degraded_success = False
    if (
        validated_lines is None
        and budget_failure_policy == "degraded_success"
        and best_candidate is not None
    ):
        validated_lines = best_candidate.lines
        selected_total_words = best_candidate.total_words
        selected_endpoint_mode = best_candidate.endpoint_mode
        selected_model = best_candidate.model
        selected_model_path = f"{best_candidate.endpoint_mode}:{best_candidate.model}"
        selected_tool_rounds = best_candidate.tool_rounds
        degraded_success = True
        validation_status = "valid"
        budget_status = "out_of_range"
        selected_naturalness_metrics = _naturalness_metrics_from_validated_lines(
            best_candidate.lines
        )
        logger.warning(
            "Using degraded-success candidate from attempt=%d endpoint=%s model=%s total_words=%s",
            best_candidate.attempt_index,
            best_candidate.endpoint_mode,
            best_candidate.model,
            best_candidate.total_words,
        )

    if validated_lines is None:
        report: SummaryReport = {
            "summary_outcome": "failure",
            "budget_status": budget_status,
            "validation_status": validation_status,
            "degraded_success": False,
            "attempts": attempts_executed,
            "max_attempts": max_attempts,
            "endpoint_mode": selected_endpoint_mode,
            "model": selected_model,
            "tool_rounds": selected_tool_rounds,
            "total_words": selected_total_words,
            "target_words": args.word_budget,
            "lower_bound": selected_lower_bound,
            "upper_bound": selected_upper_bound,
            "tool_calls_by_name": selected_tool_calls_by_name,
            "tool_loop_exhausted": selected_tool_loop_exhausted,
            "tool_loop_exhaustion_reason": selected_tool_loop_exhaustion_reason,
            "tool_round_limit": selected_tool_round_limit,
            "repeated_overwrite_count": selected_repeated_overwrite_count,
            "staged_output_present": selected_staged_output_present,
            "staged_output_valid_json": selected_staged_output_valid_json,
            "last_validation_issues": selected_last_validation_issues,
            "model_path_selected": selected_model_path,
            "naturalness_metrics": selected_naturalness_metrics,
            "issues": selected_issues,
            "output_path": args.output,
        }
        _write_summary_report(summary_report_path, report)
        logger.error(
            "No validated dialogue was produced. summary_outcome=failure report=%s",
            summary_report_path,
        )
        chat_log_writer.close()
        return 1

    final_json = post_process_script(validated_lines, args.voice_dir)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(final_json, output_file, indent=2)
    if settings.agent_tool_loop or _is_reasoner_model(selected_model):
        _emit_and_record_deepseek_stream_event(
            {"event": "summary_json_ready", "path": args.output}
        )

    summary_outcome: Literal["success", "degraded_success", "failure"] = (
        "degraded_success" if degraded_success else "success"
    )
    report = SummaryReport(
        summary_outcome=summary_outcome,
        budget_status=budget_status,
        validation_status=validation_status,
        degraded_success=degraded_success,
        attempts=attempts_executed,
        max_attempts=max_attempts,
        endpoint_mode=selected_endpoint_mode,
        model=selected_model,
        tool_rounds=selected_tool_rounds,
        total_words=selected_total_words,
        target_words=args.word_budget,
        lower_bound=selected_lower_bound,
        upper_bound=selected_upper_bound,
        tool_calls_by_name=selected_tool_calls_by_name,
        tool_loop_exhausted=selected_tool_loop_exhausted,
        tool_loop_exhaustion_reason=selected_tool_loop_exhaustion_reason,
        tool_round_limit=selected_tool_round_limit,
        repeated_overwrite_count=selected_repeated_overwrite_count,
        staged_output_present=selected_staged_output_present,
        staged_output_valid_json=selected_staged_output_valid_json,
        last_validation_issues=selected_last_validation_issues,
        model_path_selected=selected_model_path,
        naturalness_metrics=selected_naturalness_metrics,
        issues=selected_issues,
        output_path=args.output,
    )
    _write_summary_report(summary_report_path, report)

    if degraded_success:
        logger.warning(
            "Summary saved with degraded success: output=%s report=%s",
            args.output,
            summary_report_path,
        )
    else:
        logger.info(
            "Success: summarized script saved to %s (report=%s)",
            args.output,
            summary_report_path,
        )
    chat_log_writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
