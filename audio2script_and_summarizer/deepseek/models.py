"""DeepSeek shared models, typed payloads, and exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict, cast

from pydantic import BaseModel, Field

from ..speaker_validation import ValidatedDialogueLine
from .constants import (
    AgentLoopExhaustionPolicy,
    AgentToolMode,
    DEFAULT_AGENT_ALLOW_MODEL_FALLBACK,
    DEFAULT_AGENT_LOOP_EXHAUSTION_POLICY,
    DEFAULT_AGENT_MAX_REPEATED_WRITE_OVERWRITES,
    DEFAULT_AGENT_MAX_TOOL_ROUNDS,
    DEFAULT_AGENT_PERSIST_REASONING_CONTENT,
    DEFAULT_AGENT_READ_MAX_LINES,
    DEFAULT_AGENT_TOOL_MODE,
    EndpointMode,
    ErrorType,
)

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
    target_duration_seconds: float | None
    measured_duration_seconds: float | None
    duration_delta_seconds: float | None
    duration_within_tolerance: bool | None
    duration_correction_passes: int

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
    agent_tool_mode: AgentToolMode = cast(AgentToolMode, DEFAULT_AGENT_TOOL_MODE)
    agent_max_tool_rounds: int = DEFAULT_AGENT_MAX_TOOL_ROUNDS
    agent_read_max_lines: int = DEFAULT_AGENT_READ_MAX_LINES
    agent_loop_exhaustion_policy: AgentLoopExhaustionPolicy = cast(
        AgentLoopExhaustionPolicy,
        DEFAULT_AGENT_LOOP_EXHAUSTION_POLICY,
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
