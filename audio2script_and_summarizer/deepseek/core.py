"""CARD Script Summarizer with DeepSeek and transcript-grounded speaker validation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Callable, Literal, cast

from openai import OpenAI
from pydantic import ValidationError

from .chat_logs import DeepSeekChatLogWriter
from .runtime_helpers import (
    ContextUsageSnapshot,
    StreamedAssistantTurn,
    _StreamCallLogBuffer,
    _build_context_usage_snapshot,
    _emit_context_usage_event,
    _emit_deepseek_stream_event,
    _emit_stream_token_event,
    _error_digest,
    _extract_total_tokens_from_usage,
    _format_tool_event_details,
    _normalize_stream_text_piece,
    _should_persist_stream_event,
    _should_rollover_context,
    _stream_assistant_turn,
)
from ..logging_utils import configure_logging
from ..speaker_validation import (
    DialogueLinePayload,
    TranscriptSegment,
    ValidatedDialogueLine,
    build_segment_speaker_map,
    collect_allowed_speakers,
    format_segments_for_prompt,
    load_transcript_segments,
    validate_and_repair_dialogue,
)

from .client import (
    _build_deepseek_client,
    _clone_messages,
    _reset_conversation_with_summary,
    _summarize_conversation_context_via_deepseek,
)
from .models import (  # noqa: F401
    CandidateSelection,
    ConversationState,
    CountWordsToolResult,
    DeepSeekRequestSettings,
    DialogueLine,
    FinalScriptLine,
    GenerationAttemptError,
    GenerationSuccess,
    NaturalnessEvaluation,
    OutputTruncatedError,
    PodcastScript,
    ReadTranscriptToolResult,
    RetryContext,
    RetryContinuationState,
    SummaryReport,
    ToolConstraintEvaluation,
    ToolDialogueLine,
    ToolLoopDiagnostics,
    ToolLoopExhaustedError,
    TranscriptToolLine,
    ValidationEvaluation,
    WordBudgetEvaluation,
    WriteOutputSegmentToolResult,
)
from .output_helpers import (
    _count_words,
    _count_words_from_segments,
    _naturalness_metrics_from_validated_lines,
    _resolve_summary_report_path,
    _target_line_bounds,
    _write_summary_report,
    post_process_script,
)
from .parsing import (  # noqa: F401
    _classify_json_error,
    _classify_generation_error,
    _clamp_completion_tokens_for_model,
    _decode_podcast_script,
    _decode_podcast_script_with_fallback,
    _derive_completion_token_cap,
    _effective_agent_tool_mode,
    _extract_first_json_object,
    _is_reasoner_model,
    _looks_like_beta_endpoint_error,
    _looks_like_tool_protocol_error,
    _message_to_replay_dict,
    _model_completion_token_ceiling,
    _normalized_model_name,
    _normalize_assistant_content,
    _should_persist_reasoning_for_replay,
    _strip_markdown_fence,
)
from .prompting import (  # noqa: F401
    _build_initial_user_message,
    _build_rollover_continuation_message,
    _build_retry_instruction,
    _build_retry_resume_guard_message,
    _build_system_prompt,
    _build_transcript_manifest,
    _build_word_budget_retry_digest,
    _format_retry_read_ranges,
    _parse_on_off_flag,
)
from .tooling import (  # noqa: F401
    _build_constraint_hints,
    _build_tool_loop_exhaustion_error,
    _budget_bounds,
    _copy_tool_loop_diagnostics,
    _coerce_tool_dialogue_lines,
    _count_words_in_text,
    _count_words_from_payload_lines,
    _count_words_tool,
    _deepseek_tool_schemas,
    _evaluate_naturalness,
    _evaluate_script_constraints_tool,
    _hash_text,
    _handle_tool_loop_exhaustion,
    _merge_tool_loop_diagnostics,
    _new_tool_loop_diagnostics,
    _podcast_script_from_tool_dialogue,
    _read_staged_output_text,
    _read_transcript_lines_tool,
    _resolve_tool_output_staging_path,
    _tool_dialogue_payload_from_script,
    _write_output_segment_tool,
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
