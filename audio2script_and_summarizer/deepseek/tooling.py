"""DeepSeek local tool handlers and tool-loop diagnostics helpers."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import replace
from json import JSONDecodeError
from pathlib import Path
from typing import Callable, Literal, cast

from pydantic import ValidationError

from ..speaker_validation import (
    DialogueLinePayload,
    TranscriptSegment,
    validate_and_repair_dialogue,
)
from .constants import (
    AgentToolMode,
    COUNT_WORDS_TOOL_NAME,
    DISFLUENCY_PATTERN,
    EVALUATE_SCRIPT_TOOL_NAME,
    READ_TRANSCRIPT_LINES_TOOL_NAME,
    SUMMARY_LINE_WORD_FLOOR,
    SUMMARY_MAX_SHORT_QUESTION_RATIO,
    WRITE_OUTPUT_SEGMENT_TOOL_NAME,
)
from .models import (
    CountWordsToolResult,
    DeepSeekRequestSettings,
    DialogueLine,
    NaturalnessEvaluation,
    PodcastScript,
    ReadTranscriptToolResult,
    RetryContinuationState,
    ToolConstraintEvaluation,
    ToolDialogueLine,
    ToolLoopDiagnostics,
    ToolLoopExhaustedError,
    TranscriptToolLine,
    WriteOutputSegmentToolResult,
)
from .parsing import _decode_podcast_script_with_fallback
from .runtime_helpers import _emit_stream_token_event, _error_digest

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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

def _budget_bounds(word_budget: int, tolerance: float) -> tuple[int, int]:
    """Compute inclusive word-budget bounds."""
    if word_budget <= 0:
        return 0, 0
    lower = int(word_budget * (1.0 - tolerance))
    upper = int(word_budget * (1.0 + tolerance))
    return max(1, lower), max(1, upper)
