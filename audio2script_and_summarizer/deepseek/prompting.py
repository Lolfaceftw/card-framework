"""DeepSeek prompt and retry-instruction builders."""

from __future__ import annotations

from ..speaker_validation import TranscriptSegment
from .constants import (
    AgentToolMode,
    COUNT_WORDS_TOOL_NAME,
    EVALUATE_SCRIPT_TOOL_NAME,
    READ_TRANSCRIPT_LINES_TOOL_NAME,
    STAGED_OUTPUT_READY_MARKER,
    WRITE_OUTPUT_SEGMENT_TOOL_NAME,
)
from .models import RetryContext, RetryContinuationState

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
