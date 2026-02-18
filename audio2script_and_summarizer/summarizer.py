"""CARD Script Summarizer with transcript-grounded speaker validation."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Sequence, TypedDict, cast

from openai import OpenAI
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

MODEL_NAME = "gpt-4o-2024-08-06"
MAX_RETRIES_DEFAULT = 2
DEFAULT_MAX_COMPLETION_TOKENS = 8192
ERROR_DIGEST_MAX_CHARS = 280
TOKEN_BUDGET_WORD_FACTOR = 1.8
TOKEN_BUDGET_SAFETY_BUFFER = 256
TOKEN_BUDGET_MIN = 512

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

ErrorType = Literal[
    "api_error",
    "schema_validation",
    "empty_response",
    "truncated_output",
]


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
class RetryContext:
    """Represent context from the previous failed generation attempt."""

    attempt_index: int
    error_type: ErrorType
    error_digest: str


class OutputTruncatedError(ValueError):
    """Represent model output truncation due to output token limits."""


def _error_digest(error: Exception) -> str:
    """Create a compact, single-line error digest for logs and retry prompts."""
    digest = " ".join(str(error).split())
    return digest[:ERROR_DIGEST_MAX_CHARS]


def _build_system_prompt(
    allowed_speakers: set[str],
    word_budget: int | None,
    target_minutes: float | None,
    avg_wpm: float | None,
    word_budget_tolerance: float,
) -> str:
    """Build a constrained prompt with explicit objective and acceptance checks.

    Args:
        allowed_speakers: Speaker labels allowed in the response.

    Returns:
        Prompt string for the LLM.
    """
    speaker_list = ", ".join(sorted(allowed_speakers))
    sample_speaker = sorted(allowed_speakers)[0]
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

    return f"""Objective:
- Rewrite the transcript into concise, engaging podcast dialogue while preserving factual meaning.

Context:
- This is for CARD (Constraint-aware Audio Resynthesis).
- Transcript lines include evidence metadata in the form [segment_id|speaker]: text.
- Allowed speakers are exactly: {speaker_list}

Inputs:
- A transcript where each line has:
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
- Each line must include non-empty source_segment_ids.
- Keep dialogue lines around 10-15 seconds of speech.
{budget_clause}

Rules:
- Never invent, rename, or merge speaker labels.
- speaker must be exactly one of the allowed speakers.
- Every source_segment_id must come from the input transcript.
- All source_segment_ids for a single line must belong to the same speaker.
- Avoid meta-summary phrases like "the speaker discusses".
- Preserve key claims and tone.

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
"""


def _build_retry_instruction(context: RetryContext) -> str:
    """Build retry guidance that explains the previous failure."""
    return (
        "Retry correction request:\n"
        f"- Previous attempt: {context.attempt_index}\n"
        f"- Failure type: {context.error_type}\n"
        f"- Failure detail: {context.error_digest}\n"
        "- Fix the exact issue and regenerate a complete schema-valid response.\n"
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


def generate_summary(
    transcript_text: str,
    api_key: str,
    allowed_speakers: set[str],
    retry_context: RetryContext | None,
    word_budget: int | None,
    target_minutes: float | None,
    avg_wpm: float | None,
    word_budget_tolerance: float,
    max_completion_tokens: int,
) -> PodcastScript:
    """Generate a structured summary from the transcript using OpenAI.

    Args:
        transcript_text: Prompt-ready transcript with segment IDs.
        api_key: OpenAI API key.
        allowed_speakers: Speaker labels allowed in output.
        retry_context: Previous-attempt failure context.
        word_budget: Target summary word budget.
        target_minutes: Target summary duration in minutes.
        avg_wpm: Average words-per-minute calibration.
        word_budget_tolerance: Inclusive word-budget tolerance ratio.
        max_completion_tokens: Token cap for completion output.

    Returns:
        Parsed structured script from the LLM.

    Raises:
        ValueError: If the provider response is empty.
        OutputTruncatedError: If completion is truncated by token limit.
    """
    client = OpenAI(api_key=api_key)
    system_prompt = _build_system_prompt(
        allowed_speakers=allowed_speakers,
        word_budget=word_budget,
        target_minutes=target_minutes,
        avg_wpm=avg_wpm,
        word_budget_tolerance=word_budget_tolerance,
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    if retry_context is not None:
        logger.info(
            "Applying retry context: attempt=%d type=%s detail=%s",
            retry_context.attempt_index,
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
            "content": f"Here is the raw transcript:\n\n{transcript_text}",
        }
    )

    logger.info(
        "Sending transcript to OpenAI for summarization model=%s max_completion_tokens=%d.",
        MODEL_NAME,
        max_completion_tokens,
    )
    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=cast(Any, messages),
        response_format=PodcastScript,
        max_completion_tokens=max_completion_tokens,
    )

    if not completion.choices:
        raise ValueError("OpenAI returned empty choices.")

    choice = completion.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    if finish_reason == "length":
        raise OutputTruncatedError(
            "OpenAI response truncated by max_completion_tokens (finish_reason=length)."
        )

    parsed = choice.message.parsed
    if parsed is None:
        raise ValueError("OpenAI returned no parsed content.")
    return cast(PodcastScript, parsed)


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
                "Voice sample not found for speaker '%s': %s", line.speaker, voice_path
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


def _count_words_from_segments(
    segments: Sequence[dict[str, Any] | TranscriptSegment],
) -> int:
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
    """Classify generation/parsing exceptions for retry instructions."""
    if isinstance(error, OutputTruncatedError):
        return "truncated_output"
    if isinstance(error, ValidationError):
        return "schema_validation"
    if isinstance(error, ValueError) and "empty response" in str(error).lower():
        return "empty_response"
    return "api_error"


def main() -> int:
    """Run the OpenAI summarizer with strict transcript-grounded speaker validation."""
    parser = argparse.ArgumentParser(
        description="CARD Script Summarizer & Emotion Annotator"
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
        "--output", default="summarized_script.json", help="Path to save output JSON"
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API Key"
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
        "--max-retries",
        type=int,
        default=MAX_RETRIES_DEFAULT,
        help=f"Max retry attempts for LLM generation and validation (default: {MAX_RETRIES_DEFAULT})",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=DEFAULT_MAX_COMPLETION_TOKENS,
        help=(
            "Maximum output token budget for OpenAI responses; "
            f"default: {DEFAULT_MAX_COMPLETION_TOKENS}"
        ),
    )
    args = parser.parse_args()
    configure_logging(
        level=os.getenv("AUDIO2SCRIPT_LOG_LEVEL", "INFO"),
        component="summarizer_openai",
    )

    if not args.api_key:
        logger.error("No API key provided. Set OPENAI_API_KEY or pass --api-key.")
        return 1

    configured_max_tokens = max(TOKEN_BUDGET_MIN, int(args.max_completion_tokens))
    effective_max_tokens = _derive_completion_token_cap(
        word_budget=args.word_budget,
        configured_max_tokens=configured_max_tokens,
    )
    logger.info(
        "OpenAI settings: model=%s configured_max_tokens=%d effective_max_tokens=%d dynamic_cap_applied=%s",
        MODEL_NAME,
        configured_max_tokens,
        effective_max_tokens,
        args.word_budget is not None and effective_max_tokens < configured_max_tokens,
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
    validated_lines: list[ValidatedDialogueLine] | None = None
    retry_context: RetryContext | None = None

    for attempt in range(1, max_attempts + 1):
        logger.info("Summarization attempt %d/%d.", attempt, max_attempts)
        try:
            structured_script = generate_summary(
                transcript_text=transcript_text,
                api_key=args.api_key,
                allowed_speakers=allowed_speakers,
                retry_context=retry_context,
                word_budget=args.word_budget,
                target_minutes=args.target_minutes,
                avg_wpm=args.avg_wpm,
                word_budget_tolerance=args.word_budget_tolerance,
                max_completion_tokens=effective_max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            retry_context = RetryContext(
                attempt_index=attempt,
                error_type=_classify_generation_error(exc),
                error_digest=_error_digest(exc),
            )
            if attempt < max_attempts:
                logger.warning(
                    "Generation failed on attempt %d: type=%s detail=%s",
                    attempt,
                    retry_context.error_type,
                    retry_context.error_digest,
                )
                sleep_seconds = min(8.0, float(2 ** (attempt - 1)))
                logger.info("Retrying after %.1f seconds backoff.", sleep_seconds)
                time.sleep(sleep_seconds)
                continue
            logger.error(
                "Generation failed after %d attempts: type=%s detail=%s",
                max_attempts,
                retry_context.error_type,
                retry_context.error_digest,
            )
            return 1

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
            "Validation result: total_lines=%d repaired_lines=%d issues=%d",
            len(validation_report.lines),
            validation_report.repaired_lines,
            len(validation_report.issues),
        )
        if validation_report.is_valid:
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
                if total_words < lower:
                    if source_word_count > 0 and source_word_count < lower:
                        logger.warning(
                            "Transcript shorter than target budget: source=%d lower_bound=%d. "
                            "Accepting shorter summary.",
                            source_word_count,
                            lower,
                        )
                        validated_lines = validation_report.lines
                        break
                    logger.warning(
                        "Word budget out of range: total=%d target=%d range=[%d, %d]",
                        total_words,
                        args.word_budget,
                        lower,
                        upper,
                    )
                    retry_context = RetryContext(
                        attempt_index=attempt,
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
                    return 1
                if total_words > upper:
                    logger.warning(
                        "Word budget out of range: total=%d target=%d range=[%d, %d]",
                        total_words,
                        args.word_budget,
                        lower,
                        upper,
                    )
                    retry_context = RetryContext(
                        attempt_index=attempt,
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
                    return 1
            validated_lines = validation_report.lines
            break

        for issue in validation_report.issues:
            logger.error("Validation issue: %s", issue)

        retry_context = RetryContext(
            attempt_index=attempt,
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
        return 1

    if validated_lines is None:
        logger.error("No validated dialogue was produced.")
        return 1

    final_json = post_process_script(validated_lines, args.voice_dir)
    with open(args.output, "w", encoding="utf-8") as output_file:
        json.dump(final_json, output_file, indent=2)

    logger.info("Success: summarized script saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
