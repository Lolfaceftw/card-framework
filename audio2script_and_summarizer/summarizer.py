"""CARD Script Summarizer with transcript-grounded speaker validation."""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import TypedDict, cast

from openai import OpenAI
from pydantic import BaseModel, Field

from .logging_utils import configure_logging
from .speaker_validation import (
    DialogueLinePayload,
    ValidatedDialogueLine,
    build_segment_speaker_map,
    collect_allowed_speakers,
    format_segments_for_prompt,
    load_transcript_segments,
    validate_and_repair_dialogue,
)

MODEL_NAME = "gpt-4o-2024-08-06"
MAX_RETRIES_DEFAULT = 2

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DialogueLine(BaseModel):
    """Represent one model-generated dialogue line."""

    speaker: str = Field(..., description="Speaker label from the allowed transcript speakers.")
    text: str = Field(..., description="Natural dialogue text, about 10-15 seconds of speech.")
    emo_text: str = Field(..., description="Short emotion/tone description.")
    emo_alpha: float = Field(0.6, description="Emotion intensity, typically 0.5 to 0.9.")
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


def _build_system_prompt(allowed_speakers: set[str]) -> str:
    """Build a constrained prompt with explicit objective and acceptance checks.

    Args:
        allowed_speakers: Speaker labels allowed in the response.

    Returns:
        Prompt string for the LLM.
    """
    speaker_list = ", ".join(sorted(allowed_speakers))
    sample_speaker = sorted(allowed_speakers)[0]
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


def generate_summary(transcript_text: str, api_key: str, allowed_speakers: set[str]) -> PodcastScript:
    """Generate a structured summary from the transcript using OpenAI.

    Args:
        transcript_text: Prompt-ready transcript with segment IDs.
        api_key: OpenAI API key.
        allowed_speakers: Speaker labels allowed in output.

    Returns:
        Parsed structured script from the LLM.
    """
    client = OpenAI(api_key=api_key)
    system_prompt = _build_system_prompt(allowed_speakers)

    logger.info("Sending transcript to OpenAI for summarization.")
    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here is the raw transcript:\n\n{transcript_text}"},
        ],
        response_format=PodcastScript,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("OpenAI returned no parsed content.")
    return parsed


def post_process_script(lines: list[ValidatedDialogueLine], voice_sample_dir: str) -> list[FinalScriptLine]:
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
            logger.warning("Voice sample not found for speaker '%s': %s", line.speaker, voice_path)

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


def main() -> int:
    """Run the OpenAI summarizer with strict transcript-grounded speaker validation."""
    parser = argparse.ArgumentParser(description="CARD Script Summarizer & Emotion Annotator")
    parser.add_argument("--transcript", required=True, help="Path to input WhisperX JSON transcript")
    parser.add_argument("--voice-dir", required=True, help="Directory where separated speaker audios are stored")
    parser.add_argument("--output", default="summarized_script.json", help="Path to save output JSON")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=MAX_RETRIES_DEFAULT,
        help=f"Max retry attempts for LLM generation and validation (default: {MAX_RETRIES_DEFAULT})",
    )
    args = parser.parse_args()
    configure_logging()

    if not args.api_key:
        logger.error("No API key provided. Set OPENAI_API_KEY or pass --api-key.")
        return 1

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
    logger.info("Loaded %d transcript segments across %d speakers.", len(segments), len(allowed_speakers))

    max_attempts = max(1, args.max_retries + 1)
    validated_lines: list[ValidatedDialogueLine] | None = None

    for attempt in range(1, max_attempts + 1):
        logger.info("Summarization attempt %d/%d.", attempt, max_attempts)
        try:
            structured_script = generate_summary(transcript_text, args.api_key, allowed_speakers)
        except Exception as exc:  # noqa: BLE001
            if attempt < max_attempts:
                logger.warning("Generation failed on attempt %d: %s", attempt, exc)
                continue
            logger.exception("Generation failed after %d attempts.", max_attempts)
            return 1

        payload_lines: list[DialogueLinePayload] = [
            cast(DialogueLinePayload, dialogue_line.model_dump()) for dialogue_line in structured_script.dialogue
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
            validated_lines = validation_report.lines
            break

        for issue in validation_report.issues:
            logger.error("Validation issue: %s", issue)

        if attempt < max_attempts:
            logger.warning("Validation failed on attempt %d; retrying generation.", attempt)
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
