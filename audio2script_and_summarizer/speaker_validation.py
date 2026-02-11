"""Transcript-grounded speaker validation and deterministic repair helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypedDict

ValidationStatus = Literal["valid", "repaired"]


@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    """Represent one diarized transcript segment.

    Attributes:
        segment_id: Stable segment identifier used for provenance.
        speaker: Speaker label assigned by diarization.
        text: Segment text.
        start_time: Optional segment start timestamp.
        end_time: Optional segment end timestamp.
    """

    segment_id: str
    speaker: str
    text: str
    start_time: float | None
    end_time: float | None


class DialogueLinePayload(TypedDict):
    """Model-generated line payload before post-processing."""

    speaker: str
    text: str
    emo_text: str
    emo_alpha: float
    source_segment_ids: list[str]


@dataclass(slots=True, frozen=True)
class ValidatedDialogueLine:
    """Represent a validated dialogue line ready for output."""

    speaker: str
    text: str
    emo_text: str
    emo_alpha: float
    source_segment_ids: list[str]
    validation_status: ValidationStatus
    repair_reason: str | None


@dataclass(slots=True)
class ValidationReport:
    """Contain validation and repair results for generated dialogue."""

    lines: list[ValidatedDialogueLine]
    issues: list[str]
    repaired_lines: int

    @property
    def is_valid(self) -> bool:
        """Return ``True`` when no unrecoverable validation issues exist."""
        return not self.issues


def _parse_input_payload(input_data: str) -> Any:
    """Parse transcript input from JSON string or file path."""
    stripped = input_data.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return json.loads(input_data)
    with Path(input_data).open("r", encoding="utf-8") as transcript_file:
        return json.load(transcript_file)


def _extract_segments(payload: Any) -> list[dict[str, Any]]:
    """Extract raw segment dictionaries from transcript payload."""
    if isinstance(payload, list):
        return [segment for segment in payload if isinstance(segment, dict)]
    if isinstance(payload, dict):
        segments = payload.get("segments", [])
        if isinstance(segments, list):
            return [segment for segment in segments if isinstance(segment, dict)]
    return []


def _coerce_optional_float(raw_value: Any) -> float | None:
    """Convert values into optional floats without raising parsing errors."""
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def load_transcript_segments(input_data: str) -> list[TranscriptSegment]:
    """Load transcript segments from JSON file path or JSON string.

    Args:
        input_data: JSON text or transcript JSON file path.

    Returns:
        Normalized transcript segments with deterministic segment IDs.
    """
    payload = _parse_input_payload(input_data)
    raw_segments = _extract_segments(payload)

    segments: list[TranscriptSegment] = []
    seen_ids: set[str] = set()

    for idx, raw_segment in enumerate(raw_segments):
        raw_id = raw_segment.get("id") or raw_segment.get("segment_id")
        candidate_id = str(raw_id).strip() if raw_id is not None else ""
        if not candidate_id:
            candidate_id = f"seg_{idx:05d}"
        if candidate_id in seen_ids:
            candidate_id = f"{candidate_id}_{idx:05d}"
        seen_ids.add(candidate_id)

        raw_speaker = str(raw_segment.get("speaker", "Unknown")).strip()
        speaker = raw_speaker if raw_speaker else "Unknown"
        text = str(raw_segment.get("text", "")).strip()

        segments.append(
            TranscriptSegment(
                segment_id=candidate_id,
                speaker=speaker,
                text=text,
                start_time=_coerce_optional_float(raw_segment.get("start_time")),
                end_time=_coerce_optional_float(raw_segment.get("end_time")),
            )
        )
    return segments


def format_segments_for_prompt(segments: list[TranscriptSegment]) -> str:
    """Format transcript segments for model prompts with provenance IDs.

    Args:
        segments: Normalized transcript segments.

    Returns:
        Prompt-ready transcript text.
    """
    return "\n".join(
        f"[{segment.segment_id}|{segment.speaker}]: {segment.text}" for segment in segments
    )


def build_segment_speaker_map(segments: list[TranscriptSegment]) -> dict[str, str]:
    """Build a segment-to-speaker lookup map.

    Args:
        segments: Normalized transcript segments.

    Returns:
        Dictionary mapping segment IDs to speakers.
    """
    return {segment.segment_id: segment.speaker for segment in segments}


def collect_allowed_speakers(segments: list[TranscriptSegment]) -> set[str]:
    """Collect all allowed speaker labels from transcript segments.

    Args:
        segments: Normalized transcript segments.

    Returns:
        Set of speaker labels.
    """
    return {segment.speaker for segment in segments if segment.speaker}


def _canonical_speaker_for_ids(
    source_ids: list[str], segment_speaker_map: dict[str, str]
) -> tuple[str | None, str | None]:
    """Resolve the unique canonical speaker for cited segment IDs.

    Args:
        source_ids: Segment IDs cited by a generated line.
        segment_speaker_map: Segment-to-speaker lookup.

    Returns:
        Tuple of canonical speaker and failure reason.
    """
    if not source_ids:
        return None, "source_segment_ids is empty"

    missing_ids = [segment_id for segment_id in source_ids if segment_id not in segment_speaker_map]
    if missing_ids:
        return None, f"unknown source_segment_ids: {missing_ids}"

    speakers = {segment_speaker_map[segment_id] for segment_id in source_ids}
    if len(speakers) != 1:
        return None, f"source_segment_ids span multiple speakers: {sorted(speakers)}"

    return next(iter(speakers)), None


def validate_and_repair_dialogue(
    lines: list[DialogueLinePayload],
    allowed_speakers: set[str],
    segment_speaker_map: dict[str, str],
) -> ValidationReport:
    """Validate generated lines and deterministically repair speaker labels.

    Args:
        lines: Model-generated dialogue lines.
        allowed_speakers: Allowed speaker labels derived from transcript.
        segment_speaker_map: Segment-to-speaker lookup.

    Returns:
        Validation report with repaired lines and unrecoverable issues.
    """
    validated_lines: list[ValidatedDialogueLine] = []
    issues: list[str] = []
    repaired_lines = 0

    for idx, line in enumerate(lines):
        source_ids = [str(source_id).strip() for source_id in line.get("source_segment_ids", []) if str(source_id).strip()]
        canonical_speaker, canonical_error = _canonical_speaker_for_ids(source_ids, segment_speaker_map)
        line_speaker = str(line.get("speaker", "")).strip()

        if canonical_error is not None:
            issues.append(f"line {idx}: {canonical_error}")
            continue

        if canonical_speaker is None:
            issues.append(f"line {idx}: unable to resolve canonical speaker")
            continue

        repaired_speaker = line_speaker
        status: ValidationStatus = "valid"
        repair_reason: str | None = None

        if line_speaker not in allowed_speakers:
            repaired_speaker = canonical_speaker
            status = "repaired"
            repair_reason = (
                f"speaker '{line_speaker}' is not allowed; "
                f"replaced with canonical speaker '{canonical_speaker}'"
            )
        elif line_speaker != canonical_speaker:
            repaired_speaker = canonical_speaker
            status = "repaired"
            repair_reason = (
                f"speaker '{line_speaker}' mismatched evidence; "
                f"replaced with canonical speaker '{canonical_speaker}'"
            )

        if status == "repaired":
            repaired_lines += 1

        validated_lines.append(
            ValidatedDialogueLine(
                speaker=repaired_speaker,
                text=str(line.get("text", "")),
                emo_text=str(line.get("emo_text", "")),
                emo_alpha=float(line.get("emo_alpha", 0.6)),
                source_segment_ids=source_ids,
                validation_status=status,
                repair_reason=repair_reason,
            )
        )

    return ValidationReport(lines=validated_lines, issues=issues, repaired_lines=repaired_lines)
