"""Compute speaker WPM directly from diarized transcript timestamps."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    """Represent one validated diarized transcript segment.

    Attributes:
        speaker: Canonical speaker label (for example, ``SPEAKER_00``).
        start_ms: Segment start time in milliseconds.
        end_ms: Segment end time in milliseconds.
        text: Segment transcript text.
    """

    speaker: str
    start_ms: int
    end_ms: int
    text: str


def _infer_timestamp_scale(raw_segments: list[dict[str, Any]]) -> float:
    """Infer whether timestamps are expressed in seconds or milliseconds.

    Args:
        raw_segments: Unvalidated transcript segment payloads.

    Returns:
        ``1000.0`` for second-based timestamps, otherwise ``1.0``.
    """
    durations: list[float] = []
    max_end = 0.0
    for segment in raw_segments:
        start_raw = segment.get("start_time")
        end_raw = segment.get("end_time")
        if not isinstance(start_raw, (int, float)) or not isinstance(
            end_raw, (int, float)
        ):
            continue
        start = float(start_raw)
        end = float(end_raw)
        if end <= start:
            continue
        durations.append(end - start)
        max_end = max(max_end, end)

    if not durations:
        return 1.0

    median_duration = statistics.median(durations)

    # Timestamp domains used in this repository are typically either:
    # - seconds (segment durations commonly < 120)
    # - milliseconds (segment durations commonly > 120)
    if max_end >= 200_000:
        return 1.0
    if median_duration < 120:
        return 1000.0
    if max_end <= 1_000:
        return 1000.0
    return 1.0


def load_transcript_segments(transcript_json_path: str) -> list[TranscriptSegment]:
    """Load and normalize transcript segments from JSON.

    Args:
        transcript_json_path: Path to transcript JSON containing a ``segments`` list.

    Returns:
        Validated segments with millisecond timestamps.

    Raises:
        ValueError: If transcript payload is missing valid segments.
    """
    transcript_path = Path(transcript_json_path)
    with transcript_path.open("r", encoding="utf-8") as transcript_file:
        payload = json.load(transcript_file)

    if not isinstance(payload, dict):
        raise ValueError(
            "Transcript JSON must be an object with a top-level 'segments' list."
        )

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Transcript JSON must contain a 'segments' list.")

    scale = _infer_timestamp_scale(raw_segments)
    normalized_segments: list[TranscriptSegment] = []
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue

        speaker_raw = raw_segment.get("speaker")
        start_raw = raw_segment.get("start_time")
        end_raw = raw_segment.get("end_time")
        text_raw = raw_segment.get("text")

        if not isinstance(speaker_raw, str) or not speaker_raw.strip():
            continue
        if not isinstance(start_raw, (int, float)) or not isinstance(
            end_raw, (int, float)
        ):
            continue
        if not isinstance(text_raw, str):
            continue

        start = float(start_raw)
        end = float(end_raw)
        if end <= start:
            continue

        start_ms = int(round(start * scale))
        end_ms = int(round(end * scale))
        if end_ms <= start_ms:
            continue

        normalized_segments.append(
            TranscriptSegment(
                speaker=speaker_raw.strip(),
                start_ms=start_ms,
                end_ms=end_ms,
                text=text_raw.strip(),
            )
        )

    if not normalized_segments:
        raise ValueError(
            f"No valid transcript segments found in {transcript_json_path}."
        )
    return normalized_segments


def compute_per_speaker_wpm(
    segments: list[TranscriptSegment],
) -> dict[str, float]:
    """Compute words-per-minute for each speaker.

    Args:
        segments: Normalized transcript segments.

    Returns:
        Mapping of speaker label to computed WPM.

    Raises:
        ValueError: If no valid speaker-level WPM values can be computed.
    """
    word_totals: dict[str, int] = {}
    duration_seconds: dict[str, float] = {}

    for segment in segments:
        words = len(segment.text.split())
        segment_duration_seconds = max(
            0.0,
            (segment.end_ms - segment.start_ms) / 1000.0,
        )
        if words <= 0 or segment_duration_seconds <= 0.0:
            continue
        word_totals[segment.speaker] = word_totals.get(segment.speaker, 0) + words
        duration_seconds[segment.speaker] = (
            duration_seconds.get(segment.speaker, 0.0) + segment_duration_seconds
        )

    speaker_wpm: dict[str, float] = {}
    for speaker in sorted(word_totals):
        total_seconds = duration_seconds.get(speaker, 0.0)
        if total_seconds <= 0.0:
            continue
        speaker_wpm[speaker] = (word_totals[speaker] / total_seconds) * 60.0

    if not speaker_wpm:
        raise ValueError("Unable to compute WPM from transcript segments.")
    return speaker_wpm


def compute_average_wpm(per_speaker_wpm: dict[str, float]) -> float:
    """Compute rounded mean WPM across speakers.

    Args:
        per_speaker_wpm: Mapping of per-speaker WPM values.

    Returns:
        Mean speaker WPM rounded up to 2 decimals.

    Raises:
        ValueError: If ``per_speaker_wpm`` is empty.
    """
    if not per_speaker_wpm:
        raise ValueError("Cannot compute average WPM from empty speaker mapping.")
    avg_wpm = sum(per_speaker_wpm.values()) / len(per_speaker_wpm)
    avg_wpm = max(1.0, avg_wpm)
    return float(math.ceil(avg_wpm * 100.0) / 100.0)


def compute_wpm_from_transcript(
    transcript_json_path: str,
) -> tuple[float, dict[str, float]]:
    """Compute average and per-speaker WPM from a transcript JSON file.

    Args:
        transcript_json_path: Path to transcript JSON with diarized segments.

    Returns:
        Tuple of ``(average_wpm, per_speaker_wpm)``.
    """
    segments = load_transcript_segments(transcript_json_path=transcript_json_path)
    per_speaker_wpm = compute_per_speaker_wpm(segments=segments)
    average_wpm = compute_average_wpm(per_speaker_wpm)
    return average_wpm, per_speaker_wpm
