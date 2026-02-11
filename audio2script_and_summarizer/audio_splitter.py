"""Split diarized audio into per-speaker reference clips."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydub import AudioSegment

from .logging_utils import configure_logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TEN_MINUTES_MS = 10 * 60 * 1000
ONE_MINUTE_MS = 60 * 1000
DEFAULT_TARGET_DURATION_SECONDS = 30.0


@dataclass(slots=True, frozen=True)
class SpeakerInterval:
    """Represent one speaker interval in milliseconds."""

    speaker: str
    start_ms: int
    end_ms: int


def load_audio(audio_path: str) -> AudioSegment:
    """Load an input audio file.

    Args:
        audio_path: Path to source audio.

    Returns:
        Decoded audio segment.
    """
    logger.info("Loading audio from %s", audio_path)
    return AudioSegment.from_file(audio_path)


def get_safe_zone(audio_len_ms: int) -> tuple[int, int]:
    """Compute a safe region for extracting voice samples.

    For recordings longer than 10 minutes, this excludes the first and last
    minute to reduce intro/outro music contamination.

    Args:
        audio_len_ms: Audio duration in milliseconds.

    Returns:
        Inclusive start and exclusive end in milliseconds.
    """
    if audio_len_ms > TEN_MINUTES_MS:
        safe_start = ONE_MINUTE_MS
        safe_end = audio_len_ms - ONE_MINUTE_MS
        logger.info(
            "Audio >10 minutes; safe zone: %.2fs to %.2fs",
            safe_start / 1000.0,
            safe_end / 1000.0,
        )
        return safe_start, safe_end

    logger.info("Audio <=10 minutes; safe zone: full duration")
    return 0, audio_len_ms


def _normalize_timestamp_scale(
    raw_segments: list[dict[str, Any]],
    audio_len_ms: int,
) -> float:
    """Infer diarization timestamp scale.

    Args:
        raw_segments: Segments from diarization JSON.
        audio_len_ms: Audio duration in milliseconds.

    Returns:
        ``1000.0`` when input appears to be seconds, else ``1.0`` for ms.
    """
    max_end = 0.0
    for segment in raw_segments:
        end_raw = segment.get("end_time")
        if isinstance(end_raw, (int, float)):
            max_end = max(max_end, float(end_raw))
    audio_len_seconds = audio_len_ms / 1000.0
    if max_end > 0.0 and max_end <= (audio_len_seconds * 1.5):
        return 1000.0
    return 1.0


def load_speaker_intervals(
    diarization_json_path: str,
    audio_len_ms: int,
) -> list[SpeakerInterval]:
    """Load and normalize speaker intervals from diarization output.

    Args:
        diarization_json_path: Path to diarization JSON.
        audio_len_ms: Source audio duration in milliseconds.

    Returns:
        Parsed intervals in milliseconds.

    Raises:
        ValueError: If JSON does not contain a valid ``segments`` list.
    """
    with open(diarization_json_path, "r", encoding="utf-8") as json_file:
        payload = json.load(json_file)

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError("Diarization JSON must contain a 'segments' list.")

    scale = _normalize_timestamp_scale(raw_segments, audio_len_ms)
    if scale == 1000.0:
        logger.info("Detected second-based timestamps; converting to milliseconds.")

    intervals: list[SpeakerInterval] = []
    for index, raw_segment in enumerate(raw_segments):
        if not isinstance(raw_segment, dict):
            logger.warning("Skipping segment %d: not an object.", index)
            continue
        speaker_raw = raw_segment.get("speaker")
        start_raw = raw_segment.get("start_time")
        end_raw = raw_segment.get("end_time")

        if not isinstance(speaker_raw, str) or not speaker_raw.strip():
            logger.warning("Skipping segment %d: invalid speaker label.", index)
            continue
        if not isinstance(start_raw, (int, float)) or not isinstance(end_raw, (int, float)):
            logger.warning("Skipping segment %d: invalid timestamps.", index)
            continue

        start_ms = int(round(float(start_raw) * scale))
        end_ms = int(round(float(end_raw) * scale))
        if end_ms <= start_ms:
            logger.warning("Skipping segment %d: non-positive duration.", index)
            continue

        intervals.append(
            SpeakerInterval(
                speaker=speaker_raw.strip(),
                start_ms=start_ms,
                end_ms=end_ms,
            )
        )

    intervals.sort(key=lambda interval: interval.start_ms)
    return intervals


def _extract_sample_for_speaker(
    audio: AudioSegment,
    intervals: list[tuple[int, int]],
    target_duration_ms: int,
) -> AudioSegment | None:
    """Build one speaker reference sample using contiguous-first strategy.

    Args:
        audio: Full source audio.
        intervals: Clipped intervals for one speaker.
        target_duration_ms: Target sample length in milliseconds.

    Returns:
        Extracted sample audio or ``None`` when no usable audio exists.
    """
    for start_ms, end_ms in intervals:
        duration_ms = end_ms - start_ms
        if duration_ms >= target_duration_ms:
            return audio[start_ms : start_ms + target_duration_ms]

    combined_audio = AudioSegment.empty()
    for start_ms, end_ms in sorted(intervals, key=lambda item: item[1] - item[0], reverse=True):
        combined_audio += audio[start_ms:end_ms]
        if len(combined_audio) >= target_duration_ms:
            break

    if len(combined_audio) <= 0:
        return None
    return combined_audio[: min(len(combined_audio), target_duration_ms)]


def extract_speaker_samples(
    audio: AudioSegment,
    intervals: list[SpeakerInterval],
    output_dir: str,
    target_duration_seconds: float,
) -> dict[str, str]:
    """Extract per-speaker WAV samples.

    Args:
        audio: Source audio.
        intervals: Speaker intervals in milliseconds.
        output_dir: Destination directory for samples.
        target_duration_seconds: Desired sample length per speaker.

    Returns:
        Mapping of speaker label to output sample path.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_start, safe_end = get_safe_zone(len(audio))
    per_speaker_intervals: dict[str, list[tuple[int, int]]] = {}

    for interval in intervals:
        valid_start = max(interval.start_ms, safe_start)
        valid_end = min(interval.end_ms, safe_end)
        if valid_end <= valid_start:
            continue
        per_speaker_intervals.setdefault(interval.speaker, []).append((valid_start, valid_end))

    target_duration_ms = max(1, int(round(target_duration_seconds * 1000.0)))
    outputs: dict[str, str] = {}

    for speaker, speaker_intervals in per_speaker_intervals.items():
        logger.info("Extracting sample for %s", speaker)
        sample = _extract_sample_for_speaker(
            audio=audio,
            intervals=speaker_intervals,
            target_duration_ms=target_duration_ms,
        )
        if sample is None:
            logger.warning("No usable audio found for speaker %s", speaker)
            continue

        destination = output_path / f"{speaker}.wav"
        sample.export(destination, format="wav")
        outputs[speaker] = str(destination)
        logger.info(
            "Saved %s (%.2fs)",
            destination,
            len(sample) / 1000.0,
        )

    return outputs


def main() -> int:
    """Run the diarization-guided audio splitter CLI."""
    parser = argparse.ArgumentParser(description="CARD Audio Splitter")
    parser.add_argument("--audio", required=True, help="Path to source audio file")
    parser.add_argument("--json", required=True, help="Path to diarization JSON")
    parser.add_argument("--output-dir", required=True, help="Directory for speaker samples")
    parser.add_argument(
        "--target-duration-seconds",
        type=float,
        default=DEFAULT_TARGET_DURATION_SECONDS,
        help=(
            "Target duration in seconds for each speaker sample "
            f"(default: {DEFAULT_TARGET_DURATION_SECONDS})"
        ),
    )
    args = parser.parse_args()

    configure_logging()
    try:
        audio = load_audio(args.audio)
        intervals = load_speaker_intervals(args.json, audio_len_ms=len(audio))
        outputs = extract_speaker_samples(
            audio=audio,
            intervals=intervals,
            output_dir=args.output_dir,
            target_duration_seconds=args.target_duration_seconds,
        )
        if not outputs:
            logger.error("No speaker sample files were generated.")
            return 1
        logger.info("Generated %d speaker sample file(s).", len(outputs))
        return 0
    except Exception:
        logger.exception("Audio splitting failed.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
