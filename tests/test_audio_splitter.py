"""Unit tests for diarization-guided speaker audio splitting."""

from __future__ import annotations

import json
from pathlib import Path

from pydub import AudioSegment

from audio2script_and_summarizer.audio_splitter import (
    SpeakerInterval,
    extract_speaker_samples,
    get_safe_zone,
    load_speaker_intervals,
)


def test_get_safe_zone_uses_full_duration_for_short_audio() -> None:
    """Return full-range safe zone for audio shorter than 10 minutes."""
    safe_start, safe_end = get_safe_zone(audio_len_ms=2 * 60 * 1000)
    assert safe_start == 0
    assert safe_end == 2 * 60 * 1000


def test_get_safe_zone_excludes_intro_outro_for_long_audio() -> None:
    """Exclude first and last minute for long-form recordings."""
    safe_start, safe_end = get_safe_zone(audio_len_ms=15 * 60 * 1000)
    assert safe_start == 60 * 1000
    assert safe_end == 14 * 60 * 1000


def test_load_speaker_intervals_converts_second_timestamps(tmp_path: Path) -> None:
    """Normalize second-based diarization timestamps into milliseconds."""
    payload_path = tmp_path / "segments.json"
    payload_path.write_text(
        json.dumps(
            {
                "segments": [
                    {"speaker": "SPEAKER_00", "start_time": 1.25, "end_time": 2.5},
                    {"speaker": "SPEAKER_01", "start_time": 3.0, "end_time": 3.75},
                ]
            }
        ),
        encoding="utf-8",
    )

    intervals = load_speaker_intervals(
        diarization_json_path=str(payload_path),
        audio_len_ms=10 * 1000,
    )

    assert intervals == [
        SpeakerInterval(speaker="SPEAKER_00", start_ms=1250, end_ms=2500),
        SpeakerInterval(speaker="SPEAKER_01", start_ms=3000, end_ms=3750),
    ]


def test_extract_speaker_samples_writes_expected_outputs(tmp_path: Path) -> None:
    """Generate one wav sample file per speaker interval set."""
    audio = AudioSegment.silent(duration=4000)
    intervals = [
        SpeakerInterval(speaker="SPEAKER_00", start_ms=0, end_ms=2500),
        SpeakerInterval(speaker="SPEAKER_01", start_ms=1200, end_ms=3900),
    ]
    output_dir = tmp_path / "voices"

    outputs = extract_speaker_samples(
        audio=audio,
        intervals=intervals,
        output_dir=str(output_dir),
        target_duration_seconds=1.0,
    )

    assert set(outputs.keys()) == {"SPEAKER_00", "SPEAKER_01"}
    for speaker in outputs:
        sample_path = output_dir / f"{speaker}.wav"
        assert sample_path.exists()
        assert sample_path.stat().st_size > 0
