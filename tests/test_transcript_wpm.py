"""Unit tests for transcript-derived WPM computation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from audio2script_and_summarizer.transcript_wpm import (
    compute_average_wpm,
    compute_per_speaker_wpm,
    compute_wpm_from_transcript,
    load_transcript_segments,
)


def test_load_transcript_segments_converts_second_timestamps(tmp_path: Path) -> None:
    """Convert second-based timestamps to milliseconds."""
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 1.25,
                        "end_time": 3.0,
                        "text": "hello world",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    segments = load_transcript_segments(str(transcript_path))
    assert len(segments) == 1
    assert segments[0].speaker == "SPEAKER_00"
    assert segments[0].start_ms == 1250
    assert segments[0].end_ms == 3000


def test_compute_per_speaker_wpm_and_average(tmp_path: Path) -> None:
    """Compute expected per-speaker and mean WPM values."""
    transcript_path = tmp_path / "transcript_ms.json"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0,
                        "end_time": 60_000,
                        "text": "one two three",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 60_000,
                        "end_time": 120_000,
                        "text": "a b c d e f",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    segments = load_transcript_segments(str(transcript_path))
    per_speaker = compute_per_speaker_wpm(segments)
    average_wpm = compute_average_wpm(per_speaker)

    assert per_speaker["SPEAKER_00"] == pytest.approx(3.0)
    assert per_speaker["SPEAKER_01"] == pytest.approx(6.0)
    assert average_wpm == pytest.approx(4.5)

    computed_average, computed_mapping = compute_wpm_from_transcript(
        str(transcript_path)
    )
    assert computed_average == pytest.approx(average_wpm)
    assert computed_mapping == per_speaker


def test_load_transcript_segments_rejects_invalid_payload(tmp_path: Path) -> None:
    """Raise when transcript has no valid segments."""
    transcript_path = tmp_path / "invalid.json"
    transcript_path.write_text(json.dumps({"segments": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        load_transcript_segments(str(transcript_path))


def test_load_transcript_segments_rejects_top_level_list(tmp_path: Path) -> None:
    """Reject summary-like top-level list payloads."""
    transcript_path = tmp_path / "summary.json"
    transcript_path.write_text(
        json.dumps(
            [
                {
                    "speaker": "SPEAKER_00",
                    "text": "summary line",
                    "source_segment_ids": ["seg_00000"],
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="top-level 'segments' list"):
        load_transcript_segments(str(transcript_path))
