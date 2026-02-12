"""Tests for run_pipeline --skip-a2s transcript discovery helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from audio2script_and_summarizer.run_pipeline import _discover_transcript_json_files


def _write_json(path: Path, payload: object) -> None:
    """Write JSON payload to disk with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_transcript_json_files_filters_and_orders(tmp_path: Path) -> None:
    """Return only transcript-like JSON candidates sorted newest first."""
    older_transcript = tmp_path / "older.json"
    newer_transcript = tmp_path / "newer.json"
    invalid_payload = tmp_path / "config.json"
    summary_payload = tmp_path / "audio_summary.json"
    excluded_transcript = tmp_path / ".git" / "ignored.json"

    transcript_payload = {
        "segments": [
            {
                "id": "seg_00000",
                "speaker": "SPEAKER_00",
                "text": "Hello world",
                "start_time": 0.0,
                "end_time": 1.0,
            }
        ]
    }
    _write_json(older_transcript, transcript_payload)
    _write_json(newer_transcript, transcript_payload)
    _write_json(invalid_payload, {"foo": "bar"})
    _write_json(
        summary_payload,
        [
            {
                "speaker": "SPEAKER_00",
                "text": "Summary line",
                "source_segment_ids": ["seg_00000"],
            }
        ],
    )
    _write_json(excluded_transcript, transcript_payload)

    now = 1_700_000_000
    os.utime(older_transcript, (now - 60, now - 60))
    os.utime(newer_transcript, (now, now))

    results = _discover_transcript_json_files(tmp_path)

    assert results == [newer_transcript.resolve(), older_transcript.resolve()]
