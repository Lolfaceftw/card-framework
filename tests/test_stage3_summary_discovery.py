"""Tests for Stage 3 summary JSON discovery helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path

from audio2script_and_summarizer.stage3_voice import (
    discover_summary_json_files,
    select_latest_summary_json,
)


def _write_json(path: Path, payload: object) -> None:
    """Write JSON payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_summary_json_files_filters_and_orders(tmp_path: Path) -> None:
    """Return only valid summary JSON files in descending mtime order."""
    older = tmp_path / "older_summary.json"
    newer = tmp_path / "newer_summary.json"
    invalid = tmp_path / "invalid_summary.json"
    report = tmp_path / "audio_summary.json.report.json"
    buffer_file = tmp_path / "audio_summary.json.agent_buffer"

    valid_payload = [
        {
            "speaker": "SPEAKER_00",
            "voice_sample": "SPEAKER_00.wav",
            "text": "Hello there",
            "emo_text": "Warm",
            "emo_alpha": 0.6,
            "use_emo_text": True,
        }
    ]
    _write_json(older, valid_payload)
    _write_json(newer, valid_payload)
    _write_json(invalid, {"foo": "bar"})
    _write_json(report, valid_payload)
    _write_json(buffer_file, valid_payload)

    now = 1_700_000_000
    os.utime(older, (now - 60, now - 60))
    os.utime(newer, (now, now))

    discovered = discover_summary_json_files(tmp_path)
    assert discovered == [newer.resolve(), older.resolve()]


def test_select_latest_summary_json_picks_newest(tmp_path: Path) -> None:
    """Select the newest discovered summary JSON path."""
    first = tmp_path / "a_summary.json"
    second = tmp_path / "b_summary.json"
    payload = [
        {"speaker": "SPEAKER_00", "voice_sample": "s0.wav", "text": "Line one"},
    ]
    _write_json(first, payload)
    _write_json(second, payload)

    now = 1_700_000_500
    os.utime(first, (now - 10, now - 10))
    os.utime(second, (now, now))

    selected = select_latest_summary_json(tmp_path)
    assert selected == second.resolve()
