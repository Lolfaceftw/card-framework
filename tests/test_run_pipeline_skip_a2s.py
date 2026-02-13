"""Tests for run_pipeline --skip-a2s transcript discovery helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import audio2script_and_summarizer.stage3_voice as stage3_voice
from audio2script_and_summarizer import run_pipeline
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


def test_skip_a2s_summary_uses_explicit_summary_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run direct Stage 3 mode with an explicit summary JSON path."""
    summary_path = tmp_path / "audio_summary.json"
    _write_json(
        summary_path,
        [
            {
                "speaker": "SPEAKER_00",
                "voice_sample": "SPEAKER_00.wav",
                "text": "Hello world",
            }
        ],
    )

    captured: dict[str, object] = {}

    def _fake_stage3(**kwargs: object) -> tuple[Path, Path, bool, float]:
        captured.update(kwargs)
        return (
            tmp_path / "final.wav",
            tmp_path / "final_interjections.json",
            True,
            12.5,
        )

    monkeypatch.setattr(run_pipeline, "_run_stage3_from_summary", _fake_stage3)
    monkeypatch.setattr(
        run_pipeline,
        "configure_logging",
        lambda *args, **kwargs: tmp_path / "test.log",
    )
    monkeypatch.setenv("AUDIO2SCRIPT_LOG_LEVEL", "INFO")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pipeline.py",
            "--skip-a2s-summary",
            "--summary-json",
            str(summary_path),
            "--plain-ui",
        ],
    )

    result_code = run_pipeline.main()
    assert result_code == 0
    assert captured["summary_json_path"] == summary_path.resolve()


def test_skip_a2s_summary_auto_detects_newest_summary_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Auto-select newest summary JSON in direct Stage 3 mode."""
    older_summary = tmp_path / "older_summary.json"
    newer_summary = tmp_path / "newer_summary.json"
    payload = [
        {
            "speaker": "SPEAKER_00",
            "voice_sample": "SPEAKER_00.wav",
            "text": "Hello world",
        }
    ]
    _write_json(older_summary, payload)
    _write_json(newer_summary, payload)
    now = 1_700_000_000
    os.utime(older_summary, (now - 30, now - 30))
    os.utime(newer_summary, (now, now))

    captured: dict[str, object] = {}

    def _fake_stage3(**kwargs: object) -> tuple[Path, Path, bool, float]:
        captured.update(kwargs)
        return (
            tmp_path / "final.wav",
            tmp_path / "final_interjections.json",
            True,
            12.5,
        )

    monkeypatch.setattr(run_pipeline, "_run_stage3_from_summary", _fake_stage3)
    monkeypatch.setattr(
        run_pipeline,
        "configure_logging",
        lambda *args, **kwargs: tmp_path / "test.log",
    )
    monkeypatch.setenv("AUDIO2SCRIPT_LOG_LEVEL", "INFO")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_pipeline.py",
            "--skip-a2s-summary",
            "--skip-a2s-search-root",
            str(tmp_path),
            "--plain-ui",
        ],
    )

    result_code = run_pipeline.main()
    assert result_code == 0
    assert captured["summary_json_path"] == newer_summary.resolve()


def test_stage3_runtime_output_is_captured_into_dashboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route Stage 3 stdout/stderr lines into dashboard output when enabled."""

    class _FakeDashboard:
        """Capture output lines sent to dashboard logs."""

        enabled = True

        def __init__(self) -> None:
            self.lines: list[str] = []

        def log(self, message: str) -> None:
            self.lines.append(message)

    class _FakeStage3Result:
        """Mimic the minimal Stage 3 result interface."""

        def __init__(self) -> None:
            self.output_wav_path = tmp_path / "out.wav"
            self.interjection_log_path = tmp_path / "out_interjections.json"
            self.output_duration_ms = 4200
            self.mistral_enabled = True

    def _fake_run_stage3_pipeline(**kwargs: object) -> _FakeStage3Result:
        print(">> starting inference...")
        sys.stderr.write("RuntimeError('Ninja is required')\n")
        return _FakeStage3Result()

    fake_dashboard = _FakeDashboard()
    monkeypatch.setattr(run_pipeline, "_ACTIVE_DASHBOARD", fake_dashboard)
    monkeypatch.setattr(stage3_voice, "run_stage3_pipeline", _fake_run_stage3_pipeline)

    output_wav, interjection_log, mistral_enabled, duration_seconds = (
        run_pipeline._run_stage3_from_summary(
            summary_json_path=tmp_path / "summary.json",
            output_wav_path=None,
            runtime_device="cpu",
            interjection_max_ratio=0.3,
            mistral_model_id="mistralai/Mistral-7B-Instruct-v0.3",
            mistral_max_new_tokens=64,
        )
    )

    assert output_wav == tmp_path / "out.wav"
    assert interjection_log == tmp_path / "out_interjections.json"
    assert mistral_enabled is True
    assert duration_seconds == pytest.approx(4.2)
    assert any("starting inference" in line for line in fake_dashboard.lines)
    assert any("Ninja is required" in line for line in fake_dashboard.lines)


def test_stage175_runtime_output_is_captured_into_dashboard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Route Stage 1.75 stdout/stderr lines into dashboard output when enabled."""

    class _FakeDashboard:
        """Capture output lines sent to dashboard logs."""

        enabled = True

        def __init__(self) -> None:
            self.lines: list[str] = []

        def log(self, message: str) -> None:
            self.lines.append(message)

    fake_dashboard = _FakeDashboard()
    monkeypatch.setattr(run_pipeline, "_ACTIVE_DASHBOARD", fake_dashboard)

    with run_pipeline._capture_stage175_output_lines(enabled=True):  # noqa: SLF001
        print(">> starting inference...")
        sys.stderr.write("RuntimeError('Ninja is required')\n")

    assert any(
        line.startswith("[INDEXTTS2]") and "starting inference" in line
        for line in fake_dashboard.lines
    )
    assert any(
        line.startswith("[WARNING] [INDEXTTS2]") and "Ninja is required" in line
        for line in fake_dashboard.lines
    )
