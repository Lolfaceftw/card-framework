"""Unit tests for IndexTTS calibration progress callbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from audio2script_and_summarizer import wpm_calibration
from audio2script_and_summarizer.wpm_calibration import CalibrationEvent


class _FakeTTS:
    """Test double for IndexTTS inference."""

    def infer(self, **_: Any) -> None:
        """Simulate successful inference without writing audio content."""


def test_calibrate_voice_wpm_emits_progress_events(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Emit model and speaker progress events in expected order."""
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "SPEAKER_00.wav").touch()
    (voice_dir / "SPEAKER_01.wav").touch()

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("dummy: true", encoding="utf-8")
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    duration_sequence = iter([10.0, 20.0])

    monkeypatch.setattr(wpm_calibration, "_create_tts", lambda **_: _FakeTTS())
    monkeypatch.setattr(
        wpm_calibration,
        "_measure_duration_seconds",
        lambda _path: next(duration_sequence),
    )

    events: list[CalibrationEvent] = []
    avg_wpm, per_speaker_wpm = wpm_calibration.calibrate_voice_wpm(
        voice_dir=str(voice_dir),
        device="cpu",
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        progress_cb=events.append,
    )

    words = len(wpm_calibration.CALIBRATION_TEXT.split())
    expected_speaker_00 = (words / 10.0) * 60.0
    expected_speaker_01 = (words / 20.0) * 60.0
    expected_avg = (expected_speaker_00 + expected_speaker_01) / 2.0

    assert per_speaker_wpm["SPEAKER_00"] == pytest.approx(expected_speaker_00)
    assert per_speaker_wpm["SPEAKER_01"] == pytest.approx(expected_speaker_01)
    assert avg_wpm == pytest.approx(expected_avg)

    event_types = [event.event_type for event in events]
    assert event_types[0] == "model_init_started"
    assert event_types[1] == "model_init_completed"
    assert event_types.count("speaker_started") == 2
    assert event_types.count("speaker_completed") == 2
    assert event_types[-1] == "calibration_completed"
