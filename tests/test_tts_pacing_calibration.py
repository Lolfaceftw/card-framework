"""Unit tests for emotion-aware TTS pacing calibration helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from audio2script_and_summarizer import tts_pacing_calibration


class _FakeTTS:
    """Test double for IndexTTS2 inference during calibration tests."""

    def infer(self, **_: Any) -> None:
        """Simulate successful synthesis without writing real audio."""


def _write_presets(path: Path) -> None:
    """Write a compact deterministic preset file for tests."""
    payload = {
        "presets": [
            {
                "name": "neutral",
                "emo_text": "Neutral",
                "emo_alpha": 0.6,
                "calibration_text": "one two three four",
                "keywords": ["neutral", "clear"],
            },
            {
                "name": "excited_fast",
                "emo_text": "Excited",
                "emo_alpha": 0.7,
                "calibration_text": "a b c d",
                "keywords": ["excited", "fast"],
            },
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_presets_and_resolve_emotion_name(tmp_path: Path) -> None:
    """Load preset JSON and map emotion text to expected bucket."""
    presets_path = tmp_path / "presets.json"
    _write_presets(presets_path)

    presets = tts_pacing_calibration.load_emotion_pacing_presets(str(presets_path))

    assert list(presets) == ["neutral", "excited_fast"]
    assert (
        tts_pacing_calibration.resolve_emotion_preset_name(
            emo_text="Excited and fast paced",
            presets=presets,
        )
        == "excited_fast"
    )
    assert (
        tts_pacing_calibration.resolve_emotion_preset_name(
            emo_text="unknown style",
            presets=presets,
        )
        == "neutral"
    )


def test_calibration_and_weighted_wpm_estimation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Calibrate speaker rates and derive transcript-weighted effective WPM."""
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "SPEAKER_00.wav").touch()
    (voice_dir / "SPEAKER_01.wav").touch()

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("dummy: true", encoding="utf-8")
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)

    presets_path = tmp_path / "presets.json"
    _write_presets(presets_path)

    duration_sequence = iter([2.0, 1.0, 4.0, 2.0])
    monkeypatch.setattr(
        tts_pacing_calibration,
        "_create_tts",
        lambda **_: _FakeTTS(),
    )
    monkeypatch.setattr(
        tts_pacing_calibration,
        "_measure_duration_seconds",
        lambda _path: next(duration_sequence),
    )

    avg_wpm, per_speaker_wpm, calibration = (
        tts_pacing_calibration.calibrate_tts_pacing_profiles(
            voice_dir=str(voice_dir),
            device="cpu",
            cfg_path=str(cfg_path),
            model_dir=str(model_dir),
            presets_path=str(presets_path),
            progress_cb=None,
        )
    )

    assert avg_wpm == pytest.approx(90.0)
    assert per_speaker_wpm["SPEAKER_00"] == pytest.approx(120.0)
    assert per_speaker_wpm["SPEAKER_01"] == pytest.approx(60.0)
    assert calibration.get_seconds_per_word("SPEAKER_00", "neutral") == pytest.approx(0.5)
    assert calibration.get_seconds_per_word("SPEAKER_01", "neutral") == pytest.approx(1.0)

    transcript_path = tmp_path / "transcript.json"
    transcript_payload = {
        "segments": [
            {
                "id": "seg_00000",
                "speaker": "SPEAKER_00",
                "text": "word " * 90,
                "start_time": 0.0,
                "end_time": 1.0,
            },
            {
                "id": "seg_00001",
                "speaker": "SPEAKER_01",
                "text": "word " * 30,
                "start_time": 1.0,
                "end_time": 2.0,
            },
        ]
    }
    transcript_path.write_text(json.dumps(transcript_payload), encoding="utf-8")

    weighted_wpm, weighted_per_speaker = (
        tts_pacing_calibration.estimate_weighted_wpm_from_transcript(
            transcript_json_path=str(transcript_path),
            calibration=calibration,
        )
    )
    assert weighted_per_speaker["SPEAKER_00"] == pytest.approx(120.0)
    assert weighted_per_speaker["SPEAKER_01"] == pytest.approx(60.0)
    assert weighted_wpm == pytest.approx(96.0)


def test_estimate_summary_duration_seconds(tmp_path: Path) -> None:
    """Estimate summary duration from calibrated rates and transition overhead."""
    presets = tts_pacing_calibration.load_emotion_pacing_presets(None)
    calibration = tts_pacing_calibration.TTSPacingCalibration(
        presets=presets,
        seconds_per_word_by_speaker_preset={
            "SPEAKER_00": {"neutral": 0.5, "excited_fast": 0.25},
            "SPEAKER_01": {"neutral": 1.0, "excited_fast": 0.6},
        },
        speaker_default_seconds_per_word={"SPEAKER_00": 0.5, "SPEAKER_01": 1.0},
        global_default_seconds_per_word=0.75,
    )
    summary_path = tmp_path / "summary.json"
    summary_payload = [
        {"speaker": "SPEAKER_00", "text": "one two three four", "emo_text": "excited"},
        {"speaker": "SPEAKER_01", "text": "alpha beta gamma delta", "emo_text": "neutral"},
    ]
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    duration = tts_pacing_calibration.estimate_summary_duration_seconds(
        summary_json_path=str(summary_path),
        calibration=calibration,
    )

    assert duration == pytest.approx(5.1, rel=1e-4)


def test_tts_pacing_calibration_serialization_round_trip() -> None:
    """Round-trip ``TTSPacingCalibration`` through ``to_dict`` and ``from_dict``."""
    presets = tts_pacing_calibration.load_emotion_pacing_presets(None)
    calibration = tts_pacing_calibration.TTSPacingCalibration(
        presets=presets,
        seconds_per_word_by_speaker_preset={
            "SPEAKER_00": {"neutral": 0.5, "excited_fast": 0.4},
            "SPEAKER_01": {"neutral": 1.0, "excited_fast": 0.8},
        },
        speaker_default_seconds_per_word={"SPEAKER_00": 0.5, "SPEAKER_01": 1.0},
        global_default_seconds_per_word=0.75,
    )

    payload = calibration.to_dict()
    restored = tts_pacing_calibration.TTSPacingCalibration.from_dict(payload)

    assert restored.global_default_seconds_per_word == pytest.approx(0.75)
    assert restored.get_seconds_per_word("SPEAKER_00", "neutral") == pytest.approx(0.5)
    assert restored.get_seconds_per_word("SPEAKER_01", "neutral") == pytest.approx(1.0)
