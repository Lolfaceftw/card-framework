"""Unit tests for the calibrate_wpm CLI module."""

from __future__ import annotations

import json
from pathlib import Path

from audio2script_and_summarizer import calibrate_wpm
from audio2script_and_summarizer.tts_pacing_calibration import (
    EmotionPacingPreset,
    TTSPacingCalibration,
)


def test_main_writes_calibration_output(monkeypatch, tmp_path: Path) -> None:
    """Write deterministic calibration output JSON from mocked calibration calls."""
    output_path = tmp_path / "calibrated.json"
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)

    presets = {
        "neutral": EmotionPacingPreset(
            name="neutral",
            emo_text="Neutral",
            emo_alpha=0.6,
            calibration_text="one two three",
            keywords=("neutral",),
        ),
        "excited_fast": EmotionPacingPreset(
            name="excited_fast",
            emo_text="Excited",
            emo_alpha=0.7,
            calibration_text="one two three",
            keywords=("excited",),
        ),
    }
    calibration = TTSPacingCalibration(
        presets=presets,
        seconds_per_word_by_speaker_preset={
            "SPEAKER_00": {"neutral": 0.5, "excited_fast": 0.4}
        },
        speaker_default_seconds_per_word={"SPEAKER_00": 0.5},
        global_default_seconds_per_word=0.5,
    )

    def _fake_calibrate_tts_pacing_profiles(**_: object):
        return 120.0, {"SPEAKER_00": 120.0}, calibration

    def _fake_estimate_weighted_wpm_from_transcript(**_: object):
        return 118.0, {"SPEAKER_00": 120.0}

    monkeypatch.setattr(
        calibrate_wpm,
        "calibrate_tts_pacing_profiles",
        _fake_calibrate_tts_pacing_profiles,
    )
    monkeypatch.setattr(
        calibrate_wpm,
        "estimate_weighted_wpm_from_transcript",
        _fake_estimate_weighted_wpm_from_transcript,
    )

    exit_code = calibrate_wpm.main(
        [
            "--voice-dir",
            str(voice_dir),
            "--transcript-json",
            str(tmp_path / "transcript.json"),
            "--output",
            str(output_path),
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["avg_wpm_neutral"] == 120.0
    assert payload["weighted_wpm_transcript_mix"] == 118.0
    assert payload["per_speaker_wpm_neutral"]["SPEAKER_00"] == 120.0
    assert payload["per_speaker_preset_wpm"]["SPEAKER_00"]["neutral"] == 120.0
    assert payload["per_speaker_preset_wpm"]["SPEAKER_00"]["excited_fast"] == 150.0
