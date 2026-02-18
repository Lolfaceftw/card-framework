"""Unit tests for Stage 1.75 cache behavior in pipeline flow helpers."""

from __future__ import annotations

import contextlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from audio2script_and_summarizer.pipeline import flows
from audio2script_and_summarizer.tts_pacing_calibration import (
    EmotionPacingPreset,
    TTSPacingCalibration,
)


def _sample_calibration() -> TTSPacingCalibration:
    """Build a compact calibration object for helper tests."""
    preset = EmotionPacingPreset(
        name="neutral",
        emo_text="Neutral",
        emo_alpha=0.6,
        calibration_text="one two three",
        keywords=("neutral",),
    )
    return TTSPacingCalibration(
        presets={"neutral": preset},
        seconds_per_word_by_speaker_preset={"SPEAKER_00": {"neutral": 0.5}},
        speaker_default_seconds_per_word={"SPEAKER_00": 0.5},
        global_default_seconds_per_word=0.5,
    )


def _noop_capture_factory(enabled: bool) -> contextlib.AbstractContextManager[None]:
    """Return no-op capture context for helper unit tests."""
    _ = enabled
    return contextlib.nullcontext()


def test_compute_tts_preflight_wpm_uses_cache_hit(monkeypatch, tmp_path: Path) -> None:
    """Load cached calibration in auto mode and skip expensive recalibration."""
    args = SimpleNamespace(
        calibration_presets_path=str(tmp_path / "presets.json"),
        wpm_calibration_cache_mode="auto",
        wpm_calibration_cache_dir=str(tmp_path / "cache"),
    )
    calibration = _sample_calibration()

    monkeypatch.setattr(
        "audio2script_and_summarizer.wpm_calibration_cache.build_calibration_fingerprint",
        lambda **_: "fingerprint-123",
    )
    monkeypatch.setattr(
        "audio2script_and_summarizer.wpm_calibration_cache.load_cached_tts_pacing",
        lambda **_: calibration,
    )
    monkeypatch.setattr(
        "audio2script_and_summarizer.tts_pacing_calibration.calibrate_tts_pacing_profiles",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("calibrate_tts_pacing_profiles should not run on cache hit")
        ),
    )
    monkeypatch.setattr(
        "audio2script_and_summarizer.tts_pacing_calibration.estimate_weighted_wpm_from_transcript",
        lambda **_: (120.0, {"SPEAKER_00": 120.0}),
    )

    avg_wpm, per_speaker_wpm, _, cache_hit, cache_path = flows._compute_tts_preflight_wpm(  # noqa: SLF001
        args=args,
        voice_dir=str(tmp_path / "voices"),
        runtime_device="cpu",
        transcript_json_path=str(tmp_path / "transcript.json"),
        capture_stage175_runtime_output=False,
        progress_cb=None,
        _capture_stage175_output_lines=_noop_capture_factory,
        logger=flows.logging.getLogger(__name__),
    )

    assert cache_hit is True
    assert avg_wpm == 120.0
    assert per_speaker_wpm["SPEAKER_00"] == 120.0
    assert cache_path == (tmp_path / "cache" / "fingerprint-123.json").resolve()


def test_compute_tts_preflight_wpm_refresh_mode_forces_recalibration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Bypass cache reads in refresh mode and persist new calibration output."""
    args = SimpleNamespace(
        calibration_presets_path=str(tmp_path / "presets.json"),
        wpm_calibration_cache_mode="refresh",
        wpm_calibration_cache_dir=str(tmp_path / "cache"),
    )
    calibration = _sample_calibration()
    state: dict[str, Any] = {"saved": False, "calibrated": False}

    monkeypatch.setattr(
        "audio2script_and_summarizer.wpm_calibration_cache.build_calibration_fingerprint",
        lambda **_: "fingerprint-refresh",
    )
    monkeypatch.setattr(
        "audio2script_and_summarizer.wpm_calibration_cache.load_cached_tts_pacing",
        lambda **_: (_ for _ in ()).throw(
            AssertionError("load_cached_tts_pacing should not run in refresh mode")
        ),
    )

    def _fake_calibrate(**_: object) -> tuple[float, dict[str, float], TTSPacingCalibration]:
        state["calibrated"] = True
        return 120.0, {"SPEAKER_00": 120.0}, calibration

    monkeypatch.setattr(
        "audio2script_and_summarizer.tts_pacing_calibration.calibrate_tts_pacing_profiles",
        _fake_calibrate,
    )
    monkeypatch.setattr(
        "audio2script_and_summarizer.tts_pacing_calibration.estimate_weighted_wpm_from_transcript",
        lambda **_: (118.0, {"SPEAKER_00": 120.0}),
    )

    def _fake_save(**_: object) -> Path:
        state["saved"] = True
        return (tmp_path / "cache" / "fingerprint-refresh.json").resolve()

    monkeypatch.setattr(
        "audio2script_and_summarizer.wpm_calibration_cache.save_cached_tts_pacing",
        _fake_save,
    )

    avg_wpm, _, _, cache_hit, cache_path = flows._compute_tts_preflight_wpm(  # noqa: SLF001
        args=args,
        voice_dir=str(tmp_path / "voices"),
        runtime_device="cpu",
        transcript_json_path=str(tmp_path / "transcript.json"),
        capture_stage175_runtime_output=False,
        progress_cb=None,
        _capture_stage175_output_lines=_noop_capture_factory,
        logger=flows.logging.getLogger(__name__),
    )

    assert cache_hit is False
    assert avg_wpm == 118.0
    assert cache_path == (tmp_path / "cache" / "fingerprint-refresh.json").resolve()
    assert state["calibrated"] is True
    assert state["saved"] is True
