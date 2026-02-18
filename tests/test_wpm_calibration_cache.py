"""Unit tests for Stage 1.75 WPM calibration cache helpers."""

from __future__ import annotations

import json
from pathlib import Path

from audio2script_and_summarizer.tts_pacing_calibration import (
    EmotionPacingPreset,
    TTSPacingCalibration,
)
from audio2script_and_summarizer.wpm_calibration_cache import (
    build_calibration_fingerprint,
    load_cached_tts_pacing,
    save_cached_tts_pacing,
)


def _write_test_inputs(base_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Create a deterministic calibration input fixture tree.

    Returns:
        Tuple of ``voice_dir``, ``cfg_path``, ``model_dir``, and ``presets_path``.
    """
    voice_dir = base_dir / "voices"
    voice_dir.mkdir(parents=True, exist_ok=True)
    (voice_dir / "SPEAKER_00.wav").write_bytes(b"RIFFTEST0001")
    (voice_dir / "SPEAKER_01.wav").write_bytes(b"RIFFTEST0002")

    cfg_path = base_dir / "config.yaml"
    cfg_path.write_text("dummy: true\n", encoding="utf-8")

    model_dir = base_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model.safetensors").write_bytes(b"model-bytes")

    presets_path = base_dir / "presets.json"
    presets_path.write_text(
        json.dumps(
            {
                "presets": [
                    {
                        "name": "neutral",
                        "emo_text": "Neutral",
                        "emo_alpha": 0.6,
                        "calibration_text": "one two three",
                        "keywords": ["neutral"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return voice_dir, cfg_path, model_dir, presets_path


def _sample_calibration() -> TTSPacingCalibration:
    """Build a tiny in-memory calibration object for cache tests."""
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


def test_build_calibration_fingerprint_changes_when_voice_files_change(
    tmp_path: Path,
) -> None:
    """Recompute fingerprint when input speaker WAV data changes."""
    voice_dir, cfg_path, model_dir, presets_path = _write_test_inputs(tmp_path)

    first = build_calibration_fingerprint(
        voice_dir=str(voice_dir),
        device="cpu",
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        presets_path=str(presets_path),
    )
    second = build_calibration_fingerprint(
        voice_dir=str(voice_dir),
        device="cpu",
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        presets_path=str(presets_path),
    )
    assert first == second

    (voice_dir / "SPEAKER_01.wav").write_bytes(b"RIFFTEST0002_CHANGED")
    changed = build_calibration_fingerprint(
        voice_dir=str(voice_dir),
        device="cpu",
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        presets_path=str(presets_path),
    )
    assert changed != first


def test_save_and_load_cached_tts_pacing_round_trip(tmp_path: Path) -> None:
    """Persist and recover serialized ``TTSPacingCalibration`` payload."""
    cache_dir = tmp_path / "cache"
    fingerprint = "test-fingerprint"
    calibration = _sample_calibration()

    saved_path = save_cached_tts_pacing(
        cache_dir=str(cache_dir),
        fingerprint=fingerprint,
        calibration=calibration,
    )
    loaded = load_cached_tts_pacing(
        cache_dir=str(cache_dir),
        fingerprint=fingerprint,
    )

    assert saved_path.exists()
    assert loaded is not None
    assert loaded.get_seconds_per_word("SPEAKER_00", "neutral") == 0.5


def test_load_cached_tts_pacing_returns_none_for_invalid_payload(tmp_path: Path) -> None:
    """Ignore malformed cache payloads and fallback to recalibration path."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "broken.json").write_text("{not-json", encoding="utf-8")

    loaded = load_cached_tts_pacing(
        cache_dir=str(cache_dir),
        fingerprint="broken",
    )
    assert loaded is None
