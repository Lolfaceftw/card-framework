"""Configuration helpers for audio pipeline toggles."""

from __future__ import annotations

from typing import Any


def should_use_audio_stage(audio_cfg: dict[str, Any]) -> bool:
    """Return whether audio-to-script stage should run for current config."""
    input_mode = str(audio_cfg.get("input_mode", "audio_first")).strip().lower()
    if input_mode == "auto_detect":
        return bool(str(audio_cfg.get("audio_path", "")).strip())
    if input_mode == "audio_first":
        return True
    raise ValueError("audio.input_mode must be one of: audio_first, auto_detect")
