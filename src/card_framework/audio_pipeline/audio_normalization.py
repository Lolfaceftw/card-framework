"""Configurable loudness normalization helpers for rendered audio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from card_framework.audio_pipeline.errors import NonRetryableAudioStageError
from card_framework.audio_pipeline.runtime import ensure_command_available


@dataclass(slots=True, frozen=True)
class AudioNormalizationPolicy:
    """Describe one FFmpeg loudness-normalization policy."""

    enabled: bool = False
    target_lufs: float = -16.0
    true_peak_db: float = -1.5
    loudness_range_lu: float = 11.0
    linear: bool = True
    dual_mono: bool = False
    timeout_seconds: int = 300


def normalize_audio_file(
    *,
    audio_path: Path,
    policy: AudioNormalizationPolicy | None,
) -> Path:
    """Normalize one WAV file in place when the policy is enabled."""
    if policy is None or not policy.enabled:
        return audio_path
    ensure_command_available("ffmpeg")
    if not audio_path.exists():
        raise NonRetryableAudioStageError(
            f"Cannot normalize missing audio artifact: {audio_path}"
        )
    temp_output_path = audio_path.with_name(f"{audio_path.stem}.normalized{audio_path.suffix}")
    filter_expression = (
        "loudnorm="
        f"I={policy.target_lufs}:"
        f"TP={policy.true_peak_db}:"
        f"LRA={policy.loudness_range_lu}:"
        f"linear={'true' if policy.linear else 'false'}:"
        f"dual_mono={'true' if policy.dual_mono else 'false'}"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-af",
        filter_expression,
        "-c:a",
        "pcm_s24le",
        "-vn",
        "-f",
        "wav",
        str(temp_output_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=policy.timeout_seconds,
        )
        temp_output_path.replace(audio_path)
    except subprocess.TimeoutExpired as exc:
        _remove_temp_output(temp_output_path)
        raise NonRetryableAudioStageError(
            "Audio normalization timed out. "
            f"Command: {' '.join(command)}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        _remove_temp_output(temp_output_path)
        raise NonRetryableAudioStageError(
            "Audio normalization failed with ffmpeg. "
            f"Command: {' '.join(command)}. "
            f"Stderr: {(exc.stderr or '').strip()[:500]}"
        ) from exc
    except Exception as exc:
        _remove_temp_output(temp_output_path)
        raise NonRetryableAudioStageError(
            f"Failed to persist normalized audio artifact: {audio_path}"
        ) from exc
    return audio_path


def _remove_temp_output(path: Path) -> None:
    """Best-effort cleanup for failed normalization outputs."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        return
