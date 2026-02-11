"""Audio file I/O helpers for Audio2Script pipelines."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch


def write_mono_wav_pcm16(
    output_path: str | Path,
    audio: torch.Tensor,
    sample_rate_hz: int = 16000,
) -> None:
    """Write a mono PCM16 WAV file from normalized floating-point audio.

    Args:
        output_path: WAV file destination.
        audio: Audio tensor with shape ``(samples,)`` or ``(1, samples)``.
            Expected amplitude range is ``[-1.0, 1.0]``.
        sample_rate_hz: Output sample rate in Hz.

    Raises:
        ValueError: If the tensor shape is unsupported or empty.
    """
    if audio.ndim == 2:
        if audio.shape[0] != 1:
            raise ValueError(
                "Expected mono audio with shape (1, samples) when 2D tensor is provided."
            )
        mono_audio = audio[0]
    elif audio.ndim == 1:
        mono_audio = audio
    else:
        raise ValueError("Expected audio tensor with shape (samples,) or (1, samples).")

    if mono_audio.numel() == 0:
        raise ValueError("Expected non-empty audio tensor.")

    audio_array = mono_audio.detach().to("cpu", dtype=torch.float32).numpy()
    clipped = np.clip(audio_array, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16, copy=False)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(output_path_obj), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(pcm16.tobytes())
