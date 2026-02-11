"""Tests for audio file I/O helpers."""

from __future__ import annotations

import wave

import pytest
import torch

from audio2script_and_summarizer.audio_io import write_mono_wav_pcm16


def test_write_mono_wav_pcm16_writes_expected_metadata(tmp_path) -> None:
    """Write a mono WAV file with the expected PCM16 metadata."""
    output_path = tmp_path / "sample.wav"
    audio = torch.linspace(-0.5, 0.5, steps=1600, dtype=torch.float32).unsqueeze(0)

    write_mono_wav_pcm16(output_path=output_path, audio=audio, sample_rate_hz=16000)

    with wave.open(str(output_path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 1600


def test_write_mono_wav_pcm16_rejects_multichannel_tensor(tmp_path) -> None:
    """Raise when a multi-channel tensor is provided."""
    output_path = tmp_path / "invalid.wav"
    stereo_audio = torch.zeros((2, 128), dtype=torch.float32)

    with pytest.raises(ValueError, match="Expected mono audio"):
        write_mono_wav_pcm16(output_path=output_path, audio=stereo_audio)
