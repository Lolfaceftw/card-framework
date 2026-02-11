"""Pyannote-based speaker diarization."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from typing import TypeAlias

import torch

from ..audio_io import write_mono_wav_pcm16

SpeakerLabel: TypeAlias = tuple[int, int, int]


@contextmanager
def _torch_load_legacy_checkpoint_compatibility():
    """Temporarily default ``torch.load`` to ``weights_only=False``.

    PyTorch 2.6+ defaults to ``weights_only=True`` when callers do not pass the
    flag explicitly. Some pyannote/lightning checkpoints still require full
    pickle deserialization and fail under the new default.

    This shim is scoped to the pyannote model-load path only.
    """

    original_torch_load = torch.load

    def _torch_load_with_legacy_default(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = _torch_load_with_legacy_default
    try:
        yield
    finally:
        torch.load = original_torch_load


class PyannoteDiarizer:
    """Speaker diarization using ``pyannote.audio``."""

    def __init__(self, device: str | torch.device, hf_token: str | None = None) -> None:
        """Initialize the diarizer and load pyannote pipeline.

        Args:
            device: Runtime target device.
            hf_token: Optional HuggingFace token. Falls back to ``HF_TOKEN`` or
                ``HUGGINGFACE_TOKEN`` environment variables.

        Raises:
            ValueError: If no HuggingFace token is available.
            RuntimeError: If pipeline loading fails.
        """
        from pyannote.audio import Pipeline

        self.device = device
        token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

        if not token:
            print("\n" + "=" * 60)
            print("[WARN] HuggingFace token required for pyannote.audio diarization")
            print("=" * 60)
            print("To get a token:")
            print("  1. Create an account at https://huggingface.co")
            print("  2. Accept terms at https://huggingface.co/pyannote/speaker-diarization-3.1")
            print("  3. Get your token at https://huggingface.co/settings/tokens")
            print("=" * 60)
            token = input("Enter your HuggingFace token: ").strip()
            if token:
                os.environ["HF_TOKEN"] = token
                print("[INFO] HF_TOKEN set successfully.\n")
            else:
                raise ValueError(
                    "HuggingFace token required for pyannote.audio. "
                    "Set HF_TOKEN environment variable or pass hf_token parameter."
                )

        print("[INFO] Loading pyannote speaker diarization pipeline...")
        with _torch_load_legacy_checkpoint_compatibility():
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=token,
            )

        if self.pipeline is None:
            raise RuntimeError(
                "Failed to load pyannote pipeline. This usually means:\n"
                "  1. Your HuggingFace token is invalid or expired\n"
                "  2. You have not accepted model terms at:\n"
                "     https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "  3. You have not accepted segmentation model terms at:\n"
                "     https://huggingface.co/pyannote/segmentation-3.0\n"
                "Please visit those URLs and click 'Agree and access repository'."
            )

        self.pipeline.to(torch.device(device))

    def diarize(self, audio: torch.Tensor) -> list[SpeakerLabel]:
        """Perform speaker diarization.

        Args:
            audio: Audio tensor of shape ``(1, samples)`` at 16kHz.

        Returns:
            Diarization labels as ``(start_ms, end_ms, speaker_id)`` tuples.
        """

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = os.path.join(temp_dir, "audio.wav")
            write_mono_wav_pcm16(output_path=temp_path, audio=audio, sample_rate_hz=16000)

            diarization = self.pipeline(temp_path)
            labels: list[SpeakerLabel] = []
            speaker_map: dict[str, int] = {}

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                if speaker not in speaker_map:
                    speaker_map[speaker] = len(speaker_map)

                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                labels.append((start_ms, end_ms, speaker_map[speaker]))

            labels.sort(key=lambda label: label[0])
            return labels
