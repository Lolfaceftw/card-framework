"""Fallback voice-cloning gateways for testing and constrained environments."""

from __future__ import annotations

import shutil
from pathlib import Path

from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.voice_clone_contracts import VoiceCloneProvider


class PassthroughVoiceCloneGateway(VoiceCloneProvider):
    """Fallback gateway that copies reference audio as synthesized output."""

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
    ) -> Path:
        """
        Copy reference WAV to output path.

        Args:
            reference_audio_path: Source WAV path.
            text: Ignored synthesis text argument for interface compatibility.
            output_audio_path: Destination WAV path.

        Returns:
            Output WAV path.
        """
        del text
        if not reference_audio_path.exists():
            raise NonRetryableAudioStageError(
                f"Reference audio path does not exist: {reference_audio_path}"
            )
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reference_audio_path, output_audio_path)
        return output_audio_path
