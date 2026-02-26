"""Fallback adapters used for development and constrained environments."""

from __future__ import annotations

from pathlib import Path

from audio_pipeline.contracts import DiarizationTurn, SourceSeparator, SpeakerDiarizer


class PassthroughSourceSeparator(SourceSeparator):
    """No-op separator that returns the original audio path."""

    def separate_vocals(
        self,
        input_audio_path: Path,
        output_dir: Path,
        *,
        device: str,
    ) -> Path:
        del output_dir, device
        return input_audio_path


class SingleSpeakerDiarizer(SpeakerDiarizer):
    """Fallback diarizer that assigns a single speaker label."""

    def __init__(self, speaker_label: str = "SPEAKER_00", duration_ms: int = 86_400_000):
        self.speaker_label = speaker_label
        self.duration_ms = duration_ms

    def diarize(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        device: str,
    ) -> list[DiarizationTurn]:
        del audio_path, output_dir, device
        return [
            DiarizationTurn(
                speaker=self.speaker_label,
                start_time_ms=0,
                end_time_ms=self.duration_ms,
            )
        ]
