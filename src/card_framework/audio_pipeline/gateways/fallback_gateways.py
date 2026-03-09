"""Fallback adapters used for development and constrained environments."""

from __future__ import annotations

from pathlib import Path

from card_framework.audio_pipeline.contracts import DiarizationTurn, SourceSeparator, SpeakerDiarizer
from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate


class PassthroughSourceSeparator(SourceSeparator):
    """No-op separator that returns the original audio path."""

    def separate_vocals(
        self,
        input_audio_path: Path,
        output_dir: Path,
        *,
        device: str,
        progress_callback: StageProgressCallback | None = None,
    ) -> Path:
        del output_dir, device
        if progress_callback is not None:
            try:
                progress_callback(
                    StageProgressUpdate(
                        completed_units=1,
                        total_units=1,
                        note="passthrough separation completed",
                    )
                )
            except Exception:
                pass
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
        progress_callback: StageProgressCallback | None = None,
    ) -> list[DiarizationTurn]:
        del audio_path, output_dir, device
        if progress_callback is not None:
            try:
                progress_callback(
                    StageProgressUpdate(
                        completed_units=1,
                        total_units=1,
                        note="single-speaker diarization completed",
                    )
                )
            except Exception:
                pass
        return [
            DiarizationTurn(
                speaker=self.speaker_label,
                start_time_ms=0,
                end_time_ms=self.duration_ms,
            )
        ]

