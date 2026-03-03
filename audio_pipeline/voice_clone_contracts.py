"""Typed contracts for post-summary voice-cloning workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from audio_pipeline.eta import StageProgressCallback


@dataclass(slots=True, frozen=True)
class VoiceCloneTurn:
    """Represent one speaker turn extracted from summary XML."""

    speaker: str
    text: str

    def __post_init__(self) -> None:
        if not self.speaker.strip():
            raise ValueError("speaker must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")


@dataclass(slots=True, frozen=True)
class VoiceSampleReference:
    """Represent one speaker-specific reference sample artifact."""

    speaker: str
    path: Path
    duration_ms: int

    def __post_init__(self) -> None:
        if not self.speaker.strip():
            raise ValueError("speaker must be non-empty")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be > 0")


@dataclass(slots=True, frozen=True)
class VoiceCloneArtifact:
    """Describe one synthesized turn artifact."""

    turn_index: int
    speaker: str
    text: str
    reference_audio_path: Path
    output_audio_path: Path

    def __post_init__(self) -> None:
        if self.turn_index <= 0:
            raise ValueError("turn_index must be > 0")
        if not self.speaker.strip():
            raise ValueError("speaker must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")


@runtime_checkable
class VoiceCloneProvider(Protocol):
    """Strategy port for concrete voice-cloning backends."""

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        progress_callback: StageProgressCallback | None = None,
    ) -> Path:
        """
        Synthesize speech for one turn.

        Args:
            reference_audio_path: Speaker reference audio artifact.
            text: Text to synthesize.
            output_audio_path: Target output WAV path.
            progress_callback: Optional callback for synthesis progress updates.

        Returns:
            Path to synthesized audio output.
        """
