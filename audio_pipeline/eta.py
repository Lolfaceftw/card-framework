"""ETA estimation strategies and formatting utilities for audio stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

AudioStageName: TypeAlias = Literal["separation", "transcription", "diarization"]


@dataclass(slots=True, frozen=True)
class StageSpeedProfile:
    """
    Estimated wall-time multiplier for one stage.

    A multiplier is interpreted as:
    ``estimated_wall_seconds = audio_seconds * multiplier``.
    """

    cpu: float
    cuda: float

    def multiplier_for_device(self, device: str) -> float:
        """Return the configured multiplier for the requested runtime device."""
        if device.strip().lower() == "cuda":
            return self.cuda
        return self.cpu


@runtime_checkable
class StageEtaStrategy(Protocol):
    """Strategy contract for stage-level ETA estimation."""

    def estimate_total_seconds(
        self,
        *,
        stage: AudioStageName,
        audio_duration_ms: int,
        device: str,
    ) -> float | None:
        """Return estimated stage duration in seconds."""


@dataclass(slots=True, frozen=True)
class LinearStageEtaStrategy(StageEtaStrategy):
    """Estimate stage runtime from audio duration and per-stage multipliers."""

    separation: StageSpeedProfile
    transcription: StageSpeedProfile
    diarization: StageSpeedProfile

    def estimate_total_seconds(
        self,
        *,
        stage: AudioStageName,
        audio_duration_ms: int,
        device: str,
    ) -> float | None:
        """
        Estimate stage runtime in seconds.

        Args:
            stage: Stage identifier.
            audio_duration_ms: Input audio duration in milliseconds.
            device: Runtime device (`cpu` or `cuda`).

        Returns:
            Estimated runtime in seconds, or ``None`` when duration is invalid.
        """
        if audio_duration_ms <= 0:
            return None

        profile = {
            "separation": self.separation,
            "transcription": self.transcription,
            "diarization": self.diarization,
        }[stage]
        multiplier = profile.multiplier_for_device(device)
        if multiplier <= 0:
            return None
        audio_seconds = audio_duration_ms / 1000.0
        return max(1.0, audio_seconds * multiplier)


def default_stage_eta_strategy() -> LinearStageEtaStrategy:
    """
    Build default ETA strategy values.

    Defaults are conservative heuristics and can be overridden from config.
    """
    return LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=4.0, cuda=0.8),
        transcription=StageSpeedProfile(cpu=1.5, cuda=0.35),
        diarization=StageSpeedProfile(cpu=2.5, cuda=0.9),
    )


def format_eta_seconds(seconds: float | None) -> str:
    """Render an ETA value into a compact human-readable string."""
    if seconds is None:
        return "unknown"

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes > 0:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"
