"""ETA estimation strategies and formatting utilities for audio stages."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

AudioStageName: TypeAlias = Literal["separation", "transcription", "diarization"]
EtaDeviceName: TypeAlias = Literal["cpu", "cuda"]
EtaProfileContext: TypeAlias = dict[str, str]


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


@dataclass(slots=True, frozen=True)
class ObservedStageThroughput:
    """Observed stage multiplier and sample count used for adaptive ETA."""

    multiplier: float
    samples: int


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


@runtime_checkable
class StageEtaLearner(Protocol):
    """Optional contract for ETA strategies that learn from observed runtimes."""

    def observe_stage_duration(
        self,
        *,
        stage: AudioStageName,
        audio_duration_ms: int,
        elapsed_seconds: float,
        device: str,
    ) -> None:
        """Update ETA parameters using observed stage runtime."""


@runtime_checkable
class EtaProfilePersistence(Protocol):
    """Optional contract for ETA strategies that persist adaptive profiles."""

    def load_profile(
        self,
        profile_path: Path,
        *,
        context: EtaProfileContext | None = None,
    ) -> None:
        """Load persisted ETA profile from disk."""

    def save_profile(
        self,
        profile_path: Path,
        *,
        context: EtaProfileContext | None = None,
    ) -> None:
        """Persist ETA profile to disk."""


@dataclass(slots=True, frozen=True)
class LinearStageEtaStrategy(StageEtaStrategy, StageEtaLearner, EtaProfilePersistence):
    """Estimate stage runtime from audio duration and per-stage multipliers."""

    separation: StageSpeedProfile
    transcription: StageSpeedProfile
    diarization: StageSpeedProfile
    learning_rate: float = 0.35
    min_multiplier: float = 0.05
    max_multiplier: float = 20.0
    observed_throughput: dict[tuple[AudioStageName, EtaDeviceName], ObservedStageThroughput] = (
        field(default_factory=dict)
    )

    def __post_init__(self) -> None:
        """Validate adaptive ETA parameters."""
        if not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be within [0.0, 1.0].")
        if self.min_multiplier <= 0:
            raise ValueError("min_multiplier must be > 0.")
        if self.max_multiplier < self.min_multiplier:
            raise ValueError("max_multiplier must be >= min_multiplier.")

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
        normalized_device = self._normalize_device(device)
        multiplier = profile.multiplier_for_device(normalized_device)
        observed = self.observed_throughput.get((stage, normalized_device))
        if observed is not None:
            multiplier = observed.multiplier
        multiplier = min(self.max_multiplier, max(self.min_multiplier, multiplier))
        if multiplier <= 0:
            return None
        audio_seconds = audio_duration_ms / 1000.0
        return max(1.0, audio_seconds * multiplier)

    def observe_stage_duration(
        self,
        *,
        stage: AudioStageName,
        audio_duration_ms: int,
        elapsed_seconds: float,
        device: str,
    ) -> None:
        """
        Update stage throughput estimate from observed runtime.

        Args:
            stage: Stage identifier.
            audio_duration_ms: Stage input duration.
            elapsed_seconds: Actual wall-time spent in stage.
            device: Runtime device identifier.
        """
        if audio_duration_ms <= 0 or elapsed_seconds <= 0:
            return
        audio_seconds = audio_duration_ms / 1000.0
        if audio_seconds <= 0:
            return

        observed_multiplier = elapsed_seconds / audio_seconds
        observed_multiplier = min(
            self.max_multiplier,
            max(self.min_multiplier, observed_multiplier),
        )
        normalized_device = self._normalize_device(device)
        key = (stage, normalized_device)
        previous = self.observed_throughput.get(key)
        if previous is None:
            self.observed_throughput[key] = ObservedStageThroughput(
                multiplier=observed_multiplier,
                samples=1,
            )
            return

        updated_multiplier = (
            (1.0 - self.learning_rate) * previous.multiplier
            + self.learning_rate * observed_multiplier
        )
        updated_multiplier = min(
            self.max_multiplier,
            max(self.min_multiplier, updated_multiplier),
        )
        self.observed_throughput[key] = ObservedStageThroughput(
            multiplier=updated_multiplier,
            samples=previous.samples + 1,
        )

    def load_profile(
        self,
        profile_path: Path,
        *,
        context: EtaProfileContext | None = None,
    ) -> None:
        """Load adaptive ETA profile from JSON file when available."""
        self.observed_throughput.clear()
        if not profile_path.exists():
            return
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        expected_context = self._normalize_context(context)
        payload_context = self._normalize_context(payload.get("context", {}))
        if expected_context and payload_context and payload_context != expected_context:
            return
        stages_payload = payload.get("stages", {})
        loaded: dict[tuple[AudioStageName, EtaDeviceName], ObservedStageThroughput] = {}
        valid_stages: tuple[AudioStageName, ...] = (
            "separation",
            "transcription",
            "diarization",
        )
        valid_devices: tuple[EtaDeviceName, ...] = ("cpu", "cuda")
        for stage in valid_stages:
            stage_payload = stages_payload.get(stage, {})
            if not isinstance(stage_payload, dict):
                continue
            for device in valid_devices:
                device_payload = stage_payload.get(device, {})
                if not isinstance(device_payload, dict):
                    continue
                multiplier_value = device_payload.get("multiplier")
                samples_value = device_payload.get("samples")
                try:
                    multiplier = float(multiplier_value)
                    samples = int(samples_value)
                except (TypeError, ValueError):
                    continue
                if samples <= 0:
                    continue
                multiplier = min(self.max_multiplier, max(self.min_multiplier, multiplier))
                loaded[(stage, device)] = ObservedStageThroughput(
                    multiplier=multiplier,
                    samples=samples,
                )
        self.observed_throughput.update(loaded)

    def save_profile(
        self,
        profile_path: Path,
        *,
        context: EtaProfileContext | None = None,
    ) -> None:
        """Persist adaptive ETA profile to JSON file."""
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        valid_stages: tuple[AudioStageName, ...] = (
            "separation",
            "transcription",
            "diarization",
        )
        valid_devices: tuple[EtaDeviceName, ...] = ("cpu", "cuda")
        stages_payload: dict[str, dict[str, dict[str, float | int]]] = {}
        for stage in valid_stages:
            stage_payload: dict[str, dict[str, float | int]] = {}
            for device in valid_devices:
                observed = self.observed_throughput.get((stage, device))
                if observed is None:
                    continue
                stage_payload[device] = {
                    "multiplier": round(observed.multiplier, 6),
                    "samples": observed.samples,
                }
            if stage_payload:
                stages_payload[stage] = stage_payload
        payload = {
            "version": 1,
            "context": self._normalize_context(context),
            "stages": stages_payload,
        }
        tmp_path = profile_path.with_name(f"{profile_path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(profile_path)

    def _normalize_device(self, device: str) -> EtaDeviceName:
        """Normalize arbitrary device labels into ETA profile device keys."""
        return "cuda" if device.strip().lower() == "cuda" else "cpu"

    def _normalize_context(self, context: object) -> EtaProfileContext:
        """Normalize profile context into string key-value pairs."""
        if not isinstance(context, dict):
            return {}
        normalized: EtaProfileContext = {}
        for key, value in context.items():
            key_text = str(key).strip()
            if not key_text:
                continue
            value_text = str(value).strip()
            if not value_text:
                continue
            normalized[key_text] = value_text
        return normalized


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
