"""ETA estimation strategies and formatting utilities for pipeline stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

AudioStageName: TypeAlias = Literal["separation", "transcription", "diarization"]
EtaUnitStageName: TypeAlias = Literal["speaker_samples", "voice_clone"]
EtaDeviceName: TypeAlias = Literal["cpu", "cuda"]
EtaProfileContext: TypeAlias = dict[str, str]
StageProgressCallback: TypeAlias = Callable[["StageProgressUpdate"], None]

_VALID_AUDIO_STAGES: tuple[AudioStageName, ...] = (
    "separation",
    "transcription",
    "diarization",
)
_VALID_UNIT_STAGES: tuple[EtaUnitStageName, ...] = (
    "speaker_samples",
    "voice_clone",
)
_VALID_DEVICES: tuple[EtaDeviceName, ...] = ("cpu", "cuda")
_MANAGED_PROFILE_KEYS = {
    "version",
    "context",
    "stages",
    "audio_stages",
    "unit_stages",
}


@dataclass(slots=True, frozen=True)
class StageSpeedProfile:
    """
    Estimated wall-time multiplier for one audio stage.

    A multiplier is interpreted as:
    ``estimated_wall_seconds = audio_seconds * multiplier``.
    """

    cpu: float
    cuda: float

    def multiplier_for_device(self, device: str) -> float:
        """Return configured multiplier for the requested runtime device."""
        if device.strip().lower() == "cuda":
            return self.cuda
        return self.cpu


@dataclass(slots=True, frozen=True)
class ObservedStageThroughput:
    """Observed audio-stage multiplier and sample count used for adaptive ETA."""

    multiplier: float
    samples: int


@dataclass(slots=True, frozen=True)
class ObservedUnitThroughput:
    """Observed seconds-per-unit and sample count for non-audio unit stages."""

    seconds_per_unit: float
    samples: int


@dataclass(slots=True, frozen=True)
class StageProgressUpdate:
    """
    One progress snapshot emitted by a stage adapter.

    Args:
        completed_units: Number of completed output units (for example, turns/files).
        total_units: Total planned units for the stage.
        processed_audio_ms: Processed audio duration in milliseconds.
        note: Optional descriptive note for diagnostics.
    """

    completed_units: int | None = None
    total_units: int | None = None
    processed_audio_ms: int | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if self.completed_units is not None and self.completed_units < 0:
            raise ValueError("completed_units must be >= 0 when provided.")
        if self.total_units is not None and self.total_units <= 0:
            raise ValueError("total_units must be > 0 when provided.")
        if self.processed_audio_ms is not None and self.processed_audio_ms < 0:
            raise ValueError("processed_audio_ms must be >= 0 when provided.")


@dataclass(slots=True)
class DynamicEtaTracker:
    """
    Runtime ETA tracker that re-estimates completion from live progress signals.

    The tracker combines:
    1. Initial estimate from historical/profile data.
    2. Progress-ratio updates (units completed and/or processed audio).
    3. Elapsed-time overrun correction when no reliable progress is available.
    """

    initial_total_seconds: float
    progress_smoothing: float = 0.25
    overrun_factor: float = 1.15
    headroom_seconds: float = 1.0
    total_audio_ms: int | None = None
    min_progress_ratio: float = 0.01
    current_total_seconds: float = field(init=False)
    latest_progress_ratio: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.initial_total_seconds <= 0:
            raise ValueError("initial_total_seconds must be > 0.")
        if not 0.0 < self.progress_smoothing <= 1.0:
            raise ValueError("progress_smoothing must be within (0.0, 1.0].")
        if self.overrun_factor <= 1.0:
            raise ValueError("overrun_factor must be > 1.0.")
        if self.headroom_seconds < 0:
            raise ValueError("headroom_seconds must be >= 0.")
        if not 0.0 <= self.min_progress_ratio < 1.0:
            raise ValueError("min_progress_ratio must be within [0.0, 1.0).")
        if self.total_audio_ms is not None and self.total_audio_ms <= 0:
            raise ValueError("total_audio_ms must be > 0 when provided.")
        self.current_total_seconds = self.initial_total_seconds

    def observe_progress(self, *, elapsed_seconds: float, update: StageProgressUpdate) -> None:
        """
        Update ETA model from one progress snapshot.

        Args:
            elapsed_seconds: Elapsed wall time since stage start.
            update: Progress payload emitted by a stage.
        """
        elapsed = max(0.0, elapsed_seconds)
        ratio = self._extract_progress_ratio(update=update)
        if ratio is None:
            return
        if ratio < self.latest_progress_ratio:
            ratio = self.latest_progress_ratio
        self.latest_progress_ratio = ratio

        if ratio >= 1.0:
            self.current_total_seconds = max(self.current_total_seconds, elapsed)
            return

        candidate_total = max(
            elapsed + self.headroom_seconds,
            elapsed / ratio,
        )
        self.current_total_seconds = (
            (1.0 - self.progress_smoothing) * self.current_total_seconds
            + self.progress_smoothing * candidate_total
        )
        self.current_total_seconds = max(
            self.current_total_seconds,
            elapsed + self.headroom_seconds,
        )

    def estimate_signed_remaining_seconds(self, *, elapsed_seconds: float) -> float:
        """
        Return signed remaining seconds.

        Negative values mean the stage has overrun the current estimate.
        """
        elapsed = max(0.0, elapsed_seconds)
        signed_remaining = self.current_total_seconds - elapsed
        if signed_remaining < 0:
            self.current_total_seconds = max(
                self.current_total_seconds,
                max(elapsed + self.headroom_seconds, elapsed * self.overrun_factor),
            )
        return signed_remaining

    def estimate_total_seconds(self, *, elapsed_seconds: float) -> float:
        """Return current dynamic estimate of total stage duration."""
        elapsed = max(0.0, elapsed_seconds)
        if elapsed > self.current_total_seconds:
            self.estimate_signed_remaining_seconds(elapsed_seconds=elapsed)
        return max(self.current_total_seconds, elapsed)

    def _extract_progress_ratio(self, *, update: StageProgressUpdate) -> float | None:
        """Extract normalized progress ratio from available update fields."""
        ratios: list[float] = []
        if (
            self.total_audio_ms is not None
            and self.total_audio_ms > 0
            and update.processed_audio_ms is not None
        ):
            ratios.append(update.processed_audio_ms / float(self.total_audio_ms))
        if (
            update.completed_units is not None
            and update.total_units is not None
            and update.total_units > 0
        ):
            ratios.append(update.completed_units / float(update.total_units))
        if not ratios:
            return None
        ratio = min(1.0, max(0.0, max(ratios)))
        if ratio >= 1.0:
            return 1.0
        if ratio < self.min_progress_ratio:
            return None
        return ratio


@runtime_checkable
class StageEtaStrategy(Protocol):
    """Strategy contract for audio-stage ETA estimation."""

    def estimate_total_seconds(
        self,
        *,
        stage: AudioStageName,
        audio_duration_ms: int,
        device: str,
    ) -> float | None:
        """Return estimated stage duration in seconds."""


@runtime_checkable
class UnitStageEtaStrategy(Protocol):
    """Optional strategy contract for non-audio unit-stage ETA estimation."""

    def estimate_unit_stage_total_seconds(
        self,
        *,
        stage: EtaUnitStageName,
        total_units: int,
        ) -> float | None:
        """Return estimated unit-stage duration in seconds."""


@runtime_checkable
class StageEtaHistory(Protocol):
    """Optional contract for strategies that can report learned audio-stage history."""

    def has_stage_history(
        self,
        *,
        stage: AudioStageName,
        device: str,
    ) -> bool:
        """Return whether learned throughput exists for the stage and device."""


@runtime_checkable
class UnitStageEtaHistory(Protocol):
    """Optional contract for strategies that can report learned unit-stage history."""

    def has_unit_stage_history(
        self,
        *,
        stage: EtaUnitStageName,
    ) -> bool:
        """Return whether learned throughput exists for the unit stage."""


@runtime_checkable
class StageEtaLearner(Protocol):
    """Optional contract for strategies that learn from observed audio runtimes."""

    def observe_stage_duration(
        self,
        *,
        stage: AudioStageName,
        audio_duration_ms: int,
        elapsed_seconds: float,
        device: str,
    ) -> None:
        """Update audio-stage ETA parameters using observed runtime."""


@runtime_checkable
class UnitStageEtaLearner(Protocol):
    """Optional contract for strategies that learn from non-audio unit stages."""

    def observe_unit_stage_duration(
        self,
        *,
        stage: EtaUnitStageName,
        total_units: int,
        elapsed_seconds: float,
    ) -> None:
        """Update unit-stage ETA parameters using observed runtime."""


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
class LinearStageEtaStrategy(
    StageEtaStrategy,
    UnitStageEtaStrategy,
    StageEtaHistory,
    UnitStageEtaHistory,
    StageEtaLearner,
    UnitStageEtaLearner,
    EtaProfilePersistence,
):
    """Estimate stage runtime from learned and configured throughput values."""

    separation: StageSpeedProfile
    transcription: StageSpeedProfile
    diarization: StageSpeedProfile
    learning_rate: float = 0.35
    min_multiplier: float = 0.05
    max_multiplier: float = 20.0
    min_unit_seconds_per_unit: float = 0.1
    max_unit_seconds_per_unit: float = 3600.0
    unit_stage_defaults: dict[EtaUnitStageName, float] = field(
        default_factory=lambda: {
            "speaker_samples": 8.0,
            "voice_clone": 20.0,
        }
    )
    observed_throughput: dict[tuple[AudioStageName, EtaDeviceName], ObservedStageThroughput] = (
        field(default_factory=dict)
    )
    observed_unit_throughput: dict[EtaUnitStageName, ObservedUnitThroughput] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate adaptive ETA parameters."""
        if not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be within [0.0, 1.0].")
        if self.min_multiplier <= 0:
            raise ValueError("min_multiplier must be > 0.")
        if self.max_multiplier < self.min_multiplier:
            raise ValueError("max_multiplier must be >= min_multiplier.")
        if self.min_unit_seconds_per_unit <= 0:
            raise ValueError("min_unit_seconds_per_unit must be > 0.")
        if self.max_unit_seconds_per_unit < self.min_unit_seconds_per_unit:
            raise ValueError(
                "max_unit_seconds_per_unit must be >= min_unit_seconds_per_unit."
            )
        for stage_name in _VALID_UNIT_STAGES:
            configured = self.unit_stage_defaults.get(stage_name)
            if configured is None or configured <= 0:
                raise ValueError(
                    f"unit_stage_defaults['{stage_name}'] must be configured and > 0."
                )

    def estimate_total_seconds(
        self,
        *,
        stage: AudioStageName,
        audio_duration_ms: int,
        device: str,
    ) -> float | None:
        """
        Estimate one audio-stage runtime in seconds.

        Args:
            stage: Stage identifier.
            audio_duration_ms: Input audio duration in milliseconds.
            device: Runtime device (``cpu`` or ``cuda``).

        Returns:
            Estimated runtime in seconds, or ``None`` when duration is invalid.
        """
        if audio_duration_ms <= 0:
            return None
        profile = self._audio_profile_for_stage(stage=stage)
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

    def estimate_unit_stage_total_seconds(
        self,
        *,
        stage: EtaUnitStageName,
        total_units: int,
    ) -> float | None:
        """
        Estimate unit-stage runtime from learned seconds-per-unit throughput.

        Args:
            stage: Non-audio stage identifier.
            total_units: Number of planned units in this run.

        Returns:
            Estimated stage duration in seconds, or ``None`` when units are invalid.
        """
        if total_units <= 0:
            return None
        seconds_per_unit = self.unit_stage_defaults.get(stage)
        observed = self.observed_unit_throughput.get(stage)
        if observed is not None:
            seconds_per_unit = observed.seconds_per_unit
        if seconds_per_unit is None or seconds_per_unit <= 0:
            return None
        clamped = min(
            self.max_unit_seconds_per_unit,
            max(self.min_unit_seconds_per_unit, seconds_per_unit),
        )
        return max(1.0, clamped * float(total_units))

    def has_stage_history(
        self,
        *,
        stage: AudioStageName,
        device: str,
    ) -> bool:
        """Return whether learned audio-stage throughput exists for the stage/device."""
        return (stage, self._normalize_device(device)) in self.observed_throughput

    def has_unit_stage_history(
        self,
        *,
        stage: EtaUnitStageName,
    ) -> bool:
        """Return whether learned unit-stage throughput exists for the stage."""
        return stage in self.observed_unit_throughput

    def observe_stage_duration(
        self,
        *,
        stage: AudioStageName,
        audio_duration_ms: int,
        elapsed_seconds: float,
        device: str,
    ) -> None:
        """
        Update audio-stage throughput estimate from observed runtime.

        Args:
            stage: Stage identifier.
            audio_duration_ms: Stage input duration.
            elapsed_seconds: Actual wall time spent in stage.
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

    def observe_unit_stage_duration(
        self,
        *,
        stage: EtaUnitStageName,
        total_units: int,
        elapsed_seconds: float,
    ) -> None:
        """
        Update unit-stage throughput estimate from observed runtime.

        Args:
            stage: Non-audio stage identifier.
            total_units: Number of completed units in this run.
            elapsed_seconds: Wall time spent in the stage.
        """
        if total_units <= 0 or elapsed_seconds <= 0:
            return
        observed_seconds_per_unit = elapsed_seconds / float(total_units)
        observed_seconds_per_unit = min(
            self.max_unit_seconds_per_unit,
            max(self.min_unit_seconds_per_unit, observed_seconds_per_unit),
        )
        previous = self.observed_unit_throughput.get(stage)
        if previous is None:
            self.observed_unit_throughput[stage] = ObservedUnitThroughput(
                seconds_per_unit=observed_seconds_per_unit,
                samples=1,
            )
            return
        updated_seconds_per_unit = (
            (1.0 - self.learning_rate) * previous.seconds_per_unit
            + self.learning_rate * observed_seconds_per_unit
        )
        updated_seconds_per_unit = min(
            self.max_unit_seconds_per_unit,
            max(self.min_unit_seconds_per_unit, updated_seconds_per_unit),
        )
        self.observed_unit_throughput[stage] = ObservedUnitThroughput(
            seconds_per_unit=updated_seconds_per_unit,
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
        self.observed_unit_throughput.clear()
        if not profile_path.exists():
            return
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            return
        expected_context = self._normalize_context(context)
        payload_context = self._normalize_context(payload.get("context", {}))
        if expected_context and payload_context and payload_context != expected_context:
            return

        audio_stages_payload = payload.get("audio_stages")
        if not isinstance(audio_stages_payload, Mapping):
            audio_stages_payload = payload.get("stages", {})
        self.observed_throughput.update(
            self._parse_audio_stage_throughput(audio_stages_payload)
        )

        unit_stages_payload = payload.get("unit_stages", {})
        self.observed_unit_throughput.update(
            self._parse_unit_stage_throughput(unit_stages_payload)
        )

    def save_profile(
        self,
        profile_path: Path,
        *,
        context: EtaProfileContext | None = None,
    ) -> None:
        """Persist adaptive ETA profile to JSON file."""
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        audio_payload = self._serialize_audio_stage_throughput()
        unit_payload = self._serialize_unit_stage_throughput()

        preserved_payload = self._load_preserved_payload(profile_path=profile_path)
        payload = {
            **preserved_payload,
            "version": 2,
            "context": self._normalize_context(context),
            "audio_stages": audio_payload,
            "unit_stages": unit_payload,
            # Keep legacy key for downstream consumers that still expect v1 name.
            "stages": audio_payload,
        }

        tmp_path = profile_path.with_name(f"{profile_path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(profile_path)

    def _audio_profile_for_stage(self, *, stage: AudioStageName) -> StageSpeedProfile:
        """Resolve configured base profile for one audio stage."""
        return {
            "separation": self.separation,
            "transcription": self.transcription,
            "diarization": self.diarization,
        }[stage]

    def _parse_audio_stage_throughput(
        self,
        stages_payload: object,
    ) -> dict[tuple[AudioStageName, EtaDeviceName], ObservedStageThroughput]:
        """Parse serialized audio-stage throughput map."""
        if not isinstance(stages_payload, Mapping):
            return {}
        loaded: dict[tuple[AudioStageName, EtaDeviceName], ObservedStageThroughput] = {}
        for stage in _VALID_AUDIO_STAGES:
            stage_payload = stages_payload.get(stage, {})
            if not isinstance(stage_payload, Mapping):
                continue
            for device in _VALID_DEVICES:
                device_payload = stage_payload.get(device, {})
                if not isinstance(device_payload, Mapping):
                    continue
                multiplier_value = device_payload.get("multiplier")
                samples_value = device_payload.get("samples")
                if multiplier_value is None or samples_value is None:
                    continue
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
        return loaded

    def _parse_unit_stage_throughput(
        self,
        unit_payload: object,
    ) -> dict[EtaUnitStageName, ObservedUnitThroughput]:
        """Parse serialized unit-stage throughput map."""
        if not isinstance(unit_payload, Mapping):
            return {}
        loaded: dict[EtaUnitStageName, ObservedUnitThroughput] = {}
        for stage in _VALID_UNIT_STAGES:
            stage_payload = unit_payload.get(stage, {})
            if not isinstance(stage_payload, Mapping):
                continue
            seconds_per_unit_value = stage_payload.get("seconds_per_unit")
            samples_value = stage_payload.get("samples")
            if seconds_per_unit_value is None or samples_value is None:
                continue
            try:
                seconds_per_unit = float(seconds_per_unit_value)
                samples = int(samples_value)
            except (TypeError, ValueError):
                continue
            if samples <= 0:
                continue
            seconds_per_unit = min(
                self.max_unit_seconds_per_unit,
                max(self.min_unit_seconds_per_unit, seconds_per_unit),
            )
            loaded[stage] = ObservedUnitThroughput(
                seconds_per_unit=seconds_per_unit,
                samples=samples,
            )
        return loaded

    def _serialize_audio_stage_throughput(
        self,
    ) -> dict[str, dict[str, dict[str, float | int]]]:
        """Serialize observed audio-stage throughput into JSON-safe payload."""
        stages_payload: dict[str, dict[str, dict[str, float | int]]] = {}
        for stage in _VALID_AUDIO_STAGES:
            stage_payload: dict[str, dict[str, float | int]] = {}
            for device in _VALID_DEVICES:
                observed = self.observed_throughput.get((stage, device))
                if observed is None:
                    continue
                stage_payload[device] = {
                    "multiplier": round(observed.multiplier, 6),
                    "samples": observed.samples,
                }
            if stage_payload:
                stages_payload[stage] = stage_payload
        return stages_payload

    def _serialize_unit_stage_throughput(self) -> dict[str, dict[str, float | int]]:
        """Serialize observed unit-stage throughput into JSON-safe payload."""
        unit_payload: dict[str, dict[str, float | int]] = {}
        for stage in _VALID_UNIT_STAGES:
            observed = self.observed_unit_throughput.get(stage)
            if observed is None:
                continue
            unit_payload[stage] = {
                "seconds_per_unit": round(observed.seconds_per_unit, 6),
                "samples": observed.samples,
            }
        return unit_payload

    def _load_preserved_payload(self, *, profile_path: Path) -> dict[str, object]:
        """Load existing top-level keys that are not managed by ETA persistence."""
        if not profile_path.exists():
            return {}
        try:
            existing_payload = json.loads(profile_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(existing_payload, Mapping):
            return {}
        preserved: dict[str, object] = {}
        for key, value in existing_payload.items():
            if key in _MANAGED_PROFILE_KEYS:
                continue
            preserved[str(key)] = value
        return preserved

    def _normalize_device(self, device: str) -> EtaDeviceName:
        """Normalize arbitrary device labels into ETA profile device keys."""
        return "cuda" if device.strip().lower() == "cuda" else "cpu"

    def _normalize_context(self, context: object) -> EtaProfileContext:
        """Normalize profile context into string key-value pairs."""
        if not isinstance(context, Mapping):
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
