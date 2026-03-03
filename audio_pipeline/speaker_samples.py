"""Speaker-sample planning and generation for post-transcript workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from audio_pipeline.errors import ArtifactWriteError, NonRetryableAudioStageError
from audio_pipeline.runtime import utc_now_iso

SpeakerSampleStatus = Literal["ok", "shorter_than_requested"]
SpeakerSampleSourceMode = Literal["vocals", "source", "auto"]


@dataclass(slots=True, frozen=True)
class AudioSlice:
    """Contiguous source-audio interval represented in milliseconds."""

    start_time_ms: int
    end_time_ms: int

    def __post_init__(self) -> None:
        if self.start_time_ms < 0:
            raise ValueError("start_time_ms must be >= 0")
        if self.end_time_ms < self.start_time_ms:
            raise ValueError("end_time_ms must be >= start_time_ms")

    @property
    def duration_ms(self) -> int:
        """Return slice duration in milliseconds."""
        return self.end_time_ms - self.start_time_ms


@dataclass(slots=True, frozen=True)
class SpeakerSamplePlan:
    """Per-speaker extraction plan for a single output sample file."""

    speaker: str
    requested_duration_ms: int
    actual_duration_ms: int
    status: SpeakerSampleStatus
    slices: tuple[AudioSlice, ...]

    def __post_init__(self) -> None:
        if not self.speaker.strip():
            raise ValueError("speaker must be non-empty")
        if self.requested_duration_ms <= 0:
            raise ValueError("requested_duration_ms must be > 0")
        if self.actual_duration_ms <= 0:
            raise ValueError("actual_duration_ms must be > 0")
        if not self.slices:
            raise ValueError("slices must be non-empty")


@dataclass(slots=True, frozen=True)
class SpeakerSampleArtifact:
    """Generated per-speaker audio artifact with extraction metadata."""

    speaker: str
    path: Path
    requested_duration_ms: int
    actual_duration_ms: int
    status: SpeakerSampleStatus
    ranges_ms: tuple[AudioSlice, ...]


@dataclass(slots=True, frozen=True)
class SpeakerSampleGenerationResult:
    """Result bundle for a complete speaker-sample generation run."""

    output_dir: Path
    manifest_path: Path
    generated_at_utc: str
    artifacts: tuple[SpeakerSampleArtifact, ...]


class ManifestRangePayload(TypedDict):
    """Serialized slice range payload for a manifest entry."""

    start_time_ms: int
    end_time_ms: int


class SpeakerSampleManifestEntry(TypedDict):
    """Serialized per-speaker sample entry in the manifest."""

    speaker: str
    path: str
    requested_duration_ms: int
    duration_ms: int
    status: SpeakerSampleStatus
    ranges_ms: list[ManifestRangePayload]


class SpeakerSampleManifest(TypedDict):
    """Serialized manifest payload describing generated sample artifacts."""

    generated_at_utc: str
    source_audio_path: str
    target_duration_ms: int
    clip_method: Literal["concat_turns"]
    short_speaker_policy: Literal["export_shorter"]
    samples: list[SpeakerSampleManifestEntry]


@runtime_checkable
class SpeakerSampleExporter(Protocol):
    """Port interface for rendering speaker-sample audio files."""

    def export(
        self,
        *,
        source_audio_path: Path,
        slices: Sequence[AudioSlice],
        output_path: Path,
        sample_rate_hz: int,
        channels: int,
    ) -> None:
        """Render one speaker sample from ordered source slices."""


@dataclass(slots=True)
class SpeakerSampleGenerator:
    """
    Generate 30-second speaker samples by concatenating speaker turns.

    Args:
        exporter: Adapter used to render audio files from slice plans.
        target_duration_seconds: Desired sample duration per speaker.
        sample_rate_hz: Output sample rate.
        channels: Output channel count.
        clip_method: Extraction strategy identifier.
        short_speaker_policy: Behavior when speaker has less than target duration.
        manifest_filename: Output manifest file name.
    """

    exporter: SpeakerSampleExporter
    target_duration_seconds: int = 30
    sample_rate_hz: int = 16000
    channels: int = 1
    clip_method: Literal["concat_turns"] = "concat_turns"
    short_speaker_policy: Literal["export_shorter"] = "export_shorter"
    manifest_filename: str = "manifest.json"

    def generate(
        self,
        *,
        transcript_payload: Mapping[str, Any],
        source_audio_path: Path,
        output_dir: Path,
        progress_callback: StageProgressCallback | None = None,
    ) -> SpeakerSampleGenerationResult:
        """
        Generate per-speaker sample files and a machine-readable manifest.

        Args:
            transcript_payload: Normalized transcript payload containing segments.
            source_audio_path: Source audio used for clipping.
            output_dir: Target directory for generated artifacts.
            progress_callback: Optional callback invoked after progress updates.

        Returns:
            Structured generation result containing artifact and manifest paths.

        Raises:
            NonRetryableAudioStageError: If validation or sample export fails.
        """
        self._validate_configuration()

        if not source_audio_path.exists():
            raise NonRetryableAudioStageError(
                f"Speaker sample source audio does not exist: {source_audio_path}"
            )

        target_duration_ms = self.target_duration_seconds * 1000
        plans = build_speaker_sample_plans(
            transcript_payload=transcript_payload,
            target_duration_ms=target_duration_ms,
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        used_filenames: set[str] = set()
        artifacts: list[SpeakerSampleArtifact] = []
        total_plan_count = len(plans)
        if progress_callback is not None:
            try:
                progress_callback(
                    StageProgressUpdate(
                        completed_units=0,
                        total_units=total_plan_count,
                        note="speaker sample generation started",
                    )
                )
            except Exception:
                pass
        for plan in plans:
            output_path = output_dir / _build_unique_filename(
                speaker=plan.speaker,
                used_filenames=used_filenames,
            )
            self.exporter.export(
                source_audio_path=source_audio_path,
                slices=plan.slices,
                output_path=output_path,
                sample_rate_hz=self.sample_rate_hz,
                channels=self.channels,
            )
            artifacts.append(
                SpeakerSampleArtifact(
                    speaker=plan.speaker,
                    path=output_path,
                    requested_duration_ms=plan.requested_duration_ms,
                    actual_duration_ms=plan.actual_duration_ms,
                    status=plan.status,
                    ranges_ms=plan.slices,
                )
            )
            if progress_callback is not None:
                try:
                    progress_callback(
                        StageProgressUpdate(
                            completed_units=len(artifacts),
                            total_units=total_plan_count,
                            note=f"speaker sample exported for {plan.speaker}",
                        )
                    )
                except Exception:
                    pass

        generated_at_utc = utc_now_iso()
        manifest: SpeakerSampleManifest = {
            "generated_at_utc": generated_at_utc,
            "source_audio_path": str(source_audio_path),
            "target_duration_ms": target_duration_ms,
            "clip_method": self.clip_method,
            "short_speaker_policy": self.short_speaker_policy,
            "samples": [
                {
                    "speaker": artifact.speaker,
                    "path": str(artifact.path),
                    "requested_duration_ms": artifact.requested_duration_ms,
                    "duration_ms": artifact.actual_duration_ms,
                    "status": artifact.status,
                    "ranges_ms": [
                        {
                            "start_time_ms": time_range.start_time_ms,
                            "end_time_ms": time_range.end_time_ms,
                        }
                        for time_range in artifact.ranges_ms
                    ],
                }
                for artifact in artifacts
            ],
        }
        manifest_path = output_dir / self.manifest_filename
        _write_json_atomic(payload=manifest, output_path=manifest_path)
        return SpeakerSampleGenerationResult(
            output_dir=output_dir,
            manifest_path=manifest_path,
            generated_at_utc=generated_at_utc,
            artifacts=tuple(artifacts),
        )

    def _validate_configuration(self) -> None:
        """Validate sample-generation runtime configuration."""
        if self.target_duration_seconds <= 0:
            raise ValueError("target_duration_seconds must be > 0")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be > 0")
        if self.channels <= 0:
            raise ValueError("channels must be > 0")
        if self.clip_method != "concat_turns":
            raise ValueError("clip_method must be 'concat_turns'.")
        if self.short_speaker_policy != "export_shorter":
            raise ValueError("short_speaker_policy must be 'export_shorter'.")
        if not self.manifest_filename.strip():
            raise ValueError("manifest_filename must be non-empty")


def build_speaker_sample_plans(
    *,
    transcript_payload: Mapping[str, Any],
    target_duration_ms: int,
) -> list[SpeakerSamplePlan]:
    """
    Build per-speaker concatenation plans from transcript segment timings.

    Args:
        transcript_payload: Transcript payload with `segments`.
        target_duration_ms: Requested duration per speaker sample.

    Returns:
        Ordered speaker plans based on first appearance in transcript.

    Raises:
        NonRetryableAudioStageError: If transcript lacks usable speaker segments.
    """
    if target_duration_ms <= 0:
        raise ValueError("target_duration_ms must be > 0")

    segments = transcript_payload.get("segments", [])
    if not isinstance(segments, list):
        raise NonRetryableAudioStageError(
            "Transcript payload is invalid: 'segments' must be a list."
        )

    speaker_slices: dict[str, list[AudioSlice]] = {}
    normalized_segments = sorted(
        (
            _extract_segment_bounds(segment)
            for segment in segments
            if isinstance(segment, Mapping)
        ),
        key=lambda item: item[1],
    )

    for speaker, start_time, end_time in normalized_segments:
        if end_time <= start_time:
            continue
        speaker_slices.setdefault(speaker, []).append(
            AudioSlice(start_time_ms=start_time, end_time_ms=end_time)
        )

    if not speaker_slices:
        raise NonRetryableAudioStageError(
            "Speaker sample generation requires transcript segments with valid timings."
        )

    plans: list[SpeakerSamplePlan] = []
    for speaker, slices in speaker_slices.items():
        remaining_ms = target_duration_ms
        selected_slices: list[AudioSlice] = []
        for time_slice in slices:
            if remaining_ms <= 0:
                break
            if time_slice.duration_ms <= remaining_ms:
                selected_slices.append(time_slice)
                remaining_ms -= time_slice.duration_ms
                continue
            selected_slices.append(
                AudioSlice(
                    start_time_ms=time_slice.start_time_ms,
                    end_time_ms=time_slice.start_time_ms + remaining_ms,
                )
            )
            remaining_ms = 0

        actual_duration_ms = target_duration_ms - remaining_ms
        if actual_duration_ms <= 0:
            raise NonRetryableAudioStageError(
                f"Speaker '{speaker}' does not contain positive-duration segments."
            )
        status: SpeakerSampleStatus = (
            "ok" if actual_duration_ms == target_duration_ms else "shorter_than_requested"
        )
        plans.append(
            SpeakerSamplePlan(
                speaker=speaker,
                requested_duration_ms=target_duration_ms,
                actual_duration_ms=actual_duration_ms,
                status=status,
                slices=tuple(selected_slices),
            )
        )
    return plans


def resolve_sample_source_audio_path(
    *,
    source_mode: str,
    transcript_metadata: Mapping[str, Any] | None,
    configured_audio_path: str,
    base_dir: Path,
) -> Path:
    """
    Resolve speaker-sample source audio path using configured source preference.

    Args:
        source_mode: One of `vocals`, `source`, or `auto`.
        transcript_metadata: Optional transcript metadata fields.
        configured_audio_path: Optional fallback audio path from config.
        base_dir: Base directory for relative path resolution.

    Returns:
        Resolved existing audio path.

    Raises:
        NonRetryableAudioStageError: If no suitable audio path can be resolved.
    """
    normalized_mode = str(source_mode).strip().lower()
    if normalized_mode not in {"vocals", "source", "auto"}:
        raise ValueError("source_mode must be one of: vocals, source, auto")

    metadata = transcript_metadata or {}
    vocals_path = _optional_resolved_path(
        metadata.get("vocals_audio_path"),
        base_dir=base_dir,
    )
    source_path = _optional_resolved_path(
        metadata.get("source_audio_path"),
        base_dir=base_dir,
    )
    configured_path = _optional_resolved_path(
        configured_audio_path,
        base_dir=base_dir,
    )

    selected_path: Path | None = None
    if normalized_mode == "vocals":
        selected_path = vocals_path
    elif normalized_mode == "source":
        selected_path = source_path or configured_path
    else:
        selected_path = vocals_path or source_path or configured_path

    if selected_path is None:
        raise NonRetryableAudioStageError(
            "Unable to resolve source audio for speaker samples. "
            "Expected transcript metadata path(s) or audio.audio_path fallback."
        )
    if not selected_path.exists():
        raise NonRetryableAudioStageError(
            f"Resolved speaker sample audio path does not exist: {selected_path}"
        )
    return selected_path


def _extract_segment_bounds(segment: Mapping[str, Any]) -> tuple[str, int, int]:
    """Extract and normalize speaker/timing fields from one transcript segment."""
    speaker = str(segment.get("speaker", "")).strip()
    if not speaker:
        raise NonRetryableAudioStageError(
            "Transcript segment is missing a non-empty speaker label."
        )
    start_time = _coerce_milliseconds(segment.get("start_time"), field_name="start_time")
    end_time = _coerce_milliseconds(segment.get("end_time"), field_name="end_time")
    return speaker, start_time, end_time


def _coerce_milliseconds(value: Any, *, field_name: str) -> int:
    """Convert input millisecond values from int/float/string into integer ms."""
    try:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(round(value))
        if isinstance(value, str) and value.strip():
            return int(round(float(value)))
    except ValueError as exc:
        raise NonRetryableAudioStageError(
            f"Transcript segment field '{field_name}' is missing or invalid."
        ) from exc
    raise NonRetryableAudioStageError(
        f"Transcript segment field '{field_name}' is missing or invalid."
    )


def _optional_resolved_path(value: Any, *, base_dir: Path) -> Path | None:
    """Resolve optional path-like values relative to a base directory."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip())
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _sanitize_filename_component(value: str) -> str:
    """Return filesystem-safe token derived from a speaker identifier."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    sanitized = sanitized.strip("._")
    return sanitized or "speaker"


def _build_unique_filename(*, speaker: str, used_filenames: set[str]) -> str:
    """Build deterministic unique output filename for a speaker sample."""
    base_name = _sanitize_filename_component(speaker)
    candidate = f"{base_name}.wav"
    suffix = 1
    while candidate in used_filenames:
        candidate = f"{base_name}_{suffix:02d}.wav"
        suffix += 1
    used_filenames.add(candidate)
    return candidate


def _write_json_atomic(*, payload: object, output_path: Path) -> None:
    """Write JSON payload atomically using a temporary sidecar path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(output_path)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ArtifactWriteError(
            f"Failed to write speaker sample manifest to '{output_path}'."
        ) from exc
