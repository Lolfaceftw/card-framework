"""Shared type definitions for diarization benchmark execution and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class DiarizationBenchmarkSample:
    """One diarization benchmark sample entry loaded from a manifest."""

    sample_id: str
    dataset: str
    audio_path: str
    reference_rttm_path: str
    subset: str = ""
    uem_path: str | None = None
    num_speakers: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DiarizationSampleRunResult:
    """Result payload for one sample execution for one diarizer provider."""

    sample_id: str
    dataset: str
    subset: str
    provider_id: str
    status: str
    duration_seconds: float
    audio_duration_seconds: float | None
    real_time_factor: float | None
    predicted_turn_count: int
    predicted_rttm_path: str
    der: float | None = None
    jer: float | None = None
    peak_gpu_memory_mb: float | None = None
    error_message: str | None = None


@dataclass(slots=True)
class DiarizationProviderAggregate:
    """Top-level aggregate KPIs for one diarization provider."""

    provider_id: str
    total_samples: int
    passed_samples: int
    failed_samples: int
    mean_der: float | None
    mean_jer: float | None
    mean_duration_seconds: float | None
    mean_real_time_factor: float | None
    max_peak_gpu_memory_mb: float | None


@dataclass(slots=True)
class DiarizationBenchmarkReport:
    """Serializable diarization benchmark report."""

    run_id: str
    status: str
    generated_at_utc: str
    git_commit: str
    git_branch: str
    manifest_path: str
    config_path: str
    device: str
    providers: list[str]
    commands_executed: list[str]
    results: list[DiarizationSampleRunResult]
    aggregates: list[DiarizationProviderAggregate]
    failures: list[dict[str, Any]]
