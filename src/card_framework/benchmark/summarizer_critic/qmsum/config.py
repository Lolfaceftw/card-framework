"""Shared configuration and report contracts for the summarizer-critic benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from card_framework.shared.paths import REPO_ROOT

QMSUM_REPO_URL = "https://github.com/Yale-LILY/QMSum.git"
DEFAULT_REFERENCE_WPM = 150.0
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "summarizer_critic_benchmark"
DEFAULT_DATASET_ROOT = DEFAULT_ARTIFACTS_ROOT / "datasets"
DEFAULT_QMSUM_REPO_DIR = DEFAULT_DATASET_ROOT / "qmsum_official_repo"
DEFAULT_PREPARED_ROOT = DEFAULT_DATASET_ROOT / "qmsum_general_test"
DEFAULT_DATASET_MANIFEST_PATH = DEFAULT_PREPARED_ROOT / "manifest.json"
DEFAULT_RUNS_ROOT = DEFAULT_ARTIFACTS_ROOT / "runs"
DEFAULT_SUPERVISOR_ROOT = DEFAULT_ARTIFACTS_ROOT / "supervisor"
DEFAULT_RAM_THRESHOLD_PERCENT = 80.0
DEFAULT_GPU_THRESHOLD_GIB = 12.0


@dataclass(slots=True, frozen=True)
class PreparedBenchmarkSample:
    """One prepared QMSum general-summary benchmark example."""

    sample_id: str
    dataset: str
    meeting_id: str
    transcript_path: str
    reference_summary: str
    query_text: str
    target_seconds: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PreparedDatasetManifest:
    """Prepared dataset manifest consumed by benchmark execution."""

    manifest_path: Path
    dataset_id: str
    qmsum_repo_dir: Path
    sample_count: int
    samples: tuple[PreparedBenchmarkSample, ...]


@dataclass(slots=True, frozen=True)
class SampleMetricBundle:
    """Scored dimensions for one benchmark sample."""

    duration_control: float
    factualness: float
    naturalness: float
    speaker_attribution: float
    composite: float


@dataclass(slots=True)
class SampleBenchmarkResult:
    """Execution and scoring outcome for one benchmark sample."""

    sample_id: str
    dataset: str
    meeting_id: str
    status: str
    target_seconds: int
    candidate_seconds: float
    converged: bool
    iterations_run: int
    final_status: str
    duration_seconds: float
    retrieval_events: int
    tool_invocations: int
    metrics: SampleMetricBundle
    alignscore: float | None = None
    alignscore_backend: str | None = None
    judge_scores: dict[str, float] | None = None
    reference_free_status: str = "not_scored"
    reference_free_error: str | None = None
    error_message: str | None = None


@dataclass(slots=True, frozen=True)
class BenchmarkSummary:
    """Aggregate benchmark metrics across all executed samples."""

    sample_count: int
    success_count: int
    failure_count: int
    mean_duration_control: float
    mean_factualness: float
    mean_naturalness: float
    mean_speaker_attribution: float
    mean_composite: float


@dataclass(slots=True, frozen=True)
class BenchmarkExecutionReport:
    """Machine-readable top-level report for one benchmark run."""

    run_id: str
    generated_at_utc: str
    status: str
    dataset_manifest_path: str
    qmsum_repo_dir: str
    summarizer_provider: str
    critic_provider: str
    judge_provider: str
    embedding_mode: str
    commands_executed: list[str]
    git_commit: str
    git_branch: str
    report_path: str
    notes_path: str
    results_path: str
    samples: list[SampleBenchmarkResult]
    summary: BenchmarkSummary
    sample_seed: int | None = None


@dataclass(slots=True, frozen=True)
class SupervisedRunRecord:
    """One supervised child benchmark execution."""

    run_index: int
    status: str
    exit_code: int | None
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    peak_system_memory_percent: float
    peak_gpu_memory_gibibytes: float
    report_path: str | None
    notes_path: str | None
    results_path: str | None
    log_path: str
    telemetry_path: str
    kill_reason: str | None = None
