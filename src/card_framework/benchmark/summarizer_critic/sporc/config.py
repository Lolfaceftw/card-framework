"""Shared configuration and report contracts for the SPoRC benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from card_framework.shared.paths import REPO_ROOT

SPORC_DATASET_ID = "blitt/SPoRC"
DEFAULT_REFERENCE_WPM = 150.0
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "summarizer_critic_benchmark"
DEFAULT_DATASET_ROOT = DEFAULT_ARTIFACTS_ROOT / "datasets"
DEFAULT_SPORC_ROOT = DEFAULT_DATASET_ROOT / "sporc"
DEFAULT_PREPARED_ROOT = DEFAULT_SPORC_ROOT / "prepared_sample"
DEFAULT_PREPARED_FULL_ROOT = DEFAULT_SPORC_ROOT / "prepared_full"
DEFAULT_DATASET_MANIFEST_PATH = DEFAULT_PREPARED_ROOT / "manifest.json"
DEFAULT_FULL_DATASET_MANIFEST_PATH = DEFAULT_PREPARED_FULL_ROOT / "manifest.json"
DEFAULT_LONGFORM_MANIFEST_PATH = (
    DEFAULT_PREPARED_FULL_ROOT / "benchmark_longform_15k_manifest.json"
)
DEFAULT_LONGFORM_AUDIT_PATH = (
    DEFAULT_PREPARED_FULL_ROOT / "benchmark_longform_15k_audit.json"
)
DEFAULT_RUNS_ROOT = DEFAULT_ARTIFACTS_ROOT / "sporc_runs"
DEFAULT_SUPERVISOR_ROOT = DEFAULT_ARTIFACTS_ROOT / "sporc_supervisor"
DEFAULT_SWEEP_ROOT = DEFAULT_ARTIFACTS_ROOT / "sweeps"
DEFAULT_RAM_THRESHOLD_PERCENT = 80.0
DEFAULT_GPU_THRESHOLD_GIB = 12.0
DEFAULT_DATASET_VARIANT = "sample"
LONGFORM_MIN_SPOKEN_WORDS = 15_000
LONGFORM_MIN_SEGMENTS = 120
LONGFORM_MIN_SPEAKERS = 2
LONGFORM_MIN_SPEAKER_WORDS = 1_000
LONGFORM_MIN_SPEAKER_SEGMENTS = 20
LONGFORM_MAX_DOMINANT_SPEAKER_SHARE = 0.90
LONGFORM_MAX_NON_LEXICAL_RATIO = 0.10

DURATION_PRESET_SECONDS = {
    "30s": 30,
    "1m": 60,
    "5m": 300,
    "15m": 900,
}

SPORC_VARIANT_FILES = {
    "sample": {
        "episode": "episodeLevelDataSample.jsonl.gz",
        "turn": "speakerTurnDataSample.jsonl.gz",
    },
    "full": {
        "episode": "episodeLevelData.jsonl.gz",
        "turn": "speakerTurnData.jsonl.gz",
    },
}


@dataclass(slots=True, frozen=True)
class PreparedBenchmarkSample:
    """One prepared SPoRC benchmark example."""

    sample_id: str
    dataset: str
    episode_key: str
    transcript_path: str
    reference_summary: str
    target_seconds: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class PreparedDatasetManifest:
    """Prepared dataset manifest consumed by benchmark execution."""

    manifest_path: Path
    dataset_id: str
    sporc_dataset_id: str
    dataset_variant: str
    episode_data_path: Path
    turn_data_path: Path
    sample_count: int
    samples: tuple[PreparedBenchmarkSample, ...] = field(default_factory=tuple)
    samples_path: Path | None = None


@dataclass(slots=True, frozen=True)
class SampleMetricBundle:
    """Scored dimensions for one SPoRC benchmark sample."""

    factualness: float
    naturalness: float
    speaker_grammar_similarity: float
    composite: float


@dataclass(slots=True)
class SampleBenchmarkResult:
    """Execution and scoring outcome for one SPoRC benchmark sample."""

    sample_id: str
    dataset: str
    episode_key: str
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
    source_audio_path: str | None = None
    speaker_samples_manifest_path: str | None = None
    draft_audio_state_path: str | None = None
    voice_clone_output_dir: str | None = None
    voice_clone_manifest_path: str | None = None
    voice_clone_merged_audio_path: str | None = None
    transcript_per_speaker_messages: dict[str, list[str]] = field(default_factory=dict)
    summary_per_speaker_messages: dict[str, list[str]] = field(default_factory=dict)
    speaker_alias_map: dict[str, str] = field(default_factory=dict)
    speaker_notes: list[str] = field(default_factory=list)
    judge_scores: dict[str, float] | None = None
    judge_status: str = "not_scored"
    judge_error: str | None = None
    error_message: str | None = None
    preflight_reason: str | None = None
    preflight_stats: dict[str, int | float | str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class BenchmarkSummary:
    """Aggregate benchmark metrics across selected and executed samples."""

    sample_count: int
    executed_count: int
    skipped_count: int
    success_count: int
    failure_count: int
    mean_factualness: float
    mean_naturalness: float
    mean_speaker_grammar_similarity: float
    mean_composite: float


@dataclass(slots=True, frozen=True)
class BenchmarkExecutionReport:
    """Machine-readable top-level report for one SPoRC benchmark run."""

    run_id: str
    generated_at_utc: str
    status: str
    dataset_manifest_path: str
    sporc_dataset_id: str
    dataset_variant: str
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
    duration_tolerance_ratio: float = 0.0
    transcript_only: bool = False
    duration_preset: str = ""
    override_target_seconds: int | None = None


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


@dataclass(slots=True, frozen=True)
class DurationSweepPresetSummary:
    """Aggregate results for one duration preset inside a sweep."""

    duration_preset: str
    target_seconds: int
    status: str
    sample_count: int
    executed_count: int
    skipped_count: int
    success_count: int
    failure_count: int
    within_tolerance_count: int
    within_tolerance_rate: float
    mean_candidate_seconds: float
    mean_absolute_duration_delta_seconds: float
    mean_factualness: float
    mean_naturalness: float
    mean_speaker_grammar_similarity: float
    mean_composite: float
    report_path: str | None
    notes_path: str | None
    results_path: str | None
    error_message: str | None = None


@dataclass(slots=True, frozen=True)
class DurationSweepReport:
    """Top-level artifact contract for one SPoRC duration sweep."""

    sweep_id: str
    generated_at_utc: str
    mode: str
    examples_per_preset: int
    transcript_only: bool
    summarizer_provider: str
    critic_provider: str
    judge_provider: str
    embedding_mode: str
    commands_executed: list[str]
    preset_summaries: list[DurationSweepPresetSummary]
    report_path: str
    results_path: str
