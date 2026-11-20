"""Shared type definitions for benchmark execution and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class BenchmarkSample:
    """One benchmark transcript sample entry loaded from a manifest."""

    sample_id: str
    dataset: str
    transcript_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class ProviderProfile:
    """Provider profile describing one LLM backend configuration."""

    provider_id: str
    description: str
    llm_config: dict[str, Any]
    required_env: list[str] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class EmbeddingProfile:
    """Embedding profile describing one embedding backend configuration."""

    embedding_id: str
    description: str
    embedding_config: dict[str, Any]


@dataclass(slots=True, frozen=True)
class BenchmarkPreset:
    """Preset controlling matrix size and budget behavior."""

    name: str
    target_duration_seconds: int
    estimated_sample_seconds: int
    max_samples: int
    repeats: int
    include_embedding_profiles: list[str]


@dataclass(slots=True, frozen=True)
class BenchmarkCell:
    """One executable matrix cell."""

    cell_id: str
    provider_id: str
    embedding_id: str
    repeat_index: int
    llm_config: dict[str, Any]
    embedding_config: dict[str, Any]


@dataclass(slots=True)
class SampleRunResult:
    """Result payload for one sample execution inside a matrix cell."""

    sample_id: str
    dataset: str
    status: str
    duration_seconds: float
    converged: bool
    iterations_run: int
    final_status: str
    final_word_count: int
    word_budget_in_range: bool
    fallback_verdict_used: bool
    retrieval_events: int
    tool_invocations: int
    alignscore: float | None = None
    alignscore_backend: str | None = None
    judge_scores: dict[str, float] | None = None
    judge_pairwise_winner: str | None = None
    judge_order_consistent: bool | None = None
    judge_repeat_delta: float | None = None
    reference_free_status: str = "not_scored"
    reference_free_error: str | None = None
    failure_category: str | None = None
    error_message: str | None = None


@dataclass(slots=True)
class CellRunResult:
    """Aggregated result for one matrix cell."""

    cell_id: str
    provider_id: str
    embedding_id: str
    repeat_index: int
    status: str
    sample_results: list[SampleRunResult]
    skipped_reason: str | None = None


@dataclass(slots=True)
class BenchmarkAggregate:
    """Top-level aggregate KPIs across all executed cell/sample results."""

    total_cells: int
    executed_cells: int
    skipped_cells: int
    total_samples: int
    passed_samples: int
    failed_samples: int
    critic_pass_rate: float
    word_budget_in_range_rate: float
    max_iteration_failure_rate: float
    fallback_verdict_rate: float
    p50_duration_seconds: float
    p95_duration_seconds: float
    alignscore_mean: float
    judge_overall_mean: float
    judge_factuality_mean: float
    judge_order_consistency_rate: float
    judge_repeat_stability_mean: float
    reference_free_coverage_rate: float


@dataclass(slots=True)
class BenchmarkReport:
    """Serializable benchmark report."""

    run_id: str
    status: str
    generated_at_utc: str
    git_commit: str
    git_branch: str
    preset: str
    manifest_path: str
    provider_profiles_path: str
    commands_executed: list[str]
    matrix: list[dict[str, Any]]
    results: list[CellRunResult]
    aggregates: BenchmarkAggregate
    failures: list[dict[str, Any]]
