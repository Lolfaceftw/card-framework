from __future__ import annotations

from card_framework.benchmark.run import _resolve_report_outcome
from card_framework.benchmark.types import BenchmarkAggregate


def _aggregate(*, executed_cells: int, total_samples: int) -> BenchmarkAggregate:
    """Build a minimal aggregate payload for report-outcome tests."""
    return BenchmarkAggregate(
        total_cells=max(executed_cells, 1),
        executed_cells=executed_cells,
        skipped_cells=max(0, 1 - executed_cells),
        total_samples=total_samples,
        passed_samples=0,
        failed_samples=0,
        critic_pass_rate=0.0,
        word_budget_in_range_rate=0.0,
        max_iteration_failure_rate=0.0,
        fallback_verdict_rate=0.0,
        p50_duration_seconds=0.0,
        p95_duration_seconds=0.0,
        alignscore_mean=0.0,
        judge_overall_mean=0.0,
        judge_factuality_mean=0.0,
        judge_order_consistency_rate=0.0,
        judge_repeat_stability_mean=0.0,
        reference_free_coverage_rate=0.0,
    )


def test_resolve_report_outcome_fails_when_no_cells_execute() -> None:
    status, error = _resolve_report_outcome(
        aggregates=_aggregate(executed_cells=0, total_samples=0),
        failures=[{"error": "localhost:8000 refused connection"}],
    )

    assert status == "failed"
    assert error is not None
    assert "did not execute any cells" in error
    assert "refused connection" in error


def test_resolve_report_outcome_fails_when_no_samples_run() -> None:
    status, error = _resolve_report_outcome(
        aggregates=_aggregate(executed_cells=1, total_samples=0),
        failures=[],
    )

    assert status == "failed"
    assert error == "Benchmark executed zero samples. Check the manifest and sample selection."


def test_resolve_report_outcome_completes_when_cells_and_samples_exist() -> None:
    status, error = _resolve_report_outcome(
        aggregates=_aggregate(executed_cells=1, total_samples=2),
        failures=[],
    )

    assert status == "completed"
    assert error is None
