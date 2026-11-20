"""Metric aggregation utilities for benchmark reporting."""

from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

from card_framework.benchmark.types import BenchmarkAggregate, CellRunResult


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return round(sorted_values[lower], 4)
    weight = rank - lower
    return round(
        sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight,
        4,
    )


def _mean(values: list[float]) -> float:
    """Return arithmetic mean rounded to 4 decimals, defaulting to 0.0."""
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def aggregate_results(results: list[CellRunResult]) -> BenchmarkAggregate:
    """Compute aggregate benchmark KPIs from per-cell results."""
    total_cells = len(results)
    executed_cells = sum(1 for cell in results if cell.status != "skipped")
    skipped_cells = total_cells - executed_cells

    sample_results = [
        sample
        for cell in results
        if cell.status != "skipped"
        for sample in cell.sample_results
    ]
    total_samples = len(sample_results)

    passed_samples = sum(1 for sample in sample_results if sample.status == "pass")
    failed_samples = total_samples - passed_samples
    word_budget_in_range = sum(
        1 for sample in sample_results if sample.word_budget_in_range
    )
    max_iteration_failures = sum(
        1 for sample in sample_results if sample.final_status == "max_iterations_reached"
    )
    fallback_verdict_count = sum(
        1 for sample in sample_results if sample.fallback_verdict_used
    )
    reference_free_scored = sum(
        1
        for sample in sample_results
        if sample.reference_free_status in {"ok", "partial"}
    )

    durations = [sample.duration_seconds for sample in sample_results]
    alignscores = [
        sample.alignscore
        for sample in sample_results
        if sample.alignscore is not None
    ]
    judge_overall_scores = [
        sample.judge_scores.get("overall")
        for sample in sample_results
        if sample.judge_scores and sample.judge_scores.get("overall") is not None
    ]
    judge_factuality_scores = [
        sample.judge_scores.get("factuality")
        for sample in sample_results
        if sample.judge_scores and sample.judge_scores.get("factuality") is not None
    ]
    order_flags = [
        sample.judge_order_consistent
        for sample in sample_results
        if sample.judge_order_consistent is not None
    ]
    repeat_deltas = [
        sample.judge_repeat_delta
        for sample in sample_results
        if sample.judge_repeat_delta is not None
    ]

    return BenchmarkAggregate(
        total_cells=total_cells,
        executed_cells=executed_cells,
        skipped_cells=skipped_cells,
        total_samples=total_samples,
        passed_samples=passed_samples,
        failed_samples=failed_samples,
        critic_pass_rate=_safe_ratio(passed_samples, total_samples),
        word_budget_in_range_rate=_safe_ratio(word_budget_in_range, total_samples),
        max_iteration_failure_rate=_safe_ratio(max_iteration_failures, total_samples),
        fallback_verdict_rate=_safe_ratio(fallback_verdict_count, total_samples),
        p50_duration_seconds=_percentile(durations, 0.5),
        p95_duration_seconds=_percentile(durations, 0.95),
        alignscore_mean=_mean([float(score) for score in alignscores]),
        judge_overall_mean=_mean([float(score) for score in judge_overall_scores]),
        judge_factuality_mean=_mean(
            [float(score) for score in judge_factuality_scores]
        ),
        judge_order_consistency_rate=_safe_ratio(
            sum(1 for flag in order_flags if flag),
            len(order_flags),
        ),
        judge_repeat_stability_mean=_mean([float(delta) for delta in repeat_deltas]),
        reference_free_coverage_rate=_safe_ratio(reference_free_scored, total_samples),
    )


def parse_junit_totals(junit_path: Path) -> dict[str, int]:
    """Parse JUnit XML totals safely; return zeros when unavailable."""
    if not junit_path.exists():
        return {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}

    try:
        root = ET.fromstring(junit_path.read_text(encoding="utf-8"))
    except Exception:
        return {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}

    if root.tag == "testsuite":
        node = root
    else:
        suites = root.findall("testsuite")
        if not suites:
            return {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
        node = suites[0]

    def _as_int(key: str) -> int:
        raw = node.attrib.get(key, "0")
        try:
            return int(raw)
        except ValueError:
            return 0

    return {
        "tests": _as_int("tests"),
        "failures": _as_int("failures"),
        "errors": _as_int("errors"),
        "skipped": _as_int("skipped"),
    }

