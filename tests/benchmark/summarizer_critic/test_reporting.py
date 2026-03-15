"""Reporting coverage for the summarizer-critic benchmark."""

from __future__ import annotations

from pathlib import Path

from card_framework.benchmark.summarizer_critic.config import (
    BenchmarkExecutionReport,
    BenchmarkSummary,
    SampleBenchmarkResult,
    SampleMetricBundle,
)
from card_framework.benchmark.summarizer_critic.reporting import (
    render_sample_results_table,
    write_execution_artifacts,
)


def _sample_result() -> SampleBenchmarkResult:
    return SampleBenchmarkResult(
        sample_id="meeting_a__general_01",
        dataset="qmsum_general_test",
        meeting_id="meeting_a",
        status="completed",
        target_seconds=60,
        candidate_seconds=58.0,
        converged=True,
        iterations_run=3,
        final_status="pass",
        duration_seconds=5.0,
        retrieval_events=0,
        tool_invocations=12,
        metrics=SampleMetricBundle(
            duration_control=1.0,
            factualness=0.8,
            naturalness=0.7,
            speaker_attribution=0.9,
            composite=0.85,
        ),
    )


def test_render_sample_results_table_contains_expected_columns() -> None:
    table = render_sample_results_table([_sample_result()])
    assert "| sample | status |" in table
    assert "| --- | --- |" in table
    assert "meeting_a__general_01" in table
    assert "0.850" in table


def test_write_execution_artifacts_creates_report_and_markdown(tmp_path: Path) -> None:
    report = BenchmarkExecutionReport(
        run_id="run_001",
        generated_at_utc="2026-03-13T00:00:00+00:00",
        status="completed",
        dataset_manifest_path="manifest.json",
        qmsum_repo_dir="QMSum",
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
        commands_executed=[
            "python -m card_framework.benchmark.summarizer_critic execute"
        ],
        git_commit="abc123",
        git_branch="main",
        report_path="",
        notes_path="",
        results_path="",
        samples=[_sample_result()],
        summary=BenchmarkSummary(
            sample_count=1,
            success_count=1,
            failure_count=0,
            mean_duration_control=1.0,
            mean_factualness=0.8,
            mean_naturalness=0.7,
            mean_speaker_attribution=0.9,
            mean_composite=0.85,
        ),
    )
    report_path, notes_path, results_path = write_execution_artifacts(
        run_root=tmp_path,
        report=report,
    )
    assert report_path.exists()
    assert notes_path.exists()
    assert results_path.exists()

    results_text = results_path.read_text(encoding="utf-8")
    assert "## Sample Results" in results_text
    assert "| metric | mean | 95% CI |" in results_text
    assert "n/a" in results_text
    assert "```text" not in results_text
