"""Tests for the supervised QA benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from card_framework.benchmark import qa_supervisor


@pytest.mark.integration
def test_supervised_command_kills_process_on_memory_threshold(tmp_path: Path) -> None:
    """Terminate the child when host RAM telemetry breaches the configured limit."""

    def fake_sample(*, gpu_timeout_seconds: float) -> qa_supervisor.TelemetrySample:
        del gpu_timeout_seconds
        return qa_supervisor.TelemetrySample(
            sampled_at_utc="2026-03-13T00:00:00+00:00",
            system_memory=qa_supervisor.SystemMemorySnapshot(
                total_bytes=10,
                available_bytes=1,
                used_bytes=9,
                used_percent=90.0,
            ),
            gpu_devices=(),
        )

    original = qa_supervisor.collect_telemetry_sample
    qa_supervisor.collect_telemetry_sample = fake_sample
    try:
        outcome, _notes = qa_supervisor._supervise_command(
            run_label="threshold_test",
            command=[sys.executable, "-c", "import time; time.sleep(60)"],
            run_root=tmp_path / "threshold_test",
            telemetry_interval_seconds=0.05,
            ram_threshold_percent=80.0,
            gpu_threshold_mib=12_288,
            gpu_timeout_seconds=1.0,
        )
    finally:
        qa_supervisor.collect_telemetry_sample = original

    assert outcome.status == "threshold_killed"
    assert outcome.threshold_reason is not None
    assert "Host RAM reached" in outcome.threshold_reason
    telemetry_lines = (tmp_path / "threshold_test" / "telemetry.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert telemetry_lines


def test_preflight_import_returns_error_detail_when_child_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surface the child import failure detail before looping benchmark runs."""

    class _CompletedProcess:
        returncode = 1
        stdout = ""
        stderr = "ModuleNotFoundError: No module named 'hydra'"

    monkeypatch.setattr(qa_supervisor.subprocess, "run", lambda *args, **kwargs: _CompletedProcess())

    detail = qa_supervisor._preflight_qa_benchmark_import()

    assert detail is not None
    assert "hydra" in detail


def test_render_ascii_table_contains_expected_columns() -> None:
    """Render one fixed-width table row per supervised run."""
    table = qa_supervisor._render_ascii_table(
        [
            qa_supervisor.BenchmarkRunOutcome(
                run_label="benchmark_run_001",
                command=["python", "-m", "card_framework.benchmark.qa"],
                status="failed",
                exit_code=1,
                started_at_utc="2026-03-13T00:00:00+00:00",
                finished_at_utc="2026-03-13T00:01:00+00:00",
                duration_seconds=60.0,
                peak_memory_percent=55.5,
                peak_memory_used_gib=10.0,
                peak_gpu_used_mib=2048,
                threshold_reason="local vllm unavailable",
                benchmark_run_id=None,
                benchmark_score=None,
                report_path=None,
                log_path="run/process.log",
                telemetry_path="run/telemetry.jsonl",
                error_excerpt="error",
            )
        ]
    )

    assert "peak_ram%" in table
    assert "peak_gpu_mib" in table
    assert "benchmark_run_001" in table


def test_write_session_artifacts_emits_results_and_notes(tmp_path: Path) -> None:
    """Persist markdown and JSON session summaries."""
    outcome = qa_supervisor.BenchmarkRunOutcome(
        run_label="benchmark_run_001",
        command=["python", "-m", "card_framework.benchmark.qa"],
        status="completed",
        exit_code=0,
        started_at_utc="2026-03-13T00:00:00+00:00",
        finished_at_utc="2026-03-13T00:01:00+00:00",
        duration_seconds=60.0,
        peak_memory_percent=55.5,
        peak_memory_used_gib=10.0,
        peak_gpu_used_mib=2048,
        threshold_reason=None,
        benchmark_run_id="run-123",
        benchmark_score=88,
        report_path="qa_runs/run-123/qa_report.json",
        log_path="run/process.log",
        telemetry_path="run/telemetry.jsonl",
        error_excerpt=None,
    )

    qa_supervisor._write_session_artifacts(
        session_root=tmp_path,
        outcomes=[outcome],
        session_started_at_utc="2026-03-13T00:00:00+00:00",
        notes=["run completed cleanly"],
    )

    session_payload = json.loads((tmp_path / "session.json").read_text(encoding="utf-8"))
    assert session_payload["runs"][0]["benchmark_score"] == 88
    results_md = (tmp_path / "results.md").read_text(encoding="utf-8")
    notes_md = (tmp_path / "notes.md").read_text(encoding="utf-8")
    assert "QA Supervisor Results" in results_md
    assert "run completed cleanly" in notes_md
