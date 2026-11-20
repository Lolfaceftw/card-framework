"""Tests for benchmark watchdog helpers."""

from __future__ import annotations

import json
from pathlib import Path

from card_framework.benchmark.watchdog import (
    GpuMemorySnapshot,
    RunRecord,
    SystemMemorySnapshot,
    TelemetrySample,
    _resolve_command,
    _summarize_report_payload,
    detect_threshold_breaches,
    parse_nvidia_smi_memory_csv,
    render_ascii_results_table,
)


def test_parse_nvidia_smi_memory_csv_reads_multiple_devices() -> None:
    output = "0, RTX 4090, 24564, 6144\n1, RTX 4080, 16376, 2048\n"

    snapshots = parse_nvidia_smi_memory_csv(output)

    assert len(snapshots) == 2
    assert snapshots[0].gpu_index == 0
    assert snapshots[0].name == "RTX 4090"
    assert snapshots[0].used_mebibytes == 6144
    assert snapshots[1].used_gibibytes == 2.0


def test_detect_threshold_breaches_flags_ram_and_gpu() -> None:
    sample = TelemetrySample(
        timestamp_utc="2026-03-13T01:00:00+00:00",
        system_memory=SystemMemorySnapshot(
            total_bytes=100,
            available_bytes=10,
            percent_used=90.0,
        ),
        gpu_memory=(
            GpuMemorySnapshot(
                gpu_index=0,
                name="RTX 4090",
                total_mebibytes=24576,
                used_mebibytes=13 * 1024,
            ),
        ),
    )

    breaches = detect_threshold_breaches(
        sample,
        ram_threshold_percent=80.0,
        gpu_threshold_gibibytes=12.0,
    )

    assert [breach.kind for breach in breaches] == [
        "system_ram_percent",
        "gpu_vram_gibibytes",
    ]


def test_render_ascii_results_table_includes_run_statuses() -> None:
    record = RunRecord(
        run_index=1,
        started_at_utc="2026-03-13T01:00:00+00:00",
        finished_at_utc="2026-03-13T01:01:00+00:00",
        duration_seconds=60.0,
        command=["uv", "run"],
        status="completed",
        exit_code=0,
        peak_system_memory_percent=55.5,
        peak_gpu_memory_gibibytes=3.25,
        report_paths=["artifacts/benchmark/x/benchmark_report.json"],
        kill_reason="",
        stdout_path="stdout.log",
        stderr_path="stderr.log",
        telemetry_path="telemetry.jsonl",
    )

    table = render_ascii_results_table([record])

    assert "| Run | Status | Exit | Seconds | Peak RAM % | Peak GPU GiB | Reports |" in table
    assert "completed" in table
    assert "55.5" in table
    assert "3.25" in table


def test_resolve_command_uses_default_when_empty() -> None:
    command = _resolve_command([])

    assert command[:5] == (
        "uv",
        "run",
        "python",
        "-m",
        "card_framework.benchmark.run",
    )


def test_resolve_command_strips_double_dash() -> None:
    command = _resolve_command(["--", "uv", "run", "python"])

    assert command == ("uv", "run", "python")


def test_summarize_report_payload_for_benchmark_report(tmp_path: Path) -> None:
    report_path = tmp_path / "benchmark_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "aggregates": {
                    "passed_samples": 1,
                    "total_samples": 1,
                    "critic_pass_rate": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_report_payload(report_path)

    assert summary == (
        "benchmark_report.json: status=completed, passed_samples=1/1, "
        "critic_pass_rate=1.000"
    )


def test_summarize_report_payload_for_qa_report(tmp_path: Path) -> None:
    report_path = tmp_path / "qa_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "score": {
                    "score_out_of_100": 82,
                    "summary_grounding_pass_rate": 0.91,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = _summarize_report_payload(report_path)

    assert summary == "qa_report.json: status=completed, score=82/100, grounding=0.910"
