"""Artifact rendering for summarizer-critic benchmark runs."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from card_framework.benchmark.artifacts import write_json_with_hash
from card_framework.benchmark.summarizer_critic.config import (
    BenchmarkExecutionReport,
    SampleBenchmarkResult,
    SupervisedRunRecord,
)
from card_framework.benchmark.summarizer_critic.reporting_utils import (
    format_confidence_interval,
    mean_confidence_interval_95,
    render_markdown_table,
)


def render_sample_results_table(results: list[SampleBenchmarkResult]) -> str:
    """Render benchmark sample results as a Markdown table."""
    headers = [
        "sample",
        "status",
        "target_s",
        "cand_s",
        "duration",
        "factual",
        "natural",
        "speaker",
        "composite",
    ]
    rows = [
        [
            result.sample_id,
            result.status,
            str(result.target_seconds),
            f"{result.candidate_seconds:.1f}",
            f"{result.metrics.duration_control:.3f}",
            f"{result.metrics.factualness:.3f}",
            f"{result.metrics.naturalness:.3f}",
            f"{result.metrics.speaker_attribution:.3f}",
            f"{result.metrics.composite:.3f}",
        ]
        for result in results
    ]
    return render_markdown_table(headers, rows)


def render_supervised_results_table(records: list[SupervisedRunRecord]) -> str:
    """Render supervised session outcomes as a Markdown table."""
    headers = ["run", "status", "exit", "secs", "peak_ram%", "peak_gpu_gib", "report"]
    rows = [
        [
            str(record.run_index),
            record.status,
            "-" if record.exit_code is None else str(record.exit_code),
            f"{record.duration_seconds:.1f}",
            f"{record.peak_system_memory_percent:.1f}",
            f"{record.peak_gpu_memory_gibibytes:.2f}",
            "-" if not record.report_path else Path(record.report_path).name,
        ]
        for record in records
    ]
    return render_markdown_table(headers, rows)


def _render_aggregate_metrics_table(report: BenchmarkExecutionReport) -> str:
    """Render mean benchmark metrics plus 95% confidence intervals."""
    return render_markdown_table(
        ["metric", "mean", "95% CI"],
        [
            [
                "duration_control",
                f"{report.summary.mean_duration_control:.3f}",
                format_confidence_interval(
                    mean_confidence_interval_95(
                        [result.metrics.duration_control for result in report.samples]
                    )
                ),
            ],
            [
                "factualness",
                f"{report.summary.mean_factualness:.3f}",
                format_confidence_interval(
                    mean_confidence_interval_95(
                        [result.metrics.factualness for result in report.samples]
                    )
                ),
            ],
            [
                "naturalness",
                f"{report.summary.mean_naturalness:.3f}",
                format_confidence_interval(
                    mean_confidence_interval_95(
                        [result.metrics.naturalness for result in report.samples]
                    )
                ),
            ],
            [
                "speaker_attribution",
                f"{report.summary.mean_speaker_attribution:.3f}",
                format_confidence_interval(
                    mean_confidence_interval_95(
                        [
                            result.metrics.speaker_attribution
                            for result in report.samples
                        ]
                    )
                ),
            ],
            [
                "composite",
                f"{report.summary.mean_composite:.3f}",
                format_confidence_interval(
                    mean_confidence_interval_95(
                        [result.metrics.composite for result in report.samples]
                    )
                ),
            ],
        ],
    )


def write_execution_artifacts(
    *,
    run_root: Path,
    report: BenchmarkExecutionReport,
) -> tuple[Path, Path, Path]:
    """Write the machine-readable report and markdown summaries."""
    run_root.mkdir(parents=True, exist_ok=True)
    report_path = run_root / "report.json"
    notes_path = run_root / "notes.md"
    results_path = run_root / "results.md"

    write_json_with_hash(report_path, asdict(report))
    notes_lines = [
        "# Summarizer-Critic Benchmark Notes",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Dataset manifest: `{report.dataset_manifest_path}`",
        f"- QMSum repo: `{report.qmsum_repo_dir}`",
        f"- Summarizer provider: `{report.summarizer_provider}`",
        f"- Critic provider: `{report.critic_provider}`",
        f"- Judge provider: `{report.judge_provider}`",
        f"- Embedding mode: `{report.embedding_mode}`",
        f"- Success count: `{report.summary.success_count}/{report.summary.sample_count}`",
        f"- Mean composite: `{report.summary.mean_composite:.3f}`",
    ]
    notes_path.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")

    results_lines = [
        "# Summarizer-Critic Benchmark Results",
        "",
        "## Sample Results",
        "",
        render_sample_results_table(report.samples),
        "",
        "## Aggregate Metrics",
        "",
        _render_aggregate_metrics_table(report),
    ]
    results_path.write_text("\n".join(results_lines) + "\n", encoding="utf-8")
    return report_path, notes_path, results_path


def write_supervisor_session_artifacts(
    *,
    session_root: Path,
    notes: list[str],
    records: list[SupervisedRunRecord],
) -> tuple[Path, Path, Path]:
    """Write machine-readable and markdown artifacts for a supervised session."""
    session_root.mkdir(parents=True, exist_ok=True)
    session_json_path = session_root / "session.json"
    notes_path = session_root / "notes.md"
    results_path = session_root / "results.md"
    session_json_path.write_text(
        json.dumps(
            {
                "notes": notes,
                "runs": [asdict(record) for record in records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    notes_path.write_text(
        "# Summarizer-Critic Supervisor Notes\n\n"
        + "\n".join(f"- {note}" for note in notes)
        + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        "# Summarizer-Critic Supervisor Results\n\n"
        + render_supervised_results_table(records)
        + "\n",
        encoding="utf-8",
    )
    return session_json_path, notes_path, results_path
