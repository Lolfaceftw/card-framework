"""Artifact rendering for SPoRC benchmark runs."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from card_framework.benchmark.artifacts import write_json_with_hash
from card_framework.benchmark.summarizer_critic.sporc.config import (
    BenchmarkExecutionReport,
    DurationSweepReport,
    DurationSweepPresetSummary,
    SampleBenchmarkResult,
    SupervisedRunRecord,
)


def render_sample_results_table(results: list[SampleBenchmarkResult]) -> str:
    """Render benchmark sample results as a markdown-friendly ASCII table."""
    headers = [
        "sample",
        "status",
        "target_s",
        "cand_s",
        "factual",
        "natural",
        "speaker_grammar",
        "composite",
    ]
    rows = [
        [
            result.sample_id,
            result.status,
            str(result.target_seconds),
            f"{result.candidate_seconds:.1f}",
            f"{result.metrics.factualness:.3f}",
            f"{result.metrics.naturalness:.3f}",
            f"{result.metrics.speaker_grammar_similarity:.3f}",
            f"{result.metrics.composite:.3f}",
        ]
        for result in results
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def _format(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(values)
            )
            + " |"
        )

    divider = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [divider, _format(headers), divider]
    lines.extend(_format(row) for row in rows)
    lines.append(divider)
    return "\n".join(lines)


def render_supervised_results_table(records: list[SupervisedRunRecord]) -> str:
    """Render supervised session outcomes as an ASCII table."""
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
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def _format(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(values)
            )
            + " |"
        )

    divider = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [divider, _format(headers), divider]
    lines.extend(_format(row) for row in rows)
    lines.append(divider)
    return "\n".join(lines)


def render_duration_sweep_table(
    summaries: list[DurationSweepPresetSummary],
) -> str:
    """Render duration sweep aggregates as an ASCII table."""
    headers = [
        "preset",
        "status",
        "examples",
        "executed",
        "skipped",
        "ok",
        "within_tol",
        "mean_cand_s",
        "mean_abs_delta_s",
        "factual",
        "natural",
        "speaker_grammar",
        "composite",
    ]
    rows = [
        [
            summary.duration_preset,
            summary.status,
            str(summary.sample_count),
            str(summary.executed_count),
            str(summary.skipped_count),
            f"{summary.success_count}/{summary.executed_count}",
            f"{summary.within_tolerance_rate:.3f}",
            f"{summary.mean_candidate_seconds:.1f}",
            f"{summary.mean_absolute_duration_delta_seconds:.1f}",
            f"{summary.mean_factualness:.3f}",
            f"{summary.mean_naturalness:.3f}",
            f"{summary.mean_speaker_grammar_similarity:.3f}",
            f"{summary.mean_composite:.3f}",
        ]
        for summary in summaries
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def _format(values: list[str]) -> str:
        return (
            "| "
            + " | ".join(
                value.ljust(widths[index]) for index, value in enumerate(values)
            )
            + " |"
        )

    divider = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [divider, _format(headers), divider]
    lines.extend(_format(row) for row in rows)
    lines.append(divider)
    return "\n".join(lines)


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
        "# SPoRC Summarizer-Critic Benchmark Notes",
        "",
        f"- Run ID: `{report.run_id}`",
        f"- Dataset manifest: `{report.dataset_manifest_path}`",
        f"- SPoRC dataset: `{report.sporc_dataset_id}` ({report.dataset_variant})",
        f"- Summarizer provider: `{report.summarizer_provider}`",
        f"- Critic provider: `{report.critic_provider}`",
        f"- Judge provider: `{report.judge_provider}`",
        f"- Embedding mode: `{report.embedding_mode}`",
        f"- Transcript only: `{report.transcript_only}`",
        f"- Duration tolerance ratio: `{report.duration_tolerance_ratio:.3f}`",
        f"- Sample count: `{report.summary.sample_count}`",
        f"- Executed count: `{report.summary.executed_count}`",
        f"- Skipped count: `{report.summary.skipped_count}`",
        f"- Success count: `{report.summary.success_count}`",
        f"- Failure count: `{report.summary.failure_count}`",
        f"- Mean composite: `{report.summary.mean_composite:.3f}`",
        (
            f"- Duration preset override: `{report.duration_preset}`"
            if report.duration_preset
            else "- Duration preset override: `dataset_default`"
        ),
        (
            f"- Explicit target override seconds: `{report.override_target_seconds}`"
            if report.override_target_seconds is not None
            else "- Explicit target override seconds: `none`"
        ),
        "- `report.json` includes transcript and summary per-speaker message lists for speaker-style auditing.",
        "- When transcript_only is false, `report.json` also records source-audio, speaker-sample, and archived voice-clone artifact paths per sample.",
    ]
    notes_path.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")

    results_lines = [
        "# SPoRC Summarizer-Critic Benchmark Results",
        "",
        "```text",
        render_sample_results_table(report.samples),
        "```",
        "",
        "## Aggregate Metrics",
        "",
        f"- Executed samples: `{report.summary.executed_count}`",
        f"- Skipped samples: `{report.summary.skipped_count}`",
        f"- Mean factualness: `{report.summary.mean_factualness:.3f}`",
        f"- Mean naturalness: `{report.summary.mean_naturalness:.3f}`",
        f"- Mean speaker grammar similarity: `{report.summary.mean_speaker_grammar_similarity:.3f}`",
        f"- Mean composite: `{report.summary.mean_composite:.3f}`",
    ]
    skipped_samples = [sample for sample in report.samples if sample.status == "skipped"]
    if skipped_samples:
        results_lines.extend(["", "## Preflight Skips", ""])
        for sample in skipped_samples:
            results_lines.append(
                f"- `{sample.sample_id}`: `{sample.preflight_reason or sample.error_message or 'preflight_failed'}`"
            )
    results_path.write_text("\n".join(results_lines) + "\n", encoding="utf-8")
    return report_path, notes_path, results_path


def write_duration_sweep_artifacts(
    *,
    sweep_root: Path,
    report: DurationSweepReport,
) -> tuple[Path, Path]:
    """Write machine-readable and markdown artifacts for a duration sweep."""
    sweep_root.mkdir(parents=True, exist_ok=True)
    report_path = sweep_root / "report.json"
    results_path = sweep_root / "results.md"

    write_json_with_hash(report_path, asdict(report))
    results_lines = [
        "# SPoRC Duration Sweep Results",
        "",
        f"- Sweep ID: `{report.sweep_id}`",
        f"- Mode: `{report.mode}`",
        f"- Examples per preset: `{report.examples_per_preset}`",
        f"- Transcript only: `{report.transcript_only}`",
        f"- Summarizer provider: `{report.summarizer_provider}`",
        f"- Critic provider: `{report.critic_provider}`",
        f"- Judge provider: `{report.judge_provider}`",
        f"- Embedding mode: `{report.embedding_mode}`",
        "",
        "```text",
        render_duration_sweep_table(report.preset_summaries),
        "```",
        "",
        "## Per-Preset Artifacts",
        "",
    ]
    for summary in report.preset_summaries:
        results_lines.append(
            f"- `{summary.duration_preset}`: report={summary.report_path or '-'} results={summary.results_path or '-'}"
        )
        if summary.error_message:
            results_lines.append(
                f"- `{summary.duration_preset}` error: `{summary.error_message}`"
            )
    results_path.write_text("\n".join(results_lines) + "\n", encoding="utf-8")
    return report_path, results_path


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
        "# SPoRC Summarizer-Critic Supervisor Notes\n\n"
        + "\n".join(f"- {note}" for note in notes)
        + "\n",
        encoding="utf-8",
    )
    results_path.write_text(
        "# SPoRC Summarizer-Critic Supervisor Results\n\n"
        + "```text\n"
        + render_supervised_results_table(records)
        + "\n```\n",
        encoding="utf-8",
    )
    return session_json_path, notes_path, results_path
