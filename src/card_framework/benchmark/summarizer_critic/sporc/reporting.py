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
        "counted",
        "replacement",
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
            "yes" if result.counts_toward_target else "no",
            result.replacement_reason or "-",
            str(result.target_seconds),
            f"{result.candidate_seconds:.1f}",
            f"{result.metrics.factualness:.3f}",
            f"{result.metrics.naturalness:.3f}",
            f"{result.metrics.speaker_grammar_similarity:.3f}",
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


def render_duration_sweep_table(
    summaries: list[DurationSweepPresetSummary],
) -> str:
    """Render duration sweep aggregates as a Markdown table."""
    headers = [
        "preset",
        "status",
        "examples",
        "attempted",
        "replaced",
        "executed",
        "skipped",
        "restarts",
        "recovered_stalls",
        "ok",
        "within_tol",
        "mean_cand_s",
        "mean_abs_delta_s",
        "factual",
        "natural",
        "speaker_grammar",
        "composite",
        "composite_95_ci",
    ]
    rows = [
        [
            summary.duration_preset,
            summary.status,
            str(summary.sample_count),
            str(summary.attempted_candidate_count),
            str(summary.replacement_count),
            str(summary.executed_count),
            str(summary.skipped_count),
            str(summary.restart_count),
            str(summary.recovered_stall_count),
            f"{summary.success_count}/{summary.executed_count}",
            f"{summary.within_tolerance_rate:.3f}",
            f"{summary.mean_candidate_seconds:.1f}",
            f"{summary.mean_absolute_duration_delta_seconds:.1f}",
            f"{summary.mean_factualness:.3f}",
            f"{summary.mean_naturalness:.3f}",
            f"{summary.mean_speaker_grammar_similarity:.3f}",
            f"{summary.mean_composite:.3f}",
            format_confidence_interval(
                (
                    summary.mean_composite_ci95_lower,
                    summary.mean_composite_ci95_upper,
                )
                if summary.mean_composite_ci95_lower is not None
                and summary.mean_composite_ci95_upper is not None
                else None
            ),
        ]
        for summary in summaries
    ]
    return render_markdown_table(headers, rows)


def _render_aggregate_metrics_table(report: BenchmarkExecutionReport) -> str:
    """Render executed-sample means plus 95% confidence intervals."""
    executed_samples = [
        sample
        for sample in report.samples
        if sample.counts_toward_target and sample.status != "skipped"
    ]
    return render_markdown_table(
        ["metric", "mean", "95% CI"],
        [
            [
                "factualness",
                f"{report.summary.mean_factualness:.3f}",
                format_confidence_interval(
                    mean_confidence_interval_95(
                        [sample.metrics.factualness for sample in executed_samples]
                    )
                ),
            ],
            [
                "naturalness",
                f"{report.summary.mean_naturalness:.3f}",
                format_confidence_interval(
                    mean_confidence_interval_95(
                        [sample.metrics.naturalness for sample in executed_samples]
                    )
                ),
            ],
            [
                "speaker_grammar_similarity",
                f"{report.summary.mean_speaker_grammar_similarity:.3f}",
                format_confidence_interval(
                    mean_confidence_interval_95(
                        [
                            sample.metrics.speaker_grammar_similarity
                            for sample in executed_samples
                        ]
                    )
                ),
            ],
            [
                "composite",
                f"{report.summary.mean_composite:.3f}",
                format_confidence_interval(
                    mean_confidence_interval_95(
                        [sample.metrics.composite for sample in executed_samples]
                    )
                ),
            ],
        ],
    )


def _write_sample_artifact_navigation(
    *,
    report: BenchmarkExecutionReport,
) -> None:
    """Write a developer-facing sample index and README under the samples root."""
    if not report.sample_artifacts_root:
        return
    samples_root = Path(report.sample_artifacts_root)
    samples_root.mkdir(parents=True, exist_ok=True)
    index_path = samples_root / "index.json"
    readme_path = samples_root / "README.md"
    index_payload = {
        "run_id": report.run_id,
        "generated_at_utc": report.generated_at_utc,
        "samples_root": str(samples_root),
        "samples": [
            {
                "sample_id": sample.sample_id,
                "status": sample.status,
                "final_status": sample.final_status,
                "sample_artifact_dir": sample.sample_artifact_dir,
                "artifact_manifest_path": sample.artifact_manifest_path,
                "final_outputs": {
                    "summary_xml": sample.final_summary_path,
                    "critic_feedback": sample.final_critic_feedback_path,
                    "loop_diagnostics": sample.loop_diagnostics_path,
                    "sample_result": sample.sample_result_artifact_path,
                },
                "llm_trace_dirs": {
                    "summarizer": sample.summarizer_trace_dir,
                    "critic": sample.critic_trace_dir,
                    "judge": sample.judge_trace_dir,
                },
            }
            for sample in report.samples
            if sample.sample_artifact_dir
        ],
    }
    write_json_with_hash(index_path, index_payload)

    readme_lines = [
        "# SPoRC Sample Artifacts",
        "",
        "Each sample folder mirrors one benchmark sample executed in this run.",
        "",
        "## Layout",
        "",
        "- `<sample_id>/manifest.json`: sample-local index of final outputs and raw LLM call files.",
        "- `<sample_id>/final/summary.xml`: final summarizer output persisted for that sample.",
        "- `<sample_id>/final/critic_feedback.txt`: final critic feedback captured from loop diagnostics when present.",
        "- `<sample_id>/final/loop_diagnostics.json`: full loop diagnostics payload from the orchestrator.",
        "- `<sample_id>/final/sample_result.json`: the same sample result payload that appears in `report.json`, with artifact paths resolved.",
        "- `<sample_id>/llm/summarizer/call_XXX.json`: raw summarizer request, streamed reasoning/content, and final response payload.",
        "- `<sample_id>/llm/critic/call_XXX.json`: raw critic request, streamed reasoning/content, and final response payload.",
        "- `<sample_id>/llm/judge/call_XXX.json`: raw judge request, streamed reasoning/content, and final response payload.",
        "",
        f"Run-level index: `{index_path}`",
        "",
        "## Samples",
        "",
    ]
    indexed_samples = [sample for sample in report.samples if sample.sample_artifact_dir]
    if not indexed_samples:
        readme_lines.append("- No executed sample artifact folders were written for this run.")
    else:
        for sample in indexed_samples:
            sample_dir = Path(sample.sample_artifact_dir).name
            readme_lines.append(
                f"- `{sample.sample_id}`: `{sample_dir}/` status=`{sample.status}` final_status=`{sample.final_status}`"
            )
    readme_path.write_text("\n".join(readme_lines) + "\n", encoding="utf-8")


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
    _write_sample_artifact_navigation(report=report)
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
        (
            f"- Sample seed: `{report.sample_seed}`"
            if report.sample_seed is not None
            else "- Sample seed: `manifest_order`"
        ),
        (
            f"- Sample shard: `{report.sample_shard_index}/{report.sample_shard_count}`"
            if report.sample_shard_count > 1
            else "- Sample shard: `full`"
        ),
        (
            f"- Checkpoint path: `{report.checkpoint_path}`"
            if report.checkpoint_path
            else "- Checkpoint path: `none`"
        ),
        (
            f"- Sample artifacts root: `{report.sample_artifacts_root}`"
            if report.sample_artifacts_root
            else "- Sample artifacts root: `none`"
        ),
        f"- Resumed from checkpoint: `{report.resumed_from_checkpoint}`",
        f"- Resume count: `{report.resume_count}`",
        f"- Duration tolerance ratio: `{report.duration_tolerance_ratio:.3f}`",
        f"- Target retained sample count: `{report.summary.target_sample_count}`",
        f"- Retained sample count: `{report.summary.sample_count}`",
        f"- Attempted candidate count: `{report.summary.attempted_candidate_count}`",
        f"- Replacement count: `{report.summary.replacement_count}`",
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
        "- Each `samples/<sample_id>/` directory contains `final/` outputs plus raw `llm/summarizer`, `llm/critic`, and `llm/judge` call traces.",
        "- `samples/index.json` and `samples/README.md` provide a run-local entrypoint into the per-sample artifact tree.",
        "- `report.json` includes transcript and summary per-speaker message lists for speaker-style auditing.",
        "- When transcript_only is false, `report.json` also records source-audio, speaker-sample, and archived voice-clone artifact paths per sample.",
    ]
    notes_path.write_text("\n".join(notes_lines) + "\n", encoding="utf-8")

    results_lines = [
        "# SPoRC Summarizer-Critic Benchmark Results",
        "",
        "## Retained Sample Results",
        "",
        render_sample_results_table(
            [sample for sample in report.samples if sample.counts_toward_target]
        ),
        "",
        "## Aggregate Metrics",
        "",
        f"- Target retained samples: `{report.summary.target_sample_count}`",
        f"- Retained samples: `{report.summary.sample_count}`",
        f"- Attempted candidates: `{report.summary.attempted_candidate_count}`",
        f"- Replacement-excluded candidates: `{report.summary.replacement_count}`",
        f"- Executed samples: `{report.summary.executed_count}`",
        f"- Skipped samples: `{report.summary.skipped_count}`",
        "",
        _render_aggregate_metrics_table(report),
    ]
    excluded_samples = [
        sample for sample in report.samples if not sample.counts_toward_target
    ]
    if excluded_samples:
        results_lines.extend(
            [
                "",
                "## Replacement-Excluded Candidates",
                "",
                render_sample_results_table(excluded_samples),
            ]
        )
    skipped_samples = [
        sample
        for sample in report.samples
        if sample.status == "skipped" and not sample.counts_toward_target
    ]
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
        (
            f"- Sample seed: `{report.sample_seed}`"
            if report.sample_seed is not None
            else "- Sample seed: `manifest_order`"
        ),
        f"- Preset concurrency: `{report.preset_concurrency}`",
        "",
        render_duration_sweep_table(report.preset_summaries),
        "",
        "## Per-Preset Artifacts",
        "",
    ]
    for summary in report.preset_summaries:
        results_lines.append(
            f"- `{summary.duration_preset}`: report={summary.report_path or '-'} results={summary.results_path or '-'}"
        )
        results_lines.append(
            f"- `{summary.duration_preset}` restarts: `{summary.restart_count}` recovered_stalls: `{summary.recovered_stall_count}`"
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
        + render_supervised_results_table(records)
        + "\n",
        encoding="utf-8",
    )
    return session_json_path, notes_path, results_path
