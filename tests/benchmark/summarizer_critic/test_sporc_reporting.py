"""Reporting coverage for the SPoRC benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from card_framework.benchmark.summarizer_critic.sporc.config import (
    BenchmarkExecutionReport,
    BenchmarkSummary,
    DurationSweepPresetSummary,
    DurationSweepReport,
    SampleBenchmarkResult,
    SampleMetricBundle,
)
from card_framework.benchmark.summarizer_critic.sporc.reporting import (
    render_sample_results_table,
    render_duration_sweep_table,
    write_duration_sweep_artifacts,
    write_execution_artifacts,
)


def _sample_result() -> SampleBenchmarkResult:
    return SampleBenchmarkResult(
        sample_id="sporc_00001_example_episode",
        dataset="sporc_sample",
        episode_key="podcasts/example_episode.mp3",
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
            factualness=0.8,
            naturalness=0.7,
            speaker_grammar_similarity=0.9,
            composite=0.8,
        ),
        source_audio_path="audio/example.mp3",
        speaker_samples_manifest_path="speaker_samples/manifest.json",
        draft_audio_state_path="voice_clone/live_draft.state.json",
        voice_clone_output_dir="voice_clone/sample_0001",
        voice_clone_manifest_path="voice_clone/sample_0001/manifest.json",
        voice_clone_merged_audio_path="voice_clone/sample_0001/voice_cloned.wav",
        transcript_per_speaker_messages={"SPEAKER_00": ["hello there"]},
        summary_per_speaker_messages={"SPEAKER_00": ["hello there"]},
        speaker_alias_map={"Alex Host": "SPEAKER_00"},
        speaker_notes=["hostSpeakerLabels: {'Alex Host': ['SPEAKER_00']}"],
    )


def test_render_sample_results_table_contains_expected_columns() -> None:
    table = render_sample_results_table([_sample_result()])
    assert "| sample | status |" in table
    assert "| --- | --- |" in table
    assert "sporc_00001_example_episode" in table
    assert "0.800" in table


def test_write_execution_artifacts_persists_speaker_comparison_payloads(
    tmp_path: Path,
) -> None:
    sample = _sample_result()
    sample.sample_artifact_dir = str((tmp_path / "samples" / sample.sample_id).resolve())
    sample.artifact_manifest_path = str(
        (tmp_path / "samples" / sample.sample_id / "manifest.json").resolve()
    )
    sample.summarizer_trace_dir = str(
        (tmp_path / "samples" / sample.sample_id / "llm" / "summarizer").resolve()
    )
    sample.critic_trace_dir = str(
        (tmp_path / "samples" / sample.sample_id / "llm" / "critic").resolve()
    )
    sample.judge_trace_dir = str(
        (tmp_path / "samples" / sample.sample_id / "llm" / "judge").resolve()
    )
    sample.final_summary_path = str(
        (tmp_path / "samples" / sample.sample_id / "final" / "summary.xml").resolve()
    )
    sample.final_critic_feedback_path = str(
        (
            tmp_path / "samples" / sample.sample_id / "final" / "critic_feedback.txt"
        ).resolve()
    )
    sample.loop_diagnostics_path = str(
        (
            tmp_path / "samples" / sample.sample_id / "final" / "loop_diagnostics.json"
        ).resolve()
    )
    sample.sample_result_artifact_path = str(
        (tmp_path / "samples" / sample.sample_id / "final" / "sample_result.json").resolve()
    )
    report = BenchmarkExecutionReport(
        run_id="run_001",
        generated_at_utc="2026-03-13T00:00:00+00:00",
        status="completed",
        dataset_manifest_path="manifest.json",
        sporc_dataset_id="blitt/SPoRC",
        dataset_variant="sample",
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
        commands_executed=[
            "python -m card_framework.benchmark.summarizer_critic.sporc execute"
        ],
        git_commit="abc123",
        git_branch="main",
        report_path="",
        notes_path="",
        results_path="",
        samples=[sample],
        summary=BenchmarkSummary(
            sample_count=1,
            executed_count=1,
            skipped_count=0,
            success_count=1,
            failure_count=0,
            mean_factualness=0.8,
            mean_naturalness=0.7,
            mean_speaker_grammar_similarity=0.9,
            mean_composite=0.8,
        ),
        sample_artifacts_root=str((tmp_path / "samples").resolve()),
        sample_seed=23,
        sample_shard_index=2,
        sample_shard_count=3,
    )

    report_path, notes_path, results_path = write_execution_artifacts(
        run_root=tmp_path,
        report=report,
    )

    assert report_path.exists()
    assert notes_path.exists()
    assert results_path.exists()

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    sample_payload = report_payload["samples"][0]
    assert report_payload["summary"]["executed_count"] == 1
    assert report_payload["summary"]["skipped_count"] == 0
    assert sample_payload["transcript_per_speaker_messages"]["SPEAKER_00"] == [
        "hello there"
    ]
    assert sample_payload["summary_per_speaker_messages"]["SPEAKER_00"] == [
        "hello there"
    ]
    assert "- Sample shard: `2/3`" in notes_path.read_text(encoding="utf-8")
    assert (
        "- Sample artifacts root: "
        f"`{(tmp_path / 'samples').resolve()}`"
        in notes_path.read_text(encoding="utf-8")
    )
    assert sample_payload["speaker_alias_map"]["Alex Host"] == "SPEAKER_00"
    assert (
        sample_payload["voice_clone_manifest_path"]
        == "voice_clone/sample_0001/manifest.json"
    )

    results_text = results_path.read_text(encoding="utf-8")
    notes_text = notes_path.read_text(encoding="utf-8")
    assert "- Sample seed: `23`" in notes_text
    assert "samples/index.json" in notes_text
    assert "| metric | mean | 95% CI |" in results_text
    assert "n/a" in results_text
    assert "```text" not in results_text

    sample_index_path = tmp_path / "samples" / "index.json"
    sample_readme_path = tmp_path / "samples" / "README.md"
    assert sample_index_path.exists()
    assert sample_readme_path.exists()
    sample_index_payload = json.loads(sample_index_path.read_text(encoding="utf-8"))
    assert sample_index_payload["samples"][0]["sample_id"] == sample.sample_id
    assert (
        sample_index_payload["samples"][0]["llm_trace_dirs"]["judge"]
        == sample.judge_trace_dir
    )
    assert "call_XXX.json" in sample_readme_path.read_text(encoding="utf-8")


def test_render_duration_sweep_table_contains_expected_columns() -> None:
    table = render_duration_sweep_table(
        [
            DurationSweepPresetSummary(
                duration_preset="30s",
                target_seconds=30,
                status="completed",
                sample_count=10,
                executed_count=10,
                skipped_count=0,
                success_count=10,
                failure_count=0,
                within_tolerance_count=9,
                within_tolerance_rate=0.9,
                mean_candidate_seconds=29.5,
                mean_absolute_duration_delta_seconds=2.1,
                mean_factualness=0.8,
                mean_naturalness=0.7,
                mean_speaker_grammar_similarity=0.6,
                mean_composite=0.7,
                report_path="report.json",
                notes_path="notes.md",
                results_path="results.md",
                mean_composite_ci95_lower=0.65,
                mean_composite_ci95_upper=0.75,
            )
        ]
    )

    assert "| preset | status |" in table
    assert "speaker_grammar" in table
    assert "composite_95_ci" in table
    assert "30s" in table
    assert "0.700" in table
    assert "[0.650, 0.750]" in table


def test_write_duration_sweep_artifacts_persists_report(tmp_path: Path) -> None:
    report = DurationSweepReport(
        sweep_id="sweep_001",
        generated_at_utc="2026-03-13T00:00:00+00:00",
        mode="batch-all",
        examples_per_preset=10,
        transcript_only=True,
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
        commands_executed=[
            "python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep"
        ],
        preset_summaries=[
            DurationSweepPresetSummary(
                duration_preset="30s",
                target_seconds=30,
                status="completed",
                sample_count=10,
                executed_count=10,
                skipped_count=0,
                success_count=10,
                failure_count=0,
                within_tolerance_count=9,
                within_tolerance_rate=0.9,
                mean_candidate_seconds=29.5,
                mean_absolute_duration_delta_seconds=2.1,
                mean_factualness=0.8,
                mean_naturalness=0.7,
                mean_speaker_grammar_similarity=0.6,
                mean_composite=0.7,
                report_path="report.json",
                notes_path="notes.md",
                results_path="results.md",
                mean_composite_ci95_lower=0.65,
                mean_composite_ci95_upper=0.75,
            )
        ],
        report_path="",
        results_path="",
        sample_seed=31,
        preset_concurrency=3,
    )

    report_path, results_path = write_duration_sweep_artifacts(
        sweep_root=tmp_path,
        report=report,
    )

    assert report_path.exists()
    assert results_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["preset_summaries"][0]["duration_preset"] == "30s"
    assert payload["preset_summaries"][0]["mean_composite_ci95_lower"] == 0.65
    results_text = results_path.read_text(encoding="utf-8")
    assert payload["sample_seed"] == 31
    assert "- Sample seed: `31`" in results_text
    assert "- Preset concurrency: `3`" in results_text
    assert "| composite_95_ci |" in results_text


def test_write_execution_artifacts_lists_preflight_skips(tmp_path: Path) -> None:
    skipped = SampleBenchmarkResult(
        sample_id="sporc_00099_short_teaser",
        dataset="sporc_sample",
        episode_key="podcasts/short_teaser.mp3",
        status="skipped",
        target_seconds=300,
        candidate_seconds=0.0,
        converged=False,
        iterations_run=0,
        final_status="preflight_failed",
        duration_seconds=0.0,
        retrieval_events=0,
        tool_invocations=0,
        metrics=SampleMetricBundle(
            factualness=0.0,
            naturalness=0.0,
            speaker_grammar_similarity=0.0,
            composite=0.0,
        ),
        error_message="Requested target 300s exceeds transcript capacity.",
        preflight_reason="Requested target 300s exceeds transcript capacity.",
        preflight_stats={
            "segment_count": 1,
            "transcript_word_count": 124,
            "max_feasible_summary_seconds": 60.0,
        },
    )
    report = BenchmarkExecutionReport(
        run_id="run_002",
        generated_at_utc="2026-03-13T00:00:00+00:00",
        status="partial",
        dataset_manifest_path="manifest.json",
        sporc_dataset_id="blitt/SPoRC",
        dataset_variant="sample",
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
        commands_executed=[
            "python -m card_framework.benchmark.summarizer_critic.sporc execute"
        ],
        git_commit="abc123",
        git_branch="main",
        report_path="",
        notes_path="",
        results_path="",
        samples=[_sample_result(), skipped],
        summary=BenchmarkSummary(
            sample_count=2,
            executed_count=1,
            skipped_count=1,
            success_count=1,
            failure_count=0,
            mean_factualness=0.8,
            mean_naturalness=0.7,
            mean_speaker_grammar_similarity=0.9,
            mean_composite=0.8,
        ),
    )

    _, notes_path, results_path = write_execution_artifacts(
        run_root=tmp_path,
        report=report,
    )

    notes_text = notes_path.read_text(encoding="utf-8")
    results_text = results_path.read_text(encoding="utf-8")
    assert "- Executed count: `1`" in notes_text
    assert "- Skipped count: `1`" in notes_text
    assert "## Preflight Skips" in results_text
    assert "sporc_00099_short_teaser" in results_text
    assert "| metric | mean | 95% CI |" in results_text
