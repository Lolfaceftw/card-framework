"""CLI helper coverage for the SPoRC benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from card_framework.benchmark.summarizer_critic.sporc.cli import (
    _DurationSweepChildState,
    _build_execute_child_command,
    _child_environment,
    _build_parser,
    _parse_duration_preset_list,
    _render_duration_sweep_menu,
    _resolve_duration_sweep_examples,
    _resolve_prepare_root,
    _run_duration_sweep,
    main,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_PREPARED_FULL_ROOT,
    DEFAULT_PREPARED_ROOT,
    DurationSweepPresetSummary,
    PreparedDatasetManifest,
)
from card_framework.shared.paths import REPO_ROOT


class _CompletedProcessStub:
    """Minimal completed process stub for duration-sweep tests."""

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int:
        return 0

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return 0


def test_sporc_child_environment_enables_unbuffered_python_output(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    env = _child_environment()

    assert env["PYTHONUNBUFFERED"] == "1"
    pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)
    assert str(REPO_ROOT / "src") in pythonpath_entries
    assert "existing-path" in pythonpath_entries


def test_render_duration_sweep_menu_lists_presets() -> None:
    menu = _render_duration_sweep_menu()

    assert "30s" in menu
    assert "batch-all" in menu


def test_parse_duration_preset_list_splits_and_validates() -> None:
    assert _parse_duration_preset_list("30s,1m,5m") == ["30s", "1m", "5m"]


def test_resolve_prepare_root_defaults_by_dataset_variant() -> None:
    assert _resolve_prepare_root(dataset_variant="sample", prepared_root="") == (
        DEFAULT_PREPARED_ROOT.resolve()
    )
    assert _resolve_prepare_root(dataset_variant="full", prepared_root="") == (
        DEFAULT_PREPARED_FULL_ROOT.resolve()
    )


def test_resolve_duration_sweep_examples_falls_back_to_max_samples() -> None:
    args = argparse.Namespace(examples=None, max_samples=100)

    assert _resolve_duration_sweep_examples(args) == 100


def test_resolve_duration_sweep_examples_prefers_explicit_examples() -> None:
    args = argparse.Namespace(examples=12, max_samples=100)

    assert _resolve_duration_sweep_examples(args) == 12


def test_build_execute_child_command_adds_sweep_specific_flags() -> None:
    args = argparse.Namespace(
        dataset_manifest="manifest.json",
        dataset_variant="full",
        episode_data_path="episode.jsonl.gz",
        turn_data_path="turn.jsonl.gz",
        reference_wpm=150.0,
        output_root="runs",
        provider_profiles="profiles.yaml",
        config="config.yaml",
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
        judge_source_char_limit=30000,
        duration_preset="",
        target_seconds_override=90,
        max_samples=25,
        max_iterations=3,
        summarizer_timeout_seconds=None,
        critic_timeout_seconds=None,
        retrieval_timeout_seconds=None,
        index_timeout_seconds=None,
        prepare_if_missing=True,
        force_download_dataset=False,
        force_download_audio=False,
        transcript_only=False,
    )

    command = _build_execute_child_command(
        args,
        duration_preset="5m",
        target_seconds_override=None,
        max_samples=100,
        transcript_only=True,
        retrieval_port=5001,
        summarizer_port=5002,
        critic_port=5003,
        disable_stream_output=True,
        emit_progress_events=True,
    )

    assert "--duration-preset" in command
    assert "5m" in command
    assert "--max-samples" in command
    assert "100" in command
    assert "--transcript-only" in command
    assert "--disable-stream-output" in command
    assert "--emit-progress-events" in command
    assert "--retrieval-port" in command
    assert "5001" in command
    assert "--summarizer-port" in command
    assert "5002" in command
    assert "--critic-port" in command
    assert "5003" in command
    assert "--target-seconds-override" not in command


def test_build_parser_skips_runner_import_for_prepare_dataset(monkeypatch) -> None:
    def _unexpected_runner_load(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("prepare parser should not load execute runner helpers")

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._load_runner_helpers",
        _unexpected_runner_load,
    )

    parser = _build_parser(target_command="prepare-dataset")

    args = parser.parse_args(["prepare-dataset"])
    assert args.command == "prepare-dataset"


def test_main_prepare_dataset_reports_progress_to_stderr(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    manifest = PreparedDatasetManifest(
        manifest_path=(tmp_path / "prepared" / "manifest.json").resolve(),
        dataset_id="sporc_full",
        sporc_dataset_id="blitt/SPoRC",
        dataset_variant="full",
        episode_data_path=(tmp_path / "episodeLevelData.jsonl.gz").resolve(),
        turn_data_path=(tmp_path / "speakerTurnData.jsonl.gz").resolve(),
        sample_count=1,
        samples=(),
        samples_path=(tmp_path / "prepared" / "samples.jsonl.gz").resolve(),
    )

    def _unexpected_runner_load(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("prepare command should not load execute runner helpers")

    def _fake_prepare_sporc_benchmark(**kwargs):  # type: ignore[no-untyped-def]
        reporter = kwargs["status_reporter"]
        reporter("Loading SPoRC episode metadata cache...")
        reporter("Loaded SPoRC episode metadata cache in 0.10 seconds.")
        return manifest

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._load_runner_helpers",
        _unexpected_runner_load,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._load_dataset_helpers",
        lambda **kwargs: (lambda **inner_kwargs: (None, None), _fake_prepare_sporc_benchmark),
    )

    exit_code = main(
        [
            "prepare-dataset",
            "--dataset-variant",
            "full",
            "--prepared-root",
            str(tmp_path / "prepared"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Loading SPoRC episode metadata cache..." in captured.err
    assert "Loaded SPoRC episode metadata cache in 0.10 seconds." in captured.err
    payload = json.loads(captured.out)
    assert payload["status"] == "completed"
    assert payload["sample_count"] == 1


def test_run_duration_sweep_uses_auto_parallelism_and_max_samples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launched: list[tuple[str, int]] = []
    finalized: list[tuple[str, int]] = []
    written_reports = []

    def _fake_launch_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        duration_preset = kwargs["duration_preset"]
        max_samples = kwargs["max_samples"]
        launched.append((duration_preset, max_samples))
        log_path = tmp_path / f"{duration_preset}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        return _DurationSweepChildState(
            duration_preset=duration_preset,
            process=_CompletedProcessStub(pid=1000 + len(launched)),
            log_path=log_path,
            log_handle=log_handle,
            reader_thread=None,
            retrieval_port=5100 + len(launched) * 3,
            summarizer_port=5101 + len(launched) * 3,
            critic_port=5102 + len(launched) * 3,
            max_samples=max_samples,
        )

    def _fake_finalize_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        state = kwargs["state"]
        fallback_sample_count = kwargs["fallback_sample_count"]
        finalized.append((state.duration_preset, fallback_sample_count))
        if not state.log_handle.closed:
            state.log_handle.close()
        return DurationSweepPresetSummary(
            duration_preset=state.duration_preset,
            target_seconds={"30s": 30, "1m": 60, "5m": 300}[state.duration_preset],
            status="completed",
            sample_count=fallback_sample_count,
            executed_count=fallback_sample_count,
            skipped_count=0,
            success_count=fallback_sample_count,
            failure_count=0,
            within_tolerance_count=fallback_sample_count,
            within_tolerance_rate=1.0,
            mean_candidate_seconds=30.0,
            mean_absolute_duration_delta_seconds=0.0,
            mean_factualness=0.9,
            mean_naturalness=0.9,
            mean_speaker_grammar_similarity=0.9,
            mean_composite=0.9,
            report_path=str(tmp_path / f"{state.duration_preset}.report.json"),
            notes_path=str(tmp_path / f"{state.duration_preset}.notes.md"),
            results_path=str(tmp_path / f"{state.duration_preset}.results.md"),
            mean_composite_ci95_lower=None,
            mean_composite_ci95_upper=None,
            error_message=None,
        )

    def _fake_write_duration_sweep_artifacts(**kwargs):  # type: ignore[no-untyped-def]
        report = kwargs["report"]
        written_reports.append(report)
        sweep_root = kwargs["sweep_root"]
        return sweep_root / "report.json", sweep_root / "results.md"

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._launch_duration_sweep_child",
        _fake_launch_duration_sweep_child,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._finalize_duration_sweep_child",
        _fake_finalize_duration_sweep_child,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._load_reporting_helpers",
        lambda **kwargs: (_fake_write_duration_sweep_artifacts, None),
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._emit_cli_status",
        lambda message: None,
    )

    args = argparse.Namespace(
        mode="batch-all",
        duration_preset="",
        duration_presets="30s,1m,5m",
        examples=None,
        max_samples=100,
        parallelism=0,
        sweep_root=str(tmp_path),
        enable_voice_clone=False,
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
    )

    exit_code = _run_duration_sweep(args)

    assert exit_code == 0
    assert launched == [("30s", 100), ("1m", 100), ("5m", 100)]
    assert finalized == [("30s", 100), ("1m", 100), ("5m", 100)]
    assert written_reports[-1].examples_per_preset == 100
    assert [summary.duration_preset for summary in written_reports[-1].preset_summaries] == [
        "30s",
        "1m",
        "5m",
    ]


def test_main_prepare_dataset_handles_keyboard_interrupt(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    def _fake_prepare_sporc_benchmark(**kwargs):  # type: ignore[no-untyped-def]
        reporter = kwargs["status_reporter"]
        reporter("Loading SPoRC prepared samples from speaker turns...")
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._load_dataset_helpers",
        lambda **kwargs: (lambda **inner_kwargs: (None, None), _fake_prepare_sporc_benchmark),
    )

    exit_code = main(
        [
            "prepare-dataset",
            "--dataset-variant",
            "full",
            "--prepared-root",
            str(tmp_path / "prepared"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 130
    assert "Cancelled." in captured.err


def test_main_curate_longform_writes_manifest_and_audit(
    tmp_path: Path,
    capsys,
) -> None:
    transcript_path = tmp_path / "transcripts" / "passing.transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    segments = []
    current_start = 0
    for index in range(150):
        segments.append(
            {
                "speaker": "SPEAKER_00" if index % 2 == 0 else "SPEAKER_01",
                "text": " ".join(["dialogue"] * 110),
                "start_time": current_start,
                "end_time": current_start + 60_000,
            }
        )
        current_start += 60_000
    transcript_path.write_text(
        json.dumps(
            {
                "segments": segments,
                "metadata": {"dataset": "sporc", "episode_title": "Long Dialogue"},
            }
        ),
        encoding="utf-8",
    )
    source_manifest_path = tmp_path / "prepared_full" / "manifest.json"
    source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(tmp_path / "speakerTurnData.jsonl.gz"),
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "passing",
                        "dataset": "sporc_full",
                        "episode_key": "episode://passing",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "Passing long-form summary.",
                        "target_seconds": 45,
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_manifest_path = tmp_path / "prepared_full" / "benchmark_longform_15k_manifest.json"
    audit_path = tmp_path / "prepared_full" / "benchmark_longform_15k_audit.json"

    exit_code = main(
        [
            "curate-longform",
            "--source-manifest",
            str(source_manifest_path),
            "--output-manifest",
            str(output_manifest_path),
            "--audit-path",
            str(audit_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert Path(payload["dataset_manifest"]).exists()
    assert Path(payload["audit_path"]).exists()


def test_main_curate_longform_reports_progress_to_stderr(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    manifest = PreparedDatasetManifest(
        manifest_path=(tmp_path / "prepared_full" / "benchmark_longform_15k_manifest.json").resolve(),
        dataset_id="sporc_full_longform_15k",
        sporc_dataset_id="blitt/SPoRC",
        dataset_variant="full",
        episode_data_path=(tmp_path / "episodeLevelData.jsonl.gz").resolve(),
        turn_data_path=(tmp_path / "speakerTurnData.jsonl.gz").resolve(),
        sample_count=2,
        samples=(),
        samples_path=None,
    )
    audit_path = (tmp_path / "prepared_full" / "benchmark_longform_15k_audit.json").resolve()

    def _fake_curate_longform_sporc_manifest(**kwargs):  # type: ignore[no-untyped-def]
        reporter = kwargs["status_reporter"]
        reporter("Loading prepared SPoRC source manifest...")
        reporter("Loading SPoRC long-form curation sample stats...")
        reporter(
            "Loading SPoRC long-form curation sample stats... evaluated 2 / 2 prepared samples; selected 1 so far; transcript fallbacks 0."
        )
        reporter("Loaded SPoRC long-form curation sample stats in 0.20 seconds.")
        reporter("Loading curated SPoRC long-form manifest...")
        reporter("Loaded curated SPoRC long-form manifest in 0.01 seconds.")
        reporter("Loading SPoRC long-form curation audit...")
        reporter("Loaded SPoRC long-form curation audit in 0.01 seconds.")
        return manifest, audit_path

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._load_dataset_helpers",
        lambda **kwargs: (_fake_curate_longform_sporc_manifest, lambda **inner_kwargs: manifest),
    )

    exit_code = main(
        [
            "curate-longform",
            "--source-manifest",
            str(tmp_path / "prepared_full" / "manifest.json"),
            "--output-manifest",
            str(manifest.manifest_path),
            "--audit-path",
            str(audit_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Loading prepared SPoRC source manifest..." in captured.err
    assert "Loading SPoRC long-form curation sample stats..." in captured.err
    assert (
        "evaluated 2 / 2 prepared samples; selected 1 so far; transcript fallbacks 0."
        in captured.err
    )
    assert "Loaded SPoRC long-form curation audit in 0.01 seconds." in captured.err
    payload = json.loads(captured.out)
    assert payload["status"] == "completed"
    assert payload["sample_count"] == 2
