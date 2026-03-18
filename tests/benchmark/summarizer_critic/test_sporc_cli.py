"""CLI helper coverage for the SPoRC benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import time

import pytest

from card_framework.benchmark.summarizer_critic.sporc.cli import (
    _aggregate_duration_sweep_preset,
    _colorize_cli_status,
    _DurationSweepChildState,
    _build_execute_child_command,
    _launch_duration_sweep_child,
    _child_environment,
    _build_parser,
    _detect_duration_sweep_child_stall,
    _parse_duration_preset_list,
    _render_duration_sweep_menu,
    _resolve_duration_sweep_examples,
    _resolve_prepare_root,
    _run_duration_sweep,
    _update_duration_sweep_child_state,
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


class _MutableProcessStub:
    """Mutable process stub used by watchdog and restart tests."""

    def __init__(self, pid: int, returncode: int | None = None) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class _TtyStreamStub:
    """Minimal tty-like stream stub used by color helper tests."""

    def isatty(self) -> bool:
        return True


class _PopenStub:
    """Simple Popen stand-in that records launched child commands."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.stdout: list[str] = []

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


def test_colorize_cli_status_adds_ansi_codes_for_terminal_messages(monkeypatch) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")

    rendered = _colorize_cli_status(
        "Started preset 5m[s1/3] (retrieval=5001, summarizer=5002, critic=5003).",
        stream=_TtyStreamStub(),
    )

    assert rendered.startswith("\033[1m\033[36m")
    assert rendered.endswith("\033[0m")


def test_colorize_cli_status_respects_no_color(monkeypatch) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    message = "Started preset 5m."

    assert _colorize_cli_status(message, stream=_TtyStreamStub()) == message


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


def test_build_parser_accepts_human_sample_command() -> None:
    parser = _build_parser(target_command="human-sample")

    args = parser.parse_args(
        [
            "human-sample",
            "--dataset-manifest",
            "manifest.json",
            "--duration-preset",
            "30s",
            "--episodes",
            "5",
            "--words-per-minute",
            "170",
            "--sample-factor",
            "5",
        ]
    )

    assert args.command == "human-sample"
    assert args.dataset_manifest == "manifest.json"
    assert args.duration_preset == "30s"
    assert args.episodes == 5


def test_build_parser_accepts_llm_sample_command() -> None:
    parser = _build_parser(target_command="llm-sample")

    args = parser.parse_args(
        [
            "llm-sample",
            "--dataset-manifest",
            "manifest.json",
            "--duration-preset",
            "30s",
            "--samples",
            "20",
            "--episodes",
            "5",
            "--select-episode",
            "[7,10]",
            "--model",
            "claude-opus-4-6",
            "--thinking-effort",
            "medium",
        ]
    )

    assert args.command == "llm-sample"
    assert args.dataset_manifest == "manifest.json"
    assert args.duration_preset == "30s"
    assert args.samples == 20
    assert args.episodes == 5
    assert args.select_episode == "[7,10]"
    assert args.model == "claude-opus-4-6"


def test_build_parser_accepts_dedupe_manifest_command() -> None:
    parser = _build_parser(target_command="dedupe-manifest")

    args = parser.parse_args(
        [
            "dedupe-manifest",
            "--manifest-path",
            "manifest.json",
            "--audit-path",
            "audit.json",
        ]
    )

    assert args.command == "dedupe-manifest"
    assert args.manifest_path == "manifest.json"
    assert args.audit_path == "audit.json"


def test_build_parser_accepts_export_transcripts_command() -> None:
    parser = _build_parser(target_command="export-transcripts")

    args = parser.parse_args(
        [
            "export-transcripts",
            "--manifest-path",
            "manifest.json",
            "--episodes",
            "7-10",
            "--output-path",
            "combined.txt",
            "--backwards-compat",
            "--raw",
        ]
    )

    assert args.command == "export-transcripts"
    assert args.manifest_path == "manifest.json"
    assert args.episodes == "7-10"
    assert args.output_path == "combined.txt"
    assert args.backwards_compat is True
    assert args.raw is True


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
        sample_seed=41,
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
        checkpoint_path=Path("checkpoint.json"),
        resume_from_checkpoint=True,
        resume_failure_reason="sweep_watchdog_stalled",
        emit_progress_heartbeats=True,
        progress_heartbeat_seconds=30.0,
        apply_sweep_timeout_defaults=True,
        sample_shard_index=2,
        sample_shard_count=3,
    )

    assert "--duration-preset" in command
    assert "5m" in command
    assert "--max-samples" in command
    assert "100" in command
    assert "--sample-seed" in command
    assert "41" in command
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
    assert "--checkpoint-path" in command
    assert "checkpoint.json" in command
    assert "--resume-from-checkpoint" in command
    assert "--resume-failure-reason" in command
    assert "sweep_watchdog_stalled" in command
    assert "--emit-progress-heartbeats" in command
    assert "--progress-heartbeat-seconds" in command
    assert "30.0" in command
    assert "--summarizer-timeout-seconds" in command
    assert "900.0" in command
    assert "--critic-timeout-seconds" in command
    assert "--retrieval-timeout-seconds" in command
    assert "--index-timeout-seconds" in command
    assert "--sample-shard-index" in command
    assert "2" in command
    assert "--sample-shard-count" in command
    assert "3" in command


def test_launch_duration_sweep_child_uses_full_sample_scope_with_shared_queue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launched_commands: list[list[str]] = []

    def _fake_popen(command, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        launched_commands.append(list(command))
        return _PopenStub(pid=9001)

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._reserve_distinct_port_triplet",
        lambda used_ports: (5201, 5202, 5203),
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._start_duration_sweep_reader",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli.subprocess.Popen",
        _fake_popen,
    )

    args = argparse.Namespace(
        dataset_manifest="manifest.json",
        dataset_variant="full",
        episode_data_path="",
        turn_data_path="",
        reference_wpm=150.0,
        output_root=str(tmp_path / "runs"),
        provider_profiles="profiles.yaml",
        config="config.yaml",
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
        judge_source_char_limit=30000,
        duration_preset="",
        target_seconds_override=None,
        max_samples=25,
        sample_seed=41,
        max_iterations=3,
        summarizer_timeout_seconds=None,
        critic_timeout_seconds=None,
        retrieval_timeout_seconds=None,
        index_timeout_seconds=None,
        prepare_if_missing=True,
        force_download_dataset=False,
        force_download_audio=False,
        transcript_only=False,
        enable_voice_clone=False,
        progress_heartbeat_seconds=30.0,
    )

    state = _launch_duration_sweep_child(
        args=args,
        sweep_root=tmp_path / "sweep",
        duration_preset="5m",
        shard_index=7,
        shard_count=9,
        max_samples=100,
        used_ports=set(),
        event_queue=queue.Queue(),
        shared_sample_queue_root=tmp_path / "queue-root",
    )

    try:
        command = launched_commands[0]
        assert "--shared-sample-queue-root" in command
        assert command[command.index("--sample-shard-index") + 1] == "1"
        assert command[command.index("--sample-shard-count") + 1] == "1"
        assert state.shard_index == 7
        assert state.shard_count == 9
    finally:
        state.log_handle.close()


def test_launch_duration_sweep_child_uses_shared_retrieval_port_and_namespace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launched_commands: list[list[str]] = []

    def _fake_popen(command, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        launched_commands.append(list(command))
        return _PopenStub(pid=9002)

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._reserve_distinct_ports",
        lambda used_ports, count: [6202, 6203] if count == 2 else [6201],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._start_duration_sweep_reader",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli.subprocess.Popen",
        _fake_popen,
    )

    args = argparse.Namespace(
        dataset_manifest="manifest.json",
        dataset_variant="full",
        episode_data_path="",
        turn_data_path="",
        reference_wpm=150.0,
        output_root=str(tmp_path / "runs"),
        provider_profiles="profiles.yaml",
        config="config.yaml",
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="enabled",
        judge_source_char_limit=30000,
        duration_preset="",
        target_seconds_override=None,
        max_samples=25,
        sample_seed=41,
        max_iterations=3,
        summarizer_timeout_seconds=None,
        critic_timeout_seconds=None,
        retrieval_timeout_seconds=None,
        index_timeout_seconds=None,
        prepare_if_missing=True,
        force_download_dataset=False,
        force_download_audio=False,
        transcript_only=False,
        enable_voice_clone=False,
        progress_heartbeat_seconds=30.0,
    )

    state = _launch_duration_sweep_child(
        args=args,
        sweep_root=tmp_path / "sweep",
        duration_preset="5m",
        shard_index=4,
        shard_count=9,
        max_samples=100,
        used_ports=set(),
        event_queue=queue.Queue(),
        shared_retrieval_port=6201,
    )

    try:
        command = launched_commands[0]
        assert command[command.index("--retrieval-port") + 1] == "6201"
        assert command[command.index("--shared-retrieval-port") + 1] == "6201"
        assert command[command.index("--retrieval-namespace") + 1] == "sweep-worker-4"
        assert state.retrieval_port == 6201
        assert state.retrieval_namespace == "sweep-worker-4"
    finally:
        state.log_handle.close()


def test_update_duration_sweep_child_state_tracks_heartbeat_metadata(
    tmp_path: Path,
) -> None:
    state = _DurationSweepChildState(
        duration_preset="5m",
        process=_CompletedProcessStub(pid=1001),
        log_path=tmp_path / "5m.log",
        log_handle=(tmp_path / "5m.log").open("w", encoding="utf-8"),
        reader_thread=None,
        retrieval_port=5001,
        summarizer_port=5002,
        critic_port=5003,
        max_samples=100,
        checkpoint_path=tmp_path / "5m.checkpoint.json",
        output_root=tmp_path / "runs",
    )
    try:
        _update_duration_sweep_child_state(
            state,
            {
                "sporc_progress_event": "sample_start",
                "sample_index": 8,
                "sample_count": 100,
                "sample_id": "sample-8",
            },
        )
        _update_duration_sweep_child_state(
            state,
            {
                "sporc_progress_event": "sample_heartbeat",
                "sample_index": 8,
                "sample_count": 100,
                "sample_id": "sample-8",
                "stage": "summarizer",
                "iteration": 2,
                "last_activity_age_seconds": 45.0,
            },
        )

        assert state.active_sample_index == 8
        assert state.active_sample_id == "sample-8"
        assert state.active_stage == "summarizer"
        assert state.active_iteration == 2
        assert state.last_heartbeat_at is not None
        assert state.last_activity_at is not None
        assert state.last_activity_age_seconds == 45.0
    finally:
        state.log_handle.close()


def test_detect_duration_sweep_child_stall_uses_heartbeat_and_activity_age(
    tmp_path: Path,
) -> None:
    now_monotonic = time.monotonic()
    state = _DurationSweepChildState(
        duration_preset="5m",
        process=_MutableProcessStub(pid=1001, returncode=None),
        log_path=tmp_path / "5m.log",
        log_handle=(tmp_path / "5m.log").open("w", encoding="utf-8"),
        reader_thread=None,
        retrieval_port=5001,
        summarizer_port=5002,
        critic_port=5003,
        max_samples=100,
        checkpoint_path=tmp_path / "5m.checkpoint.json",
        output_root=tmp_path / "runs",
        active_sample_id="sample-8",
        active_stage="summarizer",
        last_heartbeat_at=now_monotonic - 100.0,
        last_activity_at=now_monotonic - 650.0,
        last_activity_age_seconds=650.0,
    )
    try:
        assert (
            _detect_duration_sweep_child_stall(
                state,
                now_monotonic=now_monotonic,
                child_stall_seconds=600.0,
            )
            is not None
        )
    finally:
        state.log_handle.close()


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
    launched: list[tuple[str, int, int, int]] = []
    finalized: list[tuple[str, int]] = []
    written_reports = []
    sample_counts_by_worker = {
        ("30s", 1): 100,
        ("1m", 1): 50,
        ("1m", 2): 50,
        ("5m", 1): 33,
        ("5m", 2): 33,
        ("5m", 3): 34,
    }

    def _fake_launch_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        duration_preset = kwargs["duration_preset"]
        shard_index = kwargs["shard_index"]
        shard_count = kwargs["shard_count"]
        max_samples = kwargs["max_samples"]
        launched.append((duration_preset, shard_index, shard_count, max_samples))
        log_path = tmp_path / f"{duration_preset}.worker-{shard_index}.log"
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
            checkpoint_path=tmp_path / f"{duration_preset}.worker-{shard_index}.checkpoint.json",
            output_root=tmp_path / "runs",
            shard_index=shard_index,
            shard_count=shard_count,
        )

    def _fake_finalize_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        state = kwargs["state"]
        fallback_sample_count = kwargs["fallback_sample_count"]
        finalized.append((state.duration_preset, fallback_sample_count))
        if not state.log_handle.closed:
            state.log_handle.close()
        sample_count = sample_counts_by_worker[(state.duration_preset, state.shard_index)]
        return DurationSweepPresetSummary(
            duration_preset=state.duration_preset,
            target_seconds={"30s": 30, "1m": 60, "5m": 300}[state.duration_preset],
            status="completed",
            attempted_candidate_count=sample_count,
            replacement_count=0,
            sample_count=sample_count,
            executed_count=sample_count,
            skipped_count=0,
            success_count=sample_count,
            failure_count=0,
            within_tolerance_count=sample_count,
            within_tolerance_rate=1.0,
            mean_candidate_seconds=30.0,
            mean_absolute_duration_delta_seconds=0.0,
            mean_factualness=0.9,
            mean_naturalness=0.9,
            mean_speaker_grammar_similarity=0.9,
            mean_composite=0.9,
            report_path=None,
            notes_path=None,
            results_path=None,
            mean_composite_ci95_lower=None,
            mean_composite_ci95_upper=None,
            error_message=None,
            restart_count=state.restart_count,
            recovered_stall_count=state.recovered_stall_count,
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
        sample_seed=19,
        parallelism=0,
        concurrency=1,
        sweep_root=str(tmp_path),
        enable_voice_clone=False,
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
    )

    exit_code = _run_duration_sweep(args)

    assert exit_code == 0
    assert launched == [
        ("30s", 1, 3, 100),
        ("1m", 2, 3, 100),
        ("5m", 3, 3, 100),
        ("1m", 1, 3, 100),
        ("5m", 1, 3, 100),
        ("5m", 2, 3, 100),
    ]
    assert finalized == [
        ("30s", 0),
        ("1m", 0),
        ("1m", 0),
        ("5m", 0),
        ("5m", 0),
        ("5m", 0),
    ]
    assert written_reports[-1].examples_per_preset == 100
    assert written_reports[-1].sample_seed == 19
    assert [summary.sample_count for summary in written_reports[-1].preset_summaries] == [
        100,
        100,
        100,
    ]
    assert [summary.duration_preset for summary in written_reports[-1].preset_summaries] == [
        "30s",
        "1m",
        "5m",
    ]


def test_aggregate_duration_sweep_preset_weights_multi_shard_summaries() -> None:
    aggregated = _aggregate_duration_sweep_preset(
        duration_preset="5m",
        child_summaries={
            1: DurationSweepPresetSummary(
                duration_preset="5m",
                target_seconds=300,
                status="partial",
                attempted_candidate_count=4,
                replacement_count=0,
                sample_count=4,
                executed_count=3,
                skipped_count=1,
                success_count=2,
                failure_count=1,
                within_tolerance_count=2,
                within_tolerance_rate=0.6667,
                mean_candidate_seconds=290.0,
                mean_absolute_duration_delta_seconds=15.0,
                mean_factualness=0.7,
                mean_naturalness=0.8,
                mean_speaker_grammar_similarity=0.9,
                mean_composite=0.8,
                report_path=None,
                notes_path=None,
                results_path=None,
                restart_count=1,
                recovered_stall_count=1,
            ),
            2: DurationSweepPresetSummary(
                duration_preset="5m",
                target_seconds=300,
                status="completed",
                attempted_candidate_count=2,
                replacement_count=0,
                sample_count=2,
                executed_count=2,
                skipped_count=0,
                success_count=2,
                failure_count=0,
                within_tolerance_count=1,
                within_tolerance_rate=0.5,
                mean_candidate_seconds=310.0,
                mean_absolute_duration_delta_seconds=20.0,
                mean_factualness=0.4,
                mean_naturalness=0.5,
                mean_speaker_grammar_similarity=0.6,
                mean_composite=0.5,
                report_path=None,
                notes_path=None,
                results_path=None,
                restart_count=2,
                recovered_stall_count=0,
            ),
        },
    )

    assert aggregated.status == "partial"
    assert aggregated.sample_count == 6
    assert aggregated.executed_count == 5
    assert aggregated.skipped_count == 1
    assert aggregated.success_count == 4
    assert aggregated.failure_count == 1
    assert aggregated.within_tolerance_count == 3
    assert aggregated.mean_candidate_seconds == 298.0
    assert aggregated.mean_absolute_duration_delta_seconds == 17.0
    assert aggregated.mean_factualness == 0.58
    assert aggregated.mean_naturalness == 0.68
    assert aggregated.mean_speaker_grammar_similarity == 0.78
    assert aggregated.mean_composite == 0.68
    assert aggregated.mean_composite_ci95_lower is None
    assert aggregated.mean_composite_ci95_upper is None
    assert aggregated.restart_count == 3
    assert aggregated.recovered_stall_count == 1


def test_run_duration_sweep_fans_out_each_preset_by_requested_concurrency(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launched: list[tuple[str, int, int, int]] = []
    written_reports = []
    sample_counts_by_worker = {
        ("30s", 1): 34,
        ("30s", 3): 33,
        ("30s", 5): 33,
        ("1m", 1): 17,
        ("1m", 2): 17,
        ("1m", 3): 17,
        ("1m", 4): 17,
        ("1m", 5): 16,
        ("1m", 6): 16,
    }

    def _fake_launch_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        duration_preset = kwargs["duration_preset"]
        shard_index = kwargs["shard_index"]
        shard_count = kwargs["shard_count"]
        max_samples = kwargs["max_samples"]
        launched.append((duration_preset, shard_index, shard_count, max_samples))
        log_path = tmp_path / f"{duration_preset}.shard-{shard_index}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        return _DurationSweepChildState(
            duration_preset=duration_preset,
            process=_CompletedProcessStub(pid=4000 + len(launched)),
            log_path=log_path,
            log_handle=log_handle,
            reader_thread=None,
            retrieval_port=7100 + len(launched) * 3,
            summarizer_port=7101 + len(launched) * 3,
            critic_port=7102 + len(launched) * 3,
            max_samples=max_samples,
            checkpoint_path=tmp_path / f"{duration_preset}.shard-{shard_index}.checkpoint.json",
            output_root=tmp_path / "runs",
            shard_index=shard_index,
            shard_count=shard_count,
        )

    def _fake_finalize_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        state = kwargs["state"]
        if not state.log_handle.closed:
            state.log_handle.close()
        sample_count = sample_counts_by_worker[(state.duration_preset, state.shard_index)]
        return DurationSweepPresetSummary(
            duration_preset=state.duration_preset,
            target_seconds={"30s": 30, "1m": 60}[state.duration_preset],
            status="completed",
            attempted_candidate_count=sample_count,
            replacement_count=0,
            sample_count=sample_count,
            executed_count=sample_count,
            skipped_count=0,
            success_count=sample_count,
            failure_count=0,
            within_tolerance_count=sample_count,
            within_tolerance_rate=1.0,
            mean_candidate_seconds=30.0,
            mean_absolute_duration_delta_seconds=0.0,
            mean_factualness=0.9,
            mean_naturalness=0.8,
            mean_speaker_grammar_similarity=0.7,
            mean_composite=0.8,
            report_path=None,
            notes_path=None,
            results_path=None,
            mean_composite_ci95_lower=None,
            mean_composite_ci95_upper=None,
            error_message=None,
            restart_count=0,
            recovered_stall_count=0,
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
        duration_presets="30s,1m",
        examples=None,
        max_samples=100,
        sample_seed=None,
        parallelism=0,
        concurrency=3,
        sweep_root=str(tmp_path),
        enable_voice_clone=False,
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
    )

    exit_code = _run_duration_sweep(args)

    assert exit_code == 0
    assert launched == [
        ("30s", 1, 6, 100),
        ("30s", 3, 6, 100),
        ("30s", 5, 6, 100),
        ("1m", 2, 6, 100),
        ("1m", 4, 6, 100),
        ("1m", 6, 6, 100),
        ("1m", 1, 6, 100),
        ("1m", 3, 6, 100),
        ("1m", 5, 6, 100),
    ]
    assert written_reports[-1].preset_concurrency == 3
    assert [summary.sample_count for summary in written_reports[-1].preset_summaries] == [
        100,
        100,
    ]


def test_run_duration_sweep_keyboard_interrupt_stops_active_children(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stopped: list[tuple[str, int, str]] = []

    def _fake_launch_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        duration_preset = kwargs["duration_preset"]
        shard_index = kwargs["shard_index"]
        shard_count = kwargs["shard_count"]
        log_path = tmp_path / f"{duration_preset}.interrupt-{shard_index}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        return _DurationSweepChildState(
            duration_preset=duration_preset,
            process=_MutableProcessStub(pid=5000 + shard_index, returncode=None),
            log_path=log_path,
            log_handle=log_handle,
            reader_thread=None,
            retrieval_port=8100 + shard_index * 3,
            summarizer_port=8101 + shard_index * 3,
            critic_port=8102 + shard_index * 3,
            max_samples=kwargs["max_samples"],
            checkpoint_path=tmp_path / f"{duration_preset}.interrupt-{shard_index}.checkpoint.json",
            output_root=tmp_path / "runs",
            shard_index=shard_index,
            shard_count=shard_count,
        )

    def _fake_stop_duration_sweep_child(state, *, reason):  # type: ignore[no-untyped-def]
        if state.process.poll() is not None and state.log_handle.closed:
            return state.process.poll()
        stopped.append((state.duration_preset, state.shard_index, reason))
        state.process.returncode = -9
        if not state.log_handle.closed:
            state.log_handle.close()
        return -9

    def _raise_keyboard_interrupt(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._launch_duration_sweep_child",
        _fake_launch_duration_sweep_child,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._stop_duration_sweep_child",
        _fake_stop_duration_sweep_child,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._format_duration_sweep_progress",
        _raise_keyboard_interrupt,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._emit_cli_status",
        lambda message: None,
    )

    args = argparse.Namespace(
        mode="single",
        duration_preset="5m",
        duration_presets="5m",
        examples=None,
        max_samples=12,
        sample_seed=None,
        parallelism=1,
        concurrency=2,
        sweep_root=str(tmp_path),
        enable_voice_clone=False,
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
        child_stall_seconds=600.0,
        max_child_restarts=1,
        progress_heartbeat_seconds=30.0,
        output_root=str(tmp_path / "runs"),
    )

    with pytest.raises(KeyboardInterrupt):
        _run_duration_sweep(args)

    assert stopped == [
        (
            "5m",
            1,
            "Cancellation requested by operator; shutting down child.",
        ),
        (
            "5m",
            2,
            "Cancellation requested by operator; shutting down child.",
        ),
    ]


def test_run_duration_sweep_restarts_stalled_child_and_records_recovery(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launched: list[tuple[bool, int]] = []
    written_reports = []

    def _fake_launch_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        duration_preset = kwargs["duration_preset"]
        resume_from_checkpoint = bool(kwargs.get("resume_from_checkpoint", False))
        restart_count = int(kwargs.get("restart_count", 0))
        recovered_stall_count = int(kwargs.get("recovered_stall_count", 0))
        process = _MutableProcessStub(
            pid=2000 + len(launched),
            returncode=0 if resume_from_checkpoint else None,
        )
        launched.append((resume_from_checkpoint, restart_count))
        log_path = tmp_path / f"{duration_preset}.log"
        log_handle = log_path.open("a" if resume_from_checkpoint else "w", encoding="utf-8")
        state = _DurationSweepChildState(
            duration_preset=duration_preset,
            process=process,
            log_path=log_path,
            log_handle=log_handle,
            reader_thread=None,
            retrieval_port=5100 + len(launched) * 3,
            summarizer_port=5101 + len(launched) * 3,
            critic_port=5102 + len(launched) * 3,
            max_samples=kwargs["max_samples"],
            checkpoint_path=tmp_path / f"{duration_preset}.checkpoint.json",
            output_root=tmp_path / "runs",
            sample_count=100,
            completed_samples=7,
            restart_count=restart_count,
            recovered_stall_count=recovered_stall_count,
        )
        if not resume_from_checkpoint:
            now_monotonic = time.monotonic()
            state.active_sample_index = 8
            state.active_sample_id = "sample-8"
            state.active_stage = "summarizer"
            state.last_heartbeat_at = now_monotonic - 100.0
            state.last_activity_at = now_monotonic - 650.0
            state.last_activity_age_seconds = 650.0
        else:
            kwargs["event_queue"].put(
                (
                    duration_preset,
                    1,
                    process.pid,
                    {
                        "sporc_progress_event": "sample_done",
                        "sample_index": 8,
                        "sample_count": 100,
                        "sample_id": "sample-8",
                        "final_status": "sweep_watchdog_stalled",
                    },
                )
            )
        return state

    def _fake_finalize_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        state = kwargs["state"]
        if not state.log_handle.closed:
            state.log_handle.close()
        return DurationSweepPresetSummary(
            duration_preset=state.duration_preset,
            target_seconds=300,
            status="completed",
            attempted_candidate_count=100,
            replacement_count=0,
            sample_count=100,
            executed_count=8,
            skipped_count=0,
            success_count=7,
            failure_count=1,
            within_tolerance_count=7,
            within_tolerance_rate=0.875,
            mean_candidate_seconds=300.0,
            mean_absolute_duration_delta_seconds=0.0,
            mean_factualness=0.9,
            mean_naturalness=0.9,
            mean_speaker_grammar_similarity=0.9,
            mean_composite=0.9,
            report_path=str(tmp_path / "5m.report.json"),
            notes_path=str(tmp_path / "5m.notes.md"),
            results_path=str(tmp_path / "5m.results.md"),
            mean_composite_ci95_lower=None,
            mean_composite_ci95_upper=None,
            error_message=None,
            restart_count=state.restart_count,
            recovered_stall_count=state.recovered_stall_count,
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
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._terminate_process_tree",
        lambda process: setattr(process, "returncode", -9),
    )

    args = argparse.Namespace(
        mode="single",
        duration_preset="5m",
        duration_presets="5m",
        examples=None,
        max_samples=100,
        sample_seed=None,
        parallelism=1,
        concurrency=1,
        sweep_root=str(tmp_path),
        enable_voice_clone=False,
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
        child_stall_seconds=600.0,
        max_child_restarts=2,
        progress_heartbeat_seconds=30.0,
        output_root=str(tmp_path / "runs"),
    )

    exit_code = _run_duration_sweep(args)

    assert exit_code == 0
    assert launched == [(False, 0), (True, 1)]
    assert written_reports[-1].preset_summaries[0].restart_count == 1
    assert written_reports[-1].preset_summaries[0].recovered_stall_count == 1


def test_run_duration_sweep_continues_after_multiple_stalled_samples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    launched: list[tuple[bool, int]] = []
    status_messages: list[str] = []
    written_reports = []

    def _fake_launch_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        duration_preset = kwargs["duration_preset"]
        resume_from_checkpoint = bool(kwargs.get("resume_from_checkpoint", False))
        restart_count = int(kwargs.get("restart_count", 0))
        recovered_stall_count = int(kwargs.get("recovered_stall_count", 0))
        launched.append((resume_from_checkpoint, restart_count))
        process = _MutableProcessStub(
            pid=3000 + len(launched),
            returncode=0 if restart_count >= 2 else None,
        )
        log_path = tmp_path / f"{duration_preset}.multi-stall.log"
        log_handle = log_path.open("a" if resume_from_checkpoint else "w", encoding="utf-8")
        state = _DurationSweepChildState(
            duration_preset=duration_preset,
            process=process,
            log_path=log_path,
            log_handle=log_handle,
            reader_thread=None,
            retrieval_port=6100 + len(launched) * 3,
            summarizer_port=6101 + len(launched) * 3,
            critic_port=6102 + len(launched) * 3,
            max_samples=kwargs["max_samples"],
            checkpoint_path=tmp_path / f"{duration_preset}.multi-stall.checkpoint.json",
            output_root=tmp_path / "runs",
            sample_count=100,
            completed_samples=7 + max(restart_count - 1, 0),
            restart_count=restart_count,
            recovered_stall_count=recovered_stall_count,
        )
        if not resume_from_checkpoint:
            now_monotonic = time.monotonic()
            state.active_sample_index = 8
            state.active_sample_id = "sample-8"
            state.active_stage = "summarizer"
            state.last_heartbeat_at = now_monotonic - 100.0
            state.last_activity_at = now_monotonic - 650.0
            state.last_activity_age_seconds = 650.0
            return state

        if restart_count == 1:
            kwargs["event_queue"].put(
                (
                    duration_preset,
                    1,
                    process.pid,
                    {
                        "sporc_progress_event": "sample_done",
                        "sample_index": 8,
                        "sample_count": 100,
                        "sample_id": "sample-8",
                        "final_status": "sweep_watchdog_stalled",
                    },
                )
            )
            kwargs["event_queue"].put(
                (
                    duration_preset,
                    1,
                    process.pid,
                    {
                        "sporc_progress_event": "sample_heartbeat",
                        "sample_index": 9,
                        "sample_count": 100,
                        "sample_id": "sample-9",
                        "stage": "critic",
                        "iteration": 1,
                        "last_activity_age_seconds": 650.0,
                    },
                )
            )
            return state

        kwargs["event_queue"].put(
            (
                duration_preset,
                1,
                process.pid,
                {
                    "sporc_progress_event": "sample_done",
                    "sample_index": 9,
                    "sample_count": 100,
                    "sample_id": "sample-9",
                    "final_status": "sweep_watchdog_stalled",
                },
            )
        )
        return state

    def _fake_finalize_duration_sweep_child(**kwargs):  # type: ignore[no-untyped-def]
        state = kwargs["state"]
        if not state.log_handle.closed:
            state.log_handle.close()
        return DurationSweepPresetSummary(
            duration_preset=state.duration_preset,
            target_seconds=300,
            status="completed",
            attempted_candidate_count=100,
            replacement_count=0,
            sample_count=100,
            executed_count=9,
            skipped_count=0,
            success_count=7,
            failure_count=2,
            within_tolerance_count=7,
            within_tolerance_rate=0.7778,
            mean_candidate_seconds=300.0,
            mean_absolute_duration_delta_seconds=0.0,
            mean_factualness=0.9,
            mean_naturalness=0.9,
            mean_speaker_grammar_similarity=0.9,
            mean_composite=0.9,
            report_path=str(tmp_path / "5m.multi-stall.report.json"),
            notes_path=str(tmp_path / "5m.multi-stall.notes.md"),
            results_path=str(tmp_path / "5m.multi-stall.results.md"),
            mean_composite_ci95_lower=None,
            mean_composite_ci95_upper=None,
            error_message=None,
            restart_count=state.restart_count,
            recovered_stall_count=state.recovered_stall_count,
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
        status_messages.append,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.cli._terminate_process_tree",
        lambda process: setattr(process, "returncode", -9),
    )

    args = argparse.Namespace(
        mode="single",
        duration_preset="5m",
        duration_presets="5m",
        examples=None,
        max_samples=100,
        sample_seed=None,
        parallelism=1,
        concurrency=1,
        sweep_root=str(tmp_path),
        enable_voice_clone=False,
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
        child_stall_seconds=600.0,
        max_child_restarts=1,
        progress_heartbeat_seconds=30.0,
        output_root=str(tmp_path / "runs"),
    )

    exit_code = _run_duration_sweep(args)

    assert exit_code == 0
    assert launched == [(False, 0), (True, 1), (True, 2)]
    assert any(
        "Continuing preset 5m after stalled sample exceeded the configured "
        "restart threshold (1):"
        in message
        for message in status_messages
    )
    assert all("Failed preset 5m" not in message for message in status_messages)
    assert written_reports[-1].preset_summaries[0].restart_count == 2
    assert written_reports[-1].preset_summaries[0].recovered_stall_count == 2


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
