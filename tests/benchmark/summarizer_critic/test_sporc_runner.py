"""Runner helper coverage for the SPoRC benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from card_framework.benchmark.summarizer_critic.sporc.runner import (
    _build_preflight_failed_result,
    _classify_sample_failure,
    _prepare_provider_config,
    _compute_composite_score,
    _evaluate_sample_preflight,
    _normalize_max_iterations,
    _prepare_benchmark_audio_cfg,
    _resolve_override_target_seconds,
    _resolve_sample_target_seconds,
    _resolve_orchestrator_timeouts,
    _select_samples,
    _summarize_results,
    build_execute_parser,
    execute_benchmark,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    PreparedBenchmarkSample,
    SampleBenchmarkResult,
    SampleMetricBundle,
)
from card_framework.benchmark.types import EmbeddingProfile, ProviderProfile


def _sample(
    sample_id: str,
    *,
    transcript_path: str | None = None,
) -> PreparedBenchmarkSample:
    return PreparedBenchmarkSample(
        sample_id=sample_id,
        dataset="sporc_sample",
        episode_key=sample_id,
        transcript_path=transcript_path or f"{sample_id}.json",
        reference_summary="reference",
        target_seconds=30,
    )


def _write_transcript(
    path: Path,
    *,
    text: str,
    start_time: int = 0,
    end_time: int = 60_000,
) -> None:
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "text": text,
                        "start_time": start_time,
                        "end_time": end_time,
                    }
                ],
                "metadata": {"dataset": "sporc"},
            }
        ),
        encoding="utf-8",
    )


def _write_empty_transcript(path: Path) -> None:
    path.write_text(
        json.dumps({"segments": [], "metadata": {"dataset": "sporc"}}),
        encoding="utf-8",
    )


def _completed_result(sample_id: str) -> SampleBenchmarkResult:
    return SampleBenchmarkResult(
        sample_id=sample_id,
        dataset="sporc_sample",
        episode_key=sample_id,
        status="completed",
        target_seconds=30,
        candidate_seconds=29.5,
        converged=True,
        iterations_run=2,
        final_status="pass",
        duration_seconds=3.0,
        retrieval_events=1,
        tool_invocations=4,
        metrics=SampleMetricBundle(
            factualness=0.8,
            naturalness=0.7,
            speaker_grammar_similarity=0.6,
            composite=0.7,
        ),
    )


def _failed_result(sample_id: str) -> SampleBenchmarkResult:
    return SampleBenchmarkResult(
        sample_id=sample_id,
        dataset="sporc_sample",
        episode_key=sample_id,
        status="failed",
        target_seconds=30,
        candidate_seconds=0.0,
        converged=False,
        iterations_run=0,
        final_status="execution_error",
        duration_seconds=1.0,
        retrieval_events=0,
        tool_invocations=0,
        metrics=SampleMetricBundle(0.0, 0.0, 0.0, 0.0),
        error_message="boom",
    )


def _benchmark_base_cfg(tmp_path: Path):
    return OmegaConf.create(
        {
            "orchestrator": {"duration_tolerance_ratio": 0.25},
            "audio": {
                "work_dir": str(tmp_path / "audio"),
                "speaker_samples": {"enabled": False},
                "voice_clone": {
                    "enabled": False,
                    "emo_presets": {"neutral": "steady neutral delivery"},
                    "live_drafting": {"enabled": False},
                },
            },
            "logging": {"enabled": False},
            "agents": {
                "summarizer": {"max_tool_turns": 1, "loop_guardrails": {}},
                "critic": {"max_tool_turns": 1, "reasoning_loop_guard": {}},
            },
        }
    )


def _benchmark_provider_profile() -> ProviderProfile:
    return ProviderProfile(
        provider_id="vllm_default",
        description="test",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )


def _disabled_embedding_profile() -> EmbeddingProfile:
    return EmbeddingProfile(
        embedding_id="disabled",
        description="disabled",
        embedding_config={
            "_target_": "card_framework.providers.noop.NoOpEmbeddingProvider"
        },
    )


def test_select_samples_defaults_to_all_when_zero() -> None:
    samples = (_sample("a"), _sample("b"), _sample("c"))
    assert [
        sample.sample_id
        for sample in _select_samples(
            samples,
            max_samples=0,
            sample_seed=None,
        )
    ] == ["a", "b", "c"]


def test_select_samples_uses_seed_independent_of_manifest_order() -> None:
    forward = (_sample("a"), _sample("b"), _sample("c"), _sample("d"))
    reverse = tuple(reversed(forward))

    forward_ids = [
        sample.sample_id
        for sample in _select_samples(
            forward,
            max_samples=3,
            sample_seed=23,
        )
    ]
    reverse_ids = [
        sample.sample_id
        for sample in _select_samples(
            reverse,
            max_samples=3,
            sample_seed=23,
        )
    ]

    assert forward_ids == reverse_ids


def test_normalize_max_iterations_maps_non_positive_values_to_none() -> None:
    assert _normalize_max_iterations(None) is None
    assert _normalize_max_iterations(0) is None
    assert _normalize_max_iterations(-1) is None
    assert _normalize_max_iterations(4) == 4


def test_resolve_orchestrator_timeouts_defaults_to_no_timeout() -> None:
    args = argparse.Namespace(
        summarizer_timeout_seconds=None,
        critic_timeout_seconds=None,
        retrieval_timeout_seconds=None,
        index_timeout_seconds=None,
    )
    assert _resolve_orchestrator_timeouts(args) == {
        "summarizer": None,
        "critic": None,
        "retrieval": None,
        "index": None,
    }


def test_resolve_orchestrator_timeouts_accepts_positive_overrides() -> None:
    args = argparse.Namespace(
        summarizer_timeout_seconds=25.0,
        critic_timeout_seconds=35.0,
        retrieval_timeout_seconds=45.0,
        index_timeout_seconds=55.0,
    )
    assert _resolve_orchestrator_timeouts(args) == {
        "summarizer": 25.0,
        "critic": 35.0,
        "retrieval": 45.0,
        "index": 55.0,
    }


def test_classify_sample_failure_maps_timeout_messages_to_stage_timeout() -> None:
    final_status, error_message = _classify_sample_failure(
        TimeoutError("summarizer request timed out after 900s")
    )

    assert final_status == "stage_timeout"
    assert "timed out" in error_message


def test_compute_composite_score_uses_only_three_judge_dimensions() -> None:
    assert (
        _compute_composite_score(
            factualness=0.9,
            naturalness=0.6,
            speaker_grammar_similarity=0.3,
        )
        == 0.6
    )


def test_resolve_override_target_seconds_uses_duration_preset() -> None:
    args = argparse.Namespace(duration_preset="5m", target_seconds_override=None)

    assert _resolve_override_target_seconds(args) == 300


def test_resolve_sample_target_seconds_prefers_override() -> None:
    sample = _sample("a")
    args = argparse.Namespace(duration_preset="1m", target_seconds_override=None)

    assert _resolve_sample_target_seconds(sample, args) == 60


def test_evaluate_sample_preflight_skips_incompatible_long_target(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(
        transcript_path,
        text=" ".join(["word"] * 124),
    )

    _, stats, reason = _evaluate_sample_preflight(
        sample=_sample("a", transcript_path=str(transcript_path)),
        target_seconds=300,
        reference_wpm=150.0,
    )

    assert reason is not None
    assert "exceeds transcript capacity" in reason
    assert stats["segment_count"] == 1
    assert stats["transcript_word_count"] == 124
    assert stats["max_feasible_summary_seconds"] == 60.0


def test_evaluate_sample_preflight_allows_compatible_short_target(
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(
        transcript_path,
        text=" ".join(["word"] * 124),
    )

    _, stats, reason = _evaluate_sample_preflight(
        sample=_sample("a", transcript_path=str(transcript_path)),
        target_seconds=30,
        reference_wpm=150.0,
    )

    assert reason is None
    assert stats["max_feasible_summary_seconds"] == 60.0


def test_summarize_results_excludes_skipped_samples_from_aggregates() -> None:
    skipped = _build_preflight_failed_result(
        sample=_sample("skip"),
        target_seconds=300,
        reason="too long",
        stats={"max_feasible_summary_seconds": 60.0},
    )

    summary = _summarize_results(
        [skipped, _completed_result("ok"), _failed_result("bad")]
    )

    assert summary.sample_count == 3
    assert summary.executed_count == 2
    assert summary.skipped_count == 1
    assert summary.success_count == 1
    assert summary.failure_count == 1
    assert summary.mean_composite == 0.35


def test_prepare_benchmark_audio_cfg_enables_faithful_live_draft_audio(
    tmp_path,
) -> None:
    audio_cfg = {
        "work_dir": "artifacts/audio_stage",
        "speaker_samples": {"enabled": False},
        "voice_clone": {"enabled": False, "live_drafting": {"enabled": False}},
    }

    prepared = _prepare_benchmark_audio_cfg(audio_cfg, run_root=tmp_path)

    assert prepared["speaker_samples"]["enabled"] is True
    assert prepared["voice_clone"]["enabled"] is True
    assert prepared["voice_clone"]["use_fp16"] is True
    assert prepared["voice_clone"]["live_drafting"]["enabled"] is True
    assert prepared["work_dir"] == str((tmp_path / "audio_pipeline").resolve())


def test_prepare_provider_config_applies_benchmark_vllm_timeout() -> None:
    profile = ProviderProfile(
        provider_id="vllm_default",
        description="test",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )

    prepared = _prepare_provider_config(profile)

    assert prepared["request_timeout_seconds"] == 60.0


def test_prepare_provider_config_applies_transcript_only_critic_overrides() -> None:
    profile = ProviderProfile(
        provider_id="vllm_default",
        description="test",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )

    prepared = _prepare_provider_config(
        profile,
        role="critic",
        transcript_only=True,
    )

    assert prepared["request_timeout_seconds"] == 300.0
    assert prepared["enable_thinking"] is True
    assert prepared["thinking_extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_execute_benchmark_short_circuits_when_all_samples_fail_preflight(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(
        transcript_path,
        text=" ".join(["word"] * 124),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_sample",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "sample",
                "episode_data_path": str(tmp_path / "episode.jsonl.gz"),
                "turn_data_path": str(tmp_path / "turn.jsonl.gz"),
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "sporc_00001_example",
                        "dataset": "sporc_sample",
                        "episode_key": "example.mp3",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "reference",
                        "target_seconds": 12,
                        "metadata": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--target-seconds-override",
            "300",
            "--max-samples",
            "1",
        ]
    )

    def _unexpected_instantiate(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("provider startup should not happen for all-skipped runs")

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.hydra.utils.instantiate",
        _unexpected_instantiate,
    )

    report = execute_benchmark(args)

    assert report.status == "failed"
    assert report.summary.sample_count == 1
    assert report.summary.executed_count == 0
    assert report.summary.skipped_count == 1
    assert report.samples[0].status == "skipped"
    assert report.samples[0].final_status == "preflight_failed"


def test_build_execute_parser_accepts_sweep_child_progress_flags() -> None:
    parser = argparse.ArgumentParser()
    build_execute_parser(parser)

    args = parser.parse_args(
        [
            "--retrieval-port",
            "5001",
            "--summarizer-port",
            "5002",
            "--critic-port",
            "5003",
            "--disable-stream-output",
            "--emit-progress-events",
            "--emit-progress-heartbeats",
            "--progress-heartbeat-seconds",
            "45",
            "--checkpoint-path",
            "checkpoint.json",
            "--resume-from-checkpoint",
            "--resume-failure-reason",
            "sweep_watchdog_stalled",
        ]
    )

    assert args.retrieval_port == 5001
    assert args.summarizer_port == 5002
    assert args.critic_port == 5003
    assert args.disable_stream_output is True
    assert args.emit_progress_events is True
    assert args.emit_progress_heartbeats is True
    assert args.progress_heartbeat_seconds == 45.0
    assert args.checkpoint_path == "checkpoint.json"
    assert args.resume_from_checkpoint is True
    assert args.resume_failure_reason == "sweep_watchdog_stalled"


def test_execute_benchmark_resumes_checkpoint_and_marks_orphaned_sample_failed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcript_a = tmp_path / "a.transcript.json"
    transcript_b = tmp_path / "b.transcript.json"
    transcript_c = tmp_path / "c.transcript.json"
    _write_transcript(transcript_a, text=" ".join(["word"] * 124))
    _write_transcript(transcript_b, text=" ".join(["word"] * 124))
    _write_empty_transcript(transcript_c)

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_sample",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "sample",
                "episode_data_path": str(tmp_path / "episode.jsonl.gz"),
                "turn_data_path": str(tmp_path / "turn.jsonl.gz"),
                "sample_count": 3,
                "samples": [
                    {
                        "sample_id": "a",
                        "dataset": "sporc_sample",
                        "episode_key": "a.mp3",
                        "transcript_path": str(transcript_a),
                        "reference_summary": "reference a",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                    {
                        "sample_id": "b",
                        "dataset": "sporc_sample",
                        "episode_key": "b.mp3",
                        "transcript_path": str(transcript_b),
                        "reference_summary": "reference b",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                    {
                        "sample_id": "c",
                        "dataset": "sporc_sample",
                        "episode_key": "c.mp3",
                        "transcript_path": str(transcript_c),
                        "reference_summary": "reference c",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    checkpoint_path = tmp_path / "checkpoint.json"
    checkpoint_path.write_text(
        json.dumps(
            {
                "checkpoint_version": 1,
                "run_id": "resume-run",
                "dataset_manifest_path": str(manifest_path.resolve()),
                "dataset_variant": "sample",
                "duration_preset": "",
                "override_target_seconds": None,
                "transcript_only": True,
                "sample_seed": None,
                "sample_order": ["a", "b", "c"],
                "completed_results": [asdict(_completed_result("a"))],
                "active_sample": {
                    "sample_index": 2,
                    "sample_count": 3,
                    "sample_id": "b",
                    "target_seconds": 30,
                    "stage": "summarizer",
                    "iteration": 2,
                    "elapsed_sample_seconds": 42.5,
                    "last_activity_age_seconds": 601.0,
                },
                "resume_count": 0,
                "recovered_stall_count": 0,
                "updated_at_utc": "2026-03-15T04:30:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    parser = argparse.ArgumentParser()
    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--transcript-only",
            "--max-samples",
            "3",
            "--checkpoint-path",
            str(checkpoint_path),
            "--resume-from-checkpoint",
        ]
    )
    provider_profile = _benchmark_provider_profile()
    embedding_profile = _disabled_embedding_profile()

    async def _unexpected_start_runtime(**kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        raise AssertionError("runtime startup should not happen on all-terminal resume")

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._load_base_config",
        lambda path: _benchmark_base_cfg(tmp_path),
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.load_provider_profiles",
        lambda path: [provider_profile],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._find_provider_profile",
        lambda profiles, provider_id: provider_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.missing_required_env",
        lambda profile: [],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._resolve_embedding_profile",
        lambda base_cfg, embedding_mode: embedding_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._start_managed_runtime",
        _unexpected_start_runtime,
    )

    report = execute_benchmark(args)

    assert report.run_id == "resume-run"
    assert report.status == "partial"
    assert report.resumed_from_checkpoint is True
    assert report.resume_count == 1
    assert [sample.sample_id for sample in report.samples] == ["a", "b", "c"]
    assert report.samples[1].status == "failed"
    assert report.samples[1].final_status == "resume_interrupted_sample"
    assert report.samples[1].duration_seconds == 42.5
    assert report.samples[2].status == "skipped"
    assert report.samples[2].final_status == "preflight_failed"

    checkpoint_payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint_payload["resume_count"] == 1
    assert checkpoint_payload["active_sample"] is None
    assert len(checkpoint_payload["completed_results"]) == 3


def test_execute_benchmark_continues_after_failed_sample_and_rebuilds_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcript_a = tmp_path / "a.transcript.json"
    transcript_b = tmp_path / "b.transcript.json"
    _write_transcript(transcript_a, text=" ".join(["word"] * 124))
    _write_transcript(transcript_b, text=" ".join(["word"] * 124))

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_sample",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "sample",
                "episode_data_path": str(tmp_path / "episode.jsonl.gz"),
                "turn_data_path": str(tmp_path / "turn.jsonl.gz"),
                "sample_count": 2,
                "samples": [
                    {
                        "sample_id": "a",
                        "dataset": "sporc_sample",
                        "episode_key": "a.mp3",
                        "transcript_path": str(transcript_a),
                        "reference_summary": "reference a",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                    {
                        "sample_id": "b",
                        "dataset": "sporc_sample",
                        "episode_key": "b.mp3",
                        "transcript_path": str(transcript_b),
                        "reference_summary": "reference b",
                        "target_seconds": 30,
                        "metadata": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    parser = argparse.ArgumentParser()
    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--transcript-only",
            "--max-samples",
            "2",
        ]
    )

    base_cfg = _benchmark_base_cfg(tmp_path)
    provider_profile = _benchmark_provider_profile()
    embedding_profile = _disabled_embedding_profile()
    start_calls: list[bool] = []
    stop_calls: list[str] = []

    async def _fake_start_runtime(**kwargs):  # type: ignore[no-untyped-def]
        start_calls.append(bool(kwargs["force_dynamic_ports"]))
        return SimpleNamespace(
            orchestrator=object(),
            judge=object(),
            event_capture=object(),
            retrieval_enabled=False,
            shared_agent_client=object(),
            servers=[],
            retrieval_port=5001,
            summarizer_port=5002,
            critic_port=5003,
        )

    async def _fake_stop_runtime(runtime):  # type: ignore[no-untyped-def]
        if runtime is not None:
            stop_calls.append("stopped")

    async def _fake_execute_sample(*, sample, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        if sample.sample_id == "a":
            failed = _failed_result("a")
            failed.final_status = "stage_timeout"
            failed.error_message = "summarizer request timed out after 900s"
            return failed
        return _completed_result(sample.sample_id)

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._load_base_config",
        lambda path: base_cfg,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.load_provider_profiles",
        lambda path: [provider_profile],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._find_provider_profile",
        lambda profiles, provider_id: provider_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.missing_required_env",
        lambda profile: [],
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._resolve_embedding_profile",
        lambda base_cfg, embedding_mode: embedding_profile,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._start_managed_runtime",
        _fake_start_runtime,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._stop_managed_runtime",
        _fake_stop_runtime,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner._execute_sample",
        _fake_execute_sample,
    )

    report = execute_benchmark(args)

    assert report.status == "partial"
    assert [sample.sample_id for sample in report.samples] == ["a", "b"]
    assert report.samples[0].final_status == "stage_timeout"
    assert report.samples[1].status == "completed"
    assert start_calls == [False, True]
    assert len(stop_calls) == 2


def test_build_execute_parser_accepts_sample_seed() -> None:
    parser = argparse.ArgumentParser()
    build_execute_parser(parser)

    args = parser.parse_args(["--sample-seed", "29"])

    assert args.sample_seed == 29


def test_execute_benchmark_reads_only_requested_sidecar_samples(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcript_path = tmp_path / "sample.transcript.json"
    _write_transcript(
        transcript_path,
        text=" ".join(["word"] * 124),
    )
    samples_path = tmp_path / "samples.jsonl.gz"
    with gzip.open(samples_path, "wt", encoding="utf-8") as handle:
        for sample_id in ("sporc_00001_example", "sporc_00002_unused"):
            handle.write(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "dataset": "sporc_full",
                        "episode_key": f"{sample_id}.mp3",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "reference",
                        "target_seconds": 12,
                        "metadata": {},
                    }
                )
            )
            handle.write("\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episode.jsonl.gz"),
                "turn_data_path": str(tmp_path / "turn.jsonl.gz"),
                "sample_count": 2,
                "samples_path": str(samples_path),
            }
        ),
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(tmp_path / "runs"),
            "--target-seconds-override",
            "300",
            "--max-samples",
            "1",
        ]
    )

    def _unexpected_instantiate(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("provider startup should not happen for all-skipped runs")

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.runner.hydra.utils.instantiate",
        _unexpected_instantiate,
    )

    report = execute_benchmark(args)

    assert report.summary.sample_count == 1
    assert [sample.sample_id for sample in report.samples] == ["sporc_00001_example"]
    assert report.samples[0].status == "skipped"


def test_prepare_provider_config_applies_judge_specific_vllm_overrides() -> None:
    profile = ProviderProfile(
        provider_id="vllm_default",
        description="test",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )

    prepared = _prepare_provider_config(profile, role="judge")

    assert prepared["enable_thinking"] is True
    assert prepared["thinking_extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
