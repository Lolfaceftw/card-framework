"""Coverage for the SPoRC post-critic revision helper."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any
from types import SimpleNamespace

from omegaconf import OmegaConf

from card_framework.benchmark.summarizer_critic.sporc.config import (
    BenchmarkExecutionReport,
    BenchmarkSummary,
    PreparedBenchmarkSample,
    SampleBenchmarkResult,
    SampleMetricBundle,
)
from card_framework.benchmark.summarizer_critic.sporc.cli import (
    _load_execution_report as _real_load_execution_report,
)
from card_framework.benchmark.summarizer_critic.sporc.judge import SPoRCJudgeResult
from card_framework.benchmark.summarizer_critic.sporc.runner import (
    _SampleArtifactRecorder,
)
from card_framework.benchmark.types import ProviderProfile
from card_framework.shared.paths import REPO_ROOT


def _load_module():
    script_path = (
        REPO_ROOT / "scripts" / "run_sporc_post_critic_revision.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "run_sporc_post_critic_revision", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _metric_bundle() -> SampleMetricBundle:
    return SampleMetricBundle(
        factualness=0.8,
        naturalness=0.7,
        speaker_grammar_similarity=0.6,
        composite=0.7,
    )


def _report_sample(
    tmp_path: Path, *, final_status: str = "max_iterations_reached"
) -> SampleBenchmarkResult:
    sample_root = (tmp_path / "sample-a").resolve()
    final_root = (sample_root / "final").resolve()
    final_root.mkdir(parents=True, exist_ok=True)
    summary_path = (final_root / "summary.xml").resolve()
    feedback_path = (final_root / "critic_feedback.txt").resolve()
    diagnostics_path = (final_root / "loop_diagnostics.json").resolve()
    summary_path.write_text(
        '<summary><turn speaker="SPEAKER_00">Saved draft</turn></summary>',
        encoding="utf-8",
    )
    feedback_path.write_text("Expand coverage.", encoding="utf-8")
    diagnostics_path.write_text(
        json.dumps({"iterations_run": 1, "max_iterations": 1}),
        encoding="utf-8",
    )
    return SampleBenchmarkResult(
        sample_id="sample-a",
        dataset="sporc_sample",
        episode_key="episode-1",
        status="completed",
        target_seconds=30,
        candidate_seconds=28.0,
        converged=False,
        iterations_run=1,
        final_status=final_status,
        duration_seconds=2.0,
        retrieval_events=0,
        tool_invocations=0,
        metrics=_metric_bundle(),
        final_summary_path=str(summary_path),
        final_critic_feedback_path=str(feedback_path),
        loop_diagnostics_path=str(diagnostics_path),
        counts_toward_target=True,
    )


def _report(tmp_path: Path, sample: SampleBenchmarkResult) -> BenchmarkExecutionReport:
    return BenchmarkExecutionReport(
        run_id="run-1",
        generated_at_utc="2026-04-03T00:00:00Z",
        status="completed",
        dataset_manifest_path=str((tmp_path / "manifest.json").resolve()),
        sporc_dataset_id="blitt/SPoRC",
        dataset_variant="sample",
        summarizer_provider="openai_gpt_5_4_mini",
        critic_provider="openai_gpt_5_4_mini",
        judge_provider="google_gemini_3_1_pro_preview_judge",
        embedding_mode="disabled",
        commands_executed=[],
        git_commit="abc123",
        git_branch="release/v2.4.0",
        report_path=str((tmp_path / "report.json").resolve()),
        notes_path=str((tmp_path / "results.md").resolve()),
        results_path=str((tmp_path / "results.json").resolve()),
        samples=[sample],
        summary=BenchmarkSummary(1, 1, 1, 0, 1, 0, 1, 0, 0.8, 0.7, 0.6, 0.7),
        duration_tolerance_ratio=0.05,
        transcript_only=True,
    )


def test_eligible_samples_only_selects_completed_capped_results(tmp_path: Path) -> None:
    module = _load_module()
    eligible_sample = _report_sample(tmp_path, final_status="max_iterations_reached")
    ineligible_sample = replace(
        _report_sample(tmp_path / "other", final_status="pass"),
        sample_id="sample-b",
        status="completed",
    )
    report = _report(tmp_path, eligible_sample)
    report.samples.append(ineligible_sample)

    eligible = module._eligible_samples(report, set())

    assert len(eligible) == 1
    selected_sample, previous_draft, critic_feedback, prior_diagnostics = eligible[0]
    assert selected_sample.sample_id == "sample-a"
    assert previous_draft.startswith("<summary>")
    assert critic_feedback == "Expand coverage."
    assert prior_diagnostics["iterations_run"] == 1


def test_resolve_session_report_paths_skips_incomplete_steps_by_default(
    tmp_path: Path,
) -> None:
    module = _load_module()
    completed_report = (tmp_path / "completed.json").resolve()
    completed_report.write_text("{}", encoding="utf-8")
    incomplete_report = (tmp_path / "incomplete.json").resolve()
    incomplete_report.write_text("{}", encoding="utf-8")
    session_payload = {
        "steps": {
            "openai": {
                "kind": "run",
                "provider_id": "openai_gpt_5_4_mini",
                "status": "completed",
                "report_path": str(completed_report),
            },
            "anthropic": {
                "kind": "run",
                "provider_id": "anthropic_claude_haiku_4_5",
                "status": "running",
                "report_path": str(incomplete_report),
            },
        }
    }

    resolved_default = module._resolve_session_report_paths(
        session_payload=session_payload,
        provider_filters=set(),
        allow_incomplete_steps=False,
    )
    resolved_including_incomplete = module._resolve_session_report_paths(
        session_payload=session_payload,
        provider_filters=set(),
        allow_incomplete_steps=True,
    )

    assert resolved_default == [
        ("openai_gpt_5_4_mini", completed_report),
    ]
    assert resolved_including_incomplete == [
        ("openai_gpt_5_4_mini", completed_report),
        ("anthropic_claude_haiku_4_5", incomplete_report),
    ]


def test_revise_report_in_place_runs_single_revision_and_rewrites_report(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    sample = _report_sample(tmp_path)
    report = _report(tmp_path, sample)
    report_path = Path(report.report_path)
    report_path.write_text("{}", encoding="utf-8")
    manifest_path = Path(report.dataset_manifest_path)
    manifest_path.write_text("{}", encoding="utf-8")

    revised_result = replace(
        sample,
        candidate_seconds=29.0,
        final_summary_path=str((tmp_path / "revised_summary.xml").resolve()),
        final_critic_feedback_path=str((tmp_path / "revised_feedback.txt").resolve()),
    )
    provider_profile = ProviderProfile(
        provider_id="openai_gpt_5_4_mini",
        description="test",
        llm_config={"_target_": "provider"},
        required_env=[],
    )
    judge_profile = ProviderProfile(
        provider_id="google_gemini_3_1_pro_preview_judge",
        description="test",
        llm_config={"_target_": "provider"},
        required_env=[],
    )
    rewritten_reports: list[BenchmarkExecutionReport] = []
    execute_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(module, "_load_execution_report", lambda path: report)
    monkeypatch.setattr(
        module, "ensure_compact_prepared_dataset_manifest", lambda path: path
    )
    monkeypatch.setattr(
        module,
        "load_prepared_dataset_manifest",
        lambda path, include_samples=False: SimpleNamespace(),
    )
    monkeypatch.setattr(
        module,
        "_load_sample_lookup",
        lambda manifest_path, sample_ids: {
            "sample-a": PreparedBenchmarkSample(
                sample_id="sample-a",
                dataset="sporc_sample",
                episode_key="episode-1",
                transcript_path=str((tmp_path / "transcript.json").resolve()),
                reference_summary="reference",
                target_seconds=30,
            )
        },
    )
    monkeypatch.setattr(
        module,
        "load_provider_profiles",
        lambda path: {
            "openai_gpt_5_4_mini": provider_profile,
            "google_gemini_3_1_pro_preview_judge": judge_profile,
        },
    )
    monkeypatch.setattr(
        module,
        "_find_provider_profile",
        lambda profiles, provider_id: profiles[provider_id],
    )
    monkeypatch.setattr(
        module, "_validate_provider_role_constraints", lambda profile, role: None
    )
    monkeypatch.setattr(module, "missing_required_env", lambda profile: [])
    monkeypatch.setattr(
        module,
        "_load_base_config",
        lambda path: OmegaConf.create({"audio": {}, "logging": {"enabled": False}}),
    )
    monkeypatch.setattr(
        module, "_configure_benchmark_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        module,
        "_resolve_embedding_profile",
        lambda base_cfg, embedding_mode: SimpleNamespace(embedding_id="disabled"),
    )
    monkeypatch.setattr(module, "_coerce_mapping", lambda value: {})
    monkeypatch.setattr(
        module, "_build_stub_calibration", lambda **kwargs: SimpleNamespace()
    )
    monkeypatch.setattr(module, "_prepare_provider_config", lambda *args, **kwargs: {})

    async def _fake_start_managed_runtime(**kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            orchestrator=object(),
            event_capture=object(),
            retrieval_enabled=False,
            judge=object(),
            artifact_recorder=object(),
        )

    async def _fake_execute_sample(**kwargs):  # type: ignore[no-untyped-def]
        execute_calls.append(kwargs)
        return revised_result

    async def _fake_stop_managed_runtime(runtime):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(module, "_start_managed_runtime", _fake_start_managed_runtime)
    monkeypatch.setattr(module, "_execute_sample", _fake_execute_sample)
    monkeypatch.setattr(module, "_stop_managed_runtime", _fake_stop_managed_runtime)
    monkeypatch.setattr(
        module,
        "_summarize_results",
        lambda results, target_sample_count: BenchmarkSummary(
            target_sample_count,
            len(results),
            len(results),
            0,
            len(results),
            0,
            len(results),
            0,
            0.8,
            0.7,
            0.6,
            0.7,
        ),
    )
    monkeypatch.setattr(
        module, "_report_status_from_summary", lambda summary: "completed"
    )
    monkeypatch.setattr(
        module,
        "_rewrite_report_in_place",
        lambda *, run_root, report: rewritten_reports.append(report),
    )

    result = module.asyncio.run(
        module._revise_report_in_place(
            report_path=report_path,
            provider_profiles_path=(tmp_path / "profiles.yaml").resolve(),
            config_path=(tmp_path / "config.yaml").resolve(),
            sample_filters=set(),
            stream_output=False,
        )
    )

    assert result["revised_sample_ids"] == ["sample-a"]
    assert len(execute_calls) == 1
    assert execute_calls[0]["resume_draft_xml"].startswith("<summary>")
    assert execute_calls[0]["resume_feedback"] == "Expand coverage."
    assert execute_calls[0]["max_iterations"] == 0
    assert execute_calls[0]["diagnostics_overrides"]["iterations_run"] == 2
    assert rewritten_reports[0].samples[0].iterations_run == 2
    assert (
        rewritten_reports[0].samples[0].final_status == "post_critic_revision_applied"
    )


def test_postprocess_revised_samples_are_not_selected_again(tmp_path: Path) -> None:
    module = _load_module()
    revised_sample = replace(
        _report_sample(tmp_path),
        final_status="post_critic_revision_applied",
    )
    report = _report(tmp_path, revised_sample)

    eligible = module._eligible_samples(report, set())

    assert eligible == []


def test_revise_report_in_place_persists_postprocess_feedback_artifacts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    sample = _report_sample(tmp_path)
    report = _report(tmp_path, sample)
    report_path = Path(report.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("{}", encoding="utf-8")
    manifest_path = Path(report.dataset_manifest_path)
    transcript_path = (tmp_path / "transcript.json").resolve()
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "text": "hello world from transcript",
                        "start_time": 0,
                        "end_time": 1000,
                    }
                ],
                "metadata": {},
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text("{}", encoding="utf-8")

    provider_profile = ProviderProfile(
        provider_id="openai_gpt_5_4_mini",
        description="test",
        llm_config={"_target_": "provider"},
        required_env=[],
    )
    judge_profile = ProviderProfile(
        provider_id="google_gemini_3_1_pro_preview_judge",
        description="test",
        llm_config={"_target_": "provider"},
        required_env=[],
    )

    class _RuntimeOrchestrator:
        async def run_loop_with_diagnostics(self, **kwargs):  # type: ignore[no-untyped-def]
            assert kwargs["initial_draft"].startswith("<summary>")
            assert kwargs["initial_feedback"] == "Expand coverage."
            return {
                "converged": False,
                "iterations_run": 1,
                "max_iterations": 0,
                "draft": '<summary><turn speaker="SPEAKER_00">Revised draft</turn></summary>',
                "final_status": "summarizer_only",
                "final_word_count": 2,
                "final_estimated_seconds": 0.0,
                "final_feedback": "",
                "iteration_details": [],
            }

    class _EventCapture:
        def count(self, name: str) -> int:
            return 0

    monkeypatch.setattr(module, "_load_execution_report", lambda path: report)
    monkeypatch.setattr(
        module, "ensure_compact_prepared_dataset_manifest", lambda path: path
    )
    monkeypatch.setattr(
        module,
        "load_prepared_dataset_manifest",
        lambda path, include_samples=False: SimpleNamespace(),
    )
    monkeypatch.setattr(
        module,
        "_load_sample_lookup",
        lambda manifest_path, sample_ids: {
            "sample-a": PreparedBenchmarkSample(
                sample_id="sample-a",
                dataset="sporc_sample",
                episode_key="episode-1",
                transcript_path=str(transcript_path),
                reference_summary="reference",
                target_seconds=30,
            )
        },
    )
    monkeypatch.setattr(
        module,
        "load_provider_profiles",
        lambda path: {
            "openai_gpt_5_4_mini": provider_profile,
            "google_gemini_3_1_pro_preview_judge": judge_profile,
        },
    )
    monkeypatch.setattr(
        module,
        "_find_provider_profile",
        lambda profiles, provider_id: profiles[provider_id],
    )
    monkeypatch.setattr(
        module, "_validate_provider_role_constraints", lambda profile, role: None
    )
    monkeypatch.setattr(module, "missing_required_env", lambda profile: [])
    monkeypatch.setattr(
        module,
        "_load_base_config",
        lambda path: OmegaConf.create({"audio": {}, "logging": {"enabled": False}}),
    )
    monkeypatch.setattr(
        module, "_configure_benchmark_logging", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        module,
        "_resolve_embedding_profile",
        lambda base_cfg, embedding_mode: SimpleNamespace(embedding_id="disabled"),
    )
    monkeypatch.setattr(module, "_coerce_mapping", lambda value: {})
    monkeypatch.setattr(
        module,
        "_build_stub_calibration",
        lambda **kwargs: SimpleNamespace(estimate_text_seconds=lambda **kw: 30.0),
    )
    monkeypatch.setattr(module, "_prepare_provider_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module,
        "utc_now_iso",
        lambda: "2026-04-03T20:30:00Z",
    )

    async def _fake_start_managed_runtime(**kwargs):  # type: ignore[no-untyped-def]
        return SimpleNamespace(
            orchestrator=_RuntimeOrchestrator(),
            event_capture=_EventCapture(),
            retrieval_enabled=False,
            judge=SimpleNamespace(
                evaluate=lambda **kw: SPoRCJudgeResult(
                    status="ok",
                    scores={
                        "Factualness": 0.8,
                        "Naturalness": 0.7,
                        "Speaker Grammar Similarity": 0.6,
                    },
                    error_message=None,
                )
            ),
            artifact_recorder=_SampleArtifactRecorder(run_root=report_path.parent),
        )

    async def _fake_stop_managed_runtime(runtime):  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(module, "_start_managed_runtime", _fake_start_managed_runtime)
    monkeypatch.setattr(module, "_stop_managed_runtime", _fake_stop_managed_runtime)

    result = module.asyncio.run(
        module._revise_report_in_place(
            report_path=report_path,
            provider_profiles_path=(tmp_path / "profiles.yaml").resolve(),
            config_path=(tmp_path / "config.yaml").resolve(),
            sample_filters=set(),
            stream_output=False,
        )
    )

    rewritten_report = _real_load_execution_report(report_path)
    rewritten_sample = rewritten_report.samples[0]
    diagnostics_payload = json.loads(
        Path(rewritten_sample.loop_diagnostics_path or "").read_text(encoding="utf-8")
    )
    persisted_feedback = Path(
        rewritten_sample.final_critic_feedback_path or ""
    ).read_text(encoding="utf-8")

    assert result["revised_sample_ids"] == ["sample-a"]
    assert rewritten_sample.final_status == "post_critic_revision_applied"
    assert diagnostics_payload["postprocess_revision_applied"] is True
    assert diagnostics_payload["seed_feedback"] == "Expand coverage."
    assert (
        "No new critic pass was run after this salvage revision." in persisted_feedback
    )
    assert persisted_feedback.endswith("Expand coverage.")
