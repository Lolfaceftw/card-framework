"""Apply one critic-guided summarizer revision pass to capped SPoRC reports."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from card_framework.benchmark.artifacts import utc_now_iso
from card_framework.benchmark.matrix import (
    load_provider_profiles,
    missing_required_env,
)
from card_framework.benchmark.summarizer_critic.qmsum.runner import (
    _configure_benchmark_logging,
)
from card_framework.benchmark.summarizer_critic.sporc.cli import _load_execution_report
from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    ensure_compact_prepared_dataset_manifest,
    iter_prepared_dataset_samples,
    load_prepared_dataset_manifest,
)
from card_framework.benchmark.summarizer_critic.sporc.runner import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROVIDER_PROFILES_PATH,
    _build_stub_calibration,
    _coerce_mapping,
    _execute_sample,
    _find_provider_profile,
    _load_base_config,
    _prepare_benchmark_audio_cfg,
    _prepare_provider_config,
    _report_status_from_summary,
    _resolve_embedding_profile,
    _start_managed_runtime,
    _stop_managed_runtime,
    _summarize_results,
    _validate_provider_role_constraints,
    build_voice_clone_orchestrator,
)
from card_framework.benchmark.summarizer_critic.sporc.reporting import (
    write_execution_artifacts,
)
from card_framework.shared.paths import REPO_ROOT


DEFAULT_SESSION_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "summarizer_critic_benchmark"
    / "sporc_budget_runs"
    / "_sessions"
).resolve()
_POSTPROCESS_FINAL_STATUS = "post_critic_revision_applied"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reuse existing SPoRC execute reports that stopped at max_iterations and "
            "apply exactly one extra summarizer pass from the saved critic feedback, "
            "then rerun only the judge."
        )
    )
    parser.add_argument(
        "--session-id",
        default="",
        help=(
            "Optional SPoRC budget-wrapper session UUID. Completed provider reports in "
            "that session become revision targets."
        ),
    )
    parser.add_argument(
        "--report-path",
        action="append",
        default=[],
        help=(
            "Explicit execute report.json path to revise in place. Repeat to process "
            "multiple reports."
        ),
    )
    parser.add_argument(
        "--provider",
        action="append",
        default=[],
        help="Optional provider id filter when --session-id is used.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Optional sample-id filter. Repeat to revise only specific samples.",
    )
    parser.add_argument(
        "--allow-incomplete-steps",
        action="store_true",
        help=(
            "Allow reports from wrapper steps whose session status is not completed. "
            "Use with care when another child process may still be writing the report."
        ),
    )
    parser.add_argument(
        "--provider-profiles",
        default=str(DEFAULT_PROVIDER_PROFILES_PATH),
        help="Provider profile YAML used when --report-path is supplied directly.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Hydra config file used to recreate the execute runtime.",
    )
    parser.add_argument(
        "--stream-output",
        action="store_true",
        help="Show live streamed token output from the revision summarizer and judge.",
    )
    return parser


def _load_session_payload(session_id: str) -> dict[str, Any]:
    normalized_session_id = str(session_id or "").strip()
    if not normalized_session_id:
        raise ValueError("--session-id cannot be empty.")
    session_path = (
        DEFAULT_SESSION_ROOT / normalized_session_id / "session.json"
    ).resolve()
    if not session_path.exists():
        raise FileNotFoundError(
            f"No session.json found for session {normalized_session_id}."
        )
    payload = json.loads(session_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid session payload in {session_path}.")
    return {str(key): value for key, value in payload.items()}


def _resolve_session_report_paths(
    *,
    session_payload: dict[str, Any],
    provider_filters: set[str],
    allow_incomplete_steps: bool,
) -> list[tuple[str, Path]]:
    raw_steps = session_payload.get("steps", {})
    if not isinstance(raw_steps, dict):
        raise ValueError("Session payload is missing its step mapping.")
    resolved: list[tuple[str, Path]] = []
    for raw_step in raw_steps.values():
        if not isinstance(raw_step, dict):
            continue
        kind = str(raw_step.get("kind", "") or "").strip()
        if kind != "run":
            continue
        provider_id = str(raw_step.get("provider_id", "") or "").strip()
        if provider_filters and provider_id not in provider_filters:
            continue
        status = str(raw_step.get("status", "") or "").strip()
        if status != "completed" and not allow_incomplete_steps:
            continue
        report_path = str(raw_step.get("report_path", "") or "").strip()
        if not report_path:
            continue
        report = Path(report_path).expanduser().resolve()
        if report.exists():
            resolved.append((provider_id, report))
    return resolved


def _eligible_samples(
    report, sample_filters: set[str]
) -> list[tuple[Any, str, str, dict[str, Any]]]:
    eligible: list[tuple[Any, str, str, dict[str, Any]]] = []
    for sample in report.samples:
        if sample_filters and sample.sample_id not in sample_filters:
            continue
        if sample.final_status != "max_iterations_reached":
            continue
        if sample.status != "completed":
            continue
        summary_path = str(sample.final_summary_path or "").strip()
        feedback_path = str(sample.final_critic_feedback_path or "").strip()
        diagnostics_path = str(sample.loop_diagnostics_path or "").strip()
        if not summary_path or not feedback_path:
            continue
        summary_file = Path(summary_path)
        feedback_file = Path(feedback_path)
        if not summary_file.exists() or not feedback_file.exists():
            continue
        previous_draft = summary_file.read_text(encoding="utf-8").strip()
        critic_feedback = feedback_file.read_text(encoding="utf-8").strip()
        if not previous_draft or not critic_feedback:
            continue
        prior_diagnostics: dict[str, Any] = {}
        if diagnostics_path:
            diagnostics_file = Path(diagnostics_path)
            if diagnostics_file.exists():
                loaded = json.loads(diagnostics_file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    prior_diagnostics = {
                        str(key): value for key, value in loaded.items()
                    }
        eligible.append((sample, previous_draft, critic_feedback, prior_diagnostics))
    return eligible


def _load_sample_lookup(
    manifest_path: Path,
    sample_ids: set[str],
) -> dict[str, Any]:
    lookup: dict[str, Any] = {}
    for sample in iter_prepared_dataset_samples(manifest_path, max_samples=None):
        if sample.sample_id not in sample_ids:
            continue
        lookup[sample.sample_id] = sample
        if len(lookup) == len(sample_ids):
            break
    missing_sample_ids = sorted(sample_ids - set(lookup))
    if missing_sample_ids:
        raise ValueError(
            "Report references sample ids that are not present in the prepared dataset "
            f"manifest: {missing_sample_ids}"
        )
    return lookup


def _rewrite_report_in_place(*, run_root: Path, report) -> None:
    provisional = replace(report, report_path="", notes_path="", results_path="")
    report_path, notes_path, results_path = write_execution_artifacts(
        run_root=run_root,
        report=provisional,
    )
    finalized = replace(
        provisional,
        report_path=str(report_path),
        notes_path=str(notes_path),
        results_path=str(results_path),
    )
    write_execution_artifacts(run_root=run_root, report=finalized)


async def _revise_report_in_place(
    *,
    report_path: Path,
    provider_profiles_path: Path,
    config_path: Path,
    sample_filters: set[str],
    stream_output: bool,
) -> dict[str, Any]:
    report = _load_execution_report(report_path)
    run_root = report_path.parent.resolve()
    eligible_samples = _eligible_samples(report, sample_filters)
    if not eligible_samples:
        return {
            "report_path": str(report_path),
            "provider_id": report.summarizer_provider,
            "revised_sample_ids": [],
        }

    manifest_path = ensure_compact_prepared_dataset_manifest(
        Path(report.dataset_manifest_path).resolve()
    )
    load_prepared_dataset_manifest(manifest_path, include_samples=False)
    sample_lookup = _load_sample_lookup(
        manifest_path,
        {
            sample.sample_id
            for sample, _draft, _feedback, _diagnostics in eligible_samples
        },
    )
    profiles = load_provider_profiles(provider_profiles_path.resolve())
    summarizer_profile = _find_provider_profile(profiles, report.summarizer_provider)
    critic_profile = _find_provider_profile(profiles, report.critic_provider)
    judge_profile = _find_provider_profile(profiles, report.judge_provider)
    for role_name, profile in (
        ("summarizer", summarizer_profile),
        ("critic", critic_profile),
        ("judge", judge_profile),
    ):
        _validate_provider_role_constraints(profile, role=role_name)
        missing_env = missing_required_env(profile)
        if missing_env:
            raise ValueError(
                f"Provider '{profile.provider_id}' is missing required environment variables: {missing_env}"
            )

    base_cfg = _load_base_config(config_path.resolve())
    _configure_benchmark_logging(
        base_cfg,
        log_file=str((run_root / "agent_interactions.log").resolve()),
        print_to_terminal=not stream_output,
    )
    embedding_profile = _resolve_embedding_profile(base_cfg, report.embedding_mode)
    logging_enabled = bool(base_cfg.get("logging", {}).get("enabled", False))
    base_audio_cfg = _coerce_mapping(
        OmegaConf.to_container(base_cfg.audio, resolve=True)
    )
    benchmark_audio_cfg = (
        base_audio_cfg
        if report.transcript_only
        else _prepare_benchmark_audio_cfg(base_audio_cfg, run_root=run_root)
    )
    calibration = _build_stub_calibration(
        run_root=run_root,
        audio_cfg=benchmark_audio_cfg,
    )
    voice_clone_orchestrator = None
    if not report.transcript_only:
        voice_clone_orchestrator = build_voice_clone_orchestrator(
            benchmark_audio_cfg,
            project_root=REPO_ROOT,
        )
        if voice_clone_orchestrator is None:
            raise ValueError(
                "SPoRC post-critic revision requires audio.voice_clone.enabled=true "
                "unless the report is transcript-only."
            )

    summarizer_provider_config = _prepare_provider_config(
        summarizer_profile,
        role="summarizer",
        transcript_only=report.transcript_only,
    )
    critic_provider_config = _prepare_provider_config(
        critic_profile,
        role="critic",
        transcript_only=report.transcript_only,
    )
    judge_provider_config = _prepare_provider_config(
        judge_profile,
        role="judge",
        transcript_only=report.transcript_only,
    )
    runtime_args = argparse.Namespace(
        disable_stream_output=not stream_output,
        shared_retrieval_port=None,
        retrieval_namespace="default",
        retrieval_port=None,
        summarizer_port=None,
        critic_port=None,
        summarizer_timeout_seconds=None,
        critic_timeout_seconds=None,
        retrieval_timeout_seconds=None,
        index_timeout_seconds=None,
    )
    runtime = await _start_managed_runtime(
        args=runtime_args,
        run_root=run_root,
        base_cfg=base_cfg,
        summarizer_provider_config=summarizer_provider_config,
        critic_provider_config=critic_provider_config,
        judge_provider_config=judge_provider_config,
        embedding_profile=embedding_profile,
        transcript_only=report.transcript_only,
        calibration=calibration,
        voice_clone_orchestrator=voice_clone_orchestrator,
        logging_enabled=logging_enabled,
        force_dynamic_ports=True,
    )
    try:
        updated_results = {sample.sample_id: sample for sample in report.samples}
        revised_sample_ids: list[str] = []
        for (
            sample,
            previous_draft,
            critic_feedback,
            prior_diagnostics,
        ) in eligible_samples:
            prepared_sample = sample_lookup[sample.sample_id]
            prior_iterations_run = int(
                prior_diagnostics.get("iterations_run", sample.iterations_run) or 0
            )
            prior_max_iterations = prior_diagnostics.get("max_iterations")
            if prior_max_iterations is None:
                prior_max_iterations = sample.iterations_run or 1
            postprocess_feedback = (
                "No new critic pass was run after this salvage revision. The draft was "
                "revised from the prior saved feedback below, then the judge was rerun."
                f"\n\n{critic_feedback}"
            )
            revised_result = await _execute_sample(
                sample=prepared_sample,
                target_seconds=sample.target_seconds,
                preloaded_transcript=None,
                resume_draft_xml=previous_draft,
                resume_feedback=critic_feedback,
                orchestrator=runtime.orchestrator,
                event_capture=runtime.event_capture,
                retrieval_enabled=runtime.retrieval_enabled,
                duration_tolerance_ratio=report.duration_tolerance_ratio,
                max_iterations=0,
                judge=runtime.judge,
                calibration=calibration,
                run_root=run_root,
                artifact_recorder=runtime.artifact_recorder,
                audio_cfg=benchmark_audio_cfg,
                transcript_only=report.transcript_only,
                voice_clone_orchestrator=voice_clone_orchestrator,
                force_download_audio=False,
                diagnostics_overrides={
                    "final_status": _POSTPROCESS_FINAL_STATUS,
                    "final_feedback": postprocess_feedback,
                    "iterations_run": prior_iterations_run + 1,
                    "max_iterations": prior_max_iterations,
                    "converged": False,
                    "postprocess_revision_applied": True,
                    "seed_feedback": critic_feedback,
                },
            )
            updated_results[sample.sample_id] = replace(
                revised_result,
                converged=False,
                iterations_run=prior_iterations_run + 1,
                final_status=_POSTPROCESS_FINAL_STATUS,
                counts_toward_target=sample.counts_toward_target,
                replacement_reason=sample.replacement_reason,
                infra_retry_attempts=sample.infra_retry_attempts,
            )
            revised_sample_ids.append(sample.sample_id)

        ordered_results = [
            updated_results[sample.sample_id] for sample in report.samples
        ]
        summary = _summarize_results(
            ordered_results,
            target_sample_count=report.summary.target_sample_count,
        )
        status = _report_status_from_summary(summary)
        if summary.sample_count == 0 and report.sample_shard_count > 1:
            status = "completed"
        updated_report = replace(
            report,
            generated_at_utc=utc_now_iso(),
            status=status,
            samples=ordered_results,
            summary=summary,
        )
        _rewrite_report_in_place(run_root=run_root, report=updated_report)
        return {
            "report_path": str(report_path),
            "provider_id": report.summarizer_provider,
            "revised_sample_ids": revised_sample_ids,
        }
    finally:
        await _stop_managed_runtime(runtime)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    provider_filters = {
        str(value).strip() for value in args.provider if str(value).strip()
    }
    sample_filters = {
        str(value).strip() for value in args.sample_id if str(value).strip()
    }
    explicit_report_paths = [
        Path(str(path)).expanduser().resolve()
        for path in args.report_path
        if str(path).strip()
    ]
    session_payload: dict[str, Any] | None = None
    if args.session_id:
        session_payload = _load_session_payload(str(args.session_id))
        session_provider_profiles = str(
            session_payload.get("provider_profiles_path", "") or ""
        ).strip()
        provider_profiles_path = (
            Path(session_provider_profiles).resolve()
            if session_provider_profiles
            else Path(args.provider_profiles).resolve()
        )
    else:
        provider_profiles_path = Path(args.provider_profiles).resolve()

    report_targets: list[tuple[str, Path]] = [
        ("", path) for path in explicit_report_paths if path.exists()
    ]
    if session_payload is not None:
        report_targets.extend(
            _resolve_session_report_paths(
                session_payload=session_payload,
                provider_filters=provider_filters,
                allow_incomplete_steps=bool(args.allow_incomplete_steps),
            )
        )
    deduped_targets: list[Path] = []
    seen_targets: set[Path] = set()
    for _provider_id, report_path in report_targets:
        if report_path in seen_targets:
            continue
        seen_targets.add(report_path)
        deduped_targets.append(report_path)
    if not deduped_targets:
        raise SystemExit("No eligible report.json targets were found.")

    results: list[dict[str, Any]] = []
    for report_path in deduped_targets:
        result = asyncio.run(
            _revise_report_in_place(
                report_path=report_path,
                provider_profiles_path=provider_profiles_path,
                config_path=Path(args.config).resolve(),
                sample_filters=sample_filters,
                stream_output=bool(args.stream_output),
            )
        )
        results.append(result)
        revised_sample_ids = result["revised_sample_ids"]
        if revised_sample_ids:
            print(
                f"{result['provider_id']}: revised {len(revised_sample_ids)} sample(s) in {result['report_path']}"
            )
            for sample_id in revised_sample_ids:
                print(f"  - {sample_id}")
        else:
            print(
                f"{result['provider_id']}: no eligible samples in {result['report_path']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
