"""Coverage for the SPoRC budget benchmark wrapper."""

from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
import time

import pytest

from card_framework.shared.paths import REPO_ROOT


def _load_module():
    script_path = (REPO_ROOT / "scripts" / "run_sporc_budget_sweep.py").resolve()
    spec = importlib.util.spec_from_file_location("run_sporc_budget_sweep", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_prepared_manifest(
    path: Path,
    *,
    budget_ready: bool,
) -> None:
    payload: dict[str, object] = {
        "dataset_id": "sporc_full_longform",
        "sporc_dataset_id": "blitt/SPoRC",
        "dataset_variant": "full",
        "episode_data_path": str((path.parent / "episode.jsonl.gz").resolve()),
        "turn_data_path": str((path.parent / "turn.jsonl.gz").resolve()),
        "sample_count": 1,
        "samples": [
            {
                "sample_id": "sporc_00001_example",
                "dataset": "sporc_full",
                "episode_key": "episode-001",
                "transcript_path": str((path.parent / "transcript.json").resolve()),
                "reference_summary": "reference",
                "target_seconds": 300,
                "metadata": {},
            }
        ],
    }
    if budget_ready:
        payload["curation_thresholds"] = {"max_unique_speaker_count": 3}
        payload["selected_max_unique_speaker_count"] = 3
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_budget_plan_defaults_to_live_budget_matrix(tmp_path: Path) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)

    parser = module._build_parser()
    args = parser.parse_args(["--dataset-manifest", str(dataset_manifest)])

    plan = module._build_budget_plan(args)

    assert plan.mode == "validate"
    assert plan.contestants == (
        "openai_gpt_5_4_mini",
        "anthropic_claude_haiku_4_5",
        "deepseek_chat",
        "vllm_default",
    )
    assert plan.judge_provider == "google_gemini_3_1_pro_preview_judge"
    assert any(
        step.command[step.command.index("--max-samples") + 1] == "10"
        for step in plan.run_steps
    )
    assert any(
        step.command[step.command.index("--sample-seed") + 1] == "42"
        for step in plan.run_steps
    )
    assert all("--provider-profiles" in step.command for step in plan.run_steps)


def test_validate_mode_marks_live_credentials_as_later_requirements(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    for env_name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "CARD_FRAMEWORK_VLLM_MODEL",
        "VLLM_MODEL",
    ):
        monkeypatch.delenv(env_name, raising=False)

    plan = module._build_budget_plan(
        module._build_parser().parse_args(
            [
                "--dataset-manifest",
                str(dataset_manifest),
                "--skip-vllm-server-check",
            ]
        )
    )

    later_checks = [check for check in plan.requirement_checks if not check.blocking]
    assert later_checks
    later_by_label = {check.label: check for check in later_checks}
    assert later_by_label["Credentials for openai_gpt_5_4_mini"].satisfied is False
    assert (
        later_by_label["Credentials for anthropic_claude_haiku_4_5"].satisfied is False
    )
    assert later_by_label["Credentials for deepseek_chat"].satisfied is False
    assert (
        later_by_label["Credentials for google_gemini_3_1_pro_preview_judge"].satisfied
        is False
    )
    assert later_by_label["Pinned vLLM model for vllm_default"].satisfied is False


def test_build_budget_plan_dry_run_defaults_to_vllm_pair(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen/Qwen3.5-27B")

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "dry-run",
            "--dataset-manifest",
            str(dataset_manifest),
            "--skip-vllm-server-check",
        ]
    )

    plan = module._build_budget_plan(args)

    assert plan.contestants == ("vllm_default",)
    assert plan.judge_provider == "vllm_default"
    run_step = plan.run_steps[0]
    assert run_step.command[run_step.command.index("--max-samples") + 1] == "1"
    assert (
        run_step.command[run_step.command.index("--judge-provider") + 1]
        == "vllm_default"
    )
    qwen_check = next(
        check
        for check in plan.requirement_checks
        if check.label == "Qwen/Qwen3.5-27B dry-run model for vllm_default"
    )
    assert qwen_check.satisfied is True


def test_build_budget_plan_propagates_custom_provider_profiles_path(
    tmp_path: Path,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    provider_profiles = (tmp_path / "profiles.yaml").resolve()
    provider_profiles.write_text(
        "providers:\n"
        "  - id: deepseek_chat\n"
        "    description: test\n"
        "    required_env: [DEEPSEEK_API_KEY]\n"
        "    llm:\n"
        "      _target_: card_framework.providers.deepseek_provider.DeepSeekProvider\n"
        "      api_key: $ENV{DEEPSEEK_API_KEY}\n"
        "      model: deepseek-chat\n"
        "      base_url: https://api.deepseek.com\n"
        "  - id: deepseek_judge\n"
        "    description: test\n"
        "    required_env: [DEEPSEEK_API_KEY]\n"
        "    llm:\n"
        "      _target_: card_framework.providers.deepseek_provider.DeepSeekProvider\n"
        "      api_key: $ENV{DEEPSEEK_API_KEY}\n"
        "      model: deepseek-reasoner\n"
        "      base_url: https://api.deepseek.com\n",
        encoding="utf-8",
    )

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "dry-run",
            "--dataset-manifest",
            str(dataset_manifest),
            "--provider-profiles",
            str(provider_profiles),
            "--contestants",
            "deepseek_chat",
            "--judge-provider",
            "deepseek_judge",
        ]
    )

    plan = module._build_budget_plan(args)

    run_step = plan.run_steps[0]
    assert run_step.command[run_step.command.index("--provider-profiles") + 1] == str(
        provider_profiles
    )


def test_parser_accepts_zero_max_iterations() -> None:
    module = _load_module()

    args = module._build_parser().parse_args(["--max-iterations", "0"])

    assert args.max_iterations == 0


def test_build_budget_plan_requires_hf_token_when_manifests_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    source_manifest = (tmp_path / "prepared_full" / "manifest.json").resolve()
    audit_path = (tmp_path / "audit.json").resolve()
    monkeypatch.delenv("HF_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(dataset_manifest),
            "--source-manifest",
            str(source_manifest),
            "--audit-path",
            str(audit_path),
        ]
    )

    plan = module._build_budget_plan(args)

    assert len(plan.manifest_steps) == 2
    manifest_check = next(
        check
        for check in plan.requirement_checks
        if check.label == "Budget manifest ready or buildable"
    )
    assert manifest_check.satisfied is False
    assert "HF_ACCESS_TOKEN" in manifest_check.detail


def test_build_budget_plan_marks_vllm_pin_satisfied_when_env_model_is_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen/Qwen3.5-27B")

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "dry-run",
            "--dataset-manifest",
            str(dataset_manifest),
            "--contestants",
            "deepseek_chat,vllm_default",
            "--skip-vllm-server-check",
        ]
    )

    plan = module._build_budget_plan(args)

    pin_check = next(
        check
        for check in plan.requirement_checks
        if check.label == "Pinned vLLM model for vllm_default"
    )
    assert pin_check.satisfied is True
    assert "Qwen/Qwen3.5-27B" in pin_check.detail
    qwen_check = next(
        check
        for check in plan.requirement_checks
        if check.label == "Qwen/Qwen3.5-27B dry-run model for vllm_default"
    )
    assert qwen_check.satisfied is True


def test_build_budget_plan_validates_vllm_judge_pin_when_judge_is_local(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen/Qwen3.5-27B")

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "dry-run",
            "--dataset-manifest",
            str(dataset_manifest),
            "--contestants",
            "deepseek_chat",
            "--judge-provider",
            "vllm_default",
            "--skip-vllm-server-check",
        ]
    )

    plan = module._build_budget_plan(args)

    pin_check = next(
        check
        for check in plan.requirement_checks
        if check.label == "Pinned vLLM model for vllm_default"
    )
    assert pin_check.satisfied is True
    assert "Qwen/Qwen3.5-27B" in pin_check.detail
    qwen_check = next(
        check
        for check in plan.requirement_checks
        if check.label == "Qwen/Qwen3.5-27B dry-run model for vllm_default"
    )
    assert qwen_check.satisfied is True


def test_build_budget_plan_dry_run_flags_non_qwen_vllm_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen3-32B")

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "dry-run",
            "--dataset-manifest",
            str(dataset_manifest),
            "--skip-vllm-server-check",
        ]
    )

    plan = module._build_budget_plan(args)

    qwen_check = next(
        check
        for check in plan.requirement_checks
        if check.label == "Qwen/Qwen3.5-27B dry-run model for vllm_default"
    )
    assert qwen_check.satisfied is False
    assert "Qwen3-32B" in qwen_check.detail


def test_build_budget_plan_handles_invalid_vllm_base_url_without_crashing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    provider_profiles = (tmp_path / "profiles.yaml").resolve()
    provider_profiles.write_text(
        "providers:\n"
        "  - id: deepseek_chat\n"
        "    description: test\n"
        "    required_env: [DEEPSEEK_API_KEY]\n"
        "    llm:\n"
        "      _target_: card_framework.providers.deepseek_provider.DeepSeekProvider\n"
        "      api_key: $ENV{DEEPSEEK_API_KEY}\n"
        "      model: deepseek-chat\n"
        "      base_url: https://api.deepseek.com\n"
        "  - id: deepseek_judge\n"
        "    description: test\n"
        "    required_env: [DEEPSEEK_API_KEY]\n"
        "    llm:\n"
        "      _target_: card_framework.providers.deepseek_provider.DeepSeekProvider\n"
        "      api_key: $ENV{DEEPSEEK_API_KEY}\n"
        "      model: deepseek-reasoner\n"
        "      base_url: https://api.deepseek.com\n"
        "  - id: vllm_default\n"
        "    description: test\n"
        "    required_env: []\n"
        "    llm:\n"
        "      _target_: card_framework.providers.vllm_provider.VLLMProvider\n"
        "      base_url: http://://bad-url\n"
        "      api_key: EMPTY\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen3-32B")
    for env_name in ("VLLM_URL", "CARD_FRAMEWORK_VLLM_URL", "VLLM_BASE_URL"):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_BASE_URL", "://bad-url")

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--mode",
            "dry-run",
            "--dataset-manifest",
            str(dataset_manifest),
            "--provider-profiles",
            str(provider_profiles),
            "--contestants",
            "deepseek_chat,vllm_default",
        ]
    )

    plan = module._build_budget_plan(args)

    reachability_check = next(
        check
        for check in plan.requirement_checks
        if check.label == "Reachable vLLM server for vllm_default"
    )
    assert reachability_check.satisfied is False
    assert (
        "Invalid vLLM base URL" in reachability_check.detail
        or "Unable to reach" in reachability_check.detail
    )


def test_main_validate_succeeds_for_vllm_only_dry_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen/Qwen3.5-27B")
    monkeypatch.setenv("VLLM_URL", "http://127.0.0.1:8000/v1")

    exit_code = module.main(
        [
            "--mode",
            "validate",
            "--dataset-manifest",
            str(dataset_manifest),
            "--contestants",
            "vllm_default",
            "--judge-provider",
            "vllm_default",
            "--skip-vllm-server-check",
        ]
    )

    assert exit_code == 0


def test_main_validate_succeeds_for_default_prefunding_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    for env_name in (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "GEMINI_API_KEY",
        "CARD_FRAMEWORK_VLLM_MODEL",
        "VLLM_MODEL",
    ):
        monkeypatch.delenv(env_name, raising=False)

    exit_code = module.main(
        [
            "--mode",
            "validate",
            "--dataset-manifest",
            str(dataset_manifest),
            "--skip-vllm-server-check",
        ]
    )

    assert exit_code == 0


def test_invalid_existing_dataset_manifest_does_not_count_as_ready(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.md").resolve()
    dataset_manifest.write_text("not a manifest", encoding="utf-8")
    source_manifest = (tmp_path / "prepared_full" / "manifest.json").resolve()
    monkeypatch.delenv("HF_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(dataset_manifest),
            "--source-manifest",
            str(source_manifest),
        ]
    )

    plan = module._build_budget_plan(args)

    manifest_check = next(
        check
        for check in plan.requirement_checks
        if check.label == "Budget manifest ready or buildable"
    )
    assert manifest_check.satisfied is False
    assert len(plan.manifest_steps) == 2


def test_invalid_budget_manifest_is_rebuilt_from_valid_full_manifest(
    tmp_path: Path,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    dataset_manifest.write_text("not valid json", encoding="utf-8")
    source_manifest = (tmp_path / "prepared_full" / "manifest.json").resolve()
    _write_prepared_manifest(source_manifest, budget_ready=False)

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(dataset_manifest),
            "--source-manifest",
            str(source_manifest),
        ]
    )

    plan = module._build_budget_plan(args)

    assert len(plan.manifest_steps) == 1
    assert (
        plan.manifest_steps[0].label
        == "Curate the <=3-speaker long-form budget manifest"
    )


def test_missing_custom_source_manifest_falls_back_to_generated_manifest_name(
    tmp_path: Path,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    source_manifest = (tmp_path / "prepared_full" / "custom_name.json").resolve()

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(dataset_manifest),
            "--source-manifest",
            str(source_manifest),
        ]
    )

    plan = module._build_budget_plan(args)

    prepare_step = plan.manifest_steps[0]
    curate_step = plan.manifest_steps[1]
    generated_manifest = str((source_manifest.parent / "manifest.json").resolve())
    assert prepare_step.command[
        prepare_step.command.index("--prepared-root") + 1
    ] == str(source_manifest.parent)
    assert (
        curate_step.command[curate_step.command.index("--source-manifest") + 1]
        == generated_manifest
    )


def test_invalid_custom_source_manifest_falls_back_to_generated_manifest_name(
    tmp_path: Path,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    source_manifest = (tmp_path / "prepared_full" / "custom_name.json").resolve()
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text("not valid json", encoding="utf-8")

    parser = module._build_parser()
    args = parser.parse_args(
        [
            "--dataset-manifest",
            str(dataset_manifest),
            "--source-manifest",
            str(source_manifest),
        ]
    )

    plan = module._build_budget_plan(args)

    prepare_step = plan.manifest_steps[0]
    curate_step = plan.manifest_steps[1]
    generated_manifest = str((source_manifest.parent / "manifest.json").resolve())
    assert prepare_step.command[
        prepare_step.command.index("--prepared-root") + 1
    ] == str(source_manifest.parent)
    assert (
        curate_step.command[curate_step.command.index("--source-manifest") + 1]
        == generated_manifest
    )


def test_materialize_step_command_adds_checkpoint_and_verbose_progress_flags(
    tmp_path: Path,
) -> None:
    module = _load_module()
    checkpoint_path = (tmp_path / "provider.checkpoint.json").resolve()
    checkpoint_path.write_text("{}", encoding="utf-8")
    step = module.PlanStep(
        step_id="run:test_provider",
        label="Run execute for contestant test_provider",
        command=("python", "-m", "demo", "execute"),
        kind=module._RUN_STEP_KIND,
        provider_id="test_provider",
    )

    command = module._materialize_step_command(
        step=step,
        step_state={"checkpoint_path": str(checkpoint_path)},
    )

    assert "--checkpoint-path" in command
    assert str(checkpoint_path) in command
    assert "--resume-from-checkpoint" in command
    assert "--emit-progress-events" in command
    assert "--emit-progress-heartbeats" in command


def test_retryable_connection_failure_detects_report_retryable_statuses(
    tmp_path: Path,
) -> None:
    module = _load_module()
    report_path = (tmp_path / "report.json").resolve()
    report_path.write_text(
        json.dumps(
            {
                "status": "partial",
                "samples": [
                    {
                        "sample_id": "sporc_00001_example",
                        "final_status": "connection_unavailable",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = module.StepExecutionResult(
        exit_code=1,
        output_lines=("provider disconnected",),
        parsed_payloads=(({"status": "partial", "report_path": str(report_path)}),),
    )

    retry_strategy, reason = module._retry_strategy_for_step_result(result)

    assert retry_strategy == "restart"
    assert "retryable infra statuses" in reason


def test_retryable_connection_failure_ignores_mixed_status_reports(
    tmp_path: Path,
) -> None:
    module = _load_module()
    report_path = (tmp_path / "report.json").resolve()
    report_path.write_text(
        json.dumps(
            {
                "status": "partial",
                "samples": [
                    {
                        "sample_id": "sporc_00001_example",
                        "final_status": "connection_unavailable",
                    },
                    {
                        "sample_id": "sporc_00002_example",
                        "final_status": "max_iterations_reached",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    retryable, reason = module._load_retryable_report_failure(str(report_path))

    assert retryable is False
    assert reason == ""


def test_run_step_with_retries_restarts_from_fresh_checkpoint_after_retryable_report(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "DEFAULT_SESSION_ROOT", (tmp_path / "sessions").resolve()
    )
    step = module.PlanStep(
        step_id="run:vllm_default",
        label="Run execute for contestant vllm_default",
        command=("python", "-m", "demo", "execute"),
        kind=module._RUN_STEP_KIND,
        provider_id="vllm_default",
    )
    plan = module.BudgetPlan(
        mode="dry-run",
        contestants=("vllm_default",),
        judge_provider="vllm_default",
        dataset_manifest=tmp_path / "dataset.json",
        source_manifest=tmp_path / "source.json",
        audit_path=tmp_path / "audit.json",
        provider_profiles_path=tmp_path / "profiles.yaml",
        manifest_steps=(),
        run_steps=(step,),
        requirement_checks=(),
    )
    session_context = module.SessionContext(
        session_id="00000000-0000-0000-0000-000000000003",
        state_path=(
            tmp_path
            / "sessions"
            / "00000000-0000-0000-0000-000000000003"
            / "session.json"
        ).resolve(),
        resume_command="resume",
    )
    session_payload = module._build_session_payload(
        session_id=session_context.session_id,
        requested_argv=[],
        plan=plan,
    )
    initial_checkpoint = session_payload["steps"][step.step_id]["checkpoint_path"]
    observed_commands: list[tuple[str, ...]] = []

    def _fake_run_logged_subprocess(**kwargs):
        observed_commands.append(kwargs["command"])
        if len(observed_commands) == 1:
            report_path = (tmp_path / "retryable-report.json").resolve()
            report_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "samples": [
                            {
                                "sample_id": "sporc_00001_example",
                                "final_status": "connection_unavailable",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return module.StepExecutionResult(
                exit_code=0,
                output_lines=("first attempt",),
                parsed_payloads=(
                    ({"status": "completed", "report_path": str(report_path)}),
                ),
            )
        return module.StepExecutionResult(
            exit_code=0,
            output_lines=("second attempt",),
            parsed_payloads=(({"status": "completed", "report_path": ""}),),
        )

    monkeypatch.setattr(module, "_run_logged_subprocess", _fake_run_logged_subprocess)
    monkeypatch.setattr(
        module,
        "_wait_for_retry_window",
        lambda **kwargs: True,
    )
    exit_code = module._run_step_with_retries(
        step=step,
        plan=plan,
        session_context=session_context,
        session_payload=session_payload,
        state_lock=threading.Lock(),
        stop_event=threading.Event(),
        active_processes={},
        process_lock=threading.Lock(),
        profiles={},
        args=module._build_parser().parse_args([]),
    )

    assert exit_code == 0
    assert len(observed_commands) == 2
    assert "--resume-from-checkpoint" not in observed_commands[0]
    assert "--resume-from-checkpoint" not in observed_commands[1]
    assert str(initial_checkpoint) in observed_commands[0]
    assert str(initial_checkpoint) not in observed_commands[1]


def test_run_step_with_retries_hydrates_resume_report_and_reuses_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "DEFAULT_SESSION_ROOT", (tmp_path / "sessions").resolve()
    )
    output_root = (tmp_path / "runs" / "deepseek_chat").resolve()
    step = module.PlanStep(
        step_id="run:deepseek_chat",
        label="Run execute for contestant deepseek_chat",
        command=(
            "python",
            "-m",
            "demo",
            "execute",
            "--output-root",
            str(output_root),
        ),
        kind=module._RUN_STEP_KIND,
        provider_id="deepseek_chat",
    )
    plan = module.BudgetPlan(
        mode="live",
        contestants=("deepseek_chat",),
        judge_provider="judge",
        dataset_manifest=tmp_path / "dataset.json",
        source_manifest=tmp_path / "source.json",
        audit_path=tmp_path / "audit.json",
        provider_profiles_path=tmp_path / "profiles.yaml",
        manifest_steps=(),
        run_steps=(step,),
        requirement_checks=(),
    )
    session_context = module.SessionContext(
        session_id="00000000-0000-0000-0000-000000000005",
        state_path=(
            tmp_path
            / "sessions"
            / "00000000-0000-0000-0000-000000000005"
            / "session.json"
        ).resolve(),
        resume_command="resume",
    )
    session_payload = module._build_session_payload(
        session_id=session_context.session_id,
        requested_argv=[],
        plan=plan,
    )
    checkpoint_path = Path(session_payload["steps"][step.step_id]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("{}", encoding="utf-8")
    report_path = (output_root / "20260403T080044Z_pid1" / "report.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-04-03T08:07:50.687705+00:00",
                "run_id": "20260403T080044Z_pid1",
                "checkpoint_path": str(checkpoint_path),
                "samples": [
                    {
                        "sample_id": "sporc_00001_example",
                        "final_status": "max_iterations_reached",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    observed_commands: list[tuple[str, ...]] = []

    def _fake_run_logged_subprocess(**kwargs):
        observed_commands.append(kwargs["command"])
        return module.StepExecutionResult(
            exit_code=0,
            output_lines=("resume",),
            parsed_payloads=(),
        )

    monkeypatch.setattr(module, "_run_logged_subprocess", _fake_run_logged_subprocess)
    exit_code = module._run_step_with_retries(
        step=step,
        plan=plan,
        session_context=session_context,
        session_payload=session_payload,
        state_lock=threading.Lock(),
        stop_event=threading.Event(),
        active_processes={},
        process_lock=threading.Lock(),
        profiles={},
        args=module._build_parser().parse_args([]),
    )

    step_state = session_payload["steps"][step.step_id]
    assert exit_code == 0
    assert len(observed_commands) == 1
    assert str(checkpoint_path) in observed_commands[0]
    assert "--resume-from-checkpoint" in observed_commands[0]
    assert step_state["report_path"] == str(report_path)
    assert step_state["run_id"] == "20260403T080044Z_pid1"


def test_run_step_with_retries_preserves_mixed_resume_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "DEFAULT_SESSION_ROOT", (tmp_path / "sessions").resolve()
    )
    output_root = (tmp_path / "runs" / "vllm_default").resolve()
    step = module.PlanStep(
        step_id="run:vllm_default",
        label="Run execute for contestant vllm_default",
        command=(
            "python",
            "-m",
            "demo",
            "execute",
            "--output-root",
            str(output_root),
        ),
        kind=module._RUN_STEP_KIND,
        provider_id="vllm_default",
    )
    plan = module.BudgetPlan(
        mode="live",
        contestants=("vllm_default",),
        judge_provider="judge",
        dataset_manifest=tmp_path / "dataset.json",
        source_manifest=tmp_path / "source.json",
        audit_path=tmp_path / "audit.json",
        provider_profiles_path=tmp_path / "profiles.yaml",
        manifest_steps=(),
        run_steps=(step,),
        requirement_checks=(),
    )
    session_context = module.SessionContext(
        session_id="00000000-0000-0000-0000-000000000007",
        state_path=(
            tmp_path
            / "sessions"
            / "00000000-0000-0000-0000-000000000007"
            / "session.json"
        ).resolve(),
        resume_command="resume",
    )
    session_payload = module._build_session_payload(
        session_id=session_context.session_id,
        requested_argv=[],
        plan=plan,
    )
    checkpoint_path = Path(session_payload["steps"][step.step_id]["checkpoint_path"])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text("{}", encoding="utf-8")
    report_path = (output_root / "20260403T080044Z_pid1" / "report.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-04-03T08:10:07.637732+00:00",
                "run_id": "20260403T080044Z_pid1",
                "checkpoint_path": str(checkpoint_path),
                "samples": [
                    {
                        "sample_id": "sporc_00001_example",
                        "final_status": "connection_unavailable",
                    },
                    {
                        "sample_id": "sporc_00002_example",
                        "final_status": "max_iterations_reached",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    observed_commands: list[tuple[str, ...]] = []

    def _fake_run_logged_subprocess(**kwargs):
        observed_commands.append(kwargs["command"])
        return module.StepExecutionResult(
            exit_code=0,
            output_lines=("resume",),
            parsed_payloads=(),
        )

    monkeypatch.setattr(module, "_run_logged_subprocess", _fake_run_logged_subprocess)
    exit_code = module._run_step_with_retries(
        step=step,
        plan=plan,
        session_context=session_context,
        session_payload=session_payload,
        state_lock=threading.Lock(),
        stop_event=threading.Event(),
        active_processes={},
        process_lock=threading.Lock(),
        profiles={},
        args=module._build_parser().parse_args([]),
    )

    step_state = session_payload["steps"][step.step_id]
    assert exit_code == 0
    assert len(observed_commands) == 1
    assert str(checkpoint_path) in observed_commands[0]
    assert "--resume-from-checkpoint" in observed_commands[0]
    assert step_state["checkpoint_path"] == str(checkpoint_path)
    assert step_state["report_path"] == str(report_path)


def test_run_step_with_retries_discards_retryable_resume_checkpoint_before_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "DEFAULT_SESSION_ROOT", (tmp_path / "sessions").resolve()
    )
    output_root = (tmp_path / "runs" / "openai_gpt_5_4_mini").resolve()
    step = module.PlanStep(
        step_id="run:openai_gpt_5_4_mini",
        label="Run execute for contestant openai_gpt_5_4_mini",
        command=(
            "python",
            "-m",
            "demo",
            "execute",
            "--output-root",
            str(output_root),
        ),
        kind=module._RUN_STEP_KIND,
        provider_id="openai_gpt_5_4_mini",
    )
    plan = module.BudgetPlan(
        mode="live",
        contestants=("openai_gpt_5_4_mini",),
        judge_provider="judge",
        dataset_manifest=tmp_path / "dataset.json",
        source_manifest=tmp_path / "source.json",
        audit_path=tmp_path / "audit.json",
        provider_profiles_path=tmp_path / "profiles.yaml",
        manifest_steps=(),
        run_steps=(step,),
        requirement_checks=(),
    )
    session_context = module.SessionContext(
        session_id="00000000-0000-0000-0000-000000000006",
        state_path=(
            tmp_path
            / "sessions"
            / "00000000-0000-0000-0000-000000000006"
            / "session.json"
        ).resolve(),
        resume_command="resume",
    )
    session_payload = module._build_session_payload(
        session_id=session_context.session_id,
        requested_argv=[],
        plan=plan,
    )
    initial_checkpoint = Path(session_payload["steps"][step.step_id]["checkpoint_path"])
    initial_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    initial_checkpoint.write_text("{}", encoding="utf-8")
    session_payload["steps"][step.step_id]["attempt_count"] = 1
    report_path = (output_root / "20260403T080044Z_pid1" / "report.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-04-03T08:13:33.677251+00:00",
                "run_id": "20260403T080044Z_pid1",
                "checkpoint_path": str(initial_checkpoint),
                "samples": [
                    {
                        "sample_id": "sporc_00001_example",
                        "final_status": "startup_unavailable",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    observed_commands: list[tuple[str, ...]] = []

    def _fake_run_logged_subprocess(**kwargs):
        observed_commands.append(kwargs["command"])
        return module.StepExecutionResult(
            exit_code=0,
            output_lines=("restart",),
            parsed_payloads=(),
        )

    monkeypatch.setattr(module, "_run_logged_subprocess", _fake_run_logged_subprocess)
    exit_code = module._run_step_with_retries(
        step=step,
        plan=plan,
        session_context=session_context,
        session_payload=session_payload,
        state_lock=threading.Lock(),
        stop_event=threading.Event(),
        active_processes={},
        process_lock=threading.Lock(),
        profiles={},
        args=module._build_parser().parse_args([]),
    )

    step_state = session_payload["steps"][step.step_id]
    assert exit_code == 0
    assert len(observed_commands) == 1
    assert str(initial_checkpoint) not in observed_commands[0]
    assert "--resume-from-checkpoint" not in observed_commands[0]
    assert step_state["checkpoint_path"] != str(initial_checkpoint)
    assert step_state["report_path"] == ""
    assert step_state["run_id"] == ""
    assert str(step_state["checkpoint_path"]) in str(step_state["last_command"])


def test_collect_session_budget_profile_tracks_contestant_and_judge_costs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "DEFAULT_SESSION_ROOT", (tmp_path / "sessions").resolve()
    )
    output_root = (tmp_path / "runs" / "openai_gpt_5_4_mini").resolve()
    step = module.PlanStep(
        step_id="run:openai_gpt_5_4_mini",
        label="Run execute for contestant openai_gpt_5_4_mini",
        command=(
            "python",
            "-m",
            "demo",
            "execute",
            "--output-root",
            str(output_root),
        ),
        kind=module._RUN_STEP_KIND,
        provider_id="openai_gpt_5_4_mini",
    )
    plan = module.BudgetPlan(
        mode="live",
        contestants=("openai_gpt_5_4_mini",),
        judge_provider="google_gemini_3_1_pro_preview_judge",
        dataset_manifest=tmp_path / "dataset.json",
        source_manifest=tmp_path / "source.json",
        audit_path=tmp_path / "audit.json",
        provider_profiles_path=module.DEFAULT_PROVIDER_PROFILES_PATH,
        manifest_steps=(),
        run_steps=(step,),
        requirement_checks=(),
    )
    session_payload = module._build_session_payload(
        session_id="00000000-0000-0000-0000-000000000010",
        requested_argv=[],
        plan=plan,
    )
    run_root = (output_root / "20260403T091350Z_pid1").resolve()
    report_path = (run_root / "report.json").resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-04-03T09:13:50+00:00",
                "run_id": "20260403T091350Z_pid1",
                "samples": [],
            }
        ),
        encoding="utf-8",
    )
    session_payload["steps"][step.step_id]["report_path"] = str(report_path)
    session_payload["steps"][step.step_id]["run_id"] = "20260403T091350Z_pid1"
    summarizer_call = (
        run_root / "samples" / "sample-a" / "llm" / "summarizer" / "call_001.json"
    )
    summarizer_call.parent.mkdir(parents=True, exist_ok=True)
    summarizer_call.write_text(
        json.dumps(
            {
                "operation": "chat",
                "messages": [
                    {"role": "system", "content": "You are the summarizer."},
                    {"role": "user", "content": "Summarize the transcript."},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "save_draft",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "tool_choice": "auto",
                "max_tokens": 256,
                "response": {"content": "summary"},
            }
        ),
        encoding="utf-8",
    )
    judge_call = run_root / "samples" / "sample-a" / "llm" / "judge" / "call_001.json"
    judge_call.parent.mkdir(parents=True, exist_ok=True)
    judge_call.write_text(
        json.dumps(
            {
                "operation": "generate",
                "system_prompt": "You are the judge.",
                "user_prompt": "Score this summary.",
                "max_tokens": 64,
                "response_text": '{"Factualness": 1.0}',
            }
        ),
        encoding="utf-8",
    )
    rate_limited_call = (
        run_root / "samples" / "sample-a" / "llm" / "summarizer" / "call_002.json"
    )
    rate_limited_call.write_text(
        json.dumps(
            {
                "operation": "chat",
                "messages": [{"role": "user", "content": "retry me"}],
                "error": "Error code: 429 - Please try again in 7.309s.",
            }
        ),
        encoding="utf-8",
    )

    budget_profile = module._collect_session_budget_profile(
        plan=plan,
        session_payload=session_payload,
        profiles=module._provider_index(module.DEFAULT_PROVIDER_PROFILES_PATH),
    )

    openai_profile = budget_profile["providers"]["openai_gpt_5_4_mini"]
    judge_profile = budget_profile["providers"]["google_gemini_3_1_pro_preview_judge"]
    assert openai_profile["roles"] == {"summarizer": 2}
    assert judge_profile["roles"] == {"judge": 1}
    assert (
        openai_profile["attempted_input_tokens"]
        > openai_profile["billable_input_tokens"]
    )
    assert openai_profile["estimated_cost_usd"] > 0.0
    assert judge_profile["estimated_cost_usd"] > 0.0
    assert budget_profile["total_estimated_cost_usd"] == pytest.approx(
        openai_profile["estimated_cost_usd"] + judge_profile["estimated_cost_usd"],
        abs=1e-9,
    )


def test_resolve_session_context_round_trips_original_arguments(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "DEFAULT_SESSION_ROOT", (tmp_path / "sessions").resolve()
    )
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen/Qwen3.5-27B")
    requested_argv = [
        "--mode",
        "dry-run",
        "--dataset-manifest",
        str(dataset_manifest),
        "--skip-vllm-server-check",
    ]
    plan = module._build_budget_plan(module._build_parser().parse_args(requested_argv))

    session_context, existing_payload, effective_argv = module._resolve_session_context(
        resume_session_id="",
        plan=plan,
        requested_argv=requested_argv,
    )

    assert session_context is not None
    payload = module._ensure_session_payload(
        session_context=session_context,
        existing_payload=existing_payload,
        plan=plan,
        requested_argv=effective_argv,
    )
    resumed_context, resumed_payload, resumed_argv = module._resolve_session_context(
        resume_session_id=session_context.session_id,
        plan=plan,
        requested_argv=[],
    )

    assert payload["requested_argv"] == requested_argv
    assert resumed_context is not None
    assert resumed_payload is not None
    assert resumed_context.session_id == session_context.session_id
    assert resumed_argv == requested_argv


def test_run_pending_steps_skips_completed_session_steps(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "DEFAULT_SESSION_ROOT", (tmp_path / "sessions").resolve()
    )
    step_one = module.PlanStep(
        step_id="run:first",
        label="First",
        command=("python",),
        kind=module._RUN_STEP_KIND,
        provider_id="first",
    )
    step_two = module.PlanStep(
        step_id="run:second",
        label="Second",
        command=("python",),
        kind=module._RUN_STEP_KIND,
        provider_id="second",
    )
    plan = module.BudgetPlan(
        mode="live",
        contestants=("first", "second"),
        judge_provider="judge",
        dataset_manifest=tmp_path / "dataset.json",
        source_manifest=tmp_path / "source.json",
        audit_path=tmp_path / "audit.json",
        provider_profiles_path=tmp_path / "profiles.yaml",
        manifest_steps=(),
        run_steps=(step_one, step_two),
        requirement_checks=(),
    )
    session_context = module.SessionContext(
        session_id="00000000-0000-0000-0000-000000000001",
        state_path=(
            tmp_path
            / "sessions"
            / "00000000-0000-0000-0000-000000000001"
            / "session.json"
        ).resolve(),
        resume_command="resume",
    )
    session_payload = module._build_session_payload(
        session_id=session_context.session_id,
        requested_argv=[],
        plan=plan,
    )
    session_payload["steps"]["run:first"]["status"] = "completed"
    called_steps: list[str] = []

    def _fake_run_step_with_retries(**kwargs):
        called_steps.append(kwargs["step"].step_id)
        return 0

    monkeypatch.setattr(module, "_run_step_with_retries", _fake_run_step_with_retries)
    exit_code = module._run_pending_steps(
        steps=plan.run_steps,
        plan=plan,
        session_context=session_context,
        session_payload=session_payload,
        state_lock=threading.Lock(),
        stop_event=threading.Event(),
        active_processes={},
        process_lock=threading.Lock(),
        profiles={},
        args=module._build_parser().parse_args([]),
        parallel=False,
    )

    assert exit_code == 0
    assert called_steps == ["run:second"]


def test_run_pending_steps_parallelizes_provider_work_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "DEFAULT_SESSION_ROOT", (tmp_path / "sessions").resolve()
    )
    step_one = module.PlanStep(
        step_id="run:first",
        label="First",
        command=("python",),
        kind=module._RUN_STEP_KIND,
        provider_id="first",
    )
    step_two = module.PlanStep(
        step_id="run:second",
        label="Second",
        command=("python",),
        kind=module._RUN_STEP_KIND,
        provider_id="second",
    )
    plan = module.BudgetPlan(
        mode="live",
        contestants=("first", "second"),
        judge_provider="judge",
        dataset_manifest=tmp_path / "dataset.json",
        source_manifest=tmp_path / "source.json",
        audit_path=tmp_path / "audit.json",
        provider_profiles_path=tmp_path / "profiles.yaml",
        manifest_steps=(),
        run_steps=(step_one, step_two),
        requirement_checks=(),
    )
    session_context = module.SessionContext(
        session_id="00000000-0000-0000-0000-000000000002",
        state_path=(
            tmp_path
            / "sessions"
            / "00000000-0000-0000-0000-000000000002"
            / "session.json"
        ).resolve(),
        resume_command="resume",
    )
    session_payload = module._build_session_payload(
        session_id=session_context.session_id,
        requested_argv=[],
        plan=plan,
    )
    active_count = 0
    max_active = 0
    active_lock = threading.Lock()

    def _fake_run_step_with_retries(**kwargs):
        nonlocal active_count, max_active
        with active_lock:
            active_count += 1
            max_active = max(max_active, active_count)
        time.sleep(0.1)
        with active_lock:
            active_count -= 1
        return 0

    monkeypatch.setattr(module, "_run_step_with_retries", _fake_run_step_with_retries)
    exit_code = module._run_pending_steps(
        steps=plan.run_steps,
        plan=plan,
        session_context=session_context,
        session_payload=session_payload,
        state_lock=threading.Lock(),
        stop_event=threading.Event(),
        active_processes={},
        process_lock=threading.Lock(),
        profiles={},
        args=module._build_parser().parse_args([]),
        parallel=True,
    )

    assert exit_code == 0
    assert max_active == 2


def test_run_pending_steps_honors_max_concurrent_provider_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "DEFAULT_SESSION_ROOT", (tmp_path / "sessions").resolve()
    )
    step_one = module.PlanStep(
        step_id="run:first",
        label="First",
        command=("python",),
        kind=module._RUN_STEP_KIND,
        provider_id="first",
    )
    step_two = module.PlanStep(
        step_id="run:second",
        label="Second",
        command=("python",),
        kind=module._RUN_STEP_KIND,
        provider_id="second",
    )
    plan = module.BudgetPlan(
        mode="live",
        contestants=("first", "second"),
        judge_provider="judge",
        dataset_manifest=tmp_path / "dataset.json",
        source_manifest=tmp_path / "source.json",
        audit_path=tmp_path / "audit.json",
        provider_profiles_path=tmp_path / "profiles.yaml",
        manifest_steps=(),
        run_steps=(step_one, step_two),
        requirement_checks=(),
    )
    session_context = module.SessionContext(
        session_id="00000000-0000-0000-0000-000000000004",
        state_path=(
            tmp_path
            / "sessions"
            / "00000000-0000-0000-0000-000000000004"
            / "session.json"
        ).resolve(),
        resume_command="resume",
    )
    session_payload = module._build_session_payload(
        session_id=session_context.session_id,
        requested_argv=[],
        plan=plan,
    )
    active_count = 0
    max_active = 0
    active_lock = threading.Lock()

    def _fake_run_step_with_retries(**kwargs):
        nonlocal active_count, max_active
        with active_lock:
            active_count += 1
            max_active = max(max_active, active_count)
        time.sleep(0.1)
        with active_lock:
            active_count -= 1
        return 0

    monkeypatch.setattr(module, "_run_step_with_retries", _fake_run_step_with_retries)
    exit_code = module._run_pending_steps(
        steps=plan.run_steps,
        plan=plan,
        session_context=session_context,
        session_payload=session_payload,
        state_lock=threading.Lock(),
        stop_event=threading.Event(),
        active_processes={},
        process_lock=threading.Lock(),
        profiles={},
        args=module._build_parser().parse_args(["--max-concurrent-providers", "1"]),
        parallel=True,
    )

    assert exit_code == 0
    assert max_active == 1


def test_main_prints_resume_command_when_interrupted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    monkeypatch.setattr(
        module, "DEFAULT_SESSION_ROOT", (tmp_path / "sessions").resolve()
    )
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen/Qwen3.5-27B")
    monkeypatch.setenv("VLLM_URL", "http://127.0.0.1:8000/v1")

    def _fake_run_pending_steps(**kwargs):
        return 130

    monkeypatch.setattr(module, "_run_pending_steps", _fake_run_pending_steps)
    captured = io.StringIO()
    with redirect_stdout(captured):
        exit_code = module.main(
            [
                "--mode",
                "dry-run",
                "--dataset-manifest",
                str(dataset_manifest),
                "--skip-vllm-server-check",
            ]
        )

    assert exit_code == 130
    assert "--resume-session-id" in captured.getvalue()
