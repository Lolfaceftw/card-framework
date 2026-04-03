"""Coverage for the SPoRC budget benchmark wrapper."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

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


def test_build_budget_plan_dry_run_defaults_to_deepseek_pair(tmp_path: Path) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)

    parser = module._build_parser()
    args = parser.parse_args(
        ["--mode", "dry-run", "--dataset-manifest", str(dataset_manifest)]
    )

    plan = module._build_budget_plan(args)

    assert plan.contestants == ("deepseek_chat",)
    assert plan.judge_provider == "deepseek_judge"
    run_step = plan.run_steps[0]
    assert run_step.command[run_step.command.index("--max-samples") + 1] == "1"
    assert (
        run_step.command[run_step.command.index("--judge-provider") + 1]
        == "deepseek_judge"
    )


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
        ]
    )

    plan = module._build_budget_plan(args)

    run_step = plan.run_steps[0]
    assert run_step.command[run_step.command.index("--provider-profiles") + 1] == str(
        provider_profiles
    )


def test_parser_rejects_nonpositive_max_iterations() -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module._build_parser().parse_args(["--max-iterations", "0"])


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
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen3-32B")

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
    assert "Qwen3-32B" in pin_check.detail


def test_main_validate_succeeds_for_deepseek_only_dry_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    _write_prepared_manifest(dataset_manifest, budget_ready=True)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")

    exit_code = module.main(
        [
            "--mode",
            "validate",
            "--dataset-manifest",
            str(dataset_manifest),
            "--contestants",
            "deepseek_chat",
            "--judge-provider",
            "deepseek_judge",
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
