"""Coverage for the SPoRC budget benchmark wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

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


def test_build_budget_plan_defaults_to_live_budget_matrix(tmp_path: Path) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    dataset_manifest.write_text('{"samples": []}', encoding="utf-8")

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


def test_build_budget_plan_dry_run_defaults_to_deepseek_pair(tmp_path: Path) -> None:
    module = _load_module()
    dataset_manifest = (tmp_path / "budget_manifest.json").resolve()
    dataset_manifest.write_text('{"samples": []}', encoding="utf-8")

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
    dataset_manifest.write_text('{"samples": []}', encoding="utf-8")
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
    dataset_manifest.write_text('{"samples": []}', encoding="utf-8")
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
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text('{"samples": []}', encoding="utf-8")

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
