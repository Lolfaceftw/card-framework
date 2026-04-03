"""Run or validate the SPoRC budget benchmark plan from 2026-04-02."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from card_framework.benchmark.matrix import (
    MatrixConfigError,
    load_provider_profiles,
    missing_required_env,
    resolve_provider_config,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_FULL_DATASET_MANIFEST_PATH,
    DEFAULT_PREPARED_FULL_ROOT,
)
from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    DatasetPreparationError,
    load_prepared_dataset_manifest,
)
from card_framework.benchmark.types import ProviderProfile
from card_framework.shared.paths import DEFAULT_PROVIDER_PROFILES_PATH, REPO_ROOT


DEFAULT_BUDGET_DATASET_MANIFEST = (
    DEFAULT_PREPARED_FULL_ROOT / "benchmark_longform_15k_max3_manifest.json"
).resolve()
DEFAULT_BUDGET_AUDIT_PATH = (
    DEFAULT_PREPARED_FULL_ROOT / "benchmark_longform_15k_max3_audit.json"
).resolve()
DEFAULT_BUDGET_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts" / "summarizer_critic_benchmark" / "sporc_budget_runs"
).resolve()
DEFAULT_LIVE_CONTESTANTS = (
    "openai_gpt_5_4_mini",
    "anthropic_claude_haiku_4_5",
    "deepseek_chat",
    "vllm_default",
)
DEFAULT_DRY_RUN_CONTESTANTS = ("deepseek_chat",)
DEFAULT_LIVE_JUDGE_PROVIDER = "google_gemini_3_1_pro_preview_judge"
DEFAULT_DRY_RUN_JUDGE_PROVIDER = "deepseek_judge"
HF_ENV_NAMES = ("HF_ACCESS_TOKEN", "HF_TOKEN")
VLLM_MODEL_ENV_NAMES = ("CARD_FRAMEWORK_VLLM_MODEL", "VLLM_MODEL")
_DEEPSEEK_PROVIDER_TARGET = (
    "card_framework.providers.deepseek_provider.DeepSeekProvider"
)
_DEEPSEEK_REASONER_MODEL_PREFIX = "deepseek-reasoner"
_VLLM_PROVIDER_TARGET = "card_framework.providers.vllm_provider.VLLMProvider"


@dataclass(frozen=True)
class RequirementCheck:
    """One operator-facing prerequisite check."""

    label: str
    satisfied: bool
    detail: str


@dataclass(frozen=True)
class PlanStep:
    """One subprocess step in the budget workflow."""

    label: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class BudgetPlan:
    """Materialized budget-run workflow plus validation state."""

    mode: str
    contestants: tuple[str, ...]
    judge_provider: str
    dataset_manifest: Path
    source_manifest: Path
    audit_path: Path
    manifest_steps: tuple[PlanStep, ...]
    run_steps: tuple[PlanStep, ...]
    requirement_checks: tuple[RequirementCheck, ...]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or run the SPoRC budget plan from "
            "coder_docs/sporc_budget_reference_20260402.md."
        )
    )
    parser.add_argument(
        "--mode",
        choices=("validate", "dry-run", "live"),
        default="validate",
        help=(
            "validate prints the exact plan and checks prerequisites without API "
            "calls. dry-run executes a low-cost DeepSeek-backed validation pass. "
            "live executes the full budget plan."
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        default=str(DEFAULT_BUDGET_DATASET_MANIFEST),
        help="Curated <=3-speaker long-form manifest used by execute runs.",
    )
    parser.add_argument(
        "--source-manifest",
        default=str(DEFAULT_FULL_DATASET_MANIFEST_PATH),
        help=(
            "Prepared full SPoRC manifest used as the source for automatic long-form "
            "curation when --dataset-manifest does not exist."
        ),
    )
    parser.add_argument(
        "--audit-path",
        default=str(DEFAULT_BUDGET_AUDIT_PATH),
        help="Audit JSON path written when the long-form max-speaker manifest is curated.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_BUDGET_OUTPUT_ROOT),
        help="Parent output root. Each contestant gets its own execute output subtree.",
    )
    parser.add_argument(
        "--provider-profiles",
        default=str(DEFAULT_PROVIDER_PROFILES_PATH),
        help="Benchmark provider profile YAML.",
    )
    parser.add_argument(
        "--contestants",
        default="",
        help=(
            "Optional comma-separated contestant provider ids. Defaults to the full "
            "budget plan in live/validate mode and to deepseek_chat in dry-run mode."
        ),
    )
    parser.add_argument(
        "--judge-provider",
        default="",
        help=(
            "Optional judge provider id. Defaults to google_gemini_3_1_pro_preview_judge "
            "for live/validate and deepseek_judge for dry-run."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help=(
            "Retained sample count per contestant. Defaults to 10 for validate/live and "
            "1 for dry-run."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed used for deterministic sample ordering.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help="Summarizer-critic iteration cap for each execute run.",
    )
    parser.add_argument(
        "--skip-vllm-server-check",
        action="store_true",
        help="Skip the local /models probe for vllm_default after model pin validation.",
    )
    return parser


def _parse_provider_id_list(raw_value: str) -> tuple[str, ...]:
    provider_ids: list[str] = []
    seen: set[str] = set()
    for raw_item in str(raw_value or "").split(","):
        provider_id = raw_item.strip()
        if not provider_id or provider_id in seen:
            continue
        provider_ids.append(provider_id)
        seen.add(provider_id)
    return tuple(provider_ids)


def _default_contestants_for_mode(mode: str) -> tuple[str, ...]:
    if mode == "dry-run":
        return DEFAULT_DRY_RUN_CONTESTANTS
    return DEFAULT_LIVE_CONTESTANTS


def _default_judge_for_mode(mode: str) -> str:
    if mode == "dry-run":
        return DEFAULT_DRY_RUN_JUDGE_PROVIDER
    return DEFAULT_LIVE_JUDGE_PROVIDER


def _resolve_contestants(args: argparse.Namespace) -> tuple[str, ...]:
    explicit = _parse_provider_id_list(str(args.contestants or ""))
    return explicit or _default_contestants_for_mode(str(args.mode))


def _resolve_judge_provider(args: argparse.Namespace) -> str:
    explicit = str(args.judge_provider or "").strip()
    return explicit or _default_judge_for_mode(str(args.mode))


def _resolve_max_samples(args: argparse.Namespace) -> int:
    if int(args.max_samples) > 0:
        return int(args.max_samples)
    return 1 if str(args.mode) == "dry-run" else 10


def _provider_index(provider_profiles_path: Path) -> dict[str, ProviderProfile]:
    profiles = load_provider_profiles(provider_profiles_path)
    return {profile.provider_id: profile for profile in profiles}


def _build_prepare_full_dataset_command(source_manifest: Path) -> PlanStep:
    return PlanStep(
        label="Prepare full SPoRC dataset manifest",
        command=(
            sys.executable,
            "-m",
            "card_framework.benchmark.summarizer_critic.sporc",
            "prepare-dataset",
            "--dataset-variant",
            "full",
            "--prepared-root",
            str(source_manifest.parent),
        ),
    )


def _build_curate_longform_command(
    *,
    source_manifest: Path,
    dataset_manifest: Path,
    audit_path: Path,
) -> PlanStep:
    return PlanStep(
        label="Curate the <=3-speaker long-form budget manifest",
        command=(
            sys.executable,
            "-m",
            "card_framework.benchmark.summarizer_critic.sporc",
            "curate-longform",
            "--source-manifest",
            str(source_manifest),
            "--output-manifest",
            str(dataset_manifest),
            "--audit-path",
            str(audit_path),
            "--max-speakers",
            "3",
        ),
    )


def _build_execute_command(
    *,
    contestant_provider: str,
    judge_provider: str,
    dataset_manifest: Path,
    output_root: Path,
    max_samples: int,
    sample_seed: int,
    max_iterations: int,
) -> PlanStep:
    contestant_output_root = (output_root / contestant_provider).resolve()
    return PlanStep(
        label=f"Run execute for contestant {contestant_provider}",
        command=(
            sys.executable,
            "-m",
            "card_framework.benchmark.summarizer_critic.sporc",
            "execute",
            "--dataset-manifest",
            str(dataset_manifest),
            "--output-root",
            str(contestant_output_root),
            "--summarizer-provider",
            contestant_provider,
            "--critic-provider",
            contestant_provider,
            "--judge-provider",
            judge_provider,
            "--embedding-mode",
            "disabled",
            "--transcript-only",
            "--duration-preset",
            "5m",
            "--max-samples",
            str(max_samples),
            "--sample-seed",
            str(sample_seed),
            "--max-iterations",
            str(max_iterations),
        ),
    )


def _build_manifest_steps(
    *,
    dataset_manifest: Path,
    source_manifest: Path,
    audit_path: Path,
) -> tuple[PlanStep, ...]:
    dataset_ready, _ = _prepared_manifest_check(dataset_manifest)
    source_ready, _ = _prepared_manifest_check(source_manifest)
    if dataset_ready:
        return ()

    steps: list[PlanStep] = []
    if not source_ready:
        steps.append(_build_prepare_full_dataset_command(source_manifest))
    steps.append(
        _build_curate_longform_command(
            source_manifest=source_manifest,
            dataset_manifest=dataset_manifest,
            audit_path=audit_path,
        )
    )
    return tuple(steps)


def _deepseek_reasoner_is_invalid_tool_loop(profile: ProviderProfile) -> bool:
    provider_target = str(profile.llm_config.get("_target_", "") or "").strip()
    model_name = str(profile.llm_config.get("model", "") or "").strip().lower()
    return provider_target == _DEEPSEEK_PROVIDER_TARGET and model_name.startswith(
        _DEEPSEEK_REASONER_MODEL_PREFIX
    )


def _resolve_vllm_runtime_details(
    profile: ProviderProfile,
) -> tuple[str, str, str]:
    resolved = resolve_provider_config(profile)
    return (
        str(resolved.get("base_url", "") or "").strip(),
        str(resolved.get("api_key", "") or "").strip(),
        str(resolved.get("model", "") or "").strip(),
    )


def _probe_openai_compatible_models(
    *,
    base_url: str,
    api_key: str,
    timeout_seconds: float = 5.0,
) -> tuple[bool, str, tuple[str, ...]]:
    models_url = f"{base_url.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(models_url, headers=headers)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except HTTPError as exc:
        return False, f"{models_url} returned HTTP {exc.code}.", ()
    except URLError as exc:
        return False, f"Unable to reach {models_url}: {exc.reason}", ()
    except TimeoutError:
        return False, f"Timed out while probing {models_url}.", ()
    except json.JSONDecodeError as exc:
        return False, f"{models_url} did not return valid JSON: {exc}", ()

    raw_models = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = tuple(
        sorted(
            str(item.get("id", "") or "").strip()
            for item in raw_models
            if isinstance(item, dict) and str(item.get("id", "") or "").strip()
        )
    )
    if not model_ids:
        return False, f"{models_url} returned no served model ids.", ()
    return True, f"Detected served model ids from {models_url}.", model_ids


def _hf_token_present() -> bool:
    return any(str(os.getenv(env_name, "") or "").strip() for env_name in HF_ENV_NAMES)


def _prepared_manifest_check(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"Prepared manifest does not exist: {path}"
    try:
        manifest = load_prepared_dataset_manifest(path, include_samples=False)
    except (DatasetPreparationError, OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"Prepared manifest is not loadable: {path} ({exc})"
    return (
        True,
        f"Prepared manifest is loadable with sample_count={manifest.sample_count}: {path}",
    )


def _build_requirement_checks(
    *,
    contestants: tuple[str, ...],
    judge_provider: str,
    profiles: dict[str, ProviderProfile],
    dataset_manifest: Path,
    source_manifest: Path,
    skip_vllm_server_check: bool,
) -> tuple[RequirementCheck, ...]:
    checks: list[RequirementCheck] = []

    dataset_ready, dataset_detail = _prepared_manifest_check(dataset_manifest)
    source_ready, source_detail = _prepared_manifest_check(source_manifest)
    manifest_ready_or_buildable = dataset_ready or source_ready or _hf_token_present()
    manifest_detail = (
        dataset_detail
        if dataset_ready
        else (
            f"{dataset_detail} {source_detail} The wrapper can curate the <=3-speaker subset locally."
            if source_ready
            else (
                f"{dataset_manifest} and {source_manifest} are both missing, but an HF token "
                "is present so the script can prepare the full dataset then curate the budget "
                "subset."
                if _hf_token_present()
                else (
                    f"{dataset_manifest} and {source_manifest} are both missing. Set HF_ACCESS_TOKEN "
                    "or HF_TOKEN so the script can run prepare-dataset --dataset-variant full."
                )
            )
        )
    )
    checks.append(
        RequirementCheck(
            label="Budget manifest ready or buildable",
            satisfied=manifest_ready_or_buildable,
            detail=manifest_detail,
        )
    )

    provider_ids = [*contestants, judge_provider]
    for provider_id in provider_ids:
        profile = profiles.get(provider_id)
        checks.append(
            RequirementCheck(
                label=f"Provider profile {provider_id}",
                satisfied=profile is not None,
                detail=(
                    f"Provider profile '{provider_id}' is available."
                    if profile is not None
                    else f"Provider profile '{provider_id}' is missing from the benchmark provider YAML."
                ),
            )
        )
        if profile is None:
            continue

        missing_env = missing_required_env(profile)
        checks.append(
            RequirementCheck(
                label=f"Credentials for {provider_id}",
                satisfied=not missing_env,
                detail=(
                    f"All required environment variables are present for '{provider_id}'."
                    if not missing_env
                    else (
                        f"Provider '{provider_id}' is missing required environment variables: "
                        f"{', '.join(missing_env)}."
                    )
                ),
            )
        )

    for provider_id in contestants:
        profile = profiles.get(provider_id)
        if profile is None:
            continue
        invalid_reasoner = _deepseek_reasoner_is_invalid_tool_loop(profile)
        checks.append(
            RequirementCheck(
                label=f"Tool-loop compatibility for {provider_id}",
                satisfied=not invalid_reasoner,
                detail=(
                    f"'{provider_id}' can participate in SPoRC summarizer/critic tool loops."
                    if not invalid_reasoner
                    else (
                        f"'{provider_id}' resolves to deepseek-reasoner, and DeepSeek documents "
                        "that model as not supporting function calling. Use deepseek_chat for "
                        "summarizer/critic and keep deepseek_judge as judge-only."
                    )
                ),
            )
        )

        provider_target = str(profile.llm_config.get("_target_", "") or "").strip()
        if provider_target != _VLLM_PROVIDER_TARGET:
            continue
        base_url, api_key, model_name = _resolve_vllm_runtime_details(profile)
        model_pinned = bool(model_name)
        checks.append(
            RequirementCheck(
                label=f"Pinned vLLM model for {provider_id}",
                satisfied=model_pinned,
                detail=(
                    f"Local vLLM contestant is pinned to served model '{model_name}'."
                    if model_pinned
                    else (
                        f"'{provider_id}' has no explicit model pin. Set "
                        "CARD_FRAMEWORK_VLLM_MODEL or VLLM_MODEL before the budget run so the "
                        "contestant is reproducible."
                    )
                ),
            )
        )
        if not model_pinned or skip_vllm_server_check:
            continue
        reachable, detail, model_ids = _probe_openai_compatible_models(
            base_url=base_url,
            api_key=api_key,
        )
        checks.append(
            RequirementCheck(
                label=f"Reachable vLLM server for {provider_id}",
                satisfied=reachable,
                detail=detail,
            )
        )
        if reachable:
            checks.append(
                RequirementCheck(
                    label=f"Served vLLM model for {provider_id}",
                    satisfied=model_name in model_ids,
                    detail=(
                        f"Served model ids include '{model_name}'."
                        if model_name in model_ids
                        else (
                            f"The vLLM server at {base_url} does not expose '{model_name}'. "
                            f"Served ids: {', '.join(model_ids)}."
                        )
                    ),
                )
            )

    return tuple(checks)


def _build_budget_plan(args: argparse.Namespace) -> BudgetPlan:
    dataset_manifest = Path(str(args.dataset_manifest)).resolve()
    source_manifest = Path(str(args.source_manifest)).resolve()
    audit_path = Path(str(args.audit_path)).resolve()
    output_root = Path(str(args.output_root)).resolve()
    contestants = _resolve_contestants(args)
    judge_provider = _resolve_judge_provider(args)
    max_samples = _resolve_max_samples(args)
    profiles = _provider_index(Path(str(args.provider_profiles)).resolve())
    manifest_steps = _build_manifest_steps(
        dataset_manifest=dataset_manifest,
        source_manifest=source_manifest,
        audit_path=audit_path,
    )
    run_steps = tuple(
        _build_execute_command(
            contestant_provider=contestant_provider,
            judge_provider=judge_provider,
            dataset_manifest=dataset_manifest,
            output_root=output_root,
            max_samples=max_samples,
            sample_seed=int(args.sample_seed),
            max_iterations=int(args.max_iterations),
        )
        for contestant_provider in contestants
    )
    requirement_checks = _build_requirement_checks(
        contestants=contestants,
        judge_provider=judge_provider,
        profiles=profiles,
        dataset_manifest=dataset_manifest,
        source_manifest=source_manifest,
        skip_vllm_server_check=bool(args.skip_vllm_server_check),
    )
    return BudgetPlan(
        mode=str(args.mode),
        contestants=contestants,
        judge_provider=judge_provider,
        dataset_manifest=dataset_manifest,
        source_manifest=source_manifest,
        audit_path=audit_path,
        manifest_steps=manifest_steps,
        run_steps=run_steps,
        requirement_checks=requirement_checks,
    )


def _format_command(command: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(command))


def _print_plan(plan: BudgetPlan) -> None:
    print(f"Mode: {plan.mode}")
    print(f"Contestants: {', '.join(plan.contestants)}")
    print(f"Judge: {plan.judge_provider}")
    print(f"Dataset manifest: {plan.dataset_manifest}")
    print()
    print("Requirement checks:")
    for check in plan.requirement_checks:
        status = "OK" if check.satisfied else "MISSING"
        print(f"- [{status}] {check.label}: {check.detail}")
    print()
    if plan.manifest_steps:
        print("Manifest preparation steps:")
        for step in plan.manifest_steps:
            print(f"- {step.label}")
            print(f"  {_format_command(step.command)}")
        print()
    print("Benchmark commands:")
    for step in plan.run_steps:
        print(f"- {step.label}")
        print(f"  {_format_command(step.command)}")


def _run_step(step: PlanStep) -> int:
    print(f"\n==> {step.label}")
    print(_format_command(step.command))
    completed = subprocess.run(step.command, cwd=str(REPO_ROOT), check=False)
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        plan = _build_budget_plan(args)
    except MatrixConfigError as exc:
        parser.error(str(exc))
        return 2

    _print_plan(plan)
    missing_checks = [check for check in plan.requirement_checks if not check.satisfied]
    if missing_checks:
        print(
            f"\nCannot continue: {len(missing_checks)} prerequisite(s) are still missing."
        )
        return 1

    if str(args.mode) == "validate":
        print("\nValidation succeeded. No benchmark subprocesses were started.")
        return 0

    for step in (*plan.manifest_steps, *plan.run_steps):
        exit_code = _run_step(step)
        if exit_code != 0:
            print(f"\nStopping after failed step '{step.label}' (exit {exit_code}).")
            return exit_code

    print("\nBudget workflow completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
