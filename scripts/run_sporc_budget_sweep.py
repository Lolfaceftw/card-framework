"""Run or validate the SPoRC budget benchmark plan from 2026-04-02."""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
from types import FrameType
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid

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
DEFAULT_SESSION_ROOT = (DEFAULT_BUDGET_OUTPUT_ROOT / "_sessions").resolve()
DEFAULT_LIVE_CONTESTANTS = (
    "openai_gpt_5_4_mini",
    "anthropic_claude_haiku_4_5",
    "deepseek_chat",
    "vllm_default",
)
DEFAULT_DRY_RUN_CONTESTANTS = ("vllm_default",)
DEFAULT_LIVE_JUDGE_PROVIDER = "google_gemini_3_1_pro_preview_judge"
DEFAULT_DRY_RUN_JUDGE_PROVIDER = "vllm_default"
HF_ENV_NAMES = ("HF_ACCESS_TOKEN", "HF_TOKEN")
VLLM_MODEL_ENV_NAMES = ("CARD_FRAMEWORK_VLLM_MODEL", "VLLM_MODEL")
_DEEPSEEK_PROVIDER_TARGET = (
    "card_framework.providers.deepseek_provider.DeepSeekProvider"
)
_DEEPSEEK_REASONER_MODEL_PREFIX = "deepseek-reasoner"
_OPENAI_PROVIDER_TARGET = "card_framework.providers.openai_provider.OpenAIProvider"
_ANTHROPIC_PROVIDER_TARGET = (
    "card_framework.providers.anthropic_provider.AnthropicProvider"
)
_GOOGLE_PROVIDER_TARGET = (
    "card_framework.providers.google_genai_provider.GoogleGenAIProvider"
)
_VLLM_PROVIDER_TARGET = "card_framework.providers.vllm_provider.VLLMProvider"
_SESSION_STATE_VERSION = 1
_RETRYABLE_CONNECTION_TOKENS = (
    "connecterror",
    "connecttimeout",
    "connection refused",
    "actively refused",
    "connection reset by peer",
    "connection aborted",
    "software caused connection abort",
    "all connection attempts failed",
    "forcibly closed by the remote host",
    "server not responding",
    "health check failed",
    "startup_unavailable",
    "connection_unavailable",
)
_RETRYABLE_STEP_FINAL_STATUSES = {
    "startup_unavailable",
    "connection_unavailable",
}
_RUN_STEP_KIND = "run"
_MANIFEST_STEP_KIND = "manifest"
_BUDGET_RATE_REFERENCE = "coder_docs/sporc_budget_reference_20260402.md"


@dataclass(frozen=True)
class RequirementCheck:
    """One operator-facing prerequisite check."""

    label: str
    satisfied: bool
    detail: str
    blocking: bool = True


@dataclass(frozen=True)
class PlanStep:
    """One subprocess step in the budget workflow."""

    step_id: str
    label: str
    command: tuple[str, ...]
    kind: str
    provider_id: str = ""
    checkpoint_path: str = ""


@dataclass(frozen=True)
class BudgetPlan:
    """Materialized budget-run workflow plus validation state."""

    mode: str
    contestants: tuple[str, ...]
    judge_provider: str
    dataset_manifest: Path
    source_manifest: Path
    audit_path: Path
    provider_profiles_path: Path
    manifest_steps: tuple[PlanStep, ...]
    run_steps: tuple[PlanStep, ...]
    requirement_checks: tuple[RequirementCheck, ...]


@dataclass(frozen=True)
class StepExecutionResult:
    """One logged subprocess execution result."""

    exit_code: int
    output_lines: tuple[str, ...]
    parsed_payloads: tuple[dict[str, object], ...]
    stalled: bool = False


@dataclass(frozen=True)
class SessionContext:
    """Persistent wrapper session metadata."""

    session_id: str
    state_path: Path
    resume_command: str


def _requirement_status_label(check: RequirementCheck) -> str:
    if check.satisfied:
        return "OK"
    if check.blocking:
        return "MISSING"
    return "LATER"


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
            "calls. dry-run executes a local vLLM-backed validation pass. "
            "live executes the full budget plan."
        ),
    )
    parser.add_argument(
        "--resume-session-id",
        default="",
        help=(
            "Resume a previously started dry-run/live wrapper session by UUID. "
            "When provided, the wrapper reloads the original arguments and resumes "
            "unfinished steps from the saved session state."
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
            "budget plan in live/validate mode and to vllm_default in dry-run mode."
        ),
    )
    parser.add_argument(
        "--judge-provider",
        default="",
        help=(
            "Optional judge provider id. Defaults to google_gemini_3_1_pro_preview_judge "
            "for live/validate and vllm_default for dry-run."
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
        type=_nonnegative_iteration_cap,
        default=0,
        help=(
            "Critic-feedback budget for each execute run. 0 keeps the wrapper on a "
            "single summarizer pass, 1 means summarizer -> critic -> summarizer, and "
            "larger values add more critic-guided revisions."
        ),
    )
    parser.add_argument(
        "--skip-vllm-server-check",
        action="store_true",
        help="Skip the local /models probe for vllm_default after model pin validation.",
    )
    parser.add_argument(
        "--max-concurrent-providers",
        type=int,
        default=0,
        help=(
            "Maximum number of provider execute runs to launch in parallel. Use 0 to "
            "run all provider steps concurrently by default."
        ),
    )
    parser.add_argument(
        "--connection-retry-backoff-seconds",
        type=float,
        default=15.0,
        help="Initial backoff between retryable provider reconnect attempts.",
    )
    parser.add_argument(
        "--connection-retry-max-backoff-seconds",
        type=float,
        default=120.0,
        help="Maximum reconnect backoff for retryable provider failures.",
    )
    parser.add_argument(
        "--provider-stall-seconds",
        type=float,
        default=900.0,
        help=(
            "Restart one provider child from checkpoint if it stops emitting logs or "
            "heartbeats for this many seconds."
        ),
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


def _nonnegative_iteration_cap(raw_value: str) -> int:
    try:
        parsed = int(str(raw_value).strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "--max-iterations must be a nonnegative integer for the budget wrapper. "
            "Use 0 for summarizer-only or omit the flag at the execute layer for an "
            "unbounded critic-feedback budget."
        )
    return parsed


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
        step_id="manifest:prepare_full_dataset",
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
        kind=_MANIFEST_STEP_KIND,
    )


def _build_curate_longform_command(
    *,
    source_manifest: Path,
    dataset_manifest: Path,
    audit_path: Path,
) -> PlanStep:
    return PlanStep(
        step_id="manifest:curate_budget_manifest",
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
        kind=_MANIFEST_STEP_KIND,
    )


def _build_execute_command(
    *,
    contestant_provider: str,
    judge_provider: str,
    dataset_manifest: Path,
    output_root: Path,
    provider_profiles_path: Path,
    max_samples: int,
    sample_seed: int,
    max_iterations: int,
) -> PlanStep:
    contestant_output_root = (output_root / contestant_provider).resolve()
    return PlanStep(
        step_id=f"run:{contestant_provider}",
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
            "--provider-profiles",
            str(provider_profiles_path),
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
        kind=_RUN_STEP_KIND,
        provider_id=contestant_provider,
    )


def _build_manifest_steps(
    *,
    dataset_manifest: Path,
    source_manifest: Path,
    audit_path: Path,
) -> tuple[PlanStep, ...]:
    dataset_ready, _ = _prepared_budget_manifest_check(dataset_manifest)
    effective_source_manifest = _effective_source_manifest_path(source_manifest)
    source_ready, _ = _prepared_source_manifest_check(effective_source_manifest)
    if dataset_ready:
        return ()

    steps: list[PlanStep] = []
    if not source_ready:
        steps.append(_build_prepare_full_dataset_command(effective_source_manifest))
    steps.append(
        _build_curate_longform_command(
            source_manifest=effective_source_manifest,
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
    normalized_base_url = str(base_url or "").strip()
    if not normalized_base_url:
        return False, "vLLM base URL is missing.", ()
    models_url = f"{normalized_base_url.rstrip('/')}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        request = Request(models_url, headers=headers)
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.load(response)
    except ValueError as exc:
        return False, f"Invalid vLLM base URL '{normalized_base_url}': {exc}", ()
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


def _prepared_manifest_payload(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _coerce_int_or_zero(raw_value: object) -> int:
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return int(raw_value)
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return 0
        try:
            return int(stripped)
        except ValueError:
            return 0
    return 0


def _coerce_float_or_zero(raw_value: object) -> float:
    if isinstance(raw_value, bool):
        return float(raw_value)
    if isinstance(raw_value, int | float):
        return float(raw_value)
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return 0.0
        try:
            return float(stripped)
        except ValueError:
            return 0.0
    return 0.0


def _vllm_budget_qwen_model(model_name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "", str(model_name or "").strip().lower())
    return "qwen35" in normalized and "27b" in normalized


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


def _prepared_source_manifest_check(path: Path) -> tuple[bool, str]:
    base_ok, base_detail = _prepared_manifest_check(path)
    if not base_ok:
        return base_ok, base_detail
    payload = _prepared_manifest_payload(path)
    sample_count = _coerce_int_or_zero(payload.get("sample_count")) if payload else 0
    if sample_count <= 0:
        return False, f"Prepared source manifest has no samples: {path}"
    return base_ok, base_detail


def _prepared_budget_manifest_check(path: Path) -> tuple[bool, str]:
    base_ok, base_detail = _prepared_manifest_check(path)
    if not base_ok:
        return base_ok, base_detail
    payload = _prepared_manifest_payload(path)
    if payload is None:
        return False, f"Prepared budget manifest is not JSON-object shaped: {path}"
    sample_count = _coerce_int_or_zero(payload.get("sample_count"))
    if sample_count <= 0:
        return False, f"Prepared budget manifest has no samples: {path}"
    thresholds = payload.get("curation_thresholds")
    thresholds_map: dict[str, object] = (
        {str(key): value for key, value in thresholds.items() if isinstance(key, str)}
        if isinstance(thresholds, dict)
        else {}
    )
    max_unique_speaker_count = _coerce_int_or_zero(
        thresholds_map.get("max_unique_speaker_count")
    )
    if max_unique_speaker_count != 3:
        return (
            False,
            f"Prepared budget manifest is not the <=3-speaker budget subset: {path}",
        )
    return base_ok, base_detail


def _effective_source_manifest_path(source_manifest: Path) -> Path:
    generated_manifest = (source_manifest.parent / "manifest.json").resolve()
    if source_manifest.name == generated_manifest.name:
        return source_manifest
    source_ready, _ = _prepared_source_manifest_check(source_manifest)
    if source_ready:
        return source_manifest
    return generated_manifest


def _build_requirement_checks(
    *,
    mode: str,
    contestants: tuple[str, ...],
    judge_provider: str,
    profiles: dict[str, ProviderProfile],
    dataset_manifest: Path,
    source_manifest: Path,
    skip_vllm_server_check: bool,
) -> tuple[RequirementCheck, ...]:
    checks: list[RequirementCheck] = []

    effective_source_manifest = _effective_source_manifest_path(source_manifest)
    dataset_ready, dataset_detail = _prepared_budget_manifest_check(dataset_manifest)
    source_ready, source_detail = _prepared_source_manifest_check(
        effective_source_manifest
    )
    manifest_ready_or_buildable = dataset_ready or source_ready or _hf_token_present()
    manifest_detail = (
        dataset_detail
        if dataset_ready
        else (
            f"{dataset_detail} {source_detail} The wrapper can curate the <=3-speaker subset locally."
            if source_ready
            else (
                f"{dataset_manifest} and {effective_source_manifest} are both missing, but an HF token "
                "is present so the script can prepare the full dataset then curate the budget "
                "subset."
                if _hf_token_present()
                else (
                    f"{dataset_manifest} and {effective_source_manifest} are both missing. Set HF_ACCESS_TOKEN "
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
            blocking=True,
        )
    )

    provider_ids = tuple(dict.fromkeys([*contestants, judge_provider]))
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
                blocking=True,
            )
        )
        if profile is None:
            continue

        missing_env = missing_required_env(profile)
        credentials_blocking = mode in {"dry-run", "live"}
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
                blocking=credentials_blocking,
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
                blocking=True,
            )
        )

    for provider_id in provider_ids:
        profile = profiles.get(provider_id)
        if profile is None:
            continue

        provider_target = str(profile.llm_config.get("_target_", "") or "").strip()
        if provider_target != _VLLM_PROVIDER_TARGET:
            continue
        base_url, api_key, model_name = _resolve_vllm_runtime_details(profile)
        model_pinned = bool(model_name)
        qwen_budget_model = _vllm_budget_qwen_model(model_name)
        checks.append(
            RequirementCheck(
                label=f"Pinned vLLM model for {provider_id}",
                satisfied=model_pinned,
                detail=(
                    (
                        f"Local vLLM provider is pinned to served model '{model_name}' and "
                        "will keep thinking enabled in the SPoRC budget path."
                    )
                    if model_pinned and qwen_budget_model
                    else (
                        f"Local vLLM provider is pinned to served model '{model_name}'."
                        if model_pinned
                        else (
                            f"'{provider_id}' has no explicit model pin. Set "
                            "CARD_FRAMEWORK_VLLM_MODEL or VLLM_MODEL before the budget run so the "
                            "local run is reproducible."
                        )
                    )
                ),
                blocking=mode in {"dry-run", "live"},
            )
        )
        if mode == "dry-run" and model_pinned:
            checks.append(
                RequirementCheck(
                    label=f"Qwen/Qwen3.5-27B dry-run model for {provider_id}",
                    satisfied=qwen_budget_model,
                    detail=(
                        f"Dry-run vLLM provider '{provider_id}' is pinned to Qwen/Qwen3.5-27B."
                        if qwen_budget_model
                        else (
                            f"Dry-run vLLM provider '{provider_id}' is pinned to '{model_name}', "
                            "but the documented local dry-run path expects Qwen/Qwen3.5-27B so "
                            "thinking stays enabled."
                        )
                    ),
                    blocking=True,
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
                blocking=mode in {"dry-run", "live"},
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
                    blocking=mode in {"dry-run", "live"},
                )
            )

    return tuple(checks)


def _build_budget_plan(args: argparse.Namespace) -> BudgetPlan:
    dataset_manifest = Path(str(args.dataset_manifest)).resolve()
    source_manifest = Path(str(args.source_manifest)).resolve()
    audit_path = Path(str(args.audit_path)).resolve()
    output_root = Path(str(args.output_root)).resolve()
    provider_profiles_path = Path(str(args.provider_profiles)).resolve()
    contestants = _resolve_contestants(args)
    judge_provider = _resolve_judge_provider(args)
    max_samples = _resolve_max_samples(args)
    profiles = _provider_index(provider_profiles_path)
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
            provider_profiles_path=provider_profiles_path,
            max_samples=max_samples,
            sample_seed=int(args.sample_seed),
            max_iterations=int(args.max_iterations),
        )
        for contestant_provider in contestants
    )
    requirement_checks = _build_requirement_checks(
        mode=str(args.mode),
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
        provider_profiles_path=provider_profiles_path,
        manifest_steps=manifest_steps,
        run_steps=run_steps,
        requirement_checks=requirement_checks,
    )


def _format_command(command: Iterable[str]) -> str:
    return subprocess.list2cmdline(list(command))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_session_id(raw_value: str) -> str:
    normalized = str(raw_value or "").strip()
    if not normalized:
        raise MatrixConfigError("Session UUID cannot be empty.")
    try:
        return str(uuid.UUID(normalized))
    except ValueError as exc:
        raise MatrixConfigError(f"Invalid session UUID '{normalized}': {exc}") from exc


def _session_dir(session_id: str) -> Path:
    return (DEFAULT_SESSION_ROOT / session_id).resolve()


def _session_state_path(session_id: str) -> Path:
    return (_session_dir(session_id) / "session.json").resolve()


def _resume_command(session_id: str) -> str:
    return (
        "uv run python scripts/run_sporc_budget_sweep.py --resume-session-id "
        f"{session_id}"
    )


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temp_path.replace(path)


def _step_checkpoint_path(session_id: str, step: PlanStep) -> str:
    if step.kind != _RUN_STEP_KIND or not step.provider_id:
        return ""
    return str(
        (
            _session_dir(session_id)
            / "checkpoints"
            / f"{step.provider_id}.attempt-0.checkpoint.json"
        ).resolve()
    )


def _fresh_retry_checkpoint_path(
    *,
    session_id: str,
    step: PlanStep,
    attempt_count: int,
) -> str:
    if step.kind != _RUN_STEP_KIND or not step.provider_id:
        return ""
    return str(
        (
            _session_dir(session_id)
            / "checkpoints"
            / f"{step.provider_id}.attempt-{attempt_count}.checkpoint.json"
        ).resolve()
    )


def _build_session_payload(
    *,
    session_id: str,
    requested_argv: list[str],
    plan: BudgetPlan,
) -> dict[str, object]:
    steps_payload: dict[str, object] = {}
    for step in (*plan.manifest_steps, *plan.run_steps):
        steps_payload[step.step_id] = {
            "step_id": step.step_id,
            "label": step.label,
            "kind": step.kind,
            "provider_id": step.provider_id,
            "checkpoint_path": _step_checkpoint_path(session_id, step),
            "status": "pending",
            "attempt_count": 0,
            "last_exit_code": None,
            "last_error": "",
            "last_started_at_utc": "",
            "last_completed_at_utc": "",
            "last_command": "",
            "report_path": "",
            "run_id": "",
            "resume_from_checkpoint": False,
        }
    now_iso = _utc_now_iso()
    return {
        "session_state_version": _SESSION_STATE_VERSION,
        "session_id": session_id,
        "created_at_utc": now_iso,
        "updated_at_utc": now_iso,
        "status": "pending",
        "mode": plan.mode,
        "requested_argv": list(requested_argv),
        "resume_command": _resume_command(session_id),
        "provider_profiles_path": str(plan.provider_profiles_path),
        "dataset_manifest": str(plan.dataset_manifest),
        "source_manifest": str(plan.source_manifest),
        "audit_path": str(plan.audit_path),
        "steps": steps_payload,
    }


def _load_session_payload(session_id: str) -> dict[str, object]:
    state_path = _session_state_path(session_id)
    if not state_path.exists():
        raise MatrixConfigError(
            f"No SPoRC budget wrapper session found for UUID {session_id}."
        )
    raw_payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise MatrixConfigError(
            f"Invalid SPoRC budget wrapper session payload: {state_path}"
        )
    payload = {str(key): value for key, value in raw_payload.items()}
    if int(payload.get("session_state_version", 0) or 0) != _SESSION_STATE_VERSION:
        raise MatrixConfigError(
            f"Unsupported SPoRC budget wrapper session version in {state_path}."
        )
    return payload


def _resolve_session_context(
    *,
    resume_session_id: str,
    plan: BudgetPlan,
    requested_argv: list[str],
) -> tuple[SessionContext | None, dict[str, object] | None, list[str]]:
    if str(plan.mode) == "validate":
        return None, None, requested_argv

    normalized_resume_session_id = str(resume_session_id or "").strip()
    if normalized_resume_session_id:
        normalized_session_id = _validate_session_id(normalized_resume_session_id)
        payload = _load_session_payload(normalized_session_id)
        stored_argv = payload.get("requested_argv", [])
        if not isinstance(stored_argv, list) or not all(
            isinstance(item, str) for item in stored_argv
        ):
            raise MatrixConfigError(
                f"Session {normalized_session_id} is missing its original wrapper arguments."
            )
        return (
            SessionContext(
                session_id=normalized_session_id,
                state_path=_session_state_path(normalized_session_id),
                resume_command=_resume_command(normalized_session_id),
            ),
            payload,
            [str(item) for item in stored_argv],
        )

    new_session_id = str(uuid.uuid4())
    return (
        SessionContext(
            session_id=new_session_id,
            state_path=_session_state_path(new_session_id),
            resume_command=_resume_command(new_session_id),
        ),
        None,
        requested_argv,
    )


def _ensure_session_payload(
    *,
    session_context: SessionContext,
    existing_payload: dict[str, object] | None,
    plan: BudgetPlan,
    requested_argv: list[str],
) -> dict[str, object]:
    if existing_payload is not None:
        return existing_payload
    payload = _build_session_payload(
        session_id=session_context.session_id,
        requested_argv=requested_argv,
        plan=plan,
    )
    _atomic_write_json(session_context.state_path, payload)
    return payload


def _write_session_payload(
    *,
    session_context: SessionContext,
    payload: dict[str, object],
) -> None:
    payload["updated_at_utc"] = _utc_now_iso()
    _atomic_write_json(session_context.state_path, payload)


def _update_session_status(
    *,
    session_context: SessionContext,
    payload: dict[str, object],
    status: str,
) -> None:
    payload["status"] = status
    _write_session_payload(session_context=session_context, payload=payload)


def _step_state(payload: dict[str, object], step: PlanStep) -> dict[str, object]:
    raw_steps = payload.get("steps", {})
    if not isinstance(raw_steps, dict):
        raise MatrixConfigError(f"Session state is missing step '{step.step_id}'.")
    step_mapping = {str(key): value for key, value in raw_steps.items()}
    if step.step_id not in step_mapping:
        raise MatrixConfigError(f"Session state is missing step '{step.step_id}'.")
    raw_step = step_mapping[step.step_id]
    if not isinstance(raw_step, dict):
        raise MatrixConfigError(f"Session step '{step.step_id}' is malformed.")
    return {str(key): value for key, value in raw_step.items()}


def _step_completed(payload: dict[str, object], step: PlanStep) -> bool:
    return str(_step_state(payload, step).get("status", "") or "") == "completed"


def _materialize_step_command(
    *,
    step: PlanStep,
    step_state: dict[str, object],
) -> tuple[str, ...]:
    command = list(step.command)
    checkpoint_path = str(step_state.get("checkpoint_path", "") or "").strip()
    if step.kind == _RUN_STEP_KIND:
        if checkpoint_path:
            command.extend(["--checkpoint-path", checkpoint_path])
            if Path(checkpoint_path).exists():
                command.append("--resume-from-checkpoint")
        command.extend(["--emit-progress-events", "--emit-progress-heartbeats"])
    return tuple(command)


def _extract_report_payload(
    parsed_payloads: tuple[dict[str, object], ...],
) -> dict[str, object] | None:
    for payload in reversed(parsed_payloads):
        if "report_path" in payload and "status" in payload:
            return payload
    return None


def _load_json_object(path: Path) -> dict[str, object] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return {str(key): value for key, value in payload.items()}


def _step_output_root(step: PlanStep) -> Path | None:
    command = list(step.command)
    try:
        output_root_index = command.index("--output-root")
    except ValueError:
        return None
    if output_root_index + 1 >= len(command):
        return None
    return Path(command[output_root_index + 1]).expanduser().resolve()


def _report_sort_timestamp(path: Path, payload: dict[str, object]) -> float:
    raw_generated_at = str(payload.get("generated_at_utc", "") or "").strip()
    if raw_generated_at:
        try:
            return datetime.fromisoformat(raw_generated_at).timestamp()
        except ValueError:
            pass
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _discover_resume_report_for_step(
    *,
    step: PlanStep,
    step_state: dict[str, object],
) -> tuple[Path | None, dict[str, object] | None]:
    if step.kind != _RUN_STEP_KIND:
        return None, None
    output_root = _step_output_root(step)
    if output_root is None or not output_root.exists():
        return None, None
    checkpoint_path = str(step_state.get("checkpoint_path", "") or "").strip()
    run_id = str(step_state.get("run_id", "") or "").strip()
    candidates: list[tuple[int, float, Path, dict[str, object]]] = []
    for report_path in output_root.rglob("report.json"):
        payload = _load_json_object(report_path)
        if payload is None:
            continue
        score = 0
        if (
            checkpoint_path
            and str(payload.get("checkpoint_path", "") or "").strip() == checkpoint_path
        ):
            score += 2
        if run_id and str(payload.get("run_id", "") or "").strip() == run_id:
            score += 1
        candidates.append(
            (
                score,
                _report_sort_timestamp(report_path, payload),
                report_path.resolve(),
                payload,
            )
        )
    if not candidates:
        return None, None
    if checkpoint_path:
        checkpoint_matches = [
            candidate for candidate in candidates if candidate[0] >= 2
        ]
        if checkpoint_matches:
            candidates = checkpoint_matches
    elif run_id:
        run_matches = [candidate for candidate in candidates if candidate[0] >= 1]
        if run_matches:
            candidates = run_matches
    _score, _timestamp, selected_path, selected_payload = max(
        candidates,
        key=lambda item: (item[0], item[1], str(item[2])),
    )
    return selected_path, selected_payload


def _load_retryable_report_failure(report_path: str) -> tuple[bool, str]:
    normalized_path = str(report_path or "").strip()
    if not normalized_path:
        return False, ""
    path = Path(normalized_path)
    if not path.exists():
        return False, ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, ""
    if not isinstance(payload, dict):
        return False, ""
    samples = payload.get("samples", [])
    final_statuses = {
        str(sample.get("final_status", "") or "").strip()
        for sample in samples
        if isinstance(sample, dict)
    }
    non_retryable_statuses = sorted(
        status
        for status in final_statuses
        if status and status not in _RETRYABLE_STEP_FINAL_STATUSES
    )
    if non_retryable_statuses:
        return False, ""
    retryable_statuses = sorted(final_statuses & _RETRYABLE_STEP_FINAL_STATUSES)
    if retryable_statuses:
        return True, ", ".join(retryable_statuses)
    return False, ""


def _repair_incomplete_run_step_state(
    *,
    step: PlanStep,
    session_context: SessionContext,
    session_payload: dict[str, object],
    state_lock: threading.Lock,
) -> tuple[dict[str, object], str]:
    step_state = _step_state(session_payload, step)
    if step.kind != _RUN_STEP_KIND:
        return step_state, ""
    if str(step_state.get("status", "") or "") == "completed":
        return step_state, ""

    report_path = str(step_state.get("report_path", "") or "").strip()
    run_id = str(step_state.get("run_id", "") or "").strip()
    discovered_report_path: Path | None = None
    discovered_payload: dict[str, object] | None = None
    if not report_path or not Path(report_path).exists() or not run_id:
        discovered_report_path, discovered_payload = _discover_resume_report_for_step(
            step=step,
            step_state=step_state,
        )
    updates: dict[str, object] = {}
    if discovered_report_path is not None and discovered_payload is not None:
        hydrated_report_path = str(discovered_report_path)
        hydrated_run_id = str(discovered_payload.get("run_id", "") or "").strip()
        if hydrated_report_path and hydrated_report_path != report_path:
            updates["report_path"] = hydrated_report_path
        if hydrated_run_id and hydrated_run_id != run_id:
            updates["run_id"] = hydrated_run_id
        report_path = hydrated_report_path or report_path
        run_id = hydrated_run_id or run_id

    retryable_report, retryable_reason = _load_retryable_report_failure(report_path)
    note = ""
    if retryable_report:
        current_checkpoint = str(step_state.get("checkpoint_path", "") or "").strip()
        next_attempt = max(_coerce_int_or_zero(step_state.get("attempt_count")), 1)
        replacement_checkpoint = _fresh_retry_checkpoint_path(
            session_id=session_context.session_id,
            step=step,
            attempt_count=next_attempt,
        )
        if replacement_checkpoint and replacement_checkpoint != current_checkpoint:
            updates["checkpoint_path"] = replacement_checkpoint
        updates["report_path"] = ""
        updates["run_id"] = ""
        updates["resume_from_checkpoint"] = False
        updates["last_error"] = (
            "discarded checkpoint before resume after retryable report statuses: "
            f"{retryable_reason}"
        )
        note = (
            "discarded the previous checkpoint before resume because the latest report "
            f"recorded retryable statuses: {retryable_reason}"
        )
    elif discovered_report_path is not None:
        note = (
            "hydrated resume metadata from the latest provider report so resume can "
            "continue from the cancelled checkpoint"
        )

    if updates:
        _update_step_state(
            session_context=session_context,
            payload=session_payload,
            state_lock=state_lock,
            step=step,
            **updates,
        )
        step_state = _step_state(session_payload, step)
    return step_state, note


def _estimate_visible_token_count(payload: object) -> float:
    if payload in (None, "", [], {}):
        return 0.0
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return round(len(serialized) / 4.0, 1)


def _resolve_budget_pricing(
    provider_id: str,
    profiles: dict[str, ProviderProfile],
) -> dict[str, object] | None:
    normalized_provider_id = str(provider_id or "").strip()
    if not normalized_provider_id:
        return None
    profile = profiles.get(normalized_provider_id)
    if profile is None:
        if normalized_provider_id == "vllm_default":
            return {
                "input_usd_per_million": 0.0,
                "output_usd_per_million": 0.0,
                "pricing_reference": "local_vllm_zero_api_spend",
            }
        return None
    resolved = resolve_provider_config(profile)
    provider_target = str(resolved.get("_target_", "") or "").strip()
    model_name = str(resolved.get("model", "") or "").strip().lower()
    if provider_target == _VLLM_PROVIDER_TARGET:
        return {
            "input_usd_per_million": 0.0,
            "output_usd_per_million": 0.0,
            "pricing_reference": "local_vllm_zero_api_spend",
        }
    if provider_target == _OPENAI_PROVIDER_TARGET and model_name.startswith(
        "gpt-5.4-mini"
    ):
        return {
            "input_usd_per_million": 0.75,
            "output_usd_per_million": 4.5,
            "pricing_reference": _BUDGET_RATE_REFERENCE,
        }
    if provider_target == _ANTHROPIC_PROVIDER_TARGET and model_name.startswith(
        "claude-haiku-4-5"
    ):
        return {
            "input_usd_per_million": 1.0,
            "output_usd_per_million": 5.0,
            "pricing_reference": _BUDGET_RATE_REFERENCE,
        }
    if provider_target == _GOOGLE_PROVIDER_TARGET and model_name.startswith(
        "gemini-3.1-pro-preview"
    ):
        return {
            "input_usd_per_million": 2.0,
            "output_usd_per_million": 12.0,
            "pricing_reference": _BUDGET_RATE_REFERENCE,
        }
    if provider_target == _DEEPSEEK_PROVIDER_TARGET and model_name.startswith(
        "deepseek-chat"
    ):
        return {
            "input_usd_per_million": 0.28,
            "output_usd_per_million": 0.42,
            "pricing_reference": _BUDGET_RATE_REFERENCE,
        }
    return None


def _step_run_root(
    *,
    step: PlanStep,
    step_state: dict[str, object],
) -> Path | None:
    report_path = str(step_state.get("report_path", "") or "").strip()
    if report_path:
        resolved_report_path = Path(report_path).expanduser().resolve()
        if resolved_report_path.exists():
            return resolved_report_path.parent
    run_id = str(step_state.get("run_id", "") or "").strip()
    output_root = _step_output_root(step)
    if output_root is not None and run_id:
        candidate = (output_root / run_id).resolve()
        if candidate.exists():
            return candidate
    discovered_report_path, _ = _discover_resume_report_for_step(
        step=step,
        step_state=step_state,
    )
    if discovered_report_path is not None:
        return discovered_report_path.parent
    return None


def _call_request_token_estimate(call_payload: dict[str, object]) -> float:
    operation = str(call_payload.get("operation", "") or "").strip()
    if operation == "generate":
        request_payload = {
            "system_prompt": call_payload.get("system_prompt"),
            "user_prompt": call_payload.get("user_prompt"),
            "max_tokens": call_payload.get("max_tokens"),
        }
    else:
        request_payload = {
            "messages": call_payload.get("messages"),
            "tools": call_payload.get("tools"),
            "tool_choice": call_payload.get("tool_choice"),
            "max_tokens": call_payload.get("max_tokens"),
        }
    return _estimate_visible_token_count(request_payload)


def _call_response_token_estimate(call_payload: dict[str, object]) -> float:
    if "response_text" in call_payload:
        return _estimate_visible_token_count(call_payload.get("response_text"))
    if "response" in call_payload:
        return _estimate_visible_token_count(call_payload.get("response"))
    return 0.0


def _is_nonbillable_trace_error(call_payload: dict[str, object]) -> bool:
    error_message = str(call_payload.get("error", "") or "").strip().lower()
    if not error_message:
        return False
    if "rate limit" in error_message or "error code: 429" in error_message:
        return True
    if "connection refused" in error_message or "connecttimeout" in error_message:
        return True
    return False


def _merge_budget_bucket(
    *,
    aggregate: dict[str, Any],
    provider_id: str,
    role: str,
    input_tokens: float,
    output_tokens: float,
    billable: bool,
    pricing: dict[str, object] | None,
) -> None:
    bucket_value = aggregate.setdefault(
        provider_id,
        {
            "provider_id": provider_id,
            "call_count": 0,
            "roles": {},
            "attempted_input_tokens": 0.0,
            "attempted_output_tokens": 0.0,
            "billable_input_tokens": 0.0,
            "billable_output_tokens": 0.0,
            "estimated_cost_usd": 0.0,
            "pricing": pricing,
        },
    )
    bucket = bucket_value if isinstance(bucket_value, dict) else aggregate[provider_id]
    bucket["call_count"] = int(bucket.get("call_count", 0) or 0) + 1
    roles = bucket.get("roles")
    if not isinstance(roles, dict):
        roles = {}
        bucket["roles"] = roles
    roles[role] = int(roles.get(role, 0) or 0) + 1
    bucket["attempted_input_tokens"] = round(
        float(bucket.get("attempted_input_tokens", 0.0) or 0.0) + input_tokens,
        1,
    )
    bucket["attempted_output_tokens"] = round(
        float(bucket.get("attempted_output_tokens", 0.0) or 0.0) + output_tokens,
        1,
    )
    if not billable:
        return
    bucket["billable_input_tokens"] = round(
        float(bucket.get("billable_input_tokens", 0.0) or 0.0) + input_tokens,
        1,
    )
    bucket["billable_output_tokens"] = round(
        float(bucket.get("billable_output_tokens", 0.0) or 0.0) + output_tokens,
        1,
    )
    if pricing is None:
        return
    bucket["estimated_cost_usd"] = round(
        _coerce_float_or_zero(bucket.get("estimated_cost_usd", 0.0))
        + (input_tokens / 1_000_000.0)
        * _coerce_float_or_zero(pricing.get("input_usd_per_million", 0.0))
        + (output_tokens / 1_000_000.0)
        * _coerce_float_or_zero(pricing.get("output_usd_per_million", 0.0)),
        6,
    )


def _collect_session_budget_profile(
    *,
    plan: BudgetPlan,
    session_payload: dict[str, object],
    profiles: dict[str, ProviderProfile],
) -> dict[str, object]:
    aggregate: dict[str, Any] = {}
    for step in plan.run_steps:
        step_state = _step_state(session_payload, step)
        run_root = _step_run_root(step=step, step_state=step_state)
        if run_root is None or not run_root.exists():
            continue
        for call_path in run_root.rglob("call_*.json"):
            call_payload = _load_json_object(call_path)
            if call_payload is None:
                continue
            role = call_path.parent.name
            provider_id = plan.judge_provider if role == "judge" else step.provider_id
            input_tokens = _call_request_token_estimate(call_payload)
            output_tokens = _call_response_token_estimate(call_payload)
            _merge_budget_bucket(
                aggregate=aggregate,
                provider_id=provider_id,
                role=role,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                billable=not _is_nonbillable_trace_error(call_payload),
                pricing=_resolve_budget_pricing(provider_id, profiles),
            )
    providers = {
        provider_id: aggregate[provider_id]
        for provider_id in sorted(aggregate)
        if isinstance(aggregate[provider_id], dict)
    }
    total_estimated_cost_usd = round(
        sum(
            float(provider.get("estimated_cost_usd", 0.0) or 0.0)
            for provider in providers.values()
        ),
        6,
    )
    total_attempted_input_tokens = round(
        sum(
            float(provider.get("attempted_input_tokens", 0.0) or 0.0)
            for provider in providers.values()
        ),
        1,
    )
    total_attempted_output_tokens = round(
        sum(
            float(provider.get("attempted_output_tokens", 0.0) or 0.0)
            for provider in providers.values()
        ),
        1,
    )
    return {
        "generated_at_utc": _utc_now_iso(),
        "pricing_reference": _BUDGET_RATE_REFERENCE,
        "token_estimation_method": "visible_json_chars_div_4_lower_bound",
        "providers": providers,
        "total_attempted_input_tokens": total_attempted_input_tokens,
        "total_attempted_output_tokens": total_attempted_output_tokens,
        "total_estimated_cost_usd": total_estimated_cost_usd,
    }


def _refresh_session_budget_profile(
    *,
    plan: BudgetPlan,
    session_context: SessionContext,
    session_payload: dict[str, object],
    state_lock: threading.Lock,
    profiles: dict[str, ProviderProfile],
) -> dict[str, object]:
    budget_profile = _collect_session_budget_profile(
        plan=plan,
        session_payload=session_payload,
        profiles=profiles,
    )
    with state_lock:
        session_payload["budget_profile"] = budget_profile
        _write_session_payload(session_context=session_context, payload=session_payload)
    _atomic_write_json(
        _session_dir(session_context.session_id) / "budget_profile.json", budget_profile
    )
    return budget_profile


def _print_budget_profile_summary(budget_profile: dict[str, object]) -> None:
    providers = budget_profile.get("providers", {})
    if not isinstance(providers, dict) or not providers:
        return
    provider_mapping: dict[str, Any] = {
        str(key): value for key, value in providers.items()
    }
    total_cost = _coerce_float_or_zero(
        budget_profile.get("total_estimated_cost_usd", 0.0)
    )
    parts = [f"total_est=${total_cost:.4f}"]
    for provider_id, raw_bucket in sorted(provider_mapping.items()):
        if not isinstance(raw_bucket, dict):
            continue
        parts.append(
            f"{provider_id}:${_coerce_float_or_zero(raw_bucket.get('estimated_cost_usd', 0.0)):.4f}"
        )
    _print_prefixed_log("budget", " | ".join(parts))


def _retry_strategy_for_step_result(
    result: StepExecutionResult,
) -> tuple[str | None, str]:
    if result.stalled:
        return "resume", "wrapper detected a stalled child process"
    report_payload = _extract_report_payload(result.parsed_payloads)
    if report_payload is not None:
        retryable_report, reason = _load_retryable_report_failure(
            str(report_payload.get("report_path", "") or "")
        )
        if retryable_report:
            return "restart", f"report recorded retryable infra statuses: {reason}"
    lowered_output = "\n".join(result.output_lines).lower()
    if any(token in lowered_output for token in _RETRYABLE_CONNECTION_TOKENS):
        return "resume", "child output reported a connection/startup failure"
    return None, ""


def _print_prefixed_log(prefix: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}][{prefix}] {message}", flush=True)


def _register_active_process(
    *,
    active_processes: dict[str, subprocess.Popen[str]],
    process_lock: threading.Lock,
    step_id: str,
    process: subprocess.Popen[str],
) -> None:
    with process_lock:
        active_processes[step_id] = process


def _unregister_active_process(
    *,
    active_processes: dict[str, subprocess.Popen[str]],
    process_lock: threading.Lock,
    step_id: str,
) -> None:
    with process_lock:
        active_processes.pop(step_id, None)


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def _terminate_active_processes(
    *,
    active_processes: dict[str, subprocess.Popen[str]],
    process_lock: threading.Lock,
) -> None:
    with process_lock:
        processes = list(active_processes.values())
    for process in processes:
        _terminate_process(process)


def _run_logged_subprocess(
    *,
    step: PlanStep,
    command: tuple[str, ...],
    stall_seconds: float,
    stop_event: threading.Event,
    active_processes: dict[str, subprocess.Popen[str]],
    process_lock: threading.Lock,
) -> StepExecutionResult:
    _print_prefixed_log(step.step_id, f"starting: {_format_command(command)}")
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    _register_active_process(
        active_processes=active_processes,
        process_lock=process_lock,
        step_id=step.step_id,
        process=process,
    )

    output_lines: list[str] = []
    parsed_payloads: list[dict[str, object]] = []
    output_lock = threading.Lock()
    last_output_time = time.monotonic()

    def _reader() -> None:
        nonlocal last_output_time
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.rstrip("\r\n")
            with output_lock:
                output_lines.append(line)
                last_output_time = time.monotonic()
            _print_prefixed_log(step.step_id, line)
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                with output_lock:
                    parsed_payloads.append(parsed)
        process.stdout.close()

    reader_thread = threading.Thread(
        target=_reader, name=f"budget-step-{step.step_id}", daemon=True
    )
    reader_thread.start()
    stalled = False
    try:
        while process.poll() is None:
            if stop_event.is_set():
                _terminate_process(process)
                break
            with output_lock:
                idle_seconds = time.monotonic() - last_output_time
            if stall_seconds > 0 and idle_seconds > stall_seconds:
                stalled = True
                _print_prefixed_log(
                    step.step_id,
                    f"no child output for {idle_seconds:.0f}s; terminating so the wrapper can resume from checkpoint",
                )
                _terminate_process(process)
                break
            time.sleep(1.0)
        exit_code = int(process.wait())
    finally:
        reader_thread.join(timeout=5.0)
        _unregister_active_process(
            active_processes=active_processes,
            process_lock=process_lock,
            step_id=step.step_id,
        )
    return StepExecutionResult(
        exit_code=exit_code,
        output_lines=tuple(output_lines),
        parsed_payloads=tuple(parsed_payloads),
        stalled=stalled,
    )


def _sleep_with_stop(stop_event: threading.Event, seconds: float) -> bool:
    deadline = time.monotonic() + max(0.0, seconds)
    while time.monotonic() < deadline:
        if stop_event.is_set():
            return False
        time.sleep(min(1.0, deadline - time.monotonic()))
    return not stop_event.is_set()


def _wait_for_vllm_reconnect(
    *,
    provider_id: str,
    profile: ProviderProfile,
    stop_event: threading.Event,
    backoff_seconds: float,
    max_backoff_seconds: float,
) -> bool:
    wait_seconds = max(1.0, backoff_seconds)
    while not stop_event.is_set():
        base_url, api_key, model_name = _resolve_vllm_runtime_details(profile)
        reachable, detail, model_ids = _probe_openai_compatible_models(
            base_url=base_url,
            api_key=api_key,
        )
        if reachable and (not model_name or model_name in model_ids):
            _print_prefixed_log(provider_id, f"connection probe recovered: {detail}")
            return True
        _print_prefixed_log(
            provider_id,
            f"still waiting for provider connection: {detail} retry_in={wait_seconds:.0f}s",
        )
        if not _sleep_with_stop(stop_event, wait_seconds):
            return False
        wait_seconds = min(
            max(wait_seconds * 1.5, wait_seconds + 1.0), max_backoff_seconds
        )
    return False


def _wait_for_retry_window(
    *,
    step: PlanStep,
    plan: BudgetPlan,
    profiles: dict[str, ProviderProfile],
    stop_event: threading.Event,
    backoff_seconds: float,
    max_backoff_seconds: float,
) -> bool:
    if step.kind != _RUN_STEP_KIND:
        return _sleep_with_stop(stop_event, backoff_seconds)
    provider_ids = [step.provider_id]
    if plan.judge_provider and plan.judge_provider not in provider_ids:
        provider_ids.append(plan.judge_provider)
    waited_for_vllm = False
    for provider_id in provider_ids:
        profile = profiles.get(provider_id)
        if profile is None:
            continue
        provider_target = str(profile.llm_config.get("_target_", "") or "").strip()
        if provider_target != _VLLM_PROVIDER_TARGET:
            continue
        waited_for_vllm = True
        if not _wait_for_vllm_reconnect(
            provider_id=provider_id,
            profile=profile,
            stop_event=stop_event,
            backoff_seconds=backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
        ):
            return False
    if waited_for_vllm:
        return True
    _print_prefixed_log(
        step.step_id,
        f"waiting {backoff_seconds:.0f}s before retrying the provider connection",
    )
    return _sleep_with_stop(stop_event, backoff_seconds)


def _update_step_state(
    *,
    session_context: SessionContext,
    payload: dict[str, object],
    state_lock: threading.Lock,
    step: PlanStep,
    **updates: object,
) -> None:
    with state_lock:
        raw_steps = payload.get("steps", {})
        if not isinstance(raw_steps, dict):
            raise MatrixConfigError(f"Session state is missing step '{step.step_id}'.")
        step_mapping = {str(key): value for key, value in raw_steps.items()}
        raw_step = step_mapping.get(step.step_id)
        if not isinstance(raw_step, dict):
            raise MatrixConfigError(f"Session step '{step.step_id}' is malformed.")
        step_payload = {str(key): value for key, value in raw_step.items()}
        for key, value in updates.items():
            step_payload[str(key)] = value
        step_mapping[step.step_id] = step_payload
        payload["steps"] = step_mapping
        _write_session_payload(session_context=session_context, payload=payload)


def _run_step_with_retries(
    *,
    step: PlanStep,
    plan: BudgetPlan,
    session_context: SessionContext,
    session_payload: dict[str, object],
    state_lock: threading.Lock,
    stop_event: threading.Event,
    active_processes: dict[str, subprocess.Popen[str]],
    process_lock: threading.Lock,
    profiles: dict[str, ProviderProfile],
    args: argparse.Namespace,
) -> int:
    retry_backoff = max(1.0, float(args.connection_retry_backoff_seconds))
    retry_max_backoff = max(
        retry_backoff, float(args.connection_retry_max_backoff_seconds)
    )
    stall_seconds = max(0.0, float(args.provider_stall_seconds))
    _, repair_note = _repair_incomplete_run_step_state(
        step=step,
        session_context=session_context,
        session_payload=session_payload,
        state_lock=state_lock,
    )
    if repair_note:
        _print_prefixed_log(step.step_id, repair_note)
    while not stop_event.is_set():
        step_payload = _step_state(session_payload, step)
        if str(step_payload.get("status", "") or "") == "completed":
            return 0
        command = _materialize_step_command(step=step, step_state=step_payload)
        attempt_count = _coerce_int_or_zero(step_payload.get("attempt_count")) + 1
        resume_from_checkpoint = (
            step.kind == _RUN_STEP_KIND
            and Path(str(step_payload.get("checkpoint_path", "") or "")).exists()
        )
        _update_step_state(
            session_context=session_context,
            payload=session_payload,
            state_lock=state_lock,
            step=step,
            status="running",
            attempt_count=attempt_count,
            last_started_at_utc=_utc_now_iso(),
            last_command=_format_command(command),
            resume_from_checkpoint=resume_from_checkpoint,
            last_error="",
        )
        result = _run_logged_subprocess(
            step=step,
            command=command,
            stall_seconds=stall_seconds,
            stop_event=stop_event,
            active_processes=active_processes,
            process_lock=process_lock,
        )
        report_payload = _extract_report_payload(result.parsed_payloads)
        step_updates: dict[str, object] = {
            "last_exit_code": result.exit_code,
            "report_path": (
                str(report_payload.get("report_path", "") or "")
                if report_payload is not None
                else str(step_payload.get("report_path", "") or "")
            ),
            "run_id": (
                str(report_payload.get("run_id", "") or "")
                if report_payload is not None
                else str(step_payload.get("run_id", "") or "")
            ),
        }
        retry_strategy, retry_reason = _retry_strategy_for_step_result(result)
        if result.exit_code == 0 and retry_strategy is None:
            _update_step_state(
                session_context=session_context,
                payload=session_payload,
                state_lock=state_lock,
                step=step,
                status="completed",
                last_completed_at_utc=_utc_now_iso(),
                **step_updates,
            )
            _print_budget_profile_summary(
                _refresh_session_budget_profile(
                    plan=plan,
                    session_context=session_context,
                    session_payload=session_payload,
                    state_lock=state_lock,
                    profiles=profiles,
                )
            )
            return 0
        if stop_event.is_set():
            _update_step_state(
                session_context=session_context,
                payload=session_payload,
                state_lock=state_lock,
                step=step,
                status="interrupted",
                last_error="wrapper stop requested",
                **step_updates,
            )
            _print_budget_profile_summary(
                _refresh_session_budget_profile(
                    plan=plan,
                    session_context=session_context,
                    session_payload=session_payload,
                    state_lock=state_lock,
                    profiles=profiles,
                )
            )
            return 130
        if retry_strategy is None:
            _update_step_state(
                session_context=session_context,
                payload=session_payload,
                state_lock=state_lock,
                step=step,
                status="failed",
                last_completed_at_utc=_utc_now_iso(),
                last_error=(
                    retry_reason or f"step exited with code {result.exit_code}"
                ),
                **step_updates,
            )
            _print_budget_profile_summary(
                _refresh_session_budget_profile(
                    plan=plan,
                    session_context=session_context,
                    session_payload=session_payload,
                    state_lock=state_lock,
                    profiles=profiles,
                )
            )
            return result.exit_code
        if retry_strategy == "restart":
            step_updates["checkpoint_path"] = _fresh_retry_checkpoint_path(
                session_id=session_context.session_id,
                step=step,
                attempt_count=attempt_count,
            )
            step_updates["report_path"] = ""
            step_updates["run_id"] = ""
        _print_prefixed_log(
            step.step_id,
            f"retryable provider issue detected on attempt {attempt_count}: {retry_reason} "
            f"retry_strategy={retry_strategy}",
        )
        _update_step_state(
            session_context=session_context,
            payload=session_payload,
            state_lock=state_lock,
            step=step,
            status="retry_waiting",
            last_error=retry_reason,
            **step_updates,
        )
        _print_budget_profile_summary(
            _refresh_session_budget_profile(
                plan=plan,
                session_context=session_context,
                session_payload=session_payload,
                state_lock=state_lock,
                profiles=profiles,
            )
        )
        if not _wait_for_retry_window(
            step=step,
            plan=plan,
            profiles=profiles,
            stop_event=stop_event,
            backoff_seconds=retry_backoff,
            max_backoff_seconds=retry_max_backoff,
        ):
            _update_step_state(
                session_context=session_context,
                payload=session_payload,
                state_lock=state_lock,
                step=step,
                status="interrupted",
                last_error="wrapper stop requested during retry wait",
                **step_updates,
            )
            _print_budget_profile_summary(
                _refresh_session_budget_profile(
                    plan=plan,
                    session_context=session_context,
                    session_payload=session_payload,
                    state_lock=state_lock,
                    profiles=profiles,
                )
            )
            return 130
        retry_backoff = min(
            max(retry_backoff * 1.5, retry_backoff + 1.0), retry_max_backoff
        )
    return 130


def _resolve_parallel_worker_count(args: argparse.Namespace, step_count: int) -> int:
    configured_raw = getattr(args, "max_concurrent_providers", 0)
    configured = int(configured_raw) if configured_raw else 0
    if configured <= 0:
        return max(step_count, 1)
    return min(configured, max(step_count, 1))


def _print_resume_banner(session_context: SessionContext) -> None:
    print(f"Session UUID: {session_context.session_id}")
    print(f"Resume command: {session_context.resume_command}")


def _run_pending_steps(
    *,
    steps: tuple[PlanStep, ...],
    plan: BudgetPlan,
    session_context: SessionContext,
    session_payload: dict[str, object],
    state_lock: threading.Lock,
    stop_event: threading.Event,
    active_processes: dict[str, subprocess.Popen[str]],
    process_lock: threading.Lock,
    profiles: dict[str, ProviderProfile],
    args: argparse.Namespace,
    parallel: bool,
) -> int:
    pending_steps = [
        step for step in steps if not _step_completed(session_payload, step)
    ]
    if not pending_steps:
        return 0
    if not parallel or len(pending_steps) == 1:
        for step in pending_steps:
            exit_code = _run_step_with_retries(
                step=step,
                plan=plan,
                session_context=session_context,
                session_payload=session_payload,
                state_lock=state_lock,
                stop_event=stop_event,
                active_processes=active_processes,
                process_lock=process_lock,
                profiles=profiles,
                args=args,
            )
            if exit_code != 0:
                return exit_code
        return 0

    max_workers = _resolve_parallel_worker_count(args, len(pending_steps))
    _print_prefixed_log(
        session_context.session_id,
        f"launching {len(pending_steps)} provider steps with max_concurrent_providers={max_workers}",
    )
    failure_exit_code = 0
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix="sporc-budget"
    ) as executor:
        future_to_step: dict[Future[int], PlanStep] = {}
        pending_iter = iter(pending_steps)

        def _submit_next() -> None:
            try:
                next_step = next(pending_iter)
            except StopIteration:
                return
            future = executor.submit(
                _run_step_with_retries,
                step=next_step,
                plan=plan,
                session_context=session_context,
                session_payload=session_payload,
                state_lock=state_lock,
                stop_event=stop_event,
                active_processes=active_processes,
                process_lock=process_lock,
                profiles=profiles,
                args=args,
            )
            future_to_step[future] = next_step

        for _ in range(min(max_workers, len(pending_steps))):
            _submit_next()

        while future_to_step:
            done, _ = wait(tuple(future_to_step), return_when=FIRST_COMPLETED)
            for future in done:
                step = future_to_step.pop(future)
                try:
                    exit_code = int(future.result())
                except Exception as exc:
                    exit_code = 1
                    _update_step_state(
                        session_context=session_context,
                        payload=session_payload,
                        state_lock=state_lock,
                        step=step,
                        status="failed",
                        last_completed_at_utc=_utc_now_iso(),
                        last_error=str(exc),
                    )
                if exit_code != 0:
                    failure_exit_code = exit_code
                    stop_event.set()
                    _terminate_active_processes(
                        active_processes=active_processes,
                        process_lock=process_lock,
                    )
                    break
                if not stop_event.is_set():
                    _submit_next()
            if failure_exit_code != 0:
                break
    return failure_exit_code or (130 if stop_event.is_set() else 0)


def _install_stop_signal_handlers(
    *,
    stop_event: threading.Event,
    active_processes: dict[str, subprocess.Popen[str]],
    process_lock: threading.Lock,
) -> dict[int, Callable[[int, FrameType | None], Any] | int | None]:
    previous_handlers: dict[
        int, Callable[[int, FrameType | None], Any] | int | None
    ] = {}

    def _handler(signum: int, _frame: FrameType | None) -> None:
        _print_prefixed_log(
            "wrapper", f"received signal {signum}; stopping active children"
        )
        stop_event.set()
        _terminate_active_processes(
            active_processes=active_processes,
            process_lock=process_lock,
        )

    for candidate in (
        getattr(signal, "SIGINT", None),
        getattr(signal, "SIGTERM", None),
    ):
        if candidate is None:
            continue
        previous_handlers[int(candidate)] = signal.getsignal(candidate)
        signal.signal(candidate, _handler)
    return previous_handlers


def _restore_signal_handlers(
    previous_handlers: dict[int, Callable[[int, FrameType | None], Any] | int | None],
) -> None:
    for signum, previous in previous_handlers.items():
        signal.signal(signum, previous)


def _print_plan(plan: BudgetPlan) -> None:
    print(f"Mode: {plan.mode}")
    print(f"Contestants: {', '.join(plan.contestants)}")
    print(f"Judge: {plan.judge_provider}")
    print(f"Dataset manifest: {plan.dataset_manifest}")
    print()
    print("Requirement checks:")
    for check in plan.requirement_checks:
        status = _requirement_status_label(check)
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    preliminary_args = parser.parse_args(raw_argv)
    effective_argv = list(raw_argv)
    if str(getattr(preliminary_args, "resume_session_id", "") or "").strip():
        try:
            resume_payload = _load_session_payload(
                _validate_session_id(preliminary_args.resume_session_id)
            )
        except MatrixConfigError as exc:
            parser.error(str(exc))
            return 2
        stored_argv = resume_payload.get("requested_argv", [])
        if not isinstance(stored_argv, list) or not all(
            isinstance(item, str) for item in stored_argv
        ):
            parser.error(
                f"Resume session {preliminary_args.resume_session_id} is missing its original wrapper arguments."
            )
            return 2
        effective_argv = [str(item) for item in stored_argv]

    args = parser.parse_args(effective_argv)
    try:
        plan = _build_budget_plan(args)
        session_context, existing_session_payload, effective_argv = (
            _resolve_session_context(
                resume_session_id=str(
                    getattr(preliminary_args, "resume_session_id", "") or ""
                ),
                plan=plan,
                requested_argv=effective_argv,
            )
        )
    except MatrixConfigError as exc:
        parser.error(str(exc))
        return 2

    _print_plan(plan)
    if session_context is not None:
        print()
        _print_resume_banner(session_context)
    missing_checks = [
        check
        for check in plan.requirement_checks
        if check.blocking and not check.satisfied
    ]
    if missing_checks:
        print(
            f"\nCannot continue: {len(missing_checks)} prerequisite(s) are still missing."
        )
        return 1

    later_checks = [
        check
        for check in plan.requirement_checks
        if not check.blocking and not check.satisfied
    ]

    if str(args.mode) == "validate":
        if later_checks:
            print(f"\nLater live-run requirements still pending: {len(later_checks)}.")
        print("\nValidation succeeded. No benchmark subprocesses were started.")
        return 0

    if session_context is None:
        parser.error(
            "Non-validate SPoRC budget runs require a wrapper session context."
        )
        return 2

    session_payload = _ensure_session_payload(
        session_context=session_context,
        existing_payload=existing_session_payload,
        plan=plan,
        requested_argv=effective_argv,
    )
    profiles = _provider_index(plan.provider_profiles_path)
    state_lock = threading.Lock()
    process_lock = threading.Lock()
    stop_event = threading.Event()
    active_processes: dict[str, subprocess.Popen[str]] = {}
    previous_handlers = _install_stop_signal_handlers(
        stop_event=stop_event,
        active_processes=active_processes,
        process_lock=process_lock,
    )
    exit_code = 130
    try:
        _update_session_status(
            session_context=session_context,
            payload=session_payload,
            status="running",
        )
        exit_code = _run_pending_steps(
            steps=plan.manifest_steps,
            plan=plan,
            session_context=session_context,
            session_payload=session_payload,
            state_lock=state_lock,
            stop_event=stop_event,
            active_processes=active_processes,
            process_lock=process_lock,
            profiles=profiles,
            args=args,
            parallel=False,
        )
        if exit_code == 0:
            exit_code = _run_pending_steps(
                steps=plan.run_steps,
                plan=plan,
                session_context=session_context,
                session_payload=session_payload,
                state_lock=state_lock,
                stop_event=stop_event,
                active_processes=active_processes,
                process_lock=process_lock,
                profiles=profiles,
                args=args,
                parallel=True,
            )
    except KeyboardInterrupt:
        stop_event.set()
        _terminate_active_processes(
            active_processes=active_processes,
            process_lock=process_lock,
        )
        exit_code = 130
    finally:
        _restore_signal_handlers(previous_handlers)

    final_status = (
        "completed"
        if exit_code == 0
        else ("interrupted" if exit_code == 130 else "failed")
    )
    _update_session_status(
        session_context=session_context,
        payload=session_payload,
        status=final_status,
    )

    if exit_code == 0:
        print("\nBudget workflow completed successfully.")
    elif exit_code == 130:
        print(
            "\nBudget workflow interrupted. Resume with the session UUID shown below."
        )
    else:
        print("\nBudget workflow stopped after a non-retryable step failure.")
    print(f"Resume command: {session_context.resume_command}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
