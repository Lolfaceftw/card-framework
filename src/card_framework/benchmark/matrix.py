"""Matrix expansion and budget-aware pruning for benchmark execution."""

from __future__ import annotations

from dataclasses import asdict
import math
import os
import re
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from card_framework.benchmark.types import (
    BenchmarkCell,
    BenchmarkPreset,
    EmbeddingProfile,
    ProviderProfile,
)
from card_framework.shared.config_loading import normalize_runtime_config_mapping
from card_framework.shared.paths import REPO_ROOT

_VLLM_PROVIDER_TARGET = "card_framework.providers.vllm_provider.VLLMProvider"
_VLLM_URL_ENV_NAMES = (
    "VLLM_URL",
    "CARD_FRAMEWORK_VLLM_URL",
    "CARD_FRAMEWORK_VLLM_BASE_URL",
    "VLLM_BASE_URL",
)
_VLLM_MODEL_ENV_NAMES = (
    "CARD_FRAMEWORK_VLLM_MODEL",
    "VLLM_MODEL",
)
_LEGACY_CONFIG_PATH = REPO_ROOT / "conf" / "config.yaml"
_LEGACY_PROVIDER_TARGET_PREFIX = "providers."
_CURRENT_PROVIDER_TARGET_PREFIX = "card_framework.providers."
_OC_ENV_PATTERN = re.compile(r"^\$\{oc\.env:([^,}]+)(?:,(.*))?\}$")


class MatrixConfigError(RuntimeError):
    """Raised when benchmark matrix configuration is invalid."""


def default_presets() -> dict[str, BenchmarkPreset]:
    """Return built-in benchmark presets."""
    return {
        "smoke": BenchmarkPreset(
            name="smoke",
            target_duration_seconds=900,
            estimated_sample_seconds=60,
            max_samples=1,
            repeats=1,
            include_embedding_profiles=["disabled"],
        ),
        "hourly": BenchmarkPreset(
            name="hourly",
            target_duration_seconds=3600,
            estimated_sample_seconds=120,
            max_samples=24,
            repeats=2,
            include_embedding_profiles=["disabled", "enabled"],
        ),
        "full": BenchmarkPreset(
            name="full",
            target_duration_seconds=4 * 3600,
            estimated_sample_seconds=180,
            max_samples=100,
            repeats=3,
            include_embedding_profiles=["disabled", "enabled"],
        ),
    }


def load_provider_profiles(path: Path) -> list[ProviderProfile]:
    """Load provider profiles from YAML file."""
    if not path.exists():
        raise MatrixConfigError(f"Provider profiles file not found: {path}")

    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(payload, dict):
        raise MatrixConfigError("Provider profile config must be a mapping")

    raw_profiles = payload.get("providers")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise MatrixConfigError(
            "Provider profile config requires a non-empty providers list"
        )

    profiles: list[ProviderProfile] = []
    for raw_profile in raw_profiles:
        if not isinstance(raw_profile, dict):
            raise MatrixConfigError("Provider profile entries must be objects")

        provider_id = str(raw_profile.get("id", "")).strip()
        description = str(raw_profile.get("description", "")).strip()
        llm_config = raw_profile.get("llm")
        required_env = raw_profile.get("required_env", [])

        if not provider_id or not description or not isinstance(llm_config, dict):
            raise MatrixConfigError(
                "Each provider profile requires id, description, and llm mapping"
            )
        if not isinstance(required_env, list):
            raise MatrixConfigError(
                f"Provider profile '{provider_id}' required_env must be list[str]"
            )

        profiles.append(
            ProviderProfile(
                provider_id=provider_id,
                description=description,
                llm_config=llm_config,
                required_env=[str(key) for key in required_env],
            )
        )

    return profiles


def build_embedding_profiles(base_cfg: DictConfig) -> dict[str, EmbeddingProfile]:
    """Build default embedding profiles based on repository config."""
    disabled = EmbeddingProfile(
        embedding_id="disabled",
        description="Disable retrieval by using the no-op embedding provider.",
        embedding_config={
            "_target_": "card_framework.providers.null_provider.NoOpEmbeddingProvider"
        },
    )

    base_embedding = OmegaConf.to_container(base_cfg.embedding, resolve=False)
    if not isinstance(base_embedding, dict):
        raise MatrixConfigError("Base config embedding section must be a mapping")
    base_embedding_config = {
        str(key): raw_value for key, raw_value in base_embedding.items()
    }

    if "NoOpEmbeddingProvider" in str(base_embedding_config.get("_target_", "")):
        enabled_config = _load_legacy_embedding_config()
        if enabled_config is None:
            enabled_config = {
                "_target_": "card_framework.providers.sentence_transformer_provider.SentenceTransformerEmbeddingProvider",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
                "torch_dtype": "float32",
                "batch_size": 16,
            }
    else:
        enabled_config = base_embedding_config

    enabled = EmbeddingProfile(
        embedding_id="enabled",
        description="Enable retrieval with configured embedding provider.",
        embedding_config=enabled_config,
    )

    return {"disabled": disabled, "enabled": enabled}


def _load_legacy_embedding_config() -> dict[str, Any] | None:
    """Return a normalized legacy embedding config when one is available.

    During the src-layout migration, some local benchmark runs still keep their
    embedding edits in the legacy repo-root ``conf/config.yaml`` file. When the
    active packaged config still uses the no-op embedding provider, prefer that
    legacy embedding section over the hardcoded MiniLM fallback so benchmark
    retrieval can honor the user's configured model.
    """
    if not _LEGACY_CONFIG_PATH.exists():
        return None

    legacy_payload = OmegaConf.to_container(
        OmegaConf.load(_LEGACY_CONFIG_PATH), resolve=False
    )
    if not isinstance(legacy_payload, dict):
        return None
    legacy_payload = normalize_runtime_config_mapping(legacy_payload)

    raw_embedding = legacy_payload.get("embedding")
    if not isinstance(raw_embedding, dict):
        return None

    normalized_embedding = _normalize_legacy_targets(raw_embedding)
    if "NoOpEmbeddingProvider" in str(normalized_embedding.get("_target_", "")):
        return None
    return normalized_embedding


def _normalize_legacy_targets(value: Any) -> Any:
    """Translate legacy repo-root provider targets into current package paths."""
    if isinstance(value, str):
        if value.startswith(_LEGACY_PROVIDER_TARGET_PREFIX):
            suffix = value.removeprefix(_LEGACY_PROVIDER_TARGET_PREFIX)
            return f"{_CURRENT_PROVIDER_TARGET_PREFIX}{suffix}"
        return value

    if isinstance(value, dict):
        return {
            key: _normalize_legacy_targets(raw_value)
            for key, raw_value in value.items()
        }

    if isinstance(value, list):
        return [_normalize_legacy_targets(raw_value) for raw_value in value]

    return value


def resolve_env_placeholders(value: Any) -> Any:
    """Resolve values of the form ``$ENV{VAR_NAME}`` recursively."""
    if isinstance(value, str) and value.startswith("$ENV{") and value.endswith("}"):
        env_name = value[5:-1]
        resolved = os.getenv(env_name)
        if resolved is None:
            raise MatrixConfigError(
                f"Required environment variable is missing: {env_name}"
            )
        return resolved
    if isinstance(value, str) and (match := _OC_ENV_PATTERN.match(value)):
        env_name = match.group(1).strip()
        resolved = os.getenv(env_name)
        if resolved is not None:
            return resolved
        raw_default = match.group(2)
        if raw_default is None:
            raise MatrixConfigError(
                f"Required environment variable is missing: {env_name}"
            )
        return _strip_env_default(raw_default)

    if isinstance(value, dict):
        return {key: resolve_env_placeholders(raw) for key, raw in value.items()}

    if isinstance(value, list):
        return [resolve_env_placeholders(raw) for raw in value]

    return value


def _references_env_placeholder(value: Any, env_name: str) -> bool:
    """Return True if value tree contains ``$ENV{<env_name>}`` placeholder."""
    expected = f"$ENV{{{env_name}}}"
    if isinstance(value, str):
        if value == expected:
            return True
        match = _OC_ENV_PATTERN.match(value)
        return bool(match and match.group(1).strip() == env_name)
    if isinstance(value, dict):
        return any(_references_env_placeholder(raw, env_name) for raw in value.values())
    if isinstance(value, list):
        return any(_references_env_placeholder(raw, env_name) for raw in value)
    return False


def _strip_env_default(raw_default: str) -> str:
    """Normalize one default value extracted from an ``oc.env`` placeholder."""

    normalized = raw_default.strip()
    if (
        len(normalized) >= 2
        and normalized[0] == normalized[-1]
        and normalized[0] in {'"', "'"}
    ):
        return normalized[1:-1]
    return normalized


def missing_required_env(profile: ProviderProfile) -> list[str]:
    """
    Return missing required env vars for a provider profile.

    Enforcement rule:
    - If a required env is referenced via ``$ENV{VAR}`` in llm_config, it must exist.
    - If llm_config provides concrete values (no placeholder), we do not require env.
    """
    missing: list[str] = []
    for env_name in profile.required_env:
        if _references_env_placeholder(profile.llm_config, env_name) and not os.getenv(
            env_name
        ):
            missing.append(env_name)
    return missing


def _normalize_vllm_base_url(raw_value: str) -> str:
    """Normalize a vLLM base URL from environment configuration."""
    stripped = raw_value.strip().rstrip("/")
    if not stripped:
        return ""
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", stripped):
        return stripped
    return f"http://{stripped}"


def _apply_vllm_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Override vLLM provider config from benchmark-friendly environment variables."""
    if config.get("_target_") != _VLLM_PROVIDER_TARGET:
        return config
    resolved = dict(config)
    for env_name in _VLLM_URL_ENV_NAMES:
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        normalized = _normalize_vllm_base_url(raw_value)
        if not normalized:
            continue
        resolved["base_url"] = normalized
        break
    for env_name in _VLLM_MODEL_ENV_NAMES:
        raw_value = os.getenv(env_name)
        if raw_value is None:
            continue
        normalized = raw_value.strip()
        if not normalized:
            continue
        resolved["model"] = normalized
        break
    return resolved


def resolve_provider_config(profile: ProviderProfile) -> dict[str, Any]:
    """Resolve provider config placeholders into a Hydra-instantiable config mapping."""
    resolved = resolve_env_placeholders(profile.llm_config)
    if not isinstance(resolved, dict):
        raise MatrixConfigError("Resolved provider config must be a mapping.")
    return _apply_vllm_env_overrides(resolved)


def _coverage_first_prune(
    cells: list[BenchmarkCell], max_cells: int
) -> list[BenchmarkCell]:
    """Prune matrix while preserving at least one baseline cell per provider."""
    if len(cells) <= max_cells:
        return cells

    mandatory: list[BenchmarkCell] = []
    seen_providers: set[str] = set()
    for cell in cells:
        if cell.provider_id in seen_providers:
            continue
        if cell.repeat_index == 1 and cell.embedding_id == "disabled":
            mandatory.append(cell)
            seen_providers.add(cell.provider_id)

    for cell in cells:
        if len(mandatory) >= max_cells:
            break
        if cell.provider_id not in seen_providers:
            mandatory.append(cell)
            seen_providers.add(cell.provider_id)

    selected = {cell.cell_id for cell in mandatory}
    ordered_remainder = [cell for cell in cells if cell.cell_id not in selected]

    for cell in ordered_remainder:
        if len(mandatory) >= max_cells:
            break
        mandatory.append(cell)

    return mandatory


def build_cells(
    *,
    profiles: list[ProviderProfile],
    embedding_profiles: dict[str, EmbeddingProfile],
    preset: BenchmarkPreset,
    sample_count: int,
) -> list[BenchmarkCell]:
    """Build and budget-prune benchmark matrix cells."""
    if sample_count <= 0:
        raise MatrixConfigError("sample_count must be positive")

    cells: list[BenchmarkCell] = []
    ordered_profiles = sorted(profiles, key=lambda item: item.provider_id)

    for profile in ordered_profiles:
        llm_config = resolve_provider_config(profile)
        for embedding_id in preset.include_embedding_profiles:
            embedding_profile = embedding_profiles.get(embedding_id)
            if embedding_profile is None:
                raise MatrixConfigError(
                    f"Preset requested unknown embedding profile: {embedding_id}"
                )
            for repeat_index in range(1, preset.repeats + 1):
                cell_id = (
                    f"{profile.provider_id}__{embedding_profile.embedding_id}"
                    f"__r{repeat_index}"
                )
                cells.append(
                    BenchmarkCell(
                        cell_id=cell_id,
                        provider_id=profile.provider_id,
                        embedding_id=embedding_profile.embedding_id,
                        repeat_index=repeat_index,
                        llm_config=llm_config,
                        embedding_config=embedding_profile.embedding_config,
                    )
                )

    estimated_total_seconds = (
        len(cells) * sample_count * preset.estimated_sample_seconds
    )

    if estimated_total_seconds <= preset.target_duration_seconds:
        return cells

    per_cell_cost = max(1, sample_count * preset.estimated_sample_seconds)
    max_cells = max(1, math.floor(preset.target_duration_seconds / per_cell_cost))
    return _coverage_first_prune(cells, max_cells)


def cells_to_dict(cells: list[BenchmarkCell]) -> list[dict[str, Any]]:
    """Serialize cells to dictionaries for reporting."""
    return [asdict(cell) for cell in cells]
