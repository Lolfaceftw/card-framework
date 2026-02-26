"""Matrix expansion and budget-aware pruning for benchmark execution."""

from __future__ import annotations

from dataclasses import asdict
import math
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from benchmark.types import (
    BenchmarkCell,
    BenchmarkPreset,
    EmbeddingProfile,
    ProviderProfile,
)


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
        raise MatrixConfigError("Provider profile config requires a non-empty providers list")

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
        embedding_config={"_target_": "providers.null_provider.NoOpEmbeddingProvider"},
    )

    base_embedding = OmegaConf.to_container(base_cfg.embedding, resolve=False)
    if not isinstance(base_embedding, dict):
        raise MatrixConfigError("Base config embedding section must be a mapping")

    if "NoOpEmbeddingProvider" in str(base_embedding.get("_target_", "")):
        enabled_config = {
            "_target_": "providers.sentence_transformer_provider.SentenceTransformerEmbeddingProvider",
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "torch_dtype": "float32",
            "batch_size": 16,
        }
    else:
        enabled_config = base_embedding

    enabled = EmbeddingProfile(
        embedding_id="enabled",
        description="Enable retrieval with configured embedding provider.",
        embedding_config=enabled_config,
    )

    return {"disabled": disabled, "enabled": enabled}


def resolve_env_placeholders(value: Any) -> Any:
    """Resolve values of the form ``$ENV{VAR_NAME}`` recursively."""
    if isinstance(value, str) and value.startswith("$ENV{") and value.endswith("}"):
        env_name = value[5:-1]
        resolved = os.getenv(env_name)
        if resolved is None:
            raise MatrixConfigError(f"Required environment variable is missing: {env_name}")
        return resolved

    if isinstance(value, dict):
        return {key: resolve_env_placeholders(raw) for key, raw in value.items()}

    if isinstance(value, list):
        return [resolve_env_placeholders(raw) for raw in value]

    return value


def _references_env_placeholder(value: Any, env_name: str) -> bool:
    """Return True if value tree contains ``$ENV{<env_name>}`` placeholder."""
    expected = f"$ENV{{{env_name}}}"
    if isinstance(value, str):
        return value == expected
    if isinstance(value, dict):
        return any(_references_env_placeholder(raw, env_name) for raw in value.values())
    if isinstance(value, list):
        return any(_references_env_placeholder(raw, env_name) for raw in value)
    return False


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


def resolve_provider_config(profile: ProviderProfile) -> dict[str, Any]:
    """Resolve provider config placeholders into a Hydra-instantiable config mapping."""
    return resolve_env_placeholders(profile.llm_config)


def _coverage_first_prune(cells: list[BenchmarkCell], max_cells: int) -> list[BenchmarkCell]:
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
