"""Normalize human-edited runtime configs into runtime-ready mappings."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

_STAGE_LLM_NAMES: tuple[str, ...] = ("summarizer", "critic", "interjector")
_HF_ENV_NAMES: tuple[str, ...] = (
    "HF_TOKEN",
    "HF_ACCESS_TOKEN",
    "HUGGINGFACE_TOKEN",
    "HUGGINGFACE_HUB_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HF_API_TOKEN",
)
_PROVIDER_SECRET_ENV_NAMES: dict[str, tuple[str, ...]] = {
    "card_framework.providers.deepseek_provider.DeepSeekProvider": (
        "DEEPSEEK_API_KEY",
    ),
    "card_framework.providers.google_genai_provider.GoogleGenAIProvider": (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ),
    "card_framework.providers.glm_provider.GLMProvider": (
        "ZAI_API_KEY",
        "GLM_API_KEY",
    ),
    "card_framework.providers.huggingface_provider.HuggingfaceProvider": (
        *_HF_ENV_NAMES,
    ),
    "card_framework.providers.nanbeige_provider.NanbeigeProvider": (
        *_HF_ENV_NAMES,
    ),
}


class RuntimeConfigError(ValueError):
    """Describe one invalid or incomplete runtime config."""


def load_runtime_config(config_path: Path) -> DictConfig:
    """Load and normalize one runtime config file.

    Args:
        config_path: YAML config path.

    Returns:
        Normalized runtime config.

    Raises:
        RuntimeConfigError: The file could not be loaded into a mapping.
    """

    config = OmegaConf.load(config_path)
    if not isinstance(config, DictConfig):
        raise RuntimeConfigError(f"Unable to load DictConfig from {config_path}")
    return normalize_runtime_config(config)


def normalize_runtime_config(
    config: DictConfig | Mapping[str, Any],
) -> DictConfig:
    """Materialize runtime sections from selector and profile config.

    Args:
        config: Raw runtime config.

    Returns:
        Config with explicit ``llm``, ``embedding``, and ``stage_llm`` sections.
    """

    if isinstance(config, DictConfig):
        payload = OmegaConf.to_container(config, resolve=False)
        if not isinstance(payload, dict):
            raise RuntimeConfigError("Runtime config must resolve to a mapping.")
        return OmegaConf.create(normalize_runtime_config_mapping(payload))
    return OmegaConf.create(normalize_runtime_config_mapping(config))


def normalize_runtime_config_mapping(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize runtime sections from a plain config mapping.

    Args:
        config: Raw config mapping.

    Returns:
        Copy of the config with explicit runtime sections populated.
    """

    normalized = deepcopy(dict(config))
    profiles_cfg = _as_mapping(normalized.get("profiles"))
    selection_cfg = _as_mapping(normalized.get("selection"))
    if not profiles_cfg and not selection_cfg:
        explicit_stage_llm = _as_mapping(normalized.get("stage_llm"))
        if explicit_stage_llm:
            for stage_name in _STAGE_LLM_NAMES:
                explicit_stage_llm.setdefault(stage_name, {})
            normalized["stage_llm"] = explicit_stage_llm
        return normalized

    if _has_target_config(_as_mapping(normalized.get("llm"))):
        llm_cfg = deepcopy(_as_mapping(normalized["llm"]))
    else:
        llm_cfg = _resolve_selected_profile(
            profiles=_as_mapping(profiles_cfg.get("llm")),
            profile_name=selection_cfg.get("llm"),
            config_label="selection.llm",
        )
    normalized["llm"] = llm_cfg

    if _has_target_config(_as_mapping(normalized.get("embedding"))):
        embedding_cfg = deepcopy(_as_mapping(normalized["embedding"]))
    else:
        embedding_cfg = _resolve_selected_profile(
            profiles=_as_mapping(profiles_cfg.get("embedding")),
            profile_name=selection_cfg.get("embedding"),
            config_label="selection.embedding",
        )
    normalized["embedding"] = embedding_cfg

    explicit_stage_llm = _as_mapping(normalized.get("stage_llm"))
    if explicit_stage_llm:
        stage_llm_cfg = deepcopy(explicit_stage_llm)
        for stage_name in _STAGE_LLM_NAMES:
            stage_llm_cfg.setdefault(stage_name, {})
    else:
        stage_llm_cfg = _build_stage_llm_config(
            llm_profiles=_as_mapping(profiles_cfg.get("llm")),
            stage_selection=_as_mapping(selection_cfg.get("stage_llm")),
        )
    normalized["stage_llm"] = stage_llm_cfg
    return normalized


def required_env_names_for_llm_config(llm_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """Return required env names for one LLM provider config.

    Args:
        llm_cfg: One LLM provider config mapping.

    Returns:
        Environment variable names required by the selected provider.
    """

    target = str(llm_cfg.get("_target_", "")).strip()
    return _PROVIDER_SECRET_ENV_NAMES.get(target, ())


def validate_runtime_credentials(config: DictConfig | Mapping[str, Any]) -> None:
    """Validate env-backed credentials for the active runtime config.

    Args:
        config: Raw or normalized runtime config.

    Raises:
        RuntimeConfigError: One or more required credentials are missing.
    """

    normalized = normalize_runtime_config(config)
    resolved = OmegaConf.to_container(normalized, resolve=True)
    if not isinstance(resolved, dict):
        raise RuntimeConfigError("Runtime config must resolve to a mapping.")

    missing_messages: list[str] = []

    for field_path, llm_cfg in _iter_llm_sections(resolved):
        env_names = required_env_names_for_llm_config(llm_cfg)
        if not env_names:
            continue
        if str(llm_cfg.get("api_key", "")).strip():
            continue
        missing_messages.append(
            f"`{'.'.join(field_path)}.api_key` requires one of: {', '.join(env_names)}"
        )

    diarization_cfg = _as_mapping(_as_mapping(resolved.get("audio")).get("diarization"))
    if str(diarization_cfg.get("provider", "")).strip() == "pyannote_community1":
        pyannote_cfg = _as_mapping(diarization_cfg.get("pyannote"))
        if not str(pyannote_cfg.get("auth_token", "")).strip():
            auth_token_env = str(pyannote_cfg.get("auth_token_env", "")).strip()
            env_names = (auth_token_env,) if auth_token_env else _HF_ENV_NAMES
            missing_messages.append(
                "`.audio.diarization.pyannote.auth_token` requires one of: "
                + ", ".join(env_names)
            )

    if missing_messages:
        detail = "\n".join(f"- {message}" for message in missing_messages)
        raise RuntimeConfigError(
            "Runtime config is missing required environment-backed credentials:\n"
            f"{detail}"
        )


def _build_stage_llm_config(
    *,
    llm_profiles: Mapping[str, Any],
    stage_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the explicit stage LLM config mapping."""

    stage_llm_cfg: dict[str, Any] = {}
    for stage_name in _STAGE_LLM_NAMES:
        selected_profile = str(stage_selection.get(stage_name, "default")).strip()
        if selected_profile in {"", "default"}:
            stage_llm_cfg[stage_name] = {}
            continue
        stage_llm_cfg[stage_name] = _resolve_selected_profile(
            profiles=llm_profiles,
            profile_name=selected_profile,
            config_label=f"selection.stage_llm.{stage_name}",
        )
    return stage_llm_cfg


def _resolve_selected_profile(
    *,
    profiles: Mapping[str, Any],
    profile_name: object,
    config_label: str,
) -> dict[str, Any]:
    """Resolve one selected profile from the available profile mapping."""

    selected = str(profile_name or "").strip()
    if not selected:
        raise RuntimeConfigError(f"{config_label} must be set.")

    resolved_profile = _as_mapping(profiles.get(selected))
    if not resolved_profile:
        available = ", ".join(sorted(str(name) for name in profiles)) or "<none>"
        raise RuntimeConfigError(
            f"{config_label} references unknown profile '{selected}'. "
            f"Available profiles: {available}."
        )
    if not _has_target_config(resolved_profile):
        raise RuntimeConfigError(
            f"Profile '{selected}' selected by {config_label} must define `_target_`."
        )
    return deepcopy(resolved_profile)


def _iter_llm_sections(
    config: Mapping[str, Any],
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    """Return the active LLM config sections from one runtime mapping."""

    sections: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    llm_cfg = _as_mapping(config.get("llm"))
    if llm_cfg:
        sections.append((("llm",), llm_cfg))
    stage_llm_cfg = _as_mapping(config.get("stage_llm"))
    for stage_name in _STAGE_LLM_NAMES:
        stage_cfg = _as_mapping(stage_llm_cfg.get(stage_name))
        if stage_cfg:
            sections.append((("stage_llm", stage_name), stage_cfg))
    return sections


def _has_target_config(value: Mapping[str, Any]) -> bool:
    """Return whether one mapping contains a non-empty Hydra target."""

    target = value.get("_target_")
    return isinstance(target, str) and bool(target.strip())


def _as_mapping(value: Any) -> dict[str, Any]:
    """Convert a mapping-like value into a plain mutable dictionary."""

    if isinstance(value, DictConfig):
        resolved = OmegaConf.to_container(value, resolve=False)
        return resolved if isinstance(resolved, dict) else {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}
