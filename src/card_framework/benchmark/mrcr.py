"""Resolve MRCR benchmark client settings without import-time side effects."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf
import requests

from card_framework.shared.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROVIDER_PROFILES_PATH,
    DEFAULT_QA_CONFIG_PATH,
)

DEFAULT_TIMEOUT_SECONDS = 30.0
VLLM_DEFAULT_PROVIDER_ID = "vllm_default"
_VLLM_TARGET_SUFFIXES = (
    "card_framework.providers.vllm_provider.VLLMProvider",
)


@dataclass(slots=True, frozen=True)
class MRCRClientConfig:
    """Describe the resolved vLLM client settings for MRCR."""

    base_url: str
    api_key: str
    timeout_seconds: float
    extra_body: dict[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class MRCRModelMetadata:
    """Describe the discovered model metadata for one vLLM endpoint."""

    model_id: str
    max_context_window: int


def _load_config(path: Path) -> DictConfig:
    """Load one YAML config file into a DictConfig object."""
    config = OmegaConf.load(path)
    if not isinstance(config, DictConfig):
        raise ValueError(f"Unable to load DictConfig from {path}")
    return config


def _as_plain_mapping(value: Any) -> dict[str, Any]:
    """Convert OmegaConf and mapping-like values into a plain dict."""
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        plain = OmegaConf.to_container(value, resolve=True)
        return plain if isinstance(plain, dict) else {}
    if isinstance(value, dict):
        return value
    return {}


def _looks_like_vllm_target(target: Any) -> bool:
    """Return whether one provider target string points at the vLLM provider."""
    normalized = str(target).strip()
    return any(normalized.endswith(suffix) for suffix in _VLLM_TARGET_SUFFIXES)


def _extract_vllm_from_base_config(path: Path) -> dict[str, Any]:
    """Return vLLM config from the base runtime config when configured."""
    if not path.exists():
        return {}
    config = _load_config(path)
    llm_cfg = _as_plain_mapping(config.get("llm"))
    if not _looks_like_vllm_target(llm_cfg.get("_target_")):
        return {}

    extra_body = _as_plain_mapping(llm_cfg.get("extra_body"))
    thinking_extra_body = _as_plain_mapping(llm_cfg.get("thinking_extra_body"))
    resolved_extra_body = thinking_extra_body or extra_body or None
    return {
        "base_url": str(llm_cfg.get("base_url", "")).strip(),
        "api_key": str(llm_cfg.get("api_key", "EMPTY")).strip() or "EMPTY",
        "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
        "extra_body": resolved_extra_body,
    }


def _extract_vllm_from_provider_profiles(path: Path) -> dict[str, Any]:
    """Return the default MRCR vLLM config from provider profiles."""
    config = _load_config(path)
    raw_providers = config.get("providers", [])
    providers: list[Any]
    if isinstance(raw_providers, ListConfig):
        resolved_providers = OmegaConf.to_container(raw_providers, resolve=True)
        providers = resolved_providers if isinstance(resolved_providers, list) else []
    elif isinstance(raw_providers, list):
        providers = raw_providers
    else:
        raise ValueError(f"providers must be a list in {path}")

    for provider in providers:
        provider_mapping = _as_plain_mapping(provider)
        if str(provider_mapping.get("id", "")).strip() != VLLM_DEFAULT_PROVIDER_ID:
            continue
        llm_cfg = _as_plain_mapping(provider_mapping.get("llm"))
        if not _looks_like_vllm_target(llm_cfg.get("_target_")):
            continue
        extra_body = _as_plain_mapping(llm_cfg.get("extra_body"))
        thinking_extra_body = _as_plain_mapping(llm_cfg.get("thinking_extra_body"))
        resolved_extra_body = thinking_extra_body or extra_body or None
        return {
            "base_url": str(llm_cfg.get("base_url", "")).strip(),
            "api_key": str(llm_cfg.get("api_key", "EMPTY")).strip() or "EMPTY",
            "timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
            "extra_body": resolved_extra_body,
        }

    raise ValueError(
        f"Provider profiles must define '{VLLM_DEFAULT_PROVIDER_ID}' with a vLLM target."
    )


def _extract_vllm_overrides_from_qa_config(path: Path) -> dict[str, Any]:
    """Return QA-config overrides for MRCR vLLM connectivity."""
    config = _load_config(path)
    vllm_cfg = _as_plain_mapping(config.get("vllm"))
    chat_template_kwargs = _as_plain_mapping(vllm_cfg.get("chat_template_kwargs"))
    extra_body = {"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else None
    timeout_raw = vllm_cfg.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    return {
        "base_url": str(vllm_cfg.get("base_url", "")).strip(),
        "api_key": str(vllm_cfg.get("api_key", "")).strip(),
        "timeout_seconds": float(timeout_raw),
        "extra_body": extra_body,
    }


def resolve_vllm_client_config(
    *,
    provider_profiles_path: Path = DEFAULT_PROVIDER_PROFILES_PATH,
    qa_config_path: Path = DEFAULT_QA_CONFIG_PATH,
    base_config_path: Path = DEFAULT_CONFIG_PATH,
) -> MRCRClientConfig:
    """Resolve the vLLM client settings used by the MRCR runner.

    Args:
        provider_profiles_path: Provider profile YAML path.
        qa_config_path: QA benchmark config YAML path.
        base_config_path: Base runtime config YAML path.

    Returns:
        Resolved vLLM client settings with environment-variable overrides applied.

    Raises:
        ValueError: If required settings are missing or invalid.
    """
    resolved = _extract_vllm_from_provider_profiles(provider_profiles_path)

    base_config = _extract_vllm_from_base_config(base_config_path)
    if base_config:
        resolved.update(
            {
                "base_url": base_config["base_url"] or resolved.get("base_url", ""),
                "api_key": base_config["api_key"] or resolved.get("api_key", "EMPTY"),
                "timeout_seconds": base_config["timeout_seconds"],
                "extra_body": base_config.get("extra_body") or resolved.get("extra_body"),
            }
        )

    qa_overrides = _extract_vllm_overrides_from_qa_config(qa_config_path)
    if qa_overrides.get("base_url"):
        resolved["base_url"] = qa_overrides["base_url"]
    if qa_overrides.get("api_key"):
        resolved["api_key"] = qa_overrides["api_key"]
    if qa_overrides.get("timeout_seconds"):
        resolved["timeout_seconds"] = qa_overrides["timeout_seconds"]
    if qa_overrides.get("extra_body") is not None:
        resolved["extra_body"] = qa_overrides["extra_body"]

    env_base_url = os.getenv("VLLM_BASE_URL", "").strip()
    env_api_key = os.getenv("VLLM_API_KEY", "").strip()
    env_timeout = os.getenv("VLLM_TIMEOUT_SECONDS", "").strip()
    if env_base_url:
        resolved["base_url"] = env_base_url
    if env_api_key:
        resolved["api_key"] = env_api_key
    if env_timeout:
        resolved["timeout_seconds"] = float(env_timeout)

    base_url = str(resolved.get("base_url", "")).strip()
    api_key = str(resolved.get("api_key", "EMPTY")).strip() or "EMPTY"
    timeout_seconds = float(resolved.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    if not base_url:
        raise ValueError("Resolved MRCR vLLM base_url is empty.")
    if timeout_seconds <= 0:
        raise ValueError("Resolved MRCR vLLM timeout_seconds must be positive.")

    extra_body = resolved.get("extra_body")
    return MRCRClientConfig(
        base_url=base_url,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        extra_body=extra_body if isinstance(extra_body, dict) else None,
    )


def build_completion_request(
    *,
    model_id: str,
    messages: list[dict[str, Any]],
    client_config: MRCRClientConfig,
) -> dict[str, Any]:
    """Build one OpenAI-compatible chat-completions request payload."""
    payload: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    if client_config.extra_body:
        payload["extra_body"] = client_config.extra_body
    return payload


def fetch_model_metadata(client_config: MRCRClientConfig) -> MRCRModelMetadata:
    """Fetch model metadata from one OpenAI-compatible `/models` endpoint."""
    headers: dict[str, str] = {}
    if client_config.api_key != "EMPTY":
        headers["Authorization"] = f"Bearer {client_config.api_key}"
    response = requests.get(
        f"{client_config.base_url.rstrip('/')}/models",
        headers=headers,
        timeout=client_config.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", [])
    if not isinstance(models, list) or not models:
        raise ValueError(f"No models returned from {client_config.base_url}/models")

    first_model = models[0]
    if not isinstance(first_model, dict):
        raise ValueError("First model metadata entry must be an object.")

    model_id = str(first_model.get("id", "")).strip()
    max_context_window = int(first_model.get("max_model_len", 0))
    if not model_id:
        raise ValueError("Model metadata payload is missing `id`.")
    if max_context_window <= 0:
        raise ValueError("Model metadata payload is missing a positive `max_model_len`.")
    return MRCRModelMetadata(
        model_id=model_id,
        max_context_window=max_context_window,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the MRCR helper CLI argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Resolve the MRCR runner's vLLM client settings and verify endpoint "
            "metadata without triggering dataset downloads at import time."
        )
    )
    parser.add_argument(
        "--provider-profiles",
        default=str(DEFAULT_PROVIDER_PROFILES_PATH),
        help="Path to provider_profiles.yaml.",
    )
    parser.add_argument(
        "--qa-config",
        default=str(DEFAULT_QA_CONFIG_PATH),
        help="Path to qa_config.yaml.",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the base config.yaml.",
    )
    parser.add_argument(
        "--skip-model-metadata",
        action="store_true",
        help="Skip the live /models call and print only the resolved client config.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the MRCR helper CLI."""
    args = build_arg_parser().parse_args(argv)
    client_config = resolve_vllm_client_config(
        provider_profiles_path=Path(args.provider_profiles),
        qa_config_path=Path(args.qa_config),
        base_config_path=Path(args.config),
    )
    payload: dict[str, Any] = {
        "client_config": {
            "base_url": client_config.base_url,
            "api_key": client_config.api_key,
            "timeout_seconds": client_config.timeout_seconds,
            "extra_body": client_config.extra_body,
        }
    }
    if not args.skip_model_metadata:
        metadata = fetch_model_metadata(client_config)
        payload["model_metadata"] = {
            "model_id": metadata.model_id,
            "max_context_window": metadata.max_context_window,
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
