"""Live integration test for QA benchmark vLLM endpoint configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
import pytest
import requests


def _load_qa_config(path: Path) -> DictConfig:
    """Load QA benchmark config as DictConfig."""
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise AssertionError(f"Unable to load DictConfig from {path}")
    return cfg


def _resolve_vllm_base_url() -> tuple[str, str, float]:
    """Resolve vLLM connection settings from benchmark QA config."""
    config_override = os.getenv("VLLM_BASE_URL", "").strip()
    api_key_override = os.getenv("VLLM_API_KEY", "").strip()
    timeout_override = os.getenv("VLLM_TIMEOUT_SECONDS", "").strip()

    qa_cfg = _load_qa_config(Path("benchmark/qa_config.yaml"))
    base_url = (
        config_override or str(qa_cfg.get("vllm", {}).get("base_url", "")).strip()
    )
    api_key = (
        api_key_override or str(qa_cfg.get("vllm", {}).get("api_key", "EMPTY")).strip()
    )
    timeout_raw = (
        timeout_override
        or str(qa_cfg.get("vllm", {}).get("timeout_seconds", 30)).strip()
    )

    if not base_url:
        raise AssertionError("benchmark/qa_config.yaml must define vllm.base_url")
    timeout_seconds = float(timeout_raw)
    if timeout_seconds <= 0:
        raise AssertionError("vllm.timeout_seconds must be positive")
    return base_url, (api_key or "EMPTY"), timeout_seconds


@pytest.mark.integration
@pytest.mark.network
def test_live_vllm_endpoint_roundtrip_via_qa_config() -> None:
    """Perform a real `/models` and `/chat/completions` roundtrip using QA config."""
    if os.getenv("RUN_VLLM_LIVE", "0") != "1":
        pytest.skip("Set RUN_VLLM_LIVE=1 to run live vLLM integration test.")

    base_url, api_key, timeout_seconds = _resolve_vllm_base_url()
    headers = {"Authorization": f"Bearer {api_key}"} if api_key != "EMPTY" else {}

    models_response = requests.get(
        f"{base_url}/models",
        headers=headers,
        timeout=timeout_seconds,
    )
    models_response.raise_for_status()
    models_payload = models_response.json()
    model_list = models_payload.get("data", [])
    if not isinstance(model_list, list) or not model_list:
        raise AssertionError(f"No models returned from endpoint: {models_payload}")
    model_id = str(model_list[0].get("id", "")).strip()
    if not model_id:
        raise AssertionError(f"First model entry missing id: {model_list[0]}")

    completion_response = requests.post(
        f"{base_url}/chat/completions",
        headers={**headers, "Content-Type": "application/json"},
        json={
            "model": model_id,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return ONLY valid minified JSON with no prose, markdown, or thinking text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        'Output exactly this object and nothing else: {"ok":true,"message":"qa_vllm_live"}'
                    ),
                },
            ],
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": False,
            "max_tokens": 64,
            "temperature": 0,
        },
        timeout=timeout_seconds,
    )
    completion_response.raise_for_status()
    completion_payload = completion_response.json()
    choices = completion_payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        raise AssertionError(
            f"No choices returned from completion: {completion_payload}"
        )
    message = choices[0].get("message", {})
    content = str(message.get("content", "")).strip()
    if not content:
        raise AssertionError(f"Completion content is empty: {completion_payload}")

    payload = json.loads(content)
    assert payload.get("ok") is True
    assert payload.get("message") == "qa_vllm_live"
