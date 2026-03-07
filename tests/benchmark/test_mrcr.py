"""Unit tests for MRCR benchmark config resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.mrcr import (
    MRCRClientConfig,
    build_completion_request,
    fetch_model_metadata,
    resolve_vllm_client_config,
)


def _write_text(path: Path, content: str) -> None:
    """Write UTF-8 text fixtures to disk."""

    path.write_text(content, encoding="utf-8")


def test_resolve_vllm_client_config_prefers_base_config_vllm(
    tmp_path: Path,
) -> None:
    """Use the repo base config when it is already configured for vLLM."""

    base_config_path = tmp_path / "config.yaml"
    provider_profiles_path = tmp_path / "provider_profiles.yaml"
    qa_config_path = tmp_path / "qa_config.yaml"

    _write_text(
        base_config_path,
        """
llm:
  _target_: providers.vllm_provider.VLLMProvider
  base_url: "http://repo-vllm:8000/v1"
  api_key: "EMPTY"
  thinking_extra_body:
    chat_template_kwargs:
      enable_thinking: false
""".strip(),
    )
    _write_text(
        provider_profiles_path,
        """
providers:
  - id: vllm_default
    description: Test vLLM provider
    required_env: []
    llm:
      _target_: providers.vllm_provider.VLLMProvider
      base_url: "http://profile-vllm:8000/v1"
      api_key: "EMPTY"
""".strip(),
    )
    _write_text(qa_config_path, "vllm: {}\n")

    config = resolve_vllm_client_config(
        provider_profiles_path=provider_profiles_path,
        qa_config_path=qa_config_path,
        base_config_path=base_config_path,
    )

    assert config.base_url == "http://repo-vllm:8000/v1"
    assert config.api_key == "EMPTY"
    assert config.timeout_seconds == pytest.approx(30.0)
    assert config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_resolve_vllm_client_config_prefers_qa_config_values(
    tmp_path: Path,
) -> None:
    """Resolve vLLM connection settings from QA config over provider defaults."""

    provider_profiles_path = tmp_path / "provider_profiles.yaml"
    qa_config_path = tmp_path / "qa_config.yaml"

    _write_text(
        provider_profiles_path,
        """
providers:
  - id: vllm_default
    description: Test vLLM provider
    required_env: []
    llm:
      _target_: providers.vllm_provider.VLLMProvider
      base_url: "http://127.0.0.1:8000/v1"
      api_key: "EMPTY"
""".strip(),
    )
    _write_text(
        qa_config_path,
        """
vllm:
  base_url: "http://PRIVATE_ENDPOINT_REDACTED/v1"
  api_key: "test-key"
  timeout_seconds: 45
  chat_template_kwargs:
    enable_thinking: false
""".strip(),
    )

    config = resolve_vllm_client_config(
        provider_profiles_path=provider_profiles_path,
        qa_config_path=qa_config_path,
    )

    assert config.base_url == "http://PRIVATE_ENDPOINT_REDACTED/v1"
    assert config.api_key == "test-key"
    assert config.timeout_seconds == pytest.approx(45.0)
    assert config.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_resolve_vllm_client_config_allows_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow environment variables to override QA and provider config."""

    provider_profiles_path = tmp_path / "provider_profiles.yaml"
    qa_config_path = tmp_path / "qa_config.yaml"

    _write_text(
        provider_profiles_path,
        """
providers:
  - id: vllm_default
    description: Test vLLM provider
    required_env: []
    llm:
      _target_: providers.vllm_provider.VLLMProvider
      base_url: "http://127.0.0.1:8000/v1"
      api_key: "EMPTY"
""".strip(),
    )
    _write_text(
        qa_config_path,
        """
vllm:
  base_url: "http://qa-config:8000/v1"
  api_key: "qa-key"
  timeout_seconds: 15
""".strip(),
    )

    monkeypatch.setenv("VLLM_BASE_URL", "http://env-override:9000/v1")
    monkeypatch.setenv("VLLM_API_KEY", "env-key")
    monkeypatch.setenv("VLLM_TIMEOUT_SECONDS", "90")

    config = resolve_vllm_client_config(
        provider_profiles_path=provider_profiles_path,
        qa_config_path=qa_config_path,
    )

    assert config.base_url == "http://env-override:9000/v1"
    assert config.api_key == "env-key"
    assert config.timeout_seconds == pytest.approx(90.0)


def test_build_completion_request_includes_extra_body_when_present() -> None:
    """Attach vLLM-specific request extras to the completion payload."""

    client_config = MRCRClientConfig(
        base_url="http://127.0.0.1:8000/v1",
        api_key="EMPTY",
        timeout_seconds=30.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    payload = build_completion_request(
        model_id="test-model",
        messages=[{"role": "user", "content": "hello"}],
        client_config=client_config,
    )

    assert payload == {
        "model": "test-model",
        "messages": [{"role": "user", "content": "hello"}],
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }


def test_fetch_model_metadata_reads_max_context_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read max context metadata from a vLLM/OpenAI-compatible `/models` payload."""

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "data": [
                    {
                        "id": "my-vllm-model",
                        "max_model_len": 131072,
                    }
                ]
            }

    def _fake_get(*args: object, **kwargs: object) -> _FakeResponse:
        return _FakeResponse()

    monkeypatch.setattr("benchmark.mrcr.requests.get", _fake_get)

    metadata = fetch_model_metadata(
        MRCRClientConfig(
            base_url="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            timeout_seconds=30.0,
        )
    )

    assert metadata.model_id == "my-vllm-model"
    assert metadata.max_context_window == 131072
