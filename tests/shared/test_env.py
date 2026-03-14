"""Tests for repository-root dotenv loading."""

from __future__ import annotations

import os

from card_framework.shared.env import load_repo_dotenv


def test_load_repo_dotenv_loads_root_file_without_overriding(
    monkeypatch,
    tmp_path,
) -> None:
    """Load values from the repository-root .env file."""

    env_path = tmp_path / ".env"
    env_path.write_text(
        (
            "AUTO_ENV_LOADED=from_dotenv\n"
            "EXISTING_ENV_SHOULD_WIN=from_dotenv\n"
            "HF_ACCESS_TOKEN=hf_from_dotenv\n"
            "VLLM_URL=http://127.0.0.1:8000/v1\n"
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("AUTO_ENV_LOADED", raising=False)
    monkeypatch.delenv("HF_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("VLLM_URL", raising=False)
    monkeypatch.delenv("CARD_FRAMEWORK_VLLM_URL", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setenv("EXISTING_ENV_SHOULD_WIN", "already_set")

    loaded_path = load_repo_dotenv(repo_root=tmp_path)

    assert loaded_path == env_path.resolve()
    assert os.environ["AUTO_ENV_LOADED"] == "from_dotenv"
    assert os.environ["EXISTING_ENV_SHOULD_WIN"] == "already_set"
    assert os.environ["HF_TOKEN"] == "hf_from_dotenv"
    assert os.environ["HUGGINGFACE_HUB_TOKEN"] == "hf_from_dotenv"
    assert os.environ["CARD_FRAMEWORK_VLLM_URL"] == "http://127.0.0.1:8000/v1"
    assert os.environ["VLLM_BASE_URL"] == "http://127.0.0.1:8000/v1"


def test_load_repo_dotenv_returns_none_when_file_missing(tmp_path) -> None:
    """Return None when no repository-root .env file exists."""

    assert load_repo_dotenv(repo_root=tmp_path) is None
