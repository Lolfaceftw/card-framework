from __future__ import annotations

from omegaconf import OmegaConf

import card_framework.benchmark.matrix as matrix_module
from card_framework.benchmark.matrix import (
    build_cells,
    build_embedding_profiles,
    load_provider_profiles,
    resolve_provider_config,
)
from card_framework.benchmark.types import (
    BenchmarkPreset,
    EmbeddingProfile,
    ProviderProfile,
)
from card_framework.shared.paths import DEFAULT_PROVIDER_PROFILES_PATH


def test_build_cells_prunes_with_provider_coverage() -> None:
    profiles = [
        ProviderProfile(
            provider_id="p1",
            description="Provider 1",
            llm_config={
                "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
                "base_url": "http://x",
                "api_key": "EMPTY",
            },
        ),
        ProviderProfile(
            provider_id="p2",
            description="Provider 2",
            llm_config={
                "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
                "base_url": "http://x",
                "api_key": "EMPTY",
            },
        ),
        ProviderProfile(
            provider_id="p3",
            description="Provider 3",
            llm_config={
                "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
                "base_url": "http://x",
                "api_key": "EMPTY",
            },
        ),
    ]

    embedding_profiles = {
        "disabled": EmbeddingProfile(
            embedding_id="disabled",
            description="off",
            embedding_config={
                "_target_": "card_framework.providers.null_provider.NoOpEmbeddingProvider"
            },
        ),
        "enabled": EmbeddingProfile(
            embedding_id="enabled",
            description="on",
            embedding_config={
                "_target_": (
                    "card_framework.providers.sentence_transformer_provider."
                    "SentenceTransformerEmbeddingProvider"
                ),
                "model_name": "x",
            },
        ),
    }

    preset = BenchmarkPreset(
        name="custom",
        target_duration_seconds=300,
        estimated_sample_seconds=120,
        max_samples=20,
        repeats=2,
        include_embedding_profiles=["disabled", "enabled"],
    )

    cells = build_cells(
        profiles=profiles,
        embedding_profiles=embedding_profiles,
        preset=preset,
        sample_count=5,
    )

    covered = {
        (cell.provider_id, cell.embedding_id, cell.repeat_index) for cell in cells
    }
    assert ("p1", "disabled", 1) in covered
    assert ("p2", "disabled", 1) in covered
    assert ("p3", "disabled", 1) in covered
    assert len(cells) <= 3


def test_resolve_provider_config_prefers_vllm_url_env(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_URL", "vllm.example.test:8000/v1")
    profile = ProviderProfile(
        provider_id="vllm_default",
        description="Provider 1",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )

    resolved = resolve_provider_config(profile)

    assert resolved["base_url"] == "http://vllm.example.test:8000/v1"


def test_resolve_provider_config_supports_card_framework_vllm_base_url(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_URL", raising=False)
    monkeypatch.delenv("CARD_FRAMEWORK_VLLM_URL", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.setenv(
        "CARD_FRAMEWORK_VLLM_BASE_URL",
        "http://prefixed-vllm.example.test:8000/v1",
    )
    profile = ProviderProfile(
        provider_id="vllm_default",
        description="Provider 1",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )

    resolved = resolve_provider_config(profile)

    assert resolved["base_url"] == "http://prefixed-vllm.example.test:8000/v1"


def test_resolve_provider_config_supports_card_framework_vllm_model(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CARD_FRAMEWORK_VLLM_MODEL", raising=False)
    monkeypatch.delenv("VLLM_MODEL", raising=False)
    monkeypatch.setenv("CARD_FRAMEWORK_VLLM_MODEL", "Qwen3-32B")
    profile = ProviderProfile(
        provider_id="vllm_default",
        description="Provider 1",
        llm_config={
            "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key": "EMPTY",
        },
    )

    resolved = resolve_provider_config(profile)

    assert resolved["model"] == "Qwen3-32B"


def test_resolve_provider_config_supports_oc_env_placeholders(monkeypatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    profile = ProviderProfile(
        provider_id="deepseek_chat",
        description="Provider 1",
        llm_config={
            "_target_": "card_framework.providers.deepseek_provider.DeepSeekProvider",
            "api_key": '${oc.env:DEEPSEEK_API_KEY,""}',
            "model": "deepseek-chat",
            "base_url": "https://api.deepseek.com",
        },
        required_env=["DEEPSEEK_API_KEY"],
    )

    resolved = resolve_provider_config(profile)

    assert resolved["api_key"] == "deepseek-secret"


def test_build_embedding_profiles_uses_legacy_repo_config_when_base_is_noop(
    monkeypatch, tmp_path
) -> None:
    legacy_config = tmp_path / "config.yaml"
    legacy_config.write_text(
        """
embedding:
  _target_: providers.sentence_transformer_provider.SentenceTransformerEmbeddingProvider
  model_name: Qwen/Qwen3-Embedding-4B
  device: cuda
  torch_dtype: float16
  batch_size: 4
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(matrix_module, "_LEGACY_CONFIG_PATH", legacy_config)

    base_cfg = OmegaConf.create(
        {
            "embedding": {
                "_target_": "card_framework.providers.null_provider.NoOpEmbeddingProvider"
            }
        }
    )

    profiles = build_embedding_profiles(base_cfg)

    assert profiles["enabled"].embedding_config == {
        "_target_": (
            "card_framework.providers.sentence_transformer_provider."
            "SentenceTransformerEmbeddingProvider"
        ),
        "model_name": "Qwen/Qwen3-Embedding-4B",
        "device": "cuda",
        "torch_dtype": "float16",
        "batch_size": 4,
    }


def test_build_embedding_profiles_keeps_base_embedding_when_already_enabled(
    monkeypatch, tmp_path
) -> None:
    legacy_config = tmp_path / "config.yaml"
    legacy_config.write_text(
        """
embedding:
  _target_: providers.sentence_transformer_provider.SentenceTransformerEmbeddingProvider
  model_name: Qwen/Qwen3-Embedding-4B
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(matrix_module, "_LEGACY_CONFIG_PATH", legacy_config)

    base_cfg = OmegaConf.create(
        {
            "embedding": {
                "_target_": (
                    "card_framework.providers.sentence_transformer_provider."
                    "SentenceTransformerEmbeddingProvider"
                ),
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
            }
        }
    )

    profiles = build_embedding_profiles(base_cfg)

    assert profiles["enabled"].embedding_config == {
        "_target_": (
            "card_framework.providers.sentence_transformer_provider."
            "SentenceTransformerEmbeddingProvider"
        ),
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu",
    }


def test_default_provider_profiles_include_sporc_budget_sweep_profiles() -> None:
    profiles = {
        profile.provider_id: profile
        for profile in load_provider_profiles(DEFAULT_PROVIDER_PROFILES_PATH)
    }

    assert profiles["openai_gpt_5_4_mini"].llm_config == {
        "_target_": "card_framework.providers.openai_provider.OpenAIProvider",
        "api_key": "$ENV{OPENAI_API_KEY}",
        "model": "gpt-5.4-mini",
        "enable_thinking": False,
    }
    assert profiles["anthropic_claude_haiku_4_5"].llm_config == {
        "_target_": "card_framework.providers.anthropic_provider.AnthropicProvider",
        "api_key": "$ENV{ANTHROPIC_API_KEY}",
        "model": "claude-haiku-4-5",
        "enable_thinking": False,
    }
    assert profiles["deepseek_chat"].llm_config["model"] == "deepseek-chat"
    assert profiles["google_gemini_3_1_pro_preview_judge"].llm_config == {
        "_target_": "card_framework.providers.google_genai_provider.GoogleGenAIProvider",
        "api_key": "$ENV{GEMINI_API_KEY}",
        "model": "gemini-3.1-pro-preview",
        "enable_thinking": True,
    }
