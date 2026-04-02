"""Tests for runtime config normalization and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from card_framework.shared.config_loading import (
    RuntimeConfigError,
    normalize_runtime_config_mapping,
    validate_runtime_credentials,
)


def test_normalize_runtime_config_mapping_materializes_selected_profiles() -> None:
    """Materialize explicit runtime sections from selection and profiles."""

    normalized = normalize_runtime_config_mapping(
        {
            "selection": {
                "llm": "vllm_local",
                "embedding": "disabled",
                "stage_llm": {"critic": "deepseek_reasoner"},
            },
            "profiles": {
                "llm": {
                    "vllm_local": {
                        "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
                        "base_url": "http://localhost:8000/v1",
                        "api_key": "EMPTY",
                    },
                    "deepseek_reasoner": {
                        "_target_": "card_framework.providers.deepseek_provider.DeepSeekProvider",
                        "api_key": "",
                        "model": "deepseek-reasoner",
                        "base_url": "https://api.deepseek.com",
                    },
                },
                "embedding": {
                    "disabled": {
                        "_target_": "card_framework.providers.null_provider.NoOpEmbeddingProvider"
                    }
                },
            },
        }
    )

    assert normalized["llm"]["_target_"] == (
        "card_framework.providers.vllm_provider.VLLMProvider"
    )
    assert normalized["embedding"]["_target_"] == (
        "card_framework.providers.null_provider.NoOpEmbeddingProvider"
    )
    assert normalized["stage_llm"]["summarizer"] == {}
    assert normalized["stage_llm"]["interjector"] == {}
    assert normalized["stage_llm"]["critic"]["model"] == "deepseek-reasoner"


def test_normalize_runtime_config_mapping_preserves_explicit_runtime_sections() -> None:
    """Prefer explicit runtime sections over selector-derived values."""

    normalized = normalize_runtime_config_mapping(
        {
            "selection": {
                "llm": "vllm_local",
                "embedding": "disabled",
            },
            "profiles": {
                "llm": {
                    "vllm_local": {
                        "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
                        "base_url": "http://localhost:8000/v1",
                        "api_key": "EMPTY",
                    }
                },
                "embedding": {
                    "disabled": {
                        "_target_": "card_framework.providers.null_provider.NoOpEmbeddingProvider"
                    }
                },
            },
            "llm": {
                "_target_": "card_framework.providers.deepseek_provider.DeepSeekProvider",
                "api_key": "from_override",
            },
            "stage_llm": {"critic": {}},
        }
    )

    assert normalized["llm"]["_target_"] == (
        "card_framework.providers.deepseek_provider.DeepSeekProvider"
    )
    assert normalized["llm"]["api_key"] == "from_override"
    assert normalized["stage_llm"]["critic"] == {}
    assert normalized["stage_llm"]["summarizer"] == {}
    assert normalized["stage_llm"]["interjector"] == {}


def test_validate_runtime_credentials_reports_missing_selected_provider_secret() -> (
    None
):
    """Raise an actionable error when the selected provider secret is absent."""

    with pytest.raises(RuntimeConfigError, match="DEEPSEEK_API_KEY"):
        validate_runtime_credentials(
            {
                "selection": {"llm": "deepseek_chat", "embedding": "disabled"},
                "profiles": {
                    "llm": {
                        "deepseek_chat": {
                            "_target_": "card_framework.providers.deepseek_provider.DeepSeekProvider",
                            "api_key": "",
                            "model": "deepseek-chat",
                            "base_url": "https://api.deepseek.com",
                        }
                    },
                    "embedding": {
                        "disabled": {
                            "_target_": "card_framework.providers.null_provider.NoOpEmbeddingProvider"
                        }
                    },
                },
            }
        )


def test_validate_runtime_credentials_reports_missing_anthropic_secret() -> None:
    """Anthropic profile selections should fail fast without an API key."""

    with pytest.raises(RuntimeConfigError, match="ANTHROPIC_API_KEY"):
        validate_runtime_credentials(
            {
                "selection": {"llm": "anthropic_claude_opus", "embedding": "disabled"},
                "profiles": {
                    "llm": {
                        "anthropic_claude_opus": {
                            "_target_": "card_framework.providers.anthropic_provider.AnthropicProvider",
                            "api_key": "",
                            "model": "claude-opus-4-6",
                        }
                    },
                    "embedding": {
                        "disabled": {
                            "_target_": "card_framework.providers.null_provider.NoOpEmbeddingProvider"
                        }
                    },
                },
            }
        )


def test_validate_runtime_credentials_reports_missing_openai_secret() -> None:
    """OpenAI profile selections should fail fast without an API key."""

    with pytest.raises(RuntimeConfigError, match="OPENAI_API_KEY"):
        validate_runtime_credentials(
            {
                "selection": {"llm": "openai_gpt_5_4_mini", "embedding": "disabled"},
                "profiles": {
                    "llm": {
                        "openai_gpt_5_4_mini": {
                            "_target_": "card_framework.providers.openai_provider.OpenAIProvider",
                            "api_key": "",
                            "model": "gpt-5.4-mini",
                        }
                    },
                    "embedding": {
                        "disabled": {
                            "_target_": "card_framework.providers.null_provider.NoOpEmbeddingProvider"
                        }
                    },
                },
            }
        )


def test_packaged_qwen_embedding_profile_uses_tuned_defaults() -> None:
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "src"
        / "card_framework"
        / "config"
        / "config.yaml"
    )

    qwen_profile = cfg.profiles.embedding.qwen3_4b

    assert qwen_profile.batch_size == 1
    assert qwen_profile.max_seq_length == 384
    assert qwen_profile.attention_implementation == "eager"


@pytest.mark.parametrize(
    "config_path",
    [
        Path(__file__).resolve().parents[2] / "conf" / "config.yaml",
        Path(__file__).resolve().parents[2]
        / "src"
        / "card_framework"
        / "config"
        / "config.yaml",
    ],
)
def test_summary_turn_linking_defaults_do_not_force_live_draft_rerender(
    config_path: Path,
) -> None:
    cfg = OmegaConf.load(config_path)

    assert cfg.audio.summary_turn_linking.force_live_draft_rerender is False
