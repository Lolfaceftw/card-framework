"""Expose provider implementations without eager importing heavy backends."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_PROVIDER_EXPORTS = {
    "DeepSeekProvider": "card_framework.providers.deepseek_provider",
    "GLMProvider": "card_framework.providers.glm_provider",
    "HuggingfaceProvider": "card_framework.providers.huggingface_provider",
    "NanbeigeProvider": "card_framework.providers.nanbeige_provider",
    "SentenceTransformerEmbeddingProvider": "card_framework.providers.sentence_transformer_provider",
    "TransformersProvider": "card_framework.providers.transformers_provider",
    "VLLMProvider": "card_framework.providers.vllm_provider",
}

__all__ = list(_PROVIDER_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load provider symbols on first access to avoid cold-start fan-out."""
    module_path = _PROVIDER_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return module attributes plus lazily exposed provider symbols."""
    return sorted(set(globals()) | set(__all__))
