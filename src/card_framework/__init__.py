"""Top-level package for the card-framework repository."""

from __future__ import annotations

from typing import Any

__all__ = ["InferenceResult", "RuntimeBootstrapError", "infer"]


def __getattr__(name: str) -> Any:
    """Resolve public library API exports lazily."""
    if name in {"InferenceResult", "RuntimeBootstrapError", "infer"}:
        from card_framework.api import InferenceResult, RuntimeBootstrapError, infer

        exported = {
            "InferenceResult": InferenceResult,
            "RuntimeBootstrapError": RuntimeBootstrapError,
            "infer": infer,
        }
        return exported[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
