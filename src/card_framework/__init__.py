"""Top-level package for the card-framework repository."""

from __future__ import annotations

from typing import Any

from card_framework.shared.env import load_repo_dotenv

load_repo_dotenv(search_cwd=True)

__all__ = [
    "InferenceResult",
    "NotReadyError",
    "RuntimeBootstrapError",
    "infer",
    "llm_profiles",
]


def __getattr__(name: str) -> Any:
    """Resolve public library API exports lazily."""
    if name in __all__:
        from card_framework import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
