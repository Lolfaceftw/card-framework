"""
Abstract Base Classes for LLM and Embedding Providers
=====================================================
Strategy interfaces that allow swapping LLM and embedding backends
without changing any agent or retrieval logic.
"""

from abc import ABC, abstractmethod

import numpy as np


class LLMProvider(ABC):
    """Strategy interface for text-generation LLMs."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text completion and return the full response string."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
    ):
        """
        Chat completion with support for tools.
        Returns the full response object or a structured representation including tool calls.
        """
        ...


class EmbeddingProvider(ABC):
    """Strategy interface for text-embedding models."""

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
        show_progress: bool = False,
        prompt_name: str | None = None,
    ) -> np.ndarray:
        """Encode a list of texts into an (N, D) embedding matrix."""
        ...
