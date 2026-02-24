"""
Null Object pattern implementation for the Embedding Provider.
"""

import numpy as np

from llm_provider import EmbeddingProvider


class NoOpEmbeddingProvider(EmbeddingProvider):
    """
    A Null Object that disables RAG by returning zero embeddings.
    This allows the application to run the orchestrator loop without
    loading memory-heavy embedding models.
    """

    def __init__(self, **kwargs):
        # Accept any kwargs so hydra doesn't crash if extra parameters are provided
        pass

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
        show_progress: bool = False,
        prompt_name: str | None = None,
    ) -> np.ndarray:
        """
        Return an array of zeros matching the number of texts.
        The dimension (1) is arbitrary, as mmr math operates cleanly on any matching size.
        """
        return np.zeros((len(texts), 1))
