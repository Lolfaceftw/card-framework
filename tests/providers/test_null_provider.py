from __future__ import annotations

import sys
from typing import Any

from card_framework.providers.null_provider import NoOpEmbeddingProvider


def _ensure_numpy_zeros_support() -> None:
    """Backfill minimal zeros support when another test stubs numpy."""
    numpy_module = sys.modules.get("numpy")
    if numpy_module is None or hasattr(numpy_module, "zeros"):
        return

    class _Array:
        def __init__(self, shape: tuple[int, int]) -> None:
            rows, cols = shape
            self.shape = shape
            self._values = [[0.0 for _ in range(cols)] for _ in range(rows)]

        def tolist(self) -> list[list[float]]:
            return self._values

    def _zeros(shape: tuple[int, int]) -> Any:
        return _Array(shape)

    setattr(numpy_module, "zeros", _zeros)


def _matrix_values(embeddings: Any) -> list[float]:
    if hasattr(embeddings, "flatten"):
        return [float(value) for value in embeddings.flatten()]
    if hasattr(embeddings, "tolist"):
        nested = embeddings.tolist()
    else:
        nested = embeddings
    return [float(value) for row in nested for value in row]


def test_no_op_embedding_provider_returns_zero_embeddings() -> None:
    _ensure_numpy_zeros_support()
    provider = NoOpEmbeddingProvider(unused="ok")

    embeddings = provider.encode(
        ["alpha", "beta", "gamma"],
        normalize=False,
        show_progress=True,
        prompt_name="ignored",
    )

    assert embeddings.shape == (3, 1)
    assert all(value == 0.0 for value in _matrix_values(embeddings))


def test_no_op_embedding_provider_handles_empty_inputs() -> None:
    _ensure_numpy_zeros_support()
    provider = NoOpEmbeddingProvider()

    embeddings = provider.encode([])

    assert embeddings.shape == (0, 1)
