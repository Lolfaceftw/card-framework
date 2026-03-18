"""Shared retrieval registry coverage."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from card_framework.retrieval.embeddings import NamespacedTranscriptIndexRegistry
from card_framework.shared.llm_provider import EmbeddingProvider


class _FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding stub that also tracks concurrent encode calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_calls = 0
        self.max_active_calls = 0
        self.closed = False

    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
        show_progress: bool = False,
        prompt_name: str | None = None,
    ) -> np.ndarray:
        del show_progress, prompt_name
        with self._lock:
            self._active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self._active_calls)
        try:
            time.sleep(0.05)
            vectors = []
            for text in texts:
                lower = text.lower()
                if "apple" in lower:
                    vector = np.array([1.0, 0.0], dtype=np.float32)
                elif "banana" in lower:
                    vector = np.array([0.0, 1.0], dtype=np.float32)
                else:
                    vector = np.array([0.5, 0.5], dtype=np.float32)
                vectors.append(vector)
            encoded = np.stack(vectors)
            if normalize:
                norms = np.linalg.norm(encoded, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                encoded = encoded / norms
            return encoded
        finally:
            with self._lock:
                self._active_calls -= 1

    def close(self) -> None:
        self.closed = True


def test_namespaced_registry_isolates_worker_indices() -> None:
    provider = _FakeEmbeddingProvider()
    registry = NamespacedTranscriptIndexRegistry(provider)

    registry.index_segments(
        [{"speaker": "SPEAKER_00", "text": "apple transcript"}],
        namespace="worker-a",
    )
    registry.index_segments(
        [{"speaker": "SPEAKER_01", "text": "banana transcript"}],
        namespace="worker-b",
    )

    worker_a_segments = registry.retrieve_mmr("apple", namespace="worker-a", top_k=1)
    worker_b_segments = registry.retrieve_mmr("banana", namespace="worker-b", top_k=1)

    assert worker_a_segments[0]["text"] == "apple transcript"
    assert worker_b_segments[0]["text"] == "banana transcript"


def test_namespaced_registry_serializes_concurrent_embedding_requests() -> None:
    provider = _FakeEmbeddingProvider()
    registry = NamespacedTranscriptIndexRegistry(provider, serialize_requests=True)

    threads = [
        threading.Thread(
            target=registry.index_segments,
            kwargs={
                "segments": [{"speaker": "SPEAKER_00", "text": "apple transcript"}],
                "namespace": "worker-a",
            },
        ),
        threading.Thread(
            target=registry.index_segments,
            kwargs={
                "segments": [{"speaker": "SPEAKER_01", "text": "banana transcript"}],
                "namespace": "worker-b",
            },
        ),
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert provider.max_active_calls == 1


def test_namespaced_registry_clear_namespace_drops_cached_index() -> None:
    provider = _FakeEmbeddingProvider()
    registry = NamespacedTranscriptIndexRegistry(provider)
    registry.index_segments(
        [{"speaker": "SPEAKER_00", "text": "apple transcript"}],
        namespace="worker-a",
    )

    registry.clear_namespace("worker-a")

    with pytest.raises(RuntimeError, match="No segments indexed for retrieval namespace"):
        registry.retrieve_mmr("apple", namespace="worker-a", top_k=1)


def test_namespaced_registry_close_releases_embedding_provider() -> None:
    provider = _FakeEmbeddingProvider()
    registry = NamespacedTranscriptIndexRegistry(provider)

    registry.close()

    assert provider.closed is True
