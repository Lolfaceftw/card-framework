"""Shared retrieval registry coverage."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from card_framework.retrieval.embeddings import (
    NamespacedTranscriptIndexRegistry,
    TranscriptIndex,
)
from card_framework.shared.llm_provider import EmbeddingProvider


class _FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding stub that also tracks concurrent encode calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active_calls = 0
        self.max_active_calls = 0
        self.closed = False
        self.encode_calls = 0

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
            self.encode_calls += 1
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

    def cache_identity(self) -> str:
        return "fake-embedding-provider-v1"


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


def test_transcript_index_reuses_cached_embeddings_across_instances(
    tmp_path,
) -> None:
    segments = [{"speaker": "SPEAKER_00", "text": "apple transcript"}]
    first_provider = _FakeEmbeddingProvider()
    first_index = TranscriptIndex(
        embedding_provider=first_provider,
        cache_dir=tmp_path / "embedding_cache",
    )

    assert first_index.index_segments(segments, sample_id="sample-a") == 1
    assert first_provider.encode_calls == 1

    second_provider = _FakeEmbeddingProvider()
    second_index = TranscriptIndex(
        embedding_provider=second_provider,
        cache_dir=tmp_path / "embedding_cache",
    )

    assert second_index.index_segments(segments, sample_id="sample-a") == 1
    assert second_provider.encode_calls == 0
    retrieved = second_index.retrieve_mmr("apple", top_k=1)
    assert retrieved[0]["text"] == "apple transcript"


def test_transcript_index_cache_misses_when_transcript_hash_changes(
    tmp_path,
) -> None:
    provider = _FakeEmbeddingProvider()
    index = TranscriptIndex(
        embedding_provider=provider,
        cache_dir=tmp_path / "embedding_cache",
    )

    index.index_segments(
        [{"speaker": "SPEAKER_00", "text": "apple transcript"}],
        sample_id="sample-a",
    )
    index.index_segments(
        [{"speaker": "SPEAKER_00", "text": "banana transcript"}],
        sample_id="sample-a",
    )

    assert provider.encode_calls == 2


def test_transcript_index_cache_reuses_embeddings_when_affect_hint_changes(
    tmp_path,
) -> None:
    first_provider = _FakeEmbeddingProvider()
    first_index = TranscriptIndex(
        embedding_provider=first_provider,
        cache_dir=tmp_path / "embedding_cache",
    )
    second_provider = _FakeEmbeddingProvider()
    second_index = TranscriptIndex(
        embedding_provider=second_provider,
        cache_dir=tmp_path / "embedding_cache",
    )

    first_index.index_segments(
        [
            {
                "speaker": "SPEAKER_00",
                "text": "apple transcript",
                "affect_hint": "preset=neutral",
            }
        ],
        sample_id="sample-a",
    )
    second_index.index_segments(
        [
            {
                "speaker": "SPEAKER_00",
                "text": "apple transcript",
                "affect_hint": "preset=urgent",
            }
        ],
        sample_id="sample-a",
    )

    assert first_provider.encode_calls == 1
    assert second_provider.encode_calls == 0


def test_transcript_index_preserves_emotion_vector_metadata() -> None:
    provider = _FakeEmbeddingProvider()
    index = TranscriptIndex(embedding_provider=provider)

    index.index_segments(
        [
            {
                "speaker": "SPEAKER_00",
                "text": "apple transcript",
                "emotion_vector": [0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24],
                "emotion_vector_source": "affect_analysis",
            }
        ]
    )

    assert index.segments[0]["emotion_vector"] == [
        0.14,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.02,
        0.24,
    ]
    assert index.segments[0]["emotion_vector_source"] == "affect_analysis"
