"""
Transcript Embedding & Retrieval
================================
Indexes transcript segments and supports MMR (Maximal Marginal Relevance)
retrieval for diverse, relevant selection.

The actual embedding model is injected via the EmbeddingProvider strategy
interface, making it trivial to swap backends.
"""

import hashlib
import json
from pathlib import Path
import re
import threading
import uuid

import numpy as np

from card_framework.shared.llm_provider import EmbeddingProvider
from card_framework.shared.events import event_bus
from card_framework.shared.perf import report_loading_duration


def _normalize_segments(segments: list[dict]) -> list[dict[str, str]]:
    """Normalize retrieval segments into a stable cacheable payload."""
    return [
        {
            "speaker": str(segment.get("speaker", "UNKNOWN")).strip() or "UNKNOWN",
            "text": str(segment.get("text", "")),
        }
        for segment in segments
    ]


def _stable_segments_hash(segments: list[dict[str, str]]) -> str:
    """Hash a normalized retrieval-segment payload deterministically."""
    serialized = json.dumps(
        segments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _embedding_provider_cache_identity(provider: EmbeddingProvider) -> str:
    """Return a stable provider identity string for embedding cache keys."""
    cache_identity = getattr(provider, "cache_identity", None)
    if callable(cache_identity):
        resolved = str(cache_identity()).strip()
        if resolved:
            return resolved
    return f"{provider.__class__.__module__}.{provider.__class__.__qualname__}"


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write one UTF-8 text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _atomic_write_npy(path: Path, array: np.ndarray) -> None:
    """Atomically persist one numpy array without partial writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    with temp_path.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temp_path.replace(path)


class TranscriptIndex:
    """Indexes transcript segments and retrieves relevant ones via MMR."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        cache_dir: Path | None = None,
    ) -> None:
        self.provider = embedding_provider
        self.segments: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._cache_dir = cache_dir.resolve() if cache_dir is not None else None
        self._provider_cache_identity = _embedding_provider_cache_identity(
            embedding_provider
        )

    def _cache_paths(
        self,
        *,
        sample_id: str | None,
        transcript_hash: str,
    ) -> tuple[Path, Path, dict[str, str]] | None:
        if self._cache_dir is None:
            return None
        normalized_sample_id = str(sample_id or "").strip()
        cache_key = hashlib.sha256(
            (
                f"{normalized_sample_id}\n"
                f"{transcript_hash}\n"
                f"{self._provider_cache_identity}"
            ).encode("utf-8")
        ).hexdigest()
        provider_hash = hashlib.sha256(
            self._provider_cache_identity.encode("utf-8")
        ).hexdigest()[:16]
        cache_root = (self._cache_dir / provider_hash / cache_key[:2] / cache_key).resolve()
        metadata = {
            "sample_id": normalized_sample_id,
            "transcript_hash": transcript_hash,
            "provider_identity": self._provider_cache_identity,
            "cache_key": cache_key,
        }
        return (
            cache_root / "metadata.json",
            cache_root / "embeddings.npy",
            metadata,
        )

    def _load_cached_embeddings(
        self,
        *,
        metadata_path: Path,
        embeddings_path: Path,
        expected_metadata: dict[str, str],
        segment_count: int,
    ) -> np.ndarray | None:
        if not metadata_path.exists() or not embeddings_path.exists():
            return None
        try:
            cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(cached_metadata, dict):
            return None
        for key, expected_value in expected_metadata.items():
            if str(cached_metadata.get(key, "")) != expected_value:
                return None
        try:
            embeddings = np.load(embeddings_path, allow_pickle=False)
        except (OSError, ValueError):
            return None
        if embeddings.ndim != 2 or int(embeddings.shape[0]) != int(segment_count):
            return None
        return embeddings

    def _store_cached_embeddings(
        self,
        *,
        metadata_path: Path,
        embeddings_path: Path,
        metadata: dict[str, str],
        embeddings: np.ndarray,
    ) -> None:
        cache_metadata = {
            **metadata,
            "segment_count": int(embeddings.shape[0]) if embeddings.ndim >= 1 else 0,
            "embedding_shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
        }
        try:
            _atomic_write_npy(embeddings_path, embeddings)
            _atomic_write_text(
                metadata_path,
                json.dumps(cache_metadata, indent=2, ensure_ascii=False),
            )
        except OSError:
            return None

    def _restore_from_cache(
        self,
        *,
        sample_id: str | None,
        segments: list[dict[str, str]],
        texts: list[str],
    ) -> bool:
        transcript_hash = _stable_segments_hash(segments)
        cache_paths = self._cache_paths(
            sample_id=sample_id,
            transcript_hash=transcript_hash,
        )
        if cache_paths is None:
            return False
        metadata_path, embeddings_path, metadata = cache_paths
        cached_embeddings = self._load_cached_embeddings(
            metadata_path=metadata_path,
            embeddings_path=embeddings_path,
            expected_metadata=metadata,
            segment_count=len(texts),
        )
        if cached_embeddings is None:
            return False
        self.segments = list(segments)
        self.embeddings = cached_embeddings
        event_bus.publish(
            "system_message",
            (
                f"Reused cached transcript embeddings for {len(texts)} segments "
                f"(sample_id={metadata['sample_id'] or 'unknown'})."
            ),
        )
        return True

    def index_segments(
        self,
        segments: list[dict],
        *,
        sample_id: str | None = None,
    ) -> int:
        """Embed and store all transcript segments. Returns count indexed."""
        normalized_segments = _normalize_segments(segments)
        texts = [segment.get("text", "") for segment in normalized_segments]
        if self._restore_from_cache(
            sample_id=sample_id,
            segments=normalized_segments,
            texts=texts,
        ):
            return len(texts)

        self.segments = list(normalized_segments)

        event_bus.publish("system_message", f"Embedding {len(texts)} segments...")
        with report_loading_duration(
            "transcript embeddings",
            reporter=lambda message: event_bus.publish("system_message", message),
        ):
            self.embeddings = self.provider.encode(
                texts, normalize=True, show_progress=True
            )
        transcript_hash = _stable_segments_hash(normalized_segments)
        cache_paths = self._cache_paths(
            sample_id=sample_id,
            transcript_hash=transcript_hash,
        )
        if cache_paths is not None and self.embeddings is not None:
            metadata_path, embeddings_path, metadata = cache_paths
            self._store_cached_embeddings(
                metadata_path=metadata_path,
                embeddings_path=embeddings_path,
                metadata=metadata,
                embeddings=self.embeddings,
            )
        event_bus.publish("system_message", f"Embedded {len(texts)} segments")
        return len(texts)

    def retrieve_mmr(
        self,
        query: str,
        top_k: int = 15,
        lambda_param: float = 0.7,
    ) -> list[dict]:
        """
        Retrieve top_k segments using Maximal Marginal Relevance.

        MMR balances relevance to the query with diversity among selected segments:
          score = lambda * sim(query, doc) - (1 - lambda) * max(sim(doc, selected))

        Args:
            query: The retrieval query text.
            top_k: Number of segments to return.
            lambda_param: Trade-off (1.0 = pure relevance, 0.0 = pure diversity).

        Returns:
            Selected segments sorted by original start_time.
        """
        if self.embeddings is None or len(self.segments) == 0:
            raise RuntimeError("No segments indexed. Call index_segments() first.")

        # Embed the query with the "query" prompt for better retrieval
        query_emb = self.provider.encode([query], normalize=True, prompt_name="query")

        # Cosine similarities (embeddings are already normalized)
        sim_to_query = (self.embeddings @ query_emb.T).flatten()

        n = len(self.segments)
        top_k = min(top_k, n)

        selected_indices: list[int] = []
        remaining = set(range(n))

        for _ in range(top_k):
            best_idx = -1
            best_score = -float("inf")

            for idx in sorted(remaining):
                relevance = float(sim_to_query[idx])

                # Max similarity to already-selected segments
                if selected_indices:
                    selected_embs = self.embeddings[selected_indices]
                    sim_to_selected = self.embeddings[idx] @ selected_embs.T
                    max_sim = float(np.max(sim_to_selected))
                else:
                    max_sim = 0.0

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            if best_idx >= 0:
                selected_indices.append(best_idx)
                remaining.discard(best_idx)

        # Sort by original chronological order
        selected_indices.sort()
        selected_segments = [self.segments[i] for i in selected_indices]

        total_words = sum(
            len(re.sub(r"<[^>]+>", "", seg.get("text", "")).split())
            for seg in selected_segments
        )
        event_bus.publish("system_message", 
            f"MMR selected {len(selected_segments)} segments "
            f"({total_words} words) from {n} total"
        )
        return selected_segments

    def clear(self) -> None:
        """Release the currently indexed transcript payload."""
        self.segments = []
        self.embeddings = None


class NamespacedTranscriptIndexRegistry:
    """Manage multiple transcript indices over one shared embedding provider."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        *,
        serialize_requests: bool = False,
        cache_dir: Path | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._indices: dict[str, TranscriptIndex] = {}
        self._cache_dir = cache_dir.resolve() if cache_dir is not None else None
        self._request_lock: threading.Lock | None = (
            threading.Lock() if serialize_requests else None
        )

    @staticmethod
    def _normalize_namespace(namespace: str | None) -> str:
        normalized = str(namespace or "").strip()
        return normalized or "default"

    def index_segments(
        self,
        segments: list[dict],
        *,
        namespace: str | None = None,
        sample_id: str | None = None,
    ) -> int:
        """Embed and store transcript segments for one namespace."""
        normalized_namespace = self._normalize_namespace(namespace)
        if self._request_lock is not None:
            with self._request_lock:
                return self._index_segments_locked(
                    segments,
                    normalized_namespace,
                    sample_id=sample_id,
                )
        return self._index_segments_locked(
            segments,
            normalized_namespace,
            sample_id=sample_id,
        )

    def _index_segments_locked(
        self,
        segments: list[dict],
        namespace: str,
        *,
        sample_id: str | None = None,
    ) -> int:
        self._indices.pop(namespace, None)
        index = TranscriptIndex(
            self._embedding_provider,
            cache_dir=getattr(self, "_cache_dir", None),
        )
        count = index.index_segments(segments, sample_id=sample_id)
        self._indices[namespace] = index
        return count

    def retrieve_mmr(
        self,
        query: str,
        *,
        namespace: str | None = None,
        top_k: int = 15,
        lambda_param: float = 0.7,
    ) -> list[dict]:
        """Retrieve segments from one namespace's transcript index."""
        normalized_namespace = self._normalize_namespace(namespace)
        if self._request_lock is not None:
            with self._request_lock:
                return self._retrieve_mmr_locked(
                    query=query,
                    namespace=normalized_namespace,
                    top_k=top_k,
                    lambda_param=lambda_param,
                )
        return self._retrieve_mmr_locked(
            query=query,
            namespace=normalized_namespace,
            top_k=top_k,
            lambda_param=lambda_param,
        )

    def _retrieve_mmr_locked(
        self,
        *,
        query: str,
        namespace: str,
        top_k: int,
        lambda_param: float,
    ) -> list[dict]:
        index = self._indices.get(namespace)
        if index is None:
            raise RuntimeError(
                f"No segments indexed for retrieval namespace '{namespace}'."
            )
        return index.retrieve_mmr(
            query=query,
            top_k=top_k,
            lambda_param=lambda_param,
        )

    def clear_namespace(self, namespace: str | None = None) -> None:
        """Drop one namespace's retained transcript index."""
        normalized_namespace = self._normalize_namespace(namespace)
        if self._request_lock is not None:
            with self._request_lock:
                self._clear_namespace_locked(normalized_namespace)
            return
        self._clear_namespace_locked(normalized_namespace)

    def _clear_namespace_locked(self, namespace: str) -> None:
        self._indices.pop(namespace, None)

    def close(self) -> None:
        """Release all cached indices and the shared embedding provider."""
        if self._request_lock is not None:
            with self._request_lock:
                self._indices.clear()
                self._embedding_provider.close()
            return
        self._indices.clear()
        self._embedding_provider.close()

