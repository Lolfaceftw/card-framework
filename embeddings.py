"""
Transcript Embedding & Retrieval
================================
Indexes transcript segments and supports MMR (Maximal Marginal Relevance)
retrieval for diverse, relevant selection.

The actual embedding model is injected via the EmbeddingProvider strategy
interface, making it trivial to swap backends.
"""

import re

import numpy as np

from llm_provider import EmbeddingProvider
from events import event_bus


class TranscriptIndex:
    """Indexes transcript segments and retrieves relevant ones via MMR."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self.provider = embedding_provider
        self.segments: list[dict] = []
        self.embeddings: np.ndarray | None = None

    def index_segments(self, segments: list[dict]) -> int:
        """Embed and store all transcript segments. Returns count indexed."""
        self.segments = segments
        texts = [seg.get("text", "") for seg in segments]

        event_bus.publish("system_message", f"Embedding {len(texts)} segments...")
        self.embeddings = self.provider.encode(
            texts, normalize=True, show_progress=True
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
