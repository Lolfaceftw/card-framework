"""AlignScore-style reference-free scorer with robust fallback backends."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from benchmark.reference_free.contracts import AlignScoreEvaluation, clamp_score


@dataclass(slots=True, frozen=True)
class AlignScoreRunnerConfig:
    """Configuration for alignscore-style evaluation.

    Args:
        model_name: SentenceTransformer model used in semantic fallback backend.
        source_chunk_words: Word chunk size for long source segmentation.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    source_chunk_words: int = 256


class AlignScoreRunner:
    """Compute reference-free alignment score between source and summary.

    Backend preference order:
    1) Native AlignScore package if available.
    2) SentenceTransformer semantic proxy.
    """

    def __init__(self, config: AlignScoreRunnerConfig | None = None) -> None:
        self._config = config or AlignScoreRunnerConfig()
        self._align_model: Any | None = None
        self._sentence_model: Any | None = None
        self._native_error: str | None = None

        self._init_native_alignscore_backend()

    def _init_native_alignscore_backend(self) -> None:
        """Attempt to initialize native AlignScore backend lazily."""
        try:
            from alignscore import AlignScore  # type: ignore

            self._align_model = AlignScore(
                model="roberta-base",
                batch_size=8,
                device="cpu",
            )
        except Exception as exc:  # pragma: no cover - optional dependency path
            self._native_error = str(exc)
            self._align_model = None

    def _init_sentence_backend(self) -> None:
        """Initialize sentence-transformer fallback backend on first use."""
        if self._sentence_model is not None:
            return

        from sentence_transformers import SentenceTransformer

        self._sentence_model = SentenceTransformer(self._config.model_name)

    @staticmethod
    def _chunk_source_text(source_text: str, chunk_words: int) -> list[str]:
        """Split source text into word chunks to avoid long-context collapse."""
        words = source_text.split()
        if not words:
            return []

        chunk_words = max(32, chunk_words)
        return [
            " ".join(words[index : index + chunk_words])
            for index in range(0, len(words), chunk_words)
        ]

    def _score_with_native_alignscore(
        self,
        source_text: str,
        summary_text: str,
    ) -> AlignScoreEvaluation:
        """Compute score using native AlignScore package."""
        assert self._align_model is not None
        try:
            scores = self._align_model.score(
                contexts=[source_text],
                claims=[summary_text],
            )
            if isinstance(scores, list) and scores:
                score = clamp_score(float(scores[0]))
            else:
                score = None
            if score is None:
                return AlignScoreEvaluation(
                    status="error",
                    score=None,
                    backend="alignscore_native",
                    error_message="Native AlignScore returned no score",
                )
            return AlignScoreEvaluation(
                status="ok",
                score=score,
                backend="alignscore_native",
            )
        except Exception as exc:
            return AlignScoreEvaluation(
                status="error",
                score=None,
                backend="alignscore_native",
                error_message=f"Native AlignScore scoring failed: {exc}",
            )

    def _score_with_sentence_transformer(
        self,
        source_text: str,
        summary_text: str,
    ) -> AlignScoreEvaluation:
        """Compute semantic proxy score using sentence-transformer similarity."""
        try:
            self._init_sentence_backend()
        except Exception as exc:
            return AlignScoreEvaluation(
                status="error",
                score=None,
                backend="sentence_transformer_proxy",
                error_message=f"Failed to load sentence-transformer backend: {exc}",
            )

        assert self._sentence_model is not None

        source_chunks = self._chunk_source_text(
            source_text,
            self._config.source_chunk_words,
        )
        if not source_chunks:
            return AlignScoreEvaluation(
                status="error",
                score=None,
                backend="sentence_transformer_proxy",
                error_message="Source text is empty after chunking",
            )

        embeddings = self._sentence_model.encode(
            source_chunks + [summary_text],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if len(embeddings) < 2:
            return AlignScoreEvaluation(
                status="error",
                score=None,
                backend="sentence_transformer_proxy",
                error_message="Sentence-transformer returned insufficient embeddings",
            )

        chunk_embeddings = [list(vector) for vector in embeddings[:-1]]
        summary_embedding = list(embeddings[-1])

        def _dot(left: list[float], right: list[float]) -> float:
            return sum(float(a) * float(b) for a, b in zip(left, right))

        similarities = [_dot(chunk, summary_embedding) for chunk in chunk_embeddings]
        if not similarities or math.isnan(float(max(similarities))):
            return AlignScoreEvaluation(
                status="error",
                score=None,
                backend="sentence_transformer_proxy",
                error_message="Similarity computation returned invalid values",
            )

        max_similarity = float(max(similarities))
        normalized_score = clamp_score((max_similarity + 1.0) / 2.0)

        return AlignScoreEvaluation(
            status="ok",
            score=normalized_score,
            backend="sentence_transformer_proxy",
        )

    def score(self, source_text: str, summary_text: str) -> AlignScoreEvaluation:
        """Score one source-summary pair.

        Args:
            source_text: Full source transcript text.
            summary_text: Candidate summary.

        Returns:
            Alignscore evaluation result with status and backend metadata.
        """
        if not source_text.strip() or not summary_text.strip():
            return AlignScoreEvaluation(
                status="error",
                score=None,
                backend="none",
                error_message="Source and summary must both be non-empty",
            )

        if self._align_model is not None:
            native = self._score_with_native_alignscore(source_text, summary_text)
            if native.status == "ok":
                return native

        fallback = self._score_with_sentence_transformer(source_text, summary_text)
        if fallback.status == "ok":
            return fallback

        native_error = self._native_error or "native backend unavailable"
        merged_error = f"{native_error}; {fallback.error_message or 'fallback failed'}"
        return AlignScoreEvaluation(
            status="error",
            score=None,
            backend=fallback.backend,
            error_message=merged_error,
        )
