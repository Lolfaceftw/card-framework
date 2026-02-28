"""Compare transcript retrieval with and without MMR using a Qwen embedding model.

This script embeds transcript chunks once, then runs:
1) Pure top-k retrieval by cosine similarity (no MMR).
2) Maximal Marginal Relevance (MMR) retrieval.

It prints both chunk lists and a simple overlap/difference summary.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from providers.sentence_transformer_provider import SentenceTransformerEmbeddingProvider


@dataclass(slots=True, frozen=True)
class Segment:
    """Normalized transcript chunk used for retrieval comparison."""

    index: int
    speaker: str
    start_time: int | None
    end_time: int | None
    text: str


@dataclass(slots=True, frozen=True)
class RankedChunk:
    """One ranked retrieval result with optional MMR diagnostics."""

    rank: int
    segment: Segment
    similarity: float
    mmr_score: float | None = None
    redundancy: float | None = None


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compare Qwen RAG retrieval with vs without MMR on a transcript.",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        required=True,
        help="Path to transcript JSON containing a 'segments' list.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Retrieval query. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of chunks to retrieve for each method.",
    )
    parser.add_argument(
        "--lambda-param",
        type=float,
        default=0.7,
        help="MMR lambda (1.0=relevance only, 0.0=diversity only).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-Embedding-4B",
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Embedding device (e.g., cuda or cpu).",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="float16",
        help="Embedding torch dtype (e.g., float16, float32, bfloat16).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=220,
        help="Max characters printed for each chunk preview.",
    )
    return parser.parse_args()


def _load_segments(transcript_path: Path) -> list[Segment]:
    """Load and normalize transcript segments from JSON."""
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    payload = json.loads(transcript_path.read_text(encoding="utf-8-sig"))
    raw_segments = payload.get("segments", [])
    if not isinstance(raw_segments, list):
        raise ValueError("Transcript JSON must contain a list field: 'segments'.")

    normalized: list[Segment] = []
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue

        speaker = str(raw.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
        start_time = _parse_optional_int(raw.get("start_time"))
        end_time = _parse_optional_int(raw.get("end_time"))
        normalized.append(
            Segment(
                index=index,
                speaker=speaker,
                start_time=start_time,
                end_time=end_time,
                text=text,
            )
        )

    if not normalized:
        raise ValueError("Transcript contains no non-empty segments.")
    return normalized


def _parse_optional_int(value: Any) -> int | None:
    """Parse integer-like values for transcript timestamps."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str) and value.strip():
        try:
            return int(round(float(value.strip())))
        except ValueError:
            return None
    return None


def _pure_top_k(
    similarities: np.ndarray,
    segments: list[Segment],
    *,
    top_k: int,
) -> list[RankedChunk]:
    """Return pure relevance top-k ranking (no MMR)."""
    k = max(1, min(top_k, len(segments)))
    ranked_indices = list(np.argsort(similarities)[::-1][:k])
    return [
        RankedChunk(
            rank=rank + 1,
            segment=segments[idx],
            similarity=float(similarities[idx]),
        )
        for rank, idx in enumerate(ranked_indices)
    ]


def _mmr_top_k(
    embeddings: np.ndarray,
    similarities: np.ndarray,
    segments: list[Segment],
    *,
    top_k: int,
    lambda_param: float,
) -> list[RankedChunk]:
    """Return MMR ranking with relevance/redundancy diagnostics."""
    k = max(1, min(top_k, len(segments)))
    lam = min(1.0, max(0.0, lambda_param))

    selected: list[int] = []
    remaining = set(range(len(segments)))
    results: list[RankedChunk] = []

    while remaining and len(selected) < k:
        best_idx = -1
        best_score = -float("inf")
        best_redundancy = 0.0

        for idx in sorted(remaining):
            relevance = float(similarities[idx])
            if selected:
                selected_embs = embeddings[selected]
                max_redundancy = float(np.max(embeddings[idx] @ selected_embs.T))
            else:
                max_redundancy = 0.0

            mmr_score = lam * relevance - (1.0 - lam) * max_redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx
                best_redundancy = max_redundancy

        if best_idx < 0:
            break

        selected.append(best_idx)
        remaining.remove(best_idx)
        results.append(
            RankedChunk(
                rank=len(results) + 1,
                segment=segments[best_idx],
                similarity=float(similarities[best_idx]),
                mmr_score=float(best_score),
                redundancy=float(best_redundancy),
            )
        )

    return results


def _format_time_range(segment: Segment) -> str:
    """Format segment start/end milliseconds for display."""
    start = "?"
    end = "?"
    if segment.start_time is not None:
        start = str(segment.start_time)
    if segment.end_time is not None:
        end = str(segment.end_time)
    return f"{start}-{end}ms"


def _preview_text(text: str, *, max_chars: int) -> str:
    """Create single-line chunk preview text."""
    collapsed = " ".join(text.split())
    if max_chars <= 3 or len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3]}..."


def _print_ranked_chunks(
    title: str,
    ranked: list[RankedChunk],
    *,
    max_chars: int,
) -> None:
    """Print ranked retrieval chunks."""
    print(f"\n=== {title} ===")
    for row in ranked:
        segment = row.segment
        header = (
            f"[{row.rank:02d}] idx={segment.index} "
            f"sim={row.similarity:.4f} "
            f"speaker={segment.speaker} "
            f"time={_format_time_range(segment)}"
        )
        if row.mmr_score is not None and row.redundancy is not None:
            header += f" mmr={row.mmr_score:.4f} redundancy={row.redundancy:.4f}"
        print(header)
        print(f"     {_preview_text(segment.text, max_chars=max_chars)}")


def _print_overlap_summary(
    pure_top_k: list[RankedChunk],
    mmr_top_k: list[RankedChunk],
) -> None:
    """Print overlap and differences between both retrieval methods."""
    pure_indices = [row.segment.index for row in pure_top_k]
    mmr_indices = [row.segment.index for row in mmr_top_k]
    pure_set = set(pure_indices)
    mmr_set = set(mmr_indices)
    overlap = pure_set & mmr_set
    only_pure = pure_set - mmr_set
    only_mmr = mmr_set - pure_set

    print("\n=== Comparison Summary ===")
    print(f"Pure top-k indices: {pure_indices}")
    print(f"MMR indices      : {mmr_indices}")
    print(f"Overlap ({len(overlap)}): {sorted(overlap)}")
    print(f"Only pure ({len(only_pure)}): {sorted(only_pure)}")
    print(f"Only MMR  ({len(only_mmr)}): {sorted(only_mmr)}")


def main() -> None:
    """Run retrieval comparison for one transcript/query pair."""
    args = _parse_args()
    query = args.query.strip()
    if not query:
        query = input("Enter retrieval query for this transcript: ").strip()
    if not query:
        raise ValueError("A non-empty query is required.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero.")

    segments = _load_segments(args.transcript)
    texts = [segment.text for segment in segments]

    print(f"Transcript: {args.transcript}")
    print(f"Segments  : {len(segments)}")
    print(f"Query     : {query}")
    print(
        f"Model     : {args.model_name} (device={args.device}, "
        f"dtype={args.torch_dtype}, batch_size={args.batch_size})"
    )

    provider = SentenceTransformerEmbeddingProvider(
        model_name=args.model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
        batch_size=args.batch_size,
    )

    print("\nEmbedding transcript segments...")
    embeddings = provider.encode(texts, normalize=True, show_progress=True)

    print("Embedding query...")
    query_embedding = provider.encode([query], normalize=True, prompt_name="query")
    similarities = (embeddings @ query_embedding.T).flatten()

    pure_results = _pure_top_k(similarities, segments, top_k=args.top_k)
    mmr_results = _mmr_top_k(
        embeddings,
        similarities,
        segments,
        top_k=args.top_k,
        lambda_param=args.lambda_param,
    )

    _print_ranked_chunks("WITHOUT MMR (Pure Similarity Top-K)", pure_results, max_chars=args.max_chars)
    _print_ranked_chunks(
        f"WITH MMR (lambda={min(1.0, max(0.0, args.lambda_param)):.2f})",
        mmr_results,
        max_chars=args.max_chars,
    )
    _print_overlap_summary(pure_results, mmr_results)


if __name__ == "__main__":
    main()
