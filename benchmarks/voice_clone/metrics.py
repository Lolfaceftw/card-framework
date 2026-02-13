"""Objective scoring helpers for voice cloning benchmarks."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from benchmarks.voice_clone.types import ManifestItem, PairScoreRow


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute equal error rate from similarity scores and binary labels.

    Args:
        scores: Similarity scores where larger means closer match.
        labels: Binary labels where True means same speaker.

    Returns:
        EER fraction in ``[0, 1]`` or ``None`` if undefined.
    """
    if scores.size == 0 or labels.size == 0:
        return None
    positive_total = int(np.sum(labels.astype(np.int64)))
    negative_total = int(labels.size - positive_total)
    if positive_total == 0 or negative_total == 0:
        return None

    order = np.argsort(scores)[::-1]
    ordered_scores = scores[order]
    ordered_labels = labels[order].astype(bool)

    false_positive = 0
    false_negative = positive_total
    min_gap = float("inf")
    best_fpr = 0.0
    best_fnr = 1.0
    index = 0
    while index < ordered_scores.size:
        threshold = ordered_scores[index]
        while index < ordered_scores.size and ordered_scores[index] == threshold:
            if ordered_labels[index]:
                false_negative -= 1
            else:
                false_positive += 1
            index += 1
        fpr = false_positive / negative_total
        fnr = false_negative / positive_total
        gap = abs(fpr - fnr)
        if gap < min_gap:
            min_gap = gap
            best_fpr = fpr
            best_fnr = fnr
    return float((best_fpr + best_fnr) / 2.0)


def bootstrap_ci95_mean(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    sample_count: int,
) -> tuple[float, float] | None:
    """Estimate 95% confidence interval of sample mean with bootstrap."""
    if values.size == 0:
        return None
    means = np.empty(sample_count, dtype=np.float64)
    for idx in range(sample_count):
        sampled = rng.choice(values, size=values.size, replace=True)
        means[idx] = float(np.mean(sampled))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def bootstrap_ci95_eer(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    rng: np.random.Generator,
    sample_count: int,
) -> tuple[float, float] | None:
    """Estimate 95% confidence interval of EER with bootstrap."""
    if scores.size == 0:
        return None
    sampled_eers: list[float] = []
    for _ in range(sample_count):
        sampled_index = rng.integers(0, scores.size, size=scores.size)
        sampled_eer = compute_eer(scores[sampled_index], labels[sampled_index])
        if sampled_eer is not None:
            sampled_eers.append(sampled_eer)
    if not sampled_eers:
        return None
    eer_array = np.array(sampled_eers, dtype=np.float64)
    return float(np.quantile(eer_array, 0.025)), float(np.quantile(eer_array, 0.975))


def compute_reference_centroids(
    *,
    items: list[ManifestItem],
    reference_embeddings: dict[Path, np.ndarray],
) -> dict[str, np.ndarray]:
    """Compute speaker-level centroids from reference embeddings."""
    grouped: dict[str, list[np.ndarray]] = {}
    for item in items:
        grouped.setdefault(item.speaker_id, []).append(reference_embeddings[item.reference_wav])
    centroids: dict[str, np.ndarray] = {}
    for speaker_id, vectors in grouped.items():
        centroids[speaker_id] = np.mean(np.stack(vectors, axis=0), axis=0)
    return centroids


def make_pair_rows(
    *,
    items: list[ManifestItem],
    generated_paths: dict[int, Path],
    generated_embeddings: dict[int, np.ndarray],
    reference_embeddings: dict[Path, np.ndarray],
) -> list[PairScoreRow]:
    """Build per-item objective score rows."""
    centroids = compute_reference_centroids(
        items=items,
        reference_embeddings=reference_embeddings,
    )
    rows: list[PairScoreRow] = []
    for item in items:
        generated_embedding = generated_embeddings[item.item_id]
        reference_embedding = reference_embeddings[item.reference_wav]
        same_item_cosine = cosine_similarity(generated_embedding, reference_embedding)
        centroid_scores = {
            speaker_id: cosine_similarity(generated_embedding, centroid)
            for speaker_id, centroid in centroids.items()
        }
        predicted_speaker = max(centroid_scores, key=centroid_scores.get)
        rows.append(
            PairScoreRow(
                item_id=item.item_id,
                speaker_id=item.speaker_id,
                prompt_wav=item.prompt_wav,
                reference_wav=item.reference_wav,
                generated_wav=generated_paths[item.item_id],
                same_item_cosine=same_item_cosine,
                predicted_speaker=predicted_speaker,
                top1_correct=(predicted_speaker == item.speaker_id),
            )
        )
    return rows


def collect_asv_pairs(
    *,
    items: list[ManifestItem],
    generated_embeddings: dict[int, np.ndarray],
    reference_embeddings: dict[Path, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Build same/different speaker pairs for ASV-EER scoring."""
    reference_rows = [(item.speaker_id, item.reference_wav) for item in items]
    score_values: list[float] = []
    label_values: list[bool] = []
    positive_pairs = 0
    negative_pairs = 0
    for item in items:
        generated_embedding = generated_embeddings[item.item_id]
        for ref_speaker_id, ref_wav in reference_rows:
            score_values.append(
                cosine_similarity(generated_embedding, reference_embeddings[ref_wav])
            )
            is_positive = ref_speaker_id == item.speaker_id
            label_values.append(is_positive)
            if is_positive:
                positive_pairs += 1
            else:
                negative_pairs += 1
    return (
        np.array(score_values, dtype=np.float64),
        np.array(label_values, dtype=np.bool_),
        positive_pairs,
        negative_pairs,
    )


def write_pair_scores_csv(rows: list[PairScoreRow], output_path: Path) -> None:
    """Write per-item scores to CSV."""
    fieldnames = [
        "item_id",
        "speaker_id",
        "prompt_wav",
        "reference_wav",
        "generated_wav",
        "same_item_cosine",
        "predicted_speaker",
        "top1_correct",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "item_id": row.item_id,
                    "speaker_id": row.speaker_id,
                    "prompt_wav": str(row.prompt_wav),
                    "reference_wav": str(row.reference_wav),
                    "generated_wav": str(row.generated_wav),
                    "same_item_cosine": f"{row.same_item_cosine:.8f}",
                    "predicted_speaker": row.predicted_speaker,
                    "top1_correct": int(row.top1_correct),
                }
            )


def compute_macro_speaker_metrics(
    rows: list[PairScoreRow],
) -> tuple[float, float] | None:
    """Compute speaker-balanced macro means for cosine and top-1 accuracy."""
    if not rows:
        return None
    grouped: dict[str, list[PairScoreRow]] = {}
    for row in rows:
        grouped.setdefault(row.speaker_id, []).append(row)
    if not grouped:
        return None

    speaker_cosine_means: list[float] = []
    speaker_top1_means: list[float] = []
    for speaker_rows in grouped.values():
        cosine_values = [row.same_item_cosine for row in speaker_rows]
        top1_values = [1.0 if row.top1_correct else 0.0 for row in speaker_rows]
        speaker_cosine_means.append(float(np.mean(cosine_values)))
        speaker_top1_means.append(float(np.mean(top1_values)))
    return float(np.mean(speaker_cosine_means)), float(np.mean(speaker_top1_means))
