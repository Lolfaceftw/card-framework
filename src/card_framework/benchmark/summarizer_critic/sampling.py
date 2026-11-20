"""Deterministic sample-selection helpers for summarizer-critic benchmarks."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import hashlib
from typing import TypeVar

SampleT = TypeVar("SampleT")


def select_benchmark_samples(
    samples: Sequence[SampleT],
    *,
    max_samples: int,
    sample_seed: int | None,
    sample_id_getter: Callable[[SampleT], str],
) -> list[SampleT]:
    """Select benchmark samples with optional seeded randomization.

    When ``sample_seed`` is omitted, the input order is preserved and the
    existing prefix-selection behavior remains unchanged. When a seed is
    supplied, samples are ranked by a stable SHA-256 digest over the pair
    ``(seed, sample_id)`` so the selected subset is reproducible and
    independent of manifest ordering.

    Args:
        samples: Candidate samples in manifest order.
        max_samples: Maximum number of samples to keep. Non-positive values mean
            "keep all samples."
        sample_seed: Optional deterministic seed for seeded-random selection.
        sample_id_getter: Callable that returns the stable sample identifier.

    Returns:
        The selected samples in preserved or seeded-random order.
    """
    if sample_seed is None:
        return _apply_sample_limit(list(samples), max_samples=max_samples)

    ranked_samples: list[tuple[bytes, str, SampleT]] = []
    for sample in samples:
        sample_id = str(sample_id_getter(sample))
        rank = hashlib.sha256(f"{sample_seed}:{sample_id}".encode("utf-8")).digest()
        ranked_samples.append((rank, sample_id, sample))
    ranked_samples.sort(key=lambda item: (item[0], item[1]))
    return _apply_sample_limit(
        [sample for _, _, sample in ranked_samples],
        max_samples=max_samples,
    )


def _apply_sample_limit(samples: list[SampleT], *, max_samples: int) -> list[SampleT]:
    """Apply the benchmark sample cap to an already ordered sample list."""
    if max_samples <= 0:
        return samples
    return samples[:max_samples]
