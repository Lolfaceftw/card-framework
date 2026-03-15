"""Helpers for detecting repeated spans in streamed or complete model output."""

from __future__ import annotations

import html
import re

_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]+")
_REPEATED_SPAN_SPLIT_PATTERN = re.compile(r"(?:\n+|(?<=[.!?])\s+)")


def normalize_repetition_text(raw_text: str) -> str:
    """Normalize text for repetition and similarity checks."""
    normalized = html.unescape(raw_text).lower()
    normalized = _NON_ALNUM_PATTERN.sub(" ", normalized)
    return " ".join(normalized.split()).strip()


def count_words(raw_text: str) -> int:
    """Return a loose word count suitable for repetition thresholds."""
    return len(re.findall(r"\b\w+\b", raw_text))


def extract_repeated_response_span(
    raw_text: str,
    *,
    min_words: int,
    repeat_threshold: int,
) -> tuple[str, int] | None:
    """Return a repeated sentence-like span when one dominates a response."""
    counts: dict[str, tuple[int, str]] = {}
    for segment in _REPEATED_SPAN_SPLIT_PATTERN.split(raw_text):
        candidate = " ".join(segment.split()).strip(" -*\u2022")
        if not candidate:
            continue
        if count_words(candidate) < min_words:
            continue
        normalized = normalize_repetition_text(candidate)
        if not normalized:
            continue
        current_count, current_preview = counts.get(normalized, (0, candidate))
        counts[normalized] = (current_count + 1, current_preview)

    best_match: tuple[str, int] | None = None
    for count, preview in counts.values():
        if count < repeat_threshold:
            continue
        if best_match is None or count > best_match[1]:
            best_match = (preview, count)
    return best_match
