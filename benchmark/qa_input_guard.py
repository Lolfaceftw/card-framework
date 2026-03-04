"""Preflight compatibility checks for candidate summary and source transcript."""

from __future__ import annotations

from dataclasses import dataclass
import re


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}

_DISTINCTIVE_EXCLUSIONS = _STOPWORDS | {
    "investing",
    "investment",
    "investor",
    "investors",
    "market",
    "markets",
    "financial",
    "finance",
    "money",
    "risk",
    "returns",
    "podcast",
    "rational",
    "reminder",
}


@dataclass(slots=True, frozen=True)
class InputCompatibilityResult:
    """Outcome of summary-to-source lexical compatibility checks."""

    is_compatible: bool
    token_overlap_ratio: float
    shared_token_count: int
    summary_token_count: int
    source_token_count: int
    shared_tokens_preview: tuple[str, ...]
    distinctive_overlap_ratio: float
    shared_distinctive_count: int
    summary_distinctive_count: int
    source_distinctive_count: int
    shared_distinctive_preview: tuple[str, ...]
    shared_name_phrase_count: int
    summary_name_phrase_count: int
    source_name_phrase_count: int
    shared_name_phrase_preview: tuple[str, ...]
    reason: str


def _strip_xml_tags(xml_text: str) -> str:
    """Remove XML tags from candidate summary text."""
    return re.sub(r"<[^>]+>", " ", xml_text)


def _tokenize(text: str) -> set[str]:
    """Tokenize text into de-duplicated lowercase lexical units."""
    candidates = re.findall(r"[A-Za-z][A-Za-z0-9_'-]{1,}", text.lower())
    return {
        token
        for token in candidates
        if len(token) >= 3 and token not in _STOPWORDS and not token.startswith("e0")
    }


def _distinctive_tokens(text: str) -> set[str]:
    """Extract stronger topic-disambiguating lexical tokens."""
    return {
        token
        for token in _tokenize(text)
        if len(token) >= 6 and token not in _DISTINCTIVE_EXCLUSIONS
    }


def _name_phrases(text: str) -> set[str]:
    """Extract multi-word proper-name-like phrases."""
    pattern = r"\b(?:[A-Z][a-zA-Z'-]{1,}\s+){1,3}[A-Z][a-zA-Z'-]{1,}\b"
    phrases = {match.strip().lower() for match in re.findall(pattern, text)}
    filtered: set[str] = set()
    for phrase in phrases:
        words = phrase.split()
        if len(words) < 2:
            continue
        if words[0] in _STOPWORDS or words[-1] in _STOPWORDS:
            continue
        filtered.add(phrase)
    return filtered


def evaluate_input_compatibility(
    *,
    summary_xml: str,
    source_text: str,
    min_overlap_ratio: float,
    min_shared_tokens: int,
    min_shared_distinctive_tokens: int,
    min_shared_name_phrases: int,
) -> InputCompatibilityResult:
    """Evaluate whether summary and source likely refer to the same content."""
    summary_plain_text = _strip_xml_tags(summary_xml)
    summary_tokens = _tokenize(summary_plain_text)
    source_tokens = _tokenize(source_text)
    shared = summary_tokens & source_tokens
    summary_distinctive = _distinctive_tokens(summary_plain_text)
    source_distinctive = _distinctive_tokens(source_text)
    shared_distinctive = summary_distinctive & source_distinctive
    summary_name_phrases = _name_phrases(summary_plain_text)
    source_name_phrases = _name_phrases(source_text)
    shared_name_phrases = summary_name_phrases & source_name_phrases

    denominator = max(len(summary_tokens), 1)
    overlap_ratio = len(shared) / float(denominator)
    distinctive_denominator = max(len(summary_distinctive), 1)
    distinctive_overlap_ratio = len(shared_distinctive) / float(distinctive_denominator)
    is_compatible = (
        overlap_ratio >= min_overlap_ratio and len(shared) >= min_shared_tokens
    ) and (
        len(shared_distinctive) >= min_shared_distinctive_tokens
        and len(shared_name_phrases) >= min_shared_name_phrases
    )

    if is_compatible:
        reason = (
            "Input compatibility check passed with sufficient lexical overlap between "
            "summary and source transcript."
        )
    else:
        reason = (
            "Input compatibility check failed: summary and source transcript appear to "
            "describe different content."
        )

    preview = tuple(sorted(shared)[:20])
    return InputCompatibilityResult(
        is_compatible=is_compatible,
        token_overlap_ratio=overlap_ratio,
        shared_token_count=len(shared),
        summary_token_count=len(summary_tokens),
        source_token_count=len(source_tokens),
        shared_tokens_preview=preview,
        distinctive_overlap_ratio=distinctive_overlap_ratio,
        shared_distinctive_count=len(shared_distinctive),
        summary_distinctive_count=len(summary_distinctive),
        source_distinctive_count=len(source_distinctive),
        shared_distinctive_preview=tuple(sorted(shared_distinctive)[:20]),
        shared_name_phrase_count=len(shared_name_phrases),
        summary_name_phrase_count=len(summary_name_phrases),
        source_name_phrase_count=len(source_name_phrases),
        shared_name_phrase_preview=tuple(sorted(shared_name_phrases)[:20]),
        reason=reason,
    )
