"""Custom metric helpers for duration-first summarizer-critic benchmarking."""

from __future__ import annotations

from collections import Counter
import re
from statistics import mean

from card_framework.agents.utils import count_words
from card_framework.benchmark.reference_free.contracts import clamp_score
from card_framework.shared.summary_xml import parse_summary_xml

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def score_duration_control(
    *,
    target_seconds: int,
    candidate_seconds: float,
    tolerance_ratio: float,
) -> float:
    """Score how closely the candidate duration matches the target."""
    if target_seconds <= 0 or candidate_seconds < 0:
        return 0.0
    tolerance_seconds = max(0.0, float(target_seconds) * max(0.0, tolerance_ratio))
    difference = abs(float(candidate_seconds) - float(target_seconds))
    if difference <= tolerance_seconds:
        return 1.0
    decay_span = max(1.0, float(target_seconds) - tolerance_seconds)
    return clamp_score(1.0 - ((difference - tolerance_seconds) / decay_span))


def combine_factualness(
    *,
    alignscore: float | None,
    judge_scores: dict[str, float] | None,
) -> float:
    """Combine reference-free factualness signals into one score."""
    values: list[float] = []
    if alignscore is not None:
        values.append(float(alignscore))
    if judge_scores and judge_scores.get("factuality") is not None:
        values.append(float(judge_scores["factuality"]))
    if not values:
        return 0.0
    return clamp_score(mean(values))


def _summary_text(summary_xml: str) -> str:
    try:
        turns = parse_summary_xml(summary_xml)
    except ValueError:
        return re.sub(r"<[^>]+>", " ", summary_xml).strip()
    return " ".join(turn.text.strip() for turn in turns if turn.text.strip()).strip()


def combine_naturalness(
    judge_scores: dict[str, float] | None,
    *,
    summary_xml: str,
) -> float:
    """Map judge fluency signals or a fallback readability heuristic to naturalness."""
    if not judge_scores:
        text = _summary_text(summary_xml)
        if not text:
            return 0.0
        sentences = [
            fragment.strip()
            for fragment in _SENTENCE_SPLIT_PATTERN.split(text)
            if fragment.strip()
        ]
        if not sentences:
            sentences = [text]
        sentence_lengths = [count_words(sentence) for sentence in sentences if sentence]
        average_sentence_words = (
            sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
        )
        sentence_band = clamp_score(1.0 - (abs(average_sentence_words - 18.0) / 24.0))
        punctuation_score = clamp_score(
            sum(1 for sentence in sentences if re.search(r"[.!?][\"')\]]*$", sentence))
            / len(sentences)
        )
        tokens = _TOKEN_PATTERN.findall(text.lower())
        repetition_score = 0.0
        if tokens:
            repetition_score = clamp_score(len(set(tokens)) / len(tokens))
        return clamp_score(mean([sentence_band, punctuation_score, repetition_score]))
    if judge_scores.get("coherence") is not None:
        return clamp_score(float(judge_scores["coherence"]))
    if judge_scores.get("overall") is not None:
        return clamp_score(float(judge_scores["overall"]))
    return 0.0


def _normalize_tokens(text: str) -> Counter[str]:
    return Counter(_TOKEN_PATTERN.findall(text.lower()))


def _speaker_token_counters(transcript: dict[str, object]) -> dict[str, Counter[str]]:
    counters: dict[str, Counter[str]] = {}
    raw_segments = transcript.get("segments", [])
    if not isinstance(raw_segments, list):
        return counters
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        speaker = str(raw_segment.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
        counters.setdefault(speaker, Counter()).update(
            _normalize_tokens(str(raw_segment.get("text", "")))
        )
    return counters


def _speaker_alias_map(transcript: dict[str, object]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    raw_segments = transcript.get("segments", [])
    if not isinstance(raw_segments, list):
        return alias_map
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        speaker = str(raw_segment.get("speaker", "UNKNOWN")).strip()
        if not speaker:
            continue
        normalized = speaker.lower()
        alias_map[normalized] = speaker
        compact = re.sub(r"\s+", " ", normalized).strip()
        alias_map[compact] = speaker
        if "(" in speaker and ")" in speaker:
            paren_content = (
                speaker[speaker.find("(") + 1 : speaker.rfind(")")].strip().lower()
            )
            if paren_content:
                alias_map[paren_content] = speaker
        speaker_tokens = re.findall(r"[A-Za-z][A-Za-z'.-]*", speaker)
        if speaker_tokens:
            alias_map[speaker_tokens[-1].lower()] = speaker
    return alias_map


def _mentioned_speakers(text: str, alias_map: dict[str, str]) -> set[str]:
    normalized_text = text.lower()
    mentioned: set[str] = set()
    for alias, speaker in alias_map.items():
        if not alias:
            continue
        if re.search(rf"\b{re.escape(alias)}\b", normalized_text):
            mentioned.add(speaker)
    return mentioned


def score_speaker_attribution(
    summary_xml: str,
    transcript: dict[str, object],
    *,
    reference_summary: str = "",
) -> float:
    """Score speaker attribution claims mentioned in summary text against source speakers."""
    summary_text = _summary_text(summary_xml)
    if not summary_text:
        return 0.0
    speaker_counters = _speaker_token_counters(transcript)
    if not speaker_counters:
        return 0.0
    alias_map = _speaker_alias_map(transcript)
    candidate_mentions = _mentioned_speakers(summary_text, alias_map)
    reference_mentions = _mentioned_speakers(reference_summary, alias_map)
    if not candidate_mentions and not reference_mentions:
        return 0.5
    if reference_mentions and not candidate_mentions:
        return 0.0

    per_claim_scores: list[float] = []
    sentences = [
        fragment.strip() for fragment in _SENTENCE_SPLIT_PATTERN.split(summary_text)
    ]
    for sentence in sentences:
        if not sentence:
            continue
        sentence_mentions = _mentioned_speakers(sentence, alias_map)
        for mentioned_speaker in sentence_mentions:
            current_counter = speaker_counters.get(mentioned_speaker)
            if current_counter is None:
                per_claim_scores.append(0.0)
                continue
            claim_text = sentence
            for alias, canonical_speaker in alias_map.items():
                if canonical_speaker != mentioned_speaker:
                    continue
                claim_text = re.sub(
                    rf"\b{re.escape(alias)}\b",
                    " ",
                    claim_text,
                    flags=re.IGNORECASE,
                )
            turn_counter = _normalize_tokens(claim_text)
            if not turn_counter:
                per_claim_scores.append(0.5)
                continue
            shared_same = sum((turn_counter & current_counter).values())
            shared_other = 0
            for other_speaker, other_counter in speaker_counters.items():
                if other_speaker == mentioned_speaker:
                    continue
                shared_other = max(
                    shared_other, sum((turn_counter & other_counter).values())
                )
            evidence_total = shared_same + shared_other
            if evidence_total <= 0:
                per_claim_scores.append(0.5)
                continue
            dominance = shared_same / float(evidence_total)
            per_claim_scores.append(clamp_score(0.5 + (0.5 * dominance)))
    if not per_claim_scores:
        return 0.5
    return clamp_score(mean(per_claim_scores))


def compute_composite_score(
    *,
    duration_control: float,
    factualness: float,
    naturalness: float,
    speaker_attribution: float,
) -> float:
    """Average the four required benchmark dimensions into one score."""
    return clamp_score(
        mean(
            [
                duration_control,
                factualness,
                naturalness,
                speaker_attribution,
            ]
        )
    )
