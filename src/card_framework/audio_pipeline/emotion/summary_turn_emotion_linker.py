"""Link final summary turns back to source segments and blend their emotion vectors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re

from card_framework.audio_pipeline.emotion.emotion_vectors import EmotionVector
from card_framework.orchestration.transcript import Transcript, TranscriptSegment
from card_framework.shared.summary_xml import SummaryTurn

_DEFAULT_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "with",
    }
)
_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


@dataclass(slots=True, frozen=True)
class SummaryTurnEmotionLinkingConfig:
    """Configurable controls for source-grounded summary-turn emotion linking."""

    enabled: bool = True
    top_k: int = 3
    min_support_score: float = 0.08
    same_speaker_bonus: float = 1.2
    default_emotion_confidence: float = 0.5
    min_emotion_confidence: float = 0.15
    lexical_similarity_weight: float = 0.45
    coverage_similarity_weight: float = 0.25
    sequence_similarity_weight: float = 0.15
    position_similarity_weight: float = 0.15
    min_sequence_similarity_without_overlap: float = 0.6
    require_emotion_vector: bool = True
    apply_indextts_bias: bool = False
    max_sum: float = 0.8
    preserve_existing_explicit_vectors: bool = True
    force_live_draft_rerender: bool = False

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError("top_k must be > 0.")
        if self.min_support_score < 0.0:
            raise ValueError("min_support_score must be >= 0.0.")
        if self.same_speaker_bonus <= 0.0:
            raise ValueError("same_speaker_bonus must be > 0.0.")
        if not 0.0 <= self.default_emotion_confidence <= 1.0:
            raise ValueError("default_emotion_confidence must be within [0.0, 1.0].")
        if not 0.0 <= self.min_emotion_confidence <= 1.0:
            raise ValueError("min_emotion_confidence must be within [0.0, 1.0].")
        if not 0.0 <= self.min_sequence_similarity_without_overlap <= 1.0:
            raise ValueError(
                "min_sequence_similarity_without_overlap must be within [0.0, 1.0]."
            )
        if self.max_sum <= 0.0:
            raise ValueError("max_sum must be > 0.0.")
        total_weight = (
            self.lexical_similarity_weight
            + self.coverage_similarity_weight
            + self.sequence_similarity_weight
            + self.position_similarity_weight
        )
        if total_weight <= 0.0:
            raise ValueError("At least one similarity weight must be > 0.0.")


@dataclass(slots=True, frozen=True)
class SummaryTurnSupport:
    """One source segment selected as evidence for a summary turn."""

    segment_index: int
    score: float
    weight: float


@dataclass(slots=True, frozen=True)
class SummaryTurnEmotionLinkingResult:
    """Structured output from linking summary turns to source emotion vectors."""

    turns: tuple[SummaryTurn, ...]
    linked_turn_count: int
    fallback_turn_count: int
    used_declared_support_count: int

    @property
    def changed(self) -> bool:
        """Return whether any turns received a linked explicit emotion vector."""
        return self.linked_turn_count > 0


@dataclass(slots=True, frozen=True)
class SummaryTurnEmotionLinker:
    """Blend transcript segment vectors onto summary turns."""

    config: SummaryTurnEmotionLinkingConfig

    def link_turns(
        self,
        *,
        turns: Sequence[SummaryTurn],
        transcript: Transcript,
    ) -> SummaryTurnEmotionLinkingResult:
        """Return summary turns enriched with explicit linked emotion vectors."""
        if not self.config.enabled or not turns or not transcript.segments:
            return SummaryTurnEmotionLinkingResult(
                turns=tuple(turns),
                linked_turn_count=0,
                fallback_turn_count=len(turns),
                used_declared_support_count=0,
            )

        enriched_turns: list[SummaryTurn] = []
        linked_turn_count = 0
        fallback_turn_count = 0
        used_declared_support_count = 0
        total_turns = len(turns)
        for turn_index, turn in enumerate(turns):
            if (
                self.config.preserve_existing_explicit_vectors
                and turn.emotion_vector is not None
            ):
                enriched_turns.append(turn)
                fallback_turn_count += 1
                continue
            supports, used_declared_support = self._select_support_segments(
                turn=turn,
                transcript=transcript,
                turn_index=turn_index,
                total_turns=total_turns,
            )
            if not supports:
                enriched_turns.append(turn)
                fallback_turn_count += 1
                continue
            vector = self._blend_support_vectors(
                transcript=transcript,
                supports=supports,
            )
            if vector is None:
                enriched_turns.append(turn)
                fallback_turn_count += 1
                continue
            linked_turn_count += 1
            if used_declared_support:
                used_declared_support_count += 1
            enriched_turns.append(
                SummaryTurn(
                    speaker=turn.speaker,
                    text=turn.text,
                    emo_preset=turn.emo_preset,
                    emotion_vector=vector,
                    emotion_vector_source="summary_turn_linking",
                    source_segment_indices=tuple(
                        support.segment_index for support in supports
                    ),
                    source_segment_weights=tuple(
                        support.weight for support in supports
                    ),
                )
            )
        return SummaryTurnEmotionLinkingResult(
            turns=tuple(enriched_turns),
            linked_turn_count=linked_turn_count,
            fallback_turn_count=fallback_turn_count,
            used_declared_support_count=used_declared_support_count,
        )

    def _select_support_segments(
        self,
        *,
        turn: SummaryTurn,
        transcript: Transcript,
        turn_index: int,
        total_turns: int,
    ) -> tuple[tuple[SummaryTurnSupport, ...], bool]:
        """Return the strongest supporting source segments for one summary turn."""
        if turn.source_segment_indices:
            declared_supports = self._resolve_declared_support_segments(
                turn=turn,
                transcript=transcript,
            )
            if declared_supports:
                return declared_supports, True
        automatic_supports = self._rank_automatic_support_segments(
            turn=turn,
            transcript=transcript,
            turn_index=turn_index,
            total_turns=total_turns,
        )
        return automatic_supports, False

    def _resolve_declared_support_segments(
        self,
        *,
        turn: SummaryTurn,
        transcript: Transcript,
    ) -> tuple[SummaryTurnSupport, ...]:
        """Resolve explicit support indices already attached to the summary turn."""
        scored_segments: list[tuple[int, float]] = []
        if turn.source_segment_weights:
            raw_weights = list(turn.source_segment_weights)
        else:
            raw_weights = []
        for position, segment_index in enumerate(turn.source_segment_indices):
            if segment_index < 0 or segment_index >= len(transcript.segments):
                continue
            segment = transcript.segments[segment_index]
            if self.config.require_emotion_vector and not self._segment_has_vector(segment):
                continue
            raw_weight = (
                float(raw_weights[position])
                if position < len(raw_weights)
                else self._segment_emotion_confidence(segment)
            )
            if raw_weight <= 0.0:
                continue
            scored_segments.append((segment_index, raw_weight))
        return _normalize_supports(scored_segments)

    def _rank_automatic_support_segments(
        self,
        *,
        turn: SummaryTurn,
        transcript: Transcript,
        turn_index: int,
        total_turns: int,
    ) -> tuple[SummaryTurnSupport, ...]:
        """Score transcript segments and keep the top support candidates."""
        scored_segments: list[tuple[int, float]] = []
        for segment_index, segment in enumerate(transcript.segments):
            if self.config.require_emotion_vector and not self._segment_has_vector(segment):
                continue
            support_score = self._score_segment_support(
                turn=turn,
                segment=segment,
                segment_index=segment_index,
                turn_index=turn_index,
                total_turns=total_turns,
                total_segments=len(transcript.segments),
            )
            if support_score < self.config.min_support_score:
                continue
            scored_segments.append((segment_index, support_score))
        scored_segments.sort(key=lambda item: item[1], reverse=True)
        return _normalize_supports(scored_segments[: self.config.top_k])

    def _score_segment_support(
        self,
        *,
        turn: SummaryTurn,
        segment: TranscriptSegment,
        segment_index: int,
        turn_index: int,
        total_turns: int,
        total_segments: int,
    ) -> float:
        """Score one transcript segment as support for a summary turn."""
        summary_tokens = _tokenize_text(turn.text)
        segment_tokens = _tokenize_text(segment.text)
        summary_token_set = set(summary_tokens)
        segment_token_set = set(segment_tokens)
        overlap_count = len(summary_token_set & segment_token_set)
        lexical_similarity = (
            overlap_count / len(summary_token_set) if summary_token_set else 0.0
        )
        coverage_similarity = (
            overlap_count / len(segment_token_set) if segment_token_set else 0.0
        )
        normalized_summary = " ".join(summary_tokens)
        normalized_segment = " ".join(segment_tokens)
        sequence_similarity = (
            SequenceMatcher(
                None,
                normalized_summary,
                normalized_segment,
            ).ratio()
            if normalized_summary and normalized_segment
            else 0.0
        )
        if overlap_count <= 0 and (
            sequence_similarity < self.config.min_sequence_similarity_without_overlap
        ):
            return 0.0
        position_similarity = _compute_position_similarity(
            turn_index=turn_index,
            total_turns=total_turns,
            segment_index=segment_index,
            total_segments=total_segments,
        )
        similarity_score = _weighted_similarity_score(
            lexical_similarity=lexical_similarity,
            coverage_similarity=coverage_similarity,
            sequence_similarity=sequence_similarity,
            position_similarity=position_similarity,
            config=self.config,
        )
        speaker_bonus = (
            self.config.same_speaker_bonus
            if segment.speaker.strip() == turn.speaker.strip()
            else 1.0
        )
        confidence = self._segment_emotion_confidence(segment)
        return similarity_score * confidence * speaker_bonus

    def _blend_support_vectors(
        self,
        *,
        transcript: Transcript,
        supports: Sequence[SummaryTurnSupport],
    ) -> EmotionVector | None:
        """Blend supporting source vectors with a weighted average."""
        if not supports:
            return None
        blended_values = [0.0] * len(EmotionVector.zero().values)
        total_weight = 0.0
        for support in supports:
            segment = transcript.segments[support.segment_index]
            source_vector = EmotionVector.from_raw(segment.extras.get("emotion_vector"))
            if source_vector is None:
                continue
            total_weight += float(support.weight)
            for index, value in enumerate(source_vector.values):
                blended_values[index] += float(value) * float(support.weight)
        if total_weight <= 0.0:
            return None
        normalized_values = [value / total_weight for value in blended_values]
        return EmotionVector(tuple(normalized_values)).with_index_tts_normalization(
            apply_bias=self.config.apply_indextts_bias,
            max_sum=self.config.max_sum,
        )

    def _segment_has_vector(self, segment: TranscriptSegment) -> bool:
        """Return whether one transcript segment already carries an emotion vector."""
        return EmotionVector.from_raw(segment.extras.get("emotion_vector")) is not None

    def _segment_emotion_confidence(self, segment: TranscriptSegment) -> float:
        """Return one bounded confidence score for a segment's emotion vector."""
        raw_value = segment.extras.get("emotion_vector_confidence")
        if raw_value is None:
            return self.config.default_emotion_confidence
        try:
            numeric = float(raw_value)
        except (TypeError, ValueError):
            return self.config.default_emotion_confidence
        if not math.isfinite(numeric):
            return self.config.default_emotion_confidence
        bounded = max(0.0, min(numeric, 1.0))
        return max(self.config.min_emotion_confidence, bounded)


def _normalize_supports(
    scored_segments: Sequence[tuple[int, float]],
) -> tuple[SummaryTurnSupport, ...]:
    """Normalize raw support scores into stable blending weights."""
    if not scored_segments:
        return ()
    total_score = sum(float(score) for _, score in scored_segments)
    if total_score <= 0.0:
        return ()
    return tuple(
        SummaryTurnSupport(
            segment_index=int(segment_index),
            score=float(score),
            weight=float(score) / total_score,
        )
        for segment_index, score in scored_segments
    )


def _compute_position_similarity(
    *,
    turn_index: int,
    total_turns: int,
    segment_index: int,
    total_segments: int,
) -> float:
    """Return a simple order-preserving similarity between summary and source positions."""
    if total_turns <= 1 or total_segments <= 1:
        return 1.0
    turn_position = float(turn_index) / float(max(1, total_turns - 1))
    segment_position = float(segment_index) / float(max(1, total_segments - 1))
    return max(0.0, 1.0 - abs(turn_position - segment_position))


def _weighted_similarity_score(
    *,
    lexical_similarity: float,
    coverage_similarity: float,
    sequence_similarity: float,
    position_similarity: float,
    config: SummaryTurnEmotionLinkingConfig,
) -> float:
    """Blend configured similarity signals into one support score."""
    total_weight = (
        config.lexical_similarity_weight
        + config.coverage_similarity_weight
        + config.sequence_similarity_weight
        + config.position_similarity_weight
    )
    return (
        (lexical_similarity * config.lexical_similarity_weight)
        + (coverage_similarity * config.coverage_similarity_weight)
        + (sequence_similarity * config.sequence_similarity_weight)
        + (position_similarity * config.position_similarity_weight)
    ) / total_weight


def _tokenize_text(text: str) -> list[str]:
    """Tokenize summary or transcript text into normalized content words."""
    lowered = str(text).lower()
    tokens = [
        token
        for token in _TOKEN_PATTERN.findall(lowered)
        if len(token) > 1 and token not in _DEFAULT_STOPWORDS
    ]
    if tokens:
        return tokens
    return _TOKEN_PATTERN.findall(lowered)

