"""Turn-level emotion resolution strategies for live-draft rendering."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from card_framework.audio_pipeline.emotion.emotion_vectors import EmotionVector
from card_framework.audio_pipeline.emotion.summary_turn_emotion_linker import (
    SummaryTurnEmotionLinker,
)
from card_framework.orchestration.transcript import Transcript
from card_framework.shared.summary_xml import DEFAULT_EMO_PRESET, SummaryTurn

_VALID_RESOLUTION_STEPS = frozenset({"source_linked", "explicit"})


@dataclass(slots=True, frozen=True)
class TurnEmotionResolutionConfig:
    """Configurable policy for resolving one summary turn's emotion control."""

    prelink_enabled: bool = True
    strategy_name: str = "source_linked"
    fallback_order: tuple[str, ...] = (
        "source_linked",
        "explicit",
    )

    def __post_init__(self) -> None:
        normalized_strategy_name = str(self.strategy_name).strip().lower()
        object.__setattr__(self, "strategy_name", normalized_strategy_name)
        if normalized_strategy_name not in {"source_linked"}:
            raise ValueError(
                "strategy_name must be 'source_linked' for the current implementation."
            )

        normalized_fallback_order: list[str] = []
        for raw_step in self.fallback_order:
            step = str(raw_step).strip().lower()
            if not step:
                continue
            if step not in _VALID_RESOLUTION_STEPS:
                raise ValueError(
                    "fallback_order entries must be drawn from "
                    f"{sorted(_VALID_RESOLUTION_STEPS)}."
                )
            if step not in normalized_fallback_order:
                normalized_fallback_order.append(step)
        if not normalized_fallback_order:
            raise ValueError("fallback_order must contain at least one step.")
        object.__setattr__(self, "fallback_order", tuple(normalized_fallback_order))


@dataclass(slots=True, frozen=True)
class ResolvedTurnEmotion:
    """Resolved turn emotion payload ready for live-draft rendering."""

    emotion_vector: EmotionVector | None
    emotion_vector_source: str
    source_segment_indices: tuple[int, ...] = ()
    source_segment_weights: tuple[float, ...] = ()
    effective_emo_preset: str = DEFAULT_EMO_PRESET

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "emotion_vector",
            EmotionVector.from_raw(self.emotion_vector),
        )
        object.__setattr__(
            self,
            "emotion_vector_source",
            str(self.emotion_vector_source).strip(),
        )
        object.__setattr__(
            self,
            "effective_emo_preset",
            str(self.effective_emo_preset).strip() or DEFAULT_EMO_PRESET,
        )
        indices = tuple(int(value) for value in self.source_segment_indices)
        if any(value < 0 for value in indices):
            raise ValueError("source_segment_indices values must be >= 0.")
        weights = tuple(float(value) for value in self.source_segment_weights)
        if any(value <= 0.0 for value in weights):
            raise ValueError("source_segment_weights values must be > 0.")
        if weights and len(indices) != len(weights):
            raise ValueError(
                "source_segment_weights must match source_segment_indices length."
            )
        object.__setattr__(self, "source_segment_indices", indices)
        object.__setattr__(self, "source_segment_weights", weights)

class TurnEmotionResolutionStrategy(Protocol):
    """Protocol for resolving one turn into a render-ready emotion payload."""

    def resolve_turn(
        self,
        *,
        turn: SummaryTurn,
        transcript: Transcript,
    ) -> ResolvedTurnEmotion:
        """Resolve one turn into an explicit vector or no extra emotion control."""


@dataclass(slots=True, frozen=True)
class SourceLinkedTurnEmotionResolutionStrategy:
    """Resolve turn emotion by preferring source-grounded linked vectors."""

    config: TurnEmotionResolutionConfig
    linker: SummaryTurnEmotionLinker | None

    def resolve_turn(
        self,
        *,
        turn: SummaryTurn,
        transcript: Transcript,
    ) -> ResolvedTurnEmotion:
        """Resolve one turn according to the configured precedence order."""
        effective_emo_preset = turn.emo_preset.strip() or DEFAULT_EMO_PRESET
        for step in self.config.fallback_order:
            if step == "source_linked":
                linked = self._resolve_source_linked(
                    turn=turn,
                    transcript=transcript,
                    effective_emo_preset=effective_emo_preset,
                )
                if linked is not None:
                    return linked
            elif step == "explicit" and turn.emotion_vector is not None:
                return ResolvedTurnEmotion(
                    emotion_vector=turn.emotion_vector,
                    emotion_vector_source=turn.emotion_vector_source or "explicit",
                    source_segment_indices=turn.source_segment_indices,
                    source_segment_weights=turn.source_segment_weights,
                    effective_emo_preset=effective_emo_preset,
                )
        return ResolvedTurnEmotion(
            emotion_vector=None,
            emotion_vector_source="",
            effective_emo_preset=effective_emo_preset,
        )

    def _resolve_source_linked(
        self,
        *,
        turn: SummaryTurn,
        transcript: Transcript,
        effective_emo_preset: str,
    ) -> ResolvedTurnEmotion | None:
        """Resolve one source-linked vector when prelinking is enabled."""
        if (
            not self.config.prelink_enabled
            or self.linker is None
            or not transcript.segments
        ):
            return None
        linking_result = self.linker.link_turns(turns=[turn], transcript=transcript)
        if not linking_result.turns:
            return None
        linked_turn = linking_result.turns[0]
        if linked_turn.emotion_vector is None:
            return None
        return ResolvedTurnEmotion(
            emotion_vector=linked_turn.emotion_vector,
            emotion_vector_source=(
                linked_turn.emotion_vector_source or "summary_turn_linking"
            ),
            source_segment_indices=linked_turn.source_segment_indices,
            source_segment_weights=linked_turn.source_segment_weights,
            effective_emo_preset=effective_emo_preset,
        )


def coerce_fallback_order(raw_value: object) -> tuple[str, ...]:
    """Normalize config payloads into the immutable fallback-order tuple."""
    if isinstance(raw_value, str):
        return tuple(
            step.strip().lower()
            for step in raw_value.split(",")
            if step.strip()
        )
    if isinstance(raw_value, Sequence) and not isinstance(raw_value, (str, bytes)):
        return tuple(str(step).strip().lower() for step in raw_value if str(step).strip())
    return ()

