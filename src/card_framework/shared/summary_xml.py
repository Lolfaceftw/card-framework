"""Summary XML parsing and serialization helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree
from xml.sax.saxutils import escape, quoteattr

from card_framework.shared.emotion_vector import EmotionVector, coerce_emotion_vector

DEFAULT_EMO_PRESET = "neutral"


@dataclass(slots=True, frozen=True)
class SummaryTurn:
    """Represent one speaker-tagged summary turn."""

    speaker: str
    text: str
    emo_preset: str = DEFAULT_EMO_PRESET
    emotion_vector: EmotionVector | None = None
    emotion_vector_source: str = ""
    source_segment_indices: tuple[int, ...] = ()
    source_segment_weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.speaker.strip():
            raise ValueError("speaker must be non-empty")
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        if not self.emo_preset.strip():
            raise ValueError("emo_preset must be non-empty")
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
        normalized_indices = _normalize_source_segment_indices(
            self.source_segment_indices
        )
        normalized_weights = _normalize_source_segment_weights(
            self.source_segment_weights
        )
        if normalized_weights and not normalized_indices:
            raise ValueError(
                "source_segment_weights require source_segment_indices."
            )
        if normalized_weights and len(normalized_weights) != len(normalized_indices):
            raise ValueError(
                "source_segment_weights must match source_segment_indices length."
            )
        object.__setattr__(self, "source_segment_indices", normalized_indices)
        object.__setattr__(self, "source_segment_weights", normalized_weights)


def parse_summary_xml(summary_xml: str) -> list[SummaryTurn]:
    """Parse speaker-tagged summary XML fragments into typed turns."""
    normalized = summary_xml.strip()
    if not normalized:
        return []

    wrapped_xml = f"<summary>{normalized}</summary>"
    try:
        root = ElementTree.fromstring(wrapped_xml)
    except ElementTree.ParseError as exc:
        raise ValueError("Summary XML is not well-formed.") from exc

    turns: list[SummaryTurn] = []
    for child in root:
        speaker = str(child.tag).strip()
        text = "".join(child.itertext()).strip()
        if not speaker or not text:
            continue
        emo_preset = str(
            child.attrib.get("emo_preset", DEFAULT_EMO_PRESET)
        ).strip() or DEFAULT_EMO_PRESET
        try:
            emotion_vector = _parse_emotion_vector_attr(child.attrib)
        except ValueError as exc:
            raise ValueError(
                "Summary XML contains an invalid emotion_vector attribute."
            ) from exc
        try:
            source_segment_indices = _parse_int_tuple_attr(
                child.attrib,
                "source_segment_indices",
            )
            source_segment_weights = _parse_float_tuple_attr(
                child.attrib,
                "source_segment_weights",
            )
        except ValueError as exc:
            raise ValueError(
                "Summary XML contains invalid source segment linkage attributes."
            ) from exc
        turns.append(
            SummaryTurn(
                speaker=speaker,
                text=text,
                emo_preset=emo_preset,
                emotion_vector=emotion_vector,
                emotion_vector_source=str(
                    child.attrib.get("emotion_vector_source", "")
                ).strip(),
                source_segment_indices=source_segment_indices,
                source_segment_weights=source_segment_weights,
            )
        )
    return turns


def serialize_summary_turns(turns: list[SummaryTurn]) -> str:
    """Serialize summary turns into speaker-tagged XML fragments."""
    return "\n".join(
        (
            f"<{turn.speaker} "
            + _serialize_turn_attributes(turn)
            + ">"
            f"{escape(turn.text)}</{turn.speaker}>"
        )
        for turn in turns
    )


def _serialize_turn_attributes(turn: SummaryTurn) -> str:
    """Render one turn's XML attribute block."""
    attributes = [f"emo_preset={quoteattr(turn.emo_preset)}"]
    if turn.emotion_vector is not None:
        attributes.append(
            f"emotion_vector={quoteattr(turn.emotion_vector.to_compact_string())}"
        )
    if turn.emotion_vector_source:
        attributes.append(
            f"emotion_vector_source={quoteattr(turn.emotion_vector_source)}"
        )
    if turn.source_segment_indices:
        attributes.append(
            "source_segment_indices="
            + quoteattr(_serialize_int_tuple(turn.source_segment_indices))
        )
    if turn.source_segment_weights:
        attributes.append(
            "source_segment_weights="
            + quoteattr(_serialize_float_tuple(turn.source_segment_weights))
        )
    return " ".join(attributes)


def count_summary_turns(summary_xml: str) -> int:
    """Return the number of speaker-tagged turns in summary XML."""
    return len(parse_summary_xml(summary_xml))


def _parse_emotion_vector_attr(attributes: dict[str, str]) -> EmotionVector | None:
    """Parse one optional emotion-vector XML attribute."""
    raw_value = attributes.get("emotion_vector", attributes.get("emo_vector", ""))
    return coerce_emotion_vector(raw_value)


def _parse_int_tuple_attr(
    attributes: dict[str, str],
    key: str,
) -> tuple[int, ...]:
    """Parse one comma-separated integer tuple XML attribute."""
    raw_value = str(attributes.get(key, "")).strip()
    if not raw_value:
        return ()
    values: list[int] = []
    for item in raw_value.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        value = int(normalized)
        if value < 0:
            raise ValueError(f"{key} values must be >= 0.")
        values.append(value)
    return tuple(values)


def _parse_float_tuple_attr(
    attributes: dict[str, str],
    key: str,
) -> tuple[float, ...]:
    """Parse one comma-separated float tuple XML attribute."""
    raw_value = str(attributes.get(key, "")).strip()
    if not raw_value:
        return ()
    values: list[float] = []
    for item in raw_value.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        value = float(normalized)
        if value <= 0.0:
            raise ValueError(f"{key} values must be > 0.")
        values.append(value)
    return tuple(values)


def _normalize_source_segment_indices(value: Any) -> tuple[int, ...]:
    """Normalize source segment indices into an immutable tuple."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return _parse_int_tuple_attr({"source_segment_indices": value}, "source_segment_indices")
    if isinstance(value, Iterable):
        indices = tuple(int(item) for item in value)
        if any(index < 0 for index in indices):
            raise ValueError("source_segment_indices values must be >= 0.")
        return indices
    raise ValueError("Unsupported source_segment_indices payload.")


def _normalize_source_segment_weights(value: Any) -> tuple[float, ...]:
    """Normalize source segment weights into an immutable tuple."""
    if value is None or value == "":
        return ()
    if isinstance(value, str):
        return _parse_float_tuple_attr(
            {"source_segment_weights": value},
            "source_segment_weights",
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        weights = tuple(float(item) for item in value)
        if any(weight <= 0.0 for weight in weights):
            raise ValueError("source_segment_weights values must be > 0.")
        return weights
    raise ValueError("Unsupported source_segment_weights payload.")


def _serialize_int_tuple(values: Sequence[int]) -> str:
    """Serialize integer tuples into stable comma-separated XML attribute text."""
    return ",".join(str(int(value)) for value in values)


def _serialize_float_tuple(values: Sequence[float]) -> str:
    """Serialize float tuples into stable comma-separated XML attribute text."""
    return ",".join(f"{float(value):.6f}" for value in values)
