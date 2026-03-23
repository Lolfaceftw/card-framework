"""Shared emotion-vector contracts and helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
from typing import Any

INDEX_TTS_EMOTION_ORDER = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)
INDEX_TTS_EMOTION_COUNT = len(INDEX_TTS_EMOTION_ORDER)
INDEX_TTS_DEFAULT_MAX_TOTAL = 0.8
INDEX_TTS_EMOTION_BIAS = (
    0.9375,
    0.875,
    1.0,
    1.0,
    0.9375,
    0.9375,
    0.6875,
    0.5625,
)


@dataclass(slots=True, frozen=True)
class EmotionVector:
    """Represent one validated IndexTTS-compatible emotion vector."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != INDEX_TTS_EMOTION_COUNT:
            raise ValueError(
                "EmotionVector must contain exactly "
                f"{INDEX_TTS_EMOTION_COUNT} values."
            )
        normalized_values: list[float] = []
        for value in self.values:
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError("EmotionVector values must be finite numbers.")
            if numeric < 0.0:
                raise ValueError("EmotionVector values must be >= 0.0.")
            normalized_values.append(numeric)
        object.__setattr__(self, "values", tuple(normalized_values))

    @classmethod
    def zero(cls) -> "EmotionVector":
        """Return an all-zero emotion vector."""
        return cls((0.0,) * INDEX_TTS_EMOTION_COUNT)

    @classmethod
    def from_iterable(
        cls,
        values: Iterable[Any],
    ) -> "EmotionVector":
        """Build a vector from an iterable of numeric values."""
        return cls(tuple(float(value) for value in values))

    @classmethod
    def from_mapping(
        cls,
        values_by_label: Mapping[str, Any],
    ) -> "EmotionVector":
        """Build a vector from a mapping keyed by emotion label."""
        return cls(
            tuple(
                float(values_by_label.get(label, 0.0))
                for label in INDEX_TTS_EMOTION_ORDER
            )
        )

    @classmethod
    def parse(cls, value: Any) -> "EmotionVector | None":
        """Parse one emotion vector from common payload shapes."""
        if value is None:
            return None
        if isinstance(value, EmotionVector):
            return value
        if isinstance(value, Mapping):
            return cls.from_mapping(value)
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return None
            parsed: Any
            if normalized.startswith("["):
                parsed = json.loads(normalized)
            else:
                parsed = [item.strip() for item in normalized.split(",")]
            if isinstance(parsed, Mapping):
                return cls.from_mapping(parsed)
            if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes)):
                return cls.from_iterable(parsed)
            raise ValueError("Unsupported emotion-vector string payload.")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return cls.from_iterable(value)
        raise ValueError("Unsupported emotion-vector payload.")

    @classmethod
    def from_raw(cls, value: Any) -> "EmotionVector | None":
        """Backward-compatible alias for ``parse``."""
        return cls.parse(value)

    def to_list(self) -> list[float]:
        """Return the vector as a mutable list of floats."""
        return [float(value) for value in self.values]

    def to_payload(self, *, precision: int = 6) -> list[float]:
        """Return a JSON-friendly rounded float list."""
        return [round(float(value), precision) for value in self.values]

    def to_compact_string(self, *, precision: int = 4) -> str:
        """Return a compact comma-separated representation."""
        return ",".join(f"{float(value):.{precision}f}" for value in self.values)

    def to_xml_attribute(self) -> str:
        """Return a stable XML-friendly comma-separated representation."""
        return self.to_compact_string(precision=6)

    def total_intensity(self) -> float:
        """Return the total emotion intensity."""
        return float(sum(self.values))

    def clamp_total(self, *, max_total: float) -> "EmotionVector":
        """Scale the vector down when its total exceeds ``max_total``."""
        if max_total <= 0.0:
            raise ValueError("max_total must be > 0.0.")
        total = self.total_intensity()
        if total <= max_total or total <= 0.0:
            return self
        scale = max_total / total
        return EmotionVector(tuple(value * scale for value in self.values))

    def with_index_tts_normalization(
        self,
        *,
        apply_bias: bool = True,
        max_sum: float = INDEX_TTS_DEFAULT_MAX_TOTAL,
    ) -> "EmotionVector":
        """Return a vector clamped to IndexTTS's documented total-intensity cap."""
        values = list(self.values)
        if apply_bias:
            values = [
                value * bias
                for value, bias in zip(values, INDEX_TTS_EMOTION_BIAS, strict=True)
            ]
        return EmotionVector(tuple(values)).clamp_total(max_total=max_sum)

    def with_component_floor(
        self,
        *,
        label: str,
        minimum: float,
        max_total: float | None = None,
    ) -> "EmotionVector":
        """Ensure one named component is at least ``minimum``."""
        normalized_label = str(label).strip().lower()
        if normalized_label not in INDEX_TTS_EMOTION_ORDER:
            raise ValueError(f"Unknown emotion label: {label!r}")
        if minimum < 0.0:
            raise ValueError("minimum must be >= 0.0.")
        index = INDEX_TTS_EMOTION_ORDER.index(normalized_label)
        values = list(self.values)
        values[index] = max(values[index], float(minimum))
        vector = EmotionVector(tuple(values))
        if max_total is not None:
            return vector.clamp_total(max_total=max_total)
        return vector

    def top_components(self, *, limit: int = 3) -> tuple[tuple[str, float], ...]:
        """Return the strongest named components in descending order."""
        if limit <= 0:
            return ()
        ranked = sorted(
            zip(INDEX_TTS_EMOTION_ORDER, self.values, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return tuple(
            (label, float(value))
            for label, value in ranked[:limit]
            if float(value) > 0.0
        )

    def to_named_hint(self, *, limit: int = 3, precision: int = 2) -> str:
        """Return a prompt-friendly named summary of the strongest components."""
        return ", ".join(
            f"{label}:{value:.{precision}f}"
            for label, value in self.top_components(limit=limit)
        )


def coerce_emotion_vector(value: Any) -> EmotionVector | None:
    """Best-effort parsing helper that returns ``None`` for empty inputs."""
    return EmotionVector.parse(value)
