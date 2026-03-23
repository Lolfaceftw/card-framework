"""Backward-compatible emotion-vector helpers for audio-pipeline modules."""

from __future__ import annotations

from typing import Any

from card_framework.shared.emotion_vector import (
    EmotionVector,
    INDEX_TTS_DEFAULT_MAX_TOTAL,
    INDEX_TTS_EMOTION_COUNT,
    INDEX_TTS_EMOTION_ORDER,
    coerce_emotion_vector,
)


def serialize_emotion_vector(raw_value: Any) -> list[float] | None:
    """Serialize one optional vector payload into a JSON-friendly list."""
    vector = coerce_emotion_vector(raw_value)
    if vector is None:
        return None
    return vector.to_payload()


__all__ = [
    "EmotionVector",
    "INDEX_TTS_DEFAULT_MAX_TOTAL",
    "INDEX_TTS_EMOTION_COUNT",
    "INDEX_TTS_EMOTION_ORDER",
    "coerce_emotion_vector",
    "serialize_emotion_vector",
]
