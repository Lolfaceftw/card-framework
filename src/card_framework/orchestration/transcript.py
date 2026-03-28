"""Transcript domain DTOs and adapters for orchestration boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias

from card_framework.audio_pipeline.emotion_vectors import serialize_emotion_vector


@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    """Represent one normalized transcript segment."""

    speaker: str
    text: str
    start_time: int
    end_time: int
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "TranscriptSegment":
        """Build a segment from a mapping payload."""
        speaker = str(payload.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
        text = str(payload.get("text", ""))
        start_time = int(payload.get("start_time", 0))
        end_time = int(payload.get("end_time", start_time))
        extras = {
            key: value
            for key, value in payload.items()
            if key not in {"speaker", "text", "start_time", "end_time"}
        }
        return cls(
            speaker=speaker,
            text=text,
            start_time=start_time,
            end_time=end_time,
            extras=extras,
        )

    def to_mapping(self) -> dict[str, Any]:
        """Serialize this segment as a JSON-compatible dictionary."""
        payload: dict[str, Any] = {
            "speaker": self.speaker,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "text": self.text,
        }
        payload.update(self.extras)
        return payload


@dataclass(slots=True, frozen=True)
class Transcript:
    """Represent a transcript payload used in orchestrator flows."""

    segments: tuple[TranscriptSegment, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "Transcript":
        """Build a transcript DTO from a mapping payload."""
        raw_segments = payload.get("segments", [])
        segments: list[TranscriptSegment] = []
        if isinstance(raw_segments, list):
            for item in raw_segments:
                if isinstance(item, Mapping):
                    segments.append(TranscriptSegment.from_mapping(item))
        raw_metadata = payload.get("metadata", {})
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        extras = {
            key: value
            for key, value in payload.items()
            if key not in {"segments", "metadata"}
        }
        return cls(segments=tuple(segments), metadata=metadata, extras=extras)

    def to_payload(self) -> dict[str, Any]:
        """Serialize this transcript into the legacy dict payload shape."""
        payload = dict(self.extras)
        payload["segments"] = [segment.to_mapping() for segment in self.segments]
        payload["metadata"] = dict(self.metadata)
        return payload

    def to_full_text(self) -> str:
        """Render the transcript in prompt-friendly speaker/text form."""
        lines: list[str] = []
        for segment in self.segments:
            affect_hint = str(segment.extras.get("affect_hint", "")).strip()
            emotion_vector = serialize_emotion_vector(segment.extras.get("emotion_vector"))
            emotion_vector_text = (
                ", ".join(f"{value:.3f}" for value in emotion_vector)
                if emotion_vector is not None
                else ""
            )
            if affect_hint:
                prefix = f"[{segment.speaker}] [affect: {affect_hint}]"
                if emotion_vector_text:
                    prefix += f" [emotion_vector: {emotion_vector_text}]"
                lines.append(
                    f"{prefix}: {segment.text}\n"
                )
                continue
            if emotion_vector_text:
                lines.append(
                    f"[{segment.speaker}] [emotion_vector: {emotion_vector_text}]: {segment.text}\n"
                )
                continue
            lines.append(f"[{segment.speaker}]: {segment.text}\n")
        return "".join(lines)

    def retrieval_segments(self) -> list[dict[str, Any]]:
        """Build retrieval DTO payload segments used for indexing."""
        payloads: list[dict[str, Any]] = []
        for segment in self.segments:
            payload = {"speaker": segment.speaker, "text": segment.text}
            affect_hint = str(segment.extras.get("affect_hint", "")).strip()
            if affect_hint:
                payload["affect_hint"] = affect_hint
            suggested_emo_preset = str(
                segment.extras.get("suggested_emo_preset", "")
            ).strip()
            if suggested_emo_preset:
                payload["suggested_emo_preset"] = suggested_emo_preset
            emotion_vector = serialize_emotion_vector(segment.extras.get("emotion_vector"))
            if emotion_vector is not None:
                payload["emotion_vector"] = emotion_vector
            emotion_vector_source = str(
                segment.extras.get("emotion_vector_source", "")
            ).strip()
            if emotion_vector_source:
                payload["emotion_vector_source"] = emotion_vector_source
            payloads.append(payload)
        return payloads

    def with_metadata(self, metadata: Mapping[str, Any]) -> "Transcript":
        """Return a copy with replaced metadata."""
        return Transcript(
            segments=self.segments,
            metadata=dict(metadata),
            extras=dict(self.extras),
        )


TranscriptLike: TypeAlias = Transcript | Mapping[str, Any]


def coerce_transcript(value: TranscriptLike) -> Transcript:
    """Convert a transcript-like input into a transcript DTO."""
    if isinstance(value, Transcript):
        return value
    return Transcript.from_mapping(value)
