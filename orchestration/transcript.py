"""Transcript domain DTOs and adapters for orchestration boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, TypeAlias


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
        return "".join(
            f"[{segment.speaker}]: {segment.text}\n" for segment in self.segments
        )

    def retrieval_segments(self) -> list[dict[str, str]]:
        """Build retrieval DTO payload segments used for indexing."""
        return [
            {"speaker": segment.speaker, "text": segment.text}
            for segment in self.segments
        ]

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
