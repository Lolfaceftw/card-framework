"""I/O utilities for transcript payload persistence."""

from __future__ import annotations

import json
from pathlib import Path

from card_framework.audio_pipeline.contracts import (
    TranscriptMetadataPayload,
    TranscriptPayload,
    TranscriptSegment,
)
from card_framework.audio_pipeline.errors import ArtifactWriteError


def build_transcript_payload(
    segments: list[TranscriptSegment],
    metadata: TranscriptMetadataPayload | None = None,
) -> TranscriptPayload:
    """
    Build the serialized transcript payload consumed by downstream stages.

    Args:
        segments: Normalized transcript segments.
        metadata: Optional metadata object.

    Returns:
        Transcript payload dictionary.
    """
    payload: TranscriptPayload = {
        "segments": [
            {
                "speaker": segment.speaker,
                "start_time": segment.start_time,
                "end_time": segment.end_time,
                "text": segment.text,
            }
            for segment in segments
        ]
    }
    if metadata:
        payload["metadata"] = metadata
    return payload


def write_transcript_atomic(payload: TranscriptPayload, output_path: Path) -> None:
    """
    Persist transcript payload atomically.

    Args:
        payload: Transcript payload to write.
        output_path: Target transcript JSON path.

    Raises:
        ArtifactWriteError: If payload cannot be written or moved atomically.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temp_path.replace(output_path)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ArtifactWriteError(
            f"Failed to write transcript to '{output_path}'."
        ) from exc

