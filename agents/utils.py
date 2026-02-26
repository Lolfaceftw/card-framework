import json
import logging
import os
import re
from typing import Any

import hydra


def count_words(text: str) -> int:
    """Deterministic tool — counts words, ignoring XML/HTML tags."""
    if not text:
        return 0
    clean_text = re.sub(r"<[^>]+>", "", text)
    return len(clean_text.split())


def format_transcript_for_prompt(transcript_data: dict) -> str:
    """Parses the raw JSON transcript into a readable format for the LLM prompt."""
    formatted_text = ""
    for segment in transcript_data.get("segments", []):
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "")
        formatted_text += f"[{speaker}]: {text}\n"
    return formatted_text


def load_transcript(path: str) -> dict:
    """Load transcript from a JSON file (resolved relative to the *original* cwd)."""
    original_cwd = hydra.utils.get_original_cwd()
    full_path = os.path.join(original_cwd, path)
    with open(full_path, "r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    return validate_transcript_schema(payload)


def validate_transcript_schema(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and normalize transcript segments for downstream stages.

    Required fields per segment are ``speaker``, ``start_time``, ``end_time``,
    and ``text``. Missing timestamps are backfilled to preserve compatibility
    with older transcript JSONs.
    """
    segments_raw = payload.get("segments", [])
    if not isinstance(segments_raw, list):
        raise ValueError("Transcript must include a 'segments' list.")

    normalized_segments: list[dict[str, Any]] = []
    synthetic_cursor_ms = 0
    for index, segment in enumerate(segments_raw):
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        speaker = str(segment.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
        start_time = _to_milliseconds(segment.get("start_time"))
        end_time = _to_milliseconds(segment.get("end_time"))

        if start_time is None or end_time is None:
            logging.getLogger(__name__).warning(
                "Segment %s missing timestamps; synthesizing monotonic timings.",
                index,
            )
            start_time = synthetic_cursor_ms
            end_time = synthetic_cursor_ms + max(1, len(text.split()) * 300)

        start_time = max(0, int(start_time))
        end_time = max(start_time, int(end_time))
        synthetic_cursor_ms = max(synthetic_cursor_ms, end_time)

        normalized_segments.append(
            {
                "speaker": speaker,
                "start_time": start_time,
                "end_time": end_time,
                "text": text,
            }
        )

    normalized_payload = dict(payload)
    normalized_payload["segments"] = normalized_segments
    return normalized_payload


def _to_milliseconds(value: Any) -> int | None:
    """Best-effort cast for timestamp values."""
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str) and value.strip():
        try:
            return int(round(float(value)))
        except ValueError:
            return None
    return None
