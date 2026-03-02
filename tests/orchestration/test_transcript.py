"""Tests for transcript domain DTO adapters."""

from __future__ import annotations

from orchestration.transcript import Transcript


def test_transcript_roundtrip_preserves_known_and_extra_fields() -> None:
    payload = {
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "start_time": 10,
                "end_time": 100,
                "text": "hello",
                "confidence": 0.95,
            }
        ],
        "metadata": {"source": "unit-test"},
        "session_id": "abc123",
    }

    transcript = Transcript.from_mapping(payload)
    serialized = transcript.to_payload()

    assert serialized["session_id"] == "abc123"
    assert serialized["metadata"] == {"source": "unit-test"}
    assert serialized["segments"][0]["confidence"] == 0.95
    assert serialized["segments"][0]["speaker"] == "SPEAKER_00"


def test_transcript_to_full_text_matches_prompt_format() -> None:
    transcript = Transcript.from_mapping(
        {"segments": [{"speaker": "SPEAKER_00", "text": "hello world"}]}
    )

    assert transcript.to_full_text() == "[SPEAKER_00]: hello world\n"
