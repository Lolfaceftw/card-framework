"""Tests for transcript domain DTO adapters."""

from __future__ import annotations

from card_framework.orchestration.transcript import Transcript


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


def test_transcript_to_full_text_includes_affect_hint_when_present() -> None:
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "hello world",
                    "affect_hint": "preset=urgent; top=angry:0.75",
                }
            ]
        }
    )

    assert (
        transcript.to_full_text()
        == "[SPEAKER_00] [affect: preset=urgent; top=angry:0.75]: hello world\n"
    )


def test_transcript_retrieval_segments_preserve_affect_fields() -> None:
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "hello world",
                    "affect_hint": "preset=warm",
                    "suggested_emo_preset": "warm",
                    "emotion_vector": [0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24],
                    "emotion_vector_source": "affect_analysis",
                }
            ]
        }
    )

    assert transcript.retrieval_segments() == [
        {
            "speaker": "SPEAKER_00",
            "text": "hello world",
            "affect_hint": "preset=warm",
            "suggested_emo_preset": "warm",
            "emotion_vector": [0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24],
            "emotion_vector_source": "affect_analysis",
        }
    ]

