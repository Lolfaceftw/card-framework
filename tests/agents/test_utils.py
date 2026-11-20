from __future__ import annotations

from card_framework.agents.utils import validate_transcript_schema


def test_validate_transcript_schema_preserves_segment_extras() -> None:
    payload = {
        "metadata": {"speaker_samples_manifest_path": "/tmp/manifest.json"},
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "start_time": 100,
                "end_time": 900,
                "text": "Segment text",
                "emotion_vector": [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.2],
                "emotion_vector_source": "affect_analysis",
                "emotion_vector_confidence": 0.91,
                "affect_hint": "preset=engaged",
                "suggested_emo_preset": "engaged",
                "custom_flag": True,
            }
        ],
    }

    normalized = validate_transcript_schema(payload)

    assert normalized["metadata"] == payload["metadata"]
    segment = normalized["segments"][0]
    assert segment["speaker"] == "SPEAKER_00"
    assert segment["start_time"] == 100
    assert segment["end_time"] == 900
    assert segment["text"] == "Segment text"
    assert segment["emotion_vector"] == [0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.2]
    assert segment["emotion_vector_source"] == "affect_analysis"
    assert segment["emotion_vector_confidence"] == 0.91
    assert segment["affect_hint"] == "preset=engaged"
    assert segment["suggested_emo_preset"] == "engaged"
    assert segment["custom_flag"] is True
