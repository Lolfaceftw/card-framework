"""Tests for summary XML emotion-vector round-tripping."""

from __future__ import annotations

from card_framework.shared.emotion_vector import EmotionVector
from card_framework.shared.summary_xml import SummaryTurn, parse_summary_xml, serialize_summary_turns


def test_summary_xml_roundtrips_emotion_vector() -> None:
    xml = serialize_summary_turns(
        [
            SummaryTurn(
                speaker="SPEAKER_00",
                text="hello there",
                emo_preset="warm",
                emotion_vector=EmotionVector.from_raw(
                    [0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24]
                ),
            )
        ]
    )

    turns = parse_summary_xml(xml)

    assert len(turns) == 1
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].emo_preset == "warm"
    assert turns[0].emotion_vector == EmotionVector.from_raw(
        [0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24]
    )


def test_summary_xml_roundtrips_linked_support_metadata() -> None:
    xml = serialize_summary_turns(
        [
            SummaryTurn(
                speaker="SPEAKER_01",
                text="linked turn",
                emo_preset="engaged",
                emotion_vector=EmotionVector.from_raw(
                    [0.1, 0.02, 0.01, 0.0, 0.0, 0.02, 0.03, 0.12]
                ),
                emotion_vector_source="summary_turn_linking",
                source_segment_indices=(2, 4, 5),
                source_segment_weights=(0.5, 0.3, 0.2),
            )
        ]
    )

    turns = parse_summary_xml(xml)

    assert len(turns) == 1
    assert turns[0].emotion_vector_source == "summary_turn_linking"
    assert turns[0].source_segment_indices == (2, 4, 5)
    assert turns[0].source_segment_weights == (0.5, 0.3, 0.2)
