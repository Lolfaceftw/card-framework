from __future__ import annotations

from card_framework.audio_pipeline.summary_turn_emotion_linker import (
    SummaryTurnEmotionLinker,
    SummaryTurnEmotionLinkingConfig,
)
from card_framework.orchestration.transcript import Transcript
from card_framework.shared.emotion_vector import EmotionVector
from card_framework.shared.summary_xml import SummaryTurn


def test_linker_blends_multiple_source_segments() -> None:
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "Midnight Sun is exciting and joyful news.",
                    "emotion_vector": [0.36, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.06],
                    "emotion_vector_confidence": 0.9,
                },
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 1000,
                    "end_time": 2000,
                    "text": "It also softens the pain of the cancelled trip.",
                    "emotion_vector": [0.08, 0.01, 0.10, 0.02, 0.0, 0.12, 0.01, 0.18],
                    "emotion_vector_confidence": 0.8,
                },
            ]
        }
    )
    turn = SummaryTurn(
        speaker="SPEAKER_00",
        text="Midnight Sun is exciting, but it softens the pain of the cancelled trip.",
        emo_preset="engaged",
    )

    result = SummaryTurnEmotionLinker(
        SummaryTurnEmotionLinkingConfig(
            top_k=2,
            min_support_score=0.0,
            apply_indextts_bias=False,
        )
    ).link_turns(turns=[turn], transcript=transcript)

    assert result.linked_turn_count == 1
    linked_turn = result.turns[0]
    assert linked_turn.emotion_vector_source == "summary_turn_linking"
    assert set(linked_turn.source_segment_indices) == {0, 1}
    assert abs(sum(linked_turn.source_segment_weights) - 1.0) < 1e-9
    assert linked_turn.emotion_vector is not None
    happy, angry, sad, afraid, disgusted, melancholic, surprised, calm = (
        linked_turn.emotion_vector.to_list()
    )
    assert happy > sad
    assert sad > angry
    assert calm > surprised


def test_linker_prefers_same_speaker_support() -> None:
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "We are thrilled about Midnight Sun today.",
                    "emotion_vector": [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.08],
                    "emotion_vector_confidence": 0.8,
                },
                {
                    "speaker": "SPEAKER_01",
                    "start_time": 1000,
                    "end_time": 2000,
                    "text": "We are thrilled about Midnight Sun today.",
                    "emotion_vector": [0.04, 0.18, 0.02, 0.14, 0.0, 0.0, 0.04, 0.02],
                    "emotion_vector_confidence": 0.8,
                },
            ]
        }
    )
    turn = SummaryTurn(
        speaker="SPEAKER_00",
        text="We are thrilled about Midnight Sun today.",
        emo_preset="engaged",
    )

    result = SummaryTurnEmotionLinker(
        SummaryTurnEmotionLinkingConfig(
            top_k=1,
            min_support_score=0.0,
            same_speaker_bonus=2.0,
            apply_indextts_bias=False,
        )
    ).link_turns(turns=[turn], transcript=transcript)

    assert result.turns[0].source_segment_indices == (0,)
    assert result.turns[0].emotion_vector == EmotionVector.from_raw(
        [0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.08]
    )


def test_linker_uses_declared_support_when_present() -> None:
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "ignored",
                    "emotion_vector": [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.08],
                },
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 1000,
                    "end_time": 2000,
                    "text": "selected",
                    "emotion_vector": [0.05, 0.0, 0.12, 0.0, 0.0, 0.14, 0.01, 0.18],
                },
            ]
        }
    )
    turn = SummaryTurn(
        speaker="SPEAKER_00",
        text="custom support turn",
        emo_preset="warm",
        source_segment_indices=(1,),
        source_segment_weights=(1.0,),
    )

    result = SummaryTurnEmotionLinker(
        SummaryTurnEmotionLinkingConfig(
            top_k=1,
            min_support_score=1.0,
            apply_indextts_bias=False,
        )
    ).link_turns(turns=[turn], transcript=transcript)

    assert result.used_declared_support_count == 1
    assert result.turns[0].source_segment_indices == (1,)
    assert result.turns[0].emotion_vector == EmotionVector.from_raw(
        [0.05, 0.0, 0.12, 0.0, 0.0, 0.14, 0.01, 0.18]
    )


def test_linker_does_not_link_on_position_similarity_alone() -> None:
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "completely unrelated opening chatter",
                    "emotion_vector": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.08],
                }
            ]
        }
    )
    turn = SummaryTurn(
        speaker="SPEAKER_00",
        text="Midnight Sun countdown reaction",
        emo_preset="engaged",
    )

    result = SummaryTurnEmotionLinker(
        SummaryTurnEmotionLinkingConfig(top_k=1, min_support_score=0.0)
    ).link_turns(turns=[turn], transcript=transcript)

    assert result.linked_turn_count == 0
    assert result.turns[0].emotion_vector is None
