from __future__ import annotations

from card_framework.audio_pipeline.summary_turn_emotion_linker import (
    SummaryTurnEmotionLinker,
    SummaryTurnEmotionLinkingConfig,
)
from card_framework.audio_pipeline.turn_emotion_resolution import (
    SourceLinkedTurnEmotionResolutionStrategy,
    TurnEmotionResolutionConfig,
)
from card_framework.orchestration.transcript import Transcript
from card_framework.shared.emotion_vector import EmotionVector
from card_framework.shared.summary_xml import SummaryTurn


def _build_transcript() -> Transcript:
    return Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 1000,
                    "text": "Midnight Sun is joyful exciting news.",
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


def _build_linker() -> SummaryTurnEmotionLinker:
    return SummaryTurnEmotionLinker(
        SummaryTurnEmotionLinkingConfig(
            top_k=2,
            min_support_score=0.0,
            apply_indextts_bias=False,
        )
    )


def test_resolution_strategy_prefers_source_linked_vectors() -> None:
    strategy = SourceLinkedTurnEmotionResolutionStrategy(
        config=TurnEmotionResolutionConfig(),
        linker=_build_linker(),
        emo_preset_vector_catalog={
            "engaged": EmotionVector.from_raw(
                [0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06, 0.20]
            )
        },
    )

    resolved = strategy.resolve_turn(
        turn=SummaryTurn(
            speaker="SPEAKER_00",
            text="Midnight Sun is exciting, but it softens the pain of the cancelled trip.",
            emo_preset="engaged",
        ),
        transcript=_build_transcript(),
    )

    assert resolved.emotion_vector is not None
    assert resolved.emotion_vector_source == "summary_turn_linking"
    assert set(resolved.source_segment_indices) == {0, 1}


def test_resolution_strategy_falls_back_to_explicit_then_preset_by_default() -> None:
    strategy = SourceLinkedTurnEmotionResolutionStrategy(
        config=TurnEmotionResolutionConfig(prelink_enabled=False),
        linker=None,
        emo_preset_vector_catalog={
            "warm": EmotionVector.from_raw([0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24])
        },
    )
    transcript = Transcript.from_mapping({"segments": []})

    explicit = strategy.resolve_turn(
        turn=SummaryTurn(
            speaker="SPEAKER_00",
            text="Explicit vector turn.",
            emo_preset="warm",
            emotion_vector=[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.1],
            emotion_vector_source="manual_override",
        ),
        transcript=transcript,
    )
    assert explicit.emotion_vector_source == "manual_override"

    preset = strategy.resolve_turn(
        turn=SummaryTurn(
            speaker="SPEAKER_00",
            text="Preset fallback turn.",
            emo_preset="warm",
        ),
        transcript=transcript,
    )
    assert preset.emotion_vector == EmotionVector.from_raw(
        [0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24]
    )
    assert preset.emotion_vector_source == "preset_vector"

    no_vector = SourceLinkedTurnEmotionResolutionStrategy(
        config=TurnEmotionResolutionConfig(prelink_enabled=False),
        linker=None,
        emo_preset_vector_catalog={},
    ).resolve_turn(
        turn=SummaryTurn(
            speaker="SPEAKER_00",
            text="no vector turn.",
            emo_preset="neutral",
        ),
        transcript=transcript,
    )
    assert no_vector.emotion_vector is None
    assert no_vector.should_use_emo_text_fallback is False


def test_resolution_strategy_allows_opt_in_emo_text_fallback() -> None:
    transcript = Transcript.from_mapping({"segments": []})
    emo_text = SourceLinkedTurnEmotionResolutionStrategy(
        config=TurnEmotionResolutionConfig(
            prelink_enabled=False,
            fallback_order=("explicit", "preset", "emo_text"),
        ),
        linker=None,
        emo_preset_vector_catalog={},
    ).resolve_turn(
        turn=SummaryTurn(
            speaker="SPEAKER_00",
            text="emo text fallback turn.",
            emo_preset="neutral",
        ),
        transcript=transcript,
    )
    assert emo_text.emotion_vector is None
    assert emo_text.should_use_emo_text_fallback is True


def test_resolution_strategy_uses_default_preset_vector_for_unknown_preset() -> None:
    strategy = SourceLinkedTurnEmotionResolutionStrategy(
        config=TurnEmotionResolutionConfig(prelink_enabled=False),
        linker=None,
        emo_preset_vector_catalog={
            "neutral": EmotionVector.from_raw(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24]
            )
        },
    )

    resolved = strategy.resolve_turn(
        turn=SummaryTurn(
            speaker="SPEAKER_00",
            text="Unknown preset turn.",
            emo_preset="mystery",
        ),
        transcript=Transcript.from_mapping({"segments": []}),
    )

    assert resolved.emotion_vector == EmotionVector.from_raw(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24]
    )
    assert resolved.emotion_vector_source == "preset_vector"
