"""Tests for summarizer tool handlers."""

from __future__ import annotations

import asyncio

from card_framework.agents.loop.message_registry import MessageRegistry
from card_framework.agents.loop.tool_handlers import (
    AddSpeakerMessageHandler,
    BudgetContext,
    EditMessageHandler,
)
from card_framework.audio_pipeline.emotion.summary_turn_emotion_linker import (
    SummaryTurnEmotionLinker,
    SummaryTurnEmotionLinkingConfig,
)
from card_framework.audio_pipeline.emotion.turn_emotion_resolution import (
    SourceLinkedTurnEmotionResolutionStrategy,
    TurnEmotionResolutionConfig,
)
from card_framework.orchestration.transcript import Transcript


class _FakeCalibration:
    def estimate_turn_seconds(self, turn) -> float:
        del turn
        return 1.0


def test_add_speaker_message_schema_exposes_available_speakers() -> None:
    """Advertise the allowed speaker IDs when a manifest constrains them."""
    handler = AddSpeakerMessageHandler(
        MessageRegistry(),
        BudgetContext(
            MessageRegistry(),
            calibration=None,
            target_seconds=30,
            duration_tolerance_ratio=0.05,
        ),
        allowed_speaker_ids=("SPEAKER_00",),
    )

    speaker_id_schema = handler.schema["parameters"]["properties"]["speaker_id"]

    assert speaker_id_schema["enum"] == ["SPEAKER_00"]
    assert handler.schema["parameters"]["required"] == ["speaker_id", "content"]
    assert "emo_preset" not in handler.schema["parameters"]["properties"]


def test_edit_message_schema_exposes_vector_only_controls() -> None:
    registry = MessageRegistry()
    handler = EditMessageHandler(
        registry,
        BudgetContext(
            registry,
            calibration=None,
            target_seconds=30,
            duration_tolerance_ratio=0.05,
        ),
    )

    assert handler.schema["parameters"]["required"] == ["line", "new_content"]
    assert "emo_preset" not in handler.schema["parameters"]["properties"]


def test_add_speaker_message_rejects_unknown_speaker_id() -> None:
    """Reject speaker IDs that are not present in the available manifest set."""
    registry = MessageRegistry()
    handler = AddSpeakerMessageHandler(
        registry,
        BudgetContext(
            registry,
            calibration=None,
            target_seconds=30,
            duration_tolerance_ratio=0.05,
        ),
        allowed_speaker_ids=("SPEAKER_00",),
    )

    result = asyncio.run(
        handler.execute(
            {
                "speaker_id": "SPEAKER_01",
                "content": "Hello.",
                "emo_preset": "neutral",
            }
        )
    )

    assert result["error_code"] == "unknown_speaker_id"
    assert result["available_speaker_ids"] == ["SPEAKER_00"]


def test_add_speaker_message_accepts_direct_emotion_vector() -> None:
    registry = MessageRegistry()
    handler = AddSpeakerMessageHandler(
        registry,
        BudgetContext(
            registry,
            calibration=_FakeCalibration(),
            target_seconds=30,
            duration_tolerance_ratio=0.05,
        ),
    )

    result = asyncio.run(
        handler.execute(
            {
                "speaker_id": "SPEAKER_00",
                "content": "Hello.",
                "emo_preset": "neutral",
                "emotion_vector": [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3],
            }
        )
    )

    assert result["emotion_vector"] == [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]


def test_add_speaker_message_prelinks_source_grounded_vector() -> None:
    registry = MessageRegistry()
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "Midnight Sun is exciting and joyful news.",
                    "start_time": 0,
                    "end_time": 1000,
                    "emotion_vector": [0.36, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.06],
                }
            ]
        }
    )
    handler = AddSpeakerMessageHandler(
        registry,
        BudgetContext(
            registry,
            calibration=_FakeCalibration(),
            target_seconds=30,
            duration_tolerance_ratio=0.05,
        ),
        turn_emotion_resolution_strategy=SourceLinkedTurnEmotionResolutionStrategy(
            config=TurnEmotionResolutionConfig(),
            linker=SummaryTurnEmotionLinker(
                SummaryTurnEmotionLinkingConfig(
                    top_k=1,
                    min_support_score=0.0,
                    apply_indextts_bias=False,
                )
            ),
        ),
        transcript=transcript,
    )

    result = asyncio.run(
        handler.execute(
            {
                "speaker_id": "SPEAKER_00",
                "content": "Midnight Sun is exciting and joyful news.",
                "emo_preset": "engaged",
            }
        )
    )

    assert result["emotion_vector_source"] == "summary_turn_linking"
    assert result["source_segment_indices"] == [0]
    assert result["source_segment_weights"] == [1.0]

