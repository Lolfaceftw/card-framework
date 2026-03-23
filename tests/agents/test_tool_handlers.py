"""Tests for summarizer tool handlers."""

from __future__ import annotations

import asyncio

from card_framework.agents.message_registry import MessageRegistry
from card_framework.agents.tool_handlers import AddSpeakerMessageHandler, BudgetContext


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
