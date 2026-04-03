"""Coverage for the duration-aware summarizer tool dispatcher."""

from __future__ import annotations

import asyncio

from card_framework.agents.loop.summarizer_tool_dispatcher import (
    _DRAFT_REVIEW_PROMPT,
    SummarizerToolDispatcher,
)
from card_framework.shared.events import EventBus


class _DummyToolRegistry:
    """Minimal async tool registry used to exercise dispatcher flows."""

    def __init__(self, results: dict[str, dict[str, object]]) -> None:
        self._results = results

    async def dispatch(self, name: str, _args: dict[str, object]) -> dict[str, object]:
        return dict(self._results.get(name, {}))


def test_process_tool_calls_emits_auto_feedback_as_user_messages() -> None:
    dispatcher = SummarizerToolDispatcher(agent_name="Summarizer", event_bus=EventBus())
    messages: list[dict[str, object]] = []
    context_data = {
        "tool_registry": _DummyToolRegistry(
            {
                "add_speaker_message": {"line": 1, "speaker_id": "SPEAKER_00"},
                "estimate_duration": {
                    "total_estimated_seconds": 33.2,
                    "budget": {"in_budget": False},
                },
                "count_words": {"total_word_count": 83},
            }
        ),
        "target_seconds": 300,
        "duration_tolerance_ratio": 0.05,
    }

    should_break, final_result = asyncio.run(
        dispatcher.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "add_speaker_message",
                    "arguments": {"speaker_id": "SPEAKER_00", "content": "hello"},
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )

    assert should_break is False
    assert final_result is None
    assert messages[0] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "name": "add_speaker_message",
        "content": '{"line": 1, "speaker_id": "SPEAKER_00"}',
    }
    assert messages[1]["role"] == "user"
    assert str(messages[1]["content"]).startswith(
        "[AUTO_TOOL_RESULT:estimate_duration]\n"
    )
    assert messages[2]["role"] == "user"
    assert str(messages[2]["content"]).startswith("[AUTO_TOOL_RESULT:count_words]\n")


def test_process_tool_calls_emits_auto_save_feedback_as_user_message() -> None:
    dispatcher = SummarizerToolDispatcher(agent_name="Summarizer", event_bus=EventBus())
    messages: list[dict[str, object]] = []
    context_data = {
        "tool_registry": _DummyToolRegistry(
            {
                "add_speaker_message": {"line": 1, "speaker_id": "SPEAKER_00"},
                "estimate_duration": {
                    "total_estimated_seconds": 300.0,
                    "budget": {
                        "in_budget": True,
                        "min_seconds": 285.0,
                        "max_seconds": 315.0,
                    },
                },
                "count_words": {"total_word_count": 750},
                "save_draft": {"total_messages": 3},
            }
        ),
        "target_seconds": 300,
        "duration_tolerance_ratio": 0.05,
    }

    should_break, final_result = asyncio.run(
        dispatcher.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "add_speaker_message",
                    "arguments": {"speaker_id": "SPEAKER_00", "content": "hello"},
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )

    assert should_break is False
    assert final_result is None
    assert any(
        message["role"] == "user"
        and str(message["content"]).startswith("[AUTO_TOOL_RESULT:save_draft]\n")
        for message in messages
    )
    assert messages[-1] == {"role": "user", "content": _DRAFT_REVIEW_PROMPT}
