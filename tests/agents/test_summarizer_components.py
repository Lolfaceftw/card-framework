from __future__ import annotations

import asyncio
import json
from typing import Any

from agents.summarizer_loop_controller import SummarizerLoopController
from agents.summarizer_tool_dispatcher import SummarizerToolDispatcher
from events import create_event_bus


class FakeToolRegistry:
    """Minimal async tool registry for summarizer component tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._line_count = 0

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        self.calls.append((name, dict(arguments)))
        if name == "add_speaker_message":
            self._line_count += 1
            return {"status": "added"}
        if name == "count_words":
            return {"total_word_count": 12}
        if name == "save_draft":
            return {"total_messages": self._line_count}
        return {"status": "ok"}

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": "noop"}}]


def test_loop_controller_builds_context_and_invokes_run_loop() -> None:
    event_bus = create_event_bus()
    captured: dict[str, Any] = {}

    async def fake_run_loop(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_turns: int,
        context_data: dict[str, Any],
    ) -> dict[str, Any] | None:
        captured["messages"] = messages
        captured["tools"] = tools
        captured["max_turns"] = max_turns
        captured["context_data"] = context_data
        return None

    controller = SummarizerLoopController(
        run_agent_loop=fake_run_loop,
        max_tool_turns=3,
        is_embedding_enabled=True,
        event_bus=event_bus,
    )

    registry = FakeToolRegistry()
    messages = [{"role": "system", "content": "x"}]
    guardrails = {
        "enable_extended_text_tool_parser": True,
        "enable_stall_guidance": True,
        "enable_noop_edit_detection": True,
        "stall_guidance_threshold_turns": 3,
        "stall_guidance_cooldown_turns": 2,
        "provider_name": "ProviderX",
        "model_id": "ModelY",
        "enable_staged_discovery": True,
        "required_discovery_queries": 2,
        "max_discovery_queries": 4,
        "require_unique_discovery_queries": True,
    }

    context_data = asyncio.run(
        controller.run(
            messages=messages,
            tool_registry=registry,
            min_words=100,
            max_words=200,
            loop_guardrails=guardrails,
        )
    )

    assert captured["messages"] is messages
    assert captured["tools"] == registry.get_tool_schemas()
    assert captured["max_turns"] == 3
    assert captured["context_data"] is context_data
    assert context_data["min_words"] == 100
    assert context_data["max_words"] == 200
    assert context_data["enable_staged_discovery"] is True
    assert context_data["required_discovery_queries"] == 2


def test_tool_dispatcher_executes_only_first_mutating_call_per_turn() -> None:
    dispatcher = SummarizerToolDispatcher(
        agent_name="Summarizer",
        event_bus=create_event_bus(),
    )
    registry = FakeToolRegistry()

    context_data: dict[str, Any] = {
        "tool_registry": registry,
        "min_words": 1,
        "max_words": 100,
        "signature_dedupe_window_turns": 1,
        "replay_dedupe_tools": {
            "add_speaker_message",
            "edit_message",
            "remove_message",
            "finalize_draft",
        },
    }
    messages: list[dict[str, Any]] = [{"role": "system", "content": "Summarizer"}]

    asyncio.run(
        dispatcher.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "add_speaker_message",
                    "arguments": {"speaker_id": "SPEAKER_00", "content": "First line"},
                },
                {
                    "id": "call_2",
                    "name": "add_speaker_message",
                    "arguments": {"speaker_id": "SPEAKER_01", "content": "Second line"},
                },
            ],
            messages=messages,
            context_data=context_data,
        )
    )

    called_tools = [name for name, _ in registry.calls]
    assert called_tools.count("add_speaker_message") == 1
    assert called_tools.count("count_words") == 1
    assert called_tools.count("save_draft") == 1

    skipped = [
        message
        for message in messages
        if message.get("role") == "tool"
        and message.get("tool_call_id") == "call_2"
        and message.get("name") == "add_speaker_message"
    ]
    assert len(skipped) == 1
    skipped_payload = json.loads(str(skipped[0]["content"]))
    assert skipped_payload["status"] == "skipped"
    assert skipped_payload["reason"] == "single_mutating_call_per_turn"
