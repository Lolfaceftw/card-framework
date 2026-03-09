"""Normalization safety tests for DeepSeek provider message handling."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from card_framework.providers.deepseek_provider import DeepSeekProvider


def _tool_call(tool_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def test_normalize_messages_is_non_mutating_and_idempotent_for_tool_calls() -> None:
    provider = DeepSeekProvider(api_key="test")
    call_a = _tool_call(
        "call_a",
        "add_speaker_message",
        {"speaker_id": "SPEAKER_00", "content": "alpha"},
    )
    call_b = _tool_call("call_b", "count_words", {})

    messages = [
        {"role": "assistant", "content": "part 1", "tool_calls": [call_a]},
        {"role": "assistant", "content": "part 2", "tool_calls": [call_a, call_b]},
        {
            "role": "tool",
            "tool_call_id": "call_a",
            "name": "add_speaker_message",
            "content": '{"status":"ok"}',
        },
        {
            "role": "tool",
            "tool_call_id": "call_b",
            "name": "count_words",
            "content": '{"total_word_count":1}',
        },
    ]
    original_messages = deepcopy(messages)

    normalized_once = provider._normalize_messages(messages)
    normalized_twice = provider._normalize_messages(messages)

    assert messages == original_messages

    assistant_once = [m for m in normalized_once if m.get("role") == "assistant"]
    assistant_twice = [m for m in normalized_twice if m.get("role") == "assistant"]
    assert len(assistant_once) == 1
    assert len(assistant_twice) == 1

    tool_call_ids_once = [tc["id"] for tc in assistant_once[0].get("tool_calls", [])]
    tool_call_ids_twice = [tc["id"] for tc in assistant_twice[0].get("tool_calls", [])]
    assert tool_call_ids_once == ["call_a", "call_b"]
    assert tool_call_ids_twice == ["call_a", "call_b"]


def test_normalize_messages_prunes_dangling_tool_calls() -> None:
    provider = DeepSeekProvider(api_key="test")
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_x", "add_speaker_message", {"speaker_id": "SPEAKER_00"})],
        }
    ]

    normalized = provider._normalize_messages(messages)

    assert len(normalized) == 1
    assert normalized[0]["role"] == "assistant"
    assert "tool_calls" not in normalized[0]
    assert normalized[0]["content"] == "[Tool call skipped or invalid]"
