"""Coverage for the shared Anthropic provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from card_framework.providers import anthropic_provider as anthropic_provider_module
from card_framework.providers.anthropic_provider import AnthropicProvider


class _RecordingCallback:
    """Capture callback events emitted during streamed response handling."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def on_start(self, agent_name: str) -> None:
        self.events.append(("start", agent_name))

    def on_thought_token(self, token: str) -> None:
        self.events.append(("thought", token))

    def on_content_token(self, token: str) -> None:
        self.events.append(("content", token))

    def on_complete(self) -> None:
        self.events.append(("complete", ""))


class _FakeAnthropicClient:
    """Fake Anthropic client used to unit test provider behavior."""

    def __init__(
        self,
        *,
        requests: list[dict[str, Any]],
        create_impl,
        api_key: str,
        timeout: float,
    ) -> None:
        del api_key, timeout
        self._requests = requests
        self._create_impl = create_impl
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self._requests.append(dict(kwargs))
        return self._create_impl(**kwargs)

    def close(self) -> None:
        return None


def _install_fake_anthropic(monkeypatch, create_impl):  # type: ignore[no-untyped-def]
    requests: list[dict[str, Any]] = []

    def _factory(*, api_key: str, timeout: float) -> _FakeAnthropicClient:
        return _FakeAnthropicClient(
            requests=requests,
            create_impl=create_impl,
            api_key=api_key,
            timeout=timeout,
        )

    monkeypatch.setattr(
        anthropic_provider_module,
        "anthropic",
        SimpleNamespace(Anthropic=_factory),
    )
    return requests


def test_chat_streams_thinking_content_and_tool_calls(monkeypatch) -> None:
    """Stream Anthropic thought/content blocks and tool_use back into DTOs."""

    def _create_impl(**kwargs: Any) -> list[object]:
        assert kwargs["stream"] is True
        return [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(
                    id="msg_1",
                    role="assistant",
                    stop_reason="",
                    usage={"input_tokens": 10, "output_tokens": 5},
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="thinking_delta", thinking="plan"),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=0,
                delta=SimpleNamespace(type="text_delta", text="answer"),
            ),
            SimpleNamespace(
                type="content_block_start",
                index=1,
                content_block=SimpleNamespace(
                    type="tool_use",
                    id="call_1",
                    name="submit_samples",
                    input={},
                ),
            ),
            SimpleNamespace(
                type="content_block_delta",
                index=1,
                delta=SimpleNamespace(
                    type="input_json_delta",
                    partial_json='{"episode_key":"episode://sample"}',
                ),
            ),
            SimpleNamespace(
                type="message_delta",
                delta=SimpleNamespace(stop_reason="tool_use"),
                usage={"output_tokens": 7},
            ),
        ]

    requests = _install_fake_anthropic(monkeypatch, _create_impl)
    provider = AnthropicProvider(api_key="test-key")
    callback = _RecordingCallback()
    provider.set_response_callback(callback)

    response = provider.chat(
        messages=[{"role": "system", "content": "Summarizer"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "submit_samples",
                    "description": "Return samples.",
                    "parameters": {"type": "object"},
                    "strict": True,
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "submit_samples"}},
    )

    assert callback.events == [
        ("start", "Summarizer"),
        ("thought", "plan"),
        ("content", "answer"),
        ("content", "\nTOOL submit_samples: "),
        ("content", '{"episode_key":"episode://sample"}'),
        ("complete", ""),
    ]
    assert response.content == "answer"
    assert response.reasoning_content == "plan"
    assert response.tool_calls is not None
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].function.name == "submit_samples"
    assert (
        response.tool_calls[0].function.arguments
        == '{"episode_key":"episode://sample"}'
    )
    assert requests[0]["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert requests[0]["output_config"] == {"effort": "medium"}
    assert "temperature" not in requests[0]
    assert requests[0]["tool_choice"] == {"type": "auto"}
    assert provider.last_response_details["response"]["stop_reason"] == "tool_use"
    provider._client.close()


def test_chat_falls_back_to_non_streaming_response(monkeypatch) -> None:
    """Retry a failed stream as a non-streaming Anthropic request."""

    def _create_impl(**kwargs: Any) -> object:
        if kwargs.get("stream"):
            raise RuntimeError("stream failed")
        return SimpleNamespace(
            id="msg_2",
            role="assistant",
            model="claude-opus-4-6",
            stop_reason="tool_use",
            usage={"input_tokens": 8, "output_tokens": 3},
            content=[
                SimpleNamespace(type="thinking", thinking="reason"),
                SimpleNamespace(type="text", text="answer"),
                SimpleNamespace(
                    type="tool_use",
                    id="call_2",
                    name="submit_samples",
                    input={"episode_key": "episode://sample"},
                ),
            ],
        )

    _install_fake_anthropic(monkeypatch, _create_impl)
    provider = AnthropicProvider(api_key="test-key")

    response = provider.chat(messages=[{"role": "system", "content": "Summarizer"}])

    assert response.content == "answer"
    assert response.reasoning_content == "reason"
    assert response.tool_calls is not None
    assert response.tool_calls[0].id == "call_2"
    assert response.tool_calls[0].function.name == "submit_samples"
    assert (
        response.tool_calls[0].function.arguments
        == '{"episode_key": "episode://sample"}'
    )
    assert provider.last_response_details["response"]["stop_reason"] == "tool_use"


def test_normalize_messages_keeps_auto_tool_feedback_as_text_not_tool_result() -> None:
    """Auto helper feedback must not become synthetic Anthropic tool_result blocks."""
    provider = AnthropicProvider(api_key="test-key")

    _system, normalized_messages = provider._normalize_messages(
        [
            {"role": "system", "content": "Summarizer"},
            {
                "role": "assistant",
                "content": "draft",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "add_speaker_message",
                            "arguments": '{"speaker_id":"SPEAKER_00","content":"hello"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "add_speaker_message",
                "tool_call_id": "call_1",
                "content": '{"line":1}',
            },
            {
                "role": "user",
                "content": (
                    "[AUTO_TOOL_RESULT:estimate_duration]\n"
                    "This helper was run automatically after your mutation. "
                    "Review the JSON result and continue from it.\n"
                    '{"total_estimated_seconds": 33.2}'
                ),
            },
        ]
    )

    assert normalized_messages == [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "draft"},
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "add_speaker_message",
                    "input": {"speaker_id": "SPEAKER_00", "content": "hello"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": '{"line":1}',
                },
                {
                    "type": "text",
                    "text": (
                        "[AUTO_TOOL_RESULT:estimate_duration]\n"
                        "This helper was run automatically after your mutation. "
                        "Review the JSON result and continue from it.\n"
                        '{"total_estimated_seconds": 33.2}'
                    ),
                },
            ],
        },
    ]


def test_provider_requires_installed_sdk(monkeypatch) -> None:
    """Fail fast when the Anthropic SDK is unavailable."""
    monkeypatch.setattr(anthropic_provider_module, "anthropic", None)

    with pytest.raises(RuntimeError, match="Anthropic support requires"):
        AnthropicProvider(api_key="test-key")
