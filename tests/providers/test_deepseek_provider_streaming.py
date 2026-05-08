"""Exercise DeepSeek streaming through a real OpenAI-compatible endpoint."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from card_framework.providers.deepseek_provider import DeepSeekProvider
from tests.support.servers import run_openai_compatible_server


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


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Build one streamed chat-completion chunk for the test server."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content

    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def test_chat_streams_reasoning_and_content_through_response_callback(
    free_tcp_port: int,
) -> None:
    """Forward real streamed reasoning and content tokens through the callback."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content="plan"),
                _chunk(content="answer"),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = DeepSeekProvider(
            api_key="test",
            base_url=f"{server.base_url}/v1",
        )
        callback = _RecordingCallback()
        provider.set_response_callback(callback)

        dumped = provider.chat(
            messages=[{"role": "system", "content": "Summarizer"}]
        ).model_dump()

        provider._client.close()

    assert server.chat_requests[-1]["stream"] is True
    assert callback.events == [
        ("start", "Summarizer"),
        ("thought", "plan"),
        ("content", "answer"),
        ("complete", ""),
    ]
    assert dumped["content"] == "answer"
    assert dumped["reasoning_content"] == "plan"


def test_generate_applies_deepseek_judge_request_controls(
    free_tcp_port: int,
) -> None:
    """Propagate non-thinking judge controls and default max tokens into requests."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(content='{"Factualness": 1.0}'),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = DeepSeekProvider(
            api_key="test",
            base_url=f"{server.base_url}/v1",
            model="deepseek-v4-pro",
            default_max_tokens=384_000,
            thinking_mode="disabled",
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        result = provider.generate("Judge system", "Judge user")

        provider._client.close()

    request = server.chat_requests[-1]
    assert result == '{"Factualness": 1.0}'
    assert request["max_tokens"] == 384_000
    assert request["temperature"] == 0.0
    assert request["response_format"] == {"type": "json_object"}
    assert request["thinking"] == {"type": "disabled"}
    assert provider.last_response_details["request"]["max_tokens"] == 384_000
    assert provider.last_response_details["request"]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


def test_generate_can_use_non_streaming_deepseek_judge_mode() -> None:
    """Match the non-streaming DeepSeek judge path used in human-alignment runs."""
    captured_kwargs: dict[str, Any] = {}
    provider = DeepSeekProvider(
        api_key="test",
        model="deepseek-v4-pro",
        default_max_tokens=384_000,
        thinking_mode="disabled",
        response_format={"type": "json_object"},
        stream=False,
    )

    def _fake_create(**kwargs: Any) -> object:
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"Factualness": 0.75}',
                        reasoning_content="",
                    )
                )
            ]
        )

    provider._client.chat.completions.create = _fake_create  # type: ignore[method-assign]

    result = provider.generate("Judge system", "Judge user")

    provider._client.close()

    assert result == '{"Factualness": 0.75}'
    assert captured_kwargs["stream"] is False
    assert captured_kwargs["max_tokens"] == 384_000
    assert captured_kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert provider.last_response_details["request"]["stream"] is False
