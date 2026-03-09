"""Exercise DeepSeek streaming through a real OpenAI-compatible endpoint."""

from __future__ import annotations

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
