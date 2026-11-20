"""Exercise the vLLM provider against a real OpenAI-compatible stream."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from card_framework.providers.vllm_provider import VLLMProvider
from card_framework.shared.llm_provider import StreamingResponseAbort
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


class _AbortOnContentCallback:
    """Abort one streamed response as soon as content arrives."""

    def on_start(self, agent_name: str) -> None:
        del agent_name

    def on_thought_token(self, token: str) -> None:
        del token

    def on_content_token(self, token: str) -> None:
        del token
        raise StreamingResponseAbort(reason="test stream abort")

    def on_complete(self) -> None:
        return None


def _chunk(
    *,
    content: str | None = None,
    reasoning_content: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Build one OpenAI chat-completion chunk payload."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls

    return {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "fake-model",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }


def _tool_call_delta(
    *,
    index: int,
    id: str | None = None,
    name: str | None = None,
    arguments: str | None = None,
) -> dict[str, Any]:
    """Build one streamed tool-call delta payload."""
    function: dict[str, Any] = {}
    if name is not None:
        function["name"] = name
    if arguments is not None:
        function["arguments"] = arguments

    payload: dict[str, Any] = {"index": index}
    if id is not None:
        payload["id"] = id
    if function:
        payload["function"] = function
    return payload


def _build_provider(
    base_url: str,
    *,
    enable_thinking: bool = True,
    fallback_to_reasoning_content: bool = True,
) -> VLLMProvider:
    """Create a provider pointed at the local OpenAI-compatible test server."""
    return VLLMProvider(
        base_url=f"{base_url}/v1",
        api_key="EMPTY",
        enable_thinking=enable_thinking,
        fallback_to_reasoning_content=fallback_to_reasoning_content,
    )


def test_generate_requests_thinking_mode(free_tcp_port: int) -> None:
    """Send the real reasoning-mode request body to a local vLLM-style server."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content="thinking"),
                _chunk(content="answer"),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = _build_provider(server.base_url)

        result = provider.generate(
            system_prompt="sys",
            user_prompt="user",
            max_tokens=32,
        )

        provider._client.close()

    assert result == "answer"
    assert server.chat_requests[-1]["chat_template_kwargs"] == {"enable_thinking": True}


def test_generate_uses_judge_name_for_evaluator_prompts(free_tcp_port: int) -> None:
    """Label evaluator-style generate streams as Judge for the terminal UI."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content="score it"),
                _chunk(content='{"Factualness": 0.8}'),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = _build_provider(server.base_url)
        callback = _RecordingCallback()
        provider.set_response_callback(callback)

        result = provider.generate(
            system_prompt=(
                "You are a strict podcast summarization evaluator. Return JSON only."
            ),
            user_prompt="Evaluate this summary.",
        )

        provider._client.close()

    assert result == '{"Factualness": 0.8}'
    assert callback.events == [
        ("start", "Judge"),
        ("thought", "score it"),
        ("content", '{"Factualness": 0.8}'),
        ("complete", ""),
    ]


def test_chat_returns_reasoning_content_field(free_tcp_port: int) -> None:
    """Capture both content and reasoning from the streamed OpenAI response."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content="plan"),
                _chunk(content="done"),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = _build_provider(server.base_url)

        dumped = provider.chat(
            messages=[{"role": "system", "content": "Summarizer"}]
        ).model_dump()

        provider._client.close()

    assert dumped["content"] == "done"
    assert dumped["reasoning_content"] == "plan"
    assert "reasoning" not in dumped


def test_chat_can_disable_thinking_mode(free_tcp_port: int) -> None:
    """Omit the extra reasoning body when thinking mode is disabled."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream([_chunk(content="ok"), _chunk(finish_reason="stop")])
        provider = _build_provider(server.base_url, enable_thinking=False)

        provider.chat(messages=[{"role": "system", "content": "Critic"}])

        provider._client.close()

    assert "chat_template_kwargs" not in server.chat_requests[-1]


def test_generate_accepts_omegaconf_extra_body(free_tcp_port: int) -> None:
    """Serialize OmegaConf-backed extra bodies before sending the request."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [_chunk(content='{"ok":true}'), _chunk(finish_reason="stop")]
        )
        provider = VLLMProvider(
            base_url=f"{server.base_url}/v1",
            api_key="EMPTY",
            enable_thinking=True,
            thinking_extra_body=cast(
                Any,
                OmegaConf.create({"chat_template_kwargs": {"enable_thinking": False}}),
            ),
        )

        result = provider.generate(system_prompt="sys", user_prompt="user")

        provider._client.close()

    assert result == '{"ok":true}'
    assert server.chat_requests[-1]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


def test_generate_falls_back_to_non_streaming_on_stream_error(
    free_tcp_port: int,
) -> None:
    """Retry as a non-streaming request when the streaming API path fails."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        provider = _build_provider(server.base_url, enable_thinking=False)

        def _fake_create(**kwargs: Any) -> object:
            if kwargs.get("stream"):
                raise RuntimeError("stream failed")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="fallback answer", reasoning_content=None
                        )
                    )
                ]
            )

        provider._client.chat.completions.create = _fake_create  # type: ignore[method-assign]

        result = provider.generate(system_prompt="sys", user_prompt="user")

        provider._client.close()

    assert result == "fallback answer"


def test_chat_falls_back_to_non_streaming_on_stream_error(free_tcp_port: int) -> None:
    """Retry chat completions without streaming when the stream path fails."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        provider = _build_provider(server.base_url, enable_thinking=False)

        def _fake_create(**kwargs: Any) -> object:
            if kwargs.get("stream"):
                raise RuntimeError("stream failed")
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content="fallback chat answer",
                            reasoning_content=None,
                            tool_calls=None,
                        )
                    )
                ]
            )

        provider._client.chat.completions.create = _fake_create  # type: ignore[method-assign]

        dumped = provider.chat(
            messages=[{"role": "system", "content": "Summarizer"}]
        ).model_dump()

        provider._client.close()

    assert dumped["content"] == "fallback chat answer"


def test_chat_reraises_stream_abort_without_non_streaming_fallback(
    free_tcp_port: int,
) -> None:
    """Propagate controlled callback aborts instead of retrying non-streaming."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream([_chunk(content="loop"), _chunk(finish_reason="stop")])
        provider = _build_provider(server.base_url, enable_thinking=False)
        provider.set_response_callback(_AbortOnContentCallback())

        with pytest.raises(StreamingResponseAbort, match="test stream abort"):
            provider.chat(messages=[{"role": "system", "content": "Critic"}])

        provider._client.close()

    assert len(server.chat_requests) == 1


def test_generate_falls_back_to_reasoning_when_content_empty(
    free_tcp_port: int,
) -> None:
    """Use structured reasoning content when the content channel stays empty."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content='{"ok":true}'),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = _build_provider(server.base_url)

        result = provider.generate(system_prompt="sys", user_prompt="user")

        provider._client.close()

    assert result == '{"ok":true}'


def test_chat_falls_back_to_reasoning_when_content_empty(
    free_tcp_port: int,
) -> None:
    """Mirror the generate fallback behavior for chat completions."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content='{"plan":"tool"}'),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = _build_provider(server.base_url)

        dumped = provider.chat(
            messages=[{"role": "system", "content": "Summarizer"}]
        ).model_dump()

        provider._client.close()

    assert dumped["content"] == '{"plan":"tool"}'
    assert dumped["reasoning_content"] == '{"plan":"tool"}'


def test_generate_can_disable_reasoning_fallback(free_tcp_port: int) -> None:
    """Return empty content when reasoning fallback is disabled."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content='{"ok":true}'),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = _build_provider(
            server.base_url,
            fallback_to_reasoning_content=False,
        )

        result = provider.generate(system_prompt="sys", user_prompt="user")

        provider._client.close()

    assert result == ""


def test_generate_does_not_fallback_for_unstructured_reasoning(
    free_tcp_port: int,
) -> None:
    """Reject narrative reasoning when the fallback requires structured output."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content="step-by-step analysis"),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = _build_provider(server.base_url)

        result = provider.generate(system_prompt="sys", user_prompt="user")

        provider._client.close()

    assert result == ""


def test_chat_can_disable_reasoning_fallback(free_tcp_port: int) -> None:
    """Keep reasoning metadata without using it as the content body."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content='{"plan":"tool"}'),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = _build_provider(
            server.base_url,
            fallback_to_reasoning_content=False,
        )

        dumped = provider.chat(
            messages=[{"role": "system", "content": "Summarizer"}]
        ).model_dump()

        provider._client.close()

    assert dumped["content"] == ""
    assert dumped["reasoning_content"] == '{"plan":"tool"}'


def test_chat_does_not_fallback_when_tool_calls_present(free_tcp_port: int) -> None:
    """Preserve streamed tool calls without replacing content from reasoning."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content='{"plan":"call tool"}'),
                _chunk(
                    tool_calls=[
                        _tool_call_delta(
                            index=0,
                            id="call_1",
                            name="submit_answer",
                        )
                    ]
                ),
                _chunk(
                    tool_calls=[
                        _tool_call_delta(
                            index=0,
                            arguments='{"question_id":"Q001"}',
                        )
                    ]
                ),
                _chunk(finish_reason="tool_calls"),
            ]
        )
        provider = _build_provider(server.base_url)
        callback = _RecordingCallback()
        provider.set_response_callback(callback)

        dumped = provider.chat(
            messages=[{"role": "system", "content": "Evaluator"}]
        ).model_dump()

        provider._client.close()

    assert callback.events == [
        ("start", "Judge"),
        ("thought", '{"plan":"call tool"}'),
        ("content", "\nTOOL submit_answer: "),
        ("content", '{"question_id":"Q001"}'),
        ("complete", ""),
    ]
    assert dumped["content"] == ""
    assert dumped["reasoning_content"] == '{"plan":"call tool"}'
    assert dumped["tool_calls"][0]["id"] == "call_1"
    assert dumped["tool_calls"][0]["function"]["name"] == "submit_answer"
    assert dumped["tool_calls"][0]["function"]["arguments"] == '{"question_id":"Q001"}'
