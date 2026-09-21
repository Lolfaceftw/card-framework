"""Exercise the vLLM provider against a real OpenAI-compatible stream."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from omegaconf import OmegaConf

from card_framework.providers.logging_provider import LoggingLLMProvider
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
    reasoning: str | None = None,
    tool_calls: list[dict[str, Any]] | None = None,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Build one OpenAI chat-completion chunk payload."""
    delta: dict[str, Any] = {}
    if content is not None:
        delta["content"] = content
    if reasoning_content is not None:
        delta["reasoning_content"] = reasoning_content
    if reasoning is not None:
        delta["reasoning"] = reasoning
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
) -> VLLMProvider:
    """Create a strict provider connected to the local integration server.

    Args:
        base_url: Root URL exposed by the local OpenAI-compatible server.
        enable_thinking: Whether the request should enable Qwen reasoning output.

    Returns:
        A provider that performs one request without transport or content fallbacks.
    """
    return VLLMProvider(
        base_url=f"{base_url}/v1",
        api_key="EMPTY",
        enable_thinking=enable_thinking,
    )


def test_provider_rejects_unadvertised_explicit_model(free_tcp_port: int) -> None:
    """Fail fast when a pinned benchmark model is not served by the endpoint."""
    with run_openai_compatible_server(
        port=free_tcp_port,
        model_id="Qwen3-32B",
    ):
        with pytest.raises(ValueError, match="Qwen3-14B"):
            VLLMProvider(
                base_url=f"http://127.0.0.1:{free_tcp_port}/v1",
                api_key="EMPTY",
                model="Qwen3-14B",
                enable_thinking=False,
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


def test_generate_uses_configured_default_max_tokens(free_tcp_port: int) -> None:
    """Send a profile-level output budget when the caller does not pass one."""
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream([_chunk(content="answer"), _chunk(finish_reason="stop")])
        provider = VLLMProvider(
            base_url=f"{server.base_url}/v1",
            api_key="EMPTY",
            enable_thinking=False,
            default_max_tokens=245_760,
        )

        result = provider.generate(system_prompt="sys", user_prompt="user")

        provider._client.close()

    assert result == "answer"
    assert server.chat_requests[-1]["max_tokens"] == 245_760


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


def test_chat_normalizes_live_vllm_reasoning_field(free_tcp_port: int) -> None:
    """Capture the reasoning field emitted by the authenticated study server.

    Args:
        free_tcp_port: Unused localhost port supplied by pytest.

    Current vLLM responses expose thinking text as ``delta.reasoning``. CARD's
    provider preserves it in the stable AssistantMessage reasoning field and
    records the original server field name for later audit.
    """
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning="live thought"),
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
    assert dumped["reasoning_content"] == "live thought"
    assert provider.last_response is not None
    assert provider.last_response["reasoning_fields"] == ["reasoning"]


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


def test_chat_rejects_response_without_terminal_finish_reason(
    free_tcp_port: int,
) -> None:
    """Reject a partial stream after the server closes it without completion.

    Args:
        free_tcp_port: Unused localhost port supplied by pytest.

    The assertion also verifies that the provider sends one request and does not
    retry with an alternate transport after receiving the incomplete response.
    """
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream([_chunk(content="partial")])
        provider = _build_provider(server.base_url, enable_thinking=False)

        with pytest.raises(ValueError, match="finish_reason=None"):
            provider.chat(messages=[{"role": "system", "content": "Summarizer"}])

        provider._client.close()

    assert len(server.chat_requests) == 1
    assert provider.last_response is not None
    assert provider.last_response["content"] == "partial"


def test_generate_rejects_length_limited_response(free_tcp_port: int) -> None:
    """Reject output that the server terminated at its token limit.

    Args:
        free_tcp_port: Unused localhost port supplied by pytest.

    A length finish cannot enter the controlled-duration study because its
    generated text is incomplete and the provider must not retry automatically.
    """
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [_chunk(content="truncated"), _chunk(finish_reason="length")]
        )
        provider = _build_provider(server.base_url, enable_thinking=False)

        with pytest.raises(ValueError, match="finish_reason='length'"):
            provider.generate(system_prompt="sys", user_prompt="user")

        provider._client.close()

    assert len(server.chat_requests) == 1


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


def test_generate_keeps_reasoning_separate_when_content_empty(
    free_tcp_port: int,
) -> None:
    """Keep reasoning out of the final generated text.

    Args:
        free_tcp_port: Unused localhost port supplied by pytest.

    The server-returned reasoning remains available in ``last_response`` for the
    call trace while the user-visible result reflects the empty content channel.
    """
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

    assert result == ""
    assert provider.last_response is not None
    assert provider.last_response["reasoning_content"] == '{"ok":true}'


def test_chat_keeps_reasoning_separate_when_content_empty(
    free_tcp_port: int,
) -> None:
    """Return reasoning metadata without copying it into message content.

    Args:
        free_tcp_port: Unused localhost port supplied by pytest.
    """
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

    assert dumped["content"] == ""
    assert dumped["reasoning_content"] == '{"plan":"tool"}'


def test_generate_omits_max_tokens_when_unconfigured(free_tcp_port: int) -> None:
    """Leave the request uncapped when no output limit is configured.

    Args:
        free_tcp_port: Unused localhost port supplied by pytest.

    The study uses the endpoint context capacity and rejects any server-side
    length finish, so the request must not inject a client output cap.
    """
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream([_chunk(content="ok"), _chunk(finish_reason="stop")])
        provider = _build_provider(server.base_url, enable_thinking=False)

        result = provider.generate(system_prompt="sys", user_prompt="user")

        provider._client.close()

    assert result == "ok"
    assert "max_tokens" not in server.chat_requests[-1]


def test_provider_rejects_prompt_truncation_configuration(free_tcp_port: int) -> None:
    """Reject a provider configuration that asks vLLM to clip the prompt.

    Args:
        free_tcp_port: Unused localhost port supplied by pytest.

    Raises are checked at call time because extra request bodies are merged only
    when the complete OpenAI request is assembled.
    """
    with run_openai_compatible_server(port=free_tcp_port) as server:
        provider = VLLMProvider(
            base_url=f"{server.base_url}/v1",
            api_key="EMPTY",
            enable_thinking=False,
            extra_body={"truncate_prompt_tokens": 4096},
        )

        with pytest.raises(ValueError, match="Prompt truncation is disabled"):
            provider.generate(system_prompt="sys", user_prompt="user")

        provider._client.close()

    assert server.chat_requests == []


def test_chat_rejects_malformed_tool_arguments(free_tcp_port: int) -> None:
    """Reject a completed tool call whose arguments are not valid JSON.

    Args:
        free_tcp_port: Unused localhost port supplied by pytest.

    The invalid call remains in the diagnostic record and is never returned for
    execution by the caller.
    """
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(
                    tool_calls=[
                        _tool_call_delta(
                            index=0,
                            id="call_bad",
                            name="submit_answer",
                            arguments='{"question_id":',
                        )
                    ]
                ),
                _chunk(finish_reason="tool_calls"),
            ]
        )
        provider = _build_provider(server.base_url)

        with pytest.raises(json.JSONDecodeError):
            provider.chat(messages=[{"role": "system", "content": "Evaluator"}])

        provider._client.close()

    assert provider.last_response is not None
    assert provider.last_response["tool_calls"][0]["id"] == "call_bad"


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


def test_logging_provider_archives_complete_reasoning_and_output(
    free_tcp_port: int,
    tmp_path: Path,
) -> None:
    """Persist one navigable call without credential or token-fragment records.

    Args:
        free_tcp_port: Unused localhost port supplied by pytest.
        tmp_path: Pytest-owned directory for the complete call archive.

    The local HTTP server exercises the real OpenAI client stream parser. The
    resulting JSON and Markdown must contain the full prompt, reasoning text,
    final output, provider metadata, and one README index entry.
    """
    trace_dir = (tmp_path / "agent_calls").resolve()
    with run_openai_compatible_server(port=free_tcp_port) as server:
        server.enqueue_stream(
            [
                _chunk(reasoning_content="complete reasoning"),
                _chunk(content="complete answer"),
                _chunk(finish_reason="stop"),
            ]
        )
        provider = VLLMProvider(
            base_url=f"{server.base_url}/v1",
            api_key="test-secret",
            enable_thinking=True,
        )
        wrapped = LoggingLLMProvider(provider, trace_dir=trace_dir)

        response = wrapped.chat(
            messages=[
                {"role": "system", "content": "Summarizer"},
                {"role": "user", "content": "Keep this complete prompt."},
            ]
        )

        provider._client.close()

    json_paths = list(trace_dir.glob("*.json"))
    markdown_paths = [path for path in trace_dir.glob("*.md") if path.name != "README.md"]
    assert response.content == "complete answer"
    assert len(json_paths) == 1
    assert len(markdown_paths) == 1
    payload_text = json_paths[0].read_text(encoding="utf-8")
    payload = json.loads(payload_text)
    markdown = markdown_paths[0].read_text(encoding="utf-8")
    assert payload["request"]["messages"][1]["content"] == "Keep this complete prompt."
    assert payload["provider_response"]["reasoning_content"] == "complete reasoning"
    assert payload["provider_response"]["content"] == "complete answer"
    assert "test-secret" not in payload_text
    assert "complete reasoning" in markdown
    assert "complete answer" in markdown
    assert markdown_paths[0].name in (trace_dir / "README.md").read_text(
        encoding="utf-8"
    )
