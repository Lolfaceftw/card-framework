"""Coverage for the shared Google Gen AI provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from google.genai import types

from card_framework.providers import (
    google_genai_provider as google_genai_provider_module,
)
from card_framework.providers.google_genai_provider import GoogleGenAIProvider


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


class _FakeModels:
    """Fake Gemini models client used for unit tests."""

    def __init__(
        self,
        *,
        requests: list[dict[str, Any]],
        stream_impl,
        non_stream_impl,
    ) -> None:
        self._requests = requests
        self._stream_impl = stream_impl
        self._non_stream_impl = non_stream_impl

    def generate_content_stream(self, **kwargs: Any) -> Any:
        self._requests.append({"kind": "stream", **kwargs})
        return self._stream_impl(**kwargs)

    def generate_content(self, **kwargs: Any) -> Any:
        self._requests.append({"kind": "non_stream", **kwargs})
        return self._non_stream_impl(**kwargs)


class _FakeGenAIClient:
    """Fake root Gemini client."""

    def __init__(
        self,
        *,
        requests: list[dict[str, Any]],
        stream_impl,
        non_stream_impl,
        api_key: str,
    ) -> None:
        del api_key
        self.models = _FakeModels(
            requests=requests,
            stream_impl=stream_impl,
            non_stream_impl=non_stream_impl,
        )


def _install_fake_client(monkeypatch, stream_impl, non_stream_impl):  # type: ignore[no-untyped-def]
    requests: list[dict[str, Any]] = []

    def _factory(*, api_key: str) -> _FakeGenAIClient:
        return _FakeGenAIClient(
            requests=requests,
            stream_impl=stream_impl,
            non_stream_impl=non_stream_impl,
            api_key=api_key,
        )

    monkeypatch.setattr(
        google_genai_provider_module,
        "genai",
        SimpleNamespace(Client=_factory),
    )
    return requests


def _chunk(*parts: types.Part) -> SimpleNamespace:
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=list(parts)))]
    )


def test_chat_with_tools_replays_exact_cached_tool_history(monkeypatch) -> None:
    """Reuse Gemini's exact function-call content, including thought signatures."""

    def _stream_impl(**kwargs: Any) -> object:
        raise AssertionError("tool-enabled chat should use non-streaming mode")

    def _non_stream_impl(**kwargs: Any) -> object:
        contents = kwargs["contents"]
        if len(contents) == 1:
            assert kwargs["model"] == "gemini-3.1-pro-preview"
            return SimpleNamespace(
                text="answer",
                candidates=[
                    SimpleNamespace(
                        content=types.Content(
                            role="model",
                            parts=[
                                types.Part(text="plan", thought=True),
                                types.Part.from_text(text="answer"),
                                types.Part(
                                    function_call=types.FunctionCall(
                                        id="call_1",
                                        name="submit_samples",
                                        args={"episode_key": "episode://sample"},
                                    ),
                                    thought_signature=b"sig-1",
                                ),
                            ],
                        )
                    )
                ],
            )

        assert len(contents) == 3
        assert contents[0].role == "user"
        assert contents[0].parts[0].text == "Question"
        assert contents[1].role == "model"
        function_call_part = next(
            part for part in contents[1].parts if part.function_call is not None
        )
        assert function_call_part.function_call.id == "call_1"
        assert function_call_part.function_call.name == "submit_samples"
        assert function_call_part.function_call.args == {
            "episode_key": "episode://sample"
        }
        assert function_call_part.thought_signature == b"sig-1"
        assert contents[2].role == "user"
        assert contents[2].parts[0].function_response.id == "call_1"
        return SimpleNamespace(
            text="done",
            candidates=[
                SimpleNamespace(
                    content=types.Content(
                        role="model",
                        parts=[types.Part.from_text(text="done")],
                    )
                )
            ],
        )

    requests = _install_fake_client(monkeypatch, _stream_impl, _non_stream_impl)
    provider = GoogleGenAIProvider(
        api_key="test-key",
        model="gemini-3.1-pro-preview",
    )
    callback = _RecordingCallback()
    provider.set_response_callback(callback)

    first_response = provider.chat(
        messages=[
            {"role": "system", "content": "Summarizer"},
            {"role": "user", "content": "Question"},
        ],
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

    assert first_response.content == "answer"
    assert first_response.reasoning_content == "plan"
    assert first_response.tool_calls is not None
    assert first_response.tool_calls[0].id == "call_1"

    response = provider.chat(
        messages=[
            {"role": "system", "content": "Summarizer"},
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": first_response.content,
                "tool_calls": [first_response.tool_calls[0].model_dump()],
            },
            {
                "role": "tool",
                "name": "submit_samples",
                "tool_call_id": "call_1",
                "content": '{"status":"ok"}',
            },
        ]
    )

    assert callback.events == [
        ("start", "Summarizer"),
        ("thought", "plan"),
        ("content", "answer"),
        ("content", "\nTOOL submit_samples: "),
        ("content", '{"episode_key":"episode://sample"}'),
        ("complete", ""),
        ("start", "Summarizer"),
        ("content", "done"),
        ("complete", ""),
    ]
    assert response.content == "done"

    assert [request["kind"] for request in requests] == ["non_stream", "non_stream"]
    request = requests[0]
    assert request["config"].automatic_function_calling.disable is True
    function_calling_config = request["config"].tool_config.function_calling_config
    assert function_calling_config.mode == types.FunctionCallingConfigMode.ANY
    assert function_calling_config.allowed_function_names == ["submit_samples"]
    assert request["config"].system_instruction == "Summarizer"
    assert request["config"].thinking_config.include_thoughts is True
    assert request["config"].tools[0].function_declarations[0].name == "submit_samples"


def test_chat_without_tools_still_streams(monkeypatch) -> None:
    """Keep streaming thought/content updates for plain non-tool chats."""

    def _stream_impl(**kwargs: Any) -> list[object]:
        assert kwargs["config"].tools is None
        return [
            _chunk(types.Part(text="plan", thought=True)),
            _chunk(types.Part.from_text(text="done")),
        ]

    def _non_stream_impl(**kwargs: Any) -> object:
        raise AssertionError("non-stream fallback should not be used")

    requests = _install_fake_client(monkeypatch, _stream_impl, _non_stream_impl)
    provider = GoogleGenAIProvider(api_key="test-key")
    callback = _RecordingCallback()
    provider.set_response_callback(callback)

    response = provider.chat(messages=[{"role": "system", "content": "Summarizer"}])

    assert response.content == "done"
    assert response.reasoning_content == "plan"
    assert callback.events == [
        ("start", "Summarizer"),
        ("thought", "plan"),
        ("content", "done"),
        ("complete", ""),
    ]
    assert requests[0]["kind"] == "stream"


def test_chat_falls_back_to_non_streaming_response(monkeypatch) -> None:
    """Retry a failed Gemini stream as a non-streaming request."""

    def _stream_impl(**kwargs: Any) -> object:
        raise RuntimeError("stream failed")

    def _non_stream_impl(**kwargs: Any) -> object:
        return SimpleNamespace(
            text="answer",
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[
                            types.Part(text="reason", thought=True),
                            types.Part.from_text(text="answer"),
                            types.Part(
                                function_call=types.FunctionCall(
                                    id="call_2",
                                    name="submit_samples",
                                    args={"episode_key": "episode://sample"},
                                )
                            ),
                        ]
                    )
                )
            ],
        )

    requests = _install_fake_client(monkeypatch, _stream_impl, _non_stream_impl)
    provider = GoogleGenAIProvider(api_key="test-key")
    callback = _RecordingCallback()
    provider.set_response_callback(callback)

    response = provider.chat(messages=[{"role": "system", "content": "Summarizer"}])

    assert [request["kind"] for request in requests] == ["stream", "non_stream"]
    assert response.content == "answer"
    assert response.reasoning_content == "reason"
    assert response.tool_calls is not None
    assert response.tool_calls[0].id == "call_2"
    assert response.tool_calls[0].function.name == "submit_samples"
    assert (
        response.tool_calls[0].function.arguments
        == '{"episode_key":"episode://sample"}'
    )
    assert callback.events == [
        ("start", "Summarizer"),
        ("thought", "reason"),
        ("content", "answer"),
        ("content", "\nTOOL submit_samples: "),
        ("content", '{"episode_key":"episode://sample"}'),
        ("complete", ""),
    ]


def test_chat_with_thinking_disabled_uses_minimal_level_for_gemini_3(
    monkeypatch,
) -> None:
    """Gemini 3 contestants should explicitly request the lowest thinking mode."""

    def _stream_impl(**kwargs: Any) -> list[object]:
        thinking_config = kwargs["config"].thinking_config
        assert thinking_config is not None
        assert thinking_config.include_thoughts in (None, False)
        assert (
            str(
                getattr(
                    thinking_config.thinking_level,
                    "value",
                    thinking_config.thinking_level,
                )
            ).lower()
            == "minimal"
        )
        return [_chunk(types.Part.from_text(text="done"))]

    def _non_stream_impl(**kwargs: Any) -> object:
        raise AssertionError("non-stream fallback should not be used")

    _install_fake_client(monkeypatch, _stream_impl, _non_stream_impl)
    provider = GoogleGenAIProvider(
        api_key="test-key",
        model="gemini-3-flash-preview",
        enable_thinking=False,
    )

    response = provider.chat(messages=[{"role": "system", "content": "Summarizer"}])

    assert response.content == "done"
    assert response.reasoning_content is None
