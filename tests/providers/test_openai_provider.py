"""Coverage for the shared OpenAI Responses provider."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from card_framework.providers import openai_provider as openai_provider_module
from card_framework.providers.openai_provider import OpenAIProvider


class _RecordingCallback:
    """Capture callback events emitted during response handling."""

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


class _FakeResponse:
    """Minimal Responses API payload used by unit tests."""

    def __init__(self, *, output: list[object], output_text: str = "") -> None:
        self.output = output
        self.output_text = output_text

    def model_dump(self) -> dict[str, Any]:
        return {
            "output_text": self.output_text,
            "output": [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in self.output
            ],
        }


class _FakeItem(SimpleNamespace):
    """Simple object that exposes a stable model_dump payload."""

    def model_dump(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                payload[key] = [
                    item.model_dump() if hasattr(item, "model_dump") else item
                    for item in value
                ]
                continue
            payload[key] = value.model_dump() if hasattr(value, "model_dump") else value
        return payload


class _FakeResponsesClient:
    """Fake Responses client that records create payloads."""

    def __init__(self, *, requests: list[dict[str, Any]], create_impl) -> None:
        self._requests = requests
        self._create_impl = create_impl

    def create(self, **kwargs: Any) -> Any:
        self._requests.append(dict(kwargs))
        return self._create_impl(**kwargs)


class _FakeOpenAIClient:
    """Fake root OpenAI client."""

    def __init__(
        self,
        *,
        requests: list[dict[str, Any]],
        create_impl,
        api_key: str,
        timeout: float,
        base_url: str | None = None,
    ) -> None:
        del api_key, timeout, base_url
        self.responses = _FakeResponsesClient(
            requests=requests,
            create_impl=create_impl,
        )


def _install_fake_openai(monkeypatch, create_impl):  # type: ignore[no-untyped-def]
    requests: list[dict[str, Any]] = []

    def _factory(
        *,
        api_key: str,
        timeout: float,
        base_url: str | None = None,
    ) -> _FakeOpenAIClient:
        return _FakeOpenAIClient(
            requests=requests,
            create_impl=create_impl,
            api_key=api_key,
            timeout=timeout,
            base_url=base_url,
        )

    monkeypatch.setattr(openai_provider_module, "OpenAI", _factory)
    return requests


def test_chat_with_tools_uses_responses_api_and_emits_callbacks(monkeypatch) -> None:
    """Translate shared tool calls into Responses API payloads."""

    def _create_impl(**kwargs: Any) -> _FakeResponse:
        assert kwargs["model"] == "gpt-5.4-mini"
        assert kwargs["reasoning"] == {"effort": "none"}
        assert kwargs["store"] is False
        assert kwargs["tools"] == [
            {
                "type": "function",
                "name": "submit_samples",
                "description": "Return samples.",
                "parameters": {"type": "object"},
                "strict": True,
            }
        ]
        assert kwargs["tool_choice"] == {"type": "function", "name": "submit_samples"}
        return _FakeResponse(
            output=[
                _FakeItem(
                    type="message",
                    content=[_FakeItem(type="output_text", text="answer")],
                ),
                _FakeItem(
                    type="function_call",
                    call_id="call_1",
                    name="submit_samples",
                    arguments='{"episode_key":"episode://sample"}',
                ),
            ],
            output_text="answer",
        )

    requests = _install_fake_openai(monkeypatch, _create_impl)
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.4-mini")
    callback = _RecordingCallback()
    provider.set_response_callback(callback)

    response = provider.chat(
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

    assert response.content == "answer"
    assert response.reasoning_content is None
    assert response.tool_calls is not None
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].function.name == "submit_samples"
    assert (
        response.tool_calls[0].function.arguments
        == '{"episode_key":"episode://sample"}'
    )
    assert callback.events == [
        ("start", "Summarizer"),
        ("content", "answer"),
        ("content", "\nTOOL submit_samples: "),
        ("content", '{"episode_key":"episode://sample"}'),
        ("complete", ""),
    ]
    assert requests[0]["input"] == [
        {"role": "system", "content": "Summarizer"},
        {"role": "user", "content": "Question"},
    ]


def test_chat_preserves_tool_history_as_response_items(monkeypatch) -> None:
    """Replay prior tool calls and tool outputs through Responses input items."""

    def _create_impl(**kwargs: Any) -> _FakeResponse:
        assert kwargs["input"] == [
            {"role": "system", "content": "Summarizer"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "draft"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "submit_samples",
                "arguments": '{"episode_key":"episode://sample"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '{"status":"ok"}',
            },
        ]
        return _FakeResponse(
            output=[
                _FakeItem(
                    type="message",
                    content=[_FakeItem(type="output_text", text="done")],
                )
            ],
            output_text="done",
        )

    _install_fake_openai(monkeypatch, _create_impl)
    provider = OpenAIProvider(api_key="test-key", model="gpt-5.4-mini")

    response = provider.chat(
        messages=[
            {"role": "system", "content": "Summarizer"},
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": "draft",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "submit_samples",
                            "arguments": '{"episode_key":"episode://sample"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "name": "submit_samples",
                "tool_call_id": "call_1",
                "content": '{"status":"ok"}',
            },
        ]
    )

    assert response.content == "done"
    assert response.tool_calls is None


def test_generate_uses_reasoning_none_by_default(monkeypatch) -> None:
    """Default OpenAI generate calls should keep reasoning disabled."""

    def _create_impl(**kwargs: Any) -> _FakeResponse:
        assert kwargs["reasoning"] == {"effort": "none"}
        assert kwargs["max_output_tokens"] == 256
        return _FakeResponse(
            output=[
                _FakeItem(
                    type="message",
                    content=[_FakeItem(type="output_text", text="Paris")],
                )
            ],
            output_text="Paris",
        )

    _install_fake_openai(monkeypatch, _create_impl)
    provider = OpenAIProvider(api_key="test-key")

    assert provider.generate("Judge", "What is the capital of France?", 256) == "Paris"
