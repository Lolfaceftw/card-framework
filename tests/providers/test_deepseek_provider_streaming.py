"""Streaming behavior tests for the DeepSeek provider."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

# Provide minimal `numpy` module stub for imports used by llm_provider typing.
if "numpy" not in sys.modules:
    numpy_module = types.ModuleType("numpy")

    class _NDArray:
        pass

    numpy_module.ndarray = _NDArray
    sys.modules["numpy"] = numpy_module

# Provide minimal `openai` module stub for unit tests.
if "openai" not in sys.modules:
    openai_module = types.ModuleType("openai")

    class _OpenAI:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=lambda **_kwargs: iter(()))
            )

    openai_module.OpenAI = _OpenAI
    sys.modules["openai"] = openai_module

_ROOT = Path(__file__).resolve().parents[2]
_PROVIDER_PATH = _ROOT / "providers" / "deepseek_provider.py"
_SPEC = importlib.util.spec_from_file_location(
    "deepseek_provider_streaming_for_tests",
    _PROVIDER_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load providers/deepseek_provider.py for tests.")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
DeepSeekProvider = _MODULE.DeepSeekProvider


class _FakeDelta:
    def __init__(
        self,
        *,
        content: str | None = None,
        reasoning_content: str | None = None,
        tool_calls: list[Any] | None = None,
    ) -> None:
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, delta: _FakeDelta) -> None:
        self.delta = delta


class _FakeChunk:
    def __init__(self, delta: _FakeDelta) -> None:
        self.choices = [_FakeChoice(delta)]


class _FakeCompletions:
    def __init__(self, chunks: list[_FakeChunk]) -> None:
        self._chunks = chunks
        self.last_kwargs: dict[str, Any] | None = None

    def create(self, **kwargs: Any):
        self.last_kwargs = kwargs
        return iter(self._chunks)


class _FakeOpenAI:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.completions = _FakeCompletions([])
        self.chat = types.SimpleNamespace(completions=self.completions)


class _RecordingCallback:
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


def _build_provider(
    monkeypatch,
    *,
    chunks: list[_FakeChunk],
) -> tuple[DeepSeekProvider, _FakeCompletions]:
    monkeypatch.setattr(_MODULE, "OpenAI", _FakeOpenAI)
    provider = DeepSeekProvider(api_key="test")
    provider._client.completions = _FakeCompletions(chunks)
    provider._client.chat = types.SimpleNamespace(
        completions=provider._client.completions
    )
    return provider, provider._client.completions


def test_chat_streams_reasoning_and_content_through_response_callback(
    monkeypatch,
) -> None:
    provider, completions = _build_provider(
        monkeypatch,
        chunks=[
            _FakeChunk(_FakeDelta(reasoning_content="plan")),
            _FakeChunk(_FakeDelta(content="answer")),
        ],
    )
    callback = _RecordingCallback()
    provider.set_response_callback(callback)

    message = provider.chat(messages=[{"role": "system", "content": "Summarizer"}])
    dumped = message.model_dump()

    assert completions.last_kwargs is not None
    assert completions.last_kwargs["stream"] is True
    assert callback.events == [
        ("start", "Summarizer"),
        ("thought", "plan"),
        ("content", "answer"),
        ("complete", ""),
    ]
    assert dumped["content"] == "answer"
    assert dumped["reasoning_content"] == "plan"
