"""Reasoning behavior tests for the vLLM provider."""

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

# Provide a minimal `openai` module stub for environments without the package.
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
_PROVIDER_PATH = _ROOT / "providers" / "vllm_provider.py"
_SPEC = importlib.util.spec_from_file_location("vllm_provider_for_tests", _PROVIDER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load providers/vllm_provider.py for tests.")
vllm_provider_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(vllm_provider_module)


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


def _build_provider(
    monkeypatch,
    *,
    chunks: list[_FakeChunk],
    enable_thinking: bool = True,
) -> tuple[vllm_provider_module.VLLMProvider, _FakeCompletions]:
    monkeypatch.setattr(vllm_provider_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(
        vllm_provider_module.VLLMProvider,
        "_fetch_model_id",
        lambda _self: "fake-model",
    )

    provider = vllm_provider_module.VLLMProvider(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
        enable_thinking=enable_thinking,
    )
    provider._client.completions = _FakeCompletions(chunks)
    provider._client.chat = types.SimpleNamespace(completions=provider._client.completions)
    return provider, provider._client.completions


def test_generate_requests_thinking_mode(monkeypatch) -> None:
    provider, completions = _build_provider(
        monkeypatch,
        chunks=[
            _FakeChunk(_FakeDelta(reasoning_content="thinking")),
            _FakeChunk(_FakeDelta(content="answer")),
        ],
    )

    result = provider.generate(
        system_prompt="sys",
        user_prompt="user",
        max_tokens=32,
    )

    assert result == "answer"
    assert completions.last_kwargs is not None
    assert completions.last_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_chat_returns_reasoning_content_field(monkeypatch) -> None:
    provider, completions = _build_provider(
        monkeypatch,
        chunks=[
            _FakeChunk(_FakeDelta(reasoning_content="plan")),
            _FakeChunk(_FakeDelta(content="done")),
        ],
    )

    message = provider.chat(messages=[{"role": "system", "content": "Summarizer"}])
    dumped = message.model_dump()

    assert completions.last_kwargs is not None
    assert "extra_body" in completions.last_kwargs
    assert dumped["content"] == "done"
    assert dumped["reasoning_content"] == "plan"
    assert "reasoning" not in dumped


def test_chat_can_disable_thinking_mode(monkeypatch) -> None:
    provider, completions = _build_provider(
        monkeypatch,
        chunks=[_FakeChunk(_FakeDelta(content="ok"))],
        enable_thinking=False,
    )

    provider.chat(messages=[{"role": "system", "content": "Critic"}])

    assert completions.last_kwargs is not None
    assert "extra_body" not in completions.last_kwargs
