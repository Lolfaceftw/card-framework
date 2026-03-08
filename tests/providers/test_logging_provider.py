from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PROVIDER_PATH = _ROOT / "providers" / "logging_provider.py"
_SPEC = importlib.util.spec_from_file_location(
    "logging_provider_for_tests",
    _PROVIDER_PATH,
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("Unable to load providers/logging_provider.py for tests.")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
LoggingLLMProvider = _MODULE.LoggingLLMProvider


class _FakeInnerProvider:
    def __init__(self, *, generate_result: str = "ok", chat_result: Any = None) -> None:
        self.generate_result = generate_result
        self.chat_result = chat_result
        self.generate_calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.generate_calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
            }
        )
        return self.generate_result

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        self.chat_calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "max_tokens": max_tokens,
            }
        )
        return self.chat_result


@dataclass(slots=True)
class _FakeMessage:
    role: str
    content: str
    tool_calls: list[dict[str, Any]] | None = None


class _FakeLogger:
    def __init__(self) -> None:
        self.debug_calls: list[str] = []
        self.info_calls: list[str] = []
        self.warning_calls: list[str] = []
        self.error_calls: list[str] = []

    def debug(self, message: str) -> None:
        self.debug_calls.append(message)

    def info(self, message: str) -> None:
        self.info_calls.append(message)

    def warning(self, message: str) -> None:
        self.warning_calls.append(message)

    def error(self, message: str) -> None:
        self.error_calls.append(message)


def test_logging_provider_generate_publishes_event(monkeypatch) -> None:
    inner = _FakeInnerProvider(generate_result="response")
    fake_logger = _FakeLogger()
    published: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(_MODULE, "logger", fake_logger)
    monkeypatch.setattr(
        _MODULE.event_bus,
        "publish",
        lambda event_type, **kwargs: published.append((event_type, kwargs)),
    )

    provider = LoggingLLMProvider(inner_provider=inner)
    result = provider.generate(
        system_prompt="sys",
        user_prompt="user",
        max_tokens=128,
    )

    assert result == "response"
    assert inner.generate_calls == [
        {
            "system_prompt": "sys",
            "user_prompt": "user",
            "max_tokens": 128,
        }
    ]
    assert published
    event_type, payload = published[-1]
    assert event_type == "llm_call_completed"
    assert payload["operation"] == "generate"
    assert payload["provider"] == "_FakeInnerProvider"
    assert payload["input_messages"] == 2
    assert payload["tool_count"] == 0
    assert isinstance(payload["latency_ms"], int)
    assert payload["latency_ms"] >= 0


def test_logging_provider_chat_publishes_event(monkeypatch) -> None:
    message = _FakeMessage(
        role="assistant",
        content="ok",
        tool_calls=[{"id": "tool_1", "function": {"name": "x", "arguments": "{}"}}],
    )
    inner = _FakeInnerProvider(chat_result=message)
    fake_logger = _FakeLogger()
    published: list[tuple[str, dict[str, Any]]] = []

    monkeypatch.setattr(_MODULE, "logger", fake_logger)
    monkeypatch.setattr(
        _MODULE.event_bus,
        "publish",
        lambda event_type, **kwargs: published.append((event_type, kwargs)),
    )

    provider = LoggingLLMProvider(inner_provider=inner)
    result = provider.chat(
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
        tools=[{"type": "function", "function": {"name": "x"}}],
        tool_choice="auto",
        max_tokens=64,
    )

    assert result is message
    assert inner.chat_calls
    event_type, payload = published[-1]
    assert event_type == "llm_call_completed"
    assert payload["operation"] == "chat"
    assert payload["provider"] == "_FakeInnerProvider"
    assert payload["input_messages"] == 2
    assert payload["tool_count"] == 1


def test_logging_provider_chat_logs_terminal_friendly_summaries(monkeypatch) -> None:
    message = _FakeMessage(
        role="assistant",
        content="Long answer body that should not be dumped in full at info level.",
        tool_calls=[{"id": "tool_1", "function": {"name": "add_message", "arguments": "{}"}}],
    )
    inner = _FakeInnerProvider(chat_result=message)
    fake_logger = _FakeLogger()

    monkeypatch.setattr(_MODULE, "logger", fake_logger)

    provider = LoggingLLMProvider(inner_provider=inner)
    provider.chat(
        messages=[
            {"role": "system", "content": "Summarizer"},
            {"role": "user", "content": "Please summarize this very long transcript."},
        ],
        tools=[{"type": "function", "function": {"name": "add_message"}}],
        tool_choice="auto",
        max_tokens=64,
    )

    assert any("Message Summary:" in line for line in fake_logger.info_calls)
    assert any("Tool Summary:" in line for line in fake_logger.info_calls)
    assert any("LLM CHAT RESPONSE SUMMARY:" in line for line in fake_logger.info_calls)
    assert not any(line.startswith("Tools: [") for line in fake_logger.info_calls)
    assert any(line.startswith("Tools: [") for line in fake_logger.debug_calls)


def test_logging_provider_generate_reraises_inner_errors(monkeypatch) -> None:
    class _FailingProvider(_FakeInnerProvider):
        def generate(
            self,
            system_prompt: str,
            user_prompt: str,
            max_tokens: int | None = None,
        ) -> str:
            del system_prompt, user_prompt, max_tokens
            raise RuntimeError("boom")

    fake_logger = _FakeLogger()
    monkeypatch.setattr(_MODULE, "logger", fake_logger)

    provider = LoggingLLMProvider(inner_provider=_FailingProvider())

    with pytest.raises(RuntimeError, match="boom"):
        provider.generate(system_prompt="sys", user_prompt="user")

    assert fake_logger.error_calls
    assert any("LLM ERROR" in line for line in fake_logger.error_calls)
