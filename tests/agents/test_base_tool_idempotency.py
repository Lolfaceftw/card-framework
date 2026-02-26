"""Tool replay/idempotency tests for agent execution loops."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

# Provide minimal `numpy` module stub for imports used by llm_provider typing.
if "numpy" not in sys.modules:
    numpy_module = types.ModuleType("numpy")

    class _NDArray:
        pass

    numpy_module.ndarray = _NDArray
    sys.modules["numpy"] = numpy_module

# Provide minimal `jinja2` module stub for prompt_manager imports.
if "jinja2" not in sys.modules:
    jinja2_module = types.ModuleType("jinja2")

    class _FileSystemLoader:
        def __init__(self, _path: str) -> None:
            self.path = _path

    class _Environment:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def get_template(self, _template_name: str):  # noqa: ANN204
            raise RuntimeError("Template rendering is not used in this unit test.")

    def _select_autoescape() -> bool:
        return False

    jinja2_module.Environment = _Environment
    jinja2_module.FileSystemLoader = _FileSystemLoader
    jinja2_module.select_autoescape = _select_autoescape
    sys.modules["jinja2"] = jinja2_module

# Provide minimal `a2a` module stubs for unit tests.
if "a2a.server.agent_execution" not in sys.modules:
    a2a_module = types.ModuleType("a2a")
    server_module = types.ModuleType("a2a.server")
    agent_execution_module = types.ModuleType("a2a.server.agent_execution")
    events_module = types.ModuleType("a2a.server.events")
    utils_module = types.ModuleType("a2a.utils")

    class _AgentExecutor:
        pass

    class _RequestContext:
        pass

    class _EventQueue:
        async def enqueue_event(self, _event: object) -> None:
            return None

    def _new_agent_text_message(text: str) -> str:
        return text

    agent_execution_module.AgentExecutor = _AgentExecutor
    agent_execution_module.RequestContext = _RequestContext
    events_module.EventQueue = _EventQueue
    utils_module.new_agent_text_message = _new_agent_text_message

    sys.modules["a2a"] = a2a_module
    sys.modules["a2a.server"] = server_module
    sys.modules["a2a.server.agent_execution"] = agent_execution_module
    sys.modules["a2a.server.events"] = events_module
    sys.modules["a2a.utils"] = utils_module

from agents.base import BaseA2AExecutor
from agents.summarizer import SummarizerExecutor


class FakeResponse:
    """Simple chat response DTO compatible with BaseA2AExecutor."""

    def __init__(self, *, content: str = "", tool_calls: list[dict[str, Any]] | None = None):
        self._content = content
        self._tool_calls = tool_calls

    def model_dump(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self._content,
            "tool_calls": self._tool_calls,
        }


class FakeLLM:
    """Deterministic LLM stub returning pre-seeded chat responses."""

    def __init__(self, responses: list[FakeResponse]) -> None:
        self._responses = responses
        self._index = 0

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> FakeResponse:
        if self._index >= len(self._responses):
            return FakeResponse(content="done", tool_calls=None)
        response = self._responses[self._index]
        self._index += 1
        return response

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        return ""


class DummyExecutor(BaseA2AExecutor):
    """Minimal executor exposing BaseA2AExecutor loop behavior for testing."""

    def __init__(self, llm: FakeLLM) -> None:
        super().__init__("Dummy")
        self.llm = llm
        self.executed_calls: list[tuple[str, str, dict[str, Any]]] = []

    async def handle_task(self, task_data: dict, context, event_queue) -> None:
        raise NotImplementedError("Not needed for these unit tests.")

    async def process_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        context_data: dict[str, Any],
    ) -> tuple[bool, dict | None]:
        for tool_call in tool_calls:
            self.executed_calls.append(
                (
                    str(tool_call["id"]),
                    str(tool_call["name"]),
                    dict(tool_call["arguments"]),
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": tool_call["name"],
                    "content": json.dumps({"status": "ok"}),
                }
            )
        return False, None


class FakeToolRegistry:
    """Async tool registry stub used by SummarizerExecutor tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._line_count = 0

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        self.calls.append((name, dict(arguments)))
        if name == "add_speaker_message":
            self._line_count += 1
            return {"status": "added"}
        if name == "count_words":
            return {"total_word_count": 10}
        if name == "save_draft":
            return {"total_messages": self._line_count}
        if name == "finalize_draft":
            return {"status": "finalized"}
        return {"status": "ok"}


def _tool_call(tool_id: str, speaker_id: str, content: str) -> dict[str, Any]:
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": "add_speaker_message",
            "arguments": json.dumps({"speaker_id": speaker_id, "content": content}),
        },
    }


def test_base_loop_skips_duplicate_signature_across_turns() -> None:
    llm = FakeLLM(
        responses=[
            FakeResponse(tool_calls=[_tool_call("call_1", "SPEAKER_00", "Hello world")]),
            FakeResponse(tool_calls=[_tool_call("call_2", "SPEAKER_00", "Hello world")]),
        ]
    )
    executor = DummyExecutor(llm)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "x"}]

    asyncio.run(
        executor.run_agent_loop(
            messages=messages,
            tools=[],
            max_turns=2,
            context_data={
                "replay_dedupe_tools": {"add_speaker_message"},
                "signature_dedupe_window_turns": 10,
            },
        )
    )

    assert len(executor.executed_calls) == 1
    assert executor.executed_calls[0][0] == "call_1"

    skipped_tool_messages = [
        m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == "call_2"
    ]
    assert len(skipped_tool_messages) == 1
    payload = json.loads(skipped_tool_messages[0]["content"])
    assert payload["status"] == "skipped_duplicate"
    assert payload["reason"] == "signature"


def test_base_loop_allows_same_signature_after_window_expires() -> None:
    llm = FakeLLM(
        responses=[
            FakeResponse(tool_calls=[_tool_call("call_1", "SPEAKER_00", "repeat me")]),
            FakeResponse(tool_calls=[_tool_call("call_2", "SPEAKER_01", "different")]),
            FakeResponse(tool_calls=[_tool_call("call_3", "SPEAKER_00", "repeat me")]),
        ]
    )
    executor = DummyExecutor(llm)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "x"}]

    asyncio.run(
        executor.run_agent_loop(
            messages=messages,
            tools=[],
            max_turns=3,
            context_data={
                "replay_dedupe_tools": {"add_speaker_message"},
                "signature_dedupe_window_turns": 1,
            },
        )
    )

    assert len(executor.executed_calls) == 3
    assert [call_id for call_id, _, _ in executor.executed_calls] == [
        "call_1",
        "call_2",
        "call_3",
    ]


def test_base_loop_allows_distinct_signatures() -> None:
    llm = FakeLLM(
        responses=[
            FakeResponse(tool_calls=[_tool_call("call_1", "SPEAKER_00", "Line one")]),
            FakeResponse(tool_calls=[_tool_call("call_2", "SPEAKER_00", "Line two")]),
        ]
    )
    executor = DummyExecutor(llm)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "x"}]

    asyncio.run(
        executor.run_agent_loop(
            messages=messages,
            tools=[],
            max_turns=2,
            context_data={
                "replay_dedupe_tools": {"add_speaker_message"},
                "signature_dedupe_window_turns": 10,
            },
        )
    )

    assert len(executor.executed_calls) == 2
    assert [call_id for call_id, _, _ in executor.executed_calls] == ["call_1", "call_2"]


def test_base_loop_enforces_max_tool_calls_per_turn() -> None:
    llm = FakeLLM(
        responses=[
            FakeResponse(
                tool_calls=[
                    _tool_call("call_1", "SPEAKER_00", "Line one"),
                    _tool_call("call_2", "SPEAKER_01", "Line two"),
                ]
            )
        ]
    )
    executor = DummyExecutor(llm)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "x"}]

    asyncio.run(
        executor.run_agent_loop(
            messages=messages,
            tools=[],
            max_turns=1,
            context_data={
                "replay_dedupe_tools": {"add_speaker_message"},
                "max_tool_calls_per_turn": 1,
            },
        )
    )

    assert len(executor.executed_calls) == 1
    assert executor.executed_calls[0][0] == "call_1"

    skipped_tool_messages = [
        m for m in messages if m.get("role") == "tool" and m.get("tool_call_id") == "call_2"
    ]
    assert len(skipped_tool_messages) == 1
    payload = json.loads(skipped_tool_messages[0]["content"])
    assert payload["status"] == "skipped"
    assert payload["reason"] == "max_tool_calls_per_turn"
    assert payload["max_tool_calls_per_turn"] == 1


def test_base_loop_does_not_id_dedupe_synthetic_fallback_ids() -> None:
    llm = FakeLLM(
        responses=[
            FakeResponse(tool_calls=[_tool_call("xml_fallback_0", "SPEAKER_00", "Line one")]),
            FakeResponse(tool_calls=[_tool_call("xml_fallback_0", "SPEAKER_01", "Line two")]),
        ]
    )
    executor = DummyExecutor(llm)
    messages: list[dict[str, Any]] = [{"role": "system", "content": "x"}]

    asyncio.run(
        executor.run_agent_loop(
            messages=messages,
            tools=[],
            max_turns=2,
            context_data={"replay_dedupe_tools": {"add_speaker_message"}},
        )
    )

    assert len(executor.executed_calls) == 2
    assert [call_id for call_id, _, _ in executor.executed_calls] == [
        "xml_fallback_0",
        "xml_fallback_0",
    ]


def test_summarizer_skips_duplicate_without_auto_count_or_save() -> None:
    llm = FakeLLM(
        responses=[
            FakeResponse(tool_calls=[_tool_call("call_1", "SPEAKER_00", "One line")]),
            FakeResponse(tool_calls=[_tool_call("call_2", "SPEAKER_00", "One line")]),
        ]
    )
    executor = SummarizerExecutor(
        llm=llm,
        retrieval_port=12345,
        max_tool_turns=2,
        is_embedding_enabled=False,
    )
    fake_registry = FakeToolRegistry()
    messages: list[dict[str, Any]] = [{"role": "system", "content": "Summarizer"}]

    asyncio.run(
        executor.run_agent_loop(
            messages=messages,
            tools=[],
            max_turns=2,
            context_data={
                "tool_registry": fake_registry,
                "min_words": 1,
                "max_words": 100,
                "signature_dedupe_window_turns": 1,
                "replay_dedupe_tools": {
                    "add_speaker_message",
                    "edit_message",
                    "remove_message",
                    "finalize_draft",
                },
            },
        )
    )

    called_tools = [name for name, _ in fake_registry.calls]
    assert called_tools.count("add_speaker_message") == 1
    assert called_tools.count("count_words") == 1
    assert called_tools.count("save_draft") == 1


def test_summarizer_executes_only_first_mutating_call_per_turn() -> None:
    llm = FakeLLM(
        responses=[
            FakeResponse(
                tool_calls=[
                    _tool_call("call_1", "SPEAKER_00", "First line"),
                    _tool_call("call_2", "SPEAKER_01", "Second line"),
                ]
            )
        ]
    )
    executor = SummarizerExecutor(
        llm=llm,
        retrieval_port=12345,
        max_tool_turns=1,
        is_embedding_enabled=False,
    )
    fake_registry = FakeToolRegistry()
    messages: list[dict[str, Any]] = [{"role": "system", "content": "Summarizer"}]

    asyncio.run(
        executor.run_agent_loop(
            messages=messages,
            tools=[],
            max_turns=1,
            context_data={
                "tool_registry": fake_registry,
                "min_words": 1,
                "max_words": 100,
                "signature_dedupe_window_turns": 1,
                "replay_dedupe_tools": {
                    "add_speaker_message",
                    "edit_message",
                    "remove_message",
                    "finalize_draft",
                },
            },
        )
    )

    called_tools = [name for name, _ in fake_registry.calls]
    assert called_tools.count("add_speaker_message") == 1
    assert called_tools.count("count_words") == 1
    assert called_tools.count("save_draft") == 1

    skipped = [
        m
        for m in messages
        if m.get("role") == "tool"
        and m.get("tool_call_id") == "call_2"
        and m.get("name") == "add_speaker_message"
    ]
    assert len(skipped) == 1
    skipped_payload = json.loads(skipped[0]["content"])
    assert skipped_payload["status"] == "skipped"
    assert skipped_payload["reason"] == "single_mutating_call_per_turn"
