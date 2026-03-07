"""Unit tests for Corrector agent guidance generation."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

# Provide minimal `numpy` module stub for llm_provider typing imports.
if "numpy" not in sys.modules:
    numpy_module = types.ModuleType("numpy")

    class _NDArray:
        pass

    numpy_module.ndarray = _NDArray
    sys.modules["numpy"] = numpy_module

# Provide minimal `jinja2` module stub for prompt_manager imports only when unavailable.
try:
    import jinja2 as _jinja2
except Exception:
    _jinja2 = None

if _jinja2 is None or getattr(_jinja2, "__spec__", None) is None:
    jinja2_module = types.ModuleType("jinja2")

    class _FileSystemLoader:
        def __init__(self, _path: str) -> None:
            self.path = _path

    class _Environment:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

        def get_template(self, template_name: str):  # noqa: ANN204
            class _Template:
                def __init__(self, name: str) -> None:
                    self.name = name

                def render(self, **kwargs: Any) -> str:
                    return f"{self.name}:{kwargs}"

            return _Template(template_name)

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

from agents.corrector import (
    CorrectorExecutor,
    LLMCorrectorAgent,
    render_correction_guidance,
)
from agents.dtos import CorrectorTaskRequest
from prompt_manager import PromptManager


class _FakeLLM:
    """Simple generate-only LLM stub with queued responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
            }
        )
        if not self._responses:
            raise RuntimeError("No fake response left")
        return self._responses.pop(0)

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):  # noqa: ANN201
        del messages, tools
        raise NotImplementedError


class _FakeEventQueue:
    """Collects enqueue_event payloads from send_response."""

    def __init__(self) -> None:
        self.events: list[object] = []

    async def enqueue_event(self, event: object) -> None:
        self.events.append(event)


def setup_function(function: object) -> None:
    """Reset prompt environment for test isolation."""
    del function
    PromptManager._env = None
    PromptManager._instance = None


def test_corrector_uses_full_failure_context_without_truncation() -> None:
    """Pass full failure context into prompt without manual truncation."""
    full_context = "CTX_" + ("A" * 6_000)
    llm = _FakeLLM(
        responses=[
            json.dumps(
                {
                    "correction_instruction": "Use exact schema and IDs.",
                    "few_shot_examples": [
                        {
                            "bad_example": "bad",
                            "corrected_example": "good",
                            "rationale": "why",
                        }
                    ],
                }
            )
        ]
    )
    corrector = LLMCorrectorAgent(llm=llm, max_tokens=500, max_examples=2)
    response = corrector.build_retry_guidance(
        CorrectorTaskRequest(
            target_agent="GroundTruthCreator",
            failure_type="json_parse_failed",
            failure_context=full_context,
            latest_output="not-json",
            attempt=1,
            max_attempts=3,
            expected_contract="Return one JSON object.",
        )
    )

    assert response.correction_instruction == "Use exact schema and IDs."
    assert full_context in llm.calls[0]["user_prompt"]


def test_corrector_falls_back_when_response_is_not_json() -> None:
    """Return deterministic fallback guidance when LLM output is malformed."""
    llm = _FakeLLM(responses=["this is not json"])
    corrector = LLMCorrectorAgent(llm=llm, max_tokens=500, max_examples=2)
    response = corrector.build_retry_guidance(
        CorrectorTaskRequest(
            target_agent="Evaluator",
            failure_type="invalid_question_id",
            failure_context="question mismatch",
            latest_output="{}",
            attempt=1,
            max_attempts=2,
            expected_contract="Use exact question_id.",
        )
    )

    assert response.correction_instruction
    assert response.few_shot_examples
    rendered = render_correction_guidance(response)
    assert "<Corrector guidance>" in rendered


def test_corrector_executor_returns_json_payload() -> None:
    """Serialize Corrector guidance payload from A2A executor wrapper."""
    llm = _FakeLLM(
        responses=[
            json.dumps(
                {
                    "correction_instruction": "Fix question id.",
                    "few_shot_examples": [
                        {
                            "bad_example": "bad",
                            "corrected_example": "good",
                            "rationale": "why",
                        }
                    ],
                }
            )
        ]
    )
    executor = CorrectorExecutor(llm=llm)
    queue = _FakeEventQueue()

    asyncio.run(
        executor.handle_task(
            {
                "target_agent": "Evaluator",
                "failure_type": "invalid_question_id",
                "failure_context": "bad id",
                "latest_output": "payload",
                "attempt": 1,
                "max_attempts": 3,
                "expected_contract": "exact id",
            },
            context=None,  # type: ignore[arg-type]
            event_queue=queue,  # type: ignore[arg-type]
        )
    )

    assert len(queue.events) == 1
    payload = json.loads(str(queue.events[0]))
    assert payload["correction_instruction"] == "Fix question id."
