"""Unit tests for critic deterministic-check tool handling."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
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

from agents.critic import CriticExecutor
from agents.utils import count_words
from audio_pipeline.calibration import VoiceCloneCalibration
from prompt_manager import PromptManager


def setup_function(function: object) -> None:
    """Reset global prompt template environment for test isolation."""
    del function
    PromptManager._env = None


class _FakeLLM:
    """Minimal LLM stub for constructor compatibility in unit tests."""

    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]):  # noqa: ANN201
        raise NotImplementedError

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        del system_prompt, user_prompt, max_tokens
        raise NotImplementedError


class _FakeAgentClient:
    """Fake transport used to verify dependency injection paths."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def send_task(
        self,
        port: int,
        task_data: object,
        timeout: float = 120.0,
        max_retries: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del task_data, max_retries, metadata
        self.calls.append({"port": port, "timeout": timeout})
        return json.dumps({"segments": [], "total_words": 0})


def _build_calibration() -> VoiceCloneCalibration:
    """Return a minimal calibration object for deterministic duration tests."""
    return VoiceCloneCalibration(
        artifact_path=Path("artifact.json"),
        generated_at_utc="2026-03-07T00:00:00+00:00",
        speaker_samples_manifest_path=Path("."),
        preset_emo_texts={"neutral": "Speak in a calm, neutral tone."},
        calibration_phrases=("Hello, world.",),
        speaker_preset_wpm={"SPEAKER_00": {"neutral": 120.0}},
        preset_default_wpm={"neutral": 120.0},
    )


def test_run_deterministic_checks_uses_canonical_task_draft() -> None:
    """Ignore LLM-provided draft payload and evaluate the server-side finalized draft."""
    executor = CriticExecutor(
        llm=_FakeLLM(),
        calibration=_build_calibration(),
        is_embedding_enabled=False,
    )
    canonical_draft = "<SPEAKER_00>Short finalized draft.</SPEAKER_00>"
    oversized_payload = f"{canonical_draft}\n\n--- FULL TRANSCRIPT ---\n" + (
        "word " * 1000
    )

    should_break, final_verdict = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "run_deterministic_checks",
                    "arguments": {"draft_text": oversized_payload},
                }
            ],
            messages=[],
            context_data={
                "draft": canonical_draft,
                "target_seconds": 2,
                "duration_tolerance_ratio": 0.6,
            },
        )
    )

    assert should_break is False
    assert final_verdict is None

    # Replay once to inspect emitted tool payload.
    messages: list[dict[str, Any]] = []
    asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "run_deterministic_checks",
                    "arguments": {"draft_text": oversized_payload},
                }
            ],
            messages=messages,
            context_data={
                "draft": canonical_draft,
                "target_seconds": 2,
                "duration_tolerance_ratio": 0.6,
            },
        )
    )
    tool_msg = next(m for m in messages if m.get("name") == "run_deterministic_checks")
    payload = json.loads(tool_msg["content"])

    assert payload["actual_word_count"] == count_words(canonical_draft)
    assert payload["status"] == "pass"


def test_submit_verdict_returns_final_result() -> None:
    """Return a final verdict payload when submit_verdict is called."""
    executor = CriticExecutor(
        llm=_FakeLLM(),
        calibration=_build_calibration(),
        is_embedding_enabled=False,
    )
    should_break, final_verdict = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_2",
                    "name": "submit_verdict",
                    "arguments": {
                        "status": "pass",
                        "actual_word_count": 5,
                        "estimated_seconds": 2.5,
                        "feedback": "ok",
                    },
                }
            ],
            messages=[],
            context_data={
                "draft": "<SPEAKER_00>Text.</SPEAKER_00>",
                "target_seconds": 2,
                "duration_tolerance_ratio": 0.6,
            },
        )
    )

    assert should_break is True
    assert final_verdict == {
        "status": "pass",
        "word_count": 5,
        "estimated_seconds": 2.5,
        "feedback": "ok",
    }


def test_verify_against_transcript_uses_injected_agent_client() -> None:
    """Route transcript verification through the executor-injected client."""
    fake_client = _FakeAgentClient()
    executor = CriticExecutor(
        llm=_FakeLLM(),
        calibration=_build_calibration(),
        is_embedding_enabled=True,
        retrieval_port=9123,
        agent_client=fake_client,
    )

    should_break, final_verdict = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_3",
                    "name": "verify_against_transcript",
                    "arguments": {"query": "architecture"},
                }
            ],
            messages=[],
            context_data={
                "draft": "<SPEAKER_00>Text.</SPEAKER_00>",
                "target_seconds": 2,
                "duration_tolerance_ratio": 0.6,
            },
        )
    )

    assert should_break is False
    assert final_verdict is None
    assert fake_client.calls == [{"port": 9123, "timeout": 120.0}]
