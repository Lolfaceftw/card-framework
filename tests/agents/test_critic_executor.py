"""Unit tests for critic deterministic-check tool handling."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import types
from typing import Any

import pytest

from tests.support.a2a import extract_agent_text_message

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

    def _new_agent_text_message(text: str, **_kwargs: object) -> str:
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

from card_framework.agents.base import BaseA2AExecutor
from card_framework.agents.loop.prompt_compaction import (
    ContextBudgetManager,
    ContextWindowProfile,
    PromptCompactionConfig,
    PromptCompactionRequest,
    PromptCompactionResult,
)
import card_framework.agents.critic as critic_module
from card_framework.agents.critic import CriticExecutor
from card_framework.agents.utils import count_words
from card_framework.audio_pipeline.calibration import VoiceCloneCalibration
from card_framework.shared.llm_provider import LLMProvider
from card_framework.shared.prompt_manager import PromptManager


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


class _LoopResponse:
    """Minimal assistant response payload for tool-loop tests."""

    def __init__(
        self,
        *,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        self._content = content
        self._tool_calls = tool_calls
        self._reasoning_content = reasoning_content

    def model_dump(self) -> dict[str, Any]:
        payload = {
            "role": "assistant",
            "content": self._content,
            "tool_calls": self._tool_calls,
        }
        if self._reasoning_content is not None:
            payload["reasoning_content"] = self._reasoning_content
        return payload


class _LoopLLM:
    """Deterministic LLM for shared-loop and critic integration tests."""

    def __init__(self, responses: list[_LoopResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | str | None = None,
        max_tokens: int | None = None,
    ) -> _LoopResponse:
        del messages, tools, tool_choice, max_tokens
        self.calls += 1
        if self._index >= len(self._responses):
            return _LoopResponse(content="")
        response = self._responses[self._index]
        self._index += 1
        return response


class _OverflowThenSuccessLLM:
    """Raise one context overflow first, then return one empty assistant response."""

    def __init__(self) -> None:
        self.calls = 0
        self.chat_messages: list[list[dict[str, Any]]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | str | None = None,
        max_tokens: int | None = None,
    ) -> _LoopResponse:
        del tools, tool_choice, max_tokens
        self.calls += 1
        self.chat_messages.append(
            [
                message.to_openai_dict()
                if hasattr(message, "to_openai_dict")
                else dict(message)
                for message in messages
            ]
        )
        if self.calls == 1:
            raise RuntimeError(
                "This model's maximum context length is 131072 tokens. "
                "However, you requested 133783 tokens."
            )
        return _LoopResponse(content="")


class _StaticPromptCompactionStrategy:
    """Return a deterministic compacted context payload for overflow retries."""

    def __init__(self) -> None:
        self.calls = 0

    def compact(
        self,
        *,
        request: PromptCompactionRequest,
    ) -> PromptCompactionResult:
        self.calls += 1
        del request
        return PromptCompactionResult(
            compressed_transcript_context="short transcript grounding",
            compressed_history_context="short history context",
            compression_reason="test_compaction",
            source_iteration_span=(0, 1),
            estimated_tokens_before=100,
            estimated_tokens_after=20,
        )


class _NullResponseCallback:
    """Ignore streamed callback events in unit-test providers."""

    def on_start(self, agent_name: str) -> None:
        del agent_name

    def on_thought_token(self, token: str) -> None:
        del token

    def on_content_token(self, token: str) -> None:
        del token

    def on_complete(self) -> None:
        return None


class _StreamingLoopLLM(LLMProvider):
    """Deterministic streaming LLM stub that can be interrupted by callbacks."""

    def __init__(self, responses: list[dict[str, Any] | _LoopResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls = 0
        self.chat_messages: list[list[dict[str, Any]]] = []
        self._response_callback = _NullResponseCallback()

    def set_response_callback(self, response_callback: Any) -> None:
        self._response_callback = response_callback

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        del system_prompt, user_prompt, max_tokens
        raise NotImplementedError

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | str | None = None,
        max_tokens: int | None = None,
    ) -> _LoopResponse:
        del tools, tool_choice, max_tokens
        self.calls += 1
        self.chat_messages.append(
            [
                message.to_openai_dict()
                if hasattr(message, "to_openai_dict")
                else dict(message)
                for message in messages
            ]
        )
        if self._index >= len(self._responses):
            return _LoopResponse(content="")
        response = self._responses[self._index]
        self._index += 1
        if isinstance(response, _LoopResponse):
            return response

        reasoning_tokens = [
            str(token) for token in response.get("reasoning_tokens", [])
        ]
        content_tokens = [str(token) for token in response.get("content_tokens", [])]
        self._response_callback.on_start("Critic")
        try:
            for token in reasoning_tokens:
                self._response_callback.on_thought_token(token)
            for token in content_tokens:
                self._response_callback.on_content_token(token)
        finally:
            self._response_callback.on_complete()

        return _LoopResponse(
            content="".join(content_tokens),
            tool_calls=response.get("tool_calls"),
            reasoning_content="".join(reasoning_tokens) or None,
        )


class _TaggedResponseCallback:
    """Record which runtime callback handled one streamed response."""

    def __init__(self, tag: str, started_tags: list[str]) -> None:
        self._tag = tag
        self._started_tags = started_tags

    def on_start(self, agent_name: str) -> None:
        del agent_name
        self._started_tags.append(self._tag)

    def on_thought_token(self, token: str) -> None:
        del token

    def on_content_token(self, token: str) -> None:
        del token

    def on_complete(self) -> None:
        return None


class _BlockingStreamingLLM(LLMProvider):
    """Block the first streamed chat so a second concurrent call can race it."""

    def __init__(self) -> None:
        self._response_callback: Any = _NullResponseCallback()
        self._call_count = 0
        self._call_count_lock = threading.Lock()
        self.first_started = threading.Event()
        self.second_started = threading.Event()
        self.release_first = threading.Event()

    def set_response_callback(self, response_callback: Any) -> None:
        self._response_callback = response_callback

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        del system_prompt, user_prompt, max_tokens
        raise NotImplementedError

    def chat(
        self,
        messages: list[Any],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | str | None = None,
        max_tokens: int | None = None,
    ) -> _LoopResponse:
        del messages, tools, tool_choice, max_tokens
        with self._call_count_lock:
            self._call_count += 1
            call_index = self._call_count

        self._response_callback.on_start("Critic")
        try:
            if call_index == 1:
                self.first_started.set()
                if not self.release_first.wait(timeout=5.0):
                    raise AssertionError("Timed out waiting to release the first call.")
            else:
                self.second_started.set()
            self._response_callback.on_content_token(f"call-{call_index}")
        finally:
            self._response_callback.on_complete()
        return _LoopResponse(content=f"call-{call_index}")


class _LoopGuardExecutor(BaseA2AExecutor):
    """Tiny concrete executor used to exercise the shared loop guard."""

    def __init__(self, llm: _LoopLLM) -> None:
        super().__init__("LoopGuard")
        self.llm = llm

    async def handle_task(self, task_data, context, event_queue) -> None:  # noqa: ANN001
        del task_data, context, event_queue


class _CallbackLockExecutor(BaseA2AExecutor):
    """Concrete executor used to exercise per-call stream callback wiring."""

    def __init__(self, llm: LLMProvider) -> None:
        super().__init__("CallbackLock")
        self.llm = llm

    async def handle_task(self, task_data, context, event_queue) -> None:  # noqa: ANN001
        del task_data, context, event_queue


class _RecordingQueue:
    """Collect response events emitted by send_response."""

    def __init__(self) -> None:
        self.events: list[object] = []

    async def enqueue_event(self, event: object) -> None:
        self.events.append(event)


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


def test_run_deterministic_checks_prefers_live_draft_audio_state(
    tmp_path: Path,
) -> None:
    """Use persisted live-draft audio durations when the state snapshot matches."""
    speaker_manifest_path = tmp_path / "speaker_samples" / "manifest.json"
    speaker_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    speaker_manifest_path.write_text('{"samples": []}', encoding="utf-8")
    output_audio_path = tmp_path / "voice_clone" / "turn.wav"
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    output_audio_path.write_bytes(b"wav")
    draft = "<SPEAKER_00>Short finalized draft.</SPEAKER_00>"
    state_path = tmp_path / "live_draft.state.json"
    state_path.write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-03-07T00:00:00+00:00",
                "speaker_samples_manifest_path": str(speaker_manifest_path),
                "synthesis_count": 1,
                "current_snapshot": [
                    {
                        "line": 1,
                        "turn_id": "turn-001",
                        "speaker_id": "SPEAKER_00",
                        "content": "Short finalized draft.",
                        "emo_preset": "neutral",
                    }
                ],
                "turn_audio": {
                    "turn-001": {
                        "turn_id": "turn-001",
                        "speaker": "SPEAKER_00",
                        "text": "Short finalized draft.",
                        "emo_preset": "neutral",
                        "output_audio_path": str(output_audio_path),
                        "duration_ms": 2_400,
                        "word_count": 3,
                        "actual_wpm": 75.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    executor = CriticExecutor(
        llm=_FakeLLM(),
        calibration=_build_calibration(),
        is_embedding_enabled=False,
    )
    messages: list[dict[str, Any]] = []

    asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_4",
                    "name": "run_deterministic_checks",
                    "arguments": {"draft_text": draft},
                }
            ],
            messages=messages,
            context_data={
                "draft": draft,
                "target_seconds": 2,
                "duration_tolerance_ratio": 0.25,
                "draft_audio_state_path": state_path,
            },
        )
    )
    tool_msg = next(m for m in messages if m.get("name") == "run_deterministic_checks")
    payload = json.loads(tool_msg["content"])

    assert payload["duration_source"] == "actual_audio"
    assert payload["actual_estimated_seconds"] == 2.4
    assert payload["status"] == "pass"


def test_run_deterministic_checks_falls_back_to_live_draft_turn_dir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Use rendered turn WAVs when the live-draft sidecar is unavailable."""
    turn_dir = tmp_path / "voice_clone" / "live_draft_turns"
    turn_dir.mkdir(parents=True)
    first_turn = turn_dir / "turn-001.wav"
    second_turn = turn_dir / "turn-002.wav"
    first_turn.write_bytes(b"wav1")
    second_turn.write_bytes(b"wav2")
    state_path = tmp_path / "voice_clone" / "live_draft.state.json"
    draft = (
        "<SPEAKER_00>Short finalized draft.</SPEAKER_00>\n"
        "<SPEAKER_01>Second short reply.</SPEAKER_01>"
    )

    durations = {
        first_turn.resolve(): 2_400,
        second_turn.resolve(): 3_600,
    }

    def _fake_probe_audio_duration_ms(audio_path: Path) -> int | None:
        return durations.get(audio_path.resolve())

    monkeypatch.setattr(
        critic_module,
        "probe_audio_duration_ms",
        _fake_probe_audio_duration_ms,
    )

    executor = CriticExecutor(
        llm=_FakeLLM(),
        calibration=_build_calibration(),
        is_embedding_enabled=False,
    )

    payload = executor._run_deterministic_checks(
        draft=draft,
        target_seconds=5,
        duration_tolerance_ratio=0.25,
        draft_audio_state_path=state_path,
    )

    assert payload["duration_source"] == "actual_audio_dir"
    assert payload["actual_estimated_seconds"] == 6.0
    assert payload["status"] == "pass"


def test_run_deterministic_checks_accepts_terminal_quoted_sentence() -> None:
    """Treat closing quotes after terminal punctuation as a valid sentence end."""
    draft = '<SPEAKER_00>Edward says it is "tolerable."</SPEAKER_00>'
    executor = CriticExecutor(
        llm=_FakeLLM(),
        calibration=_build_calibration(),
        is_embedding_enabled=False,
    )
    messages: list[dict[str, Any]] = []

    asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_quoted",
                    "name": "run_deterministic_checks",
                    "arguments": {"draft_text": draft},
                }
            ],
            messages=messages,
            context_data={
                "draft": draft,
                "target_seconds": 2,
                "duration_tolerance_ratio": 0.8,
            },
        )
    )
    tool_msg = next(m for m in messages if m.get("name") == "run_deterministic_checks")
    payload = json.loads(tool_msg["content"])

    assert payload["status"] == "pass"
    assert payload["failures"] == []


def test_critic_system_prompt_relaxes_short_form_coverage_requirements() -> None:
    """Use representative-arc coverage rules for 30-second targets."""
    prompt = PromptManager.get_prompt(
        "critic_system",
        target_seconds=30,
        target_minutes=0.5,
        duration_tolerance_ratio=0.05,
        duration_tolerance_percent=5.0,
        min_seconds=28.5,
        max_seconds=31.5,
        is_embedding_enabled=False,
    )

    assert "SHORT-FORM COVERAGE AND PROGRESSION" in prompt
    assert "representative beginning, core middle, and ending arc" in prompt
    assert "Omitting secondary subtopics or supporting analysis is acceptable" in prompt
    assert "LONG-FORM TRANSCRIPT COVERAGE AND PROGRESSION" not in prompt


def test_critic_system_prompt_uses_medium_form_coverage_for_one_minute_targets() -> None:
    """Require major-arc coverage, not exhaustive coverage, for 1-minute targets."""
    prompt = PromptManager.get_prompt(
        "critic_system",
        target_seconds=60,
        target_minutes=1.0,
        duration_tolerance_ratio=0.05,
        duration_tolerance_percent=5.0,
        min_seconds=57.0,
        max_seconds=63.0,
        is_embedding_enabled=False,
    )

    assert "MEDIUM-FORM COVERAGE AND PROGRESSION" in prompt
    assert "major middle beats" in prompt
    assert "MAY omit secondary subtopics" in prompt
    assert "LONG-FORM TRANSCRIPT COVERAGE AND PROGRESSION" not in prompt


def test_critic_system_prompt_uses_short_form_boundary_at_forty_five_seconds() -> None:
    """Keep exact 45-second targets in the short-form rubric bucket."""
    prompt = PromptManager.get_prompt(
        "critic_system",
        target_seconds=45,
        target_minutes=0.75,
        duration_tolerance_ratio=0.05,
        duration_tolerance_percent=5.0,
        min_seconds=42.75,
        max_seconds=47.25,
        is_embedding_enabled=False,
    )

    assert "SHORT-FORM COVERAGE AND PROGRESSION" in prompt
    assert "MEDIUM-FORM COVERAGE AND PROGRESSION" not in prompt


def test_critic_system_prompt_uses_medium_form_boundary_at_ninety_seconds() -> None:
    """Keep exact 90-second targets in the medium-form rubric bucket."""
    prompt = PromptManager.get_prompt(
        "critic_system",
        target_seconds=90,
        target_minutes=1.5,
        duration_tolerance_ratio=0.05,
        duration_tolerance_percent=5.0,
        min_seconds=85.5,
        max_seconds=94.5,
        is_embedding_enabled=False,
    )

    assert "MEDIUM-FORM COVERAGE AND PROGRESSION" in prompt
    assert "LONG-FORM TRANSCRIPT COVERAGE AND PROGRESSION" not in prompt


def test_critic_system_prompt_keeps_full_scope_for_long_form_targets() -> None:
    """Keep the strict full-scope requirement for long-form targets."""
    prompt = PromptManager.get_prompt(
        "critic_system",
        target_seconds=300,
        target_minutes=5.0,
        duration_tolerance_ratio=0.05,
        duration_tolerance_percent=5.0,
        min_seconds=285.0,
        max_seconds=315.0,
        is_embedding_enabled=False,
    )

    assert "LONG-FORM TRANSCRIPT COVERAGE AND PROGRESSION" in prompt
    assert "full chronological scope of the provided transcript" in prompt
    assert "Omitting secondary subtopics or supporting analysis is acceptable" not in prompt


def test_run_agent_loop_stops_on_repeated_no_tool_span() -> None:
    """Trip the shared loop guard when one assistant turn repeats itself."""
    repeated_text = (
        "Wait, the transcript says Ashley said the line already. "
        "Wait, the transcript says Ashley said the line already. "
        "Wait, the transcript says Ashley said the line already."
    )
    llm = _LoopLLM([_LoopResponse(content=repeated_text)])
    executor = _LoopGuardExecutor(llm)
    runtime_context = {
        "reasoning_loop_guard": {
            "enabled": True,
            "single_response_repeat_threshold": 3,
            "min_repeated_sentence_words": 8,
        }
    }

    result = asyncio.run(
        executor.run_agent_loop(
            messages=[{"role": "system", "content": "Critic"}],
            tools=[],
            max_turns=3,
            context_data=runtime_context,
        )
    )

    assert result is None
    assert llm.calls == 1
    assert runtime_context["loop_stop_category"] == "reasoning_loop_guard"
    assert "single-response repetition" in runtime_context["loop_stop_reason"]


def test_run_agent_loop_stops_on_repeated_no_tool_turns() -> None:
    """Trip the shared loop guard when consecutive no-tool turns stay near-identical."""
    llm = _LoopLLM(
        [
            _LoopResponse(
                content=(
                    "Let me re-check the transcript for whether Ashley said the line "
                    "about everybody loving that movie so much."
                )
            ),
            _LoopResponse(
                content=(
                    "Let me re-check the transcript for whether Ashley said the line "
                    "about everybody loving that movie so much."
                )
            ),
        ]
    )
    executor = _LoopGuardExecutor(llm)
    runtime_context = {
        "reasoning_loop_guard": {
            "enabled": True,
            "max_consecutive_no_tool_turns": 2,
            "min_repeated_sentence_words": 8,
            "cross_turn_similarity_threshold": 0.96,
        }
    }

    result = asyncio.run(
        executor.run_agent_loop(
            messages=[{"role": "system", "content": "Critic"}],
            tools=[],
            max_turns=4,
            context_data=runtime_context,
        )
    )

    assert result is None
    assert llm.calls == 2
    assert runtime_context["loop_stop_category"] == "reasoning_loop_guard"
    assert "cross-turn repetition" in runtime_context["loop_stop_reason"]


def test_run_agent_loop_retries_after_context_overflow_with_prompt_compaction() -> None:
    """Compact and retry the same turn once when the provider rejects prompt size."""
    llm = _OverflowThenSuccessLLM()
    executor = _LoopGuardExecutor(llm)
    compaction_strategy = _StaticPromptCompactionStrategy()
    runtime_context = {
        "context_budget_manager": ContextBudgetManager(
            config=PromptCompactionConfig(
                enabled=True,
                trigger_ratio=0.8,
                retry_on_overflow=True,
                max_overflow_retries=1,
                keep_latest_iterations=1,
            ),
            profiles={
                "deepseek_chat": ContextWindowProfile(
                    profile_name="deepseek_chat",
                    context_window_tokens=100,
                )
            },
            token_estimator=lambda _blocks: 10,
        ),
        "prompt_compaction_profile_name": "deepseek_chat",
        "prompt_compaction_strategy": compaction_strategy,
        "prompt_compaction_keep_latest_iterations": 1,
        "prompt_compaction_request_builder": (
            lambda **kwargs: PromptCompactionRequest(
                profile_name="deepseek_chat",
                current_draft="draft",
                latest_feedback="feedback",
                loop_context="loop",
                transcript_context="transcript",
                messages=kwargs["messages"],
            )
        ),
    }

    result = asyncio.run(
        executor.run_agent_loop(
            messages=[
                {"role": "system", "content": "Critic"},
                {"role": "user", "content": "Large prompt context"},
            ],
            tools=[],
            max_turns=1,
            context_data=runtime_context,
        )
    )

    assert result is None
    assert llm.calls == 2
    assert compaction_strategy.calls == 1
    assert any(
        "Compacted prior summarizer context" in str(message.get("content", ""))
        for message in llm.chat_messages[1]
    )


def test_handle_task_falls_back_with_loop_guard_prefix() -> None:
    """Prefix fallback feedback when the critic loop guard stops repetitive text."""
    llm = _LoopLLM(
        [
            _LoopResponse(
                content=(
                    "Wait, the transcript says Ashley said this sentence already. "
                    "Wait, the transcript says Ashley said this sentence already. "
                    "Wait, the transcript says Ashley said this sentence already."
                )
            )
        ]
    )
    executor = CriticExecutor(
        llm=llm,
        calibration=_build_calibration(),
        is_embedding_enabled=False,
    )
    queue = _RecordingQueue()
    original_get_prompt = PromptManager.__dict__["get_prompt"]

    def _fake_get_prompt(cls, key: str, **kwargs: Any) -> str:
        if key == "critic_user":
            return str(kwargs.get("draft", ""))
        return f"{key} prompt"

    PromptManager.get_prompt = classmethod(_fake_get_prompt)
    try:
        asyncio.run(
            executor.handle_task(
                {
                    "draft": "<SPEAKER_00>Short finalized draft.</SPEAKER_00>",
                    "target_seconds": 60,
                    "duration_tolerance_ratio": 0.05,
                    "full_transcript": "[E0001] [SPEAKER_00] Source text",
                },
                context=SimpleNamespace(task_id=None, context_id=None),
                event_queue=queue,  # type: ignore[arg-type]
            )
        )
    finally:
        PromptManager.get_prompt = original_get_prompt

    payload = json.loads(extract_agent_text_message(queue.events[0]))
    assert payload["feedback"].startswith("CRITIC LOOP GUARD:")
    assert "single-response repetition" in payload["feedback"]


def test_handle_task_requires_model_verdict_in_strict_mode() -> None:
    """Fail a strict critic pass when the model never submits a verdict."""
    repeated = (
        "Wait, the transcript says Ashley said this sentence already. "
        "Wait, the transcript says Ashley said this sentence already. "
        "Wait, the transcript says Ashley said this sentence already."
    )
    executor = CriticExecutor(
        llm=_LoopLLM([_LoopResponse(content=repeated)]),
        calibration=_build_calibration(),
        is_embedding_enabled=False,
        require_structured_verdict=True,
    )
    original_get_prompt = PromptManager.__dict__["get_prompt"]

    def _fake_get_prompt(cls, key: str, **kwargs: Any) -> str:
        """Return minimal prompts while retaining the submitted draft text."""
        if key == "critic_user":
            return str(kwargs.get("draft", ""))
        return f"{key} prompt"

    PromptManager.get_prompt = classmethod(_fake_get_prompt)
    try:
        with pytest.raises(
            RuntimeError,
            match="Critic completed without calling submit_verdict",
        ):
            asyncio.run(
                executor.handle_task(
                    {
                        "draft": "<SPEAKER_00>Short finalized draft.</SPEAKER_00>",
                        "target_seconds": 60,
                        "duration_tolerance_ratio": 0.05,
                        "full_transcript": "[E0001] [SPEAKER_00] Source text",
                    },
                    context=SimpleNamespace(task_id=None, context_id=None),
                    event_queue=_RecordingQueue(),  # type: ignore[arg-type]
                )
            )
    finally:
        PromptManager.get_prompt = original_get_prompt


def test_handle_task_retries_after_stream_repetition_abort() -> None:
    """Retry the critic once when stream-time repetition is detected."""
    draft = "<SPEAKER_00>Short finalized draft.</SPEAKER_00>"
    repeated_chunk = "Wait, the transcript says Ashley said the line already. "
    llm = _StreamingLoopLLM(
        [
            {"content_tokens": [repeated_chunk, repeated_chunk, repeated_chunk]},
            _LoopResponse(
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "submit_verdict",
                            "arguments": json.dumps(
                                {
                                    "status": "pass",
                                    "actual_word_count": 3,
                                    "estimated_seconds": 1.5,
                                    "feedback": "ok after retry",
                                }
                            ),
                        },
                    }
                ]
            ),
        ]
    )
    executor = CriticExecutor(
        llm=llm,
        calibration=_build_calibration(),
        is_embedding_enabled=False,
        reasoning_loop_guard={"max_stream_abort_retries": 1},
    )
    queue = _RecordingQueue()
    original_get_prompt = PromptManager.__dict__["get_prompt"]

    def _fake_get_prompt(cls, key: str, **kwargs: Any) -> str:
        if key == "critic_user":
            return str(kwargs.get("draft", ""))
        return f"{key} prompt"

    PromptManager.get_prompt = classmethod(_fake_get_prompt)
    try:
        asyncio.run(
            executor.handle_task(
                {
                    "draft": draft,
                    "target_seconds": 60,
                    "duration_tolerance_ratio": 0.05,
                    "full_transcript": "[E0001] [SPEAKER_00] Source text",
                },
                context=SimpleNamespace(task_id=None, context_id=None),
                event_queue=queue,  # type: ignore[arg-type]
            )
        )
    finally:
        PromptManager.get_prompt = original_get_prompt

    payload = json.loads(extract_agent_text_message(queue.events[0]))
    assert payload == {
        "status": "pass",
        "word_count": 3,
        "estimated_seconds": 1.5,
        "feedback": "ok after retry",
    }
    assert llm.calls == 2
    assert llm.chat_messages[1][-1]["role"] == "user"
    assert (
        "interrupted because you started repeating the same"
        in (llm.chat_messages[1][-1]["content"])
    )
    assert "Retry the same critique now." in llm.chat_messages[1][-1]["content"]


def test_handle_task_falls_back_after_repeated_stream_abort() -> None:
    """Use deterministic fallback when the retried critic stream loops again."""
    repeated_chunk = "Wait, the transcript says Ashley said the line already. "
    llm = _StreamingLoopLLM(
        [
            {"content_tokens": [repeated_chunk, repeated_chunk, repeated_chunk]},
            {"content_tokens": [repeated_chunk, repeated_chunk, repeated_chunk]},
        ]
    )
    executor = CriticExecutor(
        llm=llm,
        calibration=_build_calibration(),
        is_embedding_enabled=False,
        reasoning_loop_guard={"max_stream_abort_retries": 1},
    )
    queue = _RecordingQueue()
    original_get_prompt = PromptManager.__dict__["get_prompt"]

    def _fake_get_prompt(cls, key: str, **kwargs: Any) -> str:
        if key == "critic_user":
            return str(kwargs.get("draft", ""))
        return f"{key} prompt"

    PromptManager.get_prompt = classmethod(_fake_get_prompt)
    try:
        asyncio.run(
            executor.handle_task(
                {
                    "draft": "<SPEAKER_00>Short finalized draft.</SPEAKER_00>",
                    "target_seconds": 60,
                    "duration_tolerance_ratio": 0.05,
                    "full_transcript": "[E0001] [SPEAKER_00] Source text",
                },
                context=SimpleNamespace(task_id=None, context_id=None),
                event_queue=queue,  # type: ignore[arg-type]
            )
        )
    finally:
        PromptManager.get_prompt = original_get_prompt

    payload = json.loads(extract_agent_text_message(queue.events[0]))
    assert payload["feedback"].startswith("CRITIC LOOP GUARD:")
    assert "stream repetition detected" in payload["feedback"]
    assert llm.calls == 2


def test_chat_with_runtime_stream_callback_serializes_shared_provider_swaps() -> None:
    """Serialize temporary callback swaps so concurrent chats do not cross-wire."""
    llm = _BlockingStreamingLLM()
    executor = _CallbackLockExecutor(llm)
    callback_starts: list[str] = []

    async def _run_chat(tag: str) -> _LoopResponse:
        return await executor._chat_with_runtime_stream_callback(
            chat_kwargs={"messages": [], "tools": []},
            runtime_context={
                "stream_response_callback_factory": (
                    lambda existing: _TaggedResponseCallback(tag, callback_starts)
                )
            },
        )

    async def _exercise() -> None:
        first_task = asyncio.create_task(_run_chat("first"))
        await asyncio.to_thread(llm.first_started.wait, 1.0)

        second_task = asyncio.create_task(_run_chat("second"))
        await asyncio.sleep(0.1)
        assert not llm.second_started.is_set()

        llm.release_first.set()
        first_result = await first_task
        second_result = await second_task

        assert first_result.model_dump()["content"] == "call-1"
        assert second_result.model_dump()["content"] == "call-2"
        assert callback_starts == ["first", "second"]

    asyncio.run(_exercise())


def test_handle_task_preserves_normal_tool_flow() -> None:
    """Keep the standard structured critic loop behavior unchanged."""
    draft = "<SPEAKER_00>Short finalized draft.</SPEAKER_00>"
    llm = _LoopLLM(
        [
            _LoopResponse(
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "run_deterministic_checks",
                            "arguments": json.dumps({"draft_text": draft}),
                        },
                    }
                ]
            ),
            _LoopResponse(
                tool_calls=[
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "submit_verdict",
                            "arguments": json.dumps(
                                {
                                    "status": "pass",
                                    "actual_word_count": 3,
                                    "estimated_seconds": 1.5,
                                    "feedback": "ok",
                                }
                            ),
                        },
                    }
                ]
            ),
        ]
    )
    executor = CriticExecutor(
        llm=llm,
        calibration=_build_calibration(),
        is_embedding_enabled=False,
    )
    queue = _RecordingQueue()
    original_get_prompt = PromptManager.__dict__["get_prompt"]

    def _fake_get_prompt(cls, key: str, **kwargs: Any) -> str:
        if key == "critic_user":
            return str(kwargs.get("draft", ""))
        return f"{key} prompt"

    PromptManager.get_prompt = classmethod(_fake_get_prompt)
    try:
        asyncio.run(
            executor.handle_task(
                {
                    "draft": draft,
                    "target_seconds": 60,
                    "duration_tolerance_ratio": 0.05,
                    "full_transcript": "[E0001] [SPEAKER_00] Source text",
                },
                context=SimpleNamespace(task_id=None, context_id=None),
                event_queue=queue,  # type: ignore[arg-type]
            )
        )
    finally:
        PromptManager.get_prompt = original_get_prompt

    payload = json.loads(extract_agent_text_message(queue.events[0]))
    assert payload == {
        "status": "pass",
        "word_count": 3,
        "estimated_seconds": 1.5,
        "feedback": "ok",
    }

