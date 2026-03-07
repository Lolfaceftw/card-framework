"""Unit tests for GroundTruthCreatorExecutor."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any

import pytest

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

from agents.ground_truth_creator import GroundTruthCreatorExecutor
from agents.dtos import (
    CorrectorFewShotExample,
    CorrectorTaskRequest,
    CorrectorTaskResponse,
)
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


class _FakeCorrector:
    """Deterministic corrector stub that records incoming failure context."""

    def __init__(self) -> None:
        self.requests: list[CorrectorTaskRequest] = []

    def build_retry_guidance(
        self, request: CorrectorTaskRequest
    ) -> CorrectorTaskResponse:
        self.requests.append(request)
        return CorrectorTaskResponse(
            correction_instruction="Return JSON only and follow exact question IDs.",
            few_shot_examples=[
                CorrectorFewShotExample(
                    bad_example="Here are the questions: [...]",
                    corrected_example='{"questions":[{"question_id":"Q001","dimension":"factualness","question_text":"...","expected_answer":"yes","source_evidence_ids":["E0001"],"speaker_ids":["SPEAKER_00"]}]}',
                    rationale="No prose wrapper and strict schema fields.",
                )
            ],
        )


def _build_batch_payload(count: int, *, start_index: int = 1) -> str:
    questions: list[dict[str, object]] = []
    for offset in range(count):
        index = start_index + offset
        questions.append(
            {
                "question_id": f"X{index:03d}",
                "dimension": "factualness",
                "question_text": f"Fact question {index}?",
                "expected_answer": "yes",
                "source_evidence_ids": ["E0001"],
                "speaker_ids": ["SPEAKER_00"],
            }
        )
    return json.dumps({"questions": questions})


def _build_batch_dict(count: int) -> dict[str, object]:
    """Build parsed batch payload for direct validator tests."""
    return json.loads(_build_batch_payload(count))


def _build_batch_with_prompt(count: int, prompt: str, *, start_index: int = 1) -> str:
    """Build one batch payload with deterministic question text prefix."""
    questions: list[dict[str, object]] = []
    for offset in range(count):
        index = start_index + offset
        questions.append(
            {
                "question_id": f"X{index:03d}",
                "dimension": "factualness",
                "question_text": f"{prompt} {index}?",
                "expected_answer": "yes",
                "source_evidence_ids": ["E0001"],
                "speaker_ids": ["SPEAKER_00"],
            }
        )
    return json.dumps({"questions": questions})


def setup_function(function: object) -> None:
    """Reset prompt environment for test isolation."""
    del function
    PromptManager._env = None
    PromptManager._instance = None


def test_ground_truth_creator_retries_until_valid_payload() -> None:
    fake_llm = _FakeLLM(
        responses=[
            "not-json",
            _build_batch_payload(25, start_index=1),
            _build_batch_payload(25, start_index=26),
            _build_batch_payload(25, start_index=51),
            _build_batch_payload(25, start_index=76),
        ]
    )
    executor = GroundTruthCreatorExecutor(llm=fake_llm, max_generation_attempts=2)
    queue = _FakeEventQueue()

    asyncio.run(
        executor.handle_task(
            {"source_text": "[E0001] [SPEAKER_00] Source line"},
            context=None,  # type: ignore[arg-type]
            event_queue=queue,  # type: ignore[arg-type]
        )
    )

    assert len(fake_llm.calls) == 5
    assert len(queue.events) == 1
    payload = json.loads(str(queue.events[0]))
    assert len(payload["questions"]) == 100


def test_generate_batch_applies_corrector_guidance_on_retry() -> None:
    """Inject corrector instructions into next attempt when first attempt fails."""
    fake_llm = _FakeLLM(
        responses=[
            "not-json",
            _build_batch_payload(25, start_index=1),
        ]
    )
    fake_corrector = _FakeCorrector()
    executor = GroundTruthCreatorExecutor(
        llm=fake_llm,
        max_generation_attempts=2,
        corrector_agent=fake_corrector,
    )

    batch = executor._generate_batch(
        system_prompt="system",
        source_text="[E0001] [SPEAKER_00] Source line",
        dimension="factualness",
        expected_ids=[f"Q{index:03d}" for index in range(1, 26)],
    )

    assert len(batch) == 25
    assert len(fake_corrector.requests) == 1
    assert "batch_prompt=" in fake_corrector.requests[0].failure_context
    assert "<Corrector guidance>" in fake_llm.calls[1]["user_prompt"]
    assert (
        "Return JSON only and follow exact question IDs."
        in fake_llm.calls[1]["user_prompt"]
    )


def test_ground_truth_creator_does_not_use_candidate_summary_field() -> None:
    secret = "<SPEAKER_99>SECRET_CANDIDATE</SPEAKER_99>"
    fake_llm = _FakeLLM(
        responses=[
            _build_batch_payload(25, start_index=1),
            _build_batch_payload(25, start_index=26),
            _build_batch_payload(25, start_index=51),
            _build_batch_payload(25, start_index=76),
        ]
    )
    executor = GroundTruthCreatorExecutor(llm=fake_llm, max_generation_attempts=1)
    queue = _FakeEventQueue()

    asyncio.run(
        executor.handle_task(
            {
                "source_text": "[E0001] [SPEAKER_00] Source line",
                "summary_xml": secret,
            },
            context=None,  # type: ignore[arg-type]
            event_queue=queue,  # type: ignore[arg-type]
        )
    )

    assert len(fake_llm.calls) == 4
    assert secret not in fake_llm.calls[0]["system_prompt"]
    assert secret not in fake_llm.calls[0]["user_prompt"]


def test_validate_batch_payload_truncates_overproduced_questions() -> None:
    executor = GroundTruthCreatorExecutor(llm=_FakeLLM([]), max_generation_attempts=1)
    batch = executor._validate_batch_payload(
        payload=_build_batch_dict(26),
        dimension="factualness",
        expected_ids=[f"Q{index:03d}" for index in range(1, 26)],
    )
    assert batch is not None
    assert len(batch) == 25


def test_ground_truth_creator_rejects_non_contract_question_split() -> None:
    executor = GroundTruthCreatorExecutor(
        llm=_FakeLLM(
            [
                _build_batch_payload(25, start_index=1),
                _build_batch_payload(25, start_index=26),
                _build_batch_payload(25, start_index=51),
                _build_batch_payload(25, start_index=76),
                _build_batch_payload(25, start_index=101),
            ]
        ),
        max_generation_attempts=1,
    )
    queue = _FakeEventQueue()
    with pytest.raises(ValueError, match="exactly 50 factualness questions"):
        asyncio.run(
            executor.handle_task(
                {
                    "source_text": "[E0001] [SPEAKER_00] Source line",
                    "factual_question_count": 40,
                    "naturalness_question_count": 60,
                },
                context=None,  # type: ignore[arg-type]
                event_queue=queue,  # type: ignore[arg-type]
            )
        )


def test_generate_batch_retries_when_batch_overlaps_global_intents() -> None:
    fake_llm = _FakeLLM(
        responses=[
            _build_batch_with_prompt(25, "duplicate intent", start_index=1),
            _build_batch_with_prompt(25, "unique intent", start_index=1),
        ]
    )
    executor = GroundTruthCreatorExecutor(llm=fake_llm, max_generation_attempts=2)

    batch = executor._generate_batch(
        system_prompt="system",
        source_text="[E0001] [SPEAKER_00] Source line",
        dimension="factualness",
        expected_ids=[f"Q{index:03d}" for index in range(1, 26)],
        disallowed_intents={
            executor._intent_key(f"duplicate intent {index}?") for index in range(1, 26)
        },
    )

    assert len(batch) == 25
    assert len(fake_llm.calls) == 2
    assert all("unique intent" in question.question_text for question in batch)


def test_generate_batch_raises_when_all_attempts_overlap_global_intents() -> None:
    fake_llm = _FakeLLM(
        responses=[
            _build_batch_with_prompt(25, "duplicate intent", start_index=1),
            _build_batch_with_prompt(25, "duplicate intent", start_index=1),
        ]
    )
    executor = GroundTruthCreatorExecutor(llm=fake_llm, max_generation_attempts=2)

    with pytest.raises(RuntimeError, match="failed to generate valid batch"):
        executor._generate_batch(
            system_prompt="system",
            source_text="[E0001] [SPEAKER_00] Source line",
            dimension="factualness",
            expected_ids=[f"Q{index:03d}" for index in range(1, 26)],
            disallowed_intents={
                executor._intent_key(f"duplicate intent {index}?")
                for index in range(1, 26)
            },
        )


def test_intent_key_supports_non_latin_text() -> None:
    key_a = GroundTruthCreatorExecutor._intent_key("这是一个测试问题？")
    key_b = GroundTruthCreatorExecutor._intent_key("这是另一个测试问题？")
    assert key_a
    assert key_b
    assert key_a != key_b


def test_build_batch_prompt_prioritizes_latest_avoid_examples() -> None:
    avoid_examples = [f"avoid question {index}" for index in range(1, 16)]
    prompt = GroundTruthCreatorExecutor._build_batch_user_prompt(
        source_text="[E0001] [SPEAKER_00] Source line",
        dimension="factualness",
        expected_ids=[f"Q{index:03d}" for index in range(1, 26)],
        avoid_question_texts=avoid_examples,
    )
    assert "- avoid question 1\n" not in prompt
    assert "- avoid question 15\n" in prompt


def test_handle_task_retries_cross_batch_global_intent_overlap() -> None:
    fake_llm = _FakeLLM(
        responses=[
            _build_batch_with_prompt(25, "factual alpha", start_index=1),
            _build_batch_with_prompt(25, "factual beta", start_index=26),
            _build_batch_with_prompt(25, "factual alpha", start_index=1),
            _build_batch_with_prompt(25, "natural gamma", start_index=1),
            _build_batch_with_prompt(25, "natural delta", start_index=26),
        ]
    )
    executor = GroundTruthCreatorExecutor(llm=fake_llm, max_generation_attempts=2)
    queue = _FakeEventQueue()

    asyncio.run(
        executor.handle_task(
            {"source_text": "[E0001] [SPEAKER_00] Source line"},
            context=None,  # type: ignore[arg-type]
            event_queue=queue,  # type: ignore[arg-type]
        )
    )

    assert len(fake_llm.calls) == 5
    payload = json.loads(str(queue.events[0]))
    assert len(payload["questions"]) == 100
