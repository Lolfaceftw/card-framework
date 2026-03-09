"""Unit tests for QAEvaluatorExecutor."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from typing import Any, Literal

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

from card_framework.benchmark.qa_contracts import GroundTruthQuestion, GroundTruthSet
from card_framework.benchmark.qa_settings import EvaluatorRuntimeConfig, QuoteRelevanceConfig
from card_framework.shared.events import EventBus
from card_framework.agents.dtos import (
    CorrectorFewShotExample,
    CorrectorTaskRequest,
    CorrectorTaskResponse,
)
from card_framework.agents.qa_evaluator import QAEvaluatorExecutor
from card_framework.shared.prompt_manager import PromptManager


class _FakeResponse:
    """Simple chat response DTO for BaseA2AExecutor test paths."""

    def __init__(
        self, *, content: str = "", tool_calls: list[dict[str, Any]] | None = None
    ):
        self._content = content
        self._tool_calls = tool_calls

    def model_dump(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": self._content,
            "tool_calls": self._tool_calls,
        }


class _FakeLLM:
    """Deterministic chat LLM stub."""

    def __init__(self, responses: list[_FakeResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.calls = 0
        self.last_messages: list[dict[str, Any]] = []

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: dict[str, Any] | str | None = None,
        max_tokens: int | None = None,
    ) -> _FakeResponse:
        del tools, tool_choice, max_tokens
        self.calls += 1
        self.last_messages = list(messages)
        if self._index >= len(self._responses):
            return _FakeResponse(content="done", tool_calls=None)
        response = self._responses[self._index]
        self._index += 1
        return response

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        del system_prompt, user_prompt, max_tokens
        raise NotImplementedError


class _FakeEventQueue:
    """Collects enqueue_event payloads from send_response."""

    def __init__(self) -> None:
        self.events: list[object] = []

    async def enqueue_event(self, event: object) -> None:
        self.events.append(event)


class _FakeCorrector:
    """Deterministic corrector stub that records retry failure payloads."""

    def __init__(self) -> None:
        self.requests: list[CorrectorTaskRequest] = []

    def build_retry_guidance(
        self, request: CorrectorTaskRequest
    ) -> CorrectorTaskResponse:
        self.requests.append(request)
        return CorrectorTaskResponse(
            correction_instruction=(
                "Use the exact current question_id and include a verbatim relevant summary quote."
            ),
            few_shot_examples=[
                CorrectorFewShotExample(
                    bad_example='submit_answer({"question_id":"Q099","predicted_answer":"yes","summary_evidence_quote":"..."})',
                    corrected_example='submit_answer({"question_id":"Q001","predicted_answer":"yes","summary_evidence_quote":"exact summary quote"})',
                    rationale="Current turn id must match and quote must be grounded.",
                )
            ],
        )


def _build_questions() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    for index in range(1, 51):
        questions.append(
            {
                "question_id": f"Q{index:03d}",
                "dimension": "factualness",
                "question_text": f"Fact question {index}?",
                "expected_answer": "yes",
                "source_evidence_ids": [f"E{index:04d}"],
                "speaker_ids": ["SPEAKER_00"],
            }
        )
    for offset in range(1, 51):
        index = 50 + offset
        questions.append(
            {
                "question_id": f"Q{index:03d}",
                "dimension": "naturalness",
                "question_text": f"Nat question {offset}?",
                "expected_answer": "no",
                "source_evidence_ids": [f"E{index:04d}"],
                "speaker_ids": ["SPEAKER_01"],
            }
        )
    return questions


def _runtime_config(
    *,
    max_tool_turns: int = 10,
    max_attempts_per_question: int = 3,
    chat_max_tokens: int = 160,
    no_tool_call_patience: int = 3,
    max_tool_calls_per_turn: int = 1,
    per_question_concurrency: int = 1,
    quote_mode: Literal[
        "lexical_overlap", "semantic_similarity", "hybrid", "off"
    ] = "lexical_overlap",
    semantic_threshold: float | None = None,
    auto_repair: bool = True,
    repair_min_score: float = 0.25,
    min_candidate_chars: int = 6,
) -> EvaluatorRuntimeConfig:
    """Build evaluator runtime config fixture for deterministic unit tests."""
    return EvaluatorRuntimeConfig(
        max_tool_turns=max_tool_turns,
        max_attempts_per_question=max_attempts_per_question,
        chat_max_tokens=chat_max_tokens,
        no_tool_call_patience=no_tool_call_patience,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        per_question_concurrency=per_question_concurrency,
        quote_relevance=QuoteRelevanceConfig(
            mode=quote_mode,
            min_shared_tokens=1,
            min_distinctive_shared_tokens=1,
            min_token_length=3,
            semantic_threshold=semantic_threshold,
            auto_repair=auto_repair,
            repair_min_score=repair_min_score,
            min_candidate_chars=min_candidate_chars,
        ),
    )


def setup_function(function: object) -> None:
    """Reset prompt environment for test isolation."""
    del function
    PromptManager._env = None
    PromptManager._instance = None


def test_runtime_context_enforces_one_tool_call_per_turn() -> None:
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Candidate summary</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=3),
    )
    assert context_data["max_tool_calls_per_turn"] == 1


def test_constructor_accepts_legacy_positional_event_bus_argument() -> None:
    """Allow legacy positional EventBus argument without mis-parsing runtime config."""
    legacy_bus = EventBus()
    executor = QAEvaluatorExecutor(
        _FakeLLM([]),
        12,
        4,
        legacy_bus,  # type: ignore[arg-type]
    )

    assert executor.event_bus is legacy_bus
    assert executor.runtime_config.max_tool_turns == 12
    assert executor.runtime_config.max_attempts_per_question == 4


def test_process_tool_calls_retries_invalid_question_id_without_advancing() -> None:
    fake_corrector = _FakeCorrector()
    executor = QAEvaluatorExecutor(
        llm=_FakeLLM([]),
        max_tool_turns=1,
        corrector_agent=fake_corrector,
    )
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Candidate summary</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=3),
    )
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q999",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": "Candidate summary",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )

    assert should_break is False
    assert final_result is None
    answers = context_data["answers"]
    assert len(answers) == 0
    assert context_data["current_question_index"] == 0
    assert context_data["question_attempts"]["Q001"] == 1
    assert len(fake_corrector.requests) == 1
    assert fake_corrector.requests[0].target_agent == "Evaluator"
    assert "invalid_question_id" in fake_corrector.requests[0].failure_type
    retry_prompts = [item for item in messages if item.get("role") == "user"]
    assert retry_prompts
    assert "<Corrector guidance>" in str(retry_prompts[-1]["content"])


def test_process_tool_calls_dedupes_corrector_for_same_failure_signature() -> None:
    """Call Corrector once for repeated retries with same question and status."""
    fake_corrector = _FakeCorrector()
    executor = QAEvaluatorExecutor(
        llm=_FakeLLM([]),
        max_tool_turns=1,
        corrector_agent=fake_corrector,
    )
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Candidate summary</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=3),
    )
    messages: list[dict[str, Any]] = []

    for _ in range(2):
        should_break, final_result = asyncio.run(
            executor.process_tool_calls(
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "submit_answer",
                        "arguments": {
                            "question_id": "Q999",
                            "predicted_answer": "yes",
                            "summary_evidence_quote": "Candidate summary",
                        },
                    }
                ],
                messages=messages,
                context_data=context_data,
            )
        )
        assert should_break is False
        assert final_result is None

    assert len(fake_corrector.requests) == 1


def test_corrector_receives_original_submitted_quote_after_repair_failure() -> None:
    """Preserve original quote in Corrector context even when evaluator clears it."""
    fake_corrector = _FakeCorrector()
    executor = QAEvaluatorExecutor(
        llm=_FakeLLM([]),
        max_tool_turns=1,
        corrector_agent=fake_corrector,
    )
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Only unrelated sentence.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=2),
    )
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": "Hallucinated quote text",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )

    assert should_break is False
    assert final_result is None
    assert len(fake_corrector.requests) == 1
    request = fake_corrector.requests[0]
    assert "Hallucinated quote text" in request.failure_context
    assert "Hallucinated quote text" in request.latest_output


def test_process_tool_calls_marks_incorrect_after_max_retry_exceeded() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Candidate summary</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=2),
    )
    messages: list[dict[str, Any]] = []

    for _ in range(2):
        should_break, final_result = asyncio.run(
            executor.process_tool_calls(
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "submit_answer",
                        "arguments": {
                            "question_id": "Q999",
                            "predicted_answer": "yes",
                            "summary_evidence_quote": "Candidate summary",
                        },
                    }
                ],
                messages=messages,
                context_data=context_data,
            )
        )
        assert should_break is False
        assert final_result is None

    answers = context_data["answers"]
    assert len(answers) == 1
    assert answers[0].question_id == "Q001"
    assert answers[0].tool_status == "invalid_question_id_max_retry_exceeded"
    assert answers[0].is_correct is False
    assert context_data["current_question_index"] == 1


def test_process_tool_calls_requires_summary_quote_grounding() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Only this sentence is present.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=1),
    )
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": "missing quote",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert (
        context_data["answers"][0].tool_status
        == "missing_summary_evidence_max_retry_exceeded"
    )
    assert context_data["answers"][0].summary_grounding_pass is False


def test_process_tool_calls_rejects_irrelevant_summary_quote() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Unrelated quote about psychology.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=1),
    )
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": "Unrelated quote about psychology",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert (
        context_data["answers"][0].tool_status
        == "missing_summary_evidence_max_retry_exceeded"
    )


def test_process_tool_calls_rejects_generic_overlap_quote() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Welcome to the podcast intro.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=1),
    )
    first_question = question_set.questions[0]
    first_question.question_text = "Does the podcast mention market efficiency?"
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "no",
                        "summary_evidence_quote": "Welcome to the podcast intro",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert (
        context_data["answers"][0].tool_status
        == "missing_summary_evidence_max_retry_exceeded"
    )


def test_handle_task_fills_missing_answers_when_model_noops() -> None:
    llm = _FakeLLM([_FakeResponse(content="No tool call", tool_calls=None)])
    executor = QAEvaluatorExecutor(llm=llm, max_tool_turns=1)
    queue = _FakeEventQueue()

    asyncio.run(
        executor.handle_task(
            {
                "summary_xml": "<SPEAKER_00>Candidate</SPEAKER_00>",
                "source_text": "[E0001] [SPEAKER_00] Source",
                "questions": _build_questions(),
            },
            context=None,  # type: ignore[arg-type]
            event_queue=queue,  # type: ignore[arg-type]
        )
    )

    assert len(queue.events) == 1
    payload = json.loads(extract_agent_text_message(queue.events[0]))
    assert payload["status"] == "completed"
    assert payload["score"]["score_out_of_100"] == 0
    assert len(payload["answers"]) == 100
    assert payload["score"]["summary_grounding_pass_count"] == 0


def test_handle_task_stops_after_no_tool_call_patience() -> None:
    llm = _FakeLLM([])
    executor = QAEvaluatorExecutor(llm=llm, max_tool_turns=30)
    queue = _FakeEventQueue()

    asyncio.run(
        executor.handle_task(
            {
                "summary_xml": "<SPEAKER_00>Candidate</SPEAKER_00>",
                "source_text": "[E0001] [SPEAKER_00] Source",
                "questions": _build_questions(),
            },
            context=None,  # type: ignore[arg-type]
            event_queue=queue,  # type: ignore[arg-type]
        )
    )

    assert llm.calls == 3
    payload = json.loads(extract_agent_text_message(queue.events[0]))
    assert payload["status"] == "completed"
    assert len(payload["answers"]) == 100


def test_runtime_context_uses_configured_chat_and_patience() -> None:
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    runtime_config = _runtime_config(
        chat_max_tokens=123,
        no_tool_call_patience=5,
        max_tool_calls_per_turn=2,
    )
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Candidate summary</SPEAKER_00>",
        evaluator_runtime_config=runtime_config,
    )
    assert context_data["chat_max_tokens"] == 123
    assert context_data["no_tool_call_patience"] == 5
    assert context_data["max_tool_calls_per_turn"] == 2


def test_process_tool_calls_accepts_when_quote_matches_summary() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Fact sentence in candidate summary.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=2),
    )
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": "Fact sentence in candidate summary",
                        "confidence": 0.8,
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert context_data["answers"][0].tool_status == "accepted"
    assert context_data["answers"][0].summary_grounding_pass is True
    assert context_data["answers"][0].confidence == 0.8


def test_process_tool_calls_autocorrects_question_id_for_single_question() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    single_question = GroundTruthQuestion.model_validate(
        {
            "question_id": "Q001",
            "dimension": "factualness",
            "question_text": "Fact question 1?",
            "expected_answer": "yes",
            "source_evidence_ids": ["E0001"],
            "speaker_ids": ["SPEAKER_00"],
        }
    )
    question_set = GroundTruthSet.model_construct(questions=[single_question])
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Fact sentence in candidate summary.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=1),
    )
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "WRONG_ID",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": "Fact sentence in candidate summary",
                        "confidence": 0.8,
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is True
    assert final_result is not None
    assert len(context_data["answers"]) == 1
    assert context_data["answers"][0].tool_status == "accepted"


def test_process_tool_calls_auto_repairs_irrelevant_quote_when_enabled() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml=(
            "<SPEAKER_00>Welcome to the Rational Reminder podcast intro.</SPEAKER_00>\n"
            "<SPEAKER_01>The summary discusses financial planning decisions.</SPEAKER_01>"
        ),
        evaluator_runtime_config=_runtime_config(
            max_attempts_per_question=1,
            quote_mode="hybrid",
            auto_repair=True,
            repair_min_score=0.2,
        ),
    )
    question_set.questions[
        0
    ].question_text = "Does the summary mention financial planning?"
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": "Welcome to the Rational Reminder podcast intro.",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )

    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert context_data["answers"][0].tool_status == "accepted"
    assert context_data["answers"][0].summary_grounding_pass is True
    assert (
        "financial planning"
        in context_data["answers"][0].summary_evidence_quote.lower()
    )


def test_process_tool_calls_rejects_irrelevant_quote_when_auto_repair_disabled() -> (
    None
):
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml=(
            "<SPEAKER_00>Welcome to the Rational Reminder podcast intro.</SPEAKER_00>\n"
            "<SPEAKER_01>The summary discusses financial planning decisions.</SPEAKER_01>"
        ),
        evaluator_runtime_config=_runtime_config(
            max_attempts_per_question=1,
            quote_mode="lexical_overlap",
            auto_repair=False,
        ),
    )
    question_set.questions[
        0
    ].question_text = "Does the summary mention financial planning?"
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": "Welcome to the Rational Reminder podcast intro.",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )

    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert (
        context_data["answers"][0].tool_status
        == "irrelevant_summary_evidence_max_retry_exceeded"
    )


def test_process_tool_calls_auto_repairs_short_quote_candidate() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml=(
            "<SPEAKER_00>Intro sentence.</SPEAKER_00>\n"
            "<SPEAKER_01>Tax cuts.</SPEAKER_01>"
        ),
        evaluator_runtime_config=_runtime_config(
            max_attempts_per_question=1,
            quote_mode="hybrid",
            auto_repair=True,
            min_candidate_chars=4,
            repair_min_score=0.2,
        ),
    )
    question_set.questions[0].question_text = "Does the summary mention tax cuts?"
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": "Intro sentence.",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )

    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert context_data["answers"][0].tool_status == "accepted"
    assert "tax cuts" in context_data["answers"][0].summary_evidence_quote.lower()


def test_process_tool_calls_accepts_quote_when_relevance_mode_off() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Unrelated quote about psychology.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(
            max_attempts_per_question=1,
            quote_mode="off",
        ),
    )
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": "Unrelated quote about psychology",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert context_data["answers"][0].tool_status == "accepted"
    assert context_data["answers"][0].summary_grounding_pass is True


def test_process_tool_calls_rejects_quote_with_semantic_mode_and_high_threshold() -> (
    None
):
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>The summary says market momentum appeared briefly.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(
            max_attempts_per_question=1,
            quote_mode="semantic_similarity",
            semantic_threshold=0.95,
        ),
    )
    question_set.questions[
        0
    ].question_text = "Does the summary discuss dividend taxation?"
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": (
                            "The summary says market momentum appeared briefly"
                        ),
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert (
        context_data["answers"][0].tool_status
        == "missing_summary_evidence_max_retry_exceeded"
    )


def test_process_tool_calls_accepts_specific_overlap_quote() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>The Magellan fund outperformed peers over 10 years.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=1),
    )
    question_set.questions[
        0
    ].question_text = "Does the summary mention the Magellan fund performance?"
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": (
                            "The Magellan fund outperformed peers over 10 years."
                        ),
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert context_data["answers"][0].tool_status == "accepted"


def test_process_tool_calls_accepts_speaker_token_overlap_quote() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="SPEAKER_02 The witness statement is confirmed.",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=1),
    )
    question_set.questions[
        0
    ].question_text = "Does SPEAKER_02 provide a witness statement?"
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "yes",
                        "summary_evidence_quote": (
                            "SPEAKER_02 The witness statement is confirmed"
                        ),
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert context_data["answers"][0].tool_status == "accepted"


def test_process_tool_calls_accepts_no_answer_without_quote() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Candidate summary exists.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=1),
    )
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "no",
                        "summary_evidence_quote": "",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert context_data["answers"][0].tool_status == "accepted"
    assert context_data["answers"][0].predicted_answer == "no"
    assert context_data["answers"][0].summary_grounding_pass is False


def test_process_tool_calls_normalizes_binary_answer_variants() -> None:
    executor = QAEvaluatorExecutor(llm=_FakeLLM([]), max_tool_turns=10)
    question_set = GroundTruthSet.model_validate({"questions": _build_questions()})
    context_data = QAEvaluatorExecutor._build_runtime_context(
        question_set=question_set,
        summary_xml="<SPEAKER_00>Candidate summary exists.</SPEAKER_00>",
        evaluator_runtime_config=_runtime_config(max_attempts_per_question=1),
    )
    question_set.questions[0].question_text = "Candidate summary exists."
    messages: list[dict[str, Any]] = []

    should_break, final_result = asyncio.run(
        executor.process_tool_calls(
            tool_calls=[
                {
                    "id": "call_1",
                    "name": "submit_answer",
                    "arguments": {
                        "question_id": "Q001",
                        "predicted_answer": "TRUE",
                        "summary_evidence_quote": "Candidate summary exists.",
                    },
                }
            ],
            messages=messages,
            context_data=context_data,
        )
    )
    assert should_break is False
    assert final_result is None
    assert len(context_data["answers"]) == 1
    assert context_data["answers"][0].tool_status == "accepted"
    assert context_data["answers"][0].predicted_answer == "yes"


def test_handle_task_does_not_leak_expected_answer_to_prompt() -> None:
    llm = _FakeLLM([_FakeResponse(content="No tool call", tool_calls=None)])
    executor = QAEvaluatorExecutor(llm=llm, max_tool_turns=1)
    queue = _FakeEventQueue()

    asyncio.run(
        executor.handle_task(
            {
                "summary_xml": "<SPEAKER_00>Candidate</SPEAKER_00>",
                "source_text": "[E0001] [SPEAKER_00] Source",
                "questions": _build_questions(),
            },
            context=None,  # type: ignore[arg-type]
            event_queue=queue,  # type: ignore[arg-type]
        )
    )

    serialized_messages = " ".join(str(message) for message in llm.last_messages)
    assert "expected_answer" not in serialized_messages

