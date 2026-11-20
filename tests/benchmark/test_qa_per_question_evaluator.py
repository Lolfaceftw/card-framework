"""Tests for per-question evaluator execution in QA benchmark workflow."""

from __future__ import annotations

import json
import asyncio
from typing import Any

import pytest

from card_framework.benchmark.qa_evaluator_runner import (
    evaluate_questions_with_fresh_contexts,
    extract_single_question_answer,
)
from card_framework.benchmark.qa_contracts import GroundTruthQuestion, GroundTruthSet
from card_framework.shared.events import event_bus


class _FakeAgentClient:
    """Minimal async client stub for per-question evaluator tests."""

    def __init__(
        self,
        responses: dict[str, str | Exception | list[str | Exception]],
    ) -> None:
        """Store per-question responses keyed by question_id."""
        self._responses = responses
        self.called_question_ids: list[str] = []

    async def send_task(
        self,
        _port: int,
        task: Any,
        *,
        timeout: float,
        metadata: dict[str, str],
    ) -> str:
        """Return deterministic payload or raise mapped exception."""
        del timeout
        question_id = metadata.get("question_id", "")
        if not question_id:
            raise RuntimeError("missing question_id metadata")
        self.called_question_ids.append(question_id)
        response = self._responses[question_id]
        if isinstance(response, list):
            if not response:
                raise RuntimeError(f"no queued response for {question_id}")
            response = response.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def close(self) -> None:
        """Match AgentClient interface used by benchmark workflow."""


def _question(
    *,
    question_id: str,
    dimension: str = "factualness",
) -> GroundTruthQuestion:
    """Create a minimal valid ground-truth question for tests."""
    return GroundTruthQuestion.model_validate(
        {
            "question_id": question_id,
            "dimension": dimension,
            "question_text": f"Question text for {question_id}",
            "expected_answer": "yes",
            "source_evidence_ids": ["E0001"],
            "speaker_ids": ["SPEAKER_00"],
        }
    )


def test_extract_single_question_answer_rejects_missing_answers_list() -> None:
    """Raise runtime error when evaluator payload lacks answers list."""
    question = _question(question_id="Q001")
    with pytest.raises(ValueError):
        extract_single_question_answer(
            evaluator_payload={"status": "completed"},
            question=question,
        )


def test_extract_single_question_answer_rejects_mismatched_id() -> None:
    """Raise runtime error when evaluator payload answer does not match question ID."""
    question = _question(question_id="Q001")
    payload = {
        "status": "completed",
        "answers": [
            {
                "question_id": "Q099",
                "dimension": "factualness",
                "predicted_answer": "yes",
                "expected_answer": "yes",
                "is_correct": True,
                "tool_status": "accepted",
                "reason": "ok",
                "summary_grounding_pass": True,
                "summary_evidence_quote": "quote",
                "confidence": 0.9,
            }
        ],
    }
    with pytest.raises(ValueError):
        extract_single_question_answer(evaluator_payload=payload, question=question)


def test_evaluate_questions_with_fresh_contexts_handles_errors() -> None:
    """Return deterministic error records for failed per-question evaluator calls."""
    q1 = _question(question_id="Q001", dimension="factualness")
    q2 = _question(question_id="Q002", dimension="naturalness")
    question_set = GroundTruthSet.model_construct(questions=[q1, q2])

    success_payload = json.dumps(
        {
            "status": "completed",
            "answers": [
                {
                    "question_id": "Q001",
                    "dimension": "factualness",
                    "predicted_answer": "yes",
                    "expected_answer": "yes",
                    "is_correct": True,
                    "tool_status": "accepted",
                    "reason": "ok",
                    "summary_grounding_pass": True,
                    "summary_evidence_quote": "quote for q1",
                    "confidence": 0.8,
                }
            ],
        }
    )
    fake_client = _FakeAgentClient(
        responses={
            "Q001": success_payload,
            "Q002": RuntimeError("upstream timeout"),
        }
    )

    payload = asyncio.run(
        evaluate_questions_with_fresh_contexts(
            client=fake_client,
            evaluator_port=32123,
            summary_xml="<summary><p>text</p></summary>",
            source_text="[E0001] source text",
            question_set=question_set,
            evaluator_request_timeout_seconds=30.0,
            per_question_concurrency=2,
        )
    )

    answers = payload["answers"]
    assert [item["question_id"] for item in answers] == ["Q001", "Q002"]
    assert answers[0]["tool_status"] == "accepted"
    assert answers[1]["tool_status"] == "evaluator_request_error"
    assert answers[1]["summary_grounding_pass"] is False


def test_evaluate_questions_with_fresh_contexts_retries_transient_failures() -> None:
    """Retry transient evaluator request failures and recover on next attempt."""
    q1 = _question(question_id="Q001", dimension="factualness")
    question_set = GroundTruthSet.model_construct(questions=[q1])

    success_payload = json.dumps(
        {
            "status": "completed",
            "answers": [
                {
                    "question_id": "Q001",
                    "dimension": "factualness",
                    "predicted_answer": "yes",
                    "expected_answer": "yes",
                    "is_correct": True,
                    "tool_status": "accepted",
                    "reason": "ok",
                    "summary_grounding_pass": True,
                    "summary_evidence_quote": "quote for q1",
                    "confidence": 0.8,
                }
            ],
        }
    )
    fake_client = _FakeAgentClient(
        responses={"Q001": [RuntimeError("temporary failure"), success_payload]}
    )

    payload = asyncio.run(
        evaluate_questions_with_fresh_contexts(
            client=fake_client,
            evaluator_port=32123,
            summary_xml="<summary><p>text</p></summary>",
            source_text="[E0001] source text",
            question_set=question_set,
            evaluator_request_timeout_seconds=30.0,
            per_question_concurrency=1,
            evaluator_request_retries=2,
        )
    )

    answers = payload["answers"]
    assert [item["question_id"] for item in answers] == ["Q001"]
    assert answers[0]["tool_status"] == "accepted"
    assert fake_client.called_question_ids == ["Q001", "Q001"]


def test_evaluate_questions_with_fresh_contexts_emits_spawn_and_close_events() -> None:
    """Publish lifecycle events for each per-question evaluator request."""
    q1 = _question(question_id="Q001", dimension="factualness")
    q2 = _question(question_id="Q002", dimension="naturalness")
    question_set = GroundTruthSet.model_construct(questions=[q1, q2])

    success_payload_q1 = json.dumps(
        {
            "status": "completed",
            "answers": [
                {
                    "question_id": "Q001",
                    "dimension": "factualness",
                    "predicted_answer": "yes",
                    "expected_answer": "yes",
                    "is_correct": True,
                    "tool_status": "accepted",
                    "reason": "ok",
                    "summary_grounding_pass": True,
                    "summary_evidence_quote": "quote for q1",
                    "confidence": 0.8,
                }
            ],
        }
    )
    success_payload_q2 = json.dumps(
        {
            "status": "completed",
            "answers": [
                {
                    "question_id": "Q002",
                    "dimension": "naturalness",
                    "predicted_answer": "yes",
                    "expected_answer": "yes",
                    "is_correct": True,
                    "tool_status": "accepted",
                    "reason": "ok",
                    "summary_grounding_pass": True,
                    "summary_evidence_quote": "quote for q2",
                    "confidence": 0.8,
                }
            ],
        }
    )
    fake_client = _FakeAgentClient(
        responses={
            "Q001": success_payload_q1,
            "Q002": success_payload_q2,
        }
    )
    captured_status_messages: list[str] = []

    def _capture_status(message: str, **kwargs: Any) -> None:
        del kwargs
        captured_status_messages.append(message)

    event_bus.subscribe("status_message", _capture_status)
    try:
        payload = asyncio.run(
            evaluate_questions_with_fresh_contexts(
                client=fake_client,
                evaluator_port=32123,
                summary_xml="<summary><p>text</p></summary>",
                source_text="[E0001] source text",
                question_set=question_set,
                evaluator_request_timeout_seconds=30.0,
                per_question_concurrency=2,
                evaluator_request_retries=1,
            )
        )
    finally:
        event_bus.unsubscribe("status_message", _capture_status)

    answers = payload["answers"]
    assert [item["question_id"] for item in answers] == ["Q001", "Q002"]
    spawn_messages = [
        m for m in captured_status_messages if m.startswith("AGENT SPAWN")
    ]
    close_messages = [
        m for m in captured_status_messages if m.startswith("AGENT CLOSE")
    ]
    assert len(spawn_messages) == 2
    assert len(close_messages) == 2
    assert any("question_id=Q001" in m for m in spawn_messages)
    assert any("question_id=Q002" in m for m in spawn_messages)
    assert any("question_id=Q001" in m for m in close_messages)
    assert any("question_id=Q002" in m for m in close_messages)

    def _extract_field(message: str, key: str) -> str:
        for part in message.split(" "):
            if part.startswith(f"{key}="):
                return part.split("=", 1)[1]
        raise AssertionError(f"Missing key {key} in message: {message}")

    spawn_sequences = sorted(
        int(_extract_field(message, "spawn_seq")) for message in spawn_messages
    )
    close_sequences = sorted(
        int(_extract_field(message, "close_seq")) for message in close_messages
    )
    assert spawn_sequences == [1, 2]
    assert close_sequences == [1, 2]

    spawn_in_flight_values = [
        int(_extract_field(message, "in_flight").split("/", 1)[0])
        for message in spawn_messages
    ]
    close_in_flight_values = [
        int(_extract_field(message, "in_flight").split("/", 1)[0])
        for message in close_messages
    ]
    assert all(1 <= value <= 2 for value in spawn_in_flight_values)
    assert all(0 <= value <= 1 for value in close_in_flight_values)

