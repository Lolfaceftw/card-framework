"""Per-question evaluator runner utilities for QA benchmark workflow."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Protocol

from agents.dtos import QAEvaluatorTaskRequest
from benchmark.qa_contracts import (
    EvaluatorAnswerRecord,
    GroundTruthQuestion,
    GroundTruthSet,
    build_score,
)
from events import event_bus
from logger_utils import logger


class AgentClientLike(Protocol):
    """Minimal async client interface needed for evaluator task dispatch."""

    async def send_task(
        self,
        port: int,
        task: QAEvaluatorTaskRequest,
        *,
        timeout: float,
        metadata: dict[str, str],
    ) -> str:
        """Send one A2A task and return raw string payload."""


def build_evaluator_request_error_answer(
    *,
    question: GroundTruthQuestion,
    reason: str,
    tool_status: str = "evaluator_request_error",
) -> EvaluatorAnswerRecord:
    """Build deterministic fallback answer for per-question evaluator failures.

    Args:
        question: Ground-truth question that failed evaluator execution.
        reason: Failure explanation for trace/debug artifacts.
        tool_status: Structured status label for downstream analytics.

    Returns:
        Failed answer record bound to the expected question identifier.
    """
    return EvaluatorAnswerRecord(
        question_id=question.question_id,
        dimension=question.dimension,
        predicted_answer="missing",
        expected_answer=question.expected_answer,
        is_correct=False,
        tool_status=tool_status,
        reason=reason,
        summary_grounding_pass=False,
        summary_evidence_quote="",
        confidence=None,
    )


def extract_single_question_answer(
    *,
    evaluator_payload: dict[str, Any],
    question: GroundTruthQuestion,
) -> EvaluatorAnswerRecord:
    """Extract one validated answer from a single-question evaluator payload.

    Args:
        evaluator_payload: Parsed evaluator JSON response.
        question: Expected ground-truth question for this call.

    Returns:
        Validated answer record matching ``question.question_id``.

    Raises:
        ValueError: Payload is malformed or does not contain exactly one match.
    """
    answers_raw = evaluator_payload.get("answers")
    if not isinstance(answers_raw, list):
        raise ValueError(
            f"Evaluator payload for question_id={question.question_id} is missing an 'answers' list."
        )
    if not answers_raw:
        raise ValueError(
            f"Evaluator payload for question_id={question.question_id} returned no answers."
        )

    candidate_answers = [
        EvaluatorAnswerRecord.model_validate(item) for item in answers_raw
    ]
    matching_answers = [
        answer
        for answer in candidate_answers
        if answer.question_id == question.question_id
    ]
    if len(matching_answers) != 1:
        raise ValueError(
            "Evaluator payload for question_id="
            f"{question.question_id} returned {len(matching_answers)} matching answers."
        )
    return matching_answers[0]


async def evaluate_questions_with_fresh_contexts(
    *,
    client: AgentClientLike,
    evaluator_port: int,
    summary_xml: str,
    source_text: str,
    question_set: GroundTruthSet,
    evaluator_request_timeout_seconds: float,
    per_question_concurrency: int,
    evaluator_request_retries: int = 1,
) -> dict[str, Any]:
    """Evaluate each question in a separate evaluator request context.

    Args:
        client: Async A2A client implementation.
        evaluator_port: Evaluator service port.
        summary_xml: Candidate summary XML content.
        source_text: Source transcript text.
        question_set: Validated benchmark questions.
        evaluator_request_timeout_seconds: Timeout budget per question request.
        per_question_concurrency: Maximum concurrent in-flight evaluator requests.
        evaluator_request_retries: Maximum attempts per question request.

    Returns:
        Evaluator payload with ``status``, ``score``, and ``answers`` keys.
    """
    concurrency_limit = max(1, per_question_concurrency)
    retry_limit = max(1, evaluator_request_retries)
    semaphore = asyncio.Semaphore(concurrency_limit)
    total_questions = len(question_set.questions)
    lifecycle_lock = asyncio.Lock()
    in_flight_count = 0
    spawn_sequence = 0
    close_sequence = 0

    async def evaluate_one_question(
        question: GroundTruthQuestion,
        question_index: int,
    ) -> EvaluatorAnswerRecord:
        """Evaluate one question in an isolated evaluator request context."""
        nonlocal in_flight_count, spawn_sequence, close_sequence
        async with semaphore:
            started_at = time.perf_counter()
            async with lifecycle_lock:
                in_flight_count += 1
                spawn_sequence += 1
                spawn_seq = spawn_sequence
                in_flight_now = in_flight_count
            event_bus.publish(
                "status_message",
                (
                    "AGENT SPAWN "
                    f"question_id={question.question_id} "
                    f"index={question_index}/{total_questions} "
                    f"spawn_seq={spawn_seq} "
                    f"in_flight={in_flight_now}/{concurrency_limit}"
                ),
            )
            close_status = "unknown"
            attempts_used = 0
            try:
                logger.info(
                    "[QABenchmark] Evaluator request question_id=%s index=%s/%s timeout=%.1fs retries=%s",
                    question.question_id,
                    question_index,
                    total_questions,
                    evaluator_request_timeout_seconds,
                    retry_limit,
                )
                task = QAEvaluatorTaskRequest(
                    summary_xml=summary_xml,
                    source_text=source_text,
                    questions=[question.model_dump()],
                )
                for attempt in range(1, retry_limit + 1):
                    attempts_used = attempt
                    try:
                        evaluator_response_raw = await client.send_task(
                            evaluator_port,
                            task,
                            timeout=evaluator_request_timeout_seconds,
                            metadata={
                                "component": "qa_benchmark",
                                "stage": "evaluator_single_question",
                                "question_id": question.question_id,
                            },
                        )
                    except Exception as exc:  # pragma: no cover - network/runtime path
                        if attempt < retry_limit:
                            logger.warning(
                                "[QABenchmark] Evaluator request failed for question_id=%s attempt=%s/%s: %s. Retrying.",
                                question.question_id,
                                attempt,
                                retry_limit,
                                exc,
                            )
                            await asyncio.sleep(float(min(2, attempt)))
                            continue
                        logger.exception(
                            "[QABenchmark] Evaluator request failed for question_id=%s after %s attempts",
                            question.question_id,
                            retry_limit,
                        )
                        close_status = "evaluator_request_error"
                        return build_evaluator_request_error_answer(
                            question=question,
                            reason=(
                                "Evaluator request failed after "
                                f"{retry_limit} attempts: {exc}"
                            ),
                            tool_status="evaluator_request_error",
                        )

                    try:
                        evaluator_payload = json.loads(evaluator_response_raw)
                    except json.JSONDecodeError as exc:
                        if attempt < retry_limit:
                            logger.warning(
                                "[QABenchmark] Evaluator returned invalid JSON for question_id=%s attempt=%s/%s: %s. Retrying.",
                                question.question_id,
                                attempt,
                                retry_limit,
                                exc,
                            )
                            await asyncio.sleep(float(min(2, attempt)))
                            continue
                        logger.exception(
                            "[QABenchmark] Evaluator returned invalid JSON for question_id=%s after %s attempts",
                            question.question_id,
                            retry_limit,
                        )
                        close_status = "evaluator_invalid_json"
                        return build_evaluator_request_error_answer(
                            question=question,
                            reason=(
                                "Evaluator returned invalid JSON after "
                                f"{retry_limit} attempts: {exc}"
                            ),
                            tool_status="evaluator_invalid_json",
                        )

                    try:
                        answer = extract_single_question_answer(
                            evaluator_payload=evaluator_payload,
                            question=question,
                        )
                        close_status = answer.tool_status
                        return answer
                    except Exception as exc:
                        logger.exception(
                            "[QABenchmark] Evaluator payload validation failed for question_id=%s",
                            question.question_id,
                        )
                        close_status = "evaluator_payload_validation_failed"
                        return build_evaluator_request_error_answer(
                            question=question,
                            reason=f"Evaluator payload validation failed: {exc}",
                            tool_status="evaluator_payload_validation_failed",
                        )

                close_status = "evaluator_request_error"
                return build_evaluator_request_error_answer(
                    question=question,
                    reason="Evaluator request failed before receiving a valid response.",
                    tool_status="evaluator_request_error",
                )
            finally:
                elapsed_ms = int((time.perf_counter() - started_at) * 1000)
                async with lifecycle_lock:
                    in_flight_count = max(0, in_flight_count - 1)
                    close_sequence += 1
                    close_seq = close_sequence
                    remaining_in_flight = in_flight_count
                event_bus.publish(
                    "status_message",
                    (
                        "AGENT CLOSE "
                        f"question_id={question.question_id} "
                        f"index={question_index}/{total_questions} "
                        f"close_seq={close_seq} "
                        f"status={close_status} "
                        f"attempts={attempts_used}/{retry_limit} "
                        f"latency_ms={elapsed_ms} "
                        f"in_flight={remaining_in_flight}/{concurrency_limit}"
                    ),
                )

    tasks = [
        evaluate_one_question(question, index)
        for index, question in enumerate(question_set.questions, start=1)
    ]
    answers = await asyncio.gather(*tasks)
    score = build_score(answers)
    return {
        "status": "completed",
        "score": score.model_dump(),
        "answers": [answer.model_dump() for answer in answers],
    }
