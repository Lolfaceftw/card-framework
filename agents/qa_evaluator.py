"""A2A executor for QA-based summary benchmark scoring."""

from __future__ import annotations

import json
from typing import Any

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from agents.dtos import QAEvaluatorTaskRequest
from benchmark.qa_contracts import (
    EvaluatorAnswerRecord,
    GroundTruthQuestion,
    GroundTruthSet,
    build_score,
)
from events import EventBus, get_event_bus
from llm_provider import LLMProvider
from logger_utils import logger
from prompt_manager import PromptManager


class QAEvaluatorExecutor(BaseA2AExecutor):
    """Evaluate candidate summary answers against ground-truth QA items."""

    _QUOTE_RELEVANCE_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
        "does",
        "did",
        "do",
    }
    _QUOTE_GENERIC_TOKENS = {
        "podcast",
        "speaker",
        "transcript",
        "summary",
        "candidate",
        "question",
        "state",
        "stated",
        "mention",
        "mentions",
        "says",
        "said",
    }

    def __init__(
        self,
        llm: LLMProvider,
        max_tool_turns: int = 320,
        max_attempts_per_question: int = 3,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize evaluator executor and injected collaborators."""
        super().__init__("Evaluator")
        self.llm = llm
        self.max_tool_turns = max(1, int(max_tool_turns))
        self.max_attempts_per_question = max(1, int(max_attempts_per_question))
        self.event_bus = event_bus if event_bus is not None else get_event_bus()

    @staticmethod
    def _build_tools() -> list[dict[str, Any]]:
        """Return tool schema available to evaluator model."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "submit_answer",
                    "description": (
                        "Submit answer for exactly one question_id in current turn. "
                        "Use yes/no only."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question_id": {"type": "string"},
                            "predicted_answer": {
                                "type": "string",
                                "enum": ["yes", "no"],
                            },
                            "reason": {"type": "string"},
                            "summary_evidence_quote": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                        },
                        "required": [
                            "question_id",
                            "predicted_answer",
                            "summary_evidence_quote",
                        ],
                    },
                },
            }
        ]

    @staticmethod
    def _build_runtime_context(
        *,
        question_set: GroundTruthSet,
        summary_xml: str,
        max_attempts_per_question: int,
    ) -> dict[str, Any]:
        """Build loop runtime context for strict evaluator execution."""
        return {
            "question_queue": list(question_set.questions),
            "current_question_index": 0,
            "answers": [],
            "max_tool_calls_per_turn": 1,
            "tool_choice": {"type": "function", "function": {"name": "submit_answer"}},
            "chat_max_tokens": 160,
            "no_tool_call_patience": 3,
            "no_tool_call_count": 0,
            "summary_xml": summary_xml,
            "question_attempts": {},
            "max_attempts_per_question": max(1, int(max_attempts_per_question)),
        }

    @staticmethod
    def _append_missing_answers(context_data: dict[str, Any]) -> None:
        """Append deterministic incorrect records for unanswered questions."""
        question_queue = context_data.get("question_queue", [])
        current_index = int(context_data.get("current_question_index", 0))
        answers = context_data.get("answers", [])
        if not isinstance(question_queue, list) or not isinstance(answers, list):
            return

        while current_index < len(question_queue):
            question = question_queue[current_index]
            if not isinstance(question, GroundTruthQuestion):
                break
            answers.append(
                EvaluatorAnswerRecord(
                    question_id=question.question_id,
                    dimension=question.dimension,
                    predicted_answer="missing",
                    expected_answer=question.expected_answer,
                    is_correct=False,
                    tool_status="missing_answer",
                    reason="Model did not submit an answer for this question.",
                    summary_grounding_pass=False,
                )
            )
            current_index += 1

        context_data["current_question_index"] = current_index

    @staticmethod
    def _build_result_payload(context_data: dict[str, Any]) -> dict[str, Any]:
        """Build JSON-serializable evaluator result from runtime context."""
        answers = context_data.get("answers", [])
        typed_answers = [
            answer for answer in answers if isinstance(answer, EvaluatorAnswerRecord)
        ]
        score = build_score(typed_answers)
        return {
            "status": "completed",
            "score": score.model_dump(),
            "answers": [answer.model_dump() for answer in typed_answers],
        }

    @staticmethod
    def _build_next_question_prompt(question: GroundTruthQuestion) -> str:
        """Build one-turn instruction for the next required question."""
        return (
            "<Turn instruction>\n"
            f"Answer this exact question now: {question.question_id}\n"
            "Call submit_answer exactly once.\n"
            "Include summary_evidence_quote copied exactly from candidate summary XML.\n"
            "</Turn instruction>\n"
            "<Question>\n"
            f"question_id: {question.question_id}\n"
            f"dimension: {question.dimension}\n"
            f"question_text: {question.question_text}\n"
            "</Question>"
        )

    @staticmethod
    def _question_prompt_payload(question: GroundTruthQuestion) -> dict[str, Any]:
        """Return evaluator-visible question fields, excluding expected answer."""
        return {
            "question_id": question.question_id,
            "dimension": question.dimension,
            "question_text": question.question_text,
            "source_evidence_ids": question.source_evidence_ids,
            "speaker_ids": question.speaker_ids,
        }

    @staticmethod
    def _build_retry_prompt(
        *,
        question: GroundTruthQuestion,
        tool_status: str,
        tool_reason: str,
        attempts_remaining: int,
    ) -> str:
        """Build retry instruction when the last tool call failed validation."""
        return (
            "<Retry instruction>\n"
            f"Previous answer failed validation: {tool_status}\n"
            f"Reason: {tool_reason}\n"
            f"Attempts remaining for {question.question_id}: {attempts_remaining}\n"
            "Re-answer the same question_id using one submit_answer tool call.\n"
            "summary_evidence_quote must be an exact quote from candidate summary XML.\n"
            "</Retry instruction>"
        )

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize whitespace for robust substring matching."""
        return " ".join(value.split()).strip().lower()

    @classmethod
    def _is_summary_quote_present(cls, summary_xml: str, quote: str) -> bool:
        """Return whether quote appears verbatim in candidate summary text."""
        normalized_quote = cls._normalize_text(quote)
        if not normalized_quote:
            return False
        normalized_summary = cls._normalize_text(summary_xml)
        return normalized_quote in normalized_summary

    @staticmethod
    def _parse_confidence(value: Any) -> float | None:
        """Parse optional confidence value and clamp to ``[0.0, 1.0]``."""
        if value is None or value == "":
            return None
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(parsed, 1.0))

    @classmethod
    def _quote_tokens(cls, value: str) -> set[str]:
        """Return normalized content tokens for quote relevance checks."""
        return {
            token
            for token in cls._normalize_text(value).split(" ")
            if token and len(token) >= 3 and token not in cls._QUOTE_RELEVANCE_STOPWORDS
        }

    @classmethod
    def _is_quote_relevant_to_question(cls, *, quote: str, question_text: str) -> bool:
        """Require non-generic lexical overlap between quote and question text."""
        quote_tokens = cls._quote_tokens(quote)
        if not quote_tokens:
            return False
        question_tokens = cls._quote_tokens(question_text)
        if not question_tokens:
            return True
        shared_tokens = quote_tokens & question_tokens
        if not shared_tokens:
            return False

        non_generic_shared = {
            token for token in shared_tokens if token not in cls._QUOTE_GENERIC_TOKENS
        }
        if not non_generic_shared:
            return False

        distinctive_question_tokens = {
            token
            for token in question_tokens
            if len(token) >= 5 and token not in cls._QUOTE_GENERIC_TOKENS
        }
        if distinctive_question_tokens:
            distinctive_overlap = quote_tokens & distinctive_question_tokens
            required_overlap = 1
            return len(distinctive_overlap) >= required_overlap
        return True

    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Run strict QA evaluator loop and return score and answer trace."""
        del context
        request = QAEvaluatorTaskRequest.model_validate(task_data)
        question_set = GroundTruthSet.model_validate({"questions": request.questions})

        messages = [
            {
                "role": "system",
                "content": PromptManager.get_prompt("qa_evaluator_system"),
            },
            {
                "role": "user",
                "content": PromptManager.get_prompt(
                    "qa_evaluator_user",
                    summary_xml=request.summary_xml,
                    source_text=request.source_text,
                    total_questions=len(question_set.questions),
                ),
            },
        ]

        context_data = self._build_runtime_context(
            question_set=question_set,
            summary_xml=request.summary_xml,
            max_attempts_per_question=self.max_attempts_per_question,
        )
        if question_set.questions:
            messages.append(
                {
                    "role": "user",
                    "content": self._build_next_question_prompt(
                        question_set.questions[0]
                    ),
                }
            )
        final_result = await self.run_agent_loop(
            messages=messages,
            tools=self._build_tools(),
            max_turns=self.max_tool_turns,
            context_data=context_data,
        )
        if final_result is None:
            self._append_missing_answers(context_data)
            final_result = self._build_result_payload(context_data)

        await self.send_response(json.dumps(final_result), event_queue)

    async def process_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        context_data: dict[str, Any],
    ) -> tuple[bool, dict | None]:
        """Process one answer-tool submission for current question index."""
        question_queue = context_data.get("question_queue", [])
        answers = context_data.get("answers", [])
        current_index = int(context_data.get("current_question_index", 0))
        summary_xml = str(context_data.get("summary_xml", ""))
        attempts_raw = context_data.get("question_attempts", {})
        attempts = attempts_raw if isinstance(attempts_raw, dict) else {}
        max_attempts = int(context_data.get("max_attempts_per_question", 3))
        if not isinstance(question_queue, list) or not isinstance(answers, list):
            return True, {"status": "error", "score": {}, "answers": []}

        for tool_call in tool_calls:
            if current_index >= len(question_queue):
                break

            expected_question = question_queue[current_index]
            if not isinstance(expected_question, GroundTruthQuestion):
                current_index += 1
                continue

            name = str(tool_call.get("name", "")).strip()
            args = tool_call.get("arguments", {})
            if not isinstance(args, dict):
                args = {}

            predicted_answer = str(args.get("predicted_answer", "")).strip().lower()
            provided_question_id = str(args.get("question_id", "")).strip()
            reason = str(args.get("reason", "")).strip()
            summary_evidence_quote = str(args.get("summary_evidence_quote", "")).strip()
            confidence = self._parse_confidence(args.get("confidence"))

            tool_status = "accepted"
            is_correct = False
            tool_reason = reason

            if name != "submit_answer":
                tool_status = "invalid_tool"
                tool_reason = f"Unsupported tool '{name}'."
            elif provided_question_id != expected_question.question_id:
                tool_status = "invalid_question_id"
                tool_reason = (
                    f"Expected question_id '{expected_question.question_id}' but got "
                    f"'{provided_question_id}'."
                )
            elif predicted_answer not in {"yes", "no"}:
                tool_status = "invalid_answer_label"
                tool_reason = (
                    "predicted_answer must be 'yes' or 'no' for binary scoring."
                )
            elif not self._is_summary_quote_present(
                summary_xml, summary_evidence_quote
            ):
                tool_status = "missing_summary_evidence"
                tool_reason = "summary_evidence_quote must match text present in candidate summary XML."
            elif not self._is_quote_relevant_to_question(
                quote=summary_evidence_quote,
                question_text=expected_question.question_text,
            ):
                quote_tokens = sorted(self._quote_tokens(summary_evidence_quote))
                question_tokens = sorted(self._quote_tokens(expected_question.question_text))
                shared_tokens = sorted(set(quote_tokens) & set(question_tokens))
                logger.info(
                    "[QAEvaluator] Rejected irrelevant quote question_id=%s quote_tokens=%s question_tokens=%s shared_tokens=%s",
                    expected_question.question_id,
                    quote_tokens[:12],
                    question_tokens[:12],
                    shared_tokens[:12],
                )
                tool_status = "irrelevant_summary_evidence"
                tool_reason = (
                    "summary_evidence_quote must be relevant to current question_text."
                )
            else:
                is_correct = predicted_answer == expected_question.expected_answer
                if not tool_reason:
                    tool_reason = "Answer accepted."
            attempt_key = expected_question.question_id
            current_attempts = int(attempts.get(attempt_key, 0)) + 1
            attempts[attempt_key] = current_attempts

            tool_payload = {
                "status": tool_status,
                "question_id": expected_question.question_id,
                "is_correct": is_correct,
                "expected_answer": expected_question.expected_answer,
                "predicted_answer": predicted_answer if predicted_answer else "missing",
                "reason": tool_reason,
                "attempt": current_attempts,
                "max_attempts_per_question": max_attempts,
            }
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(tool_call.get("id", "missing_tool_call_id")),
                    "name": name if name else "submit_answer",
                    "content": json.dumps(tool_payload),
                }
            )

            if tool_status == "accepted":
                answer_record = EvaluatorAnswerRecord(
                    question_id=expected_question.question_id,
                    dimension=expected_question.dimension,
                    predicted_answer=predicted_answer,
                    expected_answer=expected_question.expected_answer,
                    is_correct=is_correct,
                    tool_status=tool_status,
                    reason=tool_reason,
                    summary_grounding_pass=True,
                    summary_evidence_quote=summary_evidence_quote,
                    confidence=confidence,
                )
                answers.append(answer_record)
                attempts.pop(attempt_key, None)
                current_index += 1
                if current_index < len(question_queue) and isinstance(
                    question_queue[current_index], GroundTruthQuestion
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": self._build_next_question_prompt(
                                question_queue[current_index]
                            ),
                        }
                    )
                if current_index % 10 == 0 or current_index >= len(question_queue):
                    self.event_bus.publish(
                        "status_message",
                        f"Evaluator progress: answered {current_index}/{len(question_queue)} questions.",
                    )
                continue

            attempts_remaining = max(0, max_attempts - current_attempts)
            if current_attempts >= max_attempts:
                answers.append(
                    EvaluatorAnswerRecord(
                        question_id=expected_question.question_id,
                        dimension=expected_question.dimension,
                        predicted_answer=(
                            predicted_answer if predicted_answer else "missing"
                        ),
                        expected_answer=expected_question.expected_answer,
                        is_correct=False,
                        tool_status=f"{tool_status}_max_retry_exceeded",
                        reason=tool_reason,
                        summary_grounding_pass=False,
                        summary_evidence_quote=summary_evidence_quote,
                        confidence=confidence,
                    )
                )
                attempts.pop(attempt_key, None)
                current_index += 1
                if current_index < len(question_queue) and isinstance(
                    question_queue[current_index], GroundTruthQuestion
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": self._build_next_question_prompt(
                                question_queue[current_index]
                            ),
                        }
                    )
                continue

            messages.append(
                {
                    "role": "user",
                    "content": self._build_retry_prompt(
                        question=expected_question,
                        tool_status=tool_status,
                        tool_reason=tool_reason,
                        attempts_remaining=attempts_remaining,
                    ),
                }
            )

        context_data["current_question_index"] = current_index
        context_data["question_attempts"] = attempts

        if current_index >= len(question_queue):
            return True, self._build_result_payload(context_data)

        return False, None
