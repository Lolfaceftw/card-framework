"""A2A executor for generating source-grounded QA benchmark questions."""

from __future__ import annotations

import json
import re
from typing import Any

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from agents.dtos import GroundTruthCreatorTaskRequest
from benchmark.qa_contracts import GroundTruthQuestion, GroundTruthSet
from events import EventBus, get_event_bus
from llm_provider import LLMProvider
from logger_utils import logger
from prompt_manager import PromptManager


class GroundTruthCreatorExecutor(BaseA2AExecutor):
    """Generate validated QA question sets from source transcript text."""

    _BATCH_SIZE = 25

    def __init__(
        self,
        llm: LLMProvider,
        max_generation_attempts: int = 3,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize generator executor and injected collaborators."""
        super().__init__("GroundTruthCreator")
        self.llm = llm
        self.max_generation_attempts = max(1, int(max_generation_attempts))
        self.event_bus = event_bus if event_bus is not None else get_event_bus()

    @staticmethod
    def _extract_json_payload(raw_text: str) -> dict[str, Any] | None:
        """Extract a JSON object payload from model output text."""
        stripped = raw_text.strip()
        if not stripped:
            return None

        direct = GroundTruthCreatorExecutor._coerce_questions_payload(stripped)
        if direct is not None:
            return direct

        fenced_match = re.search(r"```json\s*(.*?)\s*```", stripped, re.DOTALL)
        if fenced_match:
            fenced_payload = GroundTruthCreatorExecutor._coerce_questions_payload(
                fenced_match.group(1).strip()
            )
            if fenced_payload is not None:
                return fenced_payload

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end > start:
            object_payload = GroundTruthCreatorExecutor._coerce_questions_payload(
                stripped[start : end + 1]
            )
            if object_payload is not None:
                return object_payload

        array_start = stripped.find("[")
        array_end = stripped.rfind("]")
        if array_start != -1 and array_end > array_start:
            return GroundTruthCreatorExecutor._coerce_questions_payload(
                stripped[array_start : array_end + 1]
            )
        return None

    @staticmethod
    def _coerce_questions_payload(candidate: str) -> dict[str, Any] | None:
        """Coerce JSON string into canonical ``{'questions': [...]}`` payload."""
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return None

        if isinstance(payload, list):
            return {"questions": payload}
        if isinstance(payload, dict):
            questions = payload.get("questions")
            if isinstance(questions, list):
                return {"questions": questions}
            return None
        return None

    @staticmethod
    def _normalize_binary_answer(value: Any) -> str | None:
        """Normalize binary answer values into ``yes`` or ``no``."""
        raw = str(value).strip().lower()
        if raw in {"yes", "y", "true", "1"}:
            return "yes"
        if raw in {"no", "n", "false", "0"}:
            return "no"
        return None

    @staticmethod
    def _question_ids(start_index: int, count: int) -> list[str]:
        """Build contiguous ``Q###`` IDs for one generation batch."""
        return [f"Q{index:03d}" for index in range(start_index, start_index + count)]

    @staticmethod
    def _intent_key(question_text: str) -> str:
        """Normalize question text into a compact dedupe key."""
        lowered = question_text.casefold()
        word_spaced = re.sub(r"[^\w]+", " ", lowered, flags=re.UNICODE)
        no_underscore = re.sub(r"_+", " ", word_spaced)
        return re.sub(r"\s+", " ", no_underscore).strip()

    @staticmethod
    def _build_batch_user_prompt(
        *,
        source_text: str,
        dimension: str,
        expected_ids: list[str],
        avoid_question_texts: list[str] | None = None,
    ) -> str:
        """Build strict per-batch prompt to reduce truncation and schema drift."""
        avoid_lines = ""
        if avoid_question_texts:
            quoted = "\n".join(f"- {item}" for item in avoid_question_texts[-12:])
            avoid_lines = (
                "<Avoid intents>\n"
                "Do not repeat the meaning of any prior questions, including:\n"
                f"{quoted}\n"
                "</Avoid intents>\n"
            )
        return (
            "<Objective>\n"
            f"Generate exactly {len(expected_ids)} {dimension} questions.\n"
            "</Objective>\n"
            "<Context>\n"
            "Use only source transcript evidence. No prose outside JSON.\n"
            "</Context>\n"
            f"{avoid_lines}"
            "<Inputs>\n"
            f"Dimension: {dimension}\n"
            f"Required question_ids in exact order: {', '.join(expected_ids)}\n"
            f"Source transcript:\n{source_text}\n"
            "</Inputs>\n"
            "<Output contract>\n"
            "Return JSON only:\n"
            "{\n"
            '  "questions": [\n'
            "    {\n"
            '      "question_id": "Q001",\n'
            f'      "dimension": "{dimension}",\n'
            '      "question_text": "yes/no question",\n'
            '      "expected_answer": "yes" | "no",\n'
            '      "source_evidence_ids": ["E0001"],\n'
            '      "speaker_ids": ["SPEAKER_00"]\n'
            "    }\n"
            "  ]\n"
            "}\n"
            "</Output contract>\n"
            "<Rules>\n"
            "- Output exactly the requested count.\n"
            "- Use exact question_ids in exact order.\n"
            "- expected_answer must be yes/no.\n"
            "- Include at least one source_evidence_id per question.\n"
            "</Rules>"
        )

    def _validate_batch_payload(
        self,
        *,
        payload: dict[str, Any],
        dimension: str,
        expected_ids: list[str],
    ) -> list[GroundTruthQuestion] | None:
        """Validate and normalize one generated question batch."""
        raw_questions = payload.get("questions")
        if not isinstance(raw_questions, list):
            return None
        if len(raw_questions) < len(expected_ids):
            return None
        if len(raw_questions) > len(expected_ids):
            logger.warning(
                "[GroundTruthCreator] %s batch %s-%s produced %s items; truncating to %s.",
                dimension,
                expected_ids[0],
                expected_ids[-1],
                len(raw_questions),
                len(expected_ids),
            )
            raw_questions = raw_questions[: len(expected_ids)]

        normalized_questions: list[GroundTruthQuestion] = []
        seen_intents: set[str] = set()
        for index, expected_id in enumerate(expected_ids):
            raw_question = raw_questions[index]
            if not isinstance(raw_question, dict):
                return None

            normalized_answer = self._normalize_binary_answer(
                raw_question.get("expected_answer")
            )
            if normalized_answer is None:
                return None

            normalized_payload = {
                "question_id": expected_id,
                "dimension": dimension,
                "question_text": str(raw_question.get("question_text", "")).strip(),
                "expected_answer": normalized_answer,
                "source_evidence_ids": raw_question.get("source_evidence_ids", []),
                "speaker_ids": sorted(
                    set(raw_question.get("speaker_ids", []))
                    | set(
                        re.findall(
                            r"SPEAKER_\d+",
                            str(raw_question.get("question_text", "")).upper(),
                        )
                    )
                ),
            }
            try:
                question = GroundTruthQuestion.model_validate(normalized_payload)
            except Exception:
                return None
            intent_key = self._intent_key(question.question_text)
            if not intent_key or intent_key in seen_intents:
                return None
            seen_intents.add(intent_key)
            normalized_questions.append(question)
        return normalized_questions

    def _generate_batch(
        self,
        *,
        system_prompt: str,
        source_text: str,
        dimension: str,
        expected_ids: list[str],
        disallowed_intents: set[str] | None = None,
        disallowed_question_texts: list[str] | None = None,
    ) -> list[GroundTruthQuestion]:
        """Generate one validated batch of questions for a single dimension."""
        disallowed = disallowed_intents if disallowed_intents is not None else set()
        avoid_question_texts: list[str] = (
            list(disallowed_question_texts) if disallowed_question_texts else []
        )
        for attempt in range(1, self.max_generation_attempts + 1):
            user_prompt = self._build_batch_user_prompt(
                source_text=source_text,
                dimension=dimension,
                expected_ids=expected_ids,
                avoid_question_texts=avoid_question_texts,
            )
            raw_response = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=3_000,
            )
            payload = self._extract_json_payload(raw_response)
            if payload is None:
                logger.warning(
                    "[GroundTruthCreator] %s batch %s-%s attempt %s/%s: failed to parse JSON. preview=%s",
                    dimension,
                    expected_ids[0],
                    expected_ids[-1],
                    attempt,
                    self.max_generation_attempts,
                    raw_response[:500].replace("\n", "\\n"),
                )
                continue

            batch = self._validate_batch_payload(
                payload=payload,
                dimension=dimension,
                expected_ids=expected_ids,
            )
            if batch is None:
                logger.warning(
                    "[GroundTruthCreator] %s batch %s-%s attempt %s/%s: schema validation failed.",
                    dimension,
                    expected_ids[0],
                    expected_ids[-1],
                    attempt,
                    self.max_generation_attempts,
                )
                continue
            intent_to_question_text = {
                self._intent_key(question.question_text): question.question_text
                for question in batch
            }
            overlap_intents = [
                intent for intent in intent_to_question_text if intent in disallowed
            ]
            if overlap_intents:
                overlap_examples = [
                    intent_to_question_text[intent]
                    for intent in overlap_intents
                    if intent in intent_to_question_text
                ]
                avoid_question_texts = list(
                    dict.fromkeys([*avoid_question_texts, *overlap_examples])
                )
                logger.warning(
                    "[GroundTruthCreator] %s batch %s-%s attempt %s/%s: rejected due to %s global duplicate intents.",
                    dimension,
                    expected_ids[0],
                    expected_ids[-1],
                    attempt,
                    self.max_generation_attempts,
                    len(overlap_intents),
                )
                continue
            return batch

        raise RuntimeError(
            "GroundTruthCreator failed to generate valid batch "
            f"{expected_ids[0]}-{expected_ids[-1]} after {self.max_generation_attempts} attempts."
        )

    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Generate questions and return validated JSON payload to caller."""
        del context
        request = GroundTruthCreatorTaskRequest.model_validate(task_data)
        if (
            int(request.factual_question_count) != 50
            or int(request.naturalness_question_count) != 50
        ):
            raise ValueError(
                "GroundTruthCreator currently requires factual_question_count=50 and "
                "naturalness_question_count=50."
            )

        source_text = request.source_text.strip()
        if not source_text:
            raise ValueError("source_text must be non-empty")

        system_prompt = PromptManager.get_prompt("qa_ground_truth_system")

        all_questions: list[GroundTruthQuestion] = []
        seen_global_intents: set[str] = set()
        seen_global_question_texts: list[str] = []
        next_question_index = 1
        remaining_by_dimension = [
            ("factualness", request.factual_question_count),
            ("naturalness", request.naturalness_question_count),
        ]
        for dimension, total_count in remaining_by_dimension:
            remaining = int(total_count)
            while remaining > 0:
                batch_size = min(self._BATCH_SIZE, remaining)
                expected_ids = self._question_ids(next_question_index, batch_size)
                batch = self._generate_batch(
                    system_prompt=system_prompt,
                    source_text=source_text,
                    dimension=dimension,
                    expected_ids=expected_ids,
                    disallowed_intents=seen_global_intents,
                    disallowed_question_texts=seen_global_question_texts,
                )
                all_questions.extend(batch)
                seen_global_intents.update(
                    self._intent_key(question.question_text) for question in batch
                )
                seen_global_question_texts.extend(
                    question.question_text for question in batch
                )
                next_question_index += batch_size
                remaining -= batch_size

        question_set = GroundTruthSet.model_validate(
            {"questions": [question.model_dump() for question in all_questions]}
        )
        response_payload = {
            "questions": [q.model_dump() for q in question_set.questions]
        }
        self.event_bus.publish(
            "status_message",
            f"Generated QA ground truth: {len(question_set.questions)} questions.",
        )
        await self.send_response(json.dumps(response_payload), event_queue)
