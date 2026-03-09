"""Corrector agent that generates retry guidance for failed agent generations."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Protocol

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from card_framework.agents.base import BaseA2AExecutor
from card_framework.agents.dtos import (
    CorrectorFewShotExample,
    CorrectorTaskRequest,
    CorrectorTaskResponse,
)
from card_framework.shared.events import EventBus, get_event_bus
from card_framework.shared.llm_provider import LLMProvider
from card_framework.shared.logger_utils import logger
from card_framework.shared.prompt_manager import PromptManager


class CorrectionAdvisor(Protocol):
    """Protocol for components that can return correction guidance."""

    def build_retry_guidance(
        self, request: CorrectorTaskRequest
    ) -> CorrectorTaskResponse:
        """Analyze one failure payload and return retry guidance."""


class NoOpCorrectorAgent:
    """Fallback advisor used when corrective guidance is disabled."""

    def build_retry_guidance(
        self, request: CorrectorTaskRequest
    ) -> CorrectorTaskResponse:
        """Return empty guidance to preserve existing retry behavior."""
        del request
        return CorrectorTaskResponse(correction_instruction="", few_shot_examples=[])


class LLMCorrectorAgent:
    """LLM-backed corrector that turns failure logs into actionable retry guidance."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        event_bus: EventBus | None = None,
        max_tokens: int = 700,
        max_examples: int = 2,
    ) -> None:
        """Initialize corrector dependencies and output limits."""
        self.llm = llm
        self.event_bus = event_bus if event_bus is not None else get_event_bus()
        self.max_tokens = max(1, int(max_tokens))
        self.max_examples = max(1, int(max_examples))
        self._cache: dict[str, CorrectorTaskResponse] = {}

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize whitespace while preserving full text context."""
        return " ".join(str(value).split()).strip()

    @classmethod
    def _coerce_payload(cls, raw_text: str) -> dict[str, object] | None:
        """Extract JSON payload from model output text."""
        stripped = raw_text.strip()
        if not stripped:
            return None

        for candidate in cls._candidate_json_blobs(stripped):
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _candidate_json_blobs(raw_text: str) -> list[str]:
        """Return likely JSON blobs from direct, fenced, or embedded output."""
        candidates: list[str] = [raw_text]

        fenced_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
        if fenced_match:
            candidates.append(fenced_match.group(1).strip())

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end > start:
            candidates.append(raw_text[start : end + 1])

        return candidates

    def _coerce_response(
        self,
        *,
        payload: dict[str, object],
        request: CorrectorTaskRequest,
    ) -> CorrectorTaskResponse | None:
        """Validate and normalize LLM payload into typed corrector response."""
        raw_instruction = payload.get("correction_instruction")
        instruction = self._normalize_text(str(raw_instruction or ""))
        if not instruction:
            return None

        raw_examples = payload.get("few_shot_examples")
        examples: list[CorrectorFewShotExample] = []
        if isinstance(raw_examples, list):
            for raw_item in raw_examples[: self.max_examples]:
                if not isinstance(raw_item, dict):
                    continue
                bad_example = self._normalize_text(str(raw_item.get("bad_example", "")))
                corrected_example = self._normalize_text(
                    str(raw_item.get("corrected_example", ""))
                )
                rationale = self._normalize_text(str(raw_item.get("rationale", "")))
                if not bad_example or not corrected_example or not rationale:
                    continue
                examples.append(
                    CorrectorFewShotExample(
                        bad_example=bad_example,
                        corrected_example=corrected_example,
                        rationale=rationale,
                    )
                )

        if not examples:
            examples = self._fallback_examples(request=request)

        return CorrectorTaskResponse(
            correction_instruction=instruction,
            few_shot_examples=examples[: self.max_examples],
        )

    def _fallback_examples(
        self,
        *,
        request: CorrectorTaskRequest,
    ) -> list[CorrectorFewShotExample]:
        """Build deterministic examples when model output is malformed."""
        failure = request.failure_type.strip().lower()
        if request.target_agent == "GroundTruthCreator":
            if "json" in failure:
                return [
                    CorrectorFewShotExample(
                        bad_example="Here are your questions: [{...}]",
                        corrected_example='{"questions":[{"question_id":"Q001","dimension":"factualness","question_text":"...","expected_answer":"yes","source_evidence_ids":["E0001"],"speaker_ids":["SPEAKER_00"]}]}',
                        rationale="Return exactly one JSON object and no wrapper prose.",
                    )
                ]
            return [
                CorrectorFewShotExample(
                    bad_example='{"questions":[{"question_id":"Q77","expected_answer":"maybe"}]}',
                    corrected_example='{"questions":[{"question_id":"Q001","dimension":"factualness","question_text":"...","expected_answer":"yes","source_evidence_ids":["E0001"],"speaker_ids":["SPEAKER_00"]}]}',
                    rationale="Use required IDs and strict binary expected_answer.",
                )
            ]

        if "invalid_question_id" in failure:
            return [
                CorrectorFewShotExample(
                    bad_example='submit_answer({"question_id":"Q014","predicted_answer":"yes","summary_evidence_quote":"..."})',
                    corrected_example='submit_answer({"question_id":"Q013","predicted_answer":"yes","summary_evidence_quote":"..."})',
                    rationale="Use the exact current turn question_id only.",
                )
            ]
        if (
            "missing_summary_evidence" in failure
            or "irrelevant_summary_evidence" in failure
        ):
            return [
                CorrectorFewShotExample(
                    bad_example='submit_answer({"question_id":"Q013","predicted_answer":"yes","summary_evidence_quote":"general statement"})',
                    corrected_example='submit_answer({"question_id":"Q013","predicted_answer":"yes","summary_evidence_quote":"exact quote copied from summary xml"})',
                    rationale="Quote must be verbatim and directly relevant to question text.",
                )
            ]
        return [
            CorrectorFewShotExample(
                bad_example='submit_answer({"question_id":"Q013","predicted_answer":"maybe"})',
                corrected_example='submit_answer({"question_id":"Q013","predicted_answer":"no","summary_evidence_quote":"<exact quote>"})',
                rationale="Use binary label and include mandatory evidence quote.",
            )
        ]

    def _fallback_response(
        self, request: CorrectorTaskRequest
    ) -> CorrectorTaskResponse:
        """Return deterministic correction guidance if LLM response is unusable."""
        if request.target_agent == "GroundTruthCreator":
            instruction = (
                "Return JSON only with exact required question_ids and strict "
                "yes/no expected_answer values."
            )
        else:
            instruction = (
                "Answer the exact current question_id with one submit_answer call, "
                "binary predicted_answer, and a verbatim relevant summary quote."
            )
        return CorrectorTaskResponse(
            correction_instruction=instruction,
            few_shot_examples=self._fallback_examples(request=request)[
                : self.max_examples
            ],
        )

    @classmethod
    def _request_cache_key(cls, request: CorrectorTaskRequest) -> str:
        """Build stable cache key for repeated failure patterns."""
        payload = "|".join(
            [
                request.target_agent.strip(),
                request.failure_type.strip().lower(),
                request.failure_context,
                request.latest_output,
                request.expected_contract,
                str(request.attempt),
                str(request.max_attempts),
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def build_retry_guidance(
        self, request: CorrectorTaskRequest
    ) -> CorrectorTaskResponse:
        """Analyze one failed generation and return retry guidance."""
        cache_key = self._request_cache_key(request)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        system_prompt = PromptManager.get_prompt(
            "corrector_system",
            max_examples=self.max_examples,
        )
        user_prompt = PromptManager.get_prompt(
            "corrector_user",
            target_agent=request.target_agent,
            failure_type=request.failure_type,
            failure_context=request.failure_context,
            latest_output=request.latest_output,
            expected_contract=request.expected_contract,
            attempt=request.attempt,
            max_attempts=request.max_attempts,
            max_examples=self.max_examples,
        )

        try:
            raw_response = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            logger.warning(
                "[Corrector] Failed to generate guidance target_agent=%s failure_type=%s: %s",
                request.target_agent,
                request.failure_type,
                exc,
            )
            fallback = self._fallback_response(request)
            self._cache[cache_key] = fallback
            return fallback

        payload = self._coerce_payload(raw_response)
        if payload is None:
            logger.warning(
                "[Corrector] Failed to parse guidance payload target_agent=%s failure_type=%s",
                request.target_agent,
                request.failure_type,
            )
            fallback = self._fallback_response(request)
            self._cache[cache_key] = fallback
            return fallback

        response = self._coerce_response(payload=payload, request=request)
        if response is None:
            logger.warning(
                "[Corrector] Guidance payload missing required fields target_agent=%s failure_type=%s",
                request.target_agent,
                request.failure_type,
            )
            fallback = self._fallback_response(request)
            self._cache[cache_key] = fallback
            return fallback

        logger.info(
            "[Corrector] Generated guidance target_agent=%s failure_type=%s attempt=%s/%s",
            request.target_agent,
            request.failure_type,
            request.attempt,
            request.max_attempts,
        )
        self._cache[cache_key] = response
        return response


def render_correction_guidance(
    response: CorrectorTaskResponse,
    *,
    max_examples: int = 2,
) -> str:
    """Render correction payload into compact prompt-ready guidance text."""
    instruction = " ".join(response.correction_instruction.split()).strip()
    if not instruction:
        return ""

    lines: list[str] = [
        "<Corrector guidance>",
        f"Instruction: {instruction}",
    ]
    for index, example in enumerate(
        response.few_shot_examples[: max(1, max_examples)], 1
    ):
        lines.extend(
            [
                f"Example {index} bad: {' '.join(example.bad_example.split())}",
                f"Example {index} fixed: {' '.join(example.corrected_example.split())}",
                f"Example {index} rationale: {' '.join(example.rationale.split())}",
            ]
        )
    lines.append("</Corrector guidance>")
    return "\n".join(lines)


class CorrectorExecutor(BaseA2AExecutor):
    """A2A executor wrapper around ``LLMCorrectorAgent`` guidance generation."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        max_tokens: int = 700,
        max_examples: int = 2,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize the Corrector A2A executor."""
        super().__init__("Corrector")
        self.corrector = LLMCorrectorAgent(
            llm=llm,
            max_tokens=max_tokens,
            max_examples=max_examples,
            event_bus=event_bus,
        )

    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Generate correction instructions and few-shot examples."""
        del context
        request = CorrectorTaskRequest.model_validate(task_data)
        response = self.corrector.build_retry_guidance(request)
        await self.send_response(response.model_dump_json(), event_queue)

