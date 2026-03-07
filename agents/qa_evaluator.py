"""A2A executor for QA-based summary benchmark scoring."""

from __future__ import annotations

from difflib import SequenceMatcher
import html
import json
import re
from typing import Any

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from agents.corrector import CorrectionAdvisor, render_correction_guidance
from agents.dtos import CorrectorTaskRequest, QAEvaluatorTaskRequest
from benchmark.qa_contracts import (
    EvaluatorAnswerRecord,
    GroundTruthQuestion,
    GroundTruthSet,
    build_score,
)
from benchmark.qa_settings import EvaluatorRuntimeConfig, QuoteRelevanceConfig
from events import EventBus, get_event_bus
from llm_provider import EmbeddingProvider, LLMProvider
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
    _XML_TAG_PATTERN = re.compile(r"<[^>]+>")
    _NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9\s]+")
    _SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
    _SEMANTIC_THRESHOLD_DEFAULT = 0.5

    def __init__(
        self,
        llm: LLMProvider,
        max_tool_turns: int = 320,
        max_attempts_per_question: int = 3,
        evaluator_runtime_config: EvaluatorRuntimeConfig | None = None,
        event_bus: EventBus | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        corrector_agent: CorrectionAdvisor | None = None,
    ) -> None:
        """Initialize evaluator executor and injected collaborators."""
        super().__init__("Evaluator")
        self.llm = llm
        (
            resolved_runtime_config,
            resolved_event_bus,
            resolved_embedding_provider,
        ) = self._resolve_legacy_positional_args(
            evaluator_runtime_config=evaluator_runtime_config,
            event_bus=event_bus,
            embedding_provider=embedding_provider,
        )
        self.runtime_config = self._resolve_runtime_config(
            evaluator_runtime_config=resolved_runtime_config,
            max_tool_turns=max_tool_turns,
            max_attempts_per_question=max_attempts_per_question,
        )
        self.max_tool_turns = self.runtime_config.max_tool_turns
        self.max_attempts_per_question = self.runtime_config.max_attempts_per_question
        self.event_bus = (
            resolved_event_bus if resolved_event_bus is not None else get_event_bus()
        )
        self.embedding_provider = resolved_embedding_provider
        self.corrector_agent = corrector_agent
        logger.info(
            "[QAEvaluator] Runtime config max_tool_turns=%s max_attempts=%s chat_max_tokens=%s no_tool_call_patience=%s max_tool_calls_per_turn=%s quote_relevance_mode=%s semantic_threshold=%s auto_repair=%s repair_min_score=%.3f min_candidate_chars=%s",
            self.runtime_config.max_tool_turns,
            self.runtime_config.max_attempts_per_question,
            self.runtime_config.chat_max_tokens,
            self.runtime_config.no_tool_call_patience,
            self.runtime_config.max_tool_calls_per_turn,
            self.runtime_config.quote_relevance.mode,
            self.runtime_config.quote_relevance.semantic_threshold,
            self.runtime_config.quote_relevance.auto_repair,
            self.runtime_config.quote_relevance.repair_min_score,
            self.runtime_config.quote_relevance.min_candidate_chars,
        )

    @classmethod
    def _resolve_legacy_positional_args(
        cls,
        *,
        evaluator_runtime_config: EvaluatorRuntimeConfig
        | EventBus
        | EmbeddingProvider
        | None,
        event_bus: EventBus | None,
        embedding_provider: EmbeddingProvider | None,
    ) -> tuple[
        EvaluatorRuntimeConfig | None, EventBus | None, EmbeddingProvider | None
    ]:
        """Preserve compatibility with legacy positional constructor call patterns."""
        resolved_runtime_config = evaluator_runtime_config
        resolved_event_bus = event_bus
        resolved_embedding_provider = embedding_provider

        if isinstance(evaluator_runtime_config, EventBus):
            if resolved_event_bus is None:
                resolved_event_bus = evaluator_runtime_config
            resolved_runtime_config = None
            logger.warning(
                "[QAEvaluator] Detected legacy positional EventBus argument; "
                "pass event_bus=... explicitly to avoid ambiguity."
            )
        elif isinstance(evaluator_runtime_config, EmbeddingProvider):
            if resolved_embedding_provider is None:
                resolved_embedding_provider = evaluator_runtime_config
            resolved_runtime_config = None
            logger.warning(
                "[QAEvaluator] Detected legacy positional EmbeddingProvider argument; "
                "pass embedding_provider=... explicitly to avoid ambiguity."
            )
        return (
            resolved_runtime_config,
            resolved_event_bus,
            resolved_embedding_provider,
        )

    @classmethod
    def _resolve_runtime_config(
        cls,
        *,
        evaluator_runtime_config: EvaluatorRuntimeConfig | None,
        max_tool_turns: int,
        max_attempts_per_question: int,
    ) -> EvaluatorRuntimeConfig:
        """Resolve runtime config, preserving backward-compatible constructor args."""
        if evaluator_runtime_config is not None:
            if not isinstance(evaluator_runtime_config, EvaluatorRuntimeConfig):
                raise TypeError(
                    "evaluator_runtime_config must be EvaluatorRuntimeConfig when provided."
                )
            return evaluator_runtime_config
        return EvaluatorRuntimeConfig(
            max_tool_turns=max(1, int(max_tool_turns)),
            max_attempts_per_question=max(1, int(max_attempts_per_question)),
            chat_max_tokens=160,
            no_tool_call_patience=3,
            max_tool_calls_per_turn=1,
            per_question_concurrency=1,
            quote_relevance=QuoteRelevanceConfig(
                mode="hybrid",
                min_shared_tokens=1,
                min_distinctive_shared_tokens=1,
                min_token_length=3,
                semantic_threshold=None,
                auto_repair=True,
                repair_min_score=0.25,
                min_candidate_chars=6,
            ),
        )

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
    def _build_runtime_context_from_questions(
        *,
        questions: list[GroundTruthQuestion],
        summary_xml: str,
        evaluator_runtime_config: EvaluatorRuntimeConfig,
    ) -> dict[str, Any]:
        """Build loop runtime context from validated question list."""
        return {
            "question_queue": list(questions),
            "current_question_index": 0,
            "answers": [],
            "max_tool_calls_per_turn": evaluator_runtime_config.max_tool_calls_per_turn,
            "tool_choice": {"type": "function", "function": {"name": "submit_answer"}},
            "chat_max_tokens": evaluator_runtime_config.chat_max_tokens,
            "no_tool_call_patience": evaluator_runtime_config.no_tool_call_patience,
            "no_tool_call_count": 0,
            "summary_xml": summary_xml,
            "question_attempts": {},
            "max_attempts_per_question": evaluator_runtime_config.max_attempts_per_question,
            "quote_relevance_config": evaluator_runtime_config.quote_relevance,
            "correction_history": set(),
        }

    @staticmethod
    def _build_runtime_context(
        *,
        question_set: GroundTruthSet,
        summary_xml: str,
        evaluator_runtime_config: EvaluatorRuntimeConfig,
    ) -> dict[str, Any]:
        """Build loop runtime context for strict evaluator execution."""
        return QAEvaluatorExecutor._build_runtime_context_from_questions(
            questions=list(question_set.questions),
            summary_xml=summary_xml,
            evaluator_runtime_config=evaluator_runtime_config,
        )

    @staticmethod
    def _validate_request_questions(
        questions: list[dict[str, Any]],
    ) -> list[GroundTruthQuestion]:
        """Validate incoming question payload items without set-size constraints."""
        return [GroundTruthQuestion.model_validate(item) for item in questions]

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
        correction_guidance: str | None = None,
    ) -> str:
        """Build retry instruction when the last tool call failed validation."""
        guidance_section = ""
        if correction_guidance:
            guidance_section = f"\n{correction_guidance}\n"
        return (
            "<Retry instruction>\n"
            f"Previous answer failed validation: {tool_status}\n"
            f"Reason: {tool_reason}\n"
            f"Attempts remaining for {question.question_id}: {attempts_remaining}\n"
            "Re-answer the same question_id using one submit_answer tool call.\n"
            "summary_evidence_quote must be grounded in candidate summary XML and follow quote relevance settings.\n"
            f"{guidance_section}"
            "</Retry instruction>"
        )

    def _build_expected_contract(self, *, question: GroundTruthQuestion) -> str:
        """Return explicit evaluator output contract for one retry turn."""
        return (
            "Call submit_answer exactly once.\n"
            f"question_id must be exactly {question.question_id}.\n"
            "predicted_answer must be `yes` or `no`.\n"
            "summary_evidence_quote must be copied verbatim from summary XML and "
            "be relevant to the current question."
        )

    def _request_correction_guidance(
        self,
        *,
        question: GroundTruthQuestion,
        tool_status: str,
        tool_reason: str,
        provided_question_id: str,
        predicted_answer: str,
        submitted_summary_evidence_quote: str,
        normalized_summary_evidence_quote: str,
        attempts_used: int,
        max_attempts: int,
    ) -> str:
        """Request evaluator retry guidance from Corrector using full failure log."""
        if self.corrector_agent is None:
            return ""

        failure_context_payload = {
            "question": self._question_prompt_payload(question),
            "tool_status": tool_status,
            "tool_reason": tool_reason,
            "provided_question_id": provided_question_id,
            "predicted_answer": predicted_answer,
            "submitted_summary_evidence_quote": submitted_summary_evidence_quote,
            "normalized_summary_evidence_quote": normalized_summary_evidence_quote,
            "attempts_used": attempts_used,
            "max_attempts": max_attempts,
            "quote_relevance_mode": self.runtime_config.quote_relevance.mode,
            "quote_relevance_semantic_threshold": (
                self.runtime_config.quote_relevance.semantic_threshold
            ),
            "quote_relevance_auto_repair": (
                self.runtime_config.quote_relevance.auto_repair
            ),
        }
        request = CorrectorTaskRequest(
            target_agent="Evaluator",
            failure_type=tool_status,
            failure_context=json.dumps(
                failure_context_payload,
                ensure_ascii=False,
                indent=2,
            ),
            latest_output=json.dumps(
                {
                    "question_id": provided_question_id,
                    "predicted_answer": predicted_answer,
                    "submitted_summary_evidence_quote": submitted_summary_evidence_quote,
                    "normalized_summary_evidence_quote": normalized_summary_evidence_quote,
                },
                ensure_ascii=False,
            ),
            attempt=attempts_used,
            max_attempts=max_attempts,
            expected_contract=self._build_expected_contract(question=question),
        )
        try:
            guidance = self.corrector_agent.build_retry_guidance(request)
        except Exception as exc:
            logger.warning(
                "[QAEvaluator] Corrector failed question_id=%s status=%s attempt=%s/%s: %s",
                question.question_id,
                tool_status,
                attempts_used,
                max_attempts,
                exc,
            )
            return ""

        rendered = render_correction_guidance(guidance, max_examples=2)
        if rendered:
            logger.info(
                "[QAEvaluator] Applied Corrector guidance question_id=%s status=%s attempt=%s/%s",
                question.question_id,
                tool_status,
                attempts_used,
                max_attempts,
            )
            self.event_bus.publish(
                "agent_message",
                agent_name="Corrector",
                message=rendered,
                markdown=False,
            )
        return rendered

    @staticmethod
    def _normalize_text(value: str) -> str:
        """Normalize whitespace for robust substring matching."""
        normalized = html.unescape(value).lower()
        normalized = normalized.replace("\u201c", '"').replace("\u201d", '"')
        normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
        return " ".join(normalized.split()).strip()

    @classmethod
    def _strip_xml_tags(cls, value: str) -> str:
        """Remove XML tags while preserving surrounding text for loose matching."""
        return cls._XML_TAG_PATTERN.sub(" ", value)

    @classmethod
    def _alnum_only(cls, value: str) -> str:
        """Normalize text to alphanumeric and whitespace for punctuation-tolerant match."""
        compact = cls._NON_ALNUM_PATTERN.sub(" ", value)
        return " ".join(compact.split()).strip()

    @classmethod
    def _is_summary_quote_present(cls, summary_xml: str, quote: str) -> bool:
        """Return whether quote appears in candidate summary with robust normalization."""
        normalized_quote = cls._normalize_text(quote)
        if not normalized_quote:
            return False
        normalized_summary = cls._normalize_text(summary_xml)
        if normalized_quote in normalized_summary:
            return True
        normalized_summary_text = cls._normalize_text(cls._strip_xml_tags(summary_xml))
        if normalized_quote in normalized_summary_text:
            return True
        loose_quote = cls._alnum_only(normalized_quote)
        if not loose_quote:
            return False
        return loose_quote in cls._alnum_only(normalized_summary_text)

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

    @staticmethod
    def _normalize_predicted_answer(value: Any) -> str:
        """Normalize common binary-label variants to strict ``yes``/``no``."""
        normalized = str(value).strip().lower()
        mapping = {
            "yes": "yes",
            "y": "yes",
            "true": "yes",
            "1": "yes",
            "no": "no",
            "n": "no",
            "false": "no",
            "0": "no",
        }
        return mapping.get(normalized, normalized)

    @classmethod
    def _quote_tokens(cls, value: str) -> set[str]:
        """Return normalized content tokens for quote relevance checks."""
        return {
            token
            for token in cls._normalize_text(value).split(" ")
            if token and len(token) >= 3 and token not in cls._QUOTE_RELEVANCE_STOPWORDS
        }

    @classmethod
    def _quote_tokens_for_config(
        cls,
        *,
        value: str,
        min_token_length: int,
    ) -> set[str]:
        """Return normalized content tokens with configurable token-length floor."""
        return {
            token
            for token in cls._normalize_text(value).split(" ")
            if token
            and len(token) >= min_token_length
            and token not in cls._QUOTE_RELEVANCE_STOPWORDS
        }

    def _compute_semantic_similarity(
        self,
        text1: str,
        text2: str,
    ) -> float:
        """Compute semantic similarity using embeddings, with lexical fallback."""
        normalized_text1 = self._normalize_text(text1)
        normalized_text2 = self._normalize_text(text2)
        if not normalized_text1 or not normalized_text2:
            return 0.0

        if self.embedding_provider is not None:
            emb1 = self.embedding_provider.encode([normalized_text1], normalize=True)
            emb2 = self.embedding_provider.encode([normalized_text2], normalize=True)
            return float((emb1 @ emb2.T)[0, 0])

        tokens1 = {
            token
            for token in self._quote_tokens(normalized_text1)
            if token not in self._QUOTE_GENERIC_TOKENS
        }
        tokens2 = {
            token
            for token in self._quote_tokens(normalized_text2)
            if token not in self._QUOTE_GENERIC_TOKENS
        }
        if not tokens1 or not tokens2:
            return 0.0
        jaccard = len(tokens1 & tokens2) / float(len(tokens1 | tokens2))
        sequence_ratio = SequenceMatcher(
            a=" ".join(sorted(tokens2)),
            b=" ".join(sorted(tokens1)),
        ).ratio()
        return max(sequence_ratio, jaccard)

    @classmethod
    def _candidate_quotes_from_summary(
        cls, summary_xml: str, min_candidate_chars: int
    ) -> list[str]:
        """Extract candidate evidence snippets from summary text."""
        plain_text = cls._strip_xml_tags(summary_xml)
        lines = [line.strip() for line in plain_text.splitlines() if line.strip()]
        if not lines:
            lines = [plain_text.strip()] if plain_text.strip() else []

        candidates: list[str] = []
        seen: set[str] = set()
        for line in lines:
            sentences = cls._SENTENCE_SPLIT_PATTERN.split(line)
            parts = [part.strip() for part in sentences if part.strip()]
            if not parts:
                parts = [line]
            for part in parts:
                normalized = cls._normalize_text(part)
                if not normalized or normalized in seen:
                    continue
                if len(normalized) < max(1, min_candidate_chars):
                    continue
                seen.add(normalized)
                candidates.append(part)
        return candidates

    def _score_quote_candidate(
        self,
        *,
        candidate_quote: str,
        question_text: str,
        relevance_config: QuoteRelevanceConfig,
    ) -> float:
        """Score candidate quote alignment to the current question."""
        quote_tokens = self._quote_tokens_for_config(
            value=candidate_quote,
            min_token_length=relevance_config.min_token_length,
        )
        question_tokens = self._quote_tokens_for_config(
            value=question_text,
            min_token_length=relevance_config.min_token_length,
        )
        if not quote_tokens:
            return 0.0
        if not question_tokens:
            return 1.0

        shared_tokens = quote_tokens & question_tokens
        lexical_score = len(shared_tokens) / float(max(len(question_tokens), 1))
        distinctive_question_tokens = {
            token
            for token in question_tokens
            if len(token) >= max(5, relevance_config.min_token_length)
            and token not in self._QUOTE_GENERIC_TOKENS
        }
        distinctive_shared = quote_tokens & distinctive_question_tokens
        distinctive_score = len(distinctive_shared) / float(
            max(len(distinctive_question_tokens), 1)
        )
        semantic_score = self._compute_semantic_similarity(
            candidate_quote, question_text
        )
        return max(lexical_score, distinctive_score, semantic_score)

    def _select_repair_quote(
        self,
        *,
        summary_xml: str,
        question_text: str,
        relevance_config: QuoteRelevanceConfig,
    ) -> tuple[str, float] | None:
        """Select the best grounded replacement quote when model quote is invalid."""
        best_quote = ""
        best_score = -1.0
        for candidate_quote in self._candidate_quotes_from_summary(
            summary_xml,
            min_candidate_chars=relevance_config.min_candidate_chars,
        ):
            score = self._score_quote_candidate(
                candidate_quote=candidate_quote,
                question_text=question_text,
                relevance_config=relevance_config,
            )
            if score > best_score:
                best_quote = candidate_quote
                best_score = score
        if not best_quote:
            return None
        if best_score < relevance_config.repair_min_score:
            return None
        if not self._is_quote_relevant_to_question(
            quote=best_quote,
            question_text=question_text,
            relevance_config=relevance_config,
        ):
            return None
        return best_quote, best_score

    @classmethod
    def _is_quote_relevant_to_question_lexical(
        cls,
        *,
        quote: str,
        question_text: str,
        relevance_config: QuoteRelevanceConfig,
    ) -> bool:
        """Require configurable non-generic lexical overlap between quote and question."""
        quote_tokens = cls._quote_tokens_for_config(
            value=quote, min_token_length=relevance_config.min_token_length
        )
        if not quote_tokens:
            return False
        question_tokens = cls._quote_tokens_for_config(
            value=question_text, min_token_length=relevance_config.min_token_length
        )
        if not question_tokens:
            return True
        shared_tokens = quote_tokens & question_tokens
        if len(shared_tokens) < relevance_config.min_shared_tokens:
            return False

        non_generic_shared = {
            token for token in shared_tokens if token not in cls._QUOTE_GENERIC_TOKENS
        }
        if not non_generic_shared:
            return False

        distinctive_question_tokens = {
            token
            for token in question_tokens
            if len(token) >= max(5, relevance_config.min_token_length)
            and token not in cls._QUOTE_GENERIC_TOKENS
        }
        if (
            relevance_config.min_distinctive_shared_tokens > 0
            and distinctive_question_tokens
        ):
            distinctive_overlap = quote_tokens & distinctive_question_tokens
            return (
                len(distinctive_overlap)
                >= relevance_config.min_distinctive_shared_tokens
            )
        return True

    def _is_quote_relevant_to_question_semantic(
        self,
        *,
        quote: str,
        question_text: str,
        relevance_config: QuoteRelevanceConfig,
    ) -> bool:
        """Apply similarity-based relevance for quote/question alignment."""
        quote_text = self._normalize_text(quote)
        question_text_normalized = self._normalize_text(question_text)
        if not quote_text:
            return False
        if not question_text_normalized:
            return True
        quote_tokens = self._quote_tokens_for_config(
            value=quote_text, min_token_length=relevance_config.min_token_length
        )
        question_tokens = self._quote_tokens_for_config(
            value=question_text_normalized,
            min_token_length=relevance_config.min_token_length,
        )
        if not quote_tokens or not question_tokens:
            return False
        similarity_score = self._compute_semantic_similarity(quote_text, question_text)
        threshold = (
            relevance_config.semantic_threshold
            if relevance_config.semantic_threshold is not None
            else self._SEMANTIC_THRESHOLD_DEFAULT
        )
        return similarity_score >= threshold

    def _is_quote_relevant_to_question(
        self,
        *,
        quote: str,
        question_text: str,
        relevance_config: QuoteRelevanceConfig,
    ) -> bool:
        """Resolve configured relevance strategy for summary evidence quote checks."""
        if relevance_config.mode == "off":
            return True
        if relevance_config.mode == "semantic_similarity":
            return self._is_quote_relevant_to_question_semantic(
                quote=quote,
                question_text=question_text,
                relevance_config=relevance_config,
            )
        lexical_pass = self._is_quote_relevant_to_question_lexical(
            quote=quote,
            question_text=question_text,
            relevance_config=relevance_config,
        )
        if lexical_pass:
            return True
        if relevance_config.mode == "hybrid":
            return self._is_quote_relevant_to_question_semantic(
                quote=quote,
                question_text=question_text,
                relevance_config=relevance_config,
            )
        return False

    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Run strict QA evaluator loop and return score and answer trace."""
        del context
        request = QAEvaluatorTaskRequest.model_validate(task_data)
        validated_questions = self._validate_request_questions(request.questions)

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
                    total_questions=len(validated_questions),
                    quote_relevance_mode=self.runtime_config.quote_relevance.mode,
                    quote_relevance_min_shared_tokens=(
                        self.runtime_config.quote_relevance.min_shared_tokens
                    ),
                    quote_relevance_min_distinctive_shared_tokens=(
                        self.runtime_config.quote_relevance.min_distinctive_shared_tokens
                    ),
                    quote_relevance_semantic_threshold=(
                        self.runtime_config.quote_relevance.semantic_threshold
                    ),
                    quote_relevance_auto_repair=(
                        self.runtime_config.quote_relevance.auto_repair
                    ),
                    quote_relevance_repair_min_score=(
                        self.runtime_config.quote_relevance.repair_min_score
                    ),
                    quote_relevance_min_candidate_chars=(
                        self.runtime_config.quote_relevance.min_candidate_chars
                    ),
                ),
            },
        ]

        context_data = self._build_runtime_context_from_questions(
            questions=validated_questions,
            summary_xml=request.summary_xml,
            evaluator_runtime_config=self.runtime_config,
        )
        if validated_questions:
            messages.append(
                {
                    "role": "user",
                    "content": self._build_next_question_prompt(validated_questions[0]),
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
        correction_history_raw = context_data.get("correction_history")
        correction_history: set[tuple[str, str]]
        if isinstance(correction_history_raw, set):
            correction_history = correction_history_raw
        else:
            correction_history = set()
        max_attempts = int(context_data.get("max_attempts_per_question", 3))
        quote_relevance_raw = context_data.get("quote_relevance_config")
        quote_relevance_config = (
            quote_relevance_raw
            if isinstance(quote_relevance_raw, QuoteRelevanceConfig)
            else self.runtime_config.quote_relevance
        )
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

            predicted_answer = self._normalize_predicted_answer(
                args.get("predicted_answer", "")
            )
            provided_question_id = str(args.get("question_id", "")).strip()
            reason = str(args.get("reason", "")).strip()
            summary_evidence_quote = str(args.get("summary_evidence_quote", "")).strip()
            submitted_summary_evidence_quote = summary_evidence_quote
            confidence = self._parse_confidence(args.get("confidence"))
            if (
                len(question_queue) == 1
                and provided_question_id != expected_question.question_id
            ):
                logger.info(
                    "[QAEvaluator] Auto-corrected question_id from '%s' to '%s' for single-question request.",
                    provided_question_id or "<missing>",
                    expected_question.question_id,
                )
                provided_question_id = expected_question.question_id

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
            elif predicted_answer == "no" and not summary_evidence_quote:
                tool_status = "accepted"
                tool_reason = "No supporting quote provided for 'no' answer; accepted as absence claim."
            elif not self._is_summary_quote_present(
                summary_xml, summary_evidence_quote
            ):
                tool_status = "missing_summary_evidence"
                tool_reason = "summary_evidence_quote must match text present in candidate summary XML."
            elif not self._is_quote_relevant_to_question(
                quote=summary_evidence_quote,
                question_text=expected_question.question_text,
                relevance_config=quote_relevance_config,
            ):
                quote_tokens = sorted(self._quote_tokens(summary_evidence_quote))
                question_tokens = sorted(
                    self._quote_tokens(expected_question.question_text)
                )
                shared_tokens = sorted(set(quote_tokens) & set(question_tokens))
                logger.info(
                    "[QAEvaluator] Rejected irrelevant quote question_id=%s mode=%s quote_tokens=%s question_tokens=%s shared_tokens=%s",
                    expected_question.question_id,
                    quote_relevance_config.mode,
                    quote_tokens[:12],
                    question_tokens[:12],
                    shared_tokens[:12],
                )
                tool_status = "irrelevant_summary_evidence"
                tool_reason = (
                    "summary_evidence_quote must be relevant to current question_text."
                )
            if (
                tool_status
                in {"missing_summary_evidence", "irrelevant_summary_evidence"}
                and quote_relevance_config.auto_repair
            ):
                repaired_quote = self._select_repair_quote(
                    summary_xml=summary_xml,
                    question_text=expected_question.question_text,
                    relevance_config=quote_relevance_config,
                )
                if repaired_quote is not None:
                    replacement_quote, replacement_score = repaired_quote
                    summary_evidence_quote = replacement_quote
                    tool_status = "accepted"
                    tool_reason = "Answer accepted after deterministic quote repair."
                    logger.info(
                        "[QAEvaluator] Auto-repaired quote question_id=%s score=%.3f mode=%s",
                        expected_question.question_id,
                        replacement_score,
                        quote_relevance_config.mode,
                    )
                else:
                    # Avoid carrying forward hallucinated or irrelevant quote strings.
                    summary_evidence_quote = ""
                    if tool_status == "irrelevant_summary_evidence":
                        tool_status = "missing_summary_evidence"
                    tool_reason = "No grounded summary_evidence_quote could be deterministically repaired."
            if tool_status == "accepted":
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
                    summary_grounding_pass=bool(summary_evidence_quote),
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

            failure_signature = (expected_question.question_id, tool_status)
            correction_guidance = ""
            if failure_signature not in correction_history:
                correction_guidance = self._request_correction_guidance(
                    question=expected_question,
                    tool_status=tool_status,
                    tool_reason=tool_reason,
                    provided_question_id=provided_question_id,
                    predicted_answer=predicted_answer
                    if predicted_answer
                    else "missing",
                    submitted_summary_evidence_quote=submitted_summary_evidence_quote,
                    normalized_summary_evidence_quote=summary_evidence_quote,
                    attempts_used=current_attempts,
                    max_attempts=max_attempts,
                )
                correction_history.add(failure_signature)
            messages.append(
                {
                    "role": "user",
                    "content": self._build_retry_prompt(
                        question=expected_question,
                        tool_status=tool_status,
                        tool_reason=tool_reason,
                        attempts_remaining=attempts_remaining,
                        correction_guidance=correction_guidance,
                    ),
                }
            )

        context_data["current_question_index"] = current_index
        context_data["question_attempts"] = attempts
        context_data["correction_history"] = correction_history

        if current_index >= len(question_queue):
            return True, self._build_result_payload(context_data)

        return False, None
