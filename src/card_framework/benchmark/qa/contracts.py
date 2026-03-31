"""Contracts and scoring helpers for QA-style benchmark evaluation."""

from __future__ import annotations

from collections import Counter
import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


QAQuestionDimension = Literal["factualness", "naturalness"]
QAAnswerLabel = Literal["yes", "no"]


def _normalize_question_intent(text: str) -> str:
    """Normalize question text into a simple dedupe key."""
    lowered = text.casefold()
    normalized = re.sub(r"[^\w]+", " ", lowered, flags=re.UNICODE)
    no_underscore = re.sub(r"_+", " ", normalized)
    return re.sub(r"\s+", " ", no_underscore).strip()


class GroundTruthQuestion(BaseModel):
    """One source-grounded QA item used in the benchmark."""

    question_id: str = Field(min_length=1)
    dimension: QAQuestionDimension
    question_text: str = Field(min_length=1)
    expected_answer: QAAnswerLabel
    source_evidence_ids: list[str] = Field(min_length=1)
    speaker_ids: list[str] = Field(default_factory=list)

    @field_validator("source_evidence_ids")
    @classmethod
    def validate_source_evidence_ids(cls, value: list[str]) -> list[str]:
        """Ensure evidence identifiers are non-empty, unique strings."""
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError(
                "source_evidence_ids must contain at least one non-empty id"
            )
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("source_evidence_ids must be unique per question")
        return cleaned

    @field_validator("speaker_ids")
    @classmethod
    def validate_speaker_ids(cls, value: list[str]) -> list[str]:
        """Normalize speaker IDs into unique, non-empty values."""
        cleaned = [item.strip() for item in value if item and item.strip()]
        return sorted(set(cleaned))

    @model_validator(mode="after")
    def validate_speaker_mentions(self) -> GroundTruthQuestion:
        """Normalize speaker_ids to include speaker mentions from question text."""
        mentions = sorted(set(re.findall(r"SPEAKER_\d+", self.question_text.upper())))
        if mentions:
            merged_speakers = sorted(set(self.speaker_ids) | set(mentions))
            self.speaker_ids = merged_speakers
        return self


class GroundTruthSet(BaseModel):
    """Validated ground-truth question set for one benchmark run."""

    questions: list[GroundTruthQuestion]

    @model_validator(mode="after")
    def validate_question_distribution(self) -> GroundTruthSet:
        """Enforce v1 contract: 100 total questions split 50/50 by dimension."""
        total_questions = len(self.questions)
        if total_questions != 100:
            raise ValueError(
                f"question set must contain exactly 100 items; got {total_questions}"
            )

        question_ids = [question.question_id for question in self.questions]
        if len(set(question_ids)) != len(question_ids):
            raise ValueError("question_ids must be unique")
        normalized_intents = [
            _normalize_question_intent(question.question_text)
            for question in self.questions
        ]
        if len(set(normalized_intents)) != len(normalized_intents):
            raise ValueError("question_text intent must be unique across the set")

        counts = Counter(question.dimension for question in self.questions)
        if counts.get("factualness", 0) != 50:
            raise ValueError(
                "question set must include exactly 50 factualness questions"
            )
        if counts.get("naturalness", 0) != 50:
            raise ValueError(
                "question set must include exactly 50 naturalness questions"
            )
        return self


class EvaluatorAnswerRecord(BaseModel):
    """One evaluator answer outcome for a single question."""

    question_id: str
    dimension: QAQuestionDimension
    predicted_answer: str
    expected_answer: QAAnswerLabel
    is_correct: bool
    tool_status: str
    reason: str = ""
    summary_grounding_pass: bool = True
    summary_evidence_quote: str = ""
    confidence: float | None = None

    @field_validator("confidence")
    @classmethod
    def validate_confidence(cls, value: float | None) -> float | None:
        """Ensure optional confidence is a ratio in [0.0, 1.0]."""
        if value is None:
            return None
        if value < 0.0 or value > 1.0:
            raise ValueError("confidence must be in [0.0, 1.0]")
        return value


class EvaluatorScore(BaseModel):
    """Score aggregate for evaluator outcomes."""

    factualness_correct: int
    naturalness_correct: int
    total_correct: int
    total_questions: int
    score_out_of_100: int
    summary_grounding_pass_count: int
    summary_grounding_pass_rate: float


class QABenchmarkReport(BaseModel):
    """Top-level report payload for one QA benchmark execution."""

    run_id: str
    status: str
    generated_at_utc: str
    git_commit: str
    git_branch: str
    summary_xml_path: str
    source_transcript_path: str
    provider_id: str
    creator_provider_id: str
    evaluator_provider_id: str
    score: EvaluatorScore
    answers: list[EvaluatorAnswerRecord]


def validate_answer_coverage(
    *,
    question_set: GroundTruthSet,
    answers: list[EvaluatorAnswerRecord],
) -> None:
    """Validate one-to-one answer coverage against generated question IDs."""
    expected_ids = [question.question_id for question in question_set.questions]
    answer_ids = [answer.question_id for answer in answers]

    if len(answer_ids) != len(expected_ids):
        raise ValueError(
            f"answer count mismatch: expected {len(expected_ids)}, got {len(answer_ids)}"
        )
    if len(set(answer_ids)) != len(answer_ids):
        raise ValueError("answer list contains duplicate question_id values")

    expected_set = set(expected_ids)
    answer_set = set(answer_ids)
    unknown = sorted(answer_set - expected_set)
    if unknown:
        raise ValueError(
            f"answer list contains unknown question_id values: {unknown[:5]}"
        )

    missing = sorted(expected_set - answer_set)
    if missing:
        raise ValueError(f"answer list missing question_id values: {missing[:5]}")


def build_score(answers: list[EvaluatorAnswerRecord]) -> EvaluatorScore:
    """Compute benchmark score from per-question answer outcomes."""
    factualness_correct = sum(
        1
        for answer in answers
        if answer.dimension == "factualness" and answer.is_correct
    )
    naturalness_correct = sum(
        1
        for answer in answers
        if answer.dimension == "naturalness" and answer.is_correct
    )
    total_correct = factualness_correct + naturalness_correct
    total_questions = len(answers)
    summary_grounding_pass_count = sum(
        1 for answer in answers if bool(answer.summary_grounding_pass)
    )
    summary_grounding_pass_rate = (
        summary_grounding_pass_count / float(total_questions)
        if total_questions
        else 0.0
    )
    return EvaluatorScore(
        factualness_correct=factualness_correct,
        naturalness_correct=naturalness_correct,
        total_correct=total_correct,
        total_questions=total_questions,
        score_out_of_100=total_correct,
        summary_grounding_pass_count=summary_grounding_pass_count,
        summary_grounding_pass_rate=summary_grounding_pass_rate,
    )
