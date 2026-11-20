from __future__ import annotations

import pytest

from card_framework.benchmark.qa_contracts import (
    EvaluatorAnswerRecord,
    GroundTruthSet,
    _normalize_question_intent,
    build_score,
    validate_answer_coverage,
)


def _build_questions(
    *, factual_count: int, naturalness_count: int
) -> list[dict[str, object]]:
    questions: list[dict[str, object]] = []
    for index in range(1, factual_count + 1):
        questions.append(
            {
                "question_id": f"Q{index:03d}",
                "dimension": "factualness",
                "question_text": f"Factual question {index}?",
                "expected_answer": "yes",
                "source_evidence_ids": [f"E{index:04d}"],
                "speaker_ids": ["SPEAKER_00"],
            }
        )
    for offset in range(1, naturalness_count + 1):
        question_index = factual_count + offset
        questions.append(
            {
                "question_id": f"Q{question_index:03d}",
                "dimension": "naturalness",
                "question_text": f"Naturalness question {offset}?",
                "expected_answer": "no",
                "source_evidence_ids": [f"E{question_index:04d}"],
                "speaker_ids": ["SPEAKER_01"],
            }
        )
    return questions


def test_ground_truth_set_accepts_exact_contract_shape() -> None:
    question_set = GroundTruthSet.model_validate(
        {"questions": _build_questions(factual_count=50, naturalness_count=50)}
    )
    assert len(question_set.questions) == 100


def test_ground_truth_set_rejects_non_100_count() -> None:
    with pytest.raises(ValueError, match="exactly 100"):
        GroundTruthSet.model_validate(
            {"questions": _build_questions(factual_count=49, naturalness_count=50)}
        )


def test_ground_truth_set_rejects_wrong_dimension_split() -> None:
    with pytest.raises(ValueError, match="exactly 50"):
        GroundTruthSet.model_validate(
            {"questions": _build_questions(factual_count=60, naturalness_count=40)}
        )


def test_build_score_computes_dimension_and_total_counts() -> None:
    answers = [
        EvaluatorAnswerRecord(
            question_id="Q001",
            dimension="factualness",
            predicted_answer="yes",
            expected_answer="yes",
            is_correct=True,
            tool_status="accepted",
            reason="ok",
        ),
        EvaluatorAnswerRecord(
            question_id="Q051",
            dimension="naturalness",
            predicted_answer="no",
            expected_answer="yes",
            is_correct=False,
            tool_status="accepted",
            reason="wrong",
        ),
    ]
    score = build_score(answers)
    assert score.factualness_correct == 1
    assert score.naturalness_correct == 0
    assert score.total_correct == 1
    assert score.total_questions == 2
    assert score.score_out_of_100 == 1
    assert score.summary_grounding_pass_count == 2
    assert score.summary_grounding_pass_rate == 1.0


def test_ground_truth_set_rejects_duplicate_question_intents() -> None:
    questions = _build_questions(factual_count=50, naturalness_count=50)
    questions[1]["question_text"] = str(questions[0]["question_text"])
    with pytest.raises(ValueError, match="intent must be unique"):
        GroundTruthSet.model_validate({"questions": questions})


def test_ground_truth_question_normalizes_speaker_mentions() -> None:
    questions = _build_questions(factual_count=50, naturalness_count=50)
    questions[0]["question_text"] = "Does SPEAKER_99 mention this?"
    questions[0]["speaker_ids"] = ["SPEAKER_00"]
    question_set = GroundTruthSet.model_validate({"questions": questions})
    assert "SPEAKER_99" in question_set.questions[0].speaker_ids


def test_validate_answer_coverage_rejects_duplicate_question_ids() -> None:
    question_set = GroundTruthSet.model_validate(
        {"questions": _build_questions(factual_count=50, naturalness_count=50)}
    )
    answers = [
        EvaluatorAnswerRecord(
            question_id=question.question_id,
            dimension=question.dimension,
            predicted_answer=question.expected_answer,
            expected_answer=question.expected_answer,
            is_correct=True,
            tool_status="accepted",
            reason="ok",
        )
        for question in question_set.questions
    ]
    answers[1] = answers[0]
    with pytest.raises(ValueError, match="duplicate question_id"):
        validate_answer_coverage(question_set=question_set, answers=answers)


def test_normalize_question_intent_supports_non_latin_text() -> None:
    first = _normalize_question_intent("è¿™æ˜¯ä¸€ä¸ªæµ‹è¯•é—®é¢˜ï¼Ÿ")
    second = _normalize_question_intent("è¿™æ˜¯å¦ä¸€ä¸ªæµ‹è¯•é—®é¢˜ï¼Ÿ")
    assert first
    assert second
    assert first != second

