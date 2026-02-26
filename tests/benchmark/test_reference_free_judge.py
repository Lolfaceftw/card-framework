from __future__ import annotations

from benchmark.reference_free.contracts import JudgeRubric, JudgeRubricDimension
from benchmark.reference_free.judge_runner import LLMJudgeRunner


class FakeJudgeProvider:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        if not self._responses:
            raise RuntimeError("No fake responses left")
        return self._responses.pop(0)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
    ):
        raise NotImplementedError


def _rubric() -> JudgeRubric:
    return JudgeRubric(
        rubric_id="r1",
        version="1.0.0",
        pairwise_dimension="overall",
        dimensions=[
            JudgeRubricDimension("factuality", "f", 0.0, 1.0),
            JudgeRubricDimension("relevance", "r", 0.0, 1.0),
            JudgeRubricDimension("coherence", "c", 0.0, 1.0),
            JudgeRubricDimension("overall", "o", 0.0, 1.0),
        ],
    )


def test_judge_runner_direct_and_pairwise_order_swap() -> None:
    provider = FakeJudgeProvider(
        responses=[
            '{"factuality": 0.8, "relevance": 0.7, "coherence": 0.6, "overall": 0.65, "confidence": 0.8, "rationale": "ok"}',
            '{"factuality": 0.9, "relevance": 0.8, "coherence": 0.7, "overall": 0.75, "confidence": 0.8, "rationale": "ok"}',
            '{"winner": "A", "confidence": 0.9, "rationale": "A better"}',
            '{"winner": "B", "confidence": 0.9, "rationale": "B better"}',
        ]
    )

    runner = LLMJudgeRunner(judge_llm=provider, rubric=_rubric())
    result = runner.evaluate(
        source_text="source text",
        summary_text="summary text that is long enough for degradation checks",
        enable_order_swap=True,
        repeats=2,
    )

    assert result.status == "ok"
    assert result.scores is not None
    assert result.scores["overall"] == 0.7
    assert result.repeat_delta == 0.1
    assert result.pairwise_winner == "candidate"
    assert result.order_consistent is True


def test_judge_runner_handles_invalid_json() -> None:
    provider = FakeJudgeProvider(responses=["not json"])
    runner = LLMJudgeRunner(judge_llm=provider, rubric=_rubric())

    result = runner.evaluate(
        source_text="source text",
        summary_text="summary text",
        enable_order_swap=False,
        repeats=1,
    )

    assert result.status == "error"
    assert result.scores is None
    assert result.error_message is not None
