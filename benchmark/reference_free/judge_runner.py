"""LLM-as-judge runner for reference-free summary evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from statistics import mean
from typing import Any, Protocol

from benchmark.reference_free.contracts import (
    JudgeEvaluation,
    JudgeRubric,
    clamp_score,
    normalize_numeric,
)


class JudgeLLMProtocol(Protocol):
    """Protocol for judge-capable LLM providers."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text output for the provided prompts."""


@dataclass(slots=True, frozen=True)
class LLMJudgeRunnerConfig:
    """Configuration for rubric-based LLM judge scoring.

    Args:
        source_char_limit: Max characters of source transcript passed to judge.
        max_tokens: Max output tokens requested from judge model.
    """

    source_char_limit: int = 30_000
    max_tokens: int = 600


class LLMJudgeRunner:
    """Run direct and pairwise LLM-as-judge evaluations.

    The runner expects strict JSON output and normalizes parse errors into
    structured failure payloads instead of raising.
    """

    def __init__(
        self,
        *,
        judge_llm: JudgeLLMProtocol,
        rubric: JudgeRubric,
        config: LLMJudgeRunnerConfig | None = None,
    ) -> None:
        self._judge_llm = judge_llm
        self._rubric = rubric
        self._config = config or LLMJudgeRunnerConfig()

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        """Extract first JSON object from model response text."""
        stripped = text.strip()

        fence_match = re.search(r"```json\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if fence_match:
            stripped = fence_match.group(1)

        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None
        return payload

    @staticmethod
    def _degrade_summary(summary_text: str) -> str:
        """Create a deterministic weaker baseline for pairwise bias checks."""
        words = summary_text.split()
        if len(words) <= 40:
            return " ".join(words[: max(5, len(words) // 2)])
        return " ".join(words[: max(20, len(words) // 2)])

    def _build_direct_prompts(self, source_text: str, summary_text: str) -> tuple[str, str]:
        """Build system and user prompts for direct rubric scoring."""
        dimensions_block = "\n".join(
            (
                f"- {dimension.name}: {dimension.description} "
                f"(range {dimension.min_score} to {dimension.max_score})"
            )
            for dimension in self._rubric.dimensions
        )

        system_prompt = (
            "You are a strict summarization evaluator. Score the candidate summary "
            "against the source transcript using the provided rubric. "
            "Return JSON only with keys for each rubric dimension, plus 'confidence' "
            "and 'rationale'."
        )

        truncated_source = source_text[: self._config.source_char_limit]

        user_prompt = (
            f"Rubric ID: {self._rubric.rubric_id} v{self._rubric.version}\n"
            f"Dimensions:\n{dimensions_block}\n\n"
            "Output contract JSON:\n"
            "{\n"
            "  \"factuality\": <float>,\n"
            "  \"relevance\": <float>,\n"
            "  \"coherence\": <float>,\n"
            "  \"overall\": <float>,\n"
            "  \"confidence\": <float>,\n"
            "  \"rationale\": \"short explanation\"\n"
            "}\n\n"
            "Source transcript:\n"
            f"{truncated_source}\n\n"
            "Candidate summary:\n"
            f"{summary_text}"
        )

        return system_prompt, user_prompt

    def _build_pairwise_prompts(
        self,
        source_text: str,
        summary_a: str,
        summary_b: str,
    ) -> tuple[str, str]:
        """Build pairwise prompt for order-bias diagnostics."""
        system_prompt = (
            "You are a strict summarization evaluator. Compare summaries A and B. "
            "Return JSON only with winner in {A,B,tie}, confidence, rationale."
        )

        truncated_source = source_text[: self._config.source_char_limit]

        user_prompt = (
            f"Pairwise dimension: {self._rubric.pairwise_dimension}\n"
            "Output contract JSON:\n"
            "{\n"
            "  \"winner\": \"A\" | \"B\" | \"tie\",\n"
            "  \"confidence\": <float>,\n"
            "  \"rationale\": \"short explanation\"\n"
            "}\n\n"
            "Source transcript:\n"
            f"{truncated_source}\n\n"
            "Summary A:\n"
            f"{summary_a}\n\n"
            "Summary B:\n"
            f"{summary_b}"
        )

        return system_prompt, user_prompt

    def _judge_generate(self, system_prompt: str, user_prompt: str) -> str:
        """Call judge provider and return raw text output."""
        return self._judge_llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self._config.max_tokens,
        )

    def _parse_direct_scores(self, payload: dict[str, Any]) -> dict[str, float] | None:
        """Parse and clamp rubric scores from judge payload."""
        dimension_map = {dimension.name: dimension for dimension in self._rubric.dimensions}
        scores: dict[str, float] = {}

        for dimension_name, dimension in dimension_map.items():
            raw = normalize_numeric(payload.get(dimension_name))
            if raw is None:
                return None
            scores[dimension_name] = clamp_score(
                raw,
                min_score=dimension.min_score,
                max_score=dimension.max_score,
            )

        return scores

    def _score_direct_once(
        self,
        source_text: str,
        summary_text: str,
    ) -> tuple[dict[str, float] | None, str | None]:
        """Run one direct rubric scoring pass."""
        system_prompt, user_prompt = self._build_direct_prompts(source_text, summary_text)
        response = self._judge_generate(system_prompt, user_prompt)
        parsed = self._extract_json_object(response)
        if parsed is None:
            return None, "Judge direct response did not contain valid JSON"

        scores = self._parse_direct_scores(parsed)
        if scores is None:
            return None, "Judge direct response missing valid numeric rubric fields"

        return scores, None

    def _score_pairwise_once(
        self,
        source_text: str,
        summary_a: str,
        summary_b: str,
    ) -> tuple[str | None, str | None]:
        """Run one pairwise judge comparison and parse winner."""
        system_prompt, user_prompt = self._build_pairwise_prompts(
            source_text,
            summary_a,
            summary_b,
        )
        response = self._judge_generate(system_prompt, user_prompt)
        parsed = self._extract_json_object(response)
        if parsed is None:
            return None, "Judge pairwise response did not contain valid JSON"

        winner_raw = str(parsed.get("winner", "")).strip().lower()
        if winner_raw in {"a", "summary_a"}:
            return "A", None
        if winner_raw in {"b", "summary_b"}:
            return "B", None
        if winner_raw in {"tie", "equal"}:
            return "tie", None

        return None, "Judge pairwise winner must be one of A/B/tie"

    def evaluate(
        self,
        *,
        source_text: str,
        summary_text: str,
        enable_order_swap: bool,
        repeats: int,
    ) -> JudgeEvaluation:
        """Evaluate one summary with direct and pairwise judge metrics.

        Args:
            source_text: Source transcript text.
            summary_text: Candidate summary.
            enable_order_swap: Whether to run order-swapped pairwise checks.
            repeats: Number of direct scoring passes.

        Returns:
            Structured :class:`JudgeEvaluation` result.
        """
        if not source_text.strip() or not summary_text.strip():
            return JudgeEvaluation(
                status="error",
                scores=None,
                pairwise_winner=None,
                order_consistent=None,
                repeat_delta=None,
                error_message="Source and summary must both be non-empty",
            )

        repeats = max(1, repeats)
        per_pass_scores: list[dict[str, float]] = []

        for _ in range(repeats):
            scores, error = self._score_direct_once(source_text, summary_text)
            if scores is None:
                return JudgeEvaluation(
                    status="error",
                    scores=None,
                    pairwise_winner=None,
                    order_consistent=None,
                    repeat_delta=None,
                    error_message=error,
                )
            per_pass_scores.append(scores)

        averaged_scores: dict[str, float] = {}
        for dimension in self._rubric.dimensions:
            values = [scores[dimension.name] for scores in per_pass_scores]
            averaged_scores[dimension.name] = round(mean(values), 4)

        repeat_delta: float | None = None
        overall_key = "overall"
        if repeats >= 2 and overall_key in per_pass_scores[0] and overall_key in per_pass_scores[-1]:
            repeat_delta = round(
                abs(per_pass_scores[0][overall_key] - per_pass_scores[-1][overall_key]),
                4,
            )

        pairwise_winner: str | None = None
        order_consistent: bool | None = None

        if enable_order_swap:
            baseline = self._degrade_summary(summary_text)

            winner_ab, error_ab = self._score_pairwise_once(
                source_text,
                summary_text,
                baseline,
            )
            if winner_ab is None:
                return JudgeEvaluation(
                    status="error",
                    scores=averaged_scores,
                    pairwise_winner=None,
                    order_consistent=None,
                    repeat_delta=repeat_delta,
                    error_message=error_ab,
                )

            winner_ba, error_ba = self._score_pairwise_once(
                source_text,
                baseline,
                summary_text,
            )
            if winner_ba is None:
                return JudgeEvaluation(
                    status="error",
                    scores=averaged_scores,
                    pairwise_winner=None,
                    order_consistent=None,
                    repeat_delta=repeat_delta,
                    error_message=error_ba,
                )

            if winner_ab == "A":
                pairwise_winner = "candidate"
            elif winner_ab == "B":
                pairwise_winner = "baseline"
            else:
                pairwise_winner = "tie"

            order_consistent = (
                (winner_ab == "A" and winner_ba == "B")
                or (winner_ab == "B" and winner_ba == "A")
                or (winner_ab == "tie" and winner_ba == "tie")
            )

        return JudgeEvaluation(
            status="ok",
            scores=averaged_scores,
            pairwise_winner=pairwise_winner,
            order_consistent=order_consistent,
            repeat_delta=repeat_delta,
        )
