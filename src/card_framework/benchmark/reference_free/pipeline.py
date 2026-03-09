"""End-to-end reference-free evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import re

from card_framework.benchmark.reference_free.alignscore_runner import AlignScoreRunner
from card_framework.benchmark.reference_free.contracts import ReferenceFreeEvaluation
from card_framework.benchmark.reference_free.judge_runner import LLMJudgeRunner


@dataclass(slots=True, frozen=True)
class ReferenceFreePipelineConfig:
    """Pipeline controls for reference-free metrics.

    Args:
        enable_order_swap: Run pairwise order-swap diagnostics when True.
        judge_repeats: Number of direct judge passes.
    """

    enable_order_swap: bool = True
    judge_repeats: int = 1


class ReferenceFreePipeline:
    """Compute alignscore and LLM-judge metrics for one sample."""

    def __init__(
        self,
        *,
        alignscore_runner: AlignScoreRunner,
        judge_runner: LLMJudgeRunner,
        config: ReferenceFreePipelineConfig | None = None,
    ) -> None:
        self._alignscore_runner = alignscore_runner
        self._judge_runner = judge_runner
        self._config = config or ReferenceFreePipelineConfig()

    @staticmethod
    def _strip_xml_tags(text: str) -> str:
        """Remove XML-like tags before judge evaluation."""
        return re.sub(r"<[^>]+>", "", text)

    def evaluate_sample(self, source_text: str, summary_text: str) -> ReferenceFreeEvaluation:
        """Evaluate one sample with all configured reference-free metrics.

        Args:
            source_text: Formatted transcript source text.
            summary_text: Candidate summary text (XML or plain).

        Returns:
            Combined reference-free evaluation payload.
        """
        clean_summary = self._strip_xml_tags(summary_text).strip()

        align_result = self._alignscore_runner.score(source_text, clean_summary)

        judge_result = self._judge_runner.evaluate(
            source_text=source_text,
            summary_text=clean_summary,
            enable_order_swap=self._config.enable_order_swap,
            repeats=self._config.judge_repeats,
        )

        errors: list[str] = []
        if align_result.status != "ok":
            errors.append(f"alignscore: {align_result.error_message}")
        if judge_result.status != "ok":
            errors.append(f"judge: {judge_result.error_message}")

        if not errors:
            status = "ok"
            error_message = None
        elif align_result.status == "ok" or judge_result.status == "ok":
            status = "partial"
            error_message = " | ".join(error for error in errors if error)
        else:
            status = "error"
            error_message = " | ".join(error for error in errors if error)

        return ReferenceFreeEvaluation(
            status=status,
            alignscore=align_result.score,
            alignscore_backend=align_result.backend,
            judge_scores=judge_result.scores,
            judge_pairwise_winner=judge_result.pairwise_winner,
            judge_order_consistent=judge_result.order_consistent,
            judge_repeat_delta=judge_result.repeat_delta,
            error_message=error_message,
        )

