"""Plugin interface for future meeting-specific reference-free evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from benchmark.reference_free.contracts import ReferenceFreeEvaluation


class ReferenceFreePlugin(ABC):
    """Abstract plugin interface for optional reference-free evaluators."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable plugin name used in config and reporting."""

    @abstractmethod
    def evaluate(self, source_text: str, summary_text: str) -> ReferenceFreeEvaluation:
        """Evaluate one source-summary pair."""


class NotImplementedPlugin(ReferenceFreePlugin):
    """Placeholder plugin for phase-2 evaluator families."""

    def __init__(self, plugin_name: str) -> None:
        self._name = plugin_name

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, source_text: str, summary_text: str) -> ReferenceFreeEvaluation:
        return ReferenceFreeEvaluation(
            status="not_implemented",
            alignscore=None,
            alignscore_backend=None,
            judge_scores=None,
            judge_pairwise_winner=None,
            judge_order_consistent=None,
            judge_repeat_delta=None,
            error_message=f"Plugin '{self._name}' is reserved for phase-2 implementation",
        )


def default_plugin_registry() -> dict[str, ReferenceFreePlugin]:
    """Return built-in plugin registry with phase-2 placeholders."""
    return {
        "cream": NotImplementedPlugin("cream"),
        "mesa": NotImplementedPlugin("mesa"),
    }
