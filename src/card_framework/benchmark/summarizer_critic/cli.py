"""Backward-compatible root CLI for the QMSum benchmark path."""

from card_framework.benchmark.summarizer_critic.qmsum.cli import (
    _child_environment,
    main,
)

__all__ = ["main", "_child_environment"]
