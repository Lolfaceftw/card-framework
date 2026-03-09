"""Application-level orchestration use cases."""

from __future__ import annotations

from typing import Any

__all__ = ["StageOrchestrator", "Transcript", "TranscriptSegment"]


def __getattr__(name: str) -> Any:
    """Resolve orchestration exports lazily to avoid import cycles."""
    if name == "StageOrchestrator":
        from card_framework.orchestration.stage_orchestrator import StageOrchestrator

        return StageOrchestrator
    if name == "Transcript":
        from card_framework.orchestration.transcript import Transcript

        return Transcript
    if name == "TranscriptSegment":
        from card_framework.orchestration.transcript import TranscriptSegment

        return TranscriptSegment
    raise AttributeError(name)

