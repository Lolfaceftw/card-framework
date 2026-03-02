"""Application-level orchestration use cases."""

from __future__ import annotations

from typing import Any

__all__ = ["StageOrchestrator", "Transcript", "TranscriptSegment"]


def __getattr__(name: str) -> Any:
    """Resolve orchestration exports lazily to avoid import cycles."""
    if name == "StageOrchestrator":
        from orchestration.stage_orchestrator import StageOrchestrator

        return StageOrchestrator
    if name == "Transcript":
        from orchestration.transcript import Transcript

        return Transcript
    if name == "TranscriptSegment":
        from orchestration.transcript import TranscriptSegment

        return TranscriptSegment
    raise AttributeError(name)
