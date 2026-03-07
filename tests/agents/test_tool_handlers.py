"""Unit tests for summarizer tool-registry construction."""

from __future__ import annotations

from agents.message_registry import MessageRegistry
from agents.tool_handlers import build_revise_tools


def test_build_revise_tools_includes_add_speaker_message() -> None:
    """Expose add_speaker_message in revise mode for under-duration recovery."""
    tool_registry = build_revise_tools(
        registry=MessageRegistry(),
        retrieval_port=9012,
        calibration=None,
        target_seconds=60,
        duration_tolerance_ratio=0.05,
        is_embedding_enabled=False,
    )

    assert tool_registry.get("add_speaker_message") is not None
