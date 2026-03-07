"""Unit tests for summarizer tool-registry construction."""

from __future__ import annotations

import asyncio
from typing import Any

from agents.message_registry import MessageRegistry
from agents.tool_handlers import QueryTranscriptHandler, build_revise_tools


class RecordingAgentClient:
    """Capture retrieval calls made by query tool handlers in tests."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, dict[str, Any] | str | object, float]] = []

    async def send_task(
        self,
        port: int,
        task_data: dict[str, Any] | str | object,
        timeout: float = 120.0,
        max_retries: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        del max_retries, metadata
        self.calls.append((port, task_data, timeout))
        return "{}"


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


def test_add_speaker_message_missing_speaker_id_returns_error() -> None:
    """Reject malformed add_speaker_message calls instead of raising KeyError."""
    registry = MessageRegistry()
    tool_registry = build_revise_tools(
        registry=registry,
        retrieval_port=9012,
        calibration=None,
        target_seconds=60,
        duration_tolerance_ratio=0.05,
        is_embedding_enabled=False,
    )

    result = asyncio.run(tool_registry.dispatch("add_speaker_message", {}))

    assert result is not None
    assert result["error_code"] == "missing_speaker_id_argument"
    assert len(registry) == 0


def test_query_transcript_missing_query_returns_error_without_dispatch() -> None:
    """Reject empty query tool arguments before the retrieval client is called."""
    agent_client = RecordingAgentClient()
    handler = QueryTranscriptHandler(retrieval_port=9012, agent_client=agent_client)

    result = asyncio.run(handler.execute({}))

    assert result["error_code"] == "missing_query_argument"
    assert agent_client.calls == []
