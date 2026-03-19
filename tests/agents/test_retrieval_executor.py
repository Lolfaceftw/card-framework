"""Integration-style coverage for the retrieval A2A executor."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from card_framework.agents.client import AgentClient
from card_framework.agents.health import AgentHealthChecker
from card_framework.agents.retrieval import InfoRetrievalExecutor
from card_framework.benchmark.summarizer_critic.qmsum.runner import (
    _run_server_in_thread,
    _stop_server,
)
from card_framework.cli.main import _build_a2a_app


class _SlowTranscriptIndex:
    """Block indexing until the test explicitly releases the request."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def index_segments(self, segments: list[dict[str, Any]]) -> int:
        self.started.set()
        if not self.release.wait(timeout=5.0):
            raise TimeoutError("Timed out waiting to release slow retrieval index.")
        return len(segments)

    def retrieve_mmr(
        self,
        query: str,
        top_k: int = 15,
        lambda_param: float = 0.7,
    ) -> list[dict[str, Any]]:
        del query, top_k, lambda_param
        return []

    def clear(self) -> None:
        return None


def test_retrieval_health_check_succeeds_while_index_task_is_running(
    free_tcp_port: int,
) -> None:
    """Keep the agent-card endpoint responsive during slow embedding work."""
    slow_index = _SlowTranscriptIndex()
    app = _build_a2a_app(
        name="InfoRetrieval",
        description="Test retrieval service",
        port=free_tcp_port,
        executor=InfoRetrievalExecutor(slow_index),
    )
    server = _run_server_in_thread("test-retrieval-a2a", app, free_tcp_port)
    checker = AgentHealthChecker(max_retries=5, base_delay=0.05)
    assert checker.check("InfoRetrieval", free_tcp_port) is True

    async def _exercise() -> tuple[bool, str | None, dict[str, Any]]:
        client = AgentClient()
        try:
            task = asyncio.create_task(
                client.send_task(
                    free_tcp_port,
                    {
                        "action": "index",
                        "segments": [{"speaker": "SPEAKER_00", "text": "hello"}],
                    },
                    timeout=5.0,
                    max_retries=1,
                )
            )
            started = await asyncio.to_thread(slow_index.started.wait, 2.0)
            assert started is True
            ok, error = await asyncio.to_thread(
                checker.check_once,
                "InfoRetrieval",
                free_tcp_port,
                request_timeout_seconds=0.2,
            )
            slow_index.release.set()
            result = json.loads(await task)
            return ok, error, result
        finally:
            await client.close()

    try:
        ok, error, result = asyncio.run(_exercise())
    finally:
        slow_index.release.set()
        _stop_server(server)

    assert ok is True
    assert error is None
    assert result == {"status": "ok", "count": 1}
