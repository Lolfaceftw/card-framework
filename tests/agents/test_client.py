"""Exercise the A2A client against real localhost HTTP services."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from card_framework.agents.client import AgentClient
from tests.support.servers import run_starlette_server


async def _send_and_close(
    *,
    port: int,
    task_data: dict[str, Any],
    timeout: float,
    max_retries: int = 1,
) -> str:
    """Send one task request and close the pooled client in the same loop."""
    client = AgentClient()
    try:
        return await client.send_task(
            port=port,
            task_data=task_data,
            timeout=timeout,
            max_retries=max_retries,
        )
    finally:
        await client.close()


def test_send_task_reads_text_parts_from_live_a2a_response(
    free_tcp_port: int,
) -> None:
    """Read the actual JSON-RPC response shape returned by an A2A server."""
    posted_payloads: list[dict[str, Any]] = []

    async def _handle(request: Request) -> JSONResponse:
        posted_payloads.append(await request.json())
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "parts": [
                        {"kind": "text", "text": "first line"},
                        {"kind": "text", "text": "second line"},
                    ]
                },
            }
        )

    app = Starlette(routes=[Route("/", _handle, methods=["POST"])])

    with run_starlette_server(app, port=free_tcp_port):
        result = asyncio.run(
            _send_and_close(
                port=free_tcp_port,
                task_data={"x": 1},
                timeout=1.0,
            )
        )

    assert result == "first line\nsecond line"
    assert posted_payloads == [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": '{"x": 1}'}],
                    "messageId": posted_payloads[0]["params"]["message"]["messageId"],
                },
                "configuration": {"blocking": False},
            },
        }
    ]


def test_send_task_polls_non_blocking_task_until_completion(
    free_tcp_port: int,
) -> None:
    """Start non-blocking A2A work, then poll tasks/get until completion."""
    requests_seen: list[dict[str, Any]] = []
    poll_count = 0

    async def _handle(request: Request) -> JSONResponse:
        nonlocal poll_count
        payload = await request.json()
        requests_seen.append(payload)
        method = payload.get("method")
        if method == "message/send":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "id": "task_123",
                        "kind": "task",
                        "status": {"state": "working"},
                        "artifacts": [],
                    },
                }
            )
        if method == "tasks/get":
            poll_count += 1
            if poll_count == 1:
                return JSONResponse(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {
                            "id": "task_123",
                            "kind": "task",
                            "status": {"state": "working"},
                            "artifacts": [],
                        },
                    }
                )
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "id": "task_123",
                        "kind": "task",
                        "status": {"state": "completed"},
                        "artifacts": [
                            {
                                "parts": [
                                    {"kind": "text", "text": "done"},
                                ]
                            }
                        ],
                    },
                }
            )
        raise AssertionError(f"Unexpected method: {method}")

    app = Starlette(routes=[Route("/", _handle, methods=["POST"])])

    with run_starlette_server(app, port=free_tcp_port):
        result = asyncio.run(
            _send_and_close(
                port=free_tcp_port,
                task_data={"x": 1},
                timeout=1.0,
            )
        )

    assert result == "done"
    assert requests_seen[0]["method"] == "message/send"
    assert requests_seen[1]["method"] == "tasks/get"


def test_send_task_prefers_completed_status_message_over_task_history(
    free_tcp_port: int,
) -> None:
    """Return the final task message without appending request history text."""
    requests_seen: list[dict[str, Any]] = []

    async def _handle(request: Request) -> JSONResponse:
        payload = await request.json()
        requests_seen.append(payload)
        method = payload.get("method")
        if method == "message/send":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "id": "task_123",
                        "kind": "task",
                        "status": {"state": "working"},
                        "history": [
                            {
                                "parts": [
                                    {
                                        "kind": "text",
                                        "text": '{"draft":"source transcript"}',
                                    }
                                ]
                            }
                        ],
                    },
                }
            )
        if method == "tasks/get":
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "id": "task_123",
                        "kind": "task",
                        "status": {
                            "state": "completed",
                            "message": {
                                "parts": [{"kind": "text", "text": '{"status":"ok"}'}]
                            },
                        },
                        "history": [
                            {
                                "parts": [
                                    {
                                        "kind": "text",
                                        "text": '{"draft":"source transcript"}',
                                    }
                                ]
                            }
                        ],
                    },
                }
            )
        raise AssertionError(f"Unexpected method: {method}")

    app = Starlette(routes=[Route("/", _handle, methods=["POST"])])

    with run_starlette_server(app, port=free_tcp_port):
        result = asyncio.run(
            _send_and_close(
                port=free_tcp_port,
                task_data={"x": 1},
                timeout=1.0,
            )
        )

    assert result == '{"status":"ok"}'
    assert requests_seen[0]["method"] == "message/send"
    assert requests_seen[1]["method"] == "tasks/get"


def test_send_task_read_timeout_warns_against_retries(free_tcp_port: int) -> None:
    """Surface a read timeout from a slow localhost agent response."""

    async def _slow_handle(_request: Request) -> JSONResponse:
        await asyncio.sleep(0.2)
        return JSONResponse(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"parts": [{"kind": "text", "text": "late"}]},
            }
        )

    app = Starlette(routes=[Route("/", _slow_handle, methods=["POST"])])

    with run_starlette_server(app, port=free_tcp_port):
        with pytest.raises(RuntimeError) as excinfo:
            asyncio.run(
                _send_and_close(
                    port=free_tcp_port,
                    task_data={"x": 1},
                    timeout=0.05,
                )
            )

    assert "Per-request timeout hit after 0.1s" in str(excinfo.value)
    assert "ReadTimeout" in str(excinfo.value)


def test_send_task_connection_error_includes_exception_type(
    free_tcp_port: int,
) -> None:
    """Report the underlying connection error type when no agent is listening."""
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(
            _send_and_close(
                port=free_tcp_port,
                task_data={"x": 1},
                timeout=0.1,
            )
        )

    assert excinfo.value.__cause__ is not None
    assert type(excinfo.value.__cause__).__name__ in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.RequestError)


def test_send_task_rejects_invalid_retry_count() -> None:
    """Reject invalid retry counts before opening a network connection."""
    client = AgentClient()
    with pytest.raises(ValueError, match="max_retries must be >= 1"):
        asyncio.run(
            client.send_task(
                port=9010,
                task_data={"x": 1},
                timeout=1.0,
                max_retries=0,
            )
        )


def test_agent_client_defaults_preserve_shared_transport_behavior() -> None:
    """Keep the repo-wide AgentClient defaults stable outside SPoRC."""
    client = AgentClient()

    timeout = client._build_http_timeout(12.0)

    assert client._trust_env is True
    assert client._pool_timeout_seconds is None
    assert timeout.connect == 5.0
    assert timeout.pool == 12.0


def test_agent_client_allows_explicit_sporc_pool_timeout_override() -> None:
    """Let SPoRC opt into a tighter connection-pool timeout explicitly."""
    client = AgentClient(pool_timeout_seconds=5.0, trust_env=False)

    timeout = client._build_http_timeout(12.0)

    assert client._trust_env is False
    assert client._pool_timeout_seconds == 5.0
    assert timeout.connect == 5.0
    assert timeout.pool == 5.0


def test_agent_client_honors_explicit_long_request_timeout() -> None:
    """Allow benchmark timeout flags above the default request timeout."""
    client = AgentClient()

    timeout = client._build_http_timeout(300.0)

    assert timeout.connect == 5.0
    assert timeout.pool == 300.0
