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
                }
            },
        }
    ]


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

    assert "timed out after 0.05s" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)


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
