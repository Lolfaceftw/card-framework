"""Exercise agent health checks against real localhost HTTP endpoints."""

from __future__ import annotations

from collections.abc import Callable

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route

from card_framework.agents.health import AgentHealthChecker
from tests.support.servers import run_starlette_server


def test_check_once_prefers_agent_card_endpoint(free_tcp_port: int) -> None:
    """Use the current agent-card endpoint when it is available."""
    checker = AgentHealthChecker()
    requested_paths: list[str] = []

    async def _handle(request: Request):
        requested_paths.append(request.url.path)
        return JSONResponse({"name": "Summarizer"})

    app = Starlette(
        routes=[
            Route("/.well-known/agent-card.json", _handle, methods=["GET"]),
            Route("/.well-known/agent.json", _handle, methods=["GET"]),
        ]
    )

    with run_starlette_server(app, port=free_tcp_port):
        ok, error = checker.check_once(
            "Summarizer",
            free_tcp_port,
            request_timeout_seconds=0.5,
        )

    assert ok is True
    assert error is None
    assert requested_paths == ["/.well-known/agent-card.json"]


def test_check_once_falls_back_to_legacy_agent_json_for_older_servers(
    free_tcp_port: int,
) -> None:
    """Retry the legacy endpoint only when the newer path is unavailable."""
    checker = AgentHealthChecker()
    requested_paths: list[str] = []

    async def _handle(request: Request):
        requested_paths.append(request.url.path)
        if request.url.path.endswith("agent-card.json"):
            return PlainTextResponse("missing", status_code=404)
        return JSONResponse({"name": "Summarizer"})

    app = Starlette(
        routes=[
            Route("/.well-known/agent-card.json", _handle, methods=["GET"]),
            Route("/.well-known/agent.json", _handle, methods=["GET"]),
        ]
    )

    with run_starlette_server(app, port=free_tcp_port):
        ok, error = checker.check_once(
            "Summarizer",
            free_tcp_port,
            request_timeout_seconds=0.5,
        )

    assert ok is True
    assert error is None
    assert requested_paths == [
        "/.well-known/agent-card.json",
        "/.well-known/agent.json",
    ]


def test_wait_for_many_retries_only_pending_servers(
    free_tcp_port_factory: Callable[[], int],
) -> None:
    """Repoll only the unhealthy server until it comes online."""
    checker = AgentHealthChecker()
    summarizer_port = free_tcp_port_factory()
    critic_port = free_tcp_port_factory()
    summarizer_requests: list[str] = []
    critic_requests: list[str] = []

    async def _summarizer(request: Request):
        summarizer_requests.append(request.url.path)
        if len(summarizer_requests) == 1:
            return PlainTextResponse("warming up", status_code=503)
        return JSONResponse({"name": "Summarizer"})

    async def _critic(request: Request):
        critic_requests.append(request.url.path)
        return JSONResponse({"name": "Critic"})

    summarizer_app = Starlette(
        routes=[Route("/.well-known/agent-card.json", _summarizer, methods=["GET"])]
    )
    critic_app = Starlette(
        routes=[Route("/.well-known/agent-card.json", _critic, methods=["GET"])]
    )

    with run_starlette_server(summarizer_app, port=summarizer_port):
        with run_starlette_server(critic_app, port=critic_port):
            ready = checker.wait_for_many(
                [("Summarizer", summarizer_port), ("Critic", critic_port)],
                overall_timeout_seconds=1.0,
                poll_interval_seconds=0.05,
                request_timeout_seconds=0.2,
            )

    assert ready is True
    assert critic_requests == ["/.well-known/agent-card.json"]
    assert summarizer_requests == [
        "/.well-known/agent-card.json",
        "/.well-known/agent-card.json",
    ]
