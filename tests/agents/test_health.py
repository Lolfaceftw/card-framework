"""Tests for A2A service health checks."""

from __future__ import annotations

from collections import defaultdict
from http import HTTPStatus

import requests

from agents.health import AgentHealthChecker
import agents.health as health_module


class _FakeResponse:
    """Provide the minimal `requests.Response` surface used by health checks."""

    def __init__(
        self,
        *,
        status_code: int,
        url: str,
        reason: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.reason = reason or HTTPStatus(status_code).phrase

    def raise_for_status(self) -> None:
        """Raise the same error type that `requests` would for failing responses."""
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: {self.reason} for url: {self.url}"
            )


def test_check_once_prefers_agent_card_endpoint(monkeypatch) -> None:
    """Use the non-deprecated agent-card endpoint when it is available."""
    checker = AgentHealthChecker()
    requested_urls: list[str] = []

    def _fake_get(url: str, *, timeout: float) -> _FakeResponse:
        del timeout
        requested_urls.append(url)
        return _FakeResponse(status_code=200, url=url)

    monkeypatch.setattr(health_module.requests, "get", _fake_get)

    ok, error = checker.check_once("Summarizer", 9010, request_timeout_seconds=0.5)

    assert ok is True
    assert error is None
    assert requested_urls == [
        "http://127.0.0.1:9010/.well-known/agent-card.json",
    ]


def test_check_once_falls_back_to_legacy_agent_json_for_older_servers(
    monkeypatch,
) -> None:
    """Retry the legacy endpoint only when the new agent-card path is missing."""
    checker = AgentHealthChecker()
    requested_urls: list[str] = []

    def _fake_get(url: str, *, timeout: float) -> _FakeResponse:
        del timeout
        requested_urls.append(url)
        if url.endswith("/.well-known/agent-card.json"):
            return _FakeResponse(status_code=404, url=url)
        return _FakeResponse(status_code=200, url=url)

    monkeypatch.setattr(health_module.requests, "get", _fake_get)

    ok, error = checker.check_once("Summarizer", 9010, request_timeout_seconds=0.5)

    assert ok is True
    assert error is None
    assert requested_urls == [
        "http://127.0.0.1:9010/.well-known/agent-card.json",
        "http://127.0.0.1:9010/.well-known/agent.json",
    ]


def test_wait_for_many_retries_only_pending_servers(monkeypatch) -> None:
    """Poll only unhealthy servers until they become ready within one deadline."""
    checker = AgentHealthChecker()
    attempts: dict[str, int] = defaultdict(int)
    sleep_calls: list[float] = []

    def _fake_check_once(
        name: str,
        port: int,
        *,
        request_timeout_seconds: float = 1.0,
    ) -> tuple[bool, str | None]:
        del port, request_timeout_seconds
        attempts[name] += 1
        if name == "Summarizer":
            return attempts[name] >= 2, None if attempts[name] >= 2 else "warming up"
        return True, None

    monkeypatch.setattr(checker, "check_once", _fake_check_once)
    monkeypatch.setattr(health_module.time, "sleep", lambda seconds: sleep_calls.append(seconds))

    ready = checker.wait_for_many(
        [("Summarizer", 9010), ("Critic", 9011)],
        overall_timeout_seconds=5.0,
        poll_interval_seconds=0.2,
        request_timeout_seconds=0.5,
    )

    assert ready is True
    assert attempts["Critic"] == 1
    assert attempts["Summarizer"] == 2
    assert sleep_calls == [0.2]
