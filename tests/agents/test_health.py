"""Tests for A2A service health checks."""

from __future__ import annotations

from collections import defaultdict

from agents.health import AgentHealthChecker
import agents.health as health_module


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
