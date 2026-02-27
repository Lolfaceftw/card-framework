"""Tests for AgentClient transport and retry behavior."""

import asyncio

import httpx
import pytest

from agents.client import AgentClient


class _RaisingClient:
    """Fake async HTTP client that raises a configured exception for every post."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    async def post(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        self.calls += 1
        raise self.exc


def test_send_task_read_timeout_does_not_retry(monkeypatch) -> None:
    request = httpx.Request("POST", "http://127.0.0.1:9010")
    fake_client = _RaisingClient(httpx.ReadTimeout("timed out", request=request))
    client = AgentClient()

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client, "_get_client", lambda: fake_client)
    monkeypatch.setattr("agents.client.asyncio.sleep", _no_sleep)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(client.send_task(port=9010, task_data={"x": 1}, timeout=1.0))

    assert fake_client.calls == 1
    assert "timed out after 1.0s" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.ReadTimeout)


def test_send_task_error_includes_exception_type_for_empty_message(monkeypatch) -> None:
    request = httpx.Request("POST", "http://127.0.0.1:9010")
    fake_client = _RaisingClient(httpx.ConnectError("", request=request))
    client = AgentClient()

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(client, "_get_client", lambda: fake_client)
    monkeypatch.setattr("agents.client.asyncio.sleep", _no_sleep)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(
            client.send_task(
                port=9010,
                task_data={"x": 1},
                timeout=1.0,
                max_retries=2,
            )
        )

    assert fake_client.calls == 2
    assert "ConnectError" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, httpx.ConnectError)


def test_send_task_rejects_invalid_retry_count() -> None:
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
