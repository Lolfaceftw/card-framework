import asyncio
from collections.abc import Mapping
import json
import time
import threading
import uuid
from typing import Any, Protocol

import httpx

from card_framework.shared.events import EventBus, get_event_bus


class AgentTaskClient(Protocol):
    """Protocol for components that can send tasks to A2A services."""

    async def send_task(
        self,
        port: int,
        task_data: dict | str | object,
        timeout: float | None = 120.0,
        max_retries: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send one task request and return serialized response payload."""
        ...


class AgentClient:
    """
    Robust HTTP client for communicating with A2A agents via JSON-RPC.
    Features connection pooling and exponential backoff for transient errors.
    """

    _TERMINAL_TASK_STATES = frozenset(
        {
            "completed",
            "canceled",
            "failed",
            "rejected",
            "input-required",
            "auth-required",
        }
    )
    _SUCCESS_TASK_STATES = frozenset({"completed"})
    _POLL_INTERVAL_SECONDS = 2.0
    _MAX_HTTP_REQUEST_TIMEOUT_SECONDS = 30.0

    def __init__(
        self,
        event_bus: EventBus | None = None,
        *,
        limits: httpx.Limits | None = None,
        pool_timeout_seconds: float | None = None,
        trust_env: bool = True,
    ) -> None:
        """Initialize pooled transport and event bus dependency."""
        self.limits = limits or httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
        )
        self._clients: dict[asyncio.AbstractEventLoop | None, httpx.AsyncClient] = {}
        self._lock = threading.Lock()
        self._event_bus = event_bus if event_bus is not None else get_event_bus()
        self._pool_timeout_seconds = (
            max(0.1, float(pool_timeout_seconds))
            if pool_timeout_seconds is not None
            else None
        )
        self._trust_env = trust_env

    def _get_client(self) -> httpx.AsyncClient:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        with self._lock:
            # Prune closed loops to prevent memory leaks if many run_until_complete cycles happen
            closed_loops = [
                loop_obj
                for loop_obj in self._clients
                if loop_obj is not None and loop_obj.is_closed()
            ]
            for loop_obj in closed_loops:
                self._clients.pop(loop_obj, None)

            client = self._clients.get(loop)
            if client is None or client.is_closed:
                client = httpx.AsyncClient(
                    limits=self.limits,
                    trust_env=self._trust_env,
                )
                self._clients[loop] = client
            return client

    async def close(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        with self._lock:
            client = self._clients.pop(loop, None)

        if client is not None:
            await client.aclose()

    @classmethod
    def _resolve_http_timeout_seconds(cls, timeout: float | None) -> float:
        """Resolve a per-request timeout budget in seconds."""
        if timeout is None:
            return cls._MAX_HTTP_REQUEST_TIMEOUT_SECONDS
        return max(0.1, min(float(timeout), cls._MAX_HTTP_REQUEST_TIMEOUT_SECONDS))

    def _build_http_timeout(self, timeout: float | None) -> httpx.Timeout:
        """Build an explicit timeout with a bounded connection-pool wait."""
        resolved_seconds = self._resolve_http_timeout_seconds(timeout)
        pool_timeout_seconds = (
            min(self._pool_timeout_seconds, resolved_seconds)
            if self._pool_timeout_seconds is not None
            else resolved_seconds
        )
        return httpx.Timeout(
            timeout=resolved_seconds,
            connect=min(5.0, resolved_seconds),
            pool=pool_timeout_seconds,
        )

    @staticmethod
    def _build_message_send_payload(task_text: str) -> dict[str, Any]:
        """Build a non-blocking A2A message/send payload."""
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": task_text}],
                    "messageId": str(uuid.uuid4()),
                },
                "configuration": {"blocking": False},
            },
        }

    @staticmethod
    def _build_get_task_payload(task_id: str) -> dict[str, Any]:
        """Build a JSON-RPC payload for tasks/get."""
        return {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tasks/get",
            "params": {"id": task_id},
        }

    @classmethod
    def _extract_texts_from_result(cls, result: Any) -> list[str]:
        """Extract all text parts from a message or task-like payload."""
        if not isinstance(result, Mapping):
            return []

        texts: list[str] = []
        artifacts = result.get("artifacts")
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    continue
                parts = artifact.get("parts")
                if not isinstance(parts, list):
                    continue
                texts.extend(
                    str(part.get("text", ""))
                    for part in parts
                    if isinstance(part, Mapping)
                    and part.get("kind") == "text"
                    and str(part.get("text", "")).strip()
                )

        parts = result.get("parts")
        if isinstance(parts, list):
            texts.extend(
                str(part.get("text", ""))
                for part in parts
                if isinstance(part, Mapping)
                and part.get("kind") == "text"
                and str(part.get("text", "")).strip()
            )

        nested_result = result.get("result")
        if isinstance(nested_result, Mapping):
            texts.extend(cls._extract_texts_from_result(nested_result))

        status = result.get("status")
        if isinstance(status, Mapping):
            status_message = status.get("message")
            if isinstance(status_message, Mapping):
                texts.extend(cls._extract_texts_from_result(status_message))

        history = result.get("history")
        if isinstance(history, list):
            for item in history:
                if isinstance(item, Mapping):
                    texts.extend(cls._extract_texts_from_result(item))

        return texts

    @classmethod
    def _extract_text_result(cls, result: Any) -> str | None:
        """Extract a newline-joined text payload when available."""
        if not isinstance(result, Mapping):
            return None

        primary_texts: list[str] = []

        artifacts = result.get("artifacts")
        if isinstance(artifacts, list):
            for artifact in artifacts:
                if not isinstance(artifact, Mapping):
                    continue
                parts = artifact.get("parts")
                if not isinstance(parts, list):
                    continue
                primary_texts.extend(
                    str(part.get("text", ""))
                    for part in parts
                    if isinstance(part, Mapping)
                    and part.get("kind") == "text"
                    and str(part.get("text", "")).strip()
                )

        parts = result.get("parts")
        if isinstance(parts, list):
            primary_texts.extend(
                str(part.get("text", ""))
                for part in parts
                if isinstance(part, Mapping)
                and part.get("kind") == "text"
                and str(part.get("text", "")).strip()
            )

        nested_result = result.get("result")
        if isinstance(nested_result, Mapping):
            nested_text = cls._extract_text_result(nested_result)
            if nested_text:
                primary_texts.append(nested_text)

        status = result.get("status")
        if isinstance(status, Mapping):
            status_message = status.get("message")
            if isinstance(status_message, Mapping):
                primary_texts.extend(cls._extract_texts_from_result(status_message))

        if primary_texts:
            return "\n".join(primary_texts)

        history = result.get("history")
        if isinstance(history, list):
            history_texts: list[str] = []
            for item in history:
                if isinstance(item, Mapping):
                    history_texts.extend(cls._extract_texts_from_result(item))
            if history_texts:
                return "\n".join(history_texts)

        return None

    @staticmethod
    def _extract_task_id(result: Any) -> str | None:
        """Extract a task identifier from a task-like payload."""
        if not isinstance(result, Mapping):
            return None
        raw_task_id = result.get("id")
        status = result.get("status")
        if isinstance(raw_task_id, str) and isinstance(status, Mapping):
            return raw_task_id
        return None

    @staticmethod
    def _extract_task_state(result: Any) -> str | None:
        """Extract the task status state from a task-like payload."""
        if not isinstance(result, Mapping):
            return None
        status = result.get("status")
        if not isinstance(status, Mapping):
            return None
        raw_state = status.get("state")
        return str(raw_state) if isinstance(raw_state, str) else None

    async def _post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        timeout: httpx.Timeout,
    ) -> dict[str, Any]:
        """POST one JSON-RPC payload and return the decoded response."""
        client = self._get_client()
        response = await client.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    async def _poll_task_until_terminal(
        self,
        *,
        url: str,
        task_id: str,
        timeout: httpx.Timeout,
        timeout_seconds: float,
        max_retries: int,
        call_id: str,
        port: int,
        metadata: dict[str, Any],
    ) -> Any:
        """Poll tasks/get until the task reaches a terminal state."""
        consecutive_failures = 0
        poll_index = 0
        last_state: str | None = None

        while True:
            poll_index += 1
            try:
                data = await self._post_json(
                    url=url,
                    payload=self._build_get_task_payload(task_id),
                    timeout=timeout,
                )
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                consecutive_failures += 1
                if consecutive_failures < max_retries and not isinstance(
                    exc, (httpx.ReadTimeout, httpx.WriteTimeout)
                ):
                    delay_seconds = float(min(2**consecutive_failures, 8))
                    self._event_bus.publish(
                        "a2a_call_retry",
                        call_id=call_id,
                        port=port,
                        attempt=poll_index,
                        phase="poll",
                        delay_seconds=delay_seconds,
                        error=self._format_exception_detail(exc),
                        task_id=task_id,
                        **metadata,
                    )
                    await asyncio.sleep(delay_seconds)
                    continue
                raise RuntimeError(
                    "A2A task polling failed after "
                    f"{consecutive_failures} attempt(s) with per-request timeout={timeout_seconds}s. "
                    f"Task may still be running. Last error: {self._format_exception_detail(exc)}"
                ) from exc

            if "error" in data:
                raise RuntimeError(
                    f"A2A error while polling task {task_id}: {data['error']}"
                )

            result = data.get("result", {})
            state = self._extract_task_state(result)
            if state is not None and state != last_state:
                self._event_bus.publish(
                    "status_message",
                    f"[A2A] task {task_id} on port {port} state={state}",
                )
                last_state = state

            if state in self._TERMINAL_TASK_STATES:
                if state in self._SUCCESS_TASK_STATES:
                    return result
                detail = self._extract_text_result(result) or state or "unknown"
                raise RuntimeError(
                    f"A2A task {task_id} finished in state={state}: {detail}"
                )

            direct_text = self._extract_text_result(result)
            if direct_text and state is None:
                return result

            await asyncio.sleep(self._POLL_INTERVAL_SECONDS)

    async def send_task(
        self,
        port: int,
        task_data: dict | str | object,
        timeout: float | None = 120.0,
        max_retries: int = 3,
        metadata: dict | None = None,
    ) -> str:
        """
        Sends a task to the agent.
        """
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1.")

        call_id = str(uuid.uuid4())
        metadata = metadata or {}

        task_text: str
        model_dump_json = getattr(task_data, "model_dump_json", None)
        as_dict = getattr(task_data, "dict", None)
        if callable(model_dump_json):
            task_text = str(model_dump_json())
        elif callable(as_dict):
            task_text = json.dumps(as_dict())
        elif isinstance(task_data, dict):
            task_text = json.dumps(task_data)
        else:
            task_text = str(task_data)

        payload = self._build_message_send_payload(task_text)
        url = f"http://127.0.0.1:{port}"
        http_timeout_seconds = self._resolve_http_timeout_seconds(timeout)
        http_timeout = self._build_http_timeout(timeout)

        self._event_bus.publish(
            "a2a_call_started",
            call_id=call_id,
            port=port,
            timeout=http_timeout_seconds,
            max_retries=max_retries,
            **metadata,
        )

        last_exception: Exception | None = None
        for attempt in range(1, max_retries + 1):
            attempt_start = time.perf_counter()
            try:
                data = await self._post_json(
                    url=url,
                    payload=payload,
                    timeout=http_timeout,
                )

                if "error" in data:
                    raise RuntimeError(f"A2A error: {data['error']}")

                result = data.get("result", {})
                task_id = self._extract_task_id(result)
                task_state = self._extract_task_state(result)
                if task_id is not None and task_state not in self._TERMINAL_TASK_STATES:
                    result = await self._poll_task_until_terminal(
                        url=url,
                        task_id=task_id,
                        timeout=http_timeout,
                        timeout_seconds=http_timeout_seconds,
                        max_retries=max_retries,
                        call_id=call_id,
                        port=port,
                        metadata=metadata,
                    )

                text_result = self._extract_text_result(result)
                total_latency_ms = int((time.perf_counter() - attempt_start) * 1000)
                self._event_bus.publish(
                    "a2a_call_succeeded",
                    call_id=call_id,
                    port=port,
                    attempt=attempt,
                    latency_ms=total_latency_ms,
                    **metadata,
                )
                if text_result is not None:
                    return text_result
                return str(result)
            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_exception = exc
                is_unsafe_retry_timeout = isinstance(
                    exc, (httpx.ReadTimeout, httpx.WriteTimeout)
                )
                if attempt < max_retries and not is_unsafe_retry_timeout:
                    delay_seconds = float(2**attempt)
                    self._event_bus.publish(
                        "a2a_call_retry",
                        call_id=call_id,
                        port=port,
                        attempt=attempt,
                        delay_seconds=delay_seconds,
                        error=self._format_exception_detail(exc),
                        **metadata,
                    )
                    await asyncio.sleep(
                        delay_seconds
                    )  # Exponential backoff (2s, 4s, 8s)
                    continue
                break
            except RuntimeError as exc:
                last_exception = exc
                break

        last_error = self._format_exception_detail(last_exception)
        if isinstance(
            last_exception,
            (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout),
        ):
            error_message = (
                f"Failed after {max_retries} attempts. Per-request timeout hit after {http_timeout_seconds}s. "
                "The task may still be running on the agent; retrying the same HTTP request can "
                f"duplicate work. Last error: {last_error}"
            )
        elif isinstance(last_exception, RuntimeError):
            error_message = str(last_exception)
        else:
            error_message = (
                f"Failed after {max_retries} attempts. Last error: {last_error}"
            )
        self._event_bus.publish(
            "a2a_call_failed",
            call_id=call_id,
            port=port,
            error=error_message,
            **metadata,
        )
        if last_exception is not None:
            raise RuntimeError(error_message) from last_exception
        raise RuntimeError(error_message)

    @staticmethod
    def _format_exception_detail(exc: Exception | None) -> str:
        """Return a stable error summary that always includes exception type."""
        if exc is None:
            return "none"
        error_text = str(exc).strip() or repr(exc)
        return f"{type(exc).__name__}: {error_text}"


_default_agent_client: AgentClient | None = None


def create_agent_client(*, event_bus: EventBus | None = None) -> AgentClient:
    """Create a new ``AgentClient`` instance."""
    return AgentClient(event_bus=event_bus)


def get_default_agent_client() -> AgentClient:
    """Return process default ``AgentClient`` initialized lazily."""
    global _default_agent_client
    if _default_agent_client is None:
        _default_agent_client = create_agent_client()
    return _default_agent_client


# Backward-compatible default client for existing imports.
agent_client = get_default_agent_client()
