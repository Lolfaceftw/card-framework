"""Health checks for locally hosted A2A services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

import requests

from events import event_bus


class HealthCheckStrategy(ABC):
    """
    Abstract strategy for checking the health of a dependent service.
    """

    @abstractmethod
    def check(self, name: str, port: int) -> bool:
        """
        Check if the service is up.
        Returns True if healthy, False otherwise.
        """
        pass


class AgentHealthChecker(HealthCheckStrategy):
    """
    Concrete strategy that checks the health of an A2A agent by hitting
    its '/.well-known/agent.json' endpoint. Uses the Retry pattern with
    exponential backoff.
    """

    def __init__(self, max_retries: int = 5, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay

    def check(self, name: str, port: int) -> bool:
        for attempt in range(1, self.max_retries + 1):
            ok, error = self.check_once(name, port, request_timeout_seconds=5.0)
            if ok:
                event_bus.publish("status_message", f"[OK] {name} agent is up")
                return True
            if attempt == self.max_retries:
                event_bus.publish(
                    "error_message",
                    (
                        f"[ERR] {name} server not responding after "
                        f"{self.max_retries} attempts: {error}"
                    ),
                )
                return False

            delay = self.base_delay * (2 ** (attempt - 1))
            event_bus.publish(
                "system_message",
                f"Waiting for {name} server (attempt {attempt}/{self.max_retries}). Retrying in {delay}s...",
            )
            time.sleep(delay)

        return False

    def check_once(
        self,
        name: str,
        port: int,
        *,
        request_timeout_seconds: float = 1.0,
    ) -> tuple[bool, str | None]:
        """Probe one agent health endpoint without retrying."""
        url = f"http://127.0.0.1:{port}/.well-known/agent.json"
        try:
            response = requests.get(url, timeout=max(0.1, request_timeout_seconds))
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            return False, str(exc)
        return True, None

    def wait_for_many(
        self,
        servers: Sequence[tuple[str, int]],
        *,
        overall_timeout_seconds: float = 10.0,
        poll_interval_seconds: float = 0.2,
        request_timeout_seconds: float = 1.0,
    ) -> bool:
        """
        Wait for multiple agents in parallel within a shared startup deadline.

        Args:
            servers: Sequence of ``(name, port)`` pairs to probe.
            overall_timeout_seconds: Shared deadline for all servers.
            poll_interval_seconds: Delay between poll rounds for pending servers.
            request_timeout_seconds: Per-request timeout for each health probe.

        Returns:
            ``True`` when every server becomes healthy before the deadline.
        """
        pending = {name: port for name, port in servers}
        if not pending:
            return True

        healthy: set[str] = set()
        last_errors: dict[str, str] = {}
        deadline = time.monotonic() + max(0.1, overall_timeout_seconds)

        while pending:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                break

            probe_timeout = min(max(0.1, request_timeout_seconds), remaining_seconds)
            next_pending: dict[str, int] = {}
            with ThreadPoolExecutor(max_workers=len(pending)) as executor:
                future_map = {
                    executor.submit(
                        self.check_once,
                        name,
                        port,
                        request_timeout_seconds=probe_timeout,
                    ): (name, port)
                    for name, port in pending.items()
                }
                for future in as_completed(future_map):
                    name, port = future_map[future]
                    ok, error = future.result()
                    if ok:
                        if name not in healthy:
                            event_bus.publish("status_message", f"[OK] {name} agent is up")
                            healthy.add(name)
                        continue
                    next_pending[name] = port
                    if error is not None:
                        last_errors[name] = error

            pending = next_pending
            if not pending:
                return True

            sleep_seconds = min(
                max(0.0, poll_interval_seconds),
                max(0.0, deadline - time.monotonic()),
            )
            if sleep_seconds <= 0:
                break

            event_bus.publish(
                "system_message",
                (
                    "Waiting for agent servers to start: "
                    f"{', '.join(sorted(pending))}. Retrying in {sleep_seconds:.1f}s..."
                ),
            )
            time.sleep(sleep_seconds)

        for name in sorted(pending):
            detail = last_errors.get(name, "unknown startup failure")
            event_bus.publish(
                "error_message",
                (
                    f"[ERR] {name} server not responding within "
                    f"{overall_timeout_seconds:.1f}s: {detail}"
                ),
            )
        return False
