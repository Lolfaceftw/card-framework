import time
import requests
from abc import ABC, abstractmethod
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
        url = f"http://127.0.0.1:{port}/.well-known/agent.json"

        for attempt in range(1, self.max_retries + 1):
            try:
                r = requests.get(url, timeout=5)
                r.raise_for_status()
                event_bus.publish("status_message", f"[OK] {name} agent is up")
                return True
            except requests.exceptions.RequestException as e:
                # E.g., connection refused because uvicorn is still starting up
                if attempt == self.max_retries:
                    event_bus.publish(
                        "error_message",
                        f"[ERR] {name} server not responding after {self.max_retries} attempts: {e}",
                    )
                    return False

                delay = self.base_delay * (2 ** (attempt - 1))
                event_bus.publish(
                    "system_message",
                    f"Waiting for {name} server (attempt {attempt}/{self.max_retries}). Retrying in {delay}s...",
                )
                time.sleep(delay)

        return False
