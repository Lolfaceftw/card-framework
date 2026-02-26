"""Lightweight event bus used for cross-component observability signals."""

from collections.abc import Callable
from typing import Any
import logging
import inspect


class EventBus:
    """Process-wide pub/sub event bus with resilient callback dispatch."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventBus, cls).__new__(cls)
            cls._instance.subscribers: dict[str, list[Callable[..., Any]]] = {}
        return cls._instance

    def subscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Subscribe a callback for an event type."""
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        self.subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Unsubscribe a callback from an event type if present."""
        callbacks = self.subscribers.get(event_type, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def publish(self, event_type: str, *args: Any, **kwargs: Any) -> None:
        """
        Publish an event to all subscribers.

        Dispatch is resilient:
        - If a callback does not accept keyword metadata, we retry with positional args only.
        - Subscriber exceptions are isolated so one failing callback does not block others.
        """
        if event_type in self.subscribers:
            for callback in self.subscribers[event_type]:
                try:
                    callback(*args, **kwargs)
                except TypeError:
                    # Backwards compatibility with callbacks that do not accept extra metadata.
                    try:
                        signature = inspect.signature(callback)
                        accepted_kwargs = {
                            name: value
                            for name, value in kwargs.items()
                            if name in signature.parameters
                        }
                        callback(*args, **accepted_kwargs)
                    except Exception:
                        logging.getLogger(__name__).exception(
                            "Event subscriber failed",
                            extra={
                                "event_type": event_type,
                                "callback": str(callback),
                            },
                        )
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Event subscriber failed",
                        extra={"event_type": event_type, "callback": str(callback)},
                    )


event_bus = EventBus()
