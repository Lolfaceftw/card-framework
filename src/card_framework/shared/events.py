"""Lightweight event bus used for cross-component observability signals."""

from __future__ import annotations

from collections.abc import Callable
import inspect
import logging
from typing import Any


class EventBus:
    """In-memory pub/sub event bus with resilient callback dispatch."""

    def __init__(self) -> None:
        """Initialize an empty subscriber registry."""
        self.subscribers: dict[str, list[Callable[..., Any]]] = {}

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

_default_event_bus: EventBus | None = None


def create_event_bus() -> EventBus:
    """Create a new event-bus instance."""
    return EventBus()


def get_event_bus() -> EventBus:
    """Return process default bus, creating it lazily when needed."""
    global _default_event_bus
    if _default_event_bus is None:
        _default_event_bus = create_event_bus()
    return _default_event_bus


def set_event_bus(bus: EventBus) -> None:
    """Set process default bus for module-level consumers."""
    global _default_event_bus
    _default_event_bus = bus


# Backward-compatible default bus for existing imports.
event_bus = get_event_bus()
