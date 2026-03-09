"""Runtime event capture helpers for benchmark instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from card_framework.shared.events import event_bus


@dataclass(slots=True)
class CapturedEvent:
    """One captured event bus emission."""

    event_type: str
    timestamp_utc: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class EventCapture:
    """Subscribe to selected event types and collect events during one run scope."""

    def __init__(self, event_types: list[str]) -> None:
        self._event_types = list(dict.fromkeys(event_types))
        self._callbacks: dict[str, Callable[..., None]] = {}
        self._events: list[CapturedEvent] = []

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _build_callback(self, event_type: str) -> Callable[..., None]:
        def _callback(*args: Any, **kwargs: Any) -> None:
            self._events.append(
                CapturedEvent(
                    event_type=event_type,
                    timestamp_utc=self._now(),
                    args=args,
                    kwargs=kwargs,
                )
            )

        return _callback

    def start(self) -> None:
        """Start capturing configured events."""
        for event_type in self._event_types:
            callback = self._build_callback(event_type)
            self._callbacks[event_type] = callback
            event_bus.subscribe(event_type, callback)

    def stop(self) -> None:
        """Stop capturing events and detach callbacks."""
        for event_type, callback in self._callbacks.items():
            event_bus.unsubscribe(event_type, callback)
        self._callbacks.clear()

    def events(self) -> list[CapturedEvent]:
        """Return captured events in insertion order."""
        return list(self._events)

    def count(self, event_type: str) -> int:
        """Return number of captured events for a given type."""
        return sum(1 for event in self._events if event.event_type == event_type)

    def latest_kwargs(self, event_type: str) -> dict[str, Any] | None:
        """Return kwargs from most recent event of type, if present."""
        for event in reversed(self._events):
            if event.event_type == event_type:
                return event.kwargs
        return None

