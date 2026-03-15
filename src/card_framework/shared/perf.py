"""Shared wall-clock timing helpers for operator-visible loading steps."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import time

type StatusReporter = Callable[[str], None]


@dataclass(slots=True)
class PerfTimer:
    """Track elapsed wall-clock time from a captured `perf_counter` start."""

    started: float

    def elapsed_seconds(self) -> float:
        """Return the current elapsed wall-clock seconds."""
        return max(time.perf_counter() - self.started, 0.0)


@contextmanager
def measure_elapsed_time() -> Iterator[PerfTimer]:
    """Yield a timer object backed by `time.perf_counter`."""
    yield PerfTimer(started=time.perf_counter())


@contextmanager
def report_loading_duration(
    description: str,
    *,
    reporter: StatusReporter | None = None,
) -> Iterator[None]:
    """Report loading start and completion with wall-clock duration.

    Args:
        description: Human-readable label for the step being timed.
        reporter: Optional callback that receives status lines.

    Raises:
        Exception: Re-raises any exception from the wrapped block after
            reporting the failed duration.
    """
    if reporter is not None:
        reporter(f"Loading {description}...")
    started = time.perf_counter()
    try:
        yield
    except Exception:
        if reporter is not None:
            reporter(
                f"Failed loading {description} after "
                f"{time.perf_counter() - started:.2f} seconds."
            )
        raise
    if reporter is not None:
        reporter(
            f"Loaded {description} in {time.perf_counter() - started:.2f} seconds."
        )
