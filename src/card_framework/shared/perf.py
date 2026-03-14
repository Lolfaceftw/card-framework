"""Shared wall-clock timing helpers for operator-visible loading steps."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
import time

type StatusReporter = Callable[[str], None]


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
