"""Parse and format the human duration strings used by the command line."""

from __future__ import annotations

import re

_UNIT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(h|m|s)", re.IGNORECASE)
_UNIT_SECONDS = {"h": 3600, "m": 60, "s": 1}


def parse_duration(text: str) -> int:
    """Return a whole number of seconds for a duration such as ``5m`` or ``1:30``.

    Accepted forms are plain seconds (``300``), unit strings (``90s``, ``5m``,
    ``1m30s``, ``1h``), and clock strings (``1:30``, ``1:02:14``).

    Raises:
        ValueError: The text is empty, malformed, or not greater than zero.
    """
    raw = text.strip().lower()
    if not raw:
        raise ValueError("Duration is empty. Try 30s, 5m, or 1:30.")

    if raw.isdigit():
        seconds = float(raw)
    elif ":" in raw:
        seconds = _parse_clock(raw)
    else:
        seconds = _parse_units(raw)

    whole_seconds = round(seconds)
    if whole_seconds <= 0:
        raise ValueError(f"Duration must be longer than zero seconds: {text!r}.")
    return whole_seconds


def format_duration(seconds: float) -> str:
    """Format seconds as ``m:ss``, or ``h:mm:ss`` from one hour up."""
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def duration_label(seconds: int) -> str:
    """Return a short label such as ``30s``, ``5m``, or ``1m30s`` for folder names."""
    minutes, secs = divmod(seconds, 60)
    if not minutes:
        return f"{secs}s"
    if not secs:
        return f"{minutes}m"
    return f"{minutes}m{secs}s"


def _parse_clock(raw: str) -> float:
    parts = raw.split(":")
    if len(parts) > 3 or not all(part.isdigit() for part in parts):
        raise ValueError(f"Could not read {raw!r} as a duration. Try 1:30 or 1:02:14.")
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + int(part)
    return float(seconds)


def _parse_units(raw: str) -> float:
    compact = raw.replace(" ", "")
    matches = list(_UNIT_PATTERN.finditer(compact))
    if not matches or "".join(match.group(0) for match in matches) != compact:
        raise ValueError(f"Could not read {raw!r} as a duration. Try 30s, 5m, or 1m30s.")
    return sum(float(match.group(1)) * _UNIT_SECONDS[match.group(2)] for match in matches)
