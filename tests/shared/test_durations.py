"""Tests for the human duration strings used by the command line."""

from __future__ import annotations

import pytest

from card_framework.shared.durations import (
    duration_label,
    format_duration,
    parse_duration,
)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("300", 300),
        ("30s", 30),
        ("5m", 300),
        ("1m30s", 90),
        ("1m 30s", 90),
        ("1h", 3600),
        ("1.5m", 90),
        ("1:30", 90),
        ("1:02:14", 3734),
        (" 5M ", 300),
    ],
)
def test_parse_duration_accepts_common_forms(text: str, seconds: int) -> None:
    """Read the duration forms people type at a prompt."""
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "five minutes", "5x", "1:xx", "0", "0s", "1:2:3:4", "m"])
def test_parse_duration_rejects_unreadable_or_zero(text: str) -> None:
    """Reject text that isn't a positive duration, with a hint in the message."""
    with pytest.raises(ValueError):
        parse_duration(text)


def test_format_duration_switches_to_hours() -> None:
    """Show m:ss below an hour and h:mm:ss from an hour up."""
    assert format_duration(0) == "0:00"
    assert format_duration(299.6) == "5:00"
    assert format_duration(3734) == "1:02:14"


def test_duration_label_is_folder_friendly() -> None:
    """Build short labels for default output folder names."""
    assert duration_label(30) == "30s"
    assert duration_label(300) == "5m"
    assert duration_label(90) == "1m30s"
