"""Tests for shared runtime loading/perf helpers."""

from __future__ import annotations

import pytest

from card_framework.shared.perf import report_loading_duration


def test_report_loading_duration_reports_start_and_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit loading start and success lines with a rounded wall-clock duration."""
    perf_counter_values = iter((10.0, 12.345))
    monkeypatch.setattr(
        "card_framework.shared.perf.time.perf_counter",
        lambda: next(perf_counter_values),
    )
    messages: list[str] = []

    with report_loading_duration("runtime helpers", reporter=messages.append):
        pass

    assert messages == [
        "Loading runtime helpers...",
        "Loaded runtime helpers in 2.35 seconds.",
    ]


def test_report_loading_duration_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit a failed-loading line before re-raising the wrapped exception."""
    perf_counter_values = iter((5.0, 6.2))
    monkeypatch.setattr(
        "card_framework.shared.perf.time.perf_counter",
        lambda: next(perf_counter_values),
    )
    messages: list[str] = []

    with pytest.raises(RuntimeError, match="boom"):
        with report_loading_duration("runtime helpers", reporter=messages.append):
            raise RuntimeError("boom")

    assert messages == [
        "Loading runtime helpers...",
        "Failed loading runtime helpers after 1.20 seconds.",
    ]
