"""Tests for dashboard refresh throttling behavior."""

from __future__ import annotations

from audio2script_and_summarizer import run_pipeline


def test_request_refresh_throttles_rapid_non_forced_updates(monkeypatch) -> None:
    """Throttle rapid redraw requests to avoid one-refresh-per-line behavior."""
    dashboard = run_pipeline._PipelineDashboard(enabled=False)
    dashboard.enabled = True
    dashboard._last_refresh_monotonic = 0.0  # noqa: SLF001

    refresh_calls: list[str] = []
    monkeypatch.setattr(dashboard, "_refresh", lambda: refresh_calls.append("refresh"))

    monotonic_values = iter([1.00, 1.01, 1.03, 1.08])
    monkeypatch.setattr(run_pipeline.time, "monotonic", lambda: next(monotonic_values))

    dashboard._request_refresh()  # noqa: SLF001
    dashboard._request_refresh()  # noqa: SLF001
    dashboard._request_refresh()  # noqa: SLF001
    dashboard._request_refresh()  # noqa: SLF001

    assert refresh_calls == ["refresh", "refresh"]
    assert dashboard._refresh_pending is False  # noqa: SLF001


def test_prompt_numeric_choice_consumes_recent_pending_enter(monkeypatch) -> None:
    """Consume Enter pressed just before prompt open and pick default choice."""
    dashboard = run_pipeline._PipelineDashboard(enabled=False)
    dashboard.enabled = True
    dashboard._keyboard_enabled = True  # noqa: SLF001
    dashboard._started = False  # noqa: SLF001
    dashboard._pending_prompt_enter_monotonic = 10.0  # noqa: SLF001

    monkeypatch.setattr(run_pipeline.time, "monotonic", lambda: 10.1)
    monkeypatch.setattr(dashboard, "_request_refresh", lambda *args, **kwargs: None)

    selected = dashboard.prompt_numeric_choice(
        title="Select transcript JSON file",
        options=["a.json", "b.json"],
        default_choice=1,
    )

    assert selected == 1
    assert dashboard._pending_prompt_enter_monotonic is None  # noqa: SLF001
    assert dashboard._prompt_opened_monotonic is None  # noqa: SLF001
    assert dashboard._prompt_active is False  # noqa: SLF001


def test_prompt_numeric_choice_ignores_stale_pending_enter(monkeypatch) -> None:
    """Ignore stale Enter keypresses outside the configured grace window."""
    dashboard = run_pipeline._PipelineDashboard(enabled=False)
    dashboard.enabled = True
    dashboard._keyboard_enabled = True  # noqa: SLF001
    dashboard._started = False  # noqa: SLF001
    dashboard._pending_prompt_enter_monotonic = 1.0  # noqa: SLF001

    monkeypatch.setattr(run_pipeline.time, "monotonic", lambda: 2.0)
    monkeypatch.setattr(dashboard, "_request_refresh", lambda *args, **kwargs: None)

    selected = dashboard.prompt_numeric_choice(
        title="Select transcript JSON file",
        options=["a.json", "b.json"],
        default_choice=1,
    )

    assert selected is None
    assert dashboard._pending_prompt_enter_monotonic is None  # noqa: SLF001
    assert dashboard._prompt_opened_monotonic is None  # noqa: SLF001
