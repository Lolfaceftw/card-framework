"""Tests for DeepSeek stream marker parsing and dashboard routing."""

from __future__ import annotations

from audio2script_and_summarizer.run_pipeline import (
    DEEPSEEK_STREAM_EVENT_PREFIX,
    _parse_deepseek_stream_event_line,
    _route_deepseek_stream_event,
)


class _FakeDashboard:
    """Capture stream panel actions for routing assertions."""

    def __init__(self) -> None:
        self.opened_model: str | None = None
        self.tokens: list[tuple[str, str]] = []
        self.logs: list[str] = []
        self.closed = False
        self.context_usage: dict[str, float | int] | None = None

    def open_deepseek_stream_panel(self, model_name: str) -> None:
        """Record stream panel open action."""
        self.opened_model = model_name

    def append_deepseek_stream_token(self, *, phase: str, text: str) -> None:
        """Record streamed token append action."""
        self.tokens.append((phase, text))

    def close_deepseek_stream_panel(self) -> None:
        """Record stream panel close action."""
        self.closed = True

    def update_deepseek_context_usage(
        self,
        *,
        tokens_used: int,
        tokens_limit: int,
        tokens_left: int,
        percent_left: float,
        rollover_count: int = 0,
    ) -> None:
        """Record context usage subtitle data updates."""
        self.context_usage = {
            "tokens_used": tokens_used,
            "tokens_limit": tokens_limit,
            "tokens_left": tokens_left,
            "percent_left": percent_left,
            "rollover_count": rollover_count,
        }

    def log(self, message: str) -> None:
        """Record output panel log lines."""
        self.logs.append(message)


def test_parse_deepseek_stream_event_line_parses_json_payload() -> None:
    """Parse prefixed JSON marker lines into payload dictionaries."""
    line = (
        f'{DEEPSEEK_STREAM_EVENT_PREFIX}'
        '{"event":"token","phase":"reasoning","text":"abc"}'
    )

    payload = _parse_deepseek_stream_event_line(line)  # noqa: SLF001

    assert payload == {"event": "token", "phase": "reasoning", "text": "abc"}


def test_parse_deepseek_stream_event_line_returns_none_for_non_marker() -> None:
    """Ignore regular subprocess lines that are not stream markers."""
    payload = _parse_deepseek_stream_event_line("regular log line")  # noqa: SLF001
    assert payload is None


def test_route_deepseek_stream_event_dispatches_panel_actions() -> None:
    """Route start/token/close stream events to dashboard handlers."""
    dashboard = _FakeDashboard()

    _route_deepseek_stream_event(  # noqa: SLF001
        dashboard, {"event": "start", "model": "deepseek-reasoner"}
    )
    _route_deepseek_stream_event(  # noqa: SLF001
        dashboard, {"event": "token", "phase": "answer", "text": "hello"}
    )
    _route_deepseek_stream_event(  # noqa: SLF001
        dashboard, {"event": "token", "phase": "status", "text": "round 1/3"}
    )
    _route_deepseek_stream_event(  # noqa: SLF001
        dashboard,
        {"event": "token", "phase": "tool_call", "text": "Invoked count_words."},
    )
    _route_deepseek_stream_event(  # noqa: SLF001
        dashboard, {"event": "summary_json_ready", "path": "out.json"}
    )

    assert dashboard.opened_model == "deepseek-reasoner"
    assert dashboard.tokens == [("answer", "hello")]
    assert dashboard.logs == [
        "[DEEPSEEK STATUS] round 1/3",
        "[DEEPSEEK TOOL_CALL] Invoked count_words.",
    ]
    assert dashboard.closed is True


def test_route_deepseek_stream_event_updates_context_usage() -> None:
    """Route context usage payloads to subtitle-state dashboard handler."""
    dashboard = _FakeDashboard()

    _route_deepseek_stream_event(  # noqa: SLF001
        dashboard,
        {
            "event": "context_usage",
            "tokens_used": 44800,
            "tokens_limit": 64000,
            "tokens_left": 19200,
            "percent_left": 0.30,
            "rollover_count": 1,
        },
    )

    assert dashboard.context_usage == {
        "tokens_used": 44800,
        "tokens_limit": 64000,
        "tokens_left": 19200,
        "percent_left": 0.30,
        "rollover_count": 1,
    }
