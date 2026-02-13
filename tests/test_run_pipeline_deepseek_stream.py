"""Tests for DeepSeek stream marker parsing and dashboard routing."""

from __future__ import annotations

import json

from audio2script_and_summarizer import run_pipeline
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
    """Route stream events to dashboard handlers without early panel close."""
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
    _route_deepseek_stream_event(  # noqa: SLF001
        dashboard, {"event": "done"}
    )

    assert dashboard.opened_model == "deepseek-reasoner"
    assert dashboard.tokens == [("answer", "hello")]
    assert dashboard.logs == [
        "[DEEPSEEK STATUS] round 1/3",
        "[DEEPSEEK TOOL_CALL] Invoked count_words.",
        (
            "[DEEPSEEK STATUS] Summary JSON ready; finalizing Stage 2 "
            "subprocess (output=out.json)."
        ),
        "[DEEPSEEK STATUS] Stream finished for current DeepSeek call.",
    ]
    assert dashboard.closed is False


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


def test_run_stage_command_closes_stream_panel_on_stage2_exit(
    monkeypatch,
) -> None:
    """Close DeepSeek panel only after Stage 2 subprocess exits."""

    class _FakeDashboardForStageRun:
        """Capture Stage 2 lifecycle events emitted by `_run_stage_command`."""

        enabled = True

        def __init__(self) -> None:
            self.timeline: list[str] = []
            self.substeps: list[str] = []
            self.closed = False

        def open_deepseek_stream_panel(self, model_name: str) -> None:
            self.timeline.append(f"open:{model_name}")

        def append_deepseek_stream_token(self, *, phase: str, text: str) -> None:
            self.timeline.append(f"token:{phase}:{text}")

        def update_deepseek_context_usage(
            self,
            *,
            tokens_used: int,
            tokens_limit: int,
            tokens_left: int,
            percent_left: float,
            rollover_count: int = 0,
        ) -> None:
            self.timeline.append(
                "context_usage:"
                f"{tokens_used}/{tokens_limit}/{tokens_left}/{percent_left}/"
                f"{rollover_count}"
            )

        def close_deepseek_stream_panel(self) -> None:
            self.timeline.append("close")
            self.closed = True

        def log(self, message: str) -> None:
            self.timeline.append(f"log:{message}")

        def set_status(
            self,
            *,
            stage_name: str,
            substep: str,
            module_name: str,
            command_display: str,
            model_info: str,
            pid: int | None,
            reset_elapsed: bool = False,
        ) -> None:
            _ = (
                stage_name,
                module_name,
                command_display,
                model_info,
                pid,
                reset_elapsed,
            )
            self.substeps.append(substep)
            self.timeline.append(f"status:{substep}")

        def set_progress_detail(self, detail: str) -> None:
            self.timeline.append(f"progress:{detail}")

        def event(self, message: str) -> None:
            self.timeline.append(f"event:{message}")

        def start_detail_progress(self, label: str) -> None:
            self.timeline.append(f"detail_start:{label}")

        def update_detail_progress(self, percent_complete: int) -> None:
            self.timeline.append(f"detail_update:{percent_complete}")

        def finish_detail_progress(self, status: str = "complete") -> None:
            self.timeline.append(f"detail_finish:{status}")

    class _FakeProcess:
        """Provide deterministic subprocess output for Stage 2 lifecycle checks."""

        def __init__(self, lines: list[str]) -> None:
            self.pid = 43210
            self.stdout = lines

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    dashboard = _FakeDashboardForStageRun()
    marker_prefix = DEEPSEEK_STREAM_EVENT_PREFIX
    stream_lines = [
        marker_prefix
        + json.dumps({"event": "start", "model": "deepseek-reasoner"})
        + "\n",
        marker_prefix + json.dumps({"event": "summary_json_ready", "path": "out.json"}) + "\n",
        "post-summary cleanup\n",
        marker_prefix + json.dumps({"event": "done"}) + "\n",
    ]

    monkeypatch.setattr(run_pipeline, "_ACTIVE_DASHBOARD", dashboard)
    monkeypatch.setattr(
        run_pipeline.subprocess,
        "Popen",
        lambda *args, **kwargs: _FakeProcess(stream_lines),
    )

    run_pipeline._run_stage_command(  # noqa: SLF001
        cmd=["python", "-m", "audio2script_and_summarizer.summarizer_deepseek"],
        current_env={},
        use_dashboard=True,
        stage_name="Stage 2 (Summarizer)",
        heartbeat_seconds=0.01,
        model_info="Provider: deepseek",
    )

    summary_log = (
        "[DEEPSEEK STATUS] Summary JSON ready; finalizing Stage 2 "
        "subprocess (output=out.json)."
    )
    summary_log_index = dashboard.timeline.index(f"log:{summary_log}")
    close_index = dashboard.timeline.index("close")
    assert close_index > summary_log_index
    assert dashboard.closed is True
    assert "Summary JSON ready; finalizing subprocess" in dashboard.substeps
    assert (
        "DeepSeek stream done; waiting for subprocess completion"
        in dashboard.substeps
    )
