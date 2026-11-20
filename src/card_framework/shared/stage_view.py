"""Draw a CARD run in the terminal: a live stage list, or plain lines when piped."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import IO, TextIO

from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.padding import Padding
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from card_framework.runtime.progress import RunState, Stage, failure_cause, tail_lines
from card_framework.shared.durations import format_duration

THEME = Theme(
    {
        "card.voice_a": "#FF7A59",
        "card.voice_b": "#39D3C3",
        "card.done": "#FFD166",
        "card.active": "#39D3C3",
        "card.fail": "bold #FF7A59",
        "card.muted": "grey58",
        "card.faint": "grey37",
    }
)
_MARK = "▂▅█▆▃"


def make_console(stream: TextIO | None = None) -> Console:
    """Return a stderr console that honours NO_COLOR and non-terminal output."""
    return Console(file=stream or sys.stderr, theme=THEME, highlight=False)


def mark() -> Text:
    """Return the five-bar CARD mark, alternating the two speaker colours."""
    text = Text()
    for index, char in enumerate(_MARK):
        text.append(char, style="card.voice_a" if index % 2 == 0 else "card.voice_b")
    return text


def header(title: str, subtitle: str = "") -> Text:
    """Return the one-line header used by every CARD command."""
    line = Text(" ")
    line.append_text(mark())
    line.append("  CARD", style="bold")
    if title:
        line.append(f"  {title}")
    if subtitle:
        line.append(f"  {subtitle}", style="card.muted")
    return line


def make_stage_view(stream: TextIO | None = None, *, title: str = "") -> LiveStageView | PlainStageView:
    """Return a live view for terminals and a line-based view for pipes and logs."""
    console = make_console(stream)
    if console.is_terminal and not console.is_dumb_terminal:
        return LiveStageView(console, title=title)
    return PlainStageView(console.file)


class LiveStageView:
    """A redrawn stage list for interactive terminals."""

    def __init__(self, console: Console, *, title: str = "") -> None:
        self._console = console
        self._title = title
        self._spinner = Spinner("dots", style="card.active")
        self._live: Live | None = None

    def update(self, state: RunState) -> None:
        """Redraw the stage list for ``state``."""
        if self._live is None:
            self._console.print(header(self._title))
            self._console.print()
            self._live = Live(
                self._render(state),
                console=self._console,
                refresh_per_second=8,
                transient=False,
            )
            self._live.start()
        else:
            self._live.update(self._render(state))

    def close(self, state: RunState, *, succeeded: bool) -> None:
        """Draw the final list, then the outcome below it."""
        if self._live is None:
            self.update(state)
        assert self._live is not None
        self._live.update(self._render(state, failed=not succeeded))
        self._live.stop()
        self._live = None
        elapsed = format_duration(time.monotonic() - state.started_at)
        self._console.print()
        if succeeded:
            self._console.print(Text(f"   Done in {elapsed}.", style="card.done"))
            return
        self._console.print(Text(f"   Stopped after {elapsed}.", style="card.fail"))
        if state.log_path is not None:
            cause = failure_cause(state.log_path)
            if cause:
                self._console.print(Text(f"   Cause: {cause}"))
            for line in tail_lines(state.log_path, count=12):
                self._console.print(Text(f"   │ {line}", style="card.faint"), overflow="ellipsis", no_wrap=True)
            self._console.print(Text(f"   Full log: {short_path(state.log_path)}", style="card.muted"))

    def _render(self, state: RunState, *, failed: bool = False) -> RenderableType:
        now = time.monotonic()
        table = Table.grid(padding=(0, 2))
        table.add_column(width=2)
        table.add_column(no_wrap=True)
        table.add_column(style="card.muted", no_wrap=True, overflow="ellipsis")
        table.add_column(justify="right", style="card.faint", no_wrap=True)

        setup_done = state.setup_done_at is not None
        if setup_done:
            setup_seconds = state.setup_done_at - state.started_at
            table.add_row(Text("✓", style="card.done"), "Setup", "", format_duration(setup_seconds))
        elif failed:
            table.add_row(Text("✗", style="card.fail"), "Setup", state.setup_message, "")
        else:
            table.add_row(self._spinner, "Setup", state.setup_message, format_duration(now - state.started_at))

        for index, stage in enumerate(state.stages):
            table.add_row(*self._stage_cells(state, index, stage, now, failed=failed))

        rows: list[RenderableType] = [Padding(table, (0, 0, 0, 3))]
        if state.log_path is not None and not failed:
            rows.append(Text(f"\n   Log: {short_path(state.log_path)}", style="card.faint"))
        return Group(*rows)

    def _stage_cells(
        self,
        state: RunState,
        index: int,
        stage: Stage,
        now: float,
        *,
        failed: bool,
    ) -> tuple[RenderableType, RenderableType, str, str]:
        name = stage.artifact.name
        if index < state.completed:
            started = state.stage_started_at(index) or now
            took = format_duration(state.stage_finished_at[index] - started)
            return Text("✓", style="card.done"), name, state.notes.get(stage.key, ""), took
        started = state.stage_started_at(index)
        if started is None:
            return Text("·", style="card.faint"), Text(name, style="card.faint"), "", ""
        if failed:
            return Text("✗", style="card.fail"), name, stage.activity, format_duration(now - started)
        return self._spinner, name, stage.activity, format_duration(now - started)


class PlainStageView:
    """One ASCII line per change, for pipes, CI logs, and NO_COLOR-style output."""

    def __init__(self, stream: IO[str]) -> None:
        self._stream = stream
        self._printed_setup: str | None = None
        self._printed_stages = 0
        self._setup_closed = False

    def update(self, state: RunState) -> None:
        """Print what changed since the last call."""
        if state.setup_done_at is None and state.setup_message != self._printed_setup:
            self._printed_setup = state.setup_message
            self._line(f"setup: {state.setup_message}")
        if state.setup_done_at is not None and not self._setup_closed:
            self._setup_closed = True
            self._line(f"setup done in {format_duration(state.setup_done_at - state.started_at)}")
        while self._printed_stages < state.completed:
            index = self._printed_stages
            stage = state.stages[index]
            started = state.stage_started_at(index) or state.started_at
            took = format_duration(state.stage_finished_at[index] - started)
            note = state.notes.get(stage.key, "")
            detail = f" ({note})" if note else ""
            self._line(f"{stage.key} done: {stage.artifact.name}{detail} in {took}")
            self._printed_stages += 1

    def close(self, state: RunState, *, succeeded: bool) -> None:
        """Print the outcome and, on failure, where the log is."""
        self.update(state)
        elapsed = format_duration(time.monotonic() - state.started_at)
        if succeeded:
            self._line(f"done in {elapsed}")
            return
        self._line(f"stopped after {elapsed}")
        if state.log_path is not None:
            cause = failure_cause(state.log_path)
            if cause:
                self._line(f"cause: {cause}")
            for line in tail_lines(state.log_path, count=12):
                self._line(f"  | {line}")
            self._line(f"full log: {state.log_path}")

    def _line(self, message: str) -> None:
        print(f"card-framework: {message}", file=self._stream, flush=True)


def short_path(path: Path) -> str:
    """Show ``path`` relative to the working folder when it's inside it."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)
