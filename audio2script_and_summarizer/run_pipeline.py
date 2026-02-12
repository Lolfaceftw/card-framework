import argparse
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Literal, cast

if __package__:
    from .deepseek.stream_events import (
        DEEPSEEK_STREAM_EVENT_PREFIX,
        parse_deepseek_stream_event_line as _parse_deepseek_stream_event_line,
        route_deepseek_stream_event as _route_deepseek_stream_event,
    )
    from .logging_utils import configure_logging
else:
    # Allow `python path/to/run_pipeline.py` by bootstrapping repo root into sys.path.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from audio2script_and_summarizer.deepseek.stream_events import (
        DEEPSEEK_STREAM_EVENT_PREFIX,
        parse_deepseek_stream_event_line as _parse_deepseek_stream_event_line,
        route_deepseek_stream_event as _route_deepseek_stream_event,
    )
    from audio2script_and_summarizer.logging_utils import configure_logging

torch_module: ModuleType | None
try:
    import torch as _torch_module
except ImportError:
    torch_module = None
else:
    torch_module = _torch_module

rich_console: ModuleType | None
rich_layout: ModuleType | None
rich_live: ModuleType | None
rich_panel: ModuleType | None
rich_progress: ModuleType | None
try:
    import rich.console as _rich_console
    import rich.layout as _rich_layout
    import rich.live as _rich_live
    import rich.panel as _rich_panel
    import rich.progress as _rich_progress

    RICH_AVAILABLE = True
except ImportError:
    rich_console = None
    rich_layout = None
    rich_live = None
    rich_panel = None
    rich_progress = None
    RICH_AVAILABLE = False
else:
    rich_console = _rich_console
    rich_layout = _rich_layout
    rich_live = _rich_live
    rich_panel = _rich_panel
    rich_progress = _rich_progress

UI_CONSOLE: Any = rich_console.Console() if rich_console is not None else None
_ACTIVE_DASHBOARD: "_PipelineDashboard | None" = None

DEFAULT_DEVICE: Final[str] = (
    "cuda" if torch_module is not None and torch_module.cuda.is_available() else "cpu"
)
PIPELINE_TOTAL_STAGES: Final[int] = 4
LOG_HISTORY_MAX_LINES: Final[int] = 5000
PROGRESS_PANEL_HEIGHT: Final[int] = 5
CONTROLS_PANEL_HEIGHT: Final[int] = 9
DEFAULT_HEARTBEAT_SECONDS: Final[float] = 5.0
KEYBOARD_POLL_SECONDS: Final[float] = 0.05
STATUS_REFRESH_SECONDS: Final[float] = 1.0
DEFAULT_DEEPSEEK_HARD_CEILING_TOKENS: Final[int] = 64000
STREAM_PANEL_MIN_WIDTH: Final[int] = 28
SKIP_A2S_EXCLUDED_DIRS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "checkpoints",
    }
)

logger = logging.getLogger(__name__)


def _supports_unicode_output() -> bool:
    """Return True when the active terminal encoding supports Unicode output."""
    encoding = (sys.stdout.encoding or "").lower()
    return "utf" in encoding or "65001" in encoding


class _PipelineDashboard:
    """Render split-pane pipeline UI with progress and output logs."""

    def __init__(self, enabled: bool) -> None:
        """Initialize dashboard state.

        Args:
            enabled: Enable rich dashboard rendering when True.
        """
        self.enabled = bool(
            enabled
            and rich_console is not None
            and rich_layout is not None
            and rich_live is not None
            and rich_panel is not None
            and rich_progress is not None
        )
        self._started = False
        self._state_lock = threading.RLock()
        self._render_lock = threading.Lock()
        self._logs: list[str] = []
        self._follow_output = True
        self._scroll_offset = 0
        self._deepseek_stream_active = False
        self._deepseek_stream_lines: list[str] = []
        self._deepseek_stream_partial = ""
        self._deepseek_stream_follow = True
        self._deepseek_stream_scroll_offset = 0
        self._deepseek_stream_phase = ""
        self._deepseek_context_tokens_used: int | None = None
        self._deepseek_context_tokens_limit: int | None = None
        self._deepseek_context_tokens_left: int | None = None
        self._deepseek_context_percent_left: float | None = None
        self._deepseek_context_rollover_count = 0
        self._selected_panel: Literal["output", "deepseek_stream"] = "output"
        self._status_stage = "Pipeline setup"
        self._status_substep = "Initializing dashboard"
        self._status_module = "-"
        self._status_command = "-"
        self._status_model = "-"
        self._status_pid: int | None = None
        self._activity_started_monotonic = time.monotonic()
        self._last_output_monotonic: float | None = None
        self._keyboard_enabled = False
        self._keyboard_status_text = "Keyboard listener not started"
        self._keyboard_thread: threading.Thread | None = None
        self._keyboard_stop_event = threading.Event()
        self._prompt_active = False
        self._prompt_title = ""
        self._prompt_options: list[str] = []
        self._prompt_input_buffer = ""
        self._prompt_default_choice = "1"
        self._prompt_error = ""
        self._prompt_capture_enabled = False
        self._prompt_submitted_event = threading.Event()
        self._prompt_result_choice: int | None = None
        self._status_thread: threading.Thread | None = None
        self._status_stop_event = threading.Event()
        self._detail_active = False
        self._detail_label = ""
        self._detail_percent = 0
        self._detail_started_monotonic: float | None = None
        self._detail_last_update_monotonic: float | None = None
        self._detail_pace_pct_per_sec: float | None = None
        self._detail_eta_seconds: int | None = None
        self._console: Any = None
        self._layout: Any = None
        self._live: Any = None
        self._progress: Any = None
        self._task_id: Any = None
        self._detail_task_id: Any = None

        if not self.enabled:
            return

        console_module = cast(Any, rich_console)
        layout_module = cast(Any, rich_layout)
        live_module = cast(Any, rich_live)
        progress_module = cast(Any, rich_progress)
        spinner_or_marker: Any
        if _supports_unicode_output():
            spinner_or_marker = progress_module.SpinnerColumn()
        else:
            spinner_or_marker = progress_module.TextColumn(">")

        self._console = console_module.Console()
        self._layout = layout_module.Layout()
        self._layout.split_column(
            layout_module.Layout(name="progress", size=PROGRESS_PANEL_HEIGHT),
            layout_module.Layout(name="body", ratio=1, minimum_size=8),
            layout_module.Layout(name="controls", size=CONTROLS_PANEL_HEIGHT),
        )
        self._layout["body"].split_row(
            layout_module.Layout(name="output", ratio=2, minimum_size=8),
            layout_module.Layout(
                name="deepseek_stream",
                ratio=1,
                minimum_size=STREAM_PANEL_MIN_WIDTH,
                visible=False,
            ),
        )
        self._progress = progress_module.Progress(
            spinner_or_marker,
            progress_module.TextColumn("[progress.description]{task.description}"),
            progress_module.BarColumn(bar_width=None),
            progress_module.TextColumn("{task.percentage:>3.0f}%"),
            progress_module.TimeElapsedColumn(),
            progress_module.TimeRemainingColumn(),
            console=self._console,
            transient=False,
        )
        self._task_id = self._progress.add_task(
            "Pipeline: setup",
            total=PIPELINE_TOTAL_STAGES,
        )
        self._detail_task_id = self._progress.add_task(
            "Detail: idle",
            total=100,
            completed=0,
            visible=False,
        )
        self._live = live_module.Live(
            self._layout,
            console=self._console,
            refresh_per_second=8,
            auto_refresh=False,
        )

    def start(self) -> None:
        """Start live rendering."""
        if not self.enabled or self._started:
            return
        self._live.start()
        self._started = True
        self._start_keyboard_listener()
        self._start_status_refresher()
        self._refresh()

    def stop(self) -> None:
        """Stop live rendering."""
        if not self.enabled or not self._started:
            return
        self._stop_status_refresher()
        self._stop_keyboard_listener()
        self._refresh()
        self._live.stop()
        self._started = False

    def log(self, message: str) -> None:
        """Append a message to the output pane."""
        if not self.enabled:
            return
        clean_line = message.rstrip("\r")
        if not clean_line:
            return
        with self._state_lock:
            self._logs.append(clean_line)
            if len(self._logs) > LOG_HISTORY_MAX_LINES:
                overflow = len(self._logs) - LOG_HISTORY_MAX_LINES
                del self._logs[:overflow]
            if self._follow_output:
                self._scroll_offset = 0
            else:
                self._scroll_offset += 1
            self._scroll_offset = min(
                self._scroll_offset, self._max_scroll_offset_locked()
            )
            self._last_output_monotonic = time.monotonic()
        self._refresh()

    def event(self, message: str) -> None:
        """Append a lifecycle event to the output panel."""
        self.log(f"[EVENT] {message}")

    def set_status(
        self,
        *,
        stage_name: str | None = None,
        substep: str | None = None,
        module_name: str | None = None,
        command_display: str | None = None,
        model_info: str | None = None,
        pid: int | None = None,
        reset_elapsed: bool = False,
    ) -> None:
        """Update the status details rendered in the controls panel."""
        if not self.enabled:
            return
        now = time.monotonic()
        with self._state_lock:
            if stage_name is not None:
                self._status_stage = stage_name.strip() or "Working"
            if substep is not None:
                self._status_substep = substep.strip() or "Working"
            if module_name is not None:
                self._status_module = module_name.strip() or "-"
            if command_display is not None:
                self._status_command = command_display.strip() or "-"
            if model_info is not None:
                self._status_model = model_info.strip() or "-"
            self._status_pid = pid
            if reset_elapsed:
                self._activity_started_monotonic = now
            progress_detail = f"{self._status_stage} - {self._status_substep}"
            self._progress.update(
                self._task_id,
                description=f"Pipeline: {self._truncate_for_controls(progress_detail, max_chars=58)}",
            )
        self._refresh()

    def complete_stage(self, label: str) -> None:
        """Advance stage progress with a new status label."""
        if not self.enabled:
            return
        with self._state_lock:
            self._progress.update(
                self._task_id,
                advance=1,
                description=f"Pipeline: {label}",
            )
            if self._detail_task_id is not None:
                self._progress.update(
                    self._detail_task_id,
                    visible=False,
                    completed=0,
                    description="Detail: idle",
                )
            self._detail_active = False
            self._detail_label = ""
            self._detail_percent = 0
            self._detail_started_monotonic = None
            self._detail_last_update_monotonic = None
            self._detail_pace_pct_per_sec = None
            self._detail_eta_seconds = None
            self._status_substep = label
            self._status_pid = None
        self._refresh()

    def start_detail_progress(self, label: str) -> None:
        """Show and initialize the sub-progress bar for active long-running work."""
        if not self.enabled or self._detail_task_id is None:
            return
        clean = label.strip() or "working"
        now = time.monotonic()
        with self._state_lock:
            self._progress.update(
                self._detail_task_id,
                visible=True,
                completed=0,
                total=100,
                description=f"Detail: {clean}",
            )
            self._detail_active = True
            self._detail_label = clean
            self._detail_percent = 0
            self._detail_started_monotonic = now
            self._detail_last_update_monotonic = now
            self._detail_pace_pct_per_sec = None
            self._detail_eta_seconds = None
        self._refresh()

    def update_detail_progress(self, percent_complete: int) -> None:
        """Update sub-progress percentage for active work."""
        if not self.enabled or self._detail_task_id is None:
            return
        bounded = min(100, max(0, int(percent_complete)))
        now = time.monotonic()
        with self._state_lock:
            delta_percent = bounded - self._detail_percent
            last_update = self._detail_last_update_monotonic
            if last_update is not None and delta_percent > 0:
                delta_time = max(0.001, now - last_update)
                instantaneous_pace = delta_percent / delta_time
                if self._detail_pace_pct_per_sec is None:
                    self._detail_pace_pct_per_sec = instantaneous_pace
                else:
                    self._detail_pace_pct_per_sec = (
                        0.7 * self._detail_pace_pct_per_sec + 0.3 * instantaneous_pace
                    )
            self._detail_percent = bounded
            self._detail_last_update_monotonic = now
            if (
                self._detail_pace_pct_per_sec is not None
                and self._detail_pace_pct_per_sec > 0
            ):
                remaining_percent = max(0, 100 - bounded)
                self._detail_eta_seconds = int(
                    round(remaining_percent / self._detail_pace_pct_per_sec)
                )
            else:
                self._detail_eta_seconds = None
            self._progress.update(
                self._detail_task_id,
                visible=True,
                completed=bounded,
                total=100,
            )
        self._refresh()

    def finish_detail_progress(self, status: str = "complete") -> None:
        """Finalize and hide the sub-progress bar."""
        if not self.enabled or self._detail_task_id is None:
            return
        with self._state_lock:
            self._progress.update(
                self._detail_task_id,
                completed=100,
                description=f"Detail: {status.strip() or 'complete'}",
            )
            self._progress.update(
                self._detail_task_id,
                visible=False,
            )
            self._detail_active = False
            self._detail_label = ""
            self._detail_percent = 100
            self._detail_started_monotonic = None
            self._detail_last_update_monotonic = None
            self._detail_pace_pct_per_sec = None
            self._detail_eta_seconds = None
        self._refresh()

    def set_progress_detail(self, detail: str) -> None:
        """Update progress task description without advancing stage count."""
        if not self.enabled:
            return
        clean = detail.strip()
        if not clean:
            return
        with self._state_lock:
            self._progress.update(
                self._task_id,
                description=f"Pipeline: {self._truncate_for_controls(clean, max_chars=58)}",
            )
        self._refresh()

    def open_deepseek_stream_panel(self, model_name: str) -> None:
        """Show and reset the DeepSeek stream panel."""
        if not self.enabled:
            return
        with self._state_lock:
            self._deepseek_stream_active = True
            self._deepseek_stream_lines = []
            self._deepseek_stream_partial = ""
            self._deepseek_stream_follow = True
            self._deepseek_stream_scroll_offset = 0
            self._deepseek_stream_phase = ""
            self._deepseek_context_tokens_used = None
            self._deepseek_context_tokens_limit = None
            self._deepseek_context_tokens_left = None
            self._deepseek_context_percent_left = None
            self._deepseek_context_rollover_count = 0
            self._selected_panel = "output"
            self._logs.append(
                f"[EVENT] DeepSeek stream panel opened (model={model_name.strip() or 'unknown'})."
            )
            if len(self._logs) > LOG_HISTORY_MAX_LINES:
                overflow = len(self._logs) - LOG_HISTORY_MAX_LINES
                del self._logs[:overflow]
        self._refresh()

    def append_deepseek_stream_token(self, *, phase: str, text: str) -> None:
        """Append streamed DeepSeek tokens to the right-side panel."""
        if not self.enabled:
            return
        clean_text = text.replace("\r", "")
        if not clean_text:
            return
        normalized_phase = phase.strip().lower()
        with self._state_lock:
            if not self._deepseek_stream_active:
                return
            phase_heading = ""
            if normalized_phase == "reasoning":
                phase_heading = "[REASONING]"
            elif normalized_phase == "answer":
                phase_heading = "[ANSWER]"
            if phase_heading and normalized_phase != self._deepseek_stream_phase:
                self._deepseek_stream_phase = normalized_phase
                self._deepseek_stream_lines.append(phase_heading)
            added_lines = self._append_stream_text_locked(clean_text)
            if self._deepseek_stream_follow:
                self._deepseek_stream_scroll_offset = 0
            elif added_lines > 0:
                self._deepseek_stream_scroll_offset += added_lines
                self._deepseek_stream_scroll_offset = min(
                    self._deepseek_stream_scroll_offset,
                    self._max_scroll_offset_for_panel_locked("deepseek_stream"),
                )
        self._refresh()

    def update_deepseek_context_usage(
        self,
        *,
        tokens_used: int,
        tokens_limit: int,
        tokens_left: int,
        percent_left: float,
        rollover_count: int = 0,
    ) -> None:
        """Update current context window usage for stream-panel subtitle rendering."""
        if not self.enabled:
            return
        with self._state_lock:
            self._deepseek_context_tokens_used = max(0, int(tokens_used))
            self._deepseek_context_tokens_limit = max(1, int(tokens_limit))
            self._deepseek_context_tokens_left = max(0, int(tokens_left))
            self._deepseek_context_percent_left = max(0.0, min(1.0, float(percent_left)))
            self._deepseek_context_rollover_count = max(0, int(rollover_count))
        self._refresh()

    def close_deepseek_stream_panel(self) -> None:
        """Hide the DeepSeek stream panel after summarization output is ready."""
        if not self.enabled:
            return
        with self._state_lock:
            if not self._deepseek_stream_active:
                return
            self._deepseek_stream_active = False
            self._deepseek_stream_lines = []
            self._deepseek_stream_partial = ""
            self._deepseek_stream_follow = True
            self._deepseek_stream_scroll_offset = 0
            self._deepseek_stream_phase = ""
            self._deepseek_context_tokens_used = None
            self._deepseek_context_tokens_limit = None
            self._deepseek_context_tokens_left = None
            self._deepseek_context_percent_left = None
            self._deepseek_context_rollover_count = 0
            if self._selected_panel == "deepseek_stream":
                self._selected_panel = "output"
            self._logs.append("[EVENT] DeepSeek stream panel closed.")
            if len(self._logs) > LOG_HISTORY_MAX_LINES:
                overflow = len(self._logs) - LOG_HISTORY_MAX_LINES
                del self._logs[:overflow]
        self._refresh()

    def _append_stream_text_locked(self, text: str) -> int:
        """Append text to stream buffer and return count of newly closed lines."""
        combined = f"{self._deepseek_stream_partial}{text}"
        chunks = combined.split("\n")
        self._deepseek_stream_partial = chunks[-1]
        completed_lines = chunks[:-1]
        if completed_lines:
            self._deepseek_stream_lines.extend(completed_lines)
        max_stream_lines = max(1, LOG_HISTORY_MAX_LINES - 1)
        overflow = len(self._deepseek_stream_lines) - max_stream_lines
        if overflow > 0:
            del self._deepseek_stream_lines[:overflow]
        return len(completed_lines)

    def prompt_numeric_choice(
        self,
        *,
        title: str,
        options: list[str],
        default_choice: int = 1,
    ) -> int | None:
        """Capture a numeric selection inside the output panel.

        Interaction model:
        - ``P`` enables typing mode.
        - ``Esc`` exits typing mode.
        - ``Enter`` submits the current typed value, or default when empty.

        Args:
            title: Prompt heading shown in the output panel.
            options: Display strings for numbered options.
            default_choice: Default option index (1-based).

        Returns:
            Selected 1-based option index, or ``None`` if prompt mode is unavailable.
        """
        if not self.enabled or not options:
            return None
        if not self._keyboard_enabled:
            return None

        bounded_default = min(len(options), max(1, default_choice))
        with self._state_lock:
            self._prompt_active = True
            self._prompt_title = title.strip() or "Select an option"
            self._prompt_options = options[:]
            self._prompt_input_buffer = ""
            self._prompt_default_choice = str(bounded_default)
            self._prompt_error = ""
            self._prompt_capture_enabled = False
            self._prompt_result_choice = None
            self._prompt_submitted_event.clear()
            self._follow_output = True
            self._scroll_offset = 0
            self._logs.append(
                (
                    f"[PROMPT] {self._prompt_title} "
                    f"(choices: 1-{len(options)}, default: {bounded_default})"
                )
            )
            if len(self._logs) > LOG_HISTORY_MAX_LINES:
                overflow = len(self._logs) - LOG_HISTORY_MAX_LINES
                del self._logs[:overflow]
        self._refresh()

        while self._started and not self._prompt_submitted_event.wait(0.1):
            continue

        with self._state_lock:
            result = self._prompt_result_choice
            self._prompt_active = False
            self._prompt_title = ""
            self._prompt_options = []
            self._prompt_input_buffer = ""
            self._prompt_default_choice = "1"
            self._prompt_error = ""
            self._prompt_capture_enabled = False
            self._prompt_result_choice = None
            self._prompt_submitted_event.clear()
        self._refresh()
        return result

    def _submit_prompt_choice_locked(self) -> None:
        """Validate and submit the current prompt buffer as a numeric choice."""
        raw_value = self._prompt_input_buffer.strip() or self._prompt_default_choice
        if not raw_value.isdigit():
            self._prompt_error = "Enter a number, then press Enter."
            return

        selected_idx = int(raw_value)
        if selected_idx < 1 or selected_idx > len(self._prompt_options):
            self._prompt_error = (
                f"Choice out of range. Enter 1-{len(self._prompt_options)}."
            )
            return

        self._prompt_result_choice = selected_idx
        self._prompt_submitted_event.set()

    def _render_prompt_block_locked(self) -> str:
        """Render active prompt UI block appended to output panel."""
        if not self._prompt_active:
            return ""

        lines = ["[PROMPT ACTIVE] " + self._prompt_title]
        for idx, option in enumerate(self._prompt_options, start=1):
            marker = " (default)" if str(idx) == self._prompt_default_choice else ""
            lines.append(f"  {idx}. {option}{marker}")
        mode_label = "typing" if self._prompt_capture_enabled else "idle"
        lines.append(f"Input mode: {mode_label} | Press P to type | Enter submit")
        lines.append("Esc exits typing mode.")
        choice_display = self._prompt_input_buffer or self._prompt_default_choice
        lines.append(f"Choice> {choice_display}")
        if self._prompt_error:
            lines.append(f"Error: {self._prompt_error}")
        return "\n".join(lines)

    def _refresh(self) -> None:
        """Refresh dashboard panels."""
        if not self.enabled:
            return
        panel_module = cast(Any, rich_panel)
        # Rich Live rendering can be called from worker, keyboard, and status threads.
        # Serialize updates to avoid redraw contention/flicker during terminal resize.
        with self._render_lock:
            with self._state_lock:
                output_text = self._build_output_text_locked()
                stream_text = self._build_deepseek_stream_text_locked()
                output_mode = "LIVE" if self._follow_output else "SCROLL"
                stream_mode = (
                    "LIVE" if self._deepseek_stream_follow else "SCROLL"
                )
                selected_label = (
                    "DeepSeek Stream"
                    if self._selected_panel == "deepseek_stream"
                    and self._deepseek_stream_active
                    else "Output"
                )
                if self._prompt_active:
                    if self._prompt_capture_enabled:
                        controls_header = "Prompt active | Typing enabled | Enter submit | Esc exit typing"
                    else:
                        controls_header = (
                            "Prompt active | Press P to type | Enter default submit"
                        )
                else:
                    controls_header = (
                        "Tab Switch Panel | Up/Down Scroll | PgUp/PgDn Fast Scroll | "
                        "Home Top | End Follow Live"
                        if self._deepseek_stream_active
                        else "Up/Down Scroll | PgUp/PgDn Fast Scroll | Home Top | End Follow Live"
                    )
                if self._keyboard_enabled:
                    controls_text = (
                        f"{controls_header}\n"
                        f"Stage: {self._status_stage}\n"
                        f"Step: {self._status_substep}\n"
                        f"Module: {self._status_module}\n"
                        f"Command: {self._truncate_for_controls(self._status_command)}\n"
                        f"Model: {self._status_model}\n"
                        f"PID: {self._status_pid if self._status_pid is not None else '-'} | Output: {output_mode}\n"
                        f"Selected panel: {selected_label} | Stream: {stream_mode if self._deepseek_stream_active else 'hidden'}\n"
                        f"Elapsed: {self._elapsed_text_locked()} | Last output: {self._last_output_text_locked()}\n"
                        f"{self._detail_metrics_text_locked()}"
                    )
                else:
                    controls_text = (
                        f"Keys unavailable: {self._keyboard_status_text}\n"
                        f"{controls_header}\n"
                        f"Stage: {self._status_stage}\n"
                        f"Step: {self._status_substep}\n"
                        f"Module: {self._status_module}\n"
                        f"Command: {self._truncate_for_controls(self._status_command)}\n"
                        f"Model: {self._status_model}\n"
                        f"PID: {self._status_pid if self._status_pid is not None else '-'} | Output: {output_mode}\n"
                        f"Selected panel: {selected_label} | Stream: {stream_mode if self._deepseek_stream_active else 'hidden'}\n"
                        f"Elapsed: {self._elapsed_text_locked()} | Last output: {self._last_output_text_locked()}\n"
                        f"{self._detail_metrics_text_locked()}"
                    )
                self._layout["progress"].update(
                    panel_module.Panel(
                        self._progress,
                        title="Progress",
                        border_style="cyan",
                        padding=(0, 1),
                    )
                )
                output_border_style = (
                    "cyan"
                    if self._selected_panel == "output"
                    or not self._deepseek_stream_active
                    else "white"
                )
                self._layout["output"].update(
                    panel_module.Panel(
                        output_text,
                        title=(
                            "Output (Selected)"
                            if self._selected_panel == "output"
                            or not self._deepseek_stream_active
                            else "Output"
                        ),
                        border_style=output_border_style,
                        padding=(0, 1),
                    )
                )
                stream_layout = self._layout["deepseek_stream"]
                stream_layout.visible = self._deepseek_stream_active
                if self._deepseek_stream_active:
                    stream_border_style = (
                        "cyan"
                        if self._selected_panel == "deepseek_stream"
                        else "white"
                    )
                    stream_subtitle = self._build_deepseek_stream_subtitle_locked()
                    stream_layout.update(
                        panel_module.Panel(
                            stream_text,
                            title=(
                                "DeepSeek Stream (Selected)"
                                if self._selected_panel == "deepseek_stream"
                                else "DeepSeek Stream"
                            ),
                            subtitle=stream_subtitle,
                            subtitle_align="left",
                            border_style=stream_border_style,
                            padding=(0, 1),
                        )
                    )
                self._layout["controls"].update(
                    panel_module.Panel(
                        controls_text,
                        title="Controls",
                        border_style="#555555",
                        style="on #2f2f2f",
                        padding=(0, 1),
                    )
                )
            if self._started and self._live is not None:
                self._live.refresh()

    def _elapsed_text_locked(self) -> str:
        """Return elapsed duration since the latest status reset."""
        elapsed_seconds = max(
            0, int(time.monotonic() - self._activity_started_monotonic)
        )
        return self._format_duration(elapsed_seconds)

    def _last_output_text_locked(self) -> str:
        """Return the age of the latest output line."""
        if self._last_output_monotonic is None:
            return "no output yet"
        age_seconds = max(0, int(time.monotonic() - self._last_output_monotonic))
        return f"{self._format_duration(age_seconds)} ago"

    def _detail_metrics_text_locked(self) -> str:
        """Return pacing and ETA text for the active detail progress task."""
        if not self._detail_active:
            return "Detail: idle"
        pace_text = (
            f"{self._detail_pace_pct_per_sec:.2f}%/s"
            if self._detail_pace_pct_per_sec is not None
            else "calculating..."
        )
        eta_text = (
            self._format_duration(self._detail_eta_seconds)
            if self._detail_eta_seconds is not None
            else "estimating..."
        )
        return (
            f"Detail: {self._truncate_for_controls(self._detail_label, max_chars=24)} "
            f"{self._detail_percent}% | Pace: {pace_text} | ETA: {eta_text}"
        )

    @staticmethod
    def _format_duration(total_seconds: int) -> str:
        """Format integer seconds into HH:MM:SS."""
        hours, rem = divmod(total_seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _truncate_for_controls(value: str, max_chars: int = 90) -> str:
        """Truncate long command text to keep controls panel readable."""
        clean = value.strip()
        if len(clean) <= max_chars:
            return clean
        return f"{clean[: max_chars - 3]}..."

    def _build_output_text_locked(self) -> str:
        """Render the visible output viewport from log history and scroll state."""
        if not self._logs and not self._prompt_active:
            return "Waiting for output..."

        view_lines: list[str]
        if self._logs:
            visible_lines = self._visible_line_count_locked()
            max_offset = self._max_scroll_offset_locked()
            if self._follow_output:
                self._scroll_offset = 0
            else:
                self._scroll_offset = min(self._scroll_offset, max_offset)

            end_index = len(self._logs) - self._scroll_offset
            start_index = max(0, end_index - visible_lines)
            view_lines = self._logs[start_index:end_index]
            if start_index > 0:
                view_lines.insert(0, "[...] older output above")
        else:
            view_lines = ["Waiting for output..."]

        if self._prompt_active:
            prompt_block = self._render_prompt_block_locked()
            if prompt_block:
                view_lines.extend(["", prompt_block])

        return "\n".join(view_lines)

    def _build_deepseek_stream_text_locked(self) -> str:
        """Render the visible viewport for the DeepSeek token stream panel."""
        if not self._deepseek_stream_active:
            return "DeepSeek stream inactive."

        stream_lines = self._deepseek_stream_lines[:]
        if self._deepseek_stream_partial:
            stream_lines.append(self._deepseek_stream_partial)
        if not stream_lines:
            return "Waiting for streamed tokens..."

        visible_lines = self._visible_line_count_for_panel_locked("deepseek_stream")
        max_offset = self._max_scroll_offset_for_panel_locked("deepseek_stream")
        if self._deepseek_stream_follow:
            self._deepseek_stream_scroll_offset = 0
        else:
            self._deepseek_stream_scroll_offset = min(
                self._deepseek_stream_scroll_offset, max_offset
            )

        end_index = len(stream_lines) - self._deepseek_stream_scroll_offset
        start_index = max(0, end_index - visible_lines)
        view_lines = stream_lines[start_index:end_index]
        if start_index > 0:
            view_lines.insert(0, "[...] older stream output above")
        return "\n".join(view_lines)

    def _build_deepseek_stream_subtitle_locked(self) -> str:
        """Build subtitle text containing current context window usage."""
        if (
            self._deepseek_context_tokens_used is None
            or self._deepseek_context_tokens_limit is None
            or self._deepseek_context_tokens_left is None
            or self._deepseek_context_percent_left is None
        ):
            return "Ctx: pending"
        return (
            f"Ctx {self._deepseek_context_tokens_used:,}/{self._deepseek_context_tokens_limit:,} "
            f"| Left {self._deepseek_context_tokens_left:,} "
            f"({self._deepseek_context_percent_left * 100:.1f}%) "
            f"| Rollovers {self._deepseek_context_rollover_count}"
        )

    def _visible_line_count_locked(self) -> int:
        """Estimate lines visible inside the currently selected viewport."""
        return self._visible_line_count_for_panel_locked(self._selected_scroll_target_locked())

    def _visible_line_count_for_panel_locked(
        self, panel_name: Literal["output", "deepseek_stream"]
    ) -> int:
        """Estimate visible lines for a given panel name."""
        del panel_name  # Height is shared by both middle-row panels.
        output_height = 0
        if self._layout is not None:
            try:
                output_height = int(self._layout["body"].size)
            except (TypeError, ValueError, KeyError):
                output_height = 0
        if output_height <= 0 and self._console is not None:
            output_height = (
                self._console.size.height
                - PROGRESS_PANEL_HEIGHT
                - CONTROLS_PANEL_HEIGHT
            )
        # Reserve room for panel border and title.
        return max(3, output_height - 2)

    def _max_scroll_offset_locked(self) -> int:
        """Return the maximum valid scroll offset from the log tail."""
        return self._max_scroll_offset_for_panel_locked("output")

    def _max_scroll_offset_for_panel_locked(
        self, panel_name: Literal["output", "deepseek_stream"]
    ) -> int:
        """Return max valid scroll offset for a specific panel."""
        visible_lines = self._visible_line_count_for_panel_locked(panel_name)
        if panel_name == "deepseek_stream":
            stream_line_count = len(self._deepseek_stream_lines)
            if self._deepseek_stream_partial:
                stream_line_count += 1
            return max(0, stream_line_count - visible_lines)
        return max(0, len(self._logs) - visible_lines)

    def _selected_scroll_target_locked(self) -> Literal["output", "deepseek_stream"]:
        """Return the active scroll target based on focus and panel visibility."""
        if self._selected_panel == "deepseek_stream" and self._deepseek_stream_active:
            return "deepseek_stream"
        return "output"

    def _toggle_selected_panel(self) -> None:
        """Toggle selected panel focus between output and stream."""
        if not self.enabled:
            return
        with self._state_lock:
            if not self._deepseek_stream_active:
                self._selected_panel = "output"
            elif self._selected_panel == "output":
                self._selected_panel = "deepseek_stream"
            else:
                self._selected_panel = "output"
        self._refresh()

    def _scroll_up(self, lines: int) -> None:
        """Scroll the selected viewport up by line count."""
        if not self.enabled:
            return
        with self._state_lock:
            target_panel = self._selected_scroll_target_locked()
            step = max(1, lines)
            if target_panel == "deepseek_stream":
                self._deepseek_stream_follow = False
                self._deepseek_stream_scroll_offset = min(
                    self._deepseek_stream_scroll_offset + step,
                    self._max_scroll_offset_for_panel_locked("deepseek_stream"),
                )
            else:
                self._follow_output = False
                self._scroll_offset = min(
                    self._scroll_offset + step,
                    self._max_scroll_offset_for_panel_locked("output"),
                )
        self._refresh()

    def _scroll_down(self, lines: int) -> None:
        """Scroll the selected viewport down by line count."""
        if not self.enabled:
            return
        with self._state_lock:
            target_panel = self._selected_scroll_target_locked()
            step = max(1, lines)
            if target_panel == "deepseek_stream":
                self._deepseek_stream_scroll_offset = max(
                    0, self._deepseek_stream_scroll_offset - step
                )
                if self._deepseek_stream_scroll_offset == 0:
                    self._deepseek_stream_follow = True
            else:
                self._scroll_offset = max(0, self._scroll_offset - step)
                if self._scroll_offset == 0:
                    self._follow_output = True
        self._refresh()

    def _scroll_to_top(self) -> None:
        """Jump to the oldest visible content in the selected panel."""
        if not self.enabled:
            return
        with self._state_lock:
            target_panel = self._selected_scroll_target_locked()
            if target_panel == "deepseek_stream":
                self._deepseek_stream_follow = False
                self._deepseek_stream_scroll_offset = (
                    self._max_scroll_offset_for_panel_locked("deepseek_stream")
                )
            else:
                self._follow_output = False
                self._scroll_offset = self._max_scroll_offset_for_panel_locked(
                    "output"
                )
        self._refresh()

    def _scroll_to_latest(self) -> None:
        """Return selected panel to live-follow mode at latest content."""
        if not self.enabled:
            return
        with self._state_lock:
            target_panel = self._selected_scroll_target_locked()
            if target_panel == "deepseek_stream":
                self._deepseek_stream_follow = True
                self._deepseek_stream_scroll_offset = 0
            else:
                self._follow_output = True
                self._scroll_offset = 0
        self._refresh()

    def _start_keyboard_listener(self) -> None:
        """Start non-blocking keyboard listener for output scrolling."""
        if self._keyboard_thread is not None:
            return
        if sys.platform != "win32":
            self._keyboard_enabled = False
            self._keyboard_status_text = "Windows keyboard capture only"
            return
        try:
            import msvcrt  # type: ignore[attr-defined]
        except ImportError:
            self._keyboard_enabled = False
            self._keyboard_status_text = "msvcrt unavailable"
            return

        self._keyboard_enabled = True
        self._keyboard_status_text = "active"
        self._keyboard_stop_event.clear()
        self._keyboard_thread = threading.Thread(
            target=self._keyboard_loop,
            args=(msvcrt,),
            name="pipeline-dashboard-keys",
            daemon=True,
        )
        self._keyboard_thread.start()

    def _stop_keyboard_listener(self) -> None:
        """Stop keyboard listener thread."""
        self._keyboard_stop_event.set()
        if self._keyboard_thread is not None:
            self._keyboard_thread.join(timeout=0.4)
        self._keyboard_thread = None

    def _start_status_refresher(self) -> None:
        """Start the periodic status refresh thread."""
        if self._status_thread is not None:
            return
        self._status_stop_event.clear()
        self._status_thread = threading.Thread(
            target=self._status_loop,
            name="pipeline-dashboard-status",
            daemon=True,
        )
        self._status_thread.start()

    def _stop_status_refresher(self) -> None:
        """Stop periodic status refresh thread."""
        self._status_stop_event.set()
        if self._status_thread is not None:
            self._status_thread.join(timeout=0.4)
        self._status_thread = None

    def _status_loop(self) -> None:
        """Refresh controls panel at a fixed cadence for live status updates."""
        while not self._status_stop_event.wait(STATUS_REFRESH_SECONDS):
            if not self._started:
                continue
            self._refresh()

    def _keyboard_loop(self, msvcrt: Any) -> None:
        """Handle keyboard controls while the dashboard is active."""
        while not self._keyboard_stop_event.is_set():
            try:
                if not msvcrt.kbhit():
                    time.sleep(KEYBOARD_POLL_SECONDS)
                    continue
                key = msvcrt.getwch()
            except OSError as exc:
                with self._state_lock:
                    self._keyboard_enabled = False
                    self._keyboard_status_text = (
                        f"no input console: {exc.strerror or exc}"
                    )
                self._refresh()
                return

            if key in {"\x00", "\xe0"}:
                special = msvcrt.getwch()
                with self._state_lock:
                    prompt_active = self._prompt_active
                if prompt_active:
                    continue
                if special == "H":
                    self._scroll_up(1)
                elif special == "P":
                    self._scroll_down(1)
                elif special == "I":
                    self._scroll_up(self._visible_line_count_locked())
                elif special == "Q":
                    self._scroll_down(self._visible_line_count_locked())
                elif special == "G":
                    self._scroll_to_top()
                elif special == "O":
                    self._scroll_to_latest()
                continue

            if key == "\t":
                with self._state_lock:
                    prompt_active = self._prompt_active
                    stream_active = self._deepseek_stream_active
                if not prompt_active and stream_active:
                    self._toggle_selected_panel()
                continue

            with self._state_lock:
                if not self._prompt_active:
                    continue

                if key in {"p", "P"}:
                    self._prompt_capture_enabled = True
                    self._prompt_error = ""
                    self._refresh()
                    continue

                if key in {"\r", "\n"}:
                    self._submit_prompt_choice_locked()
                    self._refresh()
                    continue

                if key == "\x1b":
                    self._prompt_capture_enabled = False
                    self._prompt_error = ""
                    self._refresh()
                    continue

                if not self._prompt_capture_enabled:
                    continue

                if key in {"\b", "\x7f"}:
                    self._prompt_input_buffer = self._prompt_input_buffer[:-1]
                    self._prompt_error = ""
                    self._refresh()
                    continue

                if key.isprintable():
                    self._prompt_input_buffer += key
                    self._prompt_error = ""
                    self._refresh()


def _print_info(message: str, use_rich: bool) -> None:
    """Print an informational message."""
    logger.info(message)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[cyan]{message}[/cyan]")
        return
    print(message)


def _print_warning(message: str, use_rich: bool) -> None:
    """Print a warning message."""
    logger.warning(message)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[yellow]{message}[/yellow]")
        return
    print(message)


def _print_error(message: str, use_rich: bool) -> None:
    """Print an error message."""
    logger.error(message)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[bold red]{message}[/bold red]")
        return
    print(message)


def _print_success(message: str, use_rich: bool) -> None:
    """Print a success message."""
    logger.info(message)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log(message)
        return
    if use_rich and UI_CONSOLE is not None:
        UI_CONSOLE.print(f"[bold green]{message}[/bold green]")
        return
    print(message)


def _print_checkpoint(message: str, use_rich: bool) -> None:
    """Print a timestamped status checkpoint."""
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.event(message)
        return
    utc_stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    _print_info(f"[CHECKPOINT {utc_stamp}Z] {message}", use_rich=use_rich)


def _print_stage_banner(title: str, use_rich: bool) -> None:
    """Print a section header for a pipeline stage."""
    logger.info(title)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.log("=" * 50)
        _ACTIVE_DASHBOARD.log(title)
        _ACTIVE_DASHBOARD.log("=" * 50)
        return
    if use_rich and UI_CONSOLE is not None and rich_panel is not None:
        panel_module = rich_panel
        UI_CONSOLE.print(
            panel_module.Panel.fit(f"[bold]{title}[/bold]", border_style="cyan")
        )
        return

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def _resolve_device(requested_device: str) -> str:
    """Return a valid runtime device, falling back to CPU when CUDA is unavailable."""
    normalized = requested_device.strip().lower()
    if normalized.startswith("cuda"):
        if torch_module is None or not torch_module.cuda.is_available():
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            print("[WARN] CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        return requested_device
    return "cpu"


def _find_dotenv_file() -> Path | None:
    """Find a ``.env`` file from CWD upward, then fall back to repo root."""
    current_dir = Path.cwd().resolve()
    for directory in [current_dir, *current_dir.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate

    repo_root_candidate = Path(__file__).resolve().parent.parent / ".env"
    if repo_root_candidate.is_file():
        return repo_root_candidate
    return None


def _load_dotenv_file(dotenv_path: Path) -> int:
    """Load missing environment variables from a dotenv file.

    Args:
        dotenv_path: Path to ``.env`` file.

    Returns:
        Number of environment variables added to ``os.environ``.
    """
    loaded_count = 0
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if key not in os.environ:
            os.environ[key] = value
            loaded_count += 1

    return loaded_count


def _auto_load_dotenv() -> None:
    """Automatically load `.env` values into process environment when available."""
    dotenv_path = _find_dotenv_file()
    if dotenv_path is None:
        return

    loaded_count = _load_dotenv_file(dotenv_path)
    _print_info(
        f"[INFO] Loaded .env from: {dotenv_path} ({loaded_count} vars)", use_rich=False
    )


def _extract_module_name(cmd: list[str]) -> str:
    """Extract ``python -m`` module name from a subprocess command."""
    if "-m" in cmd:
        module_index = cmd.index("-m")
        if module_index + 1 < len(cmd):
            return cmd[module_index + 1]
    if cmd:
        return Path(cmd[0]).name
    return "-"


def _run_stage_command(
    cmd: list[str],
    current_env: dict[str, str],
    use_dashboard: bool,
    stage_name: str,
    heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
    model_info: str = "-",
) -> None:
    """Run a child command and optionally stream output into dashboard logs.

    Args:
        cmd: Subprocess command line.
        current_env: Environment variables for the child process.
        use_dashboard: Stream process output into the split UI when True.
        stage_name: Human-readable stage label for status updates.
        heartbeat_seconds: How often to emit activity heartbeat while output is silent.
        model_info: Model/runtime details shown in the controls panel.

    Raises:
        subprocess.CalledProcessError: When child process exits non-zero.
    """
    command_display = subprocess.list2cmdline(cmd)
    module_name = _extract_module_name(cmd)
    logger.info(
        "Starting stage subprocess",
        extra={
            "component": "run_pipeline",
            "job_id": stage_name,
        },
    )
    logger.debug(
        "Stage launch details: stage=%s module=%s command=%s",
        stage_name,
        module_name,
        command_display,
    )
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.set_status(
            stage_name=stage_name,
            substep="Launching subprocess",
            module_name=module_name,
            command_display=command_display,
            model_info=model_info,
            pid=None,
            reset_elapsed=True,
        )
        _ACTIVE_DASHBOARD.event(f"{stage_name}: launching {module_name}")

    if not use_dashboard:
        started_at = time.monotonic()
        subprocess.run(cmd, check=True, env=current_env)
        elapsed_total = int(time.monotonic() - started_at)
        logger.info("%s completed successfully in %ss", stage_name, elapsed_total)
        return

    child_env = current_env.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess.Popen(
        cmd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.set_status(
            stage_name=stage_name,
            substep="Subprocess running",
            module_name=module_name,
            command_display=command_display,
            model_info=model_info,
            pid=process.pid,
        )
        _ACTIVE_DASHBOARD.event(f"{stage_name}: subprocess started (pid={process.pid})")

    assert process.stdout is not None
    queue_sentinel = object()
    output_queue: queue.Queue[object] = queue.Queue()

    def _read_child_output() -> None:
        for raw_line in process.stdout:
            output_queue.put(raw_line)
        output_queue.put(queue_sentinel)

    reader_thread = threading.Thread(
        target=_read_child_output,
        name=f"pipeline-reader-{stage_name}",
        daemon=True,
    )
    reader_thread.start()

    started_at = time.monotonic()
    next_heartbeat = started_at + max(0.0, heartbeat_seconds)
    output_finished = False

    while True:
        try:
            queued_item = output_queue.get(timeout=0.2)
        except queue.Empty:
            queued_item = None

        if queued_item is queue_sentinel:
            output_finished = True
        elif isinstance(queued_item, str):
            line = queued_item.rstrip("\n")
            if line and _ACTIVE_DASHBOARD is not None:
                logger.debug("%s | %s", stage_name, line)
                stream_event_payload = _parse_deepseek_stream_event_line(line)
                if stream_event_payload is not None:
                    _route_deepseek_stream_event(
                        _ACTIVE_DASHBOARD,
                        stream_event_payload,
                    )
                    next_heartbeat = time.monotonic() + max(0.0, heartbeat_seconds)
                    continue
                handled_status_marker = False
                if line.startswith("[PROGRESS] "):
                    progress_payload = line[len("[PROGRESS] ") :].strip()
                    tokens = progress_payload.split()
                    if len(tokens) >= 2 and tokens[0].lower() == "htdemucs":
                        progress_value = tokens[1].lower()
                        if progress_value == "start":
                            _ACTIVE_DASHBOARD.start_detail_progress(
                                "htdemucs source separation"
                            )
                            _ACTIVE_DASHBOARD.set_status(
                                stage_name=stage_name,
                                substep="Running htdemucs source separation",
                                module_name=module_name,
                                command_display=command_display,
                                model_info=model_info,
                                pid=process.pid,
                            )
                            handled_status_marker = True
                        elif progress_value.isdigit():
                            _ACTIVE_DASHBOARD.update_detail_progress(
                                int(progress_value)
                            )
                            _ACTIVE_DASHBOARD.set_status(
                                stage_name=stage_name,
                                substep=f"htdemucs source separation ({progress_value}%)",
                                module_name=module_name,
                                command_display=command_display,
                                model_info=model_info,
                                pid=process.pid,
                            )
                            handled_status_marker = True
                        elif progress_value in {"done", "complete"}:
                            _ACTIVE_DASHBOARD.update_detail_progress(100)
                            _ACTIVE_DASHBOARD.finish_detail_progress(
                                "htdemucs complete"
                            )
                            handled_status_marker = True
                        elif progress_value == "failed":
                            _ACTIVE_DASHBOARD.finish_detail_progress("htdemucs failed")
                            handled_status_marker = True
                elif line.startswith("[MODEL] "):
                    model_info = line[len("[MODEL] ") :].strip() or model_info
                    _ACTIVE_DASHBOARD.set_status(
                        stage_name=stage_name,
                        substep="Loading model artifacts",
                        module_name=module_name,
                        command_display=command_display,
                        model_info=model_info,
                        pid=process.pid,
                    )
                    _ACTIVE_DASHBOARD.set_progress_detail(
                        f"{stage_name} - Loading model artifacts"
                    )
                    handled_status_marker = True
                elif line.startswith("[STATUS] "):
                    status_text = line[len("[STATUS] ") :].strip()
                    _ACTIVE_DASHBOARD.set_status(
                        stage_name=stage_name,
                        substep=status_text or "Running subprocess",
                        module_name=module_name,
                        command_display=command_display,
                        model_info=model_info,
                        pid=process.pid,
                    )
                    _ACTIVE_DASHBOARD.set_progress_detail(
                        f"{stage_name} - {status_text or 'Running subprocess'}"
                    )
                    handled_status_marker = True
                if not handled_status_marker:
                    _ACTIVE_DASHBOARD.log(line)
                    _ACTIVE_DASHBOARD.set_status(
                        stage_name=stage_name,
                        substep="Streaming subprocess output",
                        module_name=module_name,
                        command_display=command_display,
                        model_info=model_info,
                        pid=process.pid,
                    )
            next_heartbeat = time.monotonic() + max(0.0, heartbeat_seconds)

        now = time.monotonic()
        if heartbeat_seconds > 0 and now >= next_heartbeat:
            elapsed_seconds = int(now - started_at)
            if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
                _ACTIVE_DASHBOARD.set_status(
                    stage_name=stage_name,
                    substep=f"Waiting for subprocess output ({elapsed_seconds}s)",
                    module_name=module_name,
                    command_display=command_display,
                    model_info=model_info,
                    pid=process.pid,
                )
            next_heartbeat = now + heartbeat_seconds

        if output_finished and process.poll() is not None and output_queue.empty():
            break

    reader_thread.join(timeout=0.4)
    return_code = process.wait()
    elapsed_total = int(time.monotonic() - started_at)
    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        _ACTIVE_DASHBOARD.set_status(
            stage_name=stage_name,
            substep=f"Completed with exit code {return_code}",
            module_name=module_name,
            command_display=command_display,
            model_info=model_info,
            pid=None,
        )
        _ACTIVE_DASHBOARD.event(
            f"{stage_name}: completed with exit code {return_code} ({elapsed_total}s)"
        )
    logger.info(
        "%s finished with exit code %s in %ss", stage_name, return_code, elapsed_total
    )

    if return_code != 0:
        logger.error("%s failed with non-zero exit code %s", stage_name, return_code)
        raise subprocess.CalledProcessError(return_code, cmd)


def _count_wav_files(directory: str) -> int:
    """Count ``.wav`` files in a directory.

    Args:
        directory: Directory path.

    Returns:
        Number of ``.wav`` files in ``directory``.
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        return 0
    return sum(1 for _ in directory_path.glob("*.wav"))


def _format_speaker_wpm_summary(
    per_speaker_wpm: dict[str, float], max_items: int = 6
) -> str:
    """Build a compact human-readable per-speaker WPM summary.

    Args:
        per_speaker_wpm: Mapping of speaker labels to WPM values.
        max_items: Max speaker entries to include before truncating.

    Returns:
        Formatted summary string.
    """
    if not per_speaker_wpm:
        return "none"
    ordered_items = sorted(per_speaker_wpm.items())
    visible_items = ordered_items[:max_items]
    summary = ", ".join(f"{speaker}={wpm:.2f}" for speaker, wpm in visible_items)
    hidden_count = len(ordered_items) - len(visible_items)
    if hidden_count > 0:
        return f"{summary}, +{hidden_count} more"
    return summary


def _prompt_for_provider() -> str:
    """Prompt user to select an LLM provider."""
    while True:
        raw = input("Choose LLM provider [openai/deepseek]: ").strip().lower()
        if raw in {"openai", "deepseek"}:
            return raw
        print("Please type 'openai' or 'deepseek'.")


def _prompt_for_target_minutes() -> float:
    """Prompt user for target summary duration in minutes."""
    while True:
        raw = input("Enter target summary duration in minutes: ").strip()
        try:
            minutes = float(raw)
        except ValueError:
            print("Please enter a numeric value (e.g., 5 or 12.5).")
            continue
        if minutes > 0:
            return minutes
        print("Duration must be greater than 0.")


def _discover_transcript_json_files(search_root: Path) -> list[Path]:
    """Discover diarized transcript JSON files under a search root.

    A file is considered a candidate when it can be parsed by
    ``transcript_wpm.load_transcript_segments`` and yields at least one segment.
    This excludes summary outputs and other JSON files that do not include
    diarized timestamped transcript segments.

    Args:
        search_root: Root directory to scan recursively.

    Returns:
        Candidate transcript JSON paths, newest first.
    """
    from audio2script_and_summarizer.transcript_wpm import load_transcript_segments

    if not search_root.exists():
        return []

    candidates: list[Path] = []
    for current_root, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_A2S_EXCLUDED_DIRS]
        for filename in filenames:
            if not filename.lower().endswith(".json"):
                continue
            candidate_path = Path(current_root) / filename
            try:
                if candidate_path.stat().st_size > 20 * 1024 * 1024:
                    continue
            except OSError:
                continue
            try:
                segments = load_transcript_segments(str(candidate_path))
            except Exception:  # noqa: BLE001
                continue
            if segments:
                candidates.append(candidate_path.resolve())

    def _sort_key(path: Path) -> tuple[float, str]:
        try:
            return (path.stat().st_mtime, str(path))
        except OSError:
            return (0.0, str(path))

    candidates.sort(key=_sort_key, reverse=True)
    return candidates


def _prompt_for_transcript_json(search_root: Path, use_rich: bool) -> Path:
    """Prompt the operator to choose a transcript JSON file from discovered options.

    Args:
        search_root: Root directory to scan recursively.
        use_rich: Whether rich UI output is active.

    Returns:
        Selected transcript JSON path.

    Raises:
        FileNotFoundError: No transcript-like JSON files were discovered.
    """
    candidates = _discover_transcript_json_files(search_root)
    if not candidates:
        raise FileNotFoundError(f"No transcript JSON files found under {search_root}.")

    _print_info(
        f"[INFO] --skip-a2s discovered {len(candidates)} transcript JSON candidate(s).",
        use_rich=use_rich,
    )
    display_options: list[str] = []
    for candidate in candidates:
        try:
            display_path = candidate.relative_to(search_root)
        except ValueError:
            display_path = candidate
        display_options.append(str(display_path))

    if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled:
        selected_idx = _ACTIVE_DASHBOARD.prompt_numeric_choice(
            title="Select transcript JSON file",
            options=display_options,
            default_choice=1,
        )
        if selected_idx is not None:
            return candidates[selected_idx - 1]
        _print_warning(
            "[WARN] Dashboard prompt unavailable; falling back to console input.",
            use_rich=use_rich,
        )

    print("Select transcript JSON file:")
    for idx, display_path in enumerate(display_options, start=1):
        print(f"  {idx}. {display_path}")

    while True:
        raw = input(f"Enter choice [1-{len(candidates)}] (default 1): ").strip()
        if raw == "":
            return candidates[0]
        if raw.isdigit():
            selected_idx = int(raw)
            if 1 <= selected_idx <= len(candidates):
                return candidates[selected_idx - 1]
        print(f"Please enter a number between 1 and {len(candidates)}.")


def main() -> int:
    """Run the three-stage Audio2Script, splitter, and summarizer pipeline."""
    parser = argparse.ArgumentParser(description="CARD Audio2Script and Summarizer")
    parser.add_argument("--input", required=False, help="Path to input podcast audio")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help=f"Device to run on (cuda/cpu, default: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--openai-key",
        "--api-key",
        dest="openai_key",
        help="OpenAI API key (alias: --api-key)",
    )
    parser.add_argument(
        "--llm-provider",
        choices=["openai", "deepseek"],
        default=None,
        help="LLM provider to use (openai or deepseek). If omitted, prompt at runtime.",
    )
    parser.add_argument(
        "--target-minutes",
        type=float,
        default=None,
        help="Target summary duration in minutes (prompted if omitted).",
    )
    parser.add_argument(
        "--word-budget-tolerance",
        type=float,
        default=0.05,
        help="Tolerance ratio for word budget (default: 0.05 = +/-5%%)",
    )
    parser.add_argument(
        "--voice-dir",
        default=None,
        help="Directory for speaker samples (default: <input_basename>_voices)",
    )
    parser.add_argument(
        "--skip-a2s",
        action="store_true",
        default=False,
        help=(
            "Skip Stage 1/1.5 Audio2Script processing and jump to DeepSeek "
            "summarization using a selected existing transcript JSON."
        ),
    )
    parser.add_argument(
        "--skip-a2s-search-root",
        default=".",
        help=(
            "Root directory used by --skip-a2s when searching for transcript "
            "JSON files (default: current directory)."
        ),
    )
    parser.add_argument(
        "--deepseek-max-completion-tokens",
        type=int,
        default=DEFAULT_DEEPSEEK_HARD_CEILING_TOKENS,
        help=(
            "Hard output token ceiling forwarded to DeepSeek summarizer "
            f"(default: {DEFAULT_DEEPSEEK_HARD_CEILING_TOKENS})."
        ),
    )
    parser.add_argument(
        "--deepseek-agent-tool-mode",
        choices=["constraints_only", "full_agentic"],
        default="full_agentic",
        help=(
            "DeepSeek summarizer agentic tool profile: constraints_only "
            "or full_agentic (default: full_agentic)."
        ),
    )
    parser.add_argument(
        "--deepseek-agent-read-max-lines",
        type=int,
        default=120,
        help=(
            "Maximum transcript lines returned in each DeepSeek read tool call "
            "(default: 120)."
        ),
    )
    parser.add_argument(
        "--wpm-source",
        choices=["indextts", "transcript"],
        default="indextts",
        help=(
            "Source for Stage 1.75 word-rate estimation: "
            "'indextts' runs voice-cloner calibration (default), "
            "'transcript' computes from diarized transcript timestamps."
        ),
    )
    parser.add_argument(
        "--no-stem",
        action="store_true",
        default=False,
        help="Skip Demucs source separation in diarization stage.",
    )
    parser.add_argument(
        "--show-deprecation-warnings",
        action="store_true",
        default=False,
        help="Show third-party deprecation warnings from diarization dependencies.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Disable progress bars in child pipeline stages.",
    )
    parser.add_argument(
        "--plain-ui",
        action="store_true",
        default=False,
        help="Disable rich terminal UI even when rich is installed.",
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help=(
            "Heartbeat interval for status updates during silent child process periods "
            "(dashboard mode only)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("AUDIO2SCRIPT_LOG_LEVEL", "INFO").upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Console log level propagated to child processes.",
    )
    args = parser.parse_args()
    log_file = configure_logging(
        level=args.log_level,
        component="run_pipeline",
        enable_console=False,
    )
    os.environ["AUDIO2SCRIPT_LOG_LEVEL"] = args.log_level
    use_rich = RICH_AVAILABLE and not args.plain_ui and sys.stdout.isatty()
    if args.heartbeat_seconds < 0:
        _print_error("[ERROR] --heartbeat-seconds must be >= 0.", use_rich=use_rich)
        return 1
    if args.deepseek_max_completion_tokens <= 0:
        _print_error(
            "[ERROR] --deepseek-max-completion-tokens must be > 0.",
            use_rich=use_rich,
        )
        return 1
    if args.deepseek_agent_read_max_lines <= 0:
        _print_error(
            "[ERROR] --deepseek-agent-read-max-lines must be > 0.",
            use_rich=use_rich,
        )
        return 1

    dashboard = _PipelineDashboard(enabled=use_rich)
    global _ACTIVE_DASHBOARD
    _ACTIVE_DASHBOARD = dashboard if dashboard.enabled else None
    dashboard.start()
    _print_info(f"[INFO] File logging enabled: {log_file}", use_rich=use_rich)
    if dashboard.enabled:
        dashboard.set_status(
            stage_name="Pipeline setup",
            substep="Initializing runtime environment",
            module_name="audio2script_and_summarizer.run_pipeline",
            command_display="python -m audio2script_and_summarizer.run_pipeline ...",
            model_info="-",
            pid=None,
            reset_elapsed=True,
        )

    try:
        if not use_rich and not args.plain_ui:
            if not RICH_AVAILABLE:
                _print_info(
                    "[INFO] Rich not installed. Falling back to plain console output.",
                    use_rich=False,
                )
            else:
                _print_info(
                    "[INFO] Interactive terminal not detected. Falling back to plain console output.",
                    use_rich=False,
                )

        _auto_load_dotenv()
        runtime_device = _resolve_device(args.device)
        _print_checkpoint(
            f"Runtime device resolved: {runtime_device}", use_rich=use_rich
        )

        if args.skip_a2s:
            llm_provider = "deepseek"
            if args.llm_provider and args.llm_provider != "deepseek":
                _print_warning(
                    "[WARN] --skip-a2s requires DeepSeek summarizer; overriding --llm-provider to deepseek.",
                    use_rich=use_rich,
                )
        else:
            llm_provider = args.llm_provider or _prompt_for_provider()
        _print_checkpoint(f"LLM provider selected: {llm_provider}", use_rich=use_rich)
        if args.target_minutes is None:
            target_minutes = _prompt_for_target_minutes()
        else:
            target_minutes = args.target_minutes
        if target_minutes <= 0:
            _print_error("[ERROR] Target minutes must be > 0.", use_rich=use_rich)
            return 1

        input_path = os.path.abspath(args.input) if args.input else ""
        if not args.skip_a2s:
            if not input_path:
                _print_error(
                    "[ERROR] --input is required unless --skip-a2s is used.",
                    use_rich=use_rich,
                )
                return 1
            if not os.path.exists(input_path):
                _print_error(
                    f"[ERROR] Input file not found: {input_path}", use_rich=use_rich
                )
                return 1
            _print_checkpoint(f"Input file verified: {input_path}", use_rich=use_rich)

        current_env = os.environ.copy()
        try:
            # Find where python keeps packages
            site_packages = next(p for p in sys.path if "site-packages" in p)
            nvidia_path = os.path.join(site_packages, "nvidia")

            cudnn_lib = os.path.join(nvidia_path, "cudnn", "lib")
            cublas_lib = os.path.join(nvidia_path, "cublas", "lib")

            # If the folders exist, force the system to look there first
            if os.path.exists(cudnn_lib):
                current_ld = current_env.get("LD_LIBRARY_PATH", "")
                libs = [cudnn_lib, cublas_lib]
                current_env["LD_LIBRARY_PATH"] = (
                    os.pathsep.join([*libs, current_ld])
                    if current_ld
                    else os.pathsep.join(libs)
                )
        except Exception as e:
            _print_warning(
                f"[WARN] Could not auto-detect NVIDIA libs (might crash): {e}",
                use_rich=use_rich,
            )

        if args.skip_a2s:
            _print_stage_banner(
                "[SKIP] STAGE 1/1.5: Audio2Script Bypassed", use_rich=use_rich
            )
            _print_checkpoint(
                "Stage 1 and Stage 1.5 skipped via --skip-a2s.",
                use_rich=use_rich,
            )
            dashboard.complete_stage("Stage 1 skipped (--skip-a2s)")
            dashboard.complete_stage("Stage 1.5 skipped (--skip-a2s)")

            search_root = Path(args.skip_a2s_search_root).resolve()
            _print_checkpoint(
                f"Discovering transcript JSON files under: {search_root}",
                use_rich=use_rich,
            )
            try:
                selected_transcript = _prompt_for_transcript_json(
                    search_root=search_root,
                    use_rich=use_rich,
                )
            except FileNotFoundError as exc:
                _print_error(f"[ERROR] {exc}", use_rich=use_rich)
                return 1

            diarization_json = str(selected_transcript)
            _print_success(
                f"[SUCCESS] Selected transcript JSON: {diarization_json}",
                use_rich=use_rich,
            )

            base_name = os.path.splitext(diarization_json)[0]
            voice_dir = args.voice_dir or f"{base_name}_voices"
            voice_sample_count = _count_wav_files(voice_dir)
            if voice_sample_count <= 0:
                _print_error(
                    f"[ERROR] No speaker samples found in voice dir: {voice_dir}",
                    use_rich=use_rich,
                )
                _print_warning(
                    "Pass --voice-dir pointing to an existing <audio>_voices directory.",
                    use_rich=use_rich,
                )
                return 1

            _print_stage_banner(
                "[START] STAGE 1.75: Transcript WPM Derivation",
                use_rich=use_rich,
            )
            _print_checkpoint(
                "Stage 1.75: deriving WPM from selected transcript JSON.",
                use_rich=use_rich,
            )
            try:
                from audio2script_and_summarizer.transcript_wpm import (
                    compute_wpm_from_transcript,
                )

                avg_wpm, per_speaker_wpm = compute_wpm_from_transcript(diarization_json)
            except Exception as exc:  # noqa: BLE001
                _print_error(
                    f"[ERROR] Transcript WPM derivation failed: {exc}",
                    use_rich=use_rich,
                )
                return 1
            dashboard.complete_stage("Stage 1.75 complete")

            _print_info(
                f"[INFO] Stage 1.75 source=transcript; per-speaker WPM: "
                f"{_format_speaker_wpm_summary(per_speaker_wpm)}",
                use_rich=use_rich,
            )
            word_budget = max(1, int(round(avg_wpm * target_minutes)))
            _print_success(
                f"[SUCCESS] WPM source=transcript; calibrated WPM={avg_wpm:.2f}; "
                f"target_minutes={target_minutes:.2f}; word_budget={word_budget}",
                use_rich=use_rich,
            )

            _print_stage_banner("[START] STAGE 2: Summarizer", use_rich=use_rich)
            _print_checkpoint(
                "Stage 2: preparing summarizer subprocess", use_rich=use_rich
            )

            summary_output = f"{base_name}_summary.json"
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                _print_error(
                    "[ERROR] No DeepSeek API key found. Set DEEPSEEK_API_KEY.",
                    use_rich=use_rich,
                )
                return 1

            summarize_cmd = [
                sys.executable,
                "-m",
                "audio2script_and_summarizer.summarizer_deepseek",
                "--transcript",
                diarization_json,
                "--voice-dir",
                voice_dir,
                "--output",
                summary_output,
                "--api-key",
                api_key,
                "--max-completion-tokens",
                str(args.deepseek_max_completion_tokens),
                "--target-minutes",
                str(target_minutes),
                "--avg-wpm",
                f"{avg_wpm:.2f}",
                "--word-budget",
                str(word_budget),
                "--word-budget-tolerance",
                str(args.word_budget_tolerance),
                "--agent-tool-mode",
                args.deepseek_agent_tool_mode,
                "--agent-read-max-lines",
                str(args.deepseek_agent_read_max_lines),
            ]
            try:
                _run_stage_command(
                    cmd=summarize_cmd,
                    current_env=current_env,
                    use_dashboard=dashboard.enabled,
                    stage_name="Stage 2 (Summarizer)",
                    heartbeat_seconds=args.heartbeat_seconds,
                    model_info=(
                        "Provider: deepseek | Model: auto(reasoner/chat) "
                        f"| Max output tokens: {args.deepseek_max_completion_tokens}"
                    ),
                )
            except subprocess.CalledProcessError as e:
                _print_error(
                    f"[ERROR] Stage 2 crashed with code {e.returncode}",
                    use_rich=use_rich,
                )
                return 1

            dashboard.complete_stage("Stage 2 complete")
            _print_stage_banner("[DONE] PIPELINE COMPLETE", use_rich=use_rich)
            _print_success(f"Summary saved to: {summary_output}", use_rich=use_rich)
            return 0

        # ==========================================
        # STAGE 1: Diarization
        # ==========================================
        _print_stage_banner("[START] STAGE 1: Audio2Script", use_rich=use_rich)
        _print_checkpoint(
            "Stage 1: preparing diarization subprocess", use_rich=use_rich
        )

        try:
            diarize_cmd = [
                sys.executable,
                "-m",
                "audio2script_and_summarizer.diarize",
                "-a",
                input_path,
                "--device",
                runtime_device,
                "--batch-size",
                "2" if runtime_device.lower().startswith("cuda") else "1",
            ]
            if args.no_stem:
                diarize_cmd.append("--no-stem")
            if args.show_deprecation_warnings:
                diarize_cmd.append("--show-deprecation-warnings")
            if args.no_progress or dashboard.enabled:
                diarize_cmd.append("--no-progress")
            if dashboard.enabled:
                _print_info(
                    "[INFO] Stage 1 child progress bars disabled; rendering consolidated progress panel.",
                    use_rich=use_rich,
                )

            _run_stage_command(
                cmd=diarize_cmd,
                current_env=current_env,
                use_dashboard=dashboard.enabled,
                stage_name="Stage 1 (Audio2Script)",
                heartbeat_seconds=args.heartbeat_seconds,
                model_info=f"Diarizer: pyannote | Whisper: medium.en | Device: {runtime_device}",
            )
        except subprocess.CalledProcessError as e:
            _print_error(
                f"[ERROR] Stage 1 crashed with code {e.returncode}", use_rich=use_rich
            )
            _print_warning(
                "Tip: If code is -6 or -11, it's a library/driver mismatch.",
                use_rich=use_rich,
            )
            return 1
        dashboard.complete_stage("Stage 1 complete")

        base_name = os.path.splitext(input_path)[0]
        diarization_json = f"{base_name}.json"
        voice_dir = args.voice_dir or f"{base_name}_voices"

        if not os.path.exists(diarization_json):
            _print_error(
                f"[ERROR] Expected output not found: {diarization_json}",
                use_rich=use_rich,
            )
            _print_warning(
                "Did you update diarize.py to export JSON?", use_rich=use_rich
            )
            return 1

        _print_success(
            f"[SUCCESS] Stage 1 Complete. Output: {diarization_json}",
            use_rich=use_rich,
        )

        # ==========================================
        # STAGE 1.5: Audio Splitting
        # ==========================================
        _print_stage_banner("[START] STAGE 1.5: Audio Splitting", use_rich=use_rich)
        _print_checkpoint(
            "Stage 1.5: preparing audio splitting subprocess", use_rich=use_rich
        )

        try:
            splitter_cmd = [
                sys.executable,
                "-m",
                "audio2script_and_summarizer.audio_splitter",
                "--audio",
                input_path,
                "--json",
                diarization_json,
                "--output-dir",
                voice_dir,
            ]
            _run_stage_command(
                cmd=splitter_cmd,
                current_env=current_env,
                use_dashboard=dashboard.enabled,
                stage_name="Stage 1.5 (Audio Splitting)",
                heartbeat_seconds=args.heartbeat_seconds,
                model_info="Module: audio_splitter | Action: extract speaker samples",
            )
        except subprocess.CalledProcessError as e:
            _print_error(
                f"[ERROR] Stage 1.5 crashed with code {e.returncode}",
                use_rich=use_rich,
            )
            return 1
        dashboard.complete_stage("Stage 1.5 complete")

        voice_sample_count = _count_wav_files(voice_dir)
        if voice_sample_count <= 0:
            _print_error(
                f"[ERROR] No speaker samples were generated in: {voice_dir}",
                use_rich=use_rich,
            )
            return 1
        _print_success(
            f"[SUCCESS] Stage 1.5 Complete. Generated {voice_sample_count} speaker sample(s) in {voice_dir}",
            use_rich=use_rich,
        )

        # ==========================================
        # STAGE 1.75: Voice Cloner WPM Calibration
        # ==========================================
        _print_stage_banner(
            "[START] STAGE 1.75: Voice Cloner WPM Calibration", use_rich=use_rich
        )
        _print_checkpoint(
            "Stage 1.75: calibrating voice WPM (this can take a while)",
            use_rich=use_rich,
        )
        _print_checkpoint(
            f"Stage 1.75: WPM source selected: {args.wpm_source}",
            use_rich=use_rich,
        )

        stage_175_name = "Stage 1.75 (WPM Calibration)"
        stage_175_module = (
            "audio2script_and_summarizer.wpm_calibration"
            if args.wpm_source == "indextts"
            else "audio2script_and_summarizer.transcript_wpm"
        )
        stage_175_command = (
            "calibrate_voice_wpm(...)"
            if args.wpm_source == "indextts"
            else "compute_wpm_from_transcript(...)"
        )
        stage_175_model_info = (
            "IndexTTS2 checkpoint-based voice calibration"
            if args.wpm_source == "indextts"
            else "Diarized transcript timestamp-based WPM derivation"
        )
        if dashboard.enabled:
            dashboard.set_status(
                stage_name=stage_175_name,
                substep="Running in-process calibration",
                module_name=stage_175_module,
                command_display=stage_175_command,
                model_info=stage_175_model_info,
                pid=None,
                reset_elapsed=True,
            )
            detail_label = (
                "IndexTTS2 WPM calibration"
                if args.wpm_source == "indextts"
                else "Transcript WPM derivation"
            )
            dashboard.start_detail_progress(detail_label)
            dashboard.update_detail_progress(1)

        per_speaker_wpm: dict[str, float]
        try:
            if args.wpm_source == "indextts":
                from audio2script_and_summarizer.wpm_calibration import (
                    CalibrationEvent,
                    calibrate_voice_wpm,
                )

                repo_root = Path(__file__).resolve().parent.parent
                cfg_path = (
                    repo_root
                    / "voice-cloner-and-interjector"
                    / "checkpoints"
                    / "config.yaml"
                )
                model_dir = repo_root / "voice-cloner-and-interjector" / "checkpoints"

                def _on_calibration_progress(event: CalibrationEvent) -> None:
                    """Reflect IndexTTS calibration progress in the dashboard."""
                    if not dashboard.enabled:
                        return
                    if event.event_type == "model_init_started":
                        dashboard.set_status(
                            stage_name=stage_175_name,
                            substep="Initializing IndexTTS2 artifacts",
                            module_name=stage_175_module,
                            command_display=stage_175_command,
                            model_info=stage_175_model_info,
                            pid=None,
                        )
                        dashboard.update_detail_progress(5)
                        return
                    if event.event_type == "model_init_completed":
                        speaker_total = max(1, event.speaker_count or 1)
                        dashboard.set_status(
                            stage_name=stage_175_name,
                            substep=f"Model ready; calibrating {speaker_total} speaker sample(s)",
                            module_name=stage_175_module,
                            command_display=stage_175_command,
                            model_info=stage_175_model_info,
                            pid=None,
                        )
                        dashboard.update_detail_progress(12)
                        return
                    if event.event_type == "speaker_started":
                        speaker_total = max(1, event.speaker_count or 1)
                        speaker_index = max(1, event.speaker_index or 1)
                        progress_floor = 12
                        progress_span = 84
                        progress = progress_floor + int(
                            ((speaker_index - 1) / speaker_total) * progress_span
                        )
                        speaker_display = (
                            event.speaker_name or f"speaker_{speaker_index}"
                        )
                        dashboard.set_status(
                            stage_name=stage_175_name,
                            substep=f"Calibrating {speaker_display} ({speaker_index}/{speaker_total})",
                            module_name=stage_175_module,
                            command_display=stage_175_command,
                            model_info=stage_175_model_info,
                            pid=None,
                        )
                        dashboard.update_detail_progress(progress)
                        return
                    if event.event_type == "speaker_completed":
                        speaker_total = max(1, event.speaker_count or 1)
                        speaker_index = max(1, event.speaker_index or 1)
                        progress_floor = 12
                        progress_span = 84
                        progress = progress_floor + int(
                            (speaker_index / speaker_total) * progress_span
                        )
                        speaker_display = (
                            event.speaker_name or f"speaker_{speaker_index}"
                        )
                        speaker_wpm = event.speaker_wpm or 0.0
                        dashboard.set_status(
                            stage_name=stage_175_name,
                            substep=f"Calibrated {speaker_display} ({speaker_wpm:.2f} WPM)",
                            module_name=stage_175_module,
                            command_display=stage_175_command,
                            model_info=stage_175_model_info,
                            pid=None,
                        )
                        dashboard.update_detail_progress(progress)
                        return
                    if event.event_type == "calibration_completed":
                        avg_wpm_local = event.average_wpm or 0.0
                        dashboard.set_status(
                            stage_name=stage_175_name,
                            substep=f"Calibration complete (avg {avg_wpm_local:.2f} WPM)",
                            module_name=stage_175_module,
                            command_display=stage_175_command,
                            model_info=stage_175_model_info,
                            pid=None,
                        )
                        dashboard.update_detail_progress(100)

                avg_wpm, per_speaker_wpm = calibrate_voice_wpm(
                    voice_dir=voice_dir,
                    device=runtime_device,
                    cfg_path=str(cfg_path),
                    model_dir=str(model_dir),
                    progress_cb=_on_calibration_progress if dashboard.enabled else None,
                )
            else:
                from audio2script_and_summarizer.transcript_wpm import (
                    compute_wpm_from_transcript,
                )

                if dashboard.enabled:
                    dashboard.set_status(
                        stage_name=stage_175_name,
                        substep="Parsing diarized transcript segments",
                        module_name=stage_175_module,
                        command_display=stage_175_command,
                        model_info=stage_175_model_info,
                        pid=None,
                    )
                    dashboard.update_detail_progress(40)
                avg_wpm, per_speaker_wpm = compute_wpm_from_transcript(diarization_json)
                if dashboard.enabled:
                    dashboard.set_status(
                        stage_name=stage_175_name,
                        substep=f"Derived transcript WPM (avg {avg_wpm:.2f})",
                        module_name=stage_175_module,
                        command_display=stage_175_command,
                        model_info=stage_175_model_info,
                        pid=None,
                    )
                    dashboard.update_detail_progress(100)
        except Exception as exc:  # noqa: BLE001
            _print_error(f"[ERROR] WPM calibration failed: {exc}", use_rich=use_rich)
            return 1

        dashboard.complete_stage("Stage 1.75 complete")

        _print_info(
            f"[INFO] Stage 1.75 source={args.wpm_source}; per-speaker WPM: "
            f"{_format_speaker_wpm_summary(per_speaker_wpm)}",
            use_rich=use_rich,
        )

        word_budget = max(1, int(round(avg_wpm * target_minutes)))
        _print_success(
            f"[SUCCESS] WPM source={args.wpm_source}; calibrated WPM={avg_wpm:.2f}; "
            f"target_minutes={target_minutes:.2f}; word_budget={word_budget}",
            use_rich=use_rich,
        )

        # ==========================================
        # STAGE 2: Summarizer
        # ==========================================
        _print_stage_banner("[START] STAGE 2: Summarizer", use_rich=use_rich)
        _print_checkpoint("Stage 2: preparing summarizer subprocess", use_rich=use_rich)

        summary_output = f"{base_name}_summary.json"

        try:
            if llm_provider == "openai":
                api_key = (
                    args.openai_key
                    or os.environ.get("OPENAI_API_KEY")
                    or os.environ.get("LLM_API_KEY")
                    or os.environ.get("GEMINI_API_KEY")
                )
                if not api_key:
                    _print_error(
                        "[ERROR] No OpenAI API key found. Use --openai-key/--api-key or set OPENAI_API_KEY.",
                        use_rich=use_rich,
                    )
                    return 1
                summarize_cmd = [
                    sys.executable,
                    "-m",
                    "audio2script_and_summarizer.summarizer",
                    "--transcript",
                    diarization_json,
                    "--voice-dir",
                    voice_dir,
                    "--output",
                    summary_output,
                    "--api-key",
                    api_key,
                    "--target-minutes",
                    str(target_minutes),
                    "--avg-wpm",
                    f"{avg_wpm:.2f}",
                    "--word-budget",
                    str(word_budget),
                    "--word-budget-tolerance",
                    str(args.word_budget_tolerance),
                ]
                summary_model_info = "Provider: openai | Model: gpt-4o-2024-08-06"
            else:
                api_key = os.environ.get("DEEPSEEK_API_KEY")
                if not api_key:
                    _print_error(
                        "[ERROR] No DeepSeek API key found. Set DEEPSEEK_API_KEY.",
                        use_rich=use_rich,
                    )
                    return 1
                summarize_cmd = [
                    sys.executable,
                    "-m",
                    "audio2script_and_summarizer.summarizer_deepseek",
                    "--transcript",
                    diarization_json,
                    "--voice-dir",
                    voice_dir,
                    "--output",
                    summary_output,
                    "--api-key",
                    api_key,
                    "--max-completion-tokens",
                    str(args.deepseek_max_completion_tokens),
                    "--target-minutes",
                    str(target_minutes),
                    "--avg-wpm",
                    f"{avg_wpm:.2f}",
                    "--word-budget",
                    str(word_budget),
                    "--word-budget-tolerance",
                    str(args.word_budget_tolerance),
                    "--agent-tool-mode",
                    args.deepseek_agent_tool_mode,
                    "--agent-read-max-lines",
                    str(args.deepseek_agent_read_max_lines),
                ]
                summary_model_info = (
                    "Provider: deepseek | Model: auto(reasoner/chat) "
                    f"| Max output tokens: {args.deepseek_max_completion_tokens}"
                )
            _run_stage_command(
                cmd=summarize_cmd,
                current_env=current_env,
                use_dashboard=dashboard.enabled,
                stage_name="Stage 2 (Summarizer)",
                heartbeat_seconds=args.heartbeat_seconds,
                model_info=summary_model_info,
            )
        except subprocess.CalledProcessError as e:
            _print_error(
                f"[ERROR] Stage 2 crashed with code {e.returncode}", use_rich=use_rich
            )
            return 1
        dashboard.complete_stage("Stage 2 complete")

        _print_stage_banner("[DONE] PIPELINE COMPLETE", use_rich=use_rich)
        _print_success(f"Summary saved to: {summary_output}", use_rich=use_rich)
        return 0
    finally:
        dashboard.stop()
        _ACTIVE_DASHBOARD = None


if __name__ == "__main__":
    raise SystemExit(main())
