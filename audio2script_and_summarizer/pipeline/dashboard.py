import contextlib
import io
import logging
import sys
import threading
import time
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Final, Iterator, Literal, cast

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
rich_text: ModuleType | None
try:
    import rich.console as _rich_console
    import rich.layout as _rich_layout
    import rich.live as _rich_live
    import rich.panel as _rich_panel
    import rich.progress as _rich_progress
    import rich.text as _rich_text

    RICH_AVAILABLE = True
except ImportError:
    rich_console = None
    rich_layout = None
    rich_live = None
    rich_panel = None
    rich_progress = None
    rich_text = None
    RICH_AVAILABLE = False
else:
    rich_console = _rich_console
    rich_layout = _rich_layout
    rich_live = _rich_live
    rich_panel = _rich_panel
    rich_progress = _rich_progress
    rich_text = _rich_text

UI_CONSOLE: Any = rich_console.Console() if rich_console is not None else None
_ACTIVE_DASHBOARD: "_PipelineDashboard | None" = None

DEFAULT_DEVICE: Final[str] = (
    "cuda" if torch_module is not None and torch_module.cuda.is_available() else "cpu"
)
PIPELINE_TOTAL_STAGES: Final[int] = 5
LOG_HISTORY_MAX_LINES: Final[int] = 5000
PROGRESS_PANEL_HEIGHT: Final[int] = 5
CONTROLS_PANEL_HEIGHT: Final[int] = 9
DEFAULT_HEARTBEAT_SECONDS: Final[float] = 5.0
DEFAULT_DURATION_TOLERANCE_SECONDS: Final[float] = 3.0
DEFAULT_MAX_DURATION_CORRECTION_PASSES: Final[int] = 1
KEYBOARD_POLL_SECONDS: Final[float] = 0.05
STATUS_REFRESH_SECONDS: Final[float] = 1.0
REFRESH_THROTTLE_SECONDS: Final[float] = 0.05
PROMPT_ENTER_GRACE_SECONDS: Final[float] = 0.25
DEFAULT_DEEPSEEK_HARD_CEILING_TOKENS: Final[int] = 64000
DEFAULT_DEEPSEEK_AGENT_MAX_TOOL_ROUNDS: Final[int] = 0
DEFAULT_DEEPSEEK_AGENT_LOOP_EXHAUSTION_POLICY: Final[str] = "auto_salvage"
DEFAULT_DEEPSEEK_BUDGET_FAILURE_POLICY: Final[str] = "degraded_success"
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


@dataclass(slots=True, frozen=True)
class _DashboardTheme:
    """Color and animation settings for Rich terminal dashboard rendering."""

    progress_border_style: str
    progress_spinner_style: str
    progress_track_style: str
    progress_complete_style: str
    progress_finished_style: str
    progress_pulse_style: str
    progress_percent_style: str
    output_border_style_selected: str
    output_border_style_idle: str
    stream_border_style_selected: str
    stream_border_style_idle: str
    controls_border_style: str
    controls_panel_style: str
    controls_text_style: str
    controls_secondary_text_style: str
    output_text_style: str
    output_event_style: str
    output_info_style: str
    output_warning_style: str
    output_error_style: str
    output_deepseek_status_style: str
    output_deepseek_tool_style: str
    output_indextts_style: str
    output_prompt_style: str
    output_hint_style: str
    stream_text_style: str
    stream_meta_style: str
    stream_reasoning_header_style: str
    stream_answer_header_style: str
    stream_reasoning_text_style: str
    stream_answer_text_style: str
    subtitle_ok_style: str
    subtitle_warn_style: str
    subtitle_critical_style: str
    live_badge_style: str
    unicode_heartbeat_frames: tuple[str, ...]
    ascii_heartbeat_frames: tuple[str, ...]


DEFAULT_DASHBOARD_THEME: Final[_DashboardTheme] = _DashboardTheme(
    progress_border_style="#20c7b3",
    progress_spinner_style="#67f7e8 bold",
    progress_track_style="#1b3d42",
    progress_complete_style="#2ee6d2",
    progress_finished_style="#64ffda",
    progress_pulse_style="#00d4ff",
    progress_percent_style="#9ef6f2 bold",
    output_border_style_selected="#31e0cf",
    output_border_style_idle="#2d4b50",
    stream_border_style_selected="#00d2ff",
    stream_border_style_idle="#2d4b50",
    controls_border_style="#25c8ba",
    controls_panel_style="on #102126",
    controls_text_style="#ddfff9",
    controls_secondary_text_style="#9dc8c4",
    output_text_style="#d4f4f1",
    output_event_style="#62f3ff bold",
    output_info_style="#9ad0ff",
    output_warning_style="#ffd166 bold",
    output_error_style="#ff7b7b bold",
    output_deepseek_status_style="#8cebd9",
    output_deepseek_tool_style="#74b9ff bold",
    output_indextts_style="#7ad3ff",
    output_prompt_style="#ffde7a bold",
    output_hint_style="#8eb6b2",
    stream_text_style="#d8fbf7",
    stream_meta_style="#8eb6b2",
    stream_reasoning_header_style="#8cc9df bold",
    stream_answer_header_style="#8effb3 bold",
    stream_reasoning_text_style="#aed4e0",
    stream_answer_text_style="#c8ffe0",
    subtitle_ok_style="#7dffb2 bold",
    subtitle_warn_style="#ffd166 bold",
    subtitle_critical_style="#ff7b7b bold",
    live_badge_style="#67f7e8 bold",
    unicode_heartbeat_frames=("●", "◐", "○", "◑"),
    ascii_heartbeat_frames=("*", "+", ".", "+"),
)


class _DashboardOutputCaptureStream(io.TextIOBase):
    """Capture text output and forward line-delimited content to a callback."""

    def __init__(self, line_callback: Callable[[str], None]) -> None:
        """Initialize capture stream.

        Args:
            line_callback: Callback invoked for each completed output line.
        """
        super().__init__()
        self._line_callback = line_callback
        self._buffer = ""

    def writable(self) -> bool:
        """Return True because this object accepts text writes."""
        return True

    def write(self, text: str) -> int:
        """Buffer writes and emit one callback call per line."""
        if not text:
            return 0
        # Normalize carriage returns used by progress bars into new lines.
        self._buffer += text.replace("\r", "\n")
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            cleaned = line.strip()
            if cleaned:
                self._line_callback(cleaned)
        return len(text)

    def flush(self) -> None:
        """Emit any trailing buffered content."""
        trailing = self._buffer.strip()
        if trailing:
            self._line_callback(trailing)
        self._buffer = ""


INDEXTTS_RUNTIME_HINTS: Final[tuple[str, ...]] = (
    "weights restored from",
    "semantic_codec",
    "s2mel",
    "campplus",
    "bigvgan",
    "textnormalizer loaded",
    "bpe model loaded",
    "starting inference",
    "emotion vectors",
    "removing weight norm",
    "past_key_values",
)


def _looks_like_indextts_runtime_line(line: str) -> bool:
    """Return True when a runtime line likely originated from IndexTTS2."""
    normalized = line.strip().lower()
    if not normalized:
        return False
    if normalized.startswith(">>"):
        return True
    return any(hint in normalized for hint in INDEXTTS_RUNTIME_HINTS)


def _with_indextts_prefix(line: str, *, severity_prefix: str | None = None) -> str:
    """Attach an IndexTTS2 marker and optional severity prefix to a line."""
    clean_line = line.strip()
    if not clean_line:
        return ""
    if "[INDEXTTS2]" in clean_line:
        return clean_line
    if severity_prefix:
        return f"{severity_prefix} [INDEXTTS2] {clean_line}"
    return f"[INDEXTTS2] {clean_line}"


def _format_runtime_output_line_for_dashboard(
    line: str,
    *,
    prefer_indextts_tag: bool,
) -> str:
    """Normalize one captured runtime line before writing to dashboard output."""
    clean_line = line.strip()
    if not clean_line:
        return ""
    is_indextts_line = prefer_indextts_tag or _looks_like_indextts_runtime_line(
        clean_line
    )
    if not is_indextts_line:
        return clean_line

    lowered = clean_line.lower()
    if lowered.startswith("traceback"):
        return _with_indextts_prefix(clean_line, severity_prefix="[ERROR]")
    if (
        "runtimeerror(" in lowered
        or "failed to " in lowered
        or "falling back to" in lowered
        or "deprecated" in lowered
    ):
        return _with_indextts_prefix(clean_line, severity_prefix="[WARNING]")
    return _with_indextts_prefix(clean_line)


@contextlib.contextmanager
def _capture_stage175_output_lines(enabled: bool) -> Iterator[None]:
    """Capture Stage 1.75 stdout/stderr into dashboard logs when enabled."""
    if not enabled:
        yield
        return

    def _on_line(line: str) -> None:
        logger.debug("Stage 1.75 runtime | %s", line)
        formatted_line = _format_runtime_output_line_for_dashboard(
            line,
            prefer_indextts_tag=True,
        )
        if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled and formatted_line:
            _ACTIVE_DASHBOARD.log(formatted_line)

    capture_stream = _DashboardOutputCaptureStream(_on_line)
    with (
        contextlib.redirect_stdout(capture_stream),
        contextlib.redirect_stderr(capture_stream),
    ):
        try:
            yield
        finally:
            capture_stream.flush()


@contextlib.contextmanager
def _capture_stage3_output_lines(enabled: bool) -> Iterator[None]:
    """Capture Stage 3 stdout/stderr into dashboard logs when enabled."""
    if not enabled:
        yield
        return

    def _on_line(line: str) -> None:
        logger.debug("Stage 3 runtime | %s", line)
        formatted_line = _format_runtime_output_line_for_dashboard(
            line,
            prefer_indextts_tag=False,
        )
        if _ACTIVE_DASHBOARD is not None and _ACTIVE_DASHBOARD.enabled and formatted_line:
            _ACTIVE_DASHBOARD.log(formatted_line)

    capture_stream = _DashboardOutputCaptureStream(_on_line)
    with (
        contextlib.redirect_stdout(capture_stream),
        contextlib.redirect_stderr(capture_stream),
    ):
        try:
            yield
        finally:
            capture_stream.flush()


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
        self._theme = DEFAULT_DASHBOARD_THEME
        self.enabled = bool(
            enabled
            and rich_console is not None
            and rich_layout is not None
            and rich_live is not None
            and rich_panel is not None
            and rich_progress is not None
            and rich_text is not None
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
        self._last_refresh_monotonic = 0.0
        self._refresh_pending = False
        self._pending_prompt_enter_monotonic: float | None = None
        self._prompt_opened_monotonic: float | None = None

        if not self.enabled:
            return

        console_module = cast(Any, rich_console)
        layout_module = cast(Any, rich_layout)
        live_module = cast(Any, rich_live)
        progress_module = cast(Any, rich_progress)
        spinner_or_marker: Any
        if _supports_unicode_output():
            spinner_or_marker = progress_module.SpinnerColumn(
                style=self._theme.progress_spinner_style
            )
        else:
            spinner_or_marker = progress_module.TextColumn(
                ">",
                style=self._theme.progress_spinner_style,
            )

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
            progress_module.TextColumn(
                "[progress.description]{task.description}",
                style=self._theme.controls_text_style,
            ),
            progress_module.BarColumn(
                bar_width=None,
                style=self._theme.progress_track_style,
                complete_style=self._theme.progress_complete_style,
                finished_style=self._theme.progress_finished_style,
                pulse_style=self._theme.progress_pulse_style,
            ),
            progress_module.TextColumn(
                "{task.percentage:>3.0f}%",
                style=self._theme.progress_percent_style,
            ),
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
            self._deepseek_context_percent_left = max(
                0.0, min(1.0, float(percent_left))
            )
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
        now = time.monotonic()
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
            self._prompt_opened_monotonic = now
            self._logs.append(
                (
                    f"[PROMPT] {self._prompt_title} "
                    f"(choices: 1-{len(options)}, default: {bounded_default})"
                )
            )
            if len(self._logs) > LOG_HISTORY_MAX_LINES:
                overflow = len(self._logs) - LOG_HISTORY_MAX_LINES
                del self._logs[:overflow]
        self._request_refresh(force=True)

        with self._state_lock:
            pending_enter = self._pending_prompt_enter_monotonic
            should_consume_pending_enter = (
                pending_enter is not None
                and (now - pending_enter) <= PROMPT_ENTER_GRACE_SECONDS
            )
            if should_consume_pending_enter:
                self._prompt_result_choice = bounded_default
                self._prompt_submitted_event.set()
            self._pending_prompt_enter_monotonic = None

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
            self._prompt_opened_monotonic = None
        self._request_refresh(force=True)
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

    def _request_refresh(self, *, force: bool = False) -> None:
        """Throttle non-forced refresh requests to reduce redraw churn."""
        if not self.enabled:
            return
        now = time.monotonic()
        elapsed = now - self._last_refresh_monotonic
        if force or elapsed >= REFRESH_THROTTLE_SECONDS:
            self._refresh_pending = False
            self._last_refresh_monotonic = now
            self._refresh()
            return
        self._refresh_pending = True

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
                output_renderable = self._build_output_renderable_locked(output_text)
                stream_renderable = self._build_deepseek_stream_renderable_locked(
                    stream_text
                )
                output_mode = "LIVE" if self._follow_output else "SCROLL"
                stream_mode = "LIVE" if self._deepseek_stream_follow else "SCROLL"
                selected_label = (
                    "DeepSeek Stream"
                    if self._selected_panel == "deepseek_stream"
                    and self._deepseek_stream_active
                    else "Output"
                )
                heartbeat = self._heartbeat_symbol_locked()
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
                        f"PID: {self._status_pid if self._status_pid is not None else '-'} | Output: {output_mode} | Pulse: {heartbeat}\n"
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
                        f"PID: {self._status_pid if self._status_pid is not None else '-'} | Output: {output_mode} | Pulse: {heartbeat}\n"
                        f"Selected panel: {selected_label} | Stream: {stream_mode if self._deepseek_stream_active else 'hidden'}\n"
                        f"Elapsed: {self._elapsed_text_locked()} | Last output: {self._last_output_text_locked()}\n"
                        f"{self._detail_metrics_text_locked()}"
                    )
                self._layout["progress"].update(
                    panel_module.Panel(
                        self._progress,
                        title="Progress",
                        border_style=self._theme.progress_border_style,
                        padding=(0, 1),
                    )
                )
                output_border_style = (
                    self._theme.output_border_style_selected
                    if self._selected_panel == "output"
                    or not self._deepseek_stream_active
                    else self._theme.output_border_style_idle
                )
                self._layout["output"].update(
                    panel_module.Panel(
                        output_renderable,
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
                        self._theme.stream_border_style_selected
                        if self._selected_panel == "deepseek_stream"
                        else self._theme.stream_border_style_idle
                    )
                    stream_subtitle = (
                        self._build_deepseek_stream_subtitle_renderable_locked()
                    )
                    stream_layout.update(
                        panel_module.Panel(
                            stream_renderable,
                            title=(
                                f"DeepSeek Stream {heartbeat} (Selected)"
                                if self._selected_panel == "deepseek_stream"
                                else f"DeepSeek Stream {heartbeat}"
                            ),
                            subtitle=stream_subtitle,
                            subtitle_align="left",
                            border_style=stream_border_style,
                            padding=(0, 1),
                        )
                    )
                self._layout["controls"].update(
                    panel_module.Panel(
                        self._build_controls_renderable_locked(controls_text),
                        title="Controls",
                        border_style=self._theme.controls_border_style,
                        style=self._theme.controls_panel_style,
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

    def _heartbeat_symbol_locked(self) -> str:
        """Return an animated heartbeat glyph for lightweight live feedback."""
        if _supports_unicode_output():
            frames = self._theme.unicode_heartbeat_frames
        else:
            frames = self._theme.ascii_heartbeat_frames
        if not frames:
            return "*"
        frame_index = int(time.monotonic() * 2.0) % len(frames)
        return frames[frame_index]

    def _build_controls_renderable_locked(self, controls_text: str) -> Any:
        """Build controls panel renderable with theme-aware styling."""
        if rich_text is None:
            return controls_text
        text_module = cast(Any, rich_text)
        renderable = text_module.Text()
        lines = controls_text.splitlines()
        for index, line in enumerate(lines):
            line_style = self._theme.controls_text_style
            if (
                line.startswith("Keys unavailable:")
                or line.startswith("Selected panel:")
                or line.startswith("Elapsed:")
                or line.startswith("Detail:")
            ):
                line_style = self._theme.controls_secondary_text_style

            if " | Pulse: " in line:
                prefix, pulse = line.split(" | Pulse: ", maxsplit=1)
                renderable.append(prefix, style=line_style)
                renderable.append(" | Pulse: ", style=self._theme.controls_text_style)
                renderable.append(pulse, style=self._theme.live_badge_style)
            else:
                renderable.append(line, style=line_style)
            if index < len(lines) - 1:
                renderable.append("\n")
        return renderable

    def _build_output_renderable_locked(self, output_text: str) -> Any:
        """Build output panel renderable with semantic line colorization."""
        if rich_text is None:
            return output_text
        text_module = cast(Any, rich_text)
        renderable = text_module.Text()
        lines = output_text.splitlines()
        if not lines:
            lines = [output_text]
        for index, line in enumerate(lines):
            renderable.append(line, style=self._output_line_style(line))
            if index < len(lines) - 1:
                renderable.append("\n")
        return renderable

    def _build_deepseek_stream_renderable_locked(self, stream_text: str) -> Any:
        """Build DeepSeek stream renderable with phase-aware line colors."""
        if rich_text is None:
            return stream_text
        text_module = cast(Any, rich_text)
        renderable = text_module.Text()
        lines = stream_text.splitlines()
        if not lines:
            lines = [stream_text]
        active_phase = ""
        for index, line in enumerate(lines):
            style, active_phase = self._stream_line_style(
                line=line,
                active_phase=active_phase,
            )
            renderable.append(line, style=style)
            if index < len(lines) - 1:
                renderable.append("\n")
        return renderable

    def _output_line_style(self, line: str) -> str:
        """Map one output log line to a theme style."""
        normalized = line.strip()
        if not normalized:
            return self._theme.output_text_style
        if normalized.startswith("[EVENT]"):
            return self._theme.output_event_style
        if normalized.startswith("[DEEPSEEK STATUS]"):
            return self._theme.output_deepseek_status_style
        if normalized.startswith("[DEEPSEEK TOOL_CALL]"):
            return self._theme.output_deepseek_tool_style
        if normalized.startswith("[INDEXTTS2]"):
            return self._theme.output_indextts_style
        if normalized.startswith("[ERROR]") or normalized.startswith("Traceback"):
            return self._theme.output_error_style
        if normalized.startswith("[WARNING]") or normalized.startswith("[WARN]"):
            return self._theme.output_warning_style
        if normalized.startswith("[INFO]"):
            return self._theme.output_info_style
        if (
            normalized.startswith("[PROMPT")
            or normalized.startswith("Input mode:")
            or normalized.startswith("Choice>")
            or normalized.startswith("Esc exits typing mode.")
            or normalized.startswith("Error:")
        ):
            return self._theme.output_prompt_style
        if normalized.startswith("[...]"):
            return self._theme.output_hint_style
        if normalized.startswith("Waiting for output"):
            return self._theme.output_hint_style
        return self._theme.output_text_style

    def _stream_line_style(self, *, line: str, active_phase: str) -> tuple[str, str]:
        """Return stream-line style and updated active phase tracker."""
        normalized = line.strip()
        if not normalized:
            if active_phase == "reasoning":
                return self._theme.stream_reasoning_text_style, active_phase
            if active_phase == "answer":
                return self._theme.stream_answer_text_style, active_phase
            return self._theme.stream_text_style, active_phase
        if normalized == "[REASONING]":
            return self._theme.stream_reasoning_header_style, "reasoning"
        if normalized == "[ANSWER]":
            return self._theme.stream_answer_header_style, "answer"
        if normalized.startswith("[...]") or normalized.startswith("Waiting for"):
            return self._theme.stream_meta_style, active_phase
        if active_phase == "reasoning":
            return self._theme.stream_reasoning_text_style, active_phase
        if active_phase == "answer":
            return self._theme.stream_answer_text_style, active_phase
        return self._theme.stream_text_style, active_phase

    def _style_for_context_percent_left(self, percent_left: float | None) -> str:
        """Choose context subtitle color based on remaining-token percentage."""
        if percent_left is None:
            return self._theme.stream_meta_style
        if percent_left < 0.15:
            return self._theme.subtitle_critical_style
        if percent_left < 0.35:
            return self._theme.subtitle_warn_style
        return self._theme.subtitle_ok_style

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

    def _build_deepseek_stream_subtitle_renderable_locked(self) -> Any:
        """Build styled subtitle renderable with context-health color cues."""
        if rich_text is None:
            return self._build_deepseek_stream_subtitle_locked()
        text_module = cast(Any, rich_text)
        if (
            self._deepseek_context_tokens_used is None
            or self._deepseek_context_tokens_limit is None
            or self._deepseek_context_tokens_left is None
            or self._deepseek_context_percent_left is None
        ):
            return text_module.Text("Ctx: pending", style=self._theme.stream_meta_style)

        remaining_style = self._style_for_context_percent_left(
            self._deepseek_context_percent_left
        )
        subtitle = text_module.Text(style=self._theme.stream_meta_style)
        subtitle.append("Ctx ", style=self._theme.stream_meta_style)
        subtitle.append(
            f"{self._deepseek_context_tokens_used:,}/{self._deepseek_context_tokens_limit:,}",
            style=self._theme.controls_text_style,
        )
        subtitle.append(" | Left ", style=self._theme.stream_meta_style)
        subtitle.append(
            f"{self._deepseek_context_tokens_left:,} "
            f"({self._deepseek_context_percent_left * 100:.1f}%)",
            style=remaining_style,
        )
        subtitle.append(" | Rollovers ", style=self._theme.stream_meta_style)
        subtitle.append(
            str(self._deepseek_context_rollover_count),
            style=self._theme.output_info_style,
        )
        return subtitle

    def _visible_line_count_locked(self) -> int:
        """Estimate lines visible inside the currently selected viewport."""
        return self._visible_line_count_for_panel_locked(
            self._selected_scroll_target_locked()
        )

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
                self._scroll_offset = self._max_scroll_offset_for_panel_locked("output")
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
                    if key in {"\r", "\n"}:
                        self._pending_prompt_enter_monotonic = time.monotonic()
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

