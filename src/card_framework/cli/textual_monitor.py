"""Textual subprocess monitor for interactive operator workflows."""

from __future__ import annotations

import codecs
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
from typing import BinaryIO, Mapping

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import (
    Button,
    Footer,
    Header,
    LoadingIndicator,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)


_STEP_LINE_RE = re.compile(r"^\[(\d+)/(\d+)\]\s+(.+?)\s*$")
_SUBPROGRESS_PATTERNS = (
    ("chunk", re.compile(r"\bchunk\s+(\d+)/(\d+)\b", flags=re.IGNORECASE)),
    (
        "segment",
        re.compile(r"\bsegment\s+(\d+)/(\d+)\b", flags=re.IGNORECASE),
    ),
    ("turn", re.compile(r"\bturn\s+(\d+)/(\d+)\b", flags=re.IGNORECASE)),
)
_ETA_PATTERNS = (
    re.compile(r"estimated time left\s+([^)]+)", flags=re.IGNORECASE),
    re.compile(r"eta left[: ]+(.+)$", flags=re.IGNORECASE),
    re.compile(r"\beta[: ]+(.+)$", flags=re.IGNORECASE),
)
_ETA_OVERRUN_RE = re.compile(
    r"running longer than estimate by\s+(.+)$",
    flags=re.IGNORECASE,
)
_AGENT_PANEL_NOISE_PATTERNS = (
    re.compile(
        r'^\[\d{4}-\d{2}-\d{2} [^\]]+\]\s*-\s*HTTP Request:\s+\w+\s+https?://\S+\s+"HTTP/\d(?:\.\d)? \d{3}.*"$',
    ),
    re.compile(
        r'^(?:INFO|DEBUG):httpx:HTTP Request:\s+\w+\s+https?://\S+\s+"HTTP/\d(?:\.\d)? \d{3}.*"$',
    ),
)


@dataclass(slots=True, frozen=True)
class TextualMonitorRequest:
    """Describe one child command that should run inside a Textual monitor."""

    title: str
    summary_lines: tuple[str, ...]
    command: tuple[str, ...]
    cwd: Path
    expected_stages: tuple[str, ...] = ()
    env_overrides: Mapping[str, str] | None = None


@dataclass(slots=True, frozen=True)
class MonitorState:
    """Minimal user-facing execution state rendered by the monitor."""

    total_steps: int = 1
    current_step_index: int = 1
    step_labels: tuple[str, ...] = ()
    progress_total: float = 1.0
    progress_value: float = 0.0
    progress_label: str = "Starting"
    progress_detail: str = "Starting"
    current_stage: str = "Preparing launch"
    eta_text: str = "Waiting for ETA..."


@dataclass(slots=True)
class LatestAgentPanelState:
    """Rendered state for the latest summarizer/critic style stream."""

    agent_name: str = "Waiting for model output"
    section: str = ""
    reasoning: str = ""
    content: str = ""


class DetailProgressBar(Widget, can_focus=False):
    """Render one single-line progress bar with centered detail text."""

    DEFAULT_CSS = """
    DetailProgressBar {
        width: 1fr;
        height: 1;
    }
    """

    total: reactive[float] = reactive(1.0)
    progress: reactive[float] = reactive(0.0)
    detail: reactive[str] = reactive("Starting")

    def update(self, *, total: float, progress: float, detail: str) -> None:
        """Update the rendered progress-bar state."""
        self.total = max(total, 1.0)
        self.progress = min(max(progress, 0.0), self.total)
        self.detail = detail
        self.refresh()

    def render(self) -> Text:
        width = max(self.size.width, 12)
        filled_width = int(round(width * (self.progress / max(self.total, 1.0))))
        detail = self.detail.strip()
        if len(detail) > width - 1:
            detail = detail[: max(width - 2, 0)] + ("…" if width > 1 else "")
        start = max(0, (width - len(detail)) // 2)
        rendered = Text(no_wrap=True, end="")
        for index in range(width):
            in_fill = index < filled_width
            if detail and start <= index < start + len(detail):
                character = detail[index - start]
                style = "bold black on #4fd1c5" if in_fill else "bold white on #3a3f4b"
            else:
                character = " "
                style = "on #4fd1c5" if in_fill else "on #3a3f4b"
            rendered.append(character, style=style)
        return rendered


@dataclass(slots=True, frozen=True)
class ClassifiedMonitorLine:
    """One parsed output line plus display metadata."""

    text: str
    style: str
    show_in_diagnostics: bool


def _looks_loading_status(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith("loading ") or lowered.startswith("loaded ")


def _looks_failed_loading_status(text: str) -> bool:
    """Return whether one loading-timer line represents a real failure."""
    return text.strip().lower().startswith("failed loading ")


def _looks_warning_line(text: str) -> bool:
    lowered = text.strip().lower()
    return "warning" in lowered or lowered.startswith("note:") or "[warn" in lowered


def _looks_error_line(text: str) -> bool:
    lowered = text.strip().lower()
    if lowered.startswith("traceback"):
        return True
    if lowered.startswith("error:") or lowered.startswith("[error"):
        return True
    if " exception" in lowered or lowered.endswith(" exception"):
        return True
    if " failed" in lowered or lowered.startswith("failed "):
        return True
    return False


def _classify_line_style(text: str, *, is_error: bool) -> str:
    lowered = text.strip().lower()
    if _STEP_LINE_RE.match(text.strip()):
        return "bold magenta"
    if _looks_failed_loading_status(text):
        return "bold red"
    if _looks_loading_status(text):
        if lowered.startswith("loaded "):
            return "green"
        return "cyan"
    if _looks_error_line(text) or is_error:
        return "bold red"
    if _looks_warning_line(text):
        return "yellow"
    if "estimated time left" in lowered or lowered.startswith("eta"):
        return "bright_cyan"
    if lowered.startswith("[monitor]"):
        return "cyan"
    if "saved summary" in lowered or lowered.endswith(" complete"):
        return "green"
    return "white"


def _show_in_diagnostics(text: str, *, is_error: bool) -> bool:
    if not text.strip():
        return False
    if _looks_failed_loading_status(text):
        return True
    if _looks_loading_status(text):
        return False
    if _looks_error_line(text) or _looks_warning_line(text):
        return True
    return is_error


def _extract_eta_text(text: str) -> str | None:
    overrun_match = _ETA_OVERRUN_RE.search(text)
    if overrun_match is not None:
        overrun_text = overrun_match.group(1).strip().rstrip(").")
        return f"Overrun {overrun_text}"
    for pattern in _ETA_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return match.group(1).strip().rstrip(").")
    return None


def _extract_subprogress(text: str) -> tuple[int, int] | None:
    for _label, pattern in _SUBPROGRESS_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            current = int(match.group(1))
            total = int(match.group(2))
            if total > 0:
                return current, total
    return None


def _extract_subprogress_detail(text: str) -> tuple[str, int, int] | None:
    """Return subprogress label plus counts for chunk/segment/turn updates."""
    for label, pattern in _SUBPROGRESS_PATTERNS:
        match = pattern.search(text)
        if match is None:
            continue
        current = int(match.group(1))
        total = int(match.group(2))
        if total > 0:
            return label, current, total
    return None


def _infer_named_runtime_stage(text: str) -> str | None:
    """Infer one normalized operator-facing runtime stage label."""
    lowered = text.strip().lower()
    if not lowered:
        return None
    if "separation" in lowered or "source separation" in lowered or "demucs" in lowered:
        return "Source separation"
    if (
        "transcription" in lowered
        or "granite speech" in lowered
        or " asr" in lowered
        or lowered.startswith("asr ")
    ):
        return "Transcription"
    if "diarization" in lowered or "nemo" in lowered:
        return "Diarization"
    if lowered.startswith("running stage-4 interjection") or lowered.startswith(
        "stage-4 interjection complete:"
    ):
        return "Interjection"
    if (
        lowered.startswith("running voice cloning")
        or lowered.startswith("voice cloning complete:")
        or lowered.startswith("voice clone stage:")
        or lowered.startswith("running stage-3 voice cloning")
        or lowered.startswith("finalizing live draft voice-clone manifest")
        or "live draft renderer" in lowered
    ):
        return "Voice cloning"
    if "speaker sample" in lowered or "speaker-sample" in lowered:
        return "Speaker samples"
    if (
        "starting orchestration loop" in lowered
        or "draft mutations" in lowered
        or "live drafting" in lowered
        or lowered in {"[summarizer]", "[critic]"}
        or lowered.startswith("[thinking]")
        or lowered.startswith("[content]")
    ):
        return "Summarizer & Critic"
    if "benchmark preflight" in lowered:
        return "Benchmark preflight"
    if "benchmark run" in lowered or "running benchmark" in lowered:
        return "Benchmark run"
    return None


def _looks_stage_completion(text: str) -> bool:
    """Return whether one line looks like a stage-finished marker."""
    lowered = text.strip().lower()
    if lowered.startswith("audio stage:") and " finished in " in lowered:
        return True
    if lowered.startswith("audio stage complete:"):
        return True
    if lowered.startswith("voice cloning complete:"):
        return True
    if lowered.startswith("stage-4 interjection complete:"):
        return True
    if lowered.startswith("generated ") and " speaker samples " in lowered:
        return True
    return False


def _is_eta_update_line(text: str) -> bool:
    """Return whether one line is primarily an ETA/status-progress update."""
    lowered = text.strip().lower()
    return (
        "estimated time left" in lowered or "running longer than estimate by" in lowered
    )


def _is_progress_update_line(text: str) -> bool:
    """Return whether one line is better rendered in the progress widget."""
    return _extract_subprogress(text) is not None or _is_eta_update_line(text)


def _show_in_timeline(text: str, *, is_error: bool) -> bool:
    """Return whether one parsed line should stay visible in the live timeline."""
    stripped = text.strip()
    if not stripped:
        return False
    if _is_progress_update_line(text) and not (_looks_error_line(text) or is_error):
        return False
    if stripped in {"[Summarizer]", "[Critic]", "[Judge]", "[Corrector]"}:
        return False
    for prefix in ("[THINKING] ", "[CONTENT] "):
        if stripped.startswith(prefix):
            remainder = stripped[len(prefix) :].strip()
            if any(pattern.match(remainder) for pattern in _AGENT_PANEL_NOISE_PATTERNS):
                return True
            return False
    return True


def _truncate_agent_text(
    text: str, *, max_lines: int = 14, max_chars: int = 1800
) -> str:
    """Keep the latest streamed agent content compact and snappy."""
    trimmed = text.strip()
    if not trimmed:
        return ""
    if len(trimmed) > max_chars:
        trimmed = trimmed[-max_chars:]
    lines = trimmed.splitlines()
    if len(lines) > max_lines:
        trimmed = "\n".join(lines[-max_lines:])
    return trimmed


def _sanitize_agent_panel_text(text: str) -> str:
    """Drop transport/log chatter from rendered reasoning/content text."""
    if not text.strip():
        return ""
    kept_lines = [
        line
        for line in text.splitlines()
        if not any(
            pattern.match(line.strip()) for pattern in _AGENT_PANEL_NOISE_PATTERNS
        )
    ]
    return "\n".join(kept_lines).strip()


def render_stage_checklist(
    *,
    expected_stages: tuple[str, ...],
    completed_stages: tuple[str, ...],
    current_stage: str,
    finished: bool,
) -> str:
    """Render one clean checklist of expected stages."""
    if not expected_stages:
        return "Waiting for the runtime to report its first stage..."
    completed = set(completed_stages)
    lines: list[str] = []
    for label in expected_stages:
        if finished or label in completed:
            marker = "[x]"
        else:
            marker = "[ ]"
        suffix = " <- current" if not finished and label == current_stage else ""
        lines.append(f"{marker} {label}{suffix}")
    return "\n".join(lines)


class PlainAgentStreamParser:
    """Extract the latest plain-mode reasoning/content stream for one agent."""

    _header_markers: dict[str, str] = {
        f"[{name}]\n": name for name in ("Summarizer", "Critic", "Judge", "Corrector")
    }
    _marker_names: dict[str, str] = {
        "[THINKING] ": "reasoning",
        "\n[THINKING] ": "reasoning",
        "[CONTENT] ": "content",
        "\n[CONTENT] ": "content",
    }
    _all_markers = tuple([*_header_markers.keys(), *_marker_names.keys()])
    _max_marker_length = max(len(marker) for marker in _all_markers)

    def __init__(self) -> None:
        self._buffer = ""
        self._current_section = ""
        self._state = LatestAgentPanelState()

    def consume(self, text: str) -> LatestAgentPanelState:
        """Consume one chunk of plain streamed output."""
        if not text:
            return self._state
        self._buffer += text.replace("\r\n", "\n").replace("\r", "\n")
        while True:
            marker_info = self._find_next_marker(self._buffer)
            if marker_info is None:
                if self._current_section and "[" not in self._buffer:
                    self._append_to_current_section(self._buffer)
                    self._buffer = ""
                    break
                safe_length = max(0, len(self._buffer) - self._max_marker_length)
                if safe_length <= 0:
                    break
                self._append_to_current_section(self._buffer[:safe_length])
                self._buffer = self._buffer[safe_length:]
                break
            marker_index, marker_text = marker_info
            if marker_index > 0:
                self._append_to_current_section(self._buffer[:marker_index])
            self._buffer = self._buffer[marker_index + len(marker_text) :]
            if marker_text in self._header_markers:
                self._state = LatestAgentPanelState(
                    agent_name=self._header_markers[marker_text],
                )
                self._current_section = ""
                continue
            self._current_section = self._marker_names[marker_text]
            self._state.section = self._current_section
        return self._state

    def flush(self) -> LatestAgentPanelState:
        """Commit any remaining buffered text into the latest panel state."""
        if self._buffer:
            self._append_to_current_section(self._buffer)
            self._buffer = ""
        return self._state

    def _append_to_current_section(self, text: str) -> None:
        if not text or not self._current_section:
            return
        if self._current_section == "reasoning":
            self._state.reasoning += text
            return
        if self._current_section == "content":
            self._state.content += text

    def _find_next_marker(self, text: str) -> tuple[int, str] | None:
        matches = [
            (index, marker)
            for marker in self._all_markers
            for index in [text.find(marker)]
            if index >= 0
        ]
        if not matches:
            return None
        return min(matches, key=lambda item: item[0])


def _infer_stage_label(text: str) -> str | None:
    stripped = text.strip()
    lowered = stripped.lower()
    if not stripped:
        return None
    if _looks_loading_status(stripped):
        return _infer_named_runtime_stage(stripped)
    if "audio stage:" in lowered:
        inferred_audio_stage = _infer_named_runtime_stage(stripped)
        if inferred_audio_stage is not None:
            return inferred_audio_stage
        return "Audio stage"
    return _infer_named_runtime_stage(stripped)


def fold_monitor_line(
    state: MonitorState,
    raw_line: str,
    *,
    is_error: bool,
) -> tuple[MonitorState, ClassifiedMonitorLine]:
    """Project one output line into updated monitor state and diagnostics metadata."""
    text = raw_line.rstrip("\r\n")
    if not text:
        return state, ClassifiedMonitorLine(
            text="",
            style="white",
            show_in_diagnostics=False,
        )

    total_steps = state.total_steps
    current_step_index = state.current_step_index
    step_labels = list(state.step_labels)
    progress_total = state.progress_total
    progress_value = state.progress_value
    progress_label = state.progress_label
    progress_detail = state.progress_detail
    current_stage = state.current_stage
    eta_text = state.eta_text

    step_match = _STEP_LINE_RE.match(text)
    if step_match is not None:
        current_step_index = max(1, int(step_match.group(1)))
        total_steps = max(current_step_index, int(step_match.group(2)))
        current_stage = step_match.group(3).strip()
        if len(step_labels) < total_steps:
            step_labels.extend(
                f"Pending step {index}"
                for index in range(len(step_labels) + 1, total_steps + 1)
            )
        step_labels[current_step_index - 1] = current_stage
        if current_step_index < total_steps:
            progress_total = float(total_steps)
            progress_value = float(current_step_index)
            progress_label = f"Setup {current_step_index}/{total_steps}"
            progress_detail = progress_label
        else:
            progress_total = 1.0
            progress_value = 0.0
            progress_label = "Runtime starting"
            progress_detail = progress_label
        eta_text = "Waiting for ETA..."
    else:
        inferred_stage = _infer_stage_label(text)
        if inferred_stage is not None:
            stage_changed = inferred_stage != current_stage
            current_stage = inferred_stage
            if stage_changed or progress_label in {"Runtime starting", "Done"}:
                progress_total = 1.0
                progress_value = 0.0
                progress_label = "Live"
                progress_detail = progress_label
        inferred_eta = _extract_eta_text(text)
        if inferred_eta is not None:
            eta_text = inferred_eta
        subprogress = _extract_subprogress_detail(text)
        if subprogress is not None:
            unit_label, completed, total = subprogress
            progress_total = float(max(total, 1))
            progress_value = float(min(max(completed, 0), total))
            progress_label = f"{completed}/{total}"
            progress_detail = f"{unit_label} {completed}/{total}"
        elif _looks_stage_completion(text):
            progress_total = 1.0
            progress_value = 1.0
            progress_label = "Done"
            progress_detail = progress_label

    next_state = MonitorState(
        total_steps=max(total_steps, 1),
        current_step_index=max(current_step_index, 1),
        step_labels=tuple(step_labels),
        progress_total=max(progress_total, 1.0),
        progress_value=max(progress_value, 0.0),
        progress_label=progress_label,
        progress_detail=progress_detail,
        current_stage=current_stage,
        eta_text=eta_text,
    )
    return next_state, ClassifiedMonitorLine(
        text=text,
        style=_classify_line_style(text, is_error=is_error),
        show_in_diagnostics=_show_in_diagnostics(text, is_error=is_error),
    )


class TextualMonitorApp(App[int]):
    """Run one subprocess and keep the operator inside a Textual monitor."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #status_banner {
        padding: 0 1;
        color: $text-muted;
    }

    #overview {
        border: round $primary;
        padding: 1;
        margin: 0 1 1 1;
        height: auto;
    }

    #status_panel {
        border: round $panel;
        padding: 0 1;
        margin: 0 1 1 1;
        height: auto;
    }

    .status_label {
        color: $text-muted;
    }

    .status_value {
        text-style: bold;
    }

    #status_inline {
        height: auto;
        align: center middle;
    }

    .status_separator {
        color: $text-muted;
        margin: 0 1;
    }

    #progress_row {
        margin: 0 0 0 0;
        height: auto;
        align: center middle;
    }

    #overall_progress {
        width: 1fr;
        height: auto;
        padding: 1 0;
    }

    #eta_value {
        width: 1fr;
        content-align: center middle;
        color: $text-muted;
    }

    #agent_panel {
        border: round $panel;
        padding: 1;
        margin: 0 1 1 1;
        height: 12;
    }

    #stage_checklist {
        border: round $panel;
        padding: 1;
        height: 1fr;
    }

    .button-row {
        margin: 0 1 1 1;
        height: auto;
    }

    Button {
        margin-right: 1;
    }

    TabbedContent {
        height: 1fr;
        margin: 0 1 0 1;
    }

    TabPane {
        padding: 0 1;
    }

    RichLog {
        height: 1fr;
        border: round $panel;
    }

    #command_details {
        border: round $panel;
        padding: 1;
    }
    """

    BINDINGS = [
        ("ctrl+c", "stop_process", "Stop run"),
        ("q", "quit_monitor", "Close"),
    ]

    def __init__(self, *, request: TextualMonitorRequest) -> None:
        super().__init__()
        self._request = request
        self._expected_stages = tuple(request.expected_stages)
        self._completed_stages: set[str] = set()
        self._process: subprocess.Popen[bytes] | None = None
        self._exit_code: int | None = None
        self._started_at: float | None = None
        self._finished_at: float | None = None
        self._terminate_requested = False
        self._state = MonitorState()
        self._latest_agent = LatestAgentPanelState()
        self._agent_parser = PlainAgentStreamParser()
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self.title = request.title
        self.sub_title = str(request.cwd)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(id="status_banner")
        yield Static(id="overview")
        with Vertical(id="status_panel"):
            with Horizontal(id="status_inline"):
                yield LoadingIndicator(id="stage_spinner")
                yield Static("Stage", classes="status_label")
                yield Static(id="stage_value", classes="status_value")
                yield Static("•", classes="status_separator")
                yield Static("Progress", classes="status_label")
                yield Static(id="progress_value", classes="status_value")
            with Vertical(id="progress_row"):
                yield DetailProgressBar(id="overall_progress")
                yield Static(id="eta_value")
        with Horizontal(classes="button-row"):
            yield Button("Stop run", id="stop_run", variant="warning")
            yield Button("Close", id="close_monitor", variant="primary", disabled=True)
        yield Static(id="agent_panel")
        with TabbedContent(initial="stages"):
            with TabPane("Stages", id="stages"):
                yield Static(id="stage_checklist")
            with TabPane("Logs", id="logs"):
                yield RichLog(
                    id="combined_log",
                    wrap=False,
                    highlight=False,
                    markup=False,
                )
            with TabPane("Diagnostics", id="diagnostics"):
                yield RichLog(
                    id="diagnostic_log",
                    wrap=False,
                    highlight=False,
                    markup=False,
                )
            with TabPane("Command", id="command"):
                with VerticalScroll():
                    yield Static(id="command_details")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_overview()
        self._refresh_status_banner()
        self._sync_state_widgets()
        self.query_one("#command_details", Static).update(
            self._render_command_details()
        )
        self.set_interval(0.2, self._tick_status)
        threading.Thread(target=self._run_subprocess, daemon=True).start()

    def _elapsed_seconds(self) -> float:
        if self._started_at is None:
            return 0.0
        finished_at = (
            self._finished_at if self._finished_at is not None else time.monotonic()
        )
        return max(0.0, finished_at - self._started_at)

    def _refresh_status_banner(self) -> None:
        banner = self.query_one("#status_banner", Static)
        if self._started_at is None:
            banner.update("Launching Textual execution monitor...")
            return
        if self._exit_code is None:
            prefix = "Stopping" if self._terminate_requested else "Running"
            message = f"{prefix} · elapsed {self._elapsed_seconds():.1f}s"
            if self._process is not None and self._process.pid is not None:
                message += f" · pid {self._process.pid}"
            banner.update(message)
            return
        if self._terminate_requested or self._exit_code == 130:
            state = "Stopped"
        elif self._exit_code == 0:
            state = "Succeeded"
        else:
            state = f"Failed (exit code {self._exit_code})"
        banner.update(f"{state} · total time {self._elapsed_seconds():.1f}s")

    def _refresh_overview(self) -> None:
        lines = [*self._request.summary_lines]
        if self._request.summary_lines:
            lines.append("")
        lines.extend(
            [
                f"Working directory: {self._request.cwd}",
                f"Command length: {len(self._request.command)} argument(s)",
                "Progress and ETA stay in the status strip; Logs keep non-progress output; Diagnostics keeps warnings and errors.",
            ]
        )
        self.query_one("#overview", Static).update("\n".join(lines))

    def _render_command_details(self) -> str:
        return "\n".join(
            [
                "Workflow summary:",
                *(self._request.summary_lines or ("(none)",)),
                "",
                f"Working directory:\n{self._request.cwd}",
                "",
                "Command:",
                " ".join(self._request.command),
            ]
        )

    def _render_latest_agent_panel(self) -> str:
        """Render the latest summarizer/critic style stream in one compact panel."""
        reasoning = _truncate_agent_text(
            _sanitize_agent_panel_text(self._latest_agent.reasoning)
        )
        content = _truncate_agent_text(
            _sanitize_agent_panel_text(self._latest_agent.content)
        )
        lines = [self._latest_agent.agent_name]
        if reasoning:
            lines.extend(["", "Reasoning:", reasoning])
        if content:
            lines.extend(["", "Content:", content])
        if len(lines) == 1:
            lines.extend(["", "Waiting for the next Summarizer/Critic response..."])
        return "\n".join(lines)

    def _record_stage_transition(
        self, *, previous_stage: str, source_text: str
    ) -> None:
        """Update completed-stage tracking from one monitor line transition."""
        if not self._expected_stages:
            return
        current_stage = self._state.current_stage
        if previous_stage in self._expected_stages and previous_stage != current_stage:
            self._completed_stages.add(previous_stage)
        if (
            _looks_stage_completion(source_text)
            and current_stage in self._expected_stages
        ):
            self._completed_stages.add(current_stage)

    def _sync_state_widgets(self) -> None:
        self.query_one("#stage_value", Static).update(self._state.current_stage)
        self.query_one("#eta_value", Static).update(f"ETA {self._state.eta_text}")
        self.query_one("#progress_value", Static).update(self._state.progress_label)
        self.query_one("#stage_checklist", Static).update(
            render_stage_checklist(
                expected_stages=self._expected_stages or self._state.step_labels,
                completed_stages=tuple(self._completed_stages),
                current_stage=self._state.current_stage,
                finished=self._exit_code == 0,
            )
        )
        agent_panel = self.query_one("#agent_panel", Static)
        show_agent_panel = self._state.current_stage == "Summarizer & Critic"
        agent_panel.display = show_agent_panel
        agent_panel.update(self._render_latest_agent_panel())
        agent_panel.refresh(layout=True)
        progress_bar = self.query_one("#overall_progress", DetailProgressBar)
        progress_bar.update(
            total=max(self._state.progress_total, 1.0),
            progress=min(
                self._state.progress_value, max(self._state.progress_total, 1.0)
            ),
            detail=self._state.progress_detail,
        )
        self.query_one("#stage_spinner", LoadingIndicator).display = (
            self._exit_code is None
        )

    def _tick_status(self) -> None:
        self._refresh_status_banner()

    def _append_monitor_message(self, message: str, *, style: str = "cyan") -> None:
        self.query_one("#combined_log", RichLog).write(
            Text(message + "\n", style=style),
            scroll_end=True,
        )

    def _consume_chunk(self, text: str, *, is_error: bool) -> None:
        if not text:
            return
        normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
        if not is_error:
            self._latest_agent = self._agent_parser.consume(normalized_text)
        buffer_attr = "_stderr_buffer" if is_error else "_stdout_buffer"
        buffer = getattr(self, buffer_attr) + normalized_text
        combined_log = self.query_one("#combined_log", RichLog)
        diagnostic_log = self.query_one("#diagnostic_log", RichLog)
        while True:
            newline_index = buffer.find("\n")
            if newline_index < 0:
                break
            raw_line = buffer[: newline_index + 1]
            buffer = buffer[newline_index + 1 :]
            previous_stage = self._state.current_stage
            self._state, classified = fold_monitor_line(
                self._state,
                raw_line,
                is_error=is_error,
            )
            self._record_stage_transition(
                previous_stage=previous_stage,
                source_text=classified.text,
            )
            if (
                _show_in_timeline(classified.text, is_error=is_error)
                and classified.text
            ):
                combined_log.write(
                    Text(classified.text + "\n", style=classified.style),
                    scroll_end=True,
                )
            if classified.show_in_diagnostics and classified.text:
                diagnostic_log.write(
                    Text(classified.text + "\n", style=classified.style),
                    scroll_end=True,
                )
            self._sync_state_widgets()
        setattr(self, buffer_attr, buffer)

    def _flush_buffer(self, *, is_error: bool) -> None:
        buffer_attr = "_stderr_buffer" if is_error else "_stdout_buffer"
        buffer = getattr(self, buffer_attr)
        if not buffer:
            return
        if not is_error:
            self._latest_agent = self._agent_parser.flush()
        previous_stage = self._state.current_stage
        self._state, classified = fold_monitor_line(
            self._state,
            buffer,
            is_error=is_error,
        )
        self._record_stage_transition(
            previous_stage=previous_stage,
            source_text=classified.text,
        )
        if _show_in_timeline(classified.text, is_error=is_error) and classified.text:
            self.query_one("#combined_log", RichLog).write(
                Text(classified.text + "\n", style=classified.style),
                scroll_end=True,
            )
        if classified.show_in_diagnostics and classified.text:
            self.query_one("#diagnostic_log", RichLog).write(
                Text(classified.text + "\n", style=classified.style),
                scroll_end=True,
            )
        setattr(self, buffer_attr, "")
        self._sync_state_widgets()

    def _set_process_started(self) -> None:
        self._started_at = time.monotonic()
        self._refresh_status_banner()
        self._append_monitor_message("[monitor] Child process started.", style="green")

    def _set_process_finished(self, exit_code: int) -> None:
        if self._exit_code is not None:
            return
        self._exit_code = int(exit_code)
        self._finished_at = time.monotonic()
        stop_button = self.query_one("#stop_run", Button)
        stop_button.disabled = True
        stop_button.label = "Stopped" if self._terminate_requested else "Stop run"
        self.query_one("#close_monitor", Button).disabled = False
        if self._terminate_requested or exit_code == 130:
            self._state = MonitorState(
                total_steps=self._state.total_steps,
                current_step_index=self._state.current_step_index,
                step_labels=self._state.step_labels,
                progress_total=1.0,
                progress_value=0.0,
                progress_label="Stopped",
                progress_detail="Stopped",
                current_stage="Stopped",
                eta_text="Stopped",
            )
        elif self._exit_code == 0:
            self._completed_stages.update(self._expected_stages)
            self._state = MonitorState(
                total_steps=self._state.total_steps,
                current_step_index=self._state.total_steps,
                step_labels=self._state.step_labels,
                progress_total=1.0,
                progress_value=1.0,
                progress_label="Completed",
                progress_detail="Completed",
                current_stage="Completed",
                eta_text="0s",
            )
        else:
            self._state = MonitorState(
                total_steps=self._state.total_steps,
                current_step_index=self._state.current_step_index,
                step_labels=self._state.step_labels,
                progress_total=1.0,
                progress_value=0.0,
                progress_label="Failed",
                progress_detail="Failed",
                current_stage="Failed",
                eta_text="Failed",
            )
        self._sync_state_widgets()
        self._refresh_status_banner()
        if self._terminate_requested or exit_code == 130:
            self._append_monitor_message(
                "[monitor] Child process stopped.", style="yellow"
            )
            return
        if exit_code == 0:
            self._append_monitor_message(
                "[monitor] Child process finished successfully.",
                style="green",
            )
            return
        self._append_monitor_message(
            f"[monitor] Child process failed with exit code {exit_code}.",
            style="bold red",
        )

    def _read_stream(self, stream: BinaryIO | None, *, is_error: bool) -> None:
        if stream is None:
            return
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                reader = getattr(stream, "read1", stream.read)
                chunk = reader(1024)
                if not chunk:
                    break
                decoded = decoder.decode(chunk)
                if decoded:
                    self.call_from_thread(
                        self._consume_chunk, decoded, is_error=is_error
                    )
            remainder = decoder.decode(b"", final=True)
            if remainder:
                self.call_from_thread(self._consume_chunk, remainder, is_error=is_error)
            self.call_from_thread(self._flush_buffer, is_error=is_error)
        finally:
            stream.close()

    def _run_subprocess(self) -> None:
        env = dict(os.environ)
        if self._request.env_overrides:
            env.update(self._request.env_overrides)
        creationflags = 0
        if os.name == "nt":
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            if os.name == "nt":
                process = subprocess.Popen(
                    list(self._request.command),
                    cwd=str(self._request.cwd),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    env=env,
                    creationflags=creationflags,
                )
            else:
                process = subprocess.Popen(
                    list(self._request.command),
                    cwd=str(self._request.cwd),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    bufsize=0,
                    env=env,
                    start_new_session=True,
                )
        except Exception as exc:
            self.call_from_thread(
                self._append_monitor_message,
                f"[monitor] Failed to start child process: {exc}",
                style="bold red",
            )
            self.call_from_thread(self._set_process_finished, 1)
            return

        self._process = process
        self.call_from_thread(self._set_process_started)
        stdout_thread = threading.Thread(
            target=self._read_stream,
            args=(process.stdout,),
            kwargs={"is_error": False},
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=self._read_stream,
            args=(process.stderr,),
            kwargs={"is_error": True},
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        exit_code = process.wait()
        stdout_thread.join()
        stderr_thread.join()
        self.call_from_thread(self._set_process_finished, int(exit_code))

    def action_stop_process(self) -> None:
        process = self._process
        if process is None or self._exit_code is not None or self._terminate_requested:
            return
        self._terminate_requested = True
        stop_button = self.query_one("#stop_run", Button)
        stop_button.disabled = True
        stop_button.label = "Stopping..."
        self._refresh_status_banner()
        self._append_monitor_message(
            "[monitor] Stop requested. Terminating the full process tree...",
            style="yellow",
        )
        threading.Thread(target=self._terminate_process_tree, daemon=True).start()

    def _terminate_process_tree(self) -> None:
        process = self._process
        if process is None or self._exit_code is not None:
            return
        if os.name == "nt":
            if hasattr(signal, "CTRL_BREAK_EVENT"):
                try:
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                    process.wait(timeout=1.0)
                    if process.poll() is not None:
                        self.call_from_thread(
                            self._set_process_finished, process.poll()
                        )
                    return
                except Exception:
                    pass
            try:
                subprocess.run(
                    ["taskkill", "/pid", str(process.pid), "/t"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                process.wait(timeout=1.5)
                if process.poll() is not None:
                    self.call_from_thread(self._set_process_finished, process.poll())
                return
            except Exception:
                pass
            subprocess.run(
                ["taskkill", "/f", "/pid", str(process.pid), "/t"],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )
            try:
                process.wait(timeout=1.5)
            except Exception:
                pass
            if process.poll() is not None:
                self.call_from_thread(self._set_process_finished, process.poll())
            return

        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=1.5)
            if process.poll() is not None:
                self.call_from_thread(self._set_process_finished, process.poll())
            return
        except Exception:
            pass
        os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=1.5)
        except Exception:
            pass
        if process.poll() is not None:
            self.call_from_thread(self._set_process_finished, process.poll())

    def action_quit_monitor(self) -> None:
        if self._exit_code is None:
            self.action_stop_process()
            return
        self.exit(self._exit_code)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id or ""
        if button_id == "stop_run":
            self.action_stop_process()
            return
        if button_id == "close_monitor":
            self.action_quit_monitor()


def run_textual_monitor(*, request: TextualMonitorRequest) -> int:
    """Run one subprocess inside the Textual monitor and return its exit code."""
    app = TextualMonitorApp(request=request)
    result = app.run()
    return int(result) if isinstance(result, int) else 1
