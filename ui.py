import io
import json
import os
import re
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

_CONSOLE_SAFE_TRANSLATIONS = str.maketrans(
    {
        "\u2192": "->",
        "\u2014": "-",
        "\u2013": "-",
        "\u2026": "...",
    }
)


def _resolve_live_fps(default: int = 30) -> int:
    """Return live-render FPS from env with safe bounds and fallback."""
    raw_fps = os.getenv("LLM_TEST_LIVE_FPS")
    if raw_fps is None:
        return default
    try:
        parsed = int(raw_fps)
    except ValueError:
        return default
    return max(1, min(parsed, 120))


def _strip_tool_call_blocks(text: str) -> str:
    """
    Hide XML-style tool-call blocks from terminal display content.

    This preserves normal assistant prose while suppressing raw tool markup
    such as:
    <tool_call>{...}</tool_call>
    """
    without_closed_blocks = re.sub(
        r"<tool_call>\s*.*?\s*</tool_call>",
        "",
        text,
        flags=re.DOTALL,
    )
    dangling_open_tag_idx = without_closed_blocks.find("<tool_call>")
    if dangling_open_tag_idx >= 0:
        return without_closed_blocks[:dangling_open_tag_idx]
    return without_closed_blocks


def _sanitize_console_text(text: str, *, encoding: str | None) -> str:
    """Return text that is safe to print to the active console encoding."""
    sanitized = text.translate(_CONSOLE_SAFE_TRANSLATIONS)
    if not encoding:
        return sanitized
    return sanitized.encode(encoding, errors="replace").decode(encoding)


def _coerce_tool_result_payload(result: str) -> Any:
    """Parse JSON-shaped tool results into structured payloads when possible."""
    stripped = result.strip()
    if not stripped:
        return ""
    if stripped[0] not in "{[":
        return result
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return result


def _format_inline_tool_value(value: Any) -> str:
    """Render one scalar tool-result value on a single line."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _is_inline_tool_value(value: Any) -> bool:
    """Return whether one value is short enough for inline rendering."""
    if isinstance(value, str):
        return "\n" not in value and len(value) <= 120
    return not isinstance(value, (dict, list))


def _format_tool_result_lines(payload: Any, *, indent: int = 0) -> list[str]:
    """Render one tool-result payload into display lines."""
    padding = " " * indent
    if isinstance(payload, dict):
        if not payload:
            return [f"{padding}{{}}"]
        lines: list[str] = []
        for key, value in payload.items():
            label = f"{padding}{key}:"
            if _is_inline_tool_value(value):
                lines.append(f"{label} {_format_inline_tool_value(value)}")
                continue
            lines.append(label)
            lines.extend(_format_tool_result_lines(value, indent=indent + 2))
        return lines

    if isinstance(payload, list):
        if not payload:
            return [f"{padding}[]"]
        lines = []
        for item in payload:
            if _is_inline_tool_value(item):
                lines.append(f"{padding}- {_format_inline_tool_value(item)}")
                continue
            lines.append(f"{padding}-")
            lines.extend(_format_tool_result_lines(item, indent=indent + 2))
        return lines

    if isinstance(payload, str):
        normalized = payload.rstrip("\n")
        if not normalized:
            return [padding]
        return [f"{padding}{line}" for line in normalized.splitlines()]

    return [f"{padding}{_format_inline_tool_value(payload)}"]


def _format_tool_result_for_terminal(result: str) -> str:
    """Format one tool-result payload for readable terminal display."""
    payload = _coerce_tool_result_payload(result)
    return "\n".join(_format_tool_result_lines(payload))


class ConsoleManager:
    """
    Singleton Manager for Rich Console Output.
    Provides stylized formatting for system messages and agent chat outputs.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConsoleManager, cls).__new__(cls)
            cls._instance._init_console()
        return cls._instance

    def _init_console(self):
        # Setup specific theme colors for our UI
        custom_theme = Theme(
            {
                "system": "dim italic white",
                "status": "yellow",
                "status_spawn": "bold green",
                "status_close": "bold cyan",
                "error": "bold red",
                "agent_name": "bold blue",
                "thought": "grey46",
                "Retrieval": "cyan",
                "Summarizer": "green",
                "Critic": "magenta",
                "Corrector": "yellow",
                "tool_name": "bold yellow",
                "tool_args": "dim cyan",
                "tool_result": "dim green",
            }
        )
        self.console = Console(theme=custom_theme)
        self._inline_status_active = False
        self._last_inline_status_len = 0

        from events import event_bus

        event_bus.subscribe("system_message", self.print_system)
        event_bus.subscribe("status_message", self.print_status)
        event_bus.subscribe("error_message", self.print_error)
        event_bus.subscribe("agent_message", self.print_agent_message)
        event_bus.subscribe("agent_thought", self.print_thought)
        event_bus.subscribe("tool_invocation", self.print_tool_invocation)
        event_bus.subscribe("tool_result", self.print_tool_result)

    def _clear_inline_status(self):
        """Clear any in-place status line before normal line-based output."""
        if not self._inline_status_active:
            return
        stream = self.console.file
        stream.write("\r" + (" " * self._last_inline_status_len) + "\r")
        stream.flush()
        self._inline_status_active = False
        self._last_inline_status_len = 0

    def print_system(self, message: str):
        """Prints a dim/italic system-level message."""
        self._clear_inline_status()
        safe_message = _sanitize_console_text(
            message,
            encoding=getattr(self.console.file, "encoding", None),
        )
        self.console.print(f"[system]{safe_message}[/system]")

    def print_status(self, message: str, inline: bool = False, **kwargs):
        """Prints a status update; supports in-place inline updates."""
        del kwargs
        safe_message = _sanitize_console_text(
            message,
            encoding=getattr(self.console.file, "encoding", None),
        )
        rendered = f"Status: {safe_message}"
        status_style = "status"
        normalized_message = message.strip().upper()
        if normalized_message.startswith("AGENT SPAWN"):
            status_style = "status_spawn"
        elif normalized_message.startswith("AGENT CLOSE"):
            status_style = "status_close"
        if inline:
            trailing_padding = ""
            if len(rendered) < self._last_inline_status_len:
                trailing_padding = " " * (self._last_inline_status_len - len(rendered))
            self.console.print(
                f"[{status_style}]{rendered}{trailing_padding}[/{status_style}]",
                end="\r",
                highlight=False,
                soft_wrap=False,
            )
            self._inline_status_active = True
            self._last_inline_status_len = max(
                self._last_inline_status_len,
                len(rendered),
            )
            return

        self._clear_inline_status()
        self.console.print(f"[{status_style}]{rendered}[/{status_style}]")

    def print_error(self, message: str):
        """Prints an error message."""
        self._clear_inline_status()
        safe_message = _sanitize_console_text(
            message,
            encoding=getattr(self.console.file, "encoding", None),
        )
        self.console.print(f"[error]Error: {safe_message}[/error]")

    def _truncate_text(
        self, text: str, max_lines: int = 20, max_chars: int = 2000
    ) -> str:
        """Truncates text for concise display, but more generous than before."""
        lines = text.splitlines()
        if len(lines) > max_lines:
            truncated = "\n".join(lines[:max_lines])
            return f"{truncated}\n... ({len(lines) - max_lines} more lines)"
        if len(text) > max_chars:
            return f"{text[:max_chars]}... ({len(text) - max_chars} more characters)"
        return text

    def print_agent_message(
        self,
        agent_name: str,
        message: str,
        markdown: bool = True,
        truncate: bool = False,
    ):
        """
        Prints an agent's response in a stylised chat-like Panel.
        """
        self._clear_inline_status()
        if not message or not message.strip():
            return

        # Pick color based on agent name, default to white
        border_style = (
            agent_name
            if agent_name in ["Retrieval", "Summarizer", "Critic", "Corrector"]
            else "white"
        )

        content = message
        content = _strip_tool_call_blocks(content)
        content = _sanitize_console_text(
            content,
            encoding=getattr(self.console.file, "encoding", None),
        )
        if truncate:
            content = self._truncate_text(content)

        if markdown:
            content = Markdown(content)

        panel = Panel(
            content,
            title=Text(f"[ {agent_name} ]", style=border_style),
            title_align="left",
            border_style=border_style,
            padding=(1, 2),
        )
        self.console.print(panel)
        self.console.print()

    def live_agent_message(self, agent_name: str):
        """
        Returns a context manager for live-updating an agent's message.
        """
        self._clear_inline_status()
        border_style = (
            agent_name
            if agent_name in ["Retrieval", "Summarizer", "Critic", "Corrector"]
            else "white"
        )

        class LiveMessage:
            def __init__(self, console, agent_name, border_style):
                self.console = console
                self.agent_name = agent_name
                self.border_style = border_style
                self.full_content = ""
                self.full_thought = ""
                self.live = None
                self._last_render_at = 0.0
                self._live_fps = _resolve_live_fps()
                self._min_render_interval_seconds = 1.0 / float(self._live_fps)

            def __enter__(self):
                self.live = Live(
                    self._build_panel(),
                    console=self.console,
                    refresh_per_second=self._live_fps,
                    vertical_overflow="crop",
                    auto_refresh=False,
                    transient=True,
                )
                self.live.__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                final_panel = self._build_panel()
                self._render(force=True)
                self.live.__exit__(exc_type, exc_val, exc_tb)
                if self.full_content.strip() or self.full_thought.strip():
                    self.console.print(final_panel)
                    self.console.print()

            def _build_panel(self):
                display_content = _sanitize_console_text(
                    _strip_tool_call_blocks(self.full_content),
                    encoding=getattr(self.console.file, "encoding", None),
                )
                if not self.full_thought.strip() and not display_content.strip():
                    from rich.console import Group

                    return Group()

                text = Text()
                if self.full_thought.strip():
                    text.append("[THINKING]\n", style="bold yellow")
                    text.append(
                        _sanitize_console_text(
                            self.full_thought,
                            encoding=getattr(self.console.file, "encoding", None),
                        ),
                        style="thought",
                    )
                    text.append("\n" + "-" * 40 + "\n", style="dim")

                if display_content:
                    text.append(display_content)

                return Panel(
                    text,
                    title=Text(f"[ {self.agent_name} ]", style=self.border_style),
                    title_align="left",
                    border_style=self.border_style,
                    padding=(1, 2),
                )

            def _render(self, force: bool = False):
                now = time.monotonic()
                if (
                    force
                    or (now - self._last_render_at) >= self._min_render_interval_seconds
                ):
                    self.live.update(self._build_panel(), refresh=True)
                    self._last_render_at = now

            def update_thought(self, chunk: str):
                self.full_thought += chunk
                self._render()

            def update_content(self, chunk: str):
                self.full_content += chunk
                self._render()

        return LiveMessage(self.console, agent_name, border_style)

    def print_thought(self, agent_name: str, thought: str):
        """Prints the model's 'thinking' or scratchpad content."""
        self._clear_inline_status()
        if not thought.strip():
            return
        content = self._truncate_text(thought, max_lines=10)
        content = _sanitize_console_text(
            content,
            encoding=getattr(self.console.file, "encoding", None),
        )
        panel = Panel(
            Text(content, style="thought"),
            title=Text(f"[ {agent_name} Thought ]", style="dim"),
            title_align="left",
            border_style="dim",
            padding=(1, 2),
        )
        self.console.print(panel)

    def print_tool_invocation(self, tool_name: str, arguments: dict):
        """Prints a concise tool invocation message."""
        self._clear_inline_status()
        import json

        arg_str = json.dumps(arguments)
        if len(arg_str) > 100:
            arg_str = arg_str[:100] + "..."
        safe_tool_name = _sanitize_console_text(
            tool_name,
            encoding=getattr(self.console.file, "encoding", None),
        )
        safe_arg_str = _sanitize_console_text(
            arg_str,
            encoding=getattr(self.console.file, "encoding", None),
        )
        self.console.print(
            f"[tool_name]Tool Call: {safe_tool_name}[/tool_name] [tool_args]{safe_arg_str}[/tool_args]"
        )

    def print_tool_result(self, tool_name: str, result: str):
        """Prints a readable result of a tool execution."""
        self._clear_inline_status()
        formatted_result = _format_tool_result_for_terminal(result)
        formatted_result = self._truncate_text(
            formatted_result,
            max_lines=18,
            max_chars=3000,
        )
        safe_tool_name = _sanitize_console_text(
            tool_name,
            encoding=getattr(self.console.file, "encoding", None),
        )
        safe_result = _sanitize_console_text(
            formatted_result,
            encoding=getattr(self.console.file, "encoding", None),
        )

        self.console.print(f"[tool_name]Tool Result ({safe_tool_name}):[/tool_name]")
        self.console.print(Text(safe_result, style="tool_result"))
        self.console.print()


# Expose a global instance
ui = ConsoleManager()


def _capture_live_agent_message_output(
    agent_name: str,
    *,
    content: str = "",
    thought: str = "",
    trailing_status: str | None = None,
) -> str:
    """Capture one live-agent render cycle for UI regression tests."""
    stream = io.StringIO()
    console_manager = ConsoleManager()
    original_console = console_manager.console
    console_manager.console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        theme=Theme(
            {
                "Retrieval": "cyan",
                "Summarizer": "green",
                "Critic": "magenta",
                "Corrector": "yellow",
            }
        ),
        width=100,
    )
    try:
        with console_manager.live_agent_message(agent_name) as live_message:
            if thought:
                live_message.update_thought(thought)
            if content:
                live_message.update_content(content)
        if trailing_status is not None:
            console_manager.print_status(trailing_status)
    finally:
        console_manager.console = original_console
    return stream.getvalue()


def _capture_tool_result_output(tool_name: str, result: str) -> str:
    """Capture one tool-result render for UI regression tests."""
    stream = io.StringIO()
    console_manager = ConsoleManager()
    original_console = console_manager.console
    console_manager.console = Console(
        file=stream,
        force_terminal=False,
        color_system=None,
        theme=Theme({"tool_name": "bold yellow", "tool_result": "dim green"}),
        width=100,
    )
    try:
        console_manager.print_tool_result(tool_name, result)
    finally:
        console_manager.console = original_console
    return stream.getvalue()
