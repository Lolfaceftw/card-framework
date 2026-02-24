from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme


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
                "error": "bold red",
                "agent_name": "bold blue",
                "thought": "grey46",
                "Retrieval": "cyan",
                "Summarizer": "green",
                "Critic": "magenta",
                "tool_name": "bold yellow",
                "tool_args": "dim cyan",
                "tool_result": "dim green",
            }
        )
        self.console = Console(theme=custom_theme)

        from events import event_bus

        event_bus.subscribe("system_message", self.print_system)
        event_bus.subscribe("status_message", self.print_status)
        event_bus.subscribe("error_message", self.print_error)
        event_bus.subscribe("agent_message", self.print_agent_message)
        event_bus.subscribe("agent_thought", self.print_thought)
        event_bus.subscribe("tool_invocation", self.print_tool_invocation)
        event_bus.subscribe("tool_result", self.print_tool_result)

    def print_system(self, message: str):
        """Prints a dim/italic system-level message."""
        self.console.print(f"[system]{message}[/system]")

    def print_status(self, message: str):
        """Prints an important status update."""
        self.console.print(f"[status]Status: {message}[/status]")

    def print_error(self, message: str):
        """Prints an error message."""
        self.console.print(f"[error]Error: {message}[/error]")

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
        # Pick color based on agent name, default to white
        border_style = (
            agent_name
            if agent_name in ["Retrieval", "Summarizer", "Critic"]
            else "white"
        )

        content = message
        if truncate:
            content = self._truncate_text(message)

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
        border_style = (
            agent_name
            if agent_name in ["Retrieval", "Summarizer", "Critic"]
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

            def __enter__(self):
                self.live = Live(
                    self._build_panel(),
                    console=self.console,
                    refresh_per_second=10000,
                    vertical_overflow="visible",
                )
                self.live.__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.live.__exit__(exc_type, exc_val, exc_tb)
                self.console.print()

            def _build_panel(self):
                text = Text()
                if self.full_thought.strip():
                    text.append("💭 Thinking...\n", style="bold yellow")
                    text.append(self.full_thought, style="thought")
                    text.append("\n" + "─" * 40 + "\n", style="dim")

                if self.full_content:
                    text.append(self.full_content)
                elif not self.full_thought.strip():
                    text.append("Thinking...", style="dim")

                return Panel(
                    text,
                    title=Text(f"[ {self.agent_name} ]", style=self.border_style),
                    title_align="left",
                    border_style=self.border_style,
                    padding=(1, 2),
                )

            def update_thought(self, chunk: str):
                self.full_thought += chunk
                self.live.update(self._build_panel())

            def update_content(self, chunk: str):
                self.full_content += chunk
                self.live.update(self._build_panel())

        return LiveMessage(self.console, agent_name, border_style)

    def print_thought(self, agent_name: str, thought: str):
        """Prints the model's 'thinking' or scratchpad content."""
        if not thought.strip():
            return
        content = self._truncate_text(thought, max_lines=10)
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
        import json

        arg_str = json.dumps(arguments)
        if len(arg_str) > 100:
            arg_str = arg_str[:100] + "..."
        self.console.print(
            f"🔧 [tool_name]Tool Call: {tool_name}[/tool_name] [tool_args]{arg_str}[/tool_args]"
        )

    def print_tool_result(self, tool_name: str, result: str):
        """Prints a concise result of a tool execution."""
        summary = result
        if len(result) > 150:
            summary = result[:150].replace("\n", " ") + "..."

        self.console.print(
            f"✅ [tool_name]Tool Result ({tool_name}):[/tool_name] [tool_result]{summary}[/tool_result]"
        )
        self.console.print()


# Expose a global instance
ui = ConsoleManager()
