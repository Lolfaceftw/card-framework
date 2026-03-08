"""Streaming callback adapters used by provider implementations."""

from __future__ import annotations

from contextlib import AbstractContextManager
import sys
from typing import Any

from llm_provider import LLMResponseCallback


def _stdout_supports_live_rich_output() -> bool:
    """Return whether stdout behaves like an interactive terminal."""
    stream = getattr(sys, "stdout", None)
    if stream is None:
        return False
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


class RichConsoleResponseCallback(LLMResponseCallback):
    """Bridge provider token streams to Rich UI or plain piped stdout."""

    def __init__(self) -> None:
        self._live_context: AbstractContextManager[Any] | None = None
        self._live_message: Any | None = None
        self._plain_mode = False
        self._plain_agent_name = ""
        self._plain_started = False
        self._plain_thought_started = False
        self._plain_content_started = False

    def on_start(self, agent_name: str) -> None:
        """Open a new live-message session for the active agent."""
        self.on_complete()
        self._plain_mode = not _stdout_supports_live_rich_output()
        if self._plain_mode:
            self._plain_agent_name = agent_name
            self._plain_started = False
            self._plain_thought_started = False
            self._plain_content_started = False
            return

        from ui import ui

        self._live_context = ui.live_agent_message(agent_name)
        self._live_message = self._live_context.__enter__()

    def _plain_write(self, text: str) -> None:
        """Write one chunk to stdout immediately for piped streaming runs."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def _ensure_plain_header(self) -> None:
        """Print the agent header once in plain-text streaming mode."""
        if self._plain_started:
            return
        self._plain_write(f"\n[{self._plain_agent_name}]\n")
        self._plain_started = True

    def on_thought_token(self, token: str) -> None:
        """Render one thought-token chunk."""
        if self._plain_mode:
            if not token:
                return
            if not self._plain_thought_started and not token.strip():
                return
            self._ensure_plain_header()
            if not self._plain_thought_started:
                self._plain_write("[THINKING] ")
                self._plain_thought_started = True
            self._plain_write(token)
            return

        if self._live_message is None:
            return
        self._live_message.update_thought(token)

    def on_content_token(self, token: str) -> None:
        """Render one final-answer token chunk."""
        if self._plain_mode:
            if not token:
                return
            if not self._plain_content_started and not token.strip():
                return
            self._ensure_plain_header()
            if self._plain_thought_started and not self._plain_content_started:
                self._plain_write("\n[CONTENT] ")
                self._plain_content_started = True
            elif not self._plain_content_started:
                self._plain_write("[CONTENT] ")
                self._plain_content_started = True
            self._plain_write(token)
            return

        if self._live_message is None:
            return
        self._live_message.update_content(token)

    def on_complete(self) -> None:
        """Close the current live-message session when one is active."""
        if self._plain_mode and self._plain_started:
            self._plain_write("\n\n")
        self._plain_mode = False
        self._plain_agent_name = ""
        self._plain_started = False
        self._plain_thought_started = False
        self._plain_content_started = False

        if self._live_context is not None:
            self._live_context.__exit__(None, None, None)
        self._live_context = None
        self._live_message = None
