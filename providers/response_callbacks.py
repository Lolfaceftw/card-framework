"""Streaming callback adapters used by provider implementations."""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any

from llm_provider import LLMResponseCallback


class RichConsoleResponseCallback(LLMResponseCallback):
    """Bridge provider token streams to the Rich UI live-message renderer."""

    def __init__(self) -> None:
        self._live_context: AbstractContextManager[Any] | None = None
        self._live_message: Any | None = None

    def on_start(self, agent_name: str) -> None:
        """Open a new live-message session for the active agent."""
        self.on_complete()
        from ui import ui

        self._live_context = ui.live_agent_message(agent_name)
        self._live_message = self._live_context.__enter__()

    def on_thought_token(self, token: str) -> None:
        """Render one thought-token chunk."""
        if self._live_message is None:
            return
        self._live_message.update_thought(token)

    def on_content_token(self, token: str) -> None:
        """Render one final-answer token chunk."""
        if self._live_message is None:
            return
        self._live_message.update_content(token)

    def on_complete(self) -> None:
        """Close the current live-message session when one is active."""
        if self._live_context is not None:
            self._live_context.__exit__(None, None, None)
        self._live_context = None
        self._live_message = None
