"""Streaming callback adapters used by provider implementations."""

from __future__ import annotations

from contextlib import AbstractContextManager
import sys
from typing import Any

from card_framework.shared.llm_provider import (
    LLMResponseCallback,
    StreamingResponseAbort,
)
from card_framework.shared.repetition_guard import extract_repeated_response_span


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
            self._ensure_plain_header()
            return

        from card_framework.cli.ui import ui

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


class CompositeResponseCallback(LLMResponseCallback):
    """Forward streaming events to multiple callbacks in order."""

    def __init__(self, *callbacks: LLMResponseCallback) -> None:
        self._callbacks = tuple(callbacks)

    def on_start(self, agent_name: str) -> None:
        """Signal stream start to each callback."""
        for callback in self._callbacks:
            callback.on_start(agent_name)

    def on_thought_token(self, token: str) -> None:
        """Forward one reasoning token chunk to each callback."""
        for callback in self._callbacks:
            callback.on_thought_token(token)

    def on_content_token(self, token: str) -> None:
        """Forward one content token chunk to each callback."""
        for callback in self._callbacks:
            callback.on_content_token(token)

    def on_complete(self) -> None:
        """Close every callback, preserving the first raised error."""
        first_error: Exception | None = None
        for callback in self._callbacks:
            try:
                callback.on_complete()
            except Exception as exc:  # pragma: no cover - defensive cleanup path
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error


class StreamingRepeatGuardCallback(LLMResponseCallback):
    """Abort a stream when one response starts repeating the same long span."""

    def __init__(
        self,
        *,
        min_repeated_sentence_words: int,
        single_response_repeat_threshold: int,
    ) -> None:
        self._min_repeated_sentence_words = max(1, min_repeated_sentence_words)
        self._single_response_repeat_threshold = max(2, single_response_repeat_threshold)
        self._thought_buffer = ""
        self._content_buffer = ""

    def on_start(self, agent_name: str) -> None:
        """Reset per-response buffers for a new stream."""
        del agent_name
        self._thought_buffer = ""
        self._content_buffer = ""

    def _check_buffer(self, *, text: str, is_content: bool) -> None:
        """Raise a controlled abort when the streamed text becomes repetitive."""
        repeated_span = extract_repeated_response_span(
            text,
            min_words=self._min_repeated_sentence_words,
            repeat_threshold=self._single_response_repeat_threshold,
        )
        if repeated_span is None:
            return
        preview, repeat_count = repeated_span
        raise StreamingResponseAbort(
            reason=(
                "stream repetition detected "
                f"({repeat_count} repeats): {preview}"
            ),
            partial_content=text if is_content else "",
            partial_reasoning=text if not is_content else "",
        )

    def on_thought_token(self, token: str) -> None:
        """Inspect incremental reasoning text for repeated long spans."""
        self._thought_buffer += token
        self._check_buffer(text=self._thought_buffer, is_content=False)

    def on_content_token(self, token: str) -> None:
        """Inspect incremental content text for repeated long spans."""
        self._content_buffer += token
        self._check_buffer(text=self._content_buffer, is_content=True)

    def on_complete(self) -> None:
        """No-op completion hook for the stream guard."""
        return None
