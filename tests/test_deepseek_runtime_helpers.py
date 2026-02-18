"""Unit tests for DeepSeek runtime stream-event output rendering."""

from __future__ import annotations

from audio2script_and_summarizer.deepseek import runtime_helpers


class _FakeStdout:
    """Capture writes while exposing configurable TTY behavior."""

    def __init__(self, *, is_tty: bool) -> None:
        self._is_tty = is_tty
        self._parts: list[str] = []

    def write(self, text: str) -> int:
        """Record one write call and return written char count."""
        self._parts.append(text)
        return len(text)

    def flush(self) -> None:
        """No-op flush hook for compatibility."""

    def isatty(self) -> bool:
        """Return configured TTY capability."""
        return self._is_tty

    def render(self) -> str:
        """Return all captured output text."""
        return "".join(self._parts)


def _reset_plain_stream_state() -> None:
    """Reset shared plain-stream renderer state between test cases."""
    runtime_helpers._PLAIN_STREAM_RENDER_STATE.active_phase = None  # noqa: SLF001
    runtime_helpers._PLAIN_STREAM_RENDER_STATE.line_open = False  # noqa: SLF001


def test_emit_deepseek_stream_event_emits_marker_json_when_not_tty(
    monkeypatch,
) -> None:
    """Keep marker JSON output in auto mode when stdout is not a TTY."""
    fake_stdout = _FakeStdout(is_tty=False)
    monkeypatch.setattr(runtime_helpers.sys, "stdout", fake_stdout)
    monkeypatch.delenv(runtime_helpers.DEEPSEEK_STREAM_OUTPUT_MODE_ENV, raising=False)
    _reset_plain_stream_state()

    runtime_helpers._emit_deepseek_stream_event(  # noqa: SLF001
        {"event": "token", "phase": "reasoning", "text": "hello"}
    )

    rendered = fake_stdout.render()
    assert rendered.startswith(runtime_helpers.DEEPSEEK_STREAM_EVENT_PREFIX)
    assert '"event":"token"' in rendered
    assert '"phase":"reasoning"' in rendered
    assert '"text":"hello"' in rendered


def test_emit_deepseek_stream_event_coalesces_tokens_in_tty_auto_mode(
    monkeypatch,
) -> None:
    """Render token fragments as connected text lines on interactive terminals."""
    fake_stdout = _FakeStdout(is_tty=True)
    monkeypatch.setattr(runtime_helpers.sys, "stdout", fake_stdout)
    monkeypatch.delenv(runtime_helpers.DEEPSEEK_STREAM_OUTPUT_MODE_ENV, raising=False)
    _reset_plain_stream_state()

    runtime_helpers._emit_deepseek_stream_event(  # noqa: SLF001
        {"event": "start", "model": "deepseek-reasoner"}
    )
    runtime_helpers._emit_deepseek_stream_event(  # noqa: SLF001
        {"event": "token", "phase": "reasoning", "text": "."}
    )
    runtime_helpers._emit_deepseek_stream_event(  # noqa: SLF001
        {"event": "token", "phase": "reasoning", "text": " Now"}
    )
    runtime_helpers._emit_deepseek_stream_event(  # noqa: SLF001
        {"event": "token", "phase": "reasoning", "text": " evaluate"}
    )
    runtime_helpers._emit_deepseek_stream_event(  # noqa: SLF001
        {"event": "token", "phase": "reasoning", "text": "."}
    )
    runtime_helpers._emit_deepseek_stream_event(  # noqa: SLF001
        {"event": "done"}
    )

    rendered = fake_stdout.render()
    assert "[DEEPSEEK_STREAM]" not in rendered
    assert "[DEEPSEEK REASONING] . Now evaluate." in rendered
    assert "[DEEPSEEK STATUS] Stream finished for current DeepSeek call." in rendered
