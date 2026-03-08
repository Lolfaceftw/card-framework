from __future__ import annotations

import io

import providers.response_callbacks as response_callbacks


class _StdoutStub(io.StringIO):
    def isatty(self) -> bool:
        return False


def test_response_callback_falls_back_to_plain_streaming_for_piped_stdout(
    monkeypatch,
) -> None:
    stream = _StdoutStub()
    monkeypatch.setattr(response_callbacks.sys, "stdout", stream)

    callback = response_callbacks.RichConsoleResponseCallback()
    callback.on_start("Summarizer")
    callback.on_thought_token("thinking")
    callback.on_content_token("answer")
    callback.on_complete()

    rendered = stream.getvalue()
    assert "[Summarizer]" in rendered
    assert "[THINKING] thinking" in rendered
    assert "[CONTENT] answer" in rendered


def test_response_callback_skips_whitespace_only_sections_in_plain_mode(
    monkeypatch,
) -> None:
    stream = _StdoutStub()
    monkeypatch.setattr(response_callbacks.sys, "stdout", stream)

    callback = response_callbacks.RichConsoleResponseCallback()
    callback.on_start("Summarizer")
    callback.on_content_token("\n\n")
    callback.on_complete()

    rendered = stream.getvalue()
    assert "[Summarizer]" not in rendered
    assert "[CONTENT]" not in rendered


def test_stdout_supports_live_rich_output_false_when_not_tty(monkeypatch) -> None:
    stream = _StdoutStub()
    monkeypatch.setattr(response_callbacks.sys, "stdout", stream)

    assert response_callbacks._stdout_supports_live_rich_output() is False
