"""Unit tests for terminal display filtering of tool-call markup."""

from card_framework.cli.ui import (
    _capture_tool_result_output,
    _format_tool_result_for_terminal,
    _capture_live_agent_message_output,
    _sanitize_console_text,
    _strip_tool_call_blocks,
)


def test_strip_tool_call_blocks_removes_closed_blocks() -> None:
    raw = "hello\n<tool_call>{\"name\":\"x\"}</tool_call>\nworld"
    assert _strip_tool_call_blocks(raw) == "hello\n\nworld"


def test_strip_tool_call_blocks_removes_dangling_open_block() -> None:
    raw = "prefix\n<tool_call>{\"name\":\"x\"}"
    assert _strip_tool_call_blocks(raw) == "prefix\n"


def test_sanitize_console_text_rewrites_unicode_for_cp1252() -> None:
    raw = "Summarizer \u2192 pass\u2026"
    assert _sanitize_console_text(raw, encoding="cp1252") == "Summarizer -> pass..."


def test_live_agent_message_output_keeps_one_final_panel_before_next_status() -> None:
    output = _capture_live_agent_message_output(
        "Critic",
        content="full critic analysis",
        trailing_status="Voice cloning complete",
    )

    assert output.count("full critic analysis") == 1
    assert "Voice cloning complete" in output


def test_format_tool_result_for_terminal_preserves_multiline_string_values() -> None:
    formatted = _format_tool_result_for_terminal(
        '{"speaker":"SPEAKER_01","excerpt":"Line one\\nLine two","count":2}'
    )

    assert "speaker: SPEAKER_01" in formatted
    assert "excerpt:" in formatted
    assert "  Line one" in formatted
    assert "  Line two" in formatted
    assert "count: 2" in formatted


def test_tool_result_output_renders_structured_json_over_multiple_lines() -> None:
    output = _capture_tool_result_output(
        "query_transcript",
        '{"excerpt":"[SPEAKER_00]: hello\\n[SPEAKER_01]: world","match_count":2}',
    )

    assert "Tool Result (query_transcript):" in output
    assert "excerpt:" in output
    assert "[SPEAKER_00]: hello" in output
    assert "[SPEAKER_01]: world" in output
    assert "match_count: 2" in output
