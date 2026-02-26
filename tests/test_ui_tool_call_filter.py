"""Unit tests for terminal display filtering of tool-call markup."""

from ui import _strip_tool_call_blocks


def test_strip_tool_call_blocks_removes_closed_blocks() -> None:
    raw = "hello\n<tool_call>{\"name\":\"x\"}</tool_call>\nworld"
    assert _strip_tool_call_blocks(raw) == "hello\n\nworld"


def test_strip_tool_call_blocks_removes_dangling_open_block() -> None:
    raw = "prefix\n<tool_call>{\"name\":\"x\"}"
    assert _strip_tool_call_blocks(raw) == "prefix\n"
