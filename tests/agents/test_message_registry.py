"""Message registry behavior tests for edit/no-op semantics."""

from __future__ import annotations

from agents.message_registry import MessageRegistry


def test_edit_returns_noop_metadata_when_content_is_unchanged() -> None:
    registry = MessageRegistry()
    registry.add("SPEAKER_00", "alpha beta gamma")

    result = registry.edit(1, "alpha beta gamma")

    assert result["line"] == 1
    assert result["changed"] is False
    assert result["delta_words"] == 0
    assert result["stagnation_hint"] == "no_change"
    assert result["total_word_count"] == 3


def test_edit_returns_delta_when_content_changes() -> None:
    registry = MessageRegistry()
    registry.add("SPEAKER_00", "alpha beta")

    result = registry.edit(1, "alpha beta gamma delta")

    assert result["line"] == 1
    assert result["changed"] is True
    assert result["delta_words"] == 2
    assert result["total_word_count"] == 4


def test_edit_returns_structured_error_for_missing_line() -> None:
    registry = MessageRegistry()
    registry.add("SPEAKER_00", "alpha beta")

    result = registry.edit(2, "new")

    assert result["error_code"] == "line_not_found"
    assert result["line"] == 2
