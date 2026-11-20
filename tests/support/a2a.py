"""Helpers for asserting on A2A-style message payloads in tests."""

from __future__ import annotations


def extract_agent_text_message(event: object) -> str:
    """Return the text payload from either stubbed or real A2A message objects."""
    if isinstance(event, str):
        return event

    parts = getattr(event, "parts", None)
    if not isinstance(parts, list) or not parts:
        raise AssertionError(f"Expected A2A message with parts, got {type(event)!r}")

    first_part = parts[0]
    root = getattr(first_part, "root", None)
    text = getattr(root, "text", None)
    if not isinstance(text, str):
        raise AssertionError(
            "Expected first A2A message part to expose a text payload."
        )
    return text
