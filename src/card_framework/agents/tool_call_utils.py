"""Helpers for canonicalising and deduplicating LLM tool calls."""

from __future__ import annotations

import json
from typing import Any


def canonicalize_arguments(arguments: Any) -> str:
    """Serialize tool arguments into a stable string representation."""
    try:
        return json.dumps(
            arguments,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    except (TypeError, ValueError):
        return str(arguments)


def build_tool_signature(name: str | None, arguments: Any) -> str:
    """Return a canonical signature for a tool call."""
    normalized_name = name or "__unknown_tool__"
    return f"{normalized_name}|{canonicalize_arguments(arguments)}"


def dedupe_tool_calls_by_signature(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Drop duplicate tool calls that share the same name+arguments signature."""
    unique_calls: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()

    for call in tool_calls:
        signature = build_tool_signature(
            name=str(call.get("name") or "__unknown_tool__"),
            arguments=call.get("arguments", {}),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        unique_calls.append(call)

    return unique_calls
