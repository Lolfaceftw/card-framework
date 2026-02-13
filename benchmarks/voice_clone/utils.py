"""Utility helpers for voice cloning benchmark scripts."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def slugify(value: str) -> str:
    """Convert text into a stable filesystem token."""
    lowered = value.strip().lower()
    collapsed = re.sub(r"[^a-z0-9]+", "_", lowered)
    cleaned = collapsed.strip("_")
    return cleaned or "speaker"


def now_utc_compact() -> str:
    """Return a compact UTC timestamp for directory names."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso8601_now() -> str:
    """Return ISO-8601 UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_existing_path(raw_path: str, base_dir: Path) -> Path:
    """Resolve a file path from absolute path or repo-relative candidates.

    Args:
        raw_path: Raw file path string.
        base_dir: Base directory for relative path resolution.

    Returns:
        Resolved existing path.

    Raises:
        FileNotFoundError: No candidate exists.
    """
    candidate = Path(raw_path)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append((base_dir / candidate).resolve())
        candidates.append((Path.cwd() / candidate).resolve())
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(
        f"Path does not exist: {raw_path} (checked relative to {base_dir})"
    )

