"""Shared helper utilities for pipeline orchestration."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Final

SKIP_A2S_EXCLUDED_DIRS: Final[frozenset[str]] = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "node_modules",
        "checkpoints",
    }
)


def find_dotenv_file(*, script_file: Path | None = None) -> Path | None:
    """Find a ``.env`` file from CWD upward, then fall back to repo root.

    Args:
        script_file: Optional module file path used to resolve repository root.

    Returns:
        Existing dotenv file path when found, else ``None``.
    """
    current_dir = Path.cwd().resolve()
    for directory in [current_dir, *current_dir.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate

    if script_file is None:
        return None
    repo_root_candidate = script_file.resolve().parent.parent / ".env"
    if repo_root_candidate.is_file():
        return repo_root_candidate
    return None


def load_dotenv_file(dotenv_path: Path) -> int:
    """Load missing environment variables from a dotenv file.

    Args:
        dotenv_path: Path to ``.env`` file.

    Returns:
        Number of environment variables added to ``os.environ``.
    """
    loaded_count = 0
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        if key not in os.environ:
            os.environ[key] = value
            loaded_count += 1

    return loaded_count


def count_wav_files(directory: str) -> int:
    """Count ``.wav`` files in a directory."""
    directory_path = Path(directory)
    if not directory_path.exists():
        return 0
    return sum(1 for _ in directory_path.glob("*.wav"))


def format_speaker_wpm_summary(
    per_speaker_wpm: dict[str, float],
    *,
    max_items: int = 6,
) -> str:
    """Build a compact human-readable per-speaker WPM summary."""
    if not per_speaker_wpm:
        return "none"
    ordered_items = sorted(per_speaker_wpm.items())
    visible_items = ordered_items[:max_items]
    summary = ", ".join(f"{speaker}={wpm:.2f}" for speaker, wpm in visible_items)
    hidden_count = len(ordered_items) - len(visible_items)
    if hidden_count > 0:
        return f"{summary}, +{hidden_count} more"
    return summary


def discover_transcript_json_files(
    search_root: Path,
    *,
    excluded_dirs: frozenset[str] = SKIP_A2S_EXCLUDED_DIRS,
) -> list[Path]:
    """Discover diarized transcript JSON files under a search root."""
    from audio2script_and_summarizer.transcript_wpm import load_transcript_segments

    if not search_root.exists():
        return []

    candidates: list[Path] = []
    for current_root, dirnames, filenames in os.walk(search_root):
        dirnames[:] = [d for d in dirnames if d not in excluded_dirs]
        for filename in filenames:
            if not filename.lower().endswith(".json"):
                continue
            candidate_path = Path(current_root) / filename
            try:
                if candidate_path.stat().st_size > 20 * 1024 * 1024:
                    continue
            except OSError:
                continue
            try:
                segments = load_transcript_segments(str(candidate_path))
            except Exception:  # noqa: BLE001
                continue
            if segments:
                candidates.append(candidate_path.resolve())

    def _sort_key(path: Path) -> tuple[float, str]:
        try:
            return (path.stat().st_mtime, str(path))
        except OSError:
            return (0.0, str(path))

    candidates.sort(key=_sort_key, reverse=True)
    return candidates


def calculate_corrected_word_budget(
    *,
    current_word_budget: int,
    target_duration_seconds: float,
    measured_duration_seconds: float,
) -> int:
    """Scale word budget toward the target/actual duration ratio."""
    if current_word_budget <= 0:
        return 1
    if target_duration_seconds <= 0.0 or measured_duration_seconds <= 0.0:
        return max(1, current_word_budget)
    correction_factor = target_duration_seconds / measured_duration_seconds
    corrected = int(round(current_word_budget * correction_factor))
    return max(1, corrected)


def calculate_adaptive_tool_rounds(*, word_budget: int, target_minutes: float) -> int:
    """Estimate DeepSeek tool-loop rounds from summary size and duration."""
    base_rounds = 10
    normalized_budget = max(1, int(word_budget))
    normalized_minutes = max(0.0, float(target_minutes))
    extra_budget_rounds = int(math.ceil(max(0, normalized_budget - 180) / 45.0))
    extra_duration_rounds = int(math.ceil(max(0.0, normalized_minutes - 1.0) * 2.0))
    adaptive_rounds = base_rounds + extra_budget_rounds + extra_duration_rounds
    return max(10, min(30, adaptive_rounds))


def resolve_deepseek_agent_max_tool_rounds(
    *,
    configured_max_tool_rounds: int,
    current_word_budget: int,
    target_minutes: float,
) -> tuple[int, str]:
    """Resolve explicit or adaptive DeepSeek max tool rounds for current pass."""
    if configured_max_tool_rounds > 0:
        return configured_max_tool_rounds, "override"
    adaptive_rounds = calculate_adaptive_tool_rounds(
        word_budget=current_word_budget,
        target_minutes=target_minutes,
    )
    return adaptive_rounds, "adaptive"


def summary_report_path_for_output(summary_output_path: str) -> Path:
    """Resolve default summary report path for a summary JSON output path."""
    summary_output = Path(summary_output_path)
    return summary_output.with_name(f"{summary_output.name}.report.json")


def update_summary_report_duration_metrics(
    *,
    summary_output_path: str,
    target_duration_seconds: float | None,
    measured_duration_seconds: float | None,
    duration_tolerance_seconds: float,
    duration_correction_passes: int,
) -> bool:
    """Patch summary report JSON with Stage 3 duration metrics when available.

    Returns:
        ``True`` when metrics were written successfully, otherwise ``False``.
    """
    report_path = summary_report_path_for_output(summary_output_path)
    if not report_path.exists():
        return False

    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False

    duration_delta_seconds: float | None = None
    duration_within_tolerance: bool | None = None
    if (
        target_duration_seconds is not None
        and measured_duration_seconds is not None
        and target_duration_seconds > 0.0
        and measured_duration_seconds >= 0.0
    ):
        duration_delta_seconds = measured_duration_seconds - target_duration_seconds
        duration_within_tolerance = (
            abs(duration_delta_seconds) <= max(0.0, duration_tolerance_seconds)
        )

    payload["target_duration_seconds"] = (
        round(target_duration_seconds, 3)
        if target_duration_seconds is not None
        else None
    )
    payload["measured_duration_seconds"] = (
        round(measured_duration_seconds, 3)
        if measured_duration_seconds is not None
        else None
    )
    payload["duration_delta_seconds"] = (
        round(duration_delta_seconds, 3)
        if duration_delta_seconds is not None
        else None
    )
    payload["duration_within_tolerance"] = duration_within_tolerance
    payload["duration_correction_passes"] = max(0, int(duration_correction_passes))

    try:
        report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        return False

    return True
