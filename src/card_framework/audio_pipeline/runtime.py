"""Runtime helpers for device and dependency resolution."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import shutil
import subprocess

from card_framework.audio_pipeline.errors import DependencyMissingError


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_command_available(command_name: str) -> None:
    """Raise if command is not discoverable in PATH."""
    if shutil.which(command_name) is None:
        raise DependencyMissingError(
            f"Required command '{command_name}' is not available in PATH."
        )


def ensure_module_available(module_name: str) -> None:
    """Raise if Python module cannot be imported."""
    if importlib.util.find_spec(module_name) is None:
        raise DependencyMissingError(
            f"Required Python module '{module_name}' is not installed."
        )


def resolve_device(preferred_device: str) -> str:
    """
    Resolve runtime device from configuration.

    Args:
        preferred_device: One of ``auto``, ``cpu``, or ``cuda``.

    Returns:
        A concrete device string accepted by adapters.
    """
    normalized = preferred_device.strip().lower()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("preferred_device must be one of: auto, cpu, cuda")
    if normalized in {"cpu", "cuda"}:
        return normalized
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def resolve_path(path_value: str, *, base_dir: Path) -> Path:
    """Resolve a possibly-relative path against a base directory."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def probe_audio_duration_ms(audio_path: Path) -> int | None:
    """
    Best-effort probe of an audio duration in milliseconds.

    Returns ``None`` when duration cannot be resolved (missing ``ffprobe``,
    invalid media, or parsing failures).
    """
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration_seconds = float(completed.stdout.strip())
        if duration_seconds <= 0:
            return None
        return int(round(duration_seconds * 1000))
    except Exception:
        return None

