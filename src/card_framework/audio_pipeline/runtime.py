"""Runtime helpers for device and dependency resolution."""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess

from card_framework.audio_pipeline.errors import DependencyMissingError

_COMMAND_OVERRIDE_ENV_VARS = {
    "ffmpeg": "CARD_FRAMEWORK_FFMPEG_EXECUTABLE",
}
_FFMPEG_DURATION_PATTERN = re.compile(
    r"Duration:\s*(?P<hours>\d+):(?P<minutes>\d+):(?P<seconds>\d+(?:\.\d+)?)"
)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_command_available(command_name: str) -> None:
    """Raise if command is not discoverable in PATH or packaged fallbacks."""
    resolve_command_path(command_name)


def resolve_command_path(command_name: str) -> str:
    """Resolve one runtime command from PATH, env overrides, or packaged fallbacks."""
    override_env_var = _COMMAND_OVERRIDE_ENV_VARS.get(command_name)
    if override_env_var:
        override_value = str(os.environ.get(override_env_var, "")).strip()
        if override_value:
            resolved_override = _resolve_command_candidate(override_value)
            if resolved_override:
                _prepend_parent_to_path(resolved_override)
                return resolved_override
            raise DependencyMissingError(
                f"Configured command '{override_value}' from {override_env_var} is unavailable."
            )

    resolved_path = shutil.which(command_name)
    if resolved_path is not None:
        _prepend_parent_to_path(resolved_path)
        return resolved_path

    if command_name == "ffmpeg":
        packaged_ffmpeg = _resolve_packaged_ffmpeg_path()
        if packaged_ffmpeg:
            _prepend_parent_to_path(packaged_ffmpeg)
            return packaged_ffmpeg

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

    Returns ``None`` when duration cannot be resolved (missing ``ffmpeg``,
    invalid media, or parsing failures).
    """
    try:
        ffmpeg_executable = resolve_command_path("ffmpeg")
    except DependencyMissingError:
        return None

    command = [
        ffmpeg_executable,
        "-i",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        duration_match = _FFMPEG_DURATION_PATTERN.search(
            (completed.stderr or "") + "\n" + (completed.stdout or "")
        )
        if duration_match is None:
            return None
        duration_seconds = (
            int(duration_match.group("hours")) * 3600
            + int(duration_match.group("minutes")) * 60
            + float(duration_match.group("seconds"))
        )
        if duration_seconds <= 0:
            return None
        return int(round(duration_seconds * 1000))
    except Exception:
        return None


def _resolve_command_candidate(command_value: str) -> str | None:
    """Resolve one command candidate from a filesystem path or command name."""
    candidate_path = Path(command_value).expanduser()
    if candidate_path.is_file():
        return str(candidate_path.resolve())
    return shutil.which(command_value)


def _resolve_packaged_ffmpeg_path() -> str | None:
    """Return the packaged `imageio-ffmpeg` binary path when available."""
    try:
        imageio_ffmpeg = importlib.import_module("imageio_ffmpeg")
    except ImportError:
        return None

    get_ffmpeg_exe = getattr(imageio_ffmpeg, "get_ffmpeg_exe", None)
    if not callable(get_ffmpeg_exe):
        return None
    try:
        return str(Path(get_ffmpeg_exe()).resolve())
    except Exception:
        return None


def _prepend_parent_to_path(command_path: str) -> None:
    """Prepend one resolved command directory to PATH for nested subprocess users."""
    parent_dir = str(Path(command_path).resolve().parent)
    current_path = str(os.environ.get("PATH", ""))
    current_parts = [part for part in current_path.split(os.pathsep) if part]
    if parent_dir in current_parts:
        return
    os.environ["PATH"] = os.pathsep.join([parent_dir, *current_parts])
