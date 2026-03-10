"""Tests for runtime command resolution helpers."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from card_framework.audio_pipeline.errors import DependencyMissingError
from card_framework.audio_pipeline.runtime import (
    ensure_command_available,
    probe_audio_duration_ms,
    resolve_command_path,
)


def test_resolve_command_path_uses_imageio_ffmpeg_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resolve ffmpeg from the packaged imageio-ffmpeg helper when PATH is empty."""
    ffmpeg_path = tmp_path / "ffmpeg.exe"
    ffmpeg_path.write_text("", encoding="utf-8")

    monkeypatch.setattr("card_framework.audio_pipeline.runtime.shutil.which", lambda command: None)
    monkeypatch.setattr(
        "card_framework.audio_pipeline.runtime.importlib.import_module",
        lambda module_name: SimpleNamespace(get_ffmpeg_exe=lambda: str(ffmpeg_path)),
    )

    resolved = resolve_command_path("ffmpeg")

    assert resolved == str(ffmpeg_path.resolve())


def test_ensure_command_available_prepends_packaged_ffmpeg_directory_to_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Expose the packaged ffmpeg directory through PATH for nested subprocess users."""
    ffmpeg_path = tmp_path / "ffmpeg.exe"
    ffmpeg_path.write_text("", encoding="utf-8")

    monkeypatch.setenv("PATH", "")
    monkeypatch.setattr("card_framework.audio_pipeline.runtime.shutil.which", lambda command: None)
    monkeypatch.setattr(
        "card_framework.audio_pipeline.runtime.importlib.import_module",
        lambda module_name: SimpleNamespace(get_ffmpeg_exe=lambda: str(ffmpeg_path)),
    )

    ensure_command_available("ffmpeg")

    assert str(ffmpeg_path.parent.resolve()) in os.environ["PATH"].split(os.pathsep)


def test_resolve_command_path_rejects_missing_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise a dependency error when no command source is available."""
    monkeypatch.setattr("card_framework.audio_pipeline.runtime.shutil.which", lambda command: None)

    def _raise_import_error(module_name: str) -> SimpleNamespace:
        del module_name
        raise ImportError("missing")

    monkeypatch.setattr(
        "card_framework.audio_pipeline.runtime.importlib.import_module",
        _raise_import_error,
    )

    with pytest.raises(DependencyMissingError, match="Required command 'ffmpeg'"):
        resolve_command_path("ffmpeg")


def test_probe_audio_duration_ms_parses_ffmpeg_duration_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Parse duration from ffmpeg stderr when ffprobe is unavailable."""
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFF")

    monkeypatch.setattr(
        "card_framework.audio_pipeline.runtime.resolve_command_path",
        lambda command_name: "ffmpeg.exe",
    )
    observed_kwargs: dict[str, object] = {}

    def _fake_run(
        command: list[str],
        check: bool,
        stdout: int,
        stderr: int,
        text: bool,
        encoding: str,
        errors: str,
    ) -> subprocess.CompletedProcess[str]:
        observed_kwargs.update(
            {
                "check": check,
                "stdout": stdout,
                "stderr": stderr,
                "text": text,
                "encoding": encoding,
                "errors": errors,
            }
        )
        return subprocess.CompletedProcess(
            command,
            1,
            "",
            (
                "Input #0, wav, from 'sample.wav':\n"
                "  Duration: 00:01:23.45, bitrate: 128 kb/s\n"
            ),
        )

    monkeypatch.setattr("card_framework.audio_pipeline.runtime.subprocess.run", _fake_run)

    duration_ms = probe_audio_duration_ms(audio_path)

    assert duration_ms == 83_450
    assert observed_kwargs["encoding"] == "utf-8"
    assert observed_kwargs["errors"] == "replace"
