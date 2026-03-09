"""Tests for installed-package runtime layout and bootstrap helpers."""

from __future__ import annotations

from pathlib import Path
import subprocess

from card_framework.runtime.bootstrap import (
    _CTC_FORCED_ALIGNER_ARCHIVE_URL,
    ensure_ctc_forced_aligner_runtime,
    ensure_index_tts_runtime,
)
from card_framework.shared.runtime_layout import RuntimeLayout, resolve_runtime_home


def test_resolve_runtime_home_prefers_env(monkeypatch, tmp_path: Path) -> None:
    """Use `CARD_FRAMEWORK_HOME` when callers provide it explicitly."""
    configured_home = tmp_path / "runtime-home"
    monkeypatch.setenv("CARD_FRAMEWORK_HOME", str(configured_home))

    resolved = resolve_runtime_home()

    assert resolved == configured_home.resolve()


def test_ensure_index_tts_runtime_copies_vendor_syncs_and_downloads(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Prepare writable vendor/runtime assets in the runtime home."""
    vendor_source_dir = tmp_path / "vendor_source"
    vendor_source_dir.mkdir(parents=True)
    (vendor_source_dir / "pyproject.toml").write_text("[project]\nname='indextts'\n", encoding="utf-8")
    (vendor_source_dir / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (vendor_source_dir / "indextts").mkdir()

    layout = RuntimeLayout(
        runtime_home=(tmp_path / "runtime_home").resolve(),
        vendor_source_dir=vendor_source_dir.resolve(),
        vendor_runtime_dir=(tmp_path / "runtime_home" / "vendor" / "index_tts").resolve(),
        checkpoints_dir=(tmp_path / "runtime_home" / "checkpoints" / "index_tts").resolve(),
        bootstrap_state_path=(tmp_path / "runtime_home" / "bootstrap" / "state.json").resolve(),
    )

    recorded_commands: list[list[str]] = []

    def _fake_run(
        command: list[str],
        cwd: str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text
        recorded_commands.append(command)
        if command[:3] == ["uv", "sync", "--locked"]:
            assert cwd == str(layout.vendor_runtime_dir)
            (layout.vendor_runtime_dir / ".venv").mkdir(parents=True, exist_ok=True)
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:3] == ["uv", "tool", "run"]:
            assert cwd == str(layout.runtime_home)
            layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            (layout.checkpoints_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
            weight_path = layout.checkpoints_dir / "model.safetensors"
            weight_path.write_bytes(b"0" * 1_100_000)
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("card_framework.runtime.bootstrap.subprocess.run", _fake_run)

    ensure_index_tts_runtime(layout=layout, uv_executable="uv")

    assert (layout.vendor_runtime_dir / "pyproject.toml").exists()
    assert (layout.vendor_runtime_dir / "uv.lock").exists()
    assert (layout.vendor_runtime_dir / ".venv").exists()
    assert (layout.checkpoints_dir / "config.yaml").exists()
    assert (layout.checkpoints_dir / "model.safetensors").exists()
    assert layout.bootstrap_state_path.exists()
    assert any(command[:3] == ["uv", "sync", "--locked"] for command in recorded_commands)
    assert any(command[:3] == ["uv", "tool", "run"] for command in recorded_commands)


def test_ensure_ctc_forced_aligner_runtime_installs_when_missing(monkeypatch) -> None:
    """Bootstrap the pinned aligner only when the module is absent."""
    recorded_commands: list[list[str]] = []

    monkeypatch.setattr(
        "card_framework.runtime.bootstrap.importlib.util.find_spec",
        lambda module_name: None if module_name == "ctc_forced_aligner" else object(),
    )

    def _fake_run(
        command: list[str],
        cwd: str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, check, capture_output, text
        recorded_commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("card_framework.runtime.bootstrap.subprocess.run", _fake_run)

    ensure_ctc_forced_aligner_runtime(python_executable="python")

    assert recorded_commands == [
        [
            "python",
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            _CTC_FORCED_ALIGNER_ARCHIVE_URL,
        ]
    ]
