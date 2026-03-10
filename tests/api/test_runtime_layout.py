"""Tests for installed-package runtime layout and bootstrap helpers."""

from __future__ import annotations

from pathlib import Path
import re
import subprocess

from card_framework.runtime.bootstrap import (
    _CTC_FORCED_ALIGNER_ARCHIVE_URL,
    ensure_ctc_forced_aligner_runtime,
    ensure_index_tts_runtime,
    resolve_uv_executable,
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
        encoding: str | None = None,
        errors: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, encoding, errors
        recorded_commands.append(command)
        if command[:5] == ["uv", "sync", "--locked", "--python", "python312"]:
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
    monkeypatch.setattr(
        "card_framework.runtime.bootstrap.resolve_uv_executable",
        lambda *, uv_executable: uv_executable,
    )

    ensure_index_tts_runtime(
        layout=layout,
        uv_executable="uv",
        python_executable="python312",
    )

    assert (layout.vendor_runtime_dir / "pyproject.toml").exists()
    assert (layout.vendor_runtime_dir / "uv.lock").exists()
    assert (layout.vendor_runtime_dir / ".venv").exists()
    assert (layout.checkpoints_dir / "config.yaml").exists()
    assert (layout.checkpoints_dir / "model.safetensors").exists()
    assert layout.bootstrap_state_path.exists()
    assert any(
        command[:5] == ["uv", "sync", "--locked", "--python", "python312"]
        for command in recorded_commands
    )
    assert any(command[:3] == ["uv", "tool", "run"] for command in recorded_commands)


def test_ensure_index_tts_runtime_resyncs_when_requested_python_changes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Resync the vendored runtime when a cached env targets a different interpreter."""
    vendor_source_dir = tmp_path / "vendor_source"
    vendor_source_dir.mkdir(parents=True)
    pyproject_text = "[project]\nname='indextts'\n"
    lock_text = "version = 1\n"
    (vendor_source_dir / "pyproject.toml").write_text(pyproject_text, encoding="utf-8")
    (vendor_source_dir / "uv.lock").write_text(lock_text, encoding="utf-8")
    (vendor_source_dir / "indextts").mkdir()

    layout = RuntimeLayout(
        runtime_home=(tmp_path / "runtime_home").resolve(),
        vendor_source_dir=vendor_source_dir.resolve(),
        vendor_runtime_dir=(tmp_path / "runtime_home" / "vendor" / "index_tts").resolve(),
        checkpoints_dir=(tmp_path / "runtime_home" / "checkpoints" / "index_tts").resolve(),
        bootstrap_state_path=(tmp_path / "runtime_home" / "bootstrap" / "state.json").resolve(),
    )
    layout.vendor_runtime_dir.mkdir(parents=True, exist_ok=True)
    (layout.vendor_runtime_dir / "pyproject.toml").write_text(pyproject_text, encoding="utf-8")
    (layout.vendor_runtime_dir / "uv.lock").write_text(lock_text, encoding="utf-8")
    (layout.vendor_runtime_dir / "indextts").mkdir()
    (layout.vendor_runtime_dir / ".venv").mkdir()
    layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (layout.checkpoints_dir / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (layout.checkpoints_dir / "model.safetensors").write_bytes(b"0" * 1_100_000)
    layout.bootstrap_state_path.parent.mkdir(parents=True, exist_ok=True)
    layout.bootstrap_state_path.write_text(
        (
            "{\n"
            '  "vendor_project": {\n'
            '    "source_pyproject_hash": "pyproject-hash",\n'
            '    "source_lock_hash": "lock-hash",\n'
            '    "python_executable": "C:/Python314/python.exe"\n'
            "  }\n"
            "}\n"
        ),
        encoding="utf-8",
    )

    recorded_commands: list[list[str]] = []

    def _fake_run(
        command: list[str],
        cwd: str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text, encoding, errors
        recorded_commands.append(command)
        if command[:5] == ["uv", "sync", "--locked", "--python", "python312"]:
            assert cwd == str(layout.vendor_runtime_dir)
            return subprocess.CompletedProcess(command, 0, "", "")
        raise AssertionError(f"Unexpected command: {command}")

    monkeypatch.setattr("card_framework.runtime.bootstrap.subprocess.run", _fake_run)
    monkeypatch.setattr(
        "card_framework.runtime.bootstrap.resolve_uv_executable",
        lambda *, uv_executable: uv_executable,
    )
    monkeypatch.setattr(
        "card_framework.runtime.bootstrap._project_fingerprints",
        lambda project_dir: {
            "pyproject_hash": "pyproject-hash",
            "lock_hash": "lock-hash",
        },
    )

    ensure_index_tts_runtime(
        layout=layout,
        uv_executable="uv",
        python_executable="python312",
    )

    assert recorded_commands == [["uv", "sync", "--locked", "--python", "python312"]]


def test_ensure_ctc_forced_aligner_runtime_installs_when_missing(monkeypatch) -> None:
    """Bootstrap the pinned aligner only when the module is absent."""
    recorded_commands: list[tuple[list[str], str | None]] = []

    monkeypatch.setattr(
        "card_framework.runtime.bootstrap.importlib.util.find_spec",
        lambda module_name: None if module_name == "ctc_forced_aligner" else object(),
    )
    monkeypatch.setattr(
        "card_framework.runtime.bootstrap.resolve_uv_executable",
        lambda *, uv_executable: "uv",
    )

    def _fake_run(
        command: list[str],
        cwd: str | None = None,
        check: bool = False,
        capture_output: bool = True,
        text: bool = True,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        del check, capture_output, text
        assert encoding == "utf-8"
        assert errors == "replace"
        recorded_commands.append((command, cwd))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("card_framework.runtime.bootstrap.subprocess.run", _fake_run)

    ensure_ctc_forced_aligner_runtime(python_executable="python")

    assert recorded_commands == [
        (
            [
                "uv",
                "pip",
                "install",
                "--python",
                "python",
                _CTC_FORCED_ALIGNER_ARCHIVE_URL,
            ],
            str(Path.cwd()),
        )
    ]


def test_resolve_uv_executable_uses_active_scripts_dir_when_path_lookup_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Find the packaged uv executable in the interpreter scripts directory."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    uv_path = scripts_dir / "uv.exe"
    uv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr("card_framework.runtime.bootstrap.shutil.which", lambda command: None)
    monkeypatch.setattr("card_framework.runtime.bootstrap.sysconfig.get_path", lambda key: str(scripts_dir))

    resolved = resolve_uv_executable(uv_executable="uv")

    assert resolved == str(uv_path.resolve())


def test_resolve_uv_executable_prefers_active_scripts_dir_over_path_lookup(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Prefer the active interpreter's uv over an unrelated PATH entry."""
    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    bundled_uv_path = scripts_dir / "uv.exe"
    bundled_uv_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        "card_framework.runtime.bootstrap.shutil.which",
        lambda command: str((tmp_path / "Python314" / "Scripts" / "uv.exe").resolve()),
    )
    monkeypatch.setattr(
        "card_framework.runtime.bootstrap.sysconfig.get_path",
        lambda key: str(scripts_dir),
    )

    resolved = resolve_uv_executable(uv_executable="uv")

    assert resolved == str(bundled_uv_path.resolve())


def test_vendored_indextts_lock_includes_py312_windows_numba_wheels() -> None:
    """Keep the vendored IndexTTS lock compatible with packaged Python 3.12 on Windows."""
    vendor_lock_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "card_framework"
        / "_vendor"
        / "index_tts"
        / "uv.lock"
    )
    lock_text = vendor_lock_path.read_text(encoding="utf-8")

    llvmlite_match = re.search(
        r'name = "llvmlite".*?cp312-cp312-win_amd64\.whl',
        lock_text,
        re.DOTALL,
    )
    numba_match = re.search(
        r'name = "numba".*?cp312-cp312-win_amd64\.whl',
        lock_text,
        re.DOTALL,
    )

    assert llvmlite_match is not None
    assert numba_match is not None
