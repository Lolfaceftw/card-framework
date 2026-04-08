"""Coverage for the Docker setup-and-run entrypoint wrapper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


def _load_entrypoint_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "docker" / "setup_and_run_entrypoint.py"
    )
    spec = importlib.util.spec_from_file_location("docker_entrypoint", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - loader failure
        raise AssertionError("Unable to load docker entrypoint module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_docker_entrypoint_defaults_to_help_when_no_tty(
    monkeypatch, tmp_path: Path
) -> None:
    """Non-interactive no-arg Docker launches should still print help by default."""
    module = _load_entrypoint_module()
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname='demo'\n", encoding="utf-8"
    )
    monkeypatch.setenv("CARD_FRAMEWORK_REPO_DIR", str(tmp_path))

    class _FakeStream:
        def isatty(self) -> bool:
            return False

    monkeypatch.setattr(sys, "stdin", _FakeStream())
    monkeypatch.setattr(sys, "stdout", _FakeStream())

    recorded: dict[str, object] = {}

    def _fake_call(command, env=None):
        recorded["command"] = tuple(command)
        recorded["env"] = env
        return 0

    monkeypatch.setattr(module.subprocess, "call", _fake_call)

    exit_code = module.main([])

    assert exit_code == 0
    assert recorded["command"] == (
        "uv",
        "run",
        "python",
        "-m",
        "card_framework.cli.setup_and_run",
        "--help",
    )
