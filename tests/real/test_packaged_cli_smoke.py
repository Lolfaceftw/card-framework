"""Real subprocess smoke tests for packaged CLI entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_python(*args: str) -> subprocess.CompletedProcess[str]:
    """Run one real Python subprocess from the repository root."""
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    return subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env=env,
    )


@pytest.mark.integration
def test_main_help_smoke_for_packaged_entrypoint() -> None:
    """Run the packaged main help command as a real subprocess."""
    packaged_result = _run_python("-m", "card_framework.cli.main", "--help")

    assert packaged_result.returncode == 0, packaged_result.stderr

    packaged_output = packaged_result.stdout + packaged_result.stderr
    for token in (
        "main is powered by Hydra.",
        "Powered by Hydra",
        "runner_project_dir: src/card_framework/_vendor/index_tts",
    ):
        assert token in packaged_output


@pytest.mark.integration
def test_setup_and_run_help_smoke_for_packaged_entrypoint() -> None:
    """Run the packaged setup helper help command as a real subprocess."""
    packaged_result = _run_python("-m", "card_framework.cli.setup_and_run", "--help")

    assert packaged_result.returncode == 0, packaged_result.stderr

    packaged_output = packaged_result.stdout + packaged_result.stderr
    for token in ("--audio-path", "--voiceclone-from-summary", "--skip-repo-update"):
        assert token in packaged_output
