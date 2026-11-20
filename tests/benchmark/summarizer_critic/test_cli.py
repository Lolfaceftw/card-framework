"""CLI helper coverage for the summarizer-critic benchmark."""

from __future__ import annotations

import os

from card_framework.benchmark.summarizer_critic.cli import _child_environment
from card_framework.shared.paths import REPO_ROOT


def test_child_environment_enables_unbuffered_python_output(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    env = _child_environment()

    assert env["PYTHONUNBUFFERED"] == "1"
    pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)
    assert str(REPO_ROOT / "src") in pythonpath_entries
    assert "existing-path" in pythonpath_entries
