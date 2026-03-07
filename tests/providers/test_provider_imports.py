"""Regression tests for provider import side effects."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]


def _loaded_transformer_modules(import_statement: str) -> set[str]:
    """Return heavyweight transformer modules loaded by one subprocess import."""
    command = [
        "uv",
        "run",
        "python",
        "-c",
        (
            "import json, sys; "
            f"{import_statement}; "
            "mods = sorted("
            "name for name in sys.modules "
            "if name.startswith(('sentence_transformers', 'transformers'))"
            "); "
            "print(json.dumps(mods))"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    stdout = completed.stdout.strip()
    return set(json.loads(stdout or "[]"))


def test_importing_logging_provider_does_not_load_transformer_stacks() -> None:
    """Keep logging-provider imports cheap for remote-provider startup."""
    loaded_modules = _loaded_transformer_modules("import providers.logging_provider")
    assert loaded_modules == set()


def test_importing_deepseek_provider_does_not_load_transformer_stacks() -> None:
    """Avoid package-level fan-out when importing the active DeepSeek provider."""
    loaded_modules = _loaded_transformer_modules("import providers.deepseek_provider")
    assert loaded_modules == set()
