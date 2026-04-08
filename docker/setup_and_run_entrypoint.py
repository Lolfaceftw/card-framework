"""Docker entrypoint for the repo checkout setup_and_run workflow."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def main(argv: list[str] | None = None) -> int:
    """Run setup_and_run from the mounted repository checkout."""
    resolved_argv = list(sys.argv[1:] if argv is None else argv)
    if not resolved_argv and not (sys.stdin.isatty() and sys.stdout.isatty()):
        resolved_argv = ["--help"]
    repo_dir = Path(
        os.environ.get("CARD_FRAMEWORK_REPO_DIR", "/workspace/card-framework")
    ).resolve()
    pyproject_path = repo_dir / "pyproject.toml"

    if not pyproject_path.exists():
        print(
            (
                "Expected a card-framework checkout at "
                f"'{repo_dir}', but '{pyproject_path}' was not found. "
                "Run this image through `docker compose run ...` from the repo root "
                "so the checkout is bind-mounted into the container."
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2

    os.chdir(repo_dir)
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "card_framework.cli.setup_and_run",
        *resolved_argv,
    ]
    return subprocess.call(command, env=os.environ.copy())


if __name__ == "__main__":
    raise SystemExit(main())
