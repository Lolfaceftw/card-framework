"""Interactive setup wizard entrypoint for holdout split generation."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from benchmarks.voice_clone.holdout_wizard import run_holdout_wizard


def main() -> int:
    """Run the holdout split wizard entrypoint."""
    run_holdout_wizard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
