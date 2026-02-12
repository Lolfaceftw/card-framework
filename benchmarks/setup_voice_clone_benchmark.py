"""Interactive setup wizard entrypoint for voice-clone benchmarking."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from benchmarks.voice_clone.wizard import run_wizard


if __name__ == "__main__":
    raise SystemExit(run_wizard())

