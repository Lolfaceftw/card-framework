"""Compatibility facade for pipeline orchestration.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType


def _load_orchestrator_module() -> ModuleType:
    """Load and return the canonical orchestration module."""
    if __package__:
        from .pipeline import orchestrator as orchestrator_module

        return orchestrator_module

    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from audio2script_and_summarizer.pipeline import orchestrator as orchestrator_module

    return orchestrator_module


_ORCHESTRATOR = _load_orchestrator_module()
sys.modules[__name__] = _ORCHESTRATOR

if __name__ == "__main__":
    raise SystemExit(_ORCHESTRATOR.main())
