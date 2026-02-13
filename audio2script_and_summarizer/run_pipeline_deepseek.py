"""
CARD Audio2Script + Summarizer Pipeline (DeepSeek Wrapper).

This wrapper forwards to run_pipeline.py while forcing --llm-provider deepseek.
"""

from __future__ import annotations

import sys

from audio2script_and_summarizer.run_pipeline import main as run_main


def _ensure_deepseek_flag() -> None:
    """Ensure the DeepSeek provider flag is present in argv."""
    if "--llm-provider" in sys.argv:
        return
    sys.argv.insert(1, "--llm-provider")
    sys.argv.insert(2, "deepseek")


if __name__ == "__main__":
    _ensure_deepseek_flag()
    raise SystemExit(run_main())
