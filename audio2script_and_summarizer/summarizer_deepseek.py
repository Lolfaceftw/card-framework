"""Compatibility shim for DeepSeek summarizer internals.

This module preserves the historical import path
`audio2script_and_summarizer.summarizer_deepseek` while implementation now
lives in `audio2script_and_summarizer.deepseek.core`.
"""

from __future__ import annotations

import sys

from .deepseek import core as _core

if __name__ == "__main__":
    raise SystemExit(_core.main())

sys.modules[__name__] = _core
