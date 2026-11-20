"""Regression tests for vendored IndexTTS GPT sources."""

from __future__ import annotations

from pathlib import Path
import py_compile


VENDOR_GPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "card_framework"
    / "_vendor"
    / "index_tts"
    / "indextts"
    / "gpt"
)


def test_vendored_indextts_gpt_modules_compile_without_syntax_errors() -> None:
    """Compile vendored IndexTTS GPT modules without syntax errors."""
    for module_name in ("model.py", "model_v2.py"):
        py_compile.compile(VENDOR_GPT_DIR / module_name, doraise=True)
