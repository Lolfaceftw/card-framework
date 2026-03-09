from __future__ import annotations

import pytest

try:
    import jinja2 as _jinja2
except Exception:
    _jinja2 = None

if _jinja2 is None or getattr(_jinja2, "__spec__", None) is None:
    pytest.skip("jinja2 is required for prompt template rendering tests.", allow_module_level=True)

from card_framework.agents.loop_context import build_loop_context_prompt_block
from card_framework.shared.prompt_manager import PromptManager


def test_summarizer_user_prompt_appends_loop_context_only_when_present() -> None:
    prompt_without_context = PromptManager.get_prompt(
        "summarizer_user",
        num_segments=1,
        total_words=10,
        transcript_excerpt="[SPEAKER_00]: hello",
        loop_context_block="",
        feedback_block="",
    )
    assert "--- LOOP CONTEXT ---" not in prompt_without_context

    prompt_with_context = PromptManager.get_prompt(
        "summarizer_user",
        num_segments=1,
        total_words=10,
        transcript_excerpt="[SPEAKER_00]: hello",
        loop_context_block="Carry unresolved issue: chronology mismatch.",
        feedback_block="",
    )
    assert "--- LOOP CONTEXT ---" in prompt_with_context
    assert "chronology mismatch" in prompt_with_context


def test_loop_context_prompt_block_truncates_large_payload_safely() -> None:
    sentinel = "DO_NOT_LEAK_THIS_TAIL"
    raw = ("A" * 12_000) + sentinel

    bounded = build_loop_context_prompt_block(raw, char_cap=1024)

    assert len(bounded) <= 1_080
    assert sentinel not in bounded
    assert "[truncated" in bounded


