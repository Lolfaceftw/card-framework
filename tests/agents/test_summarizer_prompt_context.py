from __future__ import annotations

import pytest

try:
    import jinja2 as _jinja2
except Exception:
    _jinja2 = None

if _jinja2 is None or getattr(_jinja2, "__spec__", None) is None:
    pytest.skip(
        "jinja2 is required for prompt template rendering tests.",
        allow_module_level=True,
    )

from card_framework.agents.loop.loop_context import build_loop_context_prompt_block
from card_framework.agents.loop.message_registry import MessageRegistry
from card_framework.agents.summarizer import SummarizerExecutor
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


def test_loop_context_prompt_block_is_unbounded_by_default() -> None:
    sentinel = "DO_NOT_LEAK_THIS_TAIL"
    raw = ("A" * 12_000) + sentinel

    unbounded = build_loop_context_prompt_block(raw)

    assert unbounded == raw
    assert sentinel in unbounded
    assert "[truncated" not in unbounded


def test_loop_context_prompt_block_can_still_truncate_when_explicitly_capped() -> None:
    sentinel = "DO_NOT_LEAK_THIS_TAIL"
    raw = ("A" * 12_000) + sentinel

    bounded = build_loop_context_prompt_block(raw, char_cap=1024)

    assert len(bounded) <= 1_080
    assert sentinel not in bounded
    assert "[truncated" in bounded


def test_full_transcript_prompt_excerpt_keeps_small_transcript_unchanged() -> None:
    transcript = "\n".join(
        [
            "[SPEAKER_00]: hello there",
            "[SPEAKER_01]: general kenobi",
            "[SPEAKER_00]: this is a short exchange",
        ]
    )

    excerpt = SummarizerExecutor._build_full_transcript_prompt_text(transcript)

    assert excerpt == transcript


def test_full_transcript_prompt_excerpt_keeps_large_transcript_unchanged() -> None:
    lines = [
        f"[SPEAKER_{index % 2:02d}]: line {index} " + ("context " * 80)
        for index in range(90)
    ]
    transcript = "\n".join(lines)

    excerpt = SummarizerExecutor._build_full_transcript_prompt_text(transcript)

    assert excerpt == transcript


def test_load_draft_into_registry_ignores_malformed_xml() -> None:
    registry = MessageRegistry()

    SummarizerExecutor._load_draft_into_registry(
        registry,
        '<SPEAKER_00 emo_preset="neutral">missing close tag',
    )

    assert len(registry) == 0


def test_summarizer_excerpt_segment_includes_affect_hint_when_present() -> None:
    rendered = SummarizerExecutor._format_transcript_excerpt_segment(
        {
            "speaker": "SPEAKER_00",
            "text": "hello world",
            "affect_hint": "preset=urgent; top=angry:0.75",
        }
    )

    assert (
        rendered
        == "[SPEAKER_00] [affect: preset=urgent; top=angry:0.75]: hello world\n"
    )

