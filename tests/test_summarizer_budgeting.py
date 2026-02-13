"""Tests for summarizer word-budget semantics and token-cap heuristics."""

from __future__ import annotations

from audio2script_and_summarizer import summarizer
from audio2script_and_summarizer import summarizer_deepseek
from audio2script_and_summarizer.speaker_validation import ValidatedDialogueLine


def _line(text: str) -> ValidatedDialogueLine:
    """Build a minimal validated dialogue line for counting tests."""
    return ValidatedDialogueLine(
        speaker="SPEAKER_00",
        text=text,
        emo_text="Neutral",
        emo_alpha=0.6,
        source_segment_ids=["seg_00001"],
        validation_status="valid",
        repair_reason=None,
    )


def test_openai_count_words_uses_text_only_metric() -> None:
    """Count only the dialogue text words, not any JSON-like metadata."""
    total = summarizer._count_words(
        [  # noqa: SLF001
            _line("hello world"),
            _line("this has four words"),
        ]
    )
    assert total == 6


def test_deepseek_count_words_uses_text_only_metric() -> None:
    """Count only the dialogue text words, not any JSON-like metadata."""
    total = summarizer_deepseek._count_words(
        [  # noqa: SLF001
            _line("one two three"),
            _line("four"),
        ]
    )
    assert total == 4


def test_budget_retry_digest_contains_numeric_context() -> None:
    """Include concrete budget numbers in retry guidance."""
    digest = summarizer_deepseek._build_word_budget_retry_digest(  # noqa: SLF001
        total_words=1119,
        target_words=198,
        lower_bound=188,
        upper_bound=207,
    )
    assert "total=1119" in digest
    assert "target=198" in digest
    assert "range=[188,207]" in digest
    assert "delta=+921" in digest


def test_dynamic_token_cap_scales_with_budget() -> None:
    """Derive a reduced completion-token cap when word budget is set."""
    cap = summarizer_deepseek._derive_completion_token_cap(  # noqa: SLF001
        word_budget=198,
        configured_max_tokens=8192,
    )
    assert cap == 613


def test_dynamic_token_cap_uses_configured_when_budget_missing() -> None:
    """Keep configured max tokens when no word budget is provided."""
    cap = summarizer._derive_completion_token_cap(  # noqa: SLF001
        word_budget=None,
        configured_max_tokens=2048,
    )
    assert cap == 2048


def test_model_token_ceiling_for_reasoner_and_chat() -> None:
    """Apply model-specific token ceilings."""
    assert (
        summarizer_deepseek._model_completion_token_ceiling("deepseek-reasoner")  # noqa: SLF001
        == 64000
    )
    assert (
        summarizer_deepseek._model_completion_token_ceiling("deepseek-chat")  # noqa: SLF001
        == 8192
    )


def test_clamp_completion_tokens_for_model_caps_chat() -> None:
    """Cap chat model output tokens when configured limit is higher."""
    assert (
        summarizer_deepseek._clamp_completion_tokens_for_model(  # noqa: SLF001
            configured_max_tokens=64000,
            model="deepseek-chat",
        )
        == 8192
    )


def test_reasoning_replay_policy_forces_reasoner_tool_loop() -> None:
    """Force reasoning replay for reasoner tool loops even when toggle is off."""
    settings = summarizer_deepseek.DeepSeekRequestSettings(
        model="deepseek-reasoner",
        max_completion_tokens=64000,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=None,
        auto_beta=False,
        agent_tool_loop=True,
        agent_tool_mode="constraints_only",
        agent_persist_reasoning_content=False,
    )
    assert summarizer_deepseek._should_persist_reasoning_for_replay(  # noqa: SLF001
        settings,
        tool_mode="constraints_only",
    )


def test_reasoning_replay_policy_respects_toggle_without_tool_loop() -> None:
    """Keep replay disabled when no tool loop is active and toggle is off."""
    settings = summarizer_deepseek.DeepSeekRequestSettings(
        model="deepseek-reasoner",
        max_completion_tokens=64000,
        request_timeout_seconds=30.0,
        http_retries=1,
        temperature=None,
        auto_beta=False,
        agent_tool_loop=False,
        agent_tool_mode="off",
        agent_persist_reasoning_content=False,
    )
    assert not summarizer_deepseek._should_persist_reasoning_for_replay(  # noqa: SLF001
        settings,
        tool_mode="off",
    )
