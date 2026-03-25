"""Targeted tests for stage-4 interjector echo retiming helpers."""

from __future__ import annotations

import pytest

from card_framework.audio_pipeline.contracts import WordTimestamp
from card_framework.audio_pipeline.interjector import (
    _MatchedPhraseSpan,
    _WordLevelRetimingStep,
    _build_atempo_filter_chain,
    _build_word_level_echo_retiming_plan,
)


def _phrase_span(*words: tuple[str, int, int], similarity: float = 1.0) -> _MatchedPhraseSpan:
    matched_words = tuple(
        WordTimestamp(word=word, start_time_ms=start_ms, end_time_ms=end_ms)
        for word, start_ms, end_ms in words
    )
    return _MatchedPhraseSpan(
        start_word_index=0,
        end_word_index=len(matched_words) - 1,
        start_time_ms=matched_words[0].start_time_ms,
        end_time_ms=matched_words[-1].end_time_ms,
        similarity=similarity,
        matched_words=matched_words,
    )


def test_build_word_level_echo_retiming_plan_speeds_up_each_word_segment() -> None:
    host_phrase = _phrase_span(
        ("dam", 0, 280),
        ("broke", 280, 640),
    )
    overlay_phrase = _phrase_span(
        ("damn", 0, 420),
        ("broke", 420, 1180),
    )

    plan = _build_word_level_echo_retiming_plan(
        host_phrase_span=host_phrase,
        overlay_phrase_span=overlay_phrase,
        target_host_fill_ratio=0.95,
        min_overflow_ratio=1.05,
        max_tempo=1.75,
        min_segment_ms=80,
    )

    assert [step.word for step in plan] == ["damn", "broke"]
    assert all(isinstance(step, _WordLevelRetimingStep) for step in plan)
    assert plan[0].source_duration_ms == 420
    assert plan[0].target_duration_ms == 266
    assert plan[0].applied_tempo_ratio == pytest.approx(1.5789, rel=1e-3)
    assert plan[1].source_duration_ms == 760
    assert plan[1].target_duration_ms == 342
    assert plan[1].applied_tempo_ratio == pytest.approx(1.75)


def test_build_word_level_echo_retiming_plan_requires_matching_word_counts() -> None:
    host_phrase = _phrase_span(
        ("dam", 0, 280),
        ("broke", 280, 640),
    )
    overlay_phrase = _phrase_span(
        ("damn", 0, 420),
        ("really", 420, 760),
        ("broke", 760, 1180),
    )

    plan = _build_word_level_echo_retiming_plan(
        host_phrase_span=host_phrase,
        overlay_phrase_span=overlay_phrase,
        target_host_fill_ratio=0.95,
        min_overflow_ratio=1.05,
        max_tempo=1.75,
        min_segment_ms=80,
    )

    assert plan == ()


def test_build_word_level_echo_retiming_plan_rejects_too_small_target_segments() -> None:
    host_phrase = _phrase_span(
        ("dam", 0, 100),
        ("broke", 100, 180),
    )
    overlay_phrase = _phrase_span(
        ("damn", 0, 150),
        ("broke", 150, 340),
    )

    plan = _build_word_level_echo_retiming_plan(
        host_phrase_span=host_phrase,
        overlay_phrase_span=overlay_phrase,
        target_host_fill_ratio=0.75,
        min_overflow_ratio=1.05,
        max_tempo=1.75,
        min_segment_ms=80,
    )

    assert plan == ()


def test_build_atempo_filter_chain_supports_ratios_above_two() -> None:
    assert _build_atempo_filter_chain(1.75) == "atempo=1.75000"
    assert _build_atempo_filter_chain(3.5) == "atempo=2.00000,atempo=1.75000"
