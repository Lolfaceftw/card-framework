"""Targeted tests for stage-4 interjector echo retiming helpers."""

from __future__ import annotations

import pytest

from card_framework.audio_pipeline.contracts import WordTimestamp
from card_framework.audio_pipeline.interjector import (
    InterjectionDecision,
    _MatchedPhraseSpan,
    _WordLevelRetimingStep,
    _EligibleTurn,
    _build_atempo_filter_chain,
    _build_word_level_echo_retiming_plan,
    _render_eligible_turns_block,
    _sanitize_echo_agreement_text,
    _validate_llm_decisions,
)
from card_framework.audio_pipeline.voice_clone_contracts import VoiceCloneTurn


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


def _eligible_turn(
    *,
    host_text: str,
    next_text: str = "Next full turn.",
) -> _EligibleTurn:
    return _EligibleTurn(
        host_turn_index=1,
        host_turn=VoiceCloneTurn(speaker="SPEAKER_00", text=host_text),
        next_turn=VoiceCloneTurn(speaker="SPEAKER_02", text=next_text),
    )


def test_sanitize_echo_agreement_text_trims_trailing_incomplete_endings() -> None:
    assert _sanitize_echo_agreement_text("own business. I was") == "own business."
    assert _sanitize_echo_agreement_text("my family situation and") == "my family situation and"


def test_validate_interjection_decisions_trims_dangling_echo_endings() -> None:
    decisions = _validate_llm_decisions(
        [
            InterjectionDecision(
                host_turn_index=1,
                should_interject=True,
                interjection_style="echo_agreement",
                interjection_speaker="SPEAKER_02",
                interjection_text="own business. I was",
                anchor_start_token_index=4,
                anchor_end_token_index=7,
            )
        ],
        eligible_turns=[
            _eligible_turn(
                host_text="she really was minding her own business. I was there that night"
            ),
        ],
        max_interjection_words=6,
    )

    accepted = decisions[0]
    assert accepted.should_interject is True
    assert accepted.anchor_text == "her own business."
    assert accepted.anchor_end_token_index == 6
    assert accepted.interjection_text == "own business."


def test_validate_interjection_decisions_rejects_nonmatching_echo_rewrites() -> None:
    decisions = _validate_llm_decisions(
        [
            InterjectionDecision(
                host_turn_index=1,
                should_interject=True,
                interjection_style="echo_agreement",
                interjection_speaker="SPEAKER_02",
                interjection_text="Right.",
                anchor_start_token_index=4,
                anchor_end_token_index=7,
            )
        ],
        eligible_turns=[
            _eligible_turn(
                host_text="she really was minding her own business. I was there that night"
            ),
        ],
        max_interjection_words=6,
    )

    assert decisions[0].should_interject is False


def test_validate_interjection_decisions_allows_early_echo_anchor_inside_echo_window() -> None:
    decisions = _validate_llm_decisions(
        [
            InterjectionDecision(
                host_turn_index=1,
                should_interject=True,
                interjection_style="echo_agreement",
                interjection_speaker="SPEAKER_02",
                interjection_text="dam broke",
                anchor_start_token_index=6,
                anchor_end_token_index=7,
            )
        ],
        eligible_turns=[
            _eligible_turn(
                host_text=(
                    "Thanks man it was like a dam broke and all the creativity "
                    "that had been trapped inside me finally came out in one rush"
                )
            ),
        ],
        max_interjection_words=5,
        min_host_progress_ratio=0.35,
        echo_min_host_progress_ratio=0.05,
    )

    accepted = decisions[0]
    assert accepted.should_interject is True
    assert accepted.interjection_text == "dam broke"


def test_validate_interjection_decisions_still_rejects_early_backchannel_anchor() -> None:
    decisions = _validate_llm_decisions(
        [
            InterjectionDecision(
                host_turn_index=1,
                should_interject=True,
                interjection_style="backchannel",
                interjection_speaker="SPEAKER_02",
                interjection_text="Yeah.",
                anchor_start_token_index=6,
                anchor_end_token_index=7,
            )
        ],
        eligible_turns=[
            _eligible_turn(
                host_text=(
                    "Thanks man it was like a dam broke and all the creativity "
                    "that had been trapped inside me finally came out in one rush"
                )
            ),
        ],
        max_interjection_words=5,
        min_host_progress_ratio=0.35,
        echo_min_host_progress_ratio=0.05,
    )

    assert decisions[0].should_interject is False


def test_render_eligible_turns_block_reports_echo_window_even_when_backchannel_window_is_none() -> None:
    block = _render_eligible_turns_block(
        [
            _eligible_turn(
                host_text="dam broke fast",
            )
        ],
        min_host_progress_ratio=0.80,
        echo_min_host_progress_ratio=0.05,
        max_host_progress_ratio=0.90,
    )

    assert "Preferred backchannel anchor token window" in block
    assert "none" in block
    assert "Preferred echo anchor token window" in block
    assert "0..1" in block
    assert "If a style's window is `none`, do not use that style on this turn." in block
