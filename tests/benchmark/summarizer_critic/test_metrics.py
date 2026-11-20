"""Metric coverage for the summarizer-critic benchmark."""

from __future__ import annotations

from card_framework.benchmark.summarizer_critic.metrics import (
    combine_naturalness,
    compute_composite_score,
    score_duration_control,
    score_speaker_attribution,
)


def test_duration_score_is_perfect_within_tolerance() -> None:
    assert (
        score_duration_control(
            target_seconds=120, candidate_seconds=124, tolerance_ratio=0.05
        )
        == 1.0
    )


def test_duration_score_decays_outside_tolerance() -> None:
    assert (
        score_duration_control(
            target_seconds=120, candidate_seconds=200, tolerance_ratio=0.05
        )
        < 0.5
    )


def test_speaker_attribution_rewards_matching_claimed_speaker() -> None:
    transcript: dict[str, object] = {
        "segments": [
            {"speaker": "Alice", "text": "budget timeline launch risk"},
            {"speaker": "Bob", "text": "prototype feedback customer demo"},
        ]
    }
    aligned = (
        '<HOST emo_preset="neutral">Alice covered the budget timeline launch.</HOST>'
    )
    misaligned = (
        '<HOST emo_preset="neutral">Bob covered the budget timeline launch.</HOST>'
    )
    assert score_speaker_attribution(aligned, transcript) > score_speaker_attribution(
        misaligned, transcript
    )


def test_speaker_attribution_is_neutral_when_no_speaker_is_mentioned() -> None:
    transcript: dict[str, object] = {
        "segments": [
            {"speaker": "Alice", "text": "budget timeline launch risk"},
            {"speaker": "Bob", "text": "prototype feedback customer demo"},
        ]
    }

    score = score_speaker_attribution(
        '<HOST emo_preset="neutral">The team discussed launch timing and customer feedback.</HOST>',
        transcript,
        reference_summary="The team discussed launch timing and customer feedback.",
    )

    assert score == 0.5


def test_naturalness_falls_back_to_summary_text_heuristics() -> None:
    score = combine_naturalness(
        None,
        summary_xml=(
            '<HOST emo_preset="neutral">The team launched the pilot. '
            "They reviewed customer feedback and set the next timeline.</HOST>"
        ),
    )

    assert score > 0.5


def test_composite_score_averages_dimensions() -> None:
    assert (
        compute_composite_score(
            duration_control=1.0,
            factualness=0.5,
            naturalness=0.5,
            speaker_attribution=1.0,
        )
        == 0.75
    )
