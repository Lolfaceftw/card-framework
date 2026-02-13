"""Unit tests for Stage 3 interjection decision logic."""

from __future__ import annotations

import random

from audio2script_and_summarizer.stage3_voice import (
    InterjectionCandidate,
    SummaryEntry,
    compute_interjection_position_ms,
    select_interjections_by_confidence,
    select_nearest_alternate_speaker_index,
)


def _entry(speaker: str) -> SummaryEntry:
    """Build a minimal summary entry."""
    return SummaryEntry(
        speaker=speaker,
        voice_sample=f"{speaker}.wav",
        text="Example segment text",
        use_emo_text=True,
        emo_text="Neutral",
        emo_alpha=0.6,
    )


def test_select_nearest_alternate_speaker_index_prefers_previous_on_tie() -> None:
    """Pick nearest alternate speaker and break ties toward previous."""
    entries = [_entry("A"), _entry("B"), _entry("A"), _entry("B"), _entry("A")]
    # For index 2 ("A"), previous alternate is 1, next alternate is 3.
    assert select_nearest_alternate_speaker_index(entries, 2) == 1


def test_select_interjections_by_confidence_applies_ratio_cap() -> None:
    """Keep only highest confidence candidates under global cap."""
    candidates = [
        InterjectionCandidate(1, 0, "uh-huh", "line", "agreement", 0.55),
        InterjectionCandidate(2, 1, "hmm", "segment", "empathy", 0.90),
        InterjectionCandidate(3, 2, "right", "text", "agreement", 0.70),
    ]
    selected = select_interjections_by_confidence(
        candidates=candidates,
        eligible_segment_count=10,
        max_ratio=0.2,
    )
    # floor(10 * 0.2) = 2 => top two confidences kept.
    assert [item.segment_index for item in selected] == [2, 3]


def test_compute_interjection_position_ms_clamps_inside_audio_window() -> None:
    """Clamp computed overlay position to valid range."""
    rng = random.Random(7)
    position_ms = compute_interjection_position_ms(
        text="This is a short segment with a clear anchor",
        anchor_phrase="clear anchor",
        style="surprise",
        audio_duration_ms=1000,
        rng=rng,
    )
    assert 0 <= position_ms <= 500
