"""Unit tests for QA input compatibility guard."""

from __future__ import annotations

from benchmark.qa_input_guard import evaluate_input_compatibility


def test_input_guard_marks_compatible_inputs() -> None:
    """Accept inputs when lexical overlap is above configured thresholds."""
    summary_xml = "<SPEAKER_00>Benjamin Felix says market forecasts are hard and active funds often underperform.</SPEAKER_00>"
    source_text = (
        "[E0001] [SPEAKER_00] Benjamin Felix says market forecasts are difficult.\n"
        "[E0002] [SPEAKER_01] Active funds often underperform the index."
    )
    result = evaluate_input_compatibility(
        summary_xml=summary_xml,
        source_text=source_text,
        min_overlap_ratio=0.2,
        min_shared_tokens=4,
        min_shared_distinctive_tokens=2,
        min_shared_name_phrases=1,
    )
    assert result.is_compatible is True
    assert result.shared_token_count >= 4


def test_input_guard_marks_mismatched_inputs() -> None:
    """Reject inputs when overlap is too low."""
    summary_xml = "<SPEAKER_00>Psychology in financial planning and risk tolerance scales.</SPEAKER_00>"
    source_text = (
        "[E0001] [SPEAKER_00] Episode 299 about stock market competition and indexing."
    )
    result = evaluate_input_compatibility(
        summary_xml=summary_xml,
        source_text=source_text,
        min_overlap_ratio=0.3,
        min_shared_tokens=5,
        min_shared_distinctive_tokens=3,
        min_shared_name_phrases=2,
    )
    assert result.is_compatible is False
    assert result.shared_token_count < 5


def test_input_guard_adapts_name_phrase_threshold_to_available_mentions() -> None:
    """Avoid false negatives when only one shared proper-name phrase is available."""
    summary_xml = "<SPEAKER_00>Welcome to the Rational Reminder Podcast with Dr. Charles Chafin discussing financial planning.</SPEAKER_00>"
    source_text = (
        "[E0001] [SPEAKER_00] Welcome to the Rational Reminder Podcast with Dr. Charles Chafin.\n"
        "[E0002] [SPEAKER_01] The discussion covers financial planning and investor behavior."
    )
    result = evaluate_input_compatibility(
        summary_xml=summary_xml,
        source_text=source_text,
        min_overlap_ratio=0.05,
        min_shared_tokens=6,
        min_shared_distinctive_tokens=2,
        min_shared_name_phrases=2,
    )
    assert result.is_compatible is True
    assert result.shared_name_phrase_count >= 1
