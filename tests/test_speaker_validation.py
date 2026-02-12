"""Unit tests for transcript-grounded speaker validation logic."""

from __future__ import annotations

from audio2script_and_summarizer.speaker_validation import (
    build_segment_speaker_map,
    collect_allowed_speakers,
    format_segments_for_prompt,
    load_transcript_segments,
    validate_and_repair_dialogue,
)


def test_load_transcript_segments_assigns_stable_ids() -> None:
    """Assign synthesized IDs when transcript segments do not include IDs."""
    payload = """
    {
      "segments": [
        {"speaker": "SPEAKER_00", "text": "Hello"},
        {"speaker": "SPEAKER_01", "text": "Hi there"}
      ]
    }
    """

    segments = load_transcript_segments(payload)

    assert len(segments) == 2
    assert segments[0].segment_id == "seg_00000"
    assert segments[1].segment_id == "seg_00001"
    assert format_segments_for_prompt(segments).startswith("[seg_00000|SPEAKER_00]: Hello")


def test_validate_and_repair_fixes_speaker_mismatch() -> None:
    """Repair speaker label using transcript evidence IDs."""
    payload = """
    {
      "segments": [
        {"speaker": "SPEAKER_00", "text": "Welcome back"},
        {"speaker": "SPEAKER_01", "text": "Thanks for having me"}
      ]
    }
    """
    segments = load_transcript_segments(payload)
    allowed_speakers = collect_allowed_speakers(segments)
    speaker_map = build_segment_speaker_map(segments)

    report = validate_and_repair_dialogue(
        [
            {
                "speaker": "SPEAKER_99",
                "text": "Welcome back to the show.",
                "emo_text": "Warm and upbeat",
                "emo_alpha": 0.7,
                "source_segment_ids": ["seg_00000"],
            }
        ],
        allowed_speakers=allowed_speakers,
        segment_speaker_map=speaker_map,
    )

    assert report.is_valid
    assert report.repaired_lines == 1
    assert report.lines[0].speaker == "SPEAKER_00"
    assert report.lines[0].validation_status == "repaired"
    assert report.lines[0].repair_reason is not None


def test_validate_and_repair_normalizes_short_segment_ids() -> None:
    """Normalize short zero-padded source IDs when canonical IDs exist."""
    payload = """
    {
      "segments": [
        {"speaker": "SPEAKER_00", "text": "Welcome back"}
      ]
    }
    """
    segments = load_transcript_segments(payload)
    report = validate_and_repair_dialogue(
        [
            {
                "speaker": "SPEAKER_00",
                "text": "Welcome back.",
                "emo_text": "Neutral",
                "emo_alpha": 0.6,
                "source_segment_ids": ["seg_000"],
            }
        ],
        allowed_speakers=collect_allowed_speakers(segments),
        segment_speaker_map=build_segment_speaker_map(segments),
    )

    assert report.is_valid
    assert report.lines[0].source_segment_ids == ["seg_00000"]


def test_validate_and_repair_keeps_existing_noncanonical_segment_id() -> None:
    """Keep transcript IDs unchanged when they already exist in source data."""
    payload = """
    {
      "segments": [
        {"segment_id": "seg_0004", "speaker": "SPEAKER_00", "text": "Welcome back"}
      ]
    }
    """
    segments = load_transcript_segments(payload)
    report = validate_and_repair_dialogue(
        [
            {
                "speaker": "SPEAKER_00",
                "text": "Welcome back.",
                "emo_text": "Neutral",
                "emo_alpha": 0.6,
                "source_segment_ids": ["seg_0004"],
            }
        ],
        allowed_speakers=collect_allowed_speakers(segments),
        segment_speaker_map=build_segment_speaker_map(segments),
    )

    assert report.is_valid
    assert report.lines[0].source_segment_ids == ["seg_0004"]


def test_validate_and_repair_fails_on_unknown_source_segment() -> None:
    """Reject lines that cite segment IDs not present in transcript."""
    payload = """
    {
      "segments": [
        {"speaker": "SPEAKER_00", "text": "Only one segment"}
      ]
    }
    """
    segments = load_transcript_segments(payload)
    report = validate_and_repair_dialogue(
        [
            {
                "speaker": "SPEAKER_00",
                "text": "Only one segment",
                "emo_text": "Neutral",
                "emo_alpha": 0.6,
                "source_segment_ids": ["seg_missing"],
            }
        ],
        allowed_speakers=collect_allowed_speakers(segments),
        segment_speaker_map=build_segment_speaker_map(segments),
    )

    assert not report.is_valid
    assert report.lines == []
    assert len(report.issues) == 1
    assert "unknown source_segment_ids" in report.issues[0]


def test_validate_and_repair_fails_on_multi_speaker_sources() -> None:
    """Reject lines that combine source segment IDs from multiple speakers."""
    payload = """
    {
      "segments": [
        {"speaker": "SPEAKER_00", "text": "Line A"},
        {"speaker": "SPEAKER_01", "text": "Line B"}
      ]
    }
    """
    segments = load_transcript_segments(payload)
    report = validate_and_repair_dialogue(
        [
            {
                "speaker": "SPEAKER_00",
                "text": "Invalid merge",
                "emo_text": "Confused",
                "emo_alpha": 0.6,
                "source_segment_ids": ["seg_00000", "seg_00001"],
            }
        ],
        allowed_speakers=collect_allowed_speakers(segments),
        segment_speaker_map=build_segment_speaker_map(segments),
    )

    assert not report.is_valid
    assert report.lines == []
    assert len(report.issues) == 1
    assert "span multiple speakers" in report.issues[0]
