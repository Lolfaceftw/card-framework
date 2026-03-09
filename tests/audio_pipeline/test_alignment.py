from card_framework.audio_pipeline.alignment import align_segments_with_speakers
from card_framework.audio_pipeline.contracts import DiarizationTurn, TimedTextSegment


def test_align_segments_assigns_overlap_and_merges_adjacent() -> None:
    asr_segments = [
        TimedTextSegment(start_time_ms=0, end_time_ms=1000, text="hello"),
        TimedTextSegment(start_time_ms=1100, end_time_ms=1800, text="world"),
        TimedTextSegment(start_time_ms=2500, end_time_ms=3200, text="again"),
    ]
    turns = [
        DiarizationTurn(speaker="SPEAKER_01", start_time_ms=0, end_time_ms=2000),
        DiarizationTurn(speaker="SPEAKER_02", start_time_ms=2000, end_time_ms=5000),
    ]

    aligned = align_segments_with_speakers(
        asr_segments=asr_segments,
        diarization_turns=turns,
        merge_gap_ms=500,
    )

    assert len(aligned) == 2
    assert aligned[0].speaker == "SPEAKER_01"
    assert aligned[0].text == "hello world"
    assert aligned[1].speaker == "SPEAKER_02"
    assert aligned[1].text == "again"


def test_align_segments_uses_default_when_no_diarization() -> None:
    asr_segments = [
        TimedTextSegment(start_time_ms=0, end_time_ms=900, text="test one"),
        TimedTextSegment(start_time_ms=1200, end_time_ms=1800, text="test two"),
    ]

    aligned = align_segments_with_speakers(
        asr_segments=asr_segments,
        diarization_turns=[],
    )

    assert len(aligned) == 1
    assert aligned[0].speaker == "SPEAKER_00"
    assert aligned[0].text == "test one test two"

