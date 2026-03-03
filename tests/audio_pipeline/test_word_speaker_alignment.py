from audio_pipeline.contracts import DiarizationTurn, WordSpeakerToken, WordTimestamp
from audio_pipeline.word_speaker_alignment import (
    build_word_speaker_segments,
    map_words_to_speakers,
    realign_speakers_with_punctuation,
)


def test_map_words_to_speakers_uses_diarization_turns() -> None:
    tokens = map_words_to_speakers(
        word_timestamps=[
            WordTimestamp(word="hello", start_time_ms=0, end_time_ms=300),
            WordTimestamp(word="there", start_time_ms=400, end_time_ms=700),
            WordTimestamp(word="again", start_time_ms=1600, end_time_ms=2000),
        ],
        diarization_turns=[
            DiarizationTurn(speaker="SPEAKER_00", start_time_ms=0, end_time_ms=1000),
            DiarizationTurn(speaker="SPEAKER_01", start_time_ms=1000, end_time_ms=3000),
        ],
        default_speaker="SPEAKER_99",
    )

    assert [token.speaker for token in tokens] == [
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_01",
    ]


def test_build_word_speaker_segments_falls_back_without_punctuation_model() -> None:
    segments = build_word_speaker_segments(
        word_timestamps=[
            WordTimestamp(word="hello", start_time_ms=0, end_time_ms=500),
            WordTimestamp(word="world.", start_time_ms=500, end_time_ms=1000),
        ],
        diarization_turns=[
            DiarizationTurn(speaker="SPEAKER_00", start_time_ms=0, end_time_ms=2000)
        ],
        default_speaker="SPEAKER_00",
        language="en",
        merge_gap_ms=800,
        restore_punctuation_model=False,
    )

    assert len(segments) == 1
    assert segments[0].speaker == "SPEAKER_00"
    assert segments[0].text == "hello world."


def test_map_words_to_speakers_prefers_max_overlap_over_start_anchor() -> None:
    tokens = map_words_to_speakers(
        word_timestamps=[
            WordTimestamp(word="bridge", start_time_ms=900, end_time_ms=1400),
        ],
        diarization_turns=[
            DiarizationTurn(speaker="SPEAKER_00", start_time_ms=0, end_time_ms=1000),
            DiarizationTurn(speaker="SPEAKER_01", start_time_ms=1000, end_time_ms=2000),
        ],
        default_speaker="SPEAKER_99",
    )

    assert [token.speaker for token in tokens] == ["SPEAKER_01"]


def test_realign_speakers_with_punctuation_preserves_edge_minority_turn() -> None:
    tokens = [
        WordSpeakerToken(word="Well", speaker="SPEAKER_00", start_time_ms=0, end_time_ms=100),
        WordSpeakerToken(word="I", speaker="SPEAKER_01", start_time_ms=100, end_time_ms=200),
        WordSpeakerToken(word="think", speaker="SPEAKER_01", start_time_ms=200, end_time_ms=300),
        WordSpeakerToken(word="so.", speaker="SPEAKER_01", start_time_ms=300, end_time_ms=400),
    ]

    realigned = realign_speakers_with_punctuation(tokens)

    assert [token.speaker for token in realigned] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_01",
        "SPEAKER_01",
    ]
