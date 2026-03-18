from card_framework.audio_pipeline.contracts import (
    DiarizationTurn,
    TranscriptSegment,
    WordSpeakerToken,
    WordTimestamp,
)
from card_framework.audio_pipeline.word_speaker_alignment import (
    build_word_speaker_segments,
    map_words_to_speakers,
    realign_speakers_with_punctuation,
    smooth_short_sandwich_segments,
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


def test_realign_speakers_with_punctuation_relabels_mid_sentence_one_word_flip() -> None:
    tokens = [
        WordSpeakerToken(word="I", speaker="SPEAKER_01", start_time_ms=0, end_time_ms=100),
        WordSpeakerToken(word="really", speaker="SPEAKER_01", start_time_ms=100, end_time_ms=200),
        WordSpeakerToken(word="think", speaker="SPEAKER_01", start_time_ms=200, end_time_ms=300),
        WordSpeakerToken(word="this", speaker="SPEAKER_01", start_time_ms=300, end_time_ms=400),
        WordSpeakerToken(word="is", speaker="SPEAKER_01", start_time_ms=400, end_time_ms=500),
        WordSpeakerToken(word="a", speaker="SPEAKER_01", start_time_ms=500, end_time_ms=600),
        WordSpeakerToken(word="very", speaker="SPEAKER_01", start_time_ms=600, end_time_ms=700),
        WordSpeakerToken(word="good", speaker="SPEAKER_01", start_time_ms=700, end_time_ms=800),
        WordSpeakerToken(word="definition", speaker="SPEAKER_01", start_time_ms=800, end_time_ms=900),
        WordSpeakerToken(word="of", speaker="SPEAKER_01", start_time_ms=900, end_time_ms=1000),
        WordSpeakerToken(word="it", speaker="SPEAKER_01", start_time_ms=1000, end_time_ms=1100),
        WordSpeakerToken(word="and", speaker="SPEAKER_01", start_time_ms=1100, end_time_ms=1200),
        WordSpeakerToken(word="I", speaker="SPEAKER_05", start_time_ms=1200, end_time_ms=1300),
        WordSpeakerToken(word="think", speaker="SPEAKER_01", start_time_ms=1300, end_time_ms=1400),
        WordSpeakerToken(word="we", speaker="SPEAKER_01", start_time_ms=1400, end_time_ms=1500),
        WordSpeakerToken(word="should", speaker="SPEAKER_01", start_time_ms=1500, end_time_ms=1600),
        WordSpeakerToken(word="keep", speaker="SPEAKER_01", start_time_ms=1600, end_time_ms=1700),
        WordSpeakerToken(word="going.", speaker="SPEAKER_01", start_time_ms=1700, end_time_ms=1800),
    ]

    realigned = realign_speakers_with_punctuation(tokens)

    assert all(token.speaker == "SPEAKER_01" for token in realigned)


def test_smooth_short_sandwich_segments_relabels_zero_duration_single_word() -> None:
    smoothed = smooth_short_sandwich_segments(
        [
            TranscriptSegment(
                speaker="SPEAKER_01",
                start_time=0,
                end_time=1000,
                text="very good definition of it and",
            ),
            TranscriptSegment(
                speaker="SPEAKER_05",
                start_time=1000,
                end_time=1000,
                text="i",
            ),
            TranscriptSegment(
                speaker="SPEAKER_01",
                start_time=1020,
                end_time=2200,
                text="think we should keep going",
            ),
        ]
    )

    assert [segment.speaker for segment in smoothed] == [
        "SPEAKER_01",
        "SPEAKER_01",
        "SPEAKER_01",
    ]
