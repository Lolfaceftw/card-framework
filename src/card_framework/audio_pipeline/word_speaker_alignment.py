"""Word-level speaker alignment utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import re

from card_framework.audio_pipeline.contracts import (
    DiarizationTurn,
    TranscriptSegment,
    WordSpeakerToken,
    WordTimestamp,
)

_SENTENCE_ENDING_PUNCTUATION = ".?!"
_MODEL_PUNCTUATION = ".,;:!?"


def build_word_speaker_segments(
    *,
    word_timestamps: list[WordTimestamp],
    diarization_turns: list[DiarizationTurn],
    default_speaker: str,
    language: str,
    merge_gap_ms: int,
    restore_punctuation_model: bool = True,
) -> list[TranscriptSegment]:
    """
    Build speaker-attributed transcript segments from aligned words.

    Args:
        word_timestamps: Aligned words with timing.
        diarization_turns: Speaker diarization activity timeline.
        default_speaker: Fallback speaker label.
        language: Transcription language for punctuation model gating.
        merge_gap_ms: Merge threshold for adjacent same-speaker segments.
        restore_punctuation_model: Enable punctuation restoration model pass.

    Returns:
        Speaker-attributed transcript segments.
    """
    word_tokens = map_words_to_speakers(
        word_timestamps=word_timestamps,
        diarization_turns=diarization_turns,
        default_speaker=default_speaker,
    )
    if not word_tokens:
        return []

    punctuated = maybe_restore_punctuation(
        word_tokens=word_tokens,
        language=language,
        enabled=restore_punctuation_model,
    )
    realigned = realign_speakers_with_punctuation(punctuated)
    segmented = group_sentence_segments(realigned)
    return merge_adjacent_segments(segmented, merge_gap_ms=merge_gap_ms)


def map_words_to_speakers(
    *,
    word_timestamps: list[WordTimestamp],
    diarization_turns: list[DiarizationTurn],
    default_speaker: str,
    anchor: str = "start",
) -> list[WordSpeakerToken]:
    """Assign speakers to words using diarization overlap, with anchor fallback."""
    if not word_timestamps:
        return []
    if not diarization_turns:
        return [
            WordSpeakerToken(
                word=word.word,
                speaker=default_speaker,
                start_time_ms=word.start_time_ms,
                end_time_ms=word.end_time_ms,
            )
            for word in word_timestamps
        ]

    turns = sorted(diarization_turns, key=lambda turn: turn.start_time_ms)
    turn_index = 0
    tokens: list[WordSpeakerToken] = []
    previous_speaker = default_speaker

    for word in word_timestamps:
        while turn_index < len(turns) - 1 and turns[turn_index].end_time_ms < word.start_time_ms:
            turn_index += 1

        best_overlap_ms = -1
        best_speaker = ""
        candidate_index = turn_index
        while candidate_index < len(turns):
            candidate = turns[candidate_index]
            if candidate.start_time_ms > word.end_time_ms:
                break
            overlap_ms = _interval_overlap_ms(
                start_a=word.start_time_ms,
                end_a=word.end_time_ms,
                start_b=candidate.start_time_ms,
                end_b=candidate.end_time_ms,
            )
            if overlap_ms > best_overlap_ms:
                best_overlap_ms = overlap_ms
                best_speaker = candidate.speaker
            candidate_index += 1

        if best_overlap_ms > 0 and best_speaker.strip():
            resolved_speaker = best_speaker
        else:
            anchor_time = _word_anchor(word.start_time_ms, word.end_time_ms, anchor=anchor)
            nearest_turn = min(
                turns,
                key=lambda turn: min(
                    abs(anchor_time - turn.start_time_ms),
                    abs(anchor_time - turn.end_time_ms),
                ),
            )
            if nearest_turn.speaker.strip():
                resolved_speaker = nearest_turn.speaker
            elif previous_speaker.strip():
                resolved_speaker = previous_speaker
            else:
                resolved_speaker = default_speaker

        tokens.append(
            WordSpeakerToken(
                word=word.word,
                speaker=resolved_speaker,
                start_time_ms=word.start_time_ms,
                end_time_ms=word.end_time_ms,
            )
        )
        previous_speaker = resolved_speaker
    return tokens


def maybe_restore_punctuation(
    *,
    word_tokens: list[WordSpeakerToken],
    language: str,
    enabled: bool,
) -> list[WordSpeakerToken]:
    """Restore punctuation using deepmultilingualpunctuation when available."""
    if not enabled or not word_tokens:
        return word_tokens
    if not language.strip():
        return word_tokens

    try:
        from deepmultilingualpunctuation import PunctuationModel  # type: ignore[import-not-found]
    except Exception:
        return word_tokens

    try:
        punct_model = PunctuationModel(model="kredor/punctuate-all")
        predictions = punct_model.predict([token.word for token in word_tokens], chunk_size=230)
    except Exception:
        return word_tokens

    restored: list[WordSpeakerToken] = []
    for token, prediction in zip(word_tokens, predictions):
        punct_label = str(prediction[1]) if len(prediction) > 1 else ""
        word = token.word
        if (
            punct_label in _SENTENCE_ENDING_PUNCTUATION
            and (not word or word[-1] not in _MODEL_PUNCTUATION or _is_acronym(word))
        ):
            word = f"{word}{punct_label}"
            if word.endswith(".."):
                word = word.rstrip(".")
        restored.append(replace(token, word=word))
    if len(restored) < len(word_tokens):
        restored.extend(word_tokens[len(restored) :])
    return restored


def realign_speakers_with_punctuation(
    word_tokens: list[WordSpeakerToken],
    *,
    max_words_in_sentence: int = 50,
    max_relabel_window_words: int = 14,
    max_flip_span_words: int = 2,
    min_dominant_ratio: float = 0.75,
) -> list[WordSpeakerToken]:
    """Realign short speaker flips that split single sentence spans."""
    if not word_tokens:
        return []

    words = [token.word for token in word_tokens]
    speakers = [token.speaker for token in word_tokens]
    index = 0
    total = len(word_tokens)

    while index < total - 1:
        if speakers[index] == speakers[index + 1] or _is_sentence_end(words[index]):
            index += 1
            continue

        left = _first_word_index_of_sentence(index, words, speakers, max_words_in_sentence)
        right = (
            _last_word_index_of_sentence(index, words, max_words_in_sentence - index + left - 1)
            if left >= 0
            else -1
        )
        if min(left, right) == -1:
            index += 1
            continue

        window_size = right - left + 1
        if window_size > max_relabel_window_words:
            index += 1
            continue

        window_speakers = speakers[left : right + 1]
        dominant_speaker, dominant_count = Counter(window_speakers).most_common(1)[0]
        if dominant_count / window_size < min_dominant_ratio:
            index += 1
            continue

        minority_indices = [
            idx for idx, speaker in enumerate(window_speakers) if speaker != dominant_speaker
        ]
        if not minority_indices:
            index += 1
            continue

        minority_start = minority_indices[0]
        minority_end = minority_indices[-1]
        minority_span = minority_end - minority_start + 1
        if minority_span != len(minority_indices):
            index += 1
            continue
        if minority_span > max_flip_span_words:
            index += 1
            continue
        if minority_start == 0 or minority_end == window_size - 1:
            index += 1
            continue
        if (
            window_speakers[minority_start - 1] != dominant_speaker
            or window_speakers[minority_end + 1] != dominant_speaker
        ):
            index += 1
            continue

        for relabel_offset in range(minority_start, minority_end + 1):
            speakers[left + relabel_offset] = dominant_speaker
        index = right + 1

    return [
        replace(token, speaker=speakers[i])
        for i, token in enumerate(word_tokens)
    ]


def group_sentence_segments(word_tokens: list[WordSpeakerToken]) -> list[TranscriptSegment]:
    """Group aligned word tokens into speaker-attributed sentence segments."""
    if not word_tokens:
        return []

    sentence_break = _sentence_breaker()
    segments: list[TranscriptSegment] = []

    current_speaker = word_tokens[0].speaker
    current_start = word_tokens[0].start_time_ms
    current_end = word_tokens[0].end_time_ms
    current_words: list[str] = [word_tokens[0].word]

    for token in word_tokens[1:]:
        candidate_text = f"{' '.join(current_words)} {token.word}".strip()
        speaker_changed = token.speaker != current_speaker
        split_sentence = sentence_break(candidate_text)
        if speaker_changed or split_sentence:
            segments.append(
                TranscriptSegment(
                    speaker=current_speaker,
                    start_time=current_start,
                    end_time=current_end,
                    text=" ".join(current_words).strip(),
                )
            )
            current_speaker = token.speaker
            current_start = token.start_time_ms
            current_words = [token.word]
            current_end = token.end_time_ms
            continue

        current_words.append(token.word)
        current_end = token.end_time_ms

    segments.append(
        TranscriptSegment(
            speaker=current_speaker,
            start_time=current_start,
            end_time=current_end,
            text=" ".join(current_words).strip(),
        )
    )
    return segments


def merge_adjacent_segments(
    segments: list[TranscriptSegment],
    *,
    merge_gap_ms: int,
) -> list[TranscriptSegment]:
    """Merge adjacent segments for readability when speaker continuity holds."""
    if not segments:
        return []

    merged: list[TranscriptSegment] = [segments[0]]
    for current in segments[1:]:
        previous = merged[-1]
        same_speaker = previous.speaker == current.speaker
        gap_ms = current.start_time - previous.end_time
        if same_speaker and gap_ms <= merge_gap_ms:
            merged[-1] = TranscriptSegment(
                speaker=previous.speaker,
                start_time=previous.start_time,
                end_time=current.end_time,
                text=f"{previous.text} {current.text}".strip(),
            )
            continue
        merged.append(current)
    return merged


def _word_anchor(start_ms: int, end_ms: int, *, anchor: str) -> int:
    if anchor == "end":
        return end_ms
    if anchor == "mid":
        return (start_ms + end_ms) // 2
    return start_ms


def _interval_overlap_ms(
    *,
    start_a: int,
    end_a: int,
    start_b: int,
    end_b: int,
) -> int:
    """Return interval overlap in milliseconds."""
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _is_sentence_end(word: str) -> bool:
    return bool(word) and word[-1] in _SENTENCE_ENDING_PUNCTUATION


def _first_word_index_of_sentence(
    word_index: int,
    words: list[str],
    speakers: list[str],
    max_words: int,
) -> int:
    left = word_index
    while (
        left > 0
        and word_index - left < max_words
        and speakers[left - 1] == speakers[left]
        and not _is_sentence_end(words[left - 1])
    ):
        left -= 1
    return left if left == 0 or _is_sentence_end(words[left - 1]) else -1


def _last_word_index_of_sentence(
    word_index: int,
    words: list[str],
    max_words: int,
) -> int:
    right = word_index
    while (
        right < len(words) - 1
        and right - word_index < max_words
        and not _is_sentence_end(words[right])
    ):
        right += 1
    return right if right == len(words) - 1 or _is_sentence_end(words[right]) else -1


def _is_acronym(value: str) -> bool:
    return bool(re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", value))


def _sentence_breaker():
    try:
        import nltk  # type: ignore[import-not-found]

        return nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak
    except Exception:
        return lambda text: text.strip().endswith(tuple(_SENTENCE_ENDING_PUNCTUATION))

