"""Speaker alignment utilities for timed ASR segments."""

from __future__ import annotations

from card_framework.audio_pipeline.contracts import DiarizationTurn, TimedTextSegment, TranscriptSegment


def align_segments_with_speakers(
    asr_segments: list[TimedTextSegment],
    diarization_turns: list[DiarizationTurn],
    *,
    default_speaker: str = "SPEAKER_00",
    merge_gap_ms: int = 800,
) -> list[TranscriptSegment]:
    """
    Assign speakers to ASR segments and merge adjacent compatible segments.

    Args:
        asr_segments: Ordered ASR segments with timing.
        diarization_turns: Diarization speaker activity turns.
        default_speaker: Speaker label used when diarization is unavailable.
        merge_gap_ms: Max inter-segment gap to merge same-speaker spans.

    Returns:
        Speaker-attributed transcript segments.
    """
    if not asr_segments:
        return []

    turns = sorted(diarization_turns, key=lambda turn: turn.start_time_ms)
    attributed: list[TranscriptSegment] = []
    previous_speaker = default_speaker

    for segment in asr_segments:
        speaker, overlap_speakers = _pick_speaker_for_segment(
            segment=segment,
            turns=turns,
            fallback_speaker=previous_speaker,
            default_speaker=default_speaker,
        )
        previous_speaker = speaker
        attributed.append(
            TranscriptSegment(
                speaker=speaker,
                start_time=segment.start_time_ms,
                end_time=segment.end_time_ms,
                text=segment.text.strip(),
                extras=(
                    {"overlap_speakers": list(overlap_speakers)}
                    if overlap_speakers
                    else {}
                ),
            )
        )

    return _merge_adjacent_segments(attributed, merge_gap_ms=merge_gap_ms)


def _pick_speaker_for_segment(
    *,
    segment: TimedTextSegment,
    turns: list[DiarizationTurn],
    fallback_speaker: str,
    default_speaker: str,
) -> tuple[str, tuple[str, ...]]:
    """Pick a speaker by maximum overlap; fallback to nearest or previous."""
    if not turns:
        return default_speaker, ()

    best_speaker = ""
    best_overlap_ms = -1
    overlapping_speakers: list[str] = []
    for turn in turns:
        overlap_ms = min(segment.end_time_ms, turn.end_time_ms) - max(
            segment.start_time_ms, turn.start_time_ms
        )
        if overlap_ms > 0 and turn.speaker.strip():
            overlapping_speakers.append(turn.speaker)
        if overlap_ms > best_overlap_ms:
            best_overlap_ms = overlap_ms
            best_speaker = turn.speaker

    if best_overlap_ms > 0:
        return (
            best_speaker,
            tuple(
                speaker
                for speaker in dict.fromkeys(overlapping_speakers)
                if speaker.strip() and speaker != best_speaker
            ),
        )

    nearest_turn = min(
        turns,
        key=lambda turn: min(
            abs(segment.start_time_ms - turn.end_time_ms),
            abs(segment.end_time_ms - turn.start_time_ms),
        ),
    )
    if nearest_turn.speaker.strip():
        return nearest_turn.speaker, ()
    if fallback_speaker.strip():
        return fallback_speaker, ()
    return default_speaker, ()


def _merge_adjacent_segments(
    segments: list[TranscriptSegment], *, merge_gap_ms: int
) -> list[TranscriptSegment]:
    """Merge adjacent segments for readability when speaker continuity holds."""
    if not segments:
        return []

    merged: list[TranscriptSegment] = [segments[0]]
    for current in segments[1:]:
        previous = merged[-1]
        same_speaker = previous.speaker == current.speaker
        same_overlap = _segment_overlap_key(previous) == _segment_overlap_key(current)
        gap_ms = current.start_time - previous.end_time
        if same_speaker and same_overlap and gap_ms <= merge_gap_ms:
            merged[-1] = TranscriptSegment(
                speaker=previous.speaker,
                start_time=previous.start_time,
                end_time=current.end_time,
                text=f"{previous.text} {current.text}".strip(),
                extras=dict(previous.extras),
            )
            continue
        merged.append(current)
    return merged


def _segment_overlap_key(segment: TranscriptSegment) -> tuple[str, ...]:
    raw_value = segment.extras.get("overlap_speakers")
    if not isinstance(raw_value, list | tuple):
        return ()
    return tuple(
        str(speaker).strip()
        for speaker in raw_value
        if str(speaker).strip()
    )

