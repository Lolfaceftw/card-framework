# Granite Transcription Tuning

This note tracks repo-local Granite Speech tuning experiments so the active
chunking and decoding defaults stay explainable.

## 2026-03-18 - Fixed-vocals seam reduction pass

Context:
- Full Docker GPU stage-1 reruns of the same `audio.wav` can change Demucs
  `vocals.wav`, so Granite tuning needed to be isolated on a fixed separated
  artifact.
- The best full transcript at the start of this pass,
  `8426c902-0524-4b80-bbe5-197d2f3add2d`, had already eliminated the worst
  runaway `yeah` loops, but it still showed small seam leftovers such as
  `curve and then just ride the wave`, `index index`, and `hidden hidden`.

Baseline:
- Fixed vocals path:
  `artifacts/audio_stage/runs/8426c902-0524-4b80-bbe5-197d2f3add2d/separation/htdemucs/audio/vocals.wav`
- Granite-only Docker experiment:
  `granite-fixed-baseline`
- Effective settings:
  `chunk_length_seconds=25`
  `chunk_overlap_seconds=2`
  `max_new_tokens=96`
  `repetition_penalty=1.15`
- Summary:
  `segments_with_adjacent_repeat_ge2=34`
  `boundary_seam_candidate_count=32`

Experiment results:
- `granite-fixed-overlap1`
  `chunk_overlap_seconds=1`
  `segments_with_adjacent_repeat_ge2=32`
  `boundary_seam_candidate_count=11`
- `granite-fixed-overlap1-tightdecode`
  `chunk_overlap_seconds=1`
  `max_new_tokens=80`
  `repetition_penalty=1.2`
  `segments_with_adjacent_repeat_ge2=23`
  `boundary_seam_candidate_count=9`
- `granite-fixed-chunk30-overlap1-tightdecode`
  `chunk_length_seconds=30`
  `chunk_overlap_seconds=1`
  `max_new_tokens=80`
  `repetition_penalty=1.2`
  `segments_with_adjacent_repeat_ge2=24`
  `boundary_seam_candidate_count=9`
  `word_timestamp_count=13953`
- `granite-fixed-overlap1-tightdecode-rp125`
  `chunk_overlap_seconds=1`
  `max_new_tokens=80`
  `repetition_penalty=1.25`
  `segments_with_adjacent_repeat_ge2=14`
  `boundary_seam_candidate_count=7`
- `granite-fixed-overlap1-rp125-max96`
  `chunk_overlap_seconds=1`
  `max_new_tokens=96`
  `repetition_penalty=1.25`
  `segments_with_adjacent_repeat_ge2=14`
  `boundary_seam_candidate_count=7`
  `word_timestamp_count=14715`

Promoted defaults:
- `chunk_length_seconds: 25`
- `chunk_overlap_seconds: 1`
- `max_new_tokens: 96`
- `repetition_penalty: 1.25`

Validation:
- `granite-fixed-current-defaults`
  reran the checked-in defaults with no ASR tuning overrides beyond disabling
  forced alignment for speed and matched the promoted fixed-vocals result:
  `segments_with_adjacent_repeat_ge2=14`
  `boundary_seam_candidate_count=7`
  `word_timestamp_count=14715`

Rationale:
- Lowering overlap from `2s` to `1s` was the main seam fix. It cut
  seam-oriented duplicate candidates from `32` to `11` on the fixed-vocals
  baseline before any other Granite change.
- Raising `repetition_penalty` to `1.25` cleaned up most of the remaining
  repeated-token segments without needing a shorter `20s` or `30s` chunking
  strategy.
- Keeping `max_new_tokens` at `96` preserved more words than the tighter
  `80`-token cap while matching the same best duplication metrics on the fixed
  artifact.
