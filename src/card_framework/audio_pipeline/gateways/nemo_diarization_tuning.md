# NeMo Diarization Tuning

This note tracks repo-local NeMo diarization tuning experiments so the active
gateway settings stay explainable.

## 2026-03-18 - Podcast micro-turn smoothing pass 1

Context:
- Granite transcript text improved after shorter chunking, but the latest NeMo
  RTTM for run `20260318_080507` still emitted many micro-turns.
- The goal of this pass is to reduce abrupt same-speaker flips without forcing
  a fixed speaker count for guest-heavy podcast episodes.

Parameters now surfaced in config:
- `vad_onset`
- `vad_offset`
- `vad_pad_onset`
- `vad_pad_offset`
- `vad_min_duration_on`
- `vad_min_duration_off`

Experiment values:
- `vad_onset: 0.8`
- `vad_offset: 0.6`
- `vad_pad_onset: 0.1`
- `vad_pad_offset: 0.0`
- `vad_min_duration_on: 0.2`
- `vad_min_duration_off: 0.6`

Changes from the previous effective behavior:
- `vad_pad_offset` moved from `-0.05` to `0.0`
- `vad_min_duration_on` moved from `0.0` to `0.2`
- `vad_min_duration_off` moved from `0.2` to `0.6`

Rationale:
- Removing the negative pad offset avoids clipping speech tails too
  aggressively.
- `vad_min_duration_on: 0.2` suppresses tiny speech bursts that often become
  isolated micro-turns.
- `vad_min_duration_off: 0.6` merges short nonspeech gaps that can otherwise
  fragment one speaker into multiple turns.

Notes:
- Keep `min_speakers` and `max_speakers` unset by default for flexible
  podcast episodes with occasional guests.
- This repository still starts from NeMo's telephonic MSDD template file, so
  these values are a podcast-oriented override layer on top of that template.

## 2026-03-18 - Podcast micro-turn smoothing pass 2

Context:
- The latest rerun `b907642b-b50d-4c8a-bfed-9897903a3b64` greatly reduced raw
  RTTM fragmentation, but the transcript still showed a severe repeated
  `yeah` burst and a small number of one-token speaker flips.

Updated values:
- Granite `max_new_tokens: 128`
- `vad_onset: 0.8`
- `vad_offset: 0.6`
- `vad_pad_onset: 0.1`
- `vad_pad_offset: 0.05`
- `vad_min_duration_on: 0.3`
- `vad_min_duration_off: 0.8`

Rationale:
- Lower Granite `max_new_tokens` to reduce runaway repeated-token tails.
- Add a small positive `vad_pad_offset` so turn endings are less likely to get
  clipped into abrupt speaker switches.
- Raise the minimum speech and nonspeech durations again because the remaining
  diarization defects are now mostly tiny word-level blips rather than large
  speaker fragmentation.

## 2026-03-18 - Podcast micro-turn smoothing pass 3

Context:
- The latest rerun `5ac38992-00ce-4a36-883e-165a3a2410df` reduced the worst
  repeated-`yeah` loop, but it still contained smaller `yeah` bursts and the
  diarization regressed into many one-word speaker flips.

Updated values:
- Granite `max_new_tokens: 128`
- Granite `repetition_penalty: 1.1`
- `vad_onset: 0.8`
- `vad_offset: 0.6`
- `vad_pad_onset: 0.1`
- `vad_pad_offset: 0.0`
- `vad_min_duration_on: 0.2`
- `vad_min_duration_off: 0.6`

Companion transcript-smoothing change:
- `max_relabel_window_words: 18`
- `max_flip_span_words: 3`
- `min_dominant_ratio: 0.7`

Rationale:
- Keep the lower Granite token cap, but add a light repetition penalty so the
  model is less likely to get stuck on fillers like `yeah` without shortening
  every chunk further.
- Roll NeMo back to the prior VAD settings because the stronger pass-2 NeMo
  values increased fast speaker switching on this episode.
- Let the transcript-side smoothing absorb short mid-sentence one-word flips
  that are still visible even when NeMo only emits `0.25s` intrusions.

## 2026-03-18 - Podcast micro-turn smoothing pass 4

Context:
- The latest rerun `0da29be4-a5a6-49b0-9107-b863346f927e` still contained a
  long repeated `yeah` tail, while some visible one-word speaker flips such as
  the transcript lines near `787` and `1249` did not exist in the raw RTTM at
  all.

Updated values:
- Granite `max_new_tokens: 96`
- Granite `repetition_penalty: 1.15`
- `vad_onset: 0.8`
- `vad_offset: 0.6`
- `vad_pad_onset: 0.1`
- `vad_pad_offset: 0.0`
- `vad_min_duration_on: 0.2`
- `vad_min_duration_off: 0.6`

Companion transcript-smoothing change:
- relabel sandwiched transcript segments when duration is `<= 500ms`
- relabel sandwiched transcript segments when they contain `<= 3` words

Rationale:
- Tighten Granite generation again because the repeated `yeah` loop is still
  severe enough to justify a shorter cap and a slightly stronger penalty.
- Keep NeMo VAD stable because the latest small-turn examples are now traced to
  transcript assembly rather than raw RTTM switching.
- Smooth tiny transcript-only speaker intrusions deterministically when both
  neighboring segments already agree on the speaker.

## 2026-03-18 - Podcast micro-turn smoothing pass 5

Context:
- Full Docker GPU reruns of the same `audio.wav` produced different Demucs
  `vocals.wav` hashes, so diarization tuning needed to be isolated on a fixed
  separated artifact before promoting new defaults.
- On fixed vocals from run `9334f73e-4b08-4861-84e6-f570deecc4d9`, the old
  unconstrained NeMo search collapsed to `2` speakers and the exact `7`-speaker
  oracle experiment over-fragmented into `69` sub-`0.5s` RTTM turns.

Gateway fix:
- `min_speakers == max_speakers` now means an exact speaker count and writes
  `num_speakers` into the NeMo manifest.
- Unequal `min_speakers` / `max_speakers` no longer accidentally route through
  NeMo's oracle-count path.

Promoted defaults:
- `max_speakers: 10`
- `vad_onset: 0.8`
- `vad_offset: 0.6`
- `vad_pad_onset: 0.1`
- `vad_pad_offset: 0.0`
- `vad_min_duration_on: 0.2`
- `vad_min_duration_off: 0.6`
- `clustering_max_rp_threshold: 0.4`
- `clustering_sparse_search_volume: 60`
- `clustering_maj_vote_spk_count: true`
- Granite `max_new_tokens: 96`
- Granite `repetition_penalty: 1.15`

Evidence:
- Fixed-vocals search experiment `diar-fixed-search1` recovered `7` speakers
  with `55` RTTM turns at `<= 0.5s`, better than the exact-`7` oracle run's
  `69` while avoiding the old `2`-speaker collapse.
- Full Docker stage-1 rerun `8426c902-0524-4b80-bbe5-197d2f3add2d` produced
  `6` speakers, `372` transcript segments, `45` RTTM turns at `<= 0.5s`, and
  only `33` transcript sandwich candidates by the repo-local heuristic, down
  from `103` on `0da29be4-a5a6-49b0-9107-b863346f927e`.

Rationale:
- Keep the successful Granite anti-repetition settings because the severe
  repeated `yeah` loops are gone and the remaining text artifacts are smaller
  chunk-seam duplicates.
- Prefer a broader non-oracle NeMo search over exact speaker counts because it
  recovers multi-speaker structure without the worst oracle-induced
  fragmentation.
- Keep `min_speakers` unset by default for flexibility and use exact-count
  overrides only for tightly controlled episodes where the speaker count is
  known in advance.
