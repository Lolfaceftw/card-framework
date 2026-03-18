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
