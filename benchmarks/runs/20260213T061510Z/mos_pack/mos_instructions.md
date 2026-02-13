# Voice Cloning Similarity Rating Guide

## Objective
Rate whether generated speech preserves target speaker identity versus reference audio.

## Context
Each pair has one holdout reference recording and one generated clip. Judge
speaker identity similarity only.

## Inputs
- `mos_pairs.csv` metadata with pair ids.
- Audio files under `audio/` for each pair.

## Output contract
- Fill `ratings_template.csv` with one row per pair id.
- Scales:
  - `smos_a`: integer [1, 5]
  - `smos_b`: integer [1, 5]
  - `cmos_ab`: integer [-3, 3] (positive means A more similar)
  - `more_similar_to_reference`: `A`, `B`, or `TIE`

## Rules
- Use headphones and a quiet room.
- Focus on speaker identity cues (timbre, accent, prosodic identity).
- Ignore waveform displays and metadata labels during judging.

## Examples
- If A is clearly more similar: `smos_a=5`, `smos_b=2`, `cmos_ab=3`, `A`.
- If tied: `smos_a=4`, `smos_b=4`, `cmos_ab=0`, `TIE`.

## Evaluation
- All pair ids rated once.
- No missing required scores.
- Scores stay within allowed ranges.
