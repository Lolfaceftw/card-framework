# Summary-Turn Emotion Linking Investigation

Date: 2026-03-23
Status: Root cause identified and fixed

## Symptom

The Lion/Lamb/Meadow 1-minute GPU run completed successfully, but stage 2 logged:

- `Summary-turn emotion linking found no strong source-grounded matches; preserving fallback emotion behavior.`

The saved `summary.xml` contained only `emo_preset` attributes, and the stage-3
voice-clone manifest showed empty:

- `emotion_vector_source`
- `source_segment_indices`
- `source_segment_weights`

That meant TTS still used preset-derived vectors instead of the new
source-grounded linked vectors.

## Initial Hypotheses

1. The linker heuristic was too strict for paraphrased summary turns.
2. The final summary turns had too little lexical overlap with source segments.
3. The Docker runtime was not actually running the latest linker code.
4. The stage-2 runtime transcript had lost affect-analysis extras before the
   linker ran.

## Experiments

### Experiment 1: Replay linker scoring offline against saved artifacts

Inputs:

- `compare_lion_lamb_meadow_1m_20260323/input_bundle/transcript.json`
- `compare_lion_lamb_meadow_1m_20260323/source_linked_vector_run_final/summary.xml`

Result:

- many turns had multiple candidate source segments above the configured
  `min_support_score=0.08`
- `SummaryTurnEmotionLinker(...).link_turns(...)` linked all `11/11` turns

Conclusion:

- the scoring heuristic was not the main reason for the live-run fallback

### Experiment 2: Replay the same linker call inside Docker

Method:

- started a one-off `docker compose run --rm --entrypoint /bin/bash setup-and-run`
- loaded the same transcript and summary files
- invoked `SummaryTurnEmotionLinker(...).link_turns(...)`

Result:

- Docker also linked `11/11` turns

Conclusion:

- this was not a host-vs-container code drift issue

### Experiment 3: Inspect the live transcript loader path

Observation:

- `card_framework.cli.main` does not load transcript JSON directly with
  `json.load(...)`
- it calls `card_framework.agents.utils.load_transcript(...)`
- that function calls `validate_transcript_schema(...)`

Root cause in old behavior:

- `validate_transcript_schema(...)` rebuilt each segment with only:
  - `speaker`
  - `start_time`
  - `end_time`
  - `text`
- this silently dropped all segment extras, including:
  - `emotion_vector`
  - `emotion_vector_source`
  - `emotion_vector_confidence`
  - `affect_hint`
  - `suggested_emo_preset`
  - any future overlap or support metadata

Why that caused the observed fallback:

- the linker runs with `require_emotion_vector=True`
- after transcript validation, runtime segments no longer had emotion vectors
- `_segment_has_vector(...)` therefore returned `False` for every segment
- the linker skipped all segments and returned no supports
- stage 2 logged `preserving fallback emotion behavior`

## Fix

Updated `src/card_framework/agents/utils.py` so transcript validation preserves
all original segment extras while still normalizing the required core fields.

New behavior:

- normalize `speaker`, `start_time`, `end_time`, and `text`
- keep all other segment keys intact
- preserve top-level payload metadata as before

Regression coverage added:

- `tests/agents/test_utils.py`

The regression test verifies that transcript validation now preserves:

- `emotion_vector`
- `emotion_vector_source`
- `emotion_vector_confidence`
- `affect_hint`
- `suggested_emo_preset`
- arbitrary extra keys

## Why This Fix Is Optimal For Our Case

This is the correct fix for our pipeline because the problem was not the linker;
the problem was that the runtime transcript loader erased the structured emotion
evidence before the linker ever saw it.

Preserving extras is better than loosening the linker thresholds because it:

1. fixes the real root cause instead of masking it
2. preserves all future structured transcript metadata, not just emotion fields
3. keeps the linker conservative and grounded
4. prevents similar regressions in retrieval, overlap handling, and future
   post-processing stages

## Next Validation Step

Rerun the Lion/Lamb/Meadow stage-2 to stage-4 flow and confirm:

1. stage 2 logs linked-turn success instead of fallback
2. `summary.xml` includes `emotion_vector`, `emotion_vector_source`, and
   support indices or weights
3. stage-3 manifest preserves `summary_turn_linking` provenance
4. final TTS uses explicit linked vectors instead of preset-only fallback
