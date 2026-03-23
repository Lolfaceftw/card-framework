# IndexTTS2 Emotion Vector Roadmap

Date: 2026-03-23
Status: Implemented

## Implementation Snapshot

Completed in the repo:

- summary turns now support `emotion_vector_source`,
  `source_segment_indices`, and `source_segment_weights`
- a deterministic post-summary linker now blends transcript-segment emotion
  vectors onto final summary turns
- stage 2 and stage 3 now enrich summaries before final output persistence
- live-draft stage 3 now rerenders the accepted summary when source-linked
  vectors are active so final audio matches final XML
- stage-3 manifests now preserve linked-vector provenance metadata
- config surface added under `audio.summary_turn_linking`

## Goal

Preserve the original emotion of the source audio more faithfully by carrying
structured emotion vectors all the way into IndexTTS2, instead of collapsing
final summary turns down to coarse preset-only emotion.

## Current State

The pipeline already does these parts:

- source transcript segments can carry precise affect-analysis emotion vectors
- summary turns can optionally carry explicit `emotion_vector`
- stage 3 already prefers explicit vectors over preset-only fallback
- stage 4 already works on top of the stage-3 result

The main remaining gap is this:

- summary turns are usually written with `emo_preset` only
- that means stage 3 often uses a fixed preset vector instead of a
  source-grounded blended vector

## Verified Target Contract

IndexTTS2 accepts a direct 8-float `emo_vector` in this order:

`[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`

## Recommended End State

For each final summary turn:

1. identify the original transcript segments that best support that turn
2. blend those source segments' emotion vectors
3. normalize the blended result for IndexTTS2
4. write the explicit vector onto the summary turn
5. let stage 3 consume that explicit vector directly

This keeps the final synthesis grounded in the original audio instead of asking
the summarizer to invent raw vectors.

## Architecture Principles

1. Keep one canonical `EmotionVector` value object.
2. Keep `emo_preset` as fallback, not the primary target.
3. Run source-linking as a post-processing stage, not inside the summarizer
   prompt.
4. Keep the linker deterministic and config-driven.
5. Make artifacts explicit so operators can see what stage 3 actually used.

## Recommended Insertion Point

Run the summary-turn emotion-linking pass after the stage-2 summarizer/critic
loop has converged, but before final summary output is written and before any
stage-3 voice cloning begins.

Primary insertion point:

- `src/card_framework/orchestration/stage_orchestrator.py`
  inside the final accepted-draft path in `_run_full_loop()`

Secondary insertion point:

- `src/card_framework/orchestration/stage_orchestrator.py`
  inside the stage-3-only path so externally supplied summaries can also be
  enriched before voice cloning

This keeps the live draft loop unchanged while ensuring stage 3 sees the final
explicit vectors.

## Phase Plan

### Phase 1. Preserve And Expose The EmotionVector Contract

Keep the shared `EmotionVector` contract as the single source of truth for
IndexTTS-compatible vectors.

Files:

- `src/card_framework/shared/emotion_vector.py`
- `src/card_framework/shared/summary_xml.py`
- `src/card_framework/audio_pipeline/voice_clone_contracts.py`

### Phase 2. Add Support Metadata To Summary Turns

Extend summary-turn objects so they can optionally carry support-trace metadata
alongside `emotion_vector`.

Recommended optional fields:

- `source_segment_indices`
- `source_segment_weights`
- `emotion_vector_source`

Important note:

- store support links as XML attributes or a parseable side payload
- do not use nested XML child elements, because `parse_summary_xml()` currently
  reads child text as spoken content

Files:

- `src/card_framework/shared/summary_xml.py`
- `src/card_framework/audio_pipeline/voice_clone_contracts.py`
- `src/card_framework/orchestration/stage_orchestrator.py`

### Phase 3. Build A Post-Summary Turn Linker

Create a dedicated helper that takes:

- final summary turns
- transcript segments
- per-segment emotion vectors

and returns:

- enriched summary turns with explicit linked `emotion_vector`

Recommended new module:

- `src/card_framework/audio_pipeline/summary_turn_emotion_linker.py`

### Phase 4. Score Supporting Source Segments

For each summary turn, rank source transcript segments using a configurable
support score.

Recommended first-pass score:

`support_score = semantic_similarity * emotion_confidence * speaker_bonus`

Useful signals:

- text similarity between summary turn and source segment
- same-speaker bonus
- optional adjacency bonus for nearby segments
- optional coverage bonus for multi-segment support

Recommended default behavior:

- keep the top 3 support candidates above threshold
- optionally collapse contiguous same-speaker supports into one support span

Files:

- `src/card_framework/audio_pipeline/summary_turn_emotion_linker.py`
- `src/card_framework/orchestration/transcript.py`
- `src/card_framework/agents/dtos.py`
- `src/card_framework/retrieval/embeddings.py`

### Phase 5. Blend Source Emotion Vectors

Blend the linked source vectors using weighted averaging.

Default formula:

`blended_vector = sum(weight_i * source_vector_i) / sum(weight_i)`

Then apply:

- confidence gating
- minimum support threshold
- optional same-speaker bias
- IndexTTS normalization

Fallback order:

`linked explicit vector` > `existing explicit vector` > `preset vector` >
`emo_text`

Files:

- `src/card_framework/audio_pipeline/summary_turn_emotion_linker.py`
- `src/card_framework/shared/emotion_vector.py`
- `src/card_framework/audio_pipeline/voice_clone_orchestrator.py`

### Phase 6. Persist The Final Linked Result

Write the final explicit linked vector into:

- final `summary.xml`
- live-draft stage-3 snapshot
- live-draft state file
- stage-3 manifest

This makes it obvious what stage 3 actually consumed.

Files:

- `src/card_framework/shared/summary_xml.py`
- `src/card_framework/orchestration/stage_orchestrator.py`
- `src/card_framework/audio_pipeline/live_draft_voice_clone.py`
- `src/card_framework/audio_pipeline/voice_clone_orchestrator.py`

### Phase 7. Add Config, Diagnostics, And Tests

Config surface should include:

- `audio.summary_turn_linking.enabled`
- `audio.summary_turn_linking.top_k`
- `audio.summary_turn_linking.min_support_score`
- `audio.summary_turn_linking.same_speaker_bonus`
- `audio.summary_turn_linking.require_emotion_vector`
- `audio.summary_turn_linking.blending_strategy`
- `audio.summary_turn_linking.fallback_policy`

Diagnostics should report:

- linked source segments per summary turn
- support weights
- final blended vector
- whether fallback was used

Tests should cover:

1. one-turn to one-segment linking
2. one-turn to multi-segment weighted blending
3. same-speaker preference
4. fallback when support is weak or missing
5. summary XML persistence of explicit vectors
6. stage-3 precedence using linked vectors over presets

Files:

- `conf/config.yaml`
- `tests/audio_pipeline/test_summary_turn_emotion_linker.py`
- `tests/shared/test_summary_xml.py`
- `tests/audio_pipeline/test_voice_clone_orchestrator.py`
- related stage-orchestrator tests

## Numerical Example

Summary turn:

`"Midnight Sun is exciting, but it also softens the pain of the cancelled trip."`

Linked support segments:

- segment A, weight `0.55`, vector
  `[0.36, 0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.06]`
- segment B, weight `0.30`, vector
  `[0.25, 0.00, 0.03, 0.00, 0.00, 0.06, 0.04, 0.14]`
- segment C, weight `0.15`, vector
  `[0.08, 0.01, 0.10, 0.02, 0.00, 0.12, 0.01, 0.18]`

Weighted blend:

- `happy = 0.55*0.36 + 0.30*0.25 + 0.15*0.08 = 0.285`
- `sad = 0.55*0.01 + 0.30*0.03 + 0.15*0.10 = 0.030`
- `melancholic = 0.55*0.01 + 0.30*0.06 + 0.15*0.12 = 0.042`
- `surprised = 0.55*0.02 + 0.30*0.04 + 0.15*0.01 = 0.025`
- `calm = 0.55*0.06 + 0.30*0.14 + 0.15*0.18 = 0.102`

Result before final normalization:

`[0.285, 0.007, 0.030, 0.009, 0.006, 0.042, 0.025, 0.102]`

That vector now reflects the emotional mix of the real source moments instead
of a single fixed preset like `engaged`.

## Main Risks

1. Summary turns may compress multiple ideas, which can make support linking
   noisy if thresholds are weak.
2. Overweighting same-speaker matches may hide valid cross-speaker influence.
3. Underweighting confidence may produce bland averaged vectors.
4. Without good artifact tracing, operators will not know whether a turn used:
   linked vector, manual vector, preset vector, or `emo_text`.

## Recommended Implementation Order

1. Add support metadata to summary turns.
2. Build the post-summary turn linker helper.
3. Implement weighted vector blending.
4. Persist explicit vectors back into `summary.xml` and live-draft state.
5. Make stage 3 consume linked vectors automatically.
6. Add config, diagnostics, and tests.

## Explain Like I'm 5

The summary says something new in fewer words, so it is often made from several
original moments mixed together.

This plan teaches the system to look back at those original moments, mix their
emotion sliders together, and then use that mixed emotion for the final voice.

So instead of pressing one big button like `warm`, we rebuild the feeling from
the real source moments that the summary came from.
