# Prelinked Emotion And Context Compaction Plan

Last updated: 2026-03-24

## Purpose

Make live-draft duration measurement use the same emotion-vector-conditioned audio that
the final stage-3 export will use, and add proactive DeepSeek context compaction before
context-window failure occurs.

This file is the source of truth for the implementation sequence, QA gates, smoke-test
workflow, monitored GPU validation, and preset-expansion rollout.

## Goals

1. Resolve turn emotion vectors before live-draft duration estimation.
2. Eliminate post-convergence rerender drift caused by late summary-turn emotion linking.
3. Add provider-aware prompt compaction that triggers before DeepSeek reaches its context
   limit.
4. Keep all policies configurable.
5. Require critic sub-agent review for every code-edit phase.
6. Validate every major change with monitored Docker Compose GPU runs before broader
   rollout.

## Design Principles

- Use strategy objects for variable behavior:
  `TurnEmotionResolutionStrategy`, `PromptCompactionStrategy`.
- Use a policy object for budget decisions:
  `ContextBudgetManager`.
- Use typed value objects for resolved turn emotion and compaction results.
- Keep orchestration, provider, and rendering concerns separated.
- Favor config-driven thresholds over hardcoded constants.
- Make every implementation phase reviewable in isolation.

## Core Changes

### 1. Prelink Emotion During Live Draft

Move summary-turn emotion linking into the live-draft mutation path so each turn is
rendered with its resolved vector before `estimate_duration` and before critic review.

Resolution order:

1. `source-linked vector`
2. `explicit turn vector`
3. `preset vector`
4. `emo_text`

Expected result:

- the critic evaluates the same vector-aware audio that stage 3 will export
- final duration should match the approved live-draft duration much more closely

### 2. Introduce `TurnEmotionResolutionStrategy`

Add a dedicated strategy object that resolves one turn into:

- `emotion_vector`
- `emotion_vector_source`
- optional source-support metadata
- fallback preset label for readability

This keeps turn-emotion logic centralized and testable.

### 3. Persist Resolved Turn Emotion In Live-Draft State

Each add/edit/remove mutation should persist the resolved emotion payload into
live-draft state before audio render and duration measurement.

### 4. Remove Post-Convergence Surprise Rerenders

After critic pass, finalization should reuse already rendered vector-aware live-draft
audio whenever possible.

Only rerender if a material synthesis input changed, such as:

- final text changed
- explicit vector changed
- speaker assignment changed

### 5. Add Post-Render Duration Verification

After final stage-3 merged audio is ready, measure its actual duration again.

If it is outside tolerance:

- either fail with a structured reason
- or run one bounded repair pass

Behavior must be config-driven.

### 6. Introduce `ContextBudgetManager`

Add a provider-aware context budget policy that estimates prompt size before each
summarizer call.

For DeepSeek, the default compaction trigger should be:

- `80%` of the configured context window

### 7. Introduce Same-Provider `PromptCompactionStrategy`

When prompt usage reaches the configured threshold, compact with the same active LLM
provider rather than a separate helper model.

Compaction should preserve:

- current draft
- latest critic feedback
- compact loop-memory state
- compact transcript grounding

and summarize or discard older raw interaction history.

### 8. Structure Compaction Outputs

Compaction should emit structured outputs, not just free text:

- `compressed_transcript_context`
- `compressed_history_context`
- `compression_reason`
- `source_iteration_span`
- `estimated_tokens_before`
- `estimated_tokens_after`

### 9. Add Overflow Retry

If DeepSeek still returns a context-window error, perform one forced-compaction retry.
If that still fails, stop with a structured terminal error.

## Configuration Plan

Planned config surface:

- `audio.summary_turn_emotion.prelink_enabled`
- `audio.summary_turn_emotion.prelink_strategy`
- `audio.summary_turn_emotion.fallback_order`
- `audio.summary_turn_linking.preserve_existing_explicit_vectors`
- `audio.summary_turn_linking.force_live_draft_rerender`
- `orchestrator.post_render_duration_verification.enabled`
- `orchestrator.prompt_compaction.enabled`
- `orchestrator.prompt_compaction.trigger_ratio`
- `orchestrator.prompt_compaction.retry_on_overflow`
- `orchestrator.prompt_compaction.max_overflow_retries`
- `orchestrator.prompt_compaction.compress_with_same_provider`
- `orchestrator.prompt_compaction.keep_latest_iterations`
- `profiles.llm.<id>.context_window_tokens`
- `profiles.llm.<id>.compaction_target_ratio`

Default DeepSeek compaction trigger:

- `0.80`

## Mandatory Review Workflow

Every code-edit phase must follow this workflow:

1. Implement one bounded slice.
2. Run local lint and targeted tests.
3. Spawn a critic sub-agent to review that slice.
4. Collect bug findings, edge cases, and regression risks.
5. Fix findings.
6. Rerun lint and targeted tests.
7. Only then proceed to the next phase.

Critic review is mandatory for every phase, not optional cleanup.

## Implementation Phases

### Phase 1. Config And Policy Objects

Implement:

- config schema additions
- `TurnEmotionResolutionStrategy`
- `ContextBudgetManager`
- `PromptCompactionStrategy` interfaces

QA gate:

- config parsing tests
- strategy unit tests
- critic sub-agent review

### Phase 2. Prelinked Emotion In Live Draft

Implement:

- early turn emotion resolution in live-draft mutation flow
- persistence of resolved vectors into live-draft state
- vector-aware duration measurement path

QA gate:

- unit and integration tests for resolution precedence
- critic sub-agent review

### Phase 3. Finalization Without Rerender Drift

Implement:

- finalization from already rendered live-draft audio
- rerender only when material synthesis inputs changed

QA gate:

- regression tests proving no post-pass duration drift
- critic sub-agent review

### Phase 4. Post-Render Duration Verification

Implement:

- final merged WAV duration check
- bounded repair or terminal structured failure

QA gate:

- verification tests
- critic sub-agent review

### Phase 5. Prompt Compaction

Implement:

- preflight prompt-budget estimation
- same-provider compaction at configured threshold
- structured compaction state

QA gate:

- compaction trigger tests at 80%
- provider-aware path tests
- critic sub-agent review

### Phase 6. Overflow Retry

Implement:

- known DeepSeek context-window failure detection
- one forced-compaction retry
- structured terminal diagnostics if retry still fails

QA gate:

- overflow retry tests
- critic sub-agent review

### Phase 7. Runtime Validation And Rollout

Implement:

- monitored smoke run
- monitored preset expansion
- outlier reruns

QA gate:

- reviewed run reports
- critic sub-agent review on code changes made during validation fixes

## Runtime Validation Plan

### Step A. Single Docker Compose GPU Smoke Run

Once Phases 1 through 6 are implemented and locally tested:

1. Run one Docker Compose GPU smoke test with the monitored wrapper.
2. Use one random episode and one random duration preset.
3. Keep `selection.llm=deepseek_chat`.
4. Keep `orchestrator.max_iterations=3`.
5. Use source-separated speaker samples.
6. Capture:
   - stdout log
   - stderr log
   - resource monitor JSONL
   - run status JSON
7. Monitor GPU and RAM continuously using the loop-based monitored runner already used in
   the SPoRC sweep workflow.
8. Inspect:
   - duration tolerance result
   - prompt-compaction trigger behavior
   - context-overflow retry behavior, if exercised
   - post-render duration verification result

Pass criteria:

- no hang
- no orphan GPU memory pattern
- no uncontrolled context overflow
- final WAV within configured tolerance

### Step B. Preset Expansion After Smoke Pass

If the single smoke run passes:

1. Run the remaining presets for that same episode.
2. Keep loop-based monitoring on every run.
3. Inspect GPU, RAM, and logs during execution.
4. Fix any issue before widening to additional episodes.

### Step C. Wider Validation Sweep

After the same-episode preset sweep passes:

1. Rerun the currently known duration outliers first.
2. Use the same monitored Docker Compose GPU flow.
3. Keep loop-based monitoring for:
   - GPU memory
   - host RAM
   - stdout and stderr logs
   - run status
4. Only then consider a broader batch rerun.

## Monitoring Requirement

All GPU validation runs must use the loop-based monitored runner so the runtime is
observed continuously rather than sampled only at the start or end.

Minimum monitoring artifacts per run:

- `*.stdout.log`
- `*.stderr.log`
- `*.resource_monitor.jsonl`
- `*.run_status.json`

## Known Risks

- Early emotion linking may change turn pacing enough to expose new duration edge cases.
- Same-provider compaction may preserve too much or too little transcript grounding if the
  compaction schema is not strict enough.
- Post-render verification may reveal that some previously accepted drafts were only
  passing because of pre-rerender live-draft timing.
- DeepSeek overflow recovery must avoid silent looping or repeated compaction retries.

## Success Criteria

The change is successful when:

1. live-draft duration measurement uses the same resolved vectors as final export
2. final stage-3 WAV stays within tolerance when critic passes
3. DeepSeek prompt compaction triggers before context-window failure at the configured
   threshold
4. overflow is handled by bounded retry or structured failure
5. monitored Docker Compose GPU smoke validation passes before broader rollout
6. every implementation phase has passed critic sub-agent review
