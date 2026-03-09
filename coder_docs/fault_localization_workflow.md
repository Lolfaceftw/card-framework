# Fault Localization Workflow

Use this document as the source of truth for agent-driven fault localization in
this repository.

This workflow adapts the semi-formal reasoning method summarized in
`coder_docs/coder_academic_literature/agentic_code_reasoning.md` to this
repo's day-to-day Python debugging flow. It is for localizing faults in repo
bugs, not for running a new benchmark or emitting Defects4J-style scores.

## Purpose

- Start from a failing test or failing command and narrow the likely root cause
  to a small ranked set of code regions.
- Force evidence-backed reasoning instead of guesses based on names, priors, or
  a stack trace alone.
- Produce a reusable localization record that can guide the eventual fix or a
  targeted validation run.

## Required Inputs

- A failing command or failing test target.
- The failing test name and test source when available.
- Candidate source files, modules, config files, or prompts touched by the
  behavior under test.
- Optional supplemental evidence such as assertion text, error output, or a
  stack trace.

Treat failure output as a clue, not as sufficient proof. Final localization
must still be justified from code.

## Repo Defaults

- Read `coder_docs/codebase_guide.md` first, then inspect the exact subsystem
  files before drawing conclusions.
- Prefer `rg` for search and cite exact file and line references whenever
  practical.
- Start static-first: understand the failing test semantics and trace the code
  path before proposing a fix.
- Keep the final prediction set to 3-5 suspect regions unless the bug clearly
  needs fewer.
- Separate likely root cause locations from downstream symptom locations.
- If runtime validation is needed, do it after the semi-formal certificate is
  written and keep the validation targeted.

## Workflow

### Preparation

- Record the failing command, failing test identifier, and the observed failure.
- Open the failing test source and identify the production entry point under
  test.
- Gather the nearest candidate files with `rg` before broadening the search.
- Note whether config, prompt templates, or data contracts are part of the code
  path so they are not skipped.

### Phase 1: Test Semantics Analysis

Write explicit premises from the test.

- `PREMISE T1`: what call or scenario the test executes.
- `PREMISE T2`: what behavior, value, state transition, or exception the test
  expects.
- `PREMISE T3`: what failure is actually observed or what contradiction must
  exist for the test to fail.

The premises should describe the test in operational terms, not just restate
the method name.

### Phase 2: Code Path Tracing

Trace the execution path from the test entry point into production code.

For each significant function, method, config gate, or prompt boundary, record:

- `METHOD`: the function, method, or boundary name.
- `LOCATION`: `path:line`.
- `BEHAVIOR`: what the code does.
- `RELEVANT`: why it matters to one or more premises.

Build a clear call sequence from the test into the code that could create the
failure. If the path branches, note the branch condition that matters.

### Phase 3: Divergence Analysis

Convert the trace into explicit divergence claims.

- `CLAIM D1`: at `path:line`, the current implementation can produce a behavior
  that contradicts `PREMISE Tx` because ...
- `CLAIM D2`: at `path:line`, a missing guard, incorrect assumption, wrong data
  shape, stale config, or prompt mismatch can violate `PREMISE Ty` because ...

Every claim must cite both:

- a specific code location, and
- the premise it contradicts.

If a location is only a symptom, say so directly and continue tracing toward
the root cause.

### Phase 4: Ranked Predictions

Produce a ranked list of suspect regions.

Each prediction must include:

- rank,
- suspect region as `path:line` or `path:start-end`,
- supporting `CLAIM` references,
- confidence as `high`, `medium`, or `low`,
- why this is more likely than nearby alternatives,
- the next validation action.

The default output is a small ranked suspect list, not a fix and not a
benchmark score.

## Structured Exploration Cycle

Use this structure every time you branch into another file or subsystem.

Before reading the next file:

```text
HYPOTHESIS H1: [What you expect to find and why it may contain the bug]
EVIDENCE: [What from the test or prior files supports this hypothesis]
CONFIDENCE: [high|medium|low]
```

After reading the file:

```text
OBSERVATIONS from [path]:
O1: [Key observation with line reference]
O2: [Key observation with line reference]

HYPOTHESIS UPDATE:
H1: [CONFIRMED | REFUTED | REFINED] - [Explanation]

UNRESOLVED:
- [Question that remains open]
- [Another open question]

NEXT ACTION RATIONALE: [Why the next file, command, or stop condition is justified]
```

This structure is intended to prevent pattern-matching on filenames or
functions without evidence.

## Final Output Contract

Use this shape when concluding a localization pass:

```text
Fault Localization Result

Scope:
- Failing command/test:
- Observed failure:

Premises:
- PREMISE T1:
- PREMISE T2:
- PREMISE T3:

Key Claims:
- CLAIM D1:
- CLAIM D2:

Ranked Predictions:
1. path:line[-line] | confidence | supporting claims | next validation
2. path:line[-line] | confidence | supporting claims | next validation
3. path:line[-line] | confidence | supporting claims | next validation

Most Likely Root Cause:
- [Single best suspect and why]

Residual Uncertainty:
- [What is still unproven]
```

## Copy-Paste Scaffold

```md
# Fault Localization Certificate

## Inputs
- Failing command/test:
- Failing test name:
- Observed failure:
- Supplemental evidence:

## Preparation
- Initial candidate files:
- Search terms used:

## Phase 1: Test Semantics Analysis
- PREMISE T1:
- PREMISE T2:
- PREMISE T3:

## Phase 2: Code Path Tracing
- METHOD:
  LOCATION:
  BEHAVIOR:
  RELEVANT:

## Phase 3: Divergence Analysis
- CLAIM D1:
- CLAIM D2:

## Phase 4: Ranked Predictions
1. Suspect region:
   Supporting claims:
   Confidence:
   Next validation:
2. Suspect region:
   Supporting claims:
   Confidence:
   Next validation:
3. Suspect region:
   Supporting claims:
   Confidence:
   Next validation:

## Most Likely Root Cause
- ...

## Residual Uncertainty
- ...
```

## References

- `coder_docs/coder_academic_literature/agentic_code_reasoning.md`
- R. Just, D. Jalali, and M. D. Ernst, "Defects4J: A Database of Existing
  Faults to Enable Controlled Testing Studies for Java Programs," in
  *Proceedings of the 2014 International Symposium on Software Testing and
  Analysis*, 2014, pp. 437-440, doi:10.1145/2610384.2628055.

For future benchmark work, preserve the paper's Top-N hunk-overlap evaluation
separately from this operational workflow. This guide is intentionally about
repo debugging and ranked suspect regions.
