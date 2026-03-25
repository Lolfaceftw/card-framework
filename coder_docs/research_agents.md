# Delegation And Research Workflow Guide

Use this document as the source of truth for optional delegated-agent usage and
delegated web-research workflow in this repository.

This file keeps its historical name for compatibility, but its guidance is now
capability-based so it can work across different coding-agent runtimes.

If your environment supports delegated agents or sub-agents and exposes a
discoverable, non-empty local role inventory, follow this file.
If it does not, do the equivalent work in the main thread and keep the same
scoping, source-quality, and integration rules.

When a local runtime exposes a delegation inventory source for sub-agents, treat
that inventory as the source of truth for which delegated roles are actually
available. Use a local file, CLI-discoverable registry, or API as the source,
with `.codex/config.toml` as one optional example when present.

## Runtime Delegation Inventory

When delegation is available, treat the local runtime's delegation inventory as
the source of truth for what roles exist, their names, and their descriptions.
If the runtime has a config source (for example `.codex/config.toml`), use it as
one possible local inventory source.

Use only roles explicitly discovered in that inventory. If no roles are
discoverable, keep the work in the main thread.
Role labels vary by runtime; examples may include `explorer`, `researcher`,
`worker`, and `monitor`, but you should not assume these names are universal.

## Policy

- Sub-agents are allowed in this repository when they are configured locally
  and help move the task forward.
- Choose the configured sub-agent whose description best matches the scoped
  task instead of assuming only `researcher` is allowed.
- Delegate only to roles that are explicitly present in the discovered local
  inventory.
- Keep the main thread responsible for coordination, integrating delegated
  results, validation, and the final user-facing response.
- Every task that materially edits code must go through the required
  critic-review loop described below before it is considered done.
- When a delegated task includes open-web search, documentation lookup,
  standards research, or direct source-page retrieval, the delegated agent must
  prefer `coder_docs/scrapling.md`.
- If Scrapling is unavailable, use the closest equivalent browsing or retrieval
  capability that preserves direct URL usage, minimal-noise extraction, and
  authoritative source standards.
- When delegation is unnecessary or platform instructions disallow it, continue
  in the main thread.

## Delegation Patterns

### `explorer`-style role (example)

Use for read-only repository inspection, code search, and architecture
discovery when parallel context gathering is helpful.

Use only when the runtime exposes an `explorer`-style role in its configured
inventory.

### `researcher`-style role (example)

Use for open-web search, documentation lookup, standards research, and
source-page retrieval.

- Any `researcher`-style role should prefer `coder_docs/scrapling.md`, or use the
  closest equivalent retrieval path when Scrapling is unavailable.
- When the destination URL is already known, pass the direct URL instead of
  making the agent rediscover the source.
- Use only when the runtime exposes a `researcher`-style role in its configured
  inventory.

### `worker`-style role (example)

Use for bounded implementation, fixes, or other executor-style tasks when a
write-capable sub-agent is configured and delegation is appropriate.

Use only when the runtime exposes a `worker`-style role in its configured
inventory.

### `monitor`-style role (example)

Use for long-running checks, waits, or monitoring tasks when the collaboration
surface exposes that capability.

Use only when the runtime exposes a `monitor`-style role in its configured
inventory.

## Delegation Rules

- Scope delegated prompts to the exact question, code area, or output needed
  next.
- Do repository-level coordination and final integration in the main thread
  even when sub-agents are used.
- Use research output as evidence for local decisions; do not offload unrelated
  repo work without a clear boundary.
- Close or archive finished delegated threads when the collaboration surface
  exposes that control.

## Required Code-Edit Critic Loop

Use this loop for every task that materially edits repository code, config, or
behavioral docs.

1. After making a meaningful edit, spawn or reuse the best available review or
   critic-style sub-agent. If the runtime does not expose a dedicated critic,
   use the closest equivalent delegated reviewer. If delegated agents are not
   available at all, perform the same loop in the main thread.
2. Ask that critic to look for flaws, regressions, missing tests, unsafe
   assumptions, docs drift, and other correctness or maintainability issues in
   the changed work.
3. If the critic finds issues, fix them immediately. The critic may suggest the
   fix, validate a main-thread fix, or directly apply a bounded follow-up fix
   when the runtime and write scope allow it, but the main thread stays
   responsible for integration and final verification.
4. After each fix pass, send the updated state back to that critic and ask for
   feedback again.
5. Repeat the review-fix-review cycle with no arbitrary max-iteration or
   max-duration cap until the critic explicitly reports no material flaws, or
   until the task is blocked by an external constraint such as missing
   credentials, unavailable hardware, or a failing dependency outside the
   workspace.
6. Do not treat code generation, test success, or terminal output alone as a
   substitute for this critic signoff loop.

## Integration With Other Repo Docs

- Start each session with `coder_docs/codebase_guide.md` and
  `coder_docs/memory/errors_and_notes.md`.
- Keep delegated open-web work aligned with `coder_docs/scrapling.md`.
- Keep `AGENTS.md`, this file, and `coder_docs/codebase_guide.md` aligned when
  delegation or web-research expectations change.

## Do Not Do This

- Do not hardcode repo-local guidance to a single role name or a single example
  config path.
- Do not invent agent names or capabilities that are not present in the active
  runtime configuration.
- Do not skip the critic-review loop after code edits just because the first
  implementation appears to work locally.
- Do not use non-Scrapling web tooling for page retrieval when Scrapling can do
  the job.
- Do not ask a delegated agent to rediscover a URL that is already known.
- Do not leave finished delegated threads open when the platform exposes
  explicit cleanup controls.
