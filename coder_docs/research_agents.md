# Codex Delegation And Research Workflow Guide

Use this document as the source of truth for Codex sub-agent usage and
delegated web-research workflow in this repository.

This workflow is derived from the local Codex configuration in
`../.codex/config.toml`. Repo-local guidance should follow the configured
sub-agent inventory instead of hardcoding a single delegated role.

## Current Local Config Snapshot

The current local Codex config enables `multi_agent = true` and defines these
sub-agents:

- `explorer`: read-only codebase exploration.
- `researcher`: external research and web retrieval with Scrapling.
- `worker`: execution-focused implementation, fixes, and other executor-style
  tasks.
- `monitor`: long-running task monitoring.

Treat `../.codex/config.toml` as authoritative if this list changes.

## Policy

- Sub-agents are allowed in this repository when they are configured locally
  and help move the task forward.
- Choose the configured sub-agent whose description best matches the scoped
  task instead of assuming only `researcher` is allowed.
- Keep the main thread responsible for coordination, integrating delegated
  results, validation, and the final user-facing response.
- When a delegated task includes open-web search, documentation lookup,
  standards research, or direct source-page retrieval, the delegated agent must
  follow `coder_docs/scrapling.md`.
- When delegation is unnecessary or platform instructions disallow it, continue
  in the main thread.

## Delegation Patterns

### `explorer`

Use for read-only repository inspection, code search, and architecture
discovery when parallel context gathering is helpful.

### `researcher`

Use for open-web search, documentation lookup, standards research, and
source-page retrieval.

- `researcher` must follow `coder_docs/scrapling.md`.
- When the destination URL is already known, pass the direct URL instead of
  making the agent rediscover the source.

### `worker`

Use for bounded implementation, fixes, or other executor-style tasks when a
write-capable sub-agent is configured and delegation is appropriate.

### `monitor`

Use for long-running checks, waits, or monitoring tasks when the collaboration
surface exposes that capability.

## Delegation Rules

- Scope delegated prompts to the exact question, code area, or output needed
  next.
- Do repository-level coordination and final integration in the main thread
  even when sub-agents are used.
- Use research output as evidence for local decisions; do not offload unrelated
  repo work without a clear boundary.
- Close or archive finished delegated threads when the collaboration surface
  exposes that control.

## Integration With Other Repo Docs

- Start each session with `coder_docs/codebase_guide.md` and
  `coder_docs/memory/errors_and_notes.md`.
- Keep delegated open-web work aligned with `coder_docs/scrapling.md`.
- Keep `AGENTS.md`, this file, and `coder_docs/codebase_guide.md` aligned when
  delegation or web-research expectations change.

## Do Not Do This

- Do not hardcode repo-local guidance to a single agent name when
  `../.codex/config.toml` exposes a broader sub-agent inventory.
- Do not invent agent names or capabilities that are not present in the local
  Codex config.
- Do not use non-Scrapling web tooling for page retrieval when Scrapling can do
  the job.
- Do not ask a delegated agent to rediscover a URL that is already known.
- Do not leave finished delegated threads open when the platform exposes
  explicit cleanup controls.
