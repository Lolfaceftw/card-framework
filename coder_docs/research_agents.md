# Codex Research Agent Workflow Guide

Use this document as the source of truth for Codex research-agent usage and any
delegated research workflow in this repository.

This workflow is derived from the local Codex configuration in
`../.codex/config.toml`. Only the configured `researcher` agent is supported by
repo-local guidance.

## Policy

- Use delegated agents only for external research and web retrieval.
- The supported delegated agent ID is `researcher`.
- Keep repository reading, local command execution, validation, and
  write-capable implementation in the main thread.
- Keep the main thread responsible for integrating research results and
  delivering the final user-facing response.
- Do not document or depend on dedicated non-research agent roles in this
  repository.

## `researcher`

Use for external research and web retrieval.

- Open-web search, documentation lookup, standards research, and source-page
  retrieval belong here.
- `researcher` must follow `coder_docs/scrapling.md`.
- When the destination URL is already known, pass the direct URL instead of
  making the agent rediscover the source.
- Keep prompts scoped to the exact question, source list, or evidence needed.

## Delegation Rules

- Delegate open-web search and research to `researcher`.
- Do repository reading, code search, local validation, and implementation in
  the main thread with local tools.
- Use research output as evidence for local decisions; do not offload unrelated
  repo work to `researcher`.
- When a research task finishes, capture the result promptly and close or
  archive the finished research thread if the collaboration surface exposes that
  control.

## Integration With Other Repo Docs

- Start each session with `coder_docs/codebase_guide.md` and
  `coder_docs/memory/errors_and_notes.md`.
- Keep open-web work aligned with `coder_docs/scrapling.md`.
- Keep `AGENTS.md`, this file, and `coder_docs/codebase_guide.md` aligned when
  research-agent expectations change.

## Do Not Do This

- Do not document or rely on non-research delegated roles in repo-local
  guidance.
- Do not use non-Scrapling web tooling for page retrieval when Scrapling can do
  the job.
- Do not ask `researcher` to rediscover a URL that is already known.
- Do not leave finished research threads open when the platform exposes
  explicit cleanup controls.
