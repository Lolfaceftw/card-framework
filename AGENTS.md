# Agent Entry Point

Use this file as the repo-local pointer into `coder_docs/`. `AGENTS.md` should
stay compact. The detailed repository guidance belongs in the topic-specific
Markdown files under `coder_docs/`.

## Start Here

- Read `coder_docs/codebase_guide.md` first at the start of every new session.
- Review `coder_docs/memory/errors_and_notes.md` before repeating work in an
  area with recent fixes or pitfalls.
- Before time-sensitive work, check the current date and time using the
  strongest date or web capability your environment exposes and treat that as
  authoritative for the session.

## Capability Gates

- If your environment supports delegated agents or sub-agents, follow
  `coder_docs/research_agents.md` for delegation rules.
- If your environment does not support delegated agents, do the equivalent work
  in the main thread and keep the same scoping and integration discipline.
- If your environment exposes Scrapling MCP, follow `coder_docs/scrapling.md`
  for external web retrieval.
- If Scrapling is unavailable, use the closest equivalent browsing or retrieval
  capability while preserving the same standards for authoritative sourcing,
  direct URLs, minimal noise, and targeted extraction.

## If You Are Looking For

- Architecture, runtime flow, entrypoints, commands, and maintenance
  expectations:
  `coder_docs/codebase_guide.md`
- Delegation, optional sub-agent usage, and delegated research workflow:
  `coder_docs/research_agents.md`
- External web retrieval workflow and Scrapling usage:
  `coder_docs/scrapling.md`
- Python coding style, docstrings, typing, design, and core do-nots:
  `coder_docs/python_engineering_standards.md`
- Security expectations:
  `coder_docs/security_standards.md`
- Logging expectations:
  `coder_docs/logging_standards.md`
- Testing expectations:
  `coder_docs/testing_standards.md`
- Methodology-sensitive or academically grounded implementations:
  `coder_docs/academic_standards.md`
- Bug triage and failing-test workflow:
  `coder_docs/fault_localization_workflow.md`
- Git and GitHub contributor workflow:
  `coder_docs/git_github_workflow.md`
- Release and GitHub Actions publish behavior:
  `coder_docs/github_actions_release_spec.md`
- Linting workflow:
  `coder_docs/ruff.md`
- Dependency and environment workflow:
  `coder_docs/uv_package_manager.md`

## Repo-Wide Non-Negotiables

- Treat `coder_docs/*` as the authoritative repo-local guidance set.
- Assume configuration exists; do not hardcode deploy-varying values,
  credentials, or environment-specific endpoints.
- Use `uv` as the package manager for this repository and `ruff` as the linter.
- When implementation changes affect behavior, interfaces, contracts, outputs,
  workflows, or operator expectations, update the relevant tests and Markdown
  docs in the same change.
- When you fix a meaningful error or uncover a recurring pitfall, prepend a
  brief dated Problem/Solution note to
  `coder_docs/memory/errors_and_notes.md`.
- If `AGENTS.md` and a topic-specific `coder_docs/*.md` file drift, move the
  detailed rule into the topic file and keep `AGENTS.md` as the pointer layer.
