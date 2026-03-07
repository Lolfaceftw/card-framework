# Ruff Linting Guide

This repository uses Ruff as its Python linter.

This document is intentionally about linting, not formatting. If the project later adopts Ruff formatting or another formatter as policy, document that separately and update this file only where the workflows overlap.

## Current Project Baseline

- Ruff is installed as a dev dependency through `[dependency-groups].dev` in `pyproject.toml`.
- Current checked version in this repo is `ruff 0.15.5`.
- Current baseline status is clean: `uv run ruff check .` passes.
- There is currently no explicit `[tool.ruff]` configuration block in `pyproject.toml`.

Because there is no local Ruff config yet, the working baseline is the default Ruff behavior plus project conventions from `AGENTS.md` and `coder_docs/`.

## Canonical Commands

Run these from the repository root:

```bash
uv sync --dev
uv run ruff check .
uv run ruff check . --fix
```

Useful targeted commands:

```bash
uv run ruff check src
uv run ruff check path/to/file.py
```

## Working Rules

- Run Ruff after Python changes before considering the task complete.
- Prefer fixing violations over suppressing them.
- If you use `--fix`, review the diff instead of assuming every fix is appropriate.
- Do not use `--unsafe-fixes` unless the task explicitly calls for it and you understand the behavioral risk.
- Keep suppressions narrow and justified. Prefer rule-specific `noqa` comments over blanket ignores.
- If a suppression becomes necessary, use the smallest possible scope and include the exact rule code.

## Future Ruff Configuration Policy

If this project adds Ruff configuration later:

- Put the configuration in `pyproject.toml`.
- Keep rule selection explicit. Do not enable broad categories without intent.
- Document the selected rules, ignores, and rationale in this file.
- Update this file when the lint baseline or command workflow changes.

## Agent Expectations

- Treat `uv run ruff check .` as the repo-level lint gate.
- Do not introduce a second linter unless the repository intentionally adopts one.
- If Ruff starts failing because of new code or new configuration, fix the code or update the documented policy in the same change.
