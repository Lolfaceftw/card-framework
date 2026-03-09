# uv Package Manager Guide

This repository uses `uv` as its authoritative package manager and environment manager.

Use `uv` for environment creation, dependency installation, command execution, dependency updates, and lockfile maintenance. Avoid ad hoc `pip` workflows unless a documented exception is introduced.

## Current Project State

- The project targets Python 3.12+.
- The checked local tool version is `uv 0.10.0`.
- Runtime dependencies live in `[project.dependencies]` in `pyproject.toml`.
- Dev dependencies currently live in `[dependency-groups].dev`.
- The lockfile is `uv.lock` and is committed repository state.

## Canonical Commands

Run these from the repository root:

```bash
uv sync
uv sync --dev
uv run python -m card_framework.cli.main
uv add <package>
uv add --dev <package>
uv remove <package>
uv lock
```

Preferred workflow:

1. `uv sync --dev` when starting work on a fresh checkout or after dependency changes.
2. `uv run ...` for project commands so they execute in the locked environment.
3. `uv add`, `uv add --dev`, or `uv remove` for dependency changes instead of editing dependency lists by hand.
4. Commit both `pyproject.toml` and `uv.lock` when dependencies change.

## Dependency Management Rules

- Runtime libraries belong in `[project.dependencies]`.
- Dev-only tools belong in a dependency group, usually `dev` unless the project intentionally introduces a more specific group.
- If you manually edit dependency declarations in `pyproject.toml`, run `uv lock` afterward and review the resulting lockfile diff.
- Keep dependency changes atomic and easy to review.

## Repo-Specific PyTorch Rule

This repo pins `torch` and `torchaudio` to the explicit `pytorch-cu128` index through `tool.uv.sources` and `[[tool.uv.index]]`, while keeping `pytorch-cu126` configured as an optional fallback index for environments that need CUDA 12.6 wheels.

When changing any of those packages:

- Preserve the explicit index mapping unless the task intentionally changes the PyTorch source strategy.
- Review both the dependency declaration and the `tool.uv.sources` or index configuration together.
- Update this document if the project switches to a different CUDA build, CPU wheels, or a different source layout.

## Command Usage Notes

- Prefer `uv run python -m card_framework.cli.main` over raw `python -m card_framework.cli.main` so the locked environment is used consistently.
- Prefer `uv sync --dev` for local development because Ruff is currently a dev dependency.
- Use `uv add --dev` for developer tooling such as linters or test tools.
- Use `uv lock` after meaningful dependency edits or when reconciling a lockfile mismatch.

## Do Not Do This

- Do not use `pip install ...` for normal project dependency management.
- Do not edit `uv.lock` manually.
- Do not change package indexes or `tool.uv.sources` casually, especially for the PyTorch stack.
- Do not land dependency changes without the matching `uv.lock` update.

## When To Update This Document

Update this file whenever any of the following changes:

- The standard `uv` commands for this repo.
- Dependency group layout.
- Lockfile policy.
- Package source or index policy.
- Runtime entrypoint strategy.
