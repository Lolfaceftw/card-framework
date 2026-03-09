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
uv build --wheel
uv publish --dry-run
```

Preferred workflow:

1. `uv sync --dev` when starting work on a fresh checkout or after dependency changes.
2. `uv run ...` for project commands so they execute in the locked environment.
3. `uv add`, `uv add --dev`, or `uv remove` for dependency changes instead of editing dependency lists by hand.
4. Commit both `pyproject.toml` and `uv.lock` when dependencies change.
5. Use `uv build` for release artifacts and `uv publish --dry-run` before any real package upload.
6. Keep the GitHub trusted-publishing workflow aligned with the local release commands. This repo's `.github/workflows/publish-pypi.yml` uses `uv build --no-sources` followed by `uv publish`.

## Dependency Management Rules

- Runtime libraries belong in `[project.dependencies]`.
- Dev-only tools belong in a dependency group, usually `dev` unless the project intentionally introduces a more specific group.
- If you manually edit dependency declarations in `pyproject.toml`, run `uv lock` afterward and review the resulting lockfile diff.
- Keep dependency changes atomic and easy to review.
- Keep release-critical direct references in `[project.dependencies]`, not only in `tool.uv.sources`, when downstream `pip install` must preserve an exact non-PyPI source.

## Repo-Specific Git Dependency Rule

`ctc-forced-aligner` is intentionally declared as a direct Git requirement in
`[project.dependencies]`:

- `ctc-forced-aligner @ git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git@e23e1525bae810f0582b6e539ce7aec63fd01196`

This is not cosmetic. A bare `ctc-forced-aligner` requirement resolves on PyPI
to Deskpai's `1.0.2` source distribution, which does not provide the API this
repo imports and fails on Windows during linking with
`LNK2001: unresolved external symbol PyInit_align_ops`.

When changing this dependency:

- Review the direct reference in `[project.dependencies]` and the locked Git source together.
- Do not move it back to a bare package name unless the public PyPI package and API have been revalidated.
- Keep the publish smoke test aligned so built artifacts still expose the Git-pinned requirement in package metadata.

## Repo-Specific PyTorch Rule

This repo pins `torch` and `torchaudio` to the explicit `pytorch-cu126` index through `tool.uv.sources` and `[[tool.uv.index]]`. The repository's packaged CUDA contract is now CUDA 12.6 only.

When changing any of those packages:

- Preserve the explicit index mapping unless the task intentionally changes the PyTorch source strategy.
- Review both the dependency declaration and the `tool.uv.sources` or index configuration together.
- Update this document if the project switches to a different CUDA build, CPU wheels, or a different source layout.

## Command Usage Notes

- Prefer `uv run python -m card_framework.cli.main` over raw `python -m card_framework.cli.main` so the locked environment is used consistently.
- Prefer `uv sync --dev` for local development because Ruff is currently a dev dependency.
- Use `uv add --dev` for developer tooling such as linters or test tools.
- Use `uv lock` after meaningful dependency edits or when reconciling a lockfile mismatch.
- Use `uv build --wheel` or `uv build` to produce release artifacts for `card-framework`.
- Use `uv publish --dry-run` as the required preflight before uploading to PyPI with `uv publish`.
- For GitHub Actions releases, prefer `uv build --no-sources` so release builds do not accidentally depend on local `tool.uv.sources` overrides that downstream users will not have. Direct Git references in `[project.dependencies]`, such as `ctc-forced-aligner`, are still expected to survive in the built metadata.

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
