### Required tooling
- Use `uv` as the default and required package manager/runtime wrapper for Python workflows in this repository.
- Use `uv add` and `uv remove` for dependency changes instead of manual dependency edits.
- Run `uv lock` whenever dependencies are added, removed, or changed so the lockfile stays current.
- If dependency fields in `pyproject.toml` are edited manually, run `uv lock` immediately in the same change.
- Validate lock freshness with `uv lock --check` before claiming checks passed.
- Use `uv sync` for local environment setup and dependency installation.
- Use `uv sync --locked` for CI or any reproducible install path.
- Run Python tools via `uv run`, including tests, lint, type checks, and scripts (for example: `uv run pytest`, `uv run ruff check .`, `uv run mypy`).
- For workspaces, run dependency lock/sync from the workspace root; use package-scoped execution when needed (for example, `uv run --package <name> ...`).

### Disallowed by default
- Do not use `pip`, `pip3`, `python -m pip`, `poetry`, or `conda` for project dependency management unless a task explicitly requires it and the deviation is documented in change notes.
- Do not use `uv pip install` to manage project dependencies tracked by `pyproject.toml`/`uv.lock`.

### Verification commands
- `uv lock --check`
- `uv sync --locked`
- `uv run pytest`