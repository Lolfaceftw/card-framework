# GitHub Actions Release Spec

Use this document as the source of truth for GitHub Actions release workflow
behavior, release build standards, and post-tag verification in this
repository.

This is a repo-specific operations document. It is intentionally narrow and
describes the workflows and checks that exist in this checkout now.

## Current Workflow Inventory

As of 2026-03-09, this repository has one checked-in GitHub Actions workflow:

- `.github/workflows/publish-pypi.yml`

Current contract:

- Trigger: `push` on tags matching `v*`
- Workflow name: `Publish PyPI Package`
- Build job name: `Build Distributions`
- Publish job name: `Publish To PyPI`
- Publish environment: `pypi`
- Publisher mode: GitHub OIDC trusted publishing through PyPI

Treat these names as externally important because GitHub environment rules,
PyPI trusted-publisher settings, and any future required status checks may rely
on them.

## Release Build Standards

Before pushing a release tag, create a dedicated semver-named release branch
such as `release/vX.Y.Z` from the target integration branch. Run the version
bump, release preflight, and review flow there, merge that branch, and only
then push the matching `vX.Y.Z` tag from the merged integration-branch commit.
That release-preparation branch must satisfy all of the following:

1. The branch name and intended tag agree on the same semantic version.
   Example: branch `release/v1.0.4` pairs with tag `v1.0.4`.
2. `pyproject.toml` version matches the intended tag without the leading `v`.
   Example: tag `v1.0.4` requires `version = "1.0.4"`.
3. The release artifacts build with:

   ```bash
   uv build --no-sources
   ```

4. Local preflight checks pass:

   ```bash
   uv run ruff check .
   uv run pytest tests/api/test_runtime_layout.py tests/api/test_infer_api.py tests/real/test_packaged_infer_import.py
   ```

5. The exact release artifacts pass a targeted publish dry-run:

   ```bash
   uv publish --dry-run dist/card_framework-X.Y.Z-py3-none-any.whl dist/card_framework-X.Y.Z.tar.gz
   ```

6. Built wheel and sdist metadata remain PyPI-compatible.

   Current repo-specific requirement:

   - Published metadata must not include direct URL dependencies such as
     `Requires-Dist: ... @ git+...`.
   - Published metadata must not include the bare `ctc-forced-aligner`
     dependency name, because PyPI resolves that name to an incompatible third-
     party project rather than the upstream aligner CARD expects.

7. Built artifacts must install into a clean environment with `--no-deps` and
   expose the public packaged API import shape used by the publish workflow:

   - `from card_framework import InferenceResult, infer`
   - `infer(...)` callable
   - parameter order:
     `audio_wav`, `output_dir`, `target_duration_seconds`, `device`,
     `vllm_url`, `vllm_api_key`

## Workflow Standards

Changes to `.github/workflows/publish-pypi.yml` must preserve these invariants
unless the repository intentionally changes release policy:

- Build from a clean GitHub-hosted environment.
- Use `uv build --no-sources` for release artifacts.
- Smoke-check both wheel and sdist before publish.
- Publish only after the build job succeeds.
- Use GitHub OIDC trusted publishing in the `pypi` environment.
- Keep job names stable unless the corresponding protection or documentation is
  updated in the same change.

If workflow behavior changes materially, update this file, `AGENTS.md`,
`coder_docs/git_github_workflow.md`, and `coder_docs/codebase_guide.md` in the
same change.

## Post-Tag Verification

Pushing a version tag is not the end of release work. A release is considered
done only after the triggered GitHub Actions run is watched to completion and
its success is verified.

The pushed `vX.Y.Z` tag should point at the merged integration-branch commit
that came from `release/vX.Y.Z`, not at an unmerged topic-branch tip.

Required sequence after pushing `vX.Y.Z`:

1. Find the run:

   ```bash
   gh run list --workflow "Publish PyPI Package" --limit 1
   ```

2. Watch it to completion:

   ```bash
   gh run watch <run-id> --exit-status
   ```

3. If the run fails, inspect the failure directly:

   ```bash
   gh run view <run-id> --log-failed
   ```

4. Only treat the release as complete after both jobs succeed:

   - `Build Distributions`
   - `Publish To PyPI`

5. After success, verify the published version from a clean environment when
   practical. Prefer a fresh virtual environment and install the exact version:

   ```bash
   python -m pip install --no-cache-dir card-framework==X.Y.Z
   python -c "import inspect; from card_framework import infer; print(tuple(inspect.signature(infer).parameters))"
   ```

## Failure Handling Rules

When a tagged release workflow fails:

- Do not assume the push was good enough just because the tag exists remotely.
- Do not silently move on without checking the workflow result.
- Fix the underlying issue first.
- Cut a new version and a new tag for the replacement release.

Current repo rule:

- Do not reuse a failed release version tag ambiguously. If `vX.Y.Z` failed,
  the fixed release should normally advance to a new version such as
  `vX.Y.(Z+1)` so the repository history and publish history stay traceable.

## Operator Checklist

Use this checklist for every release-tag push:

- Release branch `release/vX.Y.Z` created from the current integration branch
- Release branch merged before the matching tag is created
- Version bumped in `pyproject.toml`
- `uv lock` updated when dependency metadata changed
- `uv build --no-sources` passed
- Ruff passed
- Targeted packaged-release tests passed
- `uv publish --dry-run` passed for the exact new wheel and sdist
- Tag pushed
- `gh run watch --exit-status` succeeded
- Failed logs reviewed immediately if the run did not pass
- Public install verification completed when practical
