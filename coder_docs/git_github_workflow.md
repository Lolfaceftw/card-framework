# Git And GitHub Workflow Guide

Use this document as the source of truth for local Git staging, commit construction, push safety, and GitHub pull request workflow in this repository.

This repository expects topic branches plus pull requests for shared history. Keep changes reviewable, preserve history clarity, and prefer explicit staging over convenience shortcuts when the change is not trivial.

## Current Repo State

- As of 2026-03-07, this workspace does not contain a checked-in `.github/` directory.
- As of 2026-03-07, this workspace does not contain a checked-in `CODEOWNERS` file.
- Shared-branch protection, required status checks, secret scanning, push protection, and CODEOWNERS enforcement should therefore be treated as GitHub-hosted repository settings unless a future change checks those files into the repo.

## Working Defaults

- Branch from the integration branch you intend to merge into. In most cases this is the repository default branch or another agreed shared branch for the task.
- Keep one topic or bug fix per branch.
- Prefer several small coherent commits over one large mixed commit.
- Do not push directly to shared protected branches.
- Do not rewrite published shared history. If a shared branch needs undoing, prefer `git revert`.
- Never add files or staged content that contain API keys, access tokens, secrets, private URLs such as vLLM endpoints, customer data, or other private information.

## Local Workflow

1. Sync the target integration branch before creating or refreshing your topic branch.
2. Create or switch to your topic branch.
3. Review local changes with:

   ```bash
   git status
   git diff
   git diff --staged
   ```

4. Stage intentionally.

   - Prefer `git add <path>` or `git add -p` for non-trivial work.
   - Remember that staging captures the file contents at the moment of `git add`.
   - If you edit a file again after staging it, run `git add` again if the newer changes belong in the commit.
   - Treat `git diff --staged` as a security review boundary. If you see a secret, token, credential, private URL, or other private data, unstage it immediately and remove it from the change before committing.

5. Commit only a logical unit that can be reviewed and understood on its own.

## Commit Rules

- Use `git add -p` or equivalent selective staging when a file contains unrelated hunks.
- Avoid defaulting to `git commit -a` for non-trivial changes. It stages tracked modifications and deletions, but not new files, and it makes mixed commits easier.
- Use an imperative subject line. Aim for a short subject and add a blank line plus explanatory body when the change is not trivial.
- The commit body should explain the problem, why this approach is correct, and any important tradeoffs or discarded alternatives.
- Amend or interactively rebase only unpublished local history. Once others may depend on the branch, stop rewriting unless the branch is explicitly your own topic branch and you will push with `--force-with-lease`.

## Push Workflow

- First publish a new topic branch with:

  ```bash
  git push -u origin <branch>
  ```

- Treat a non-fast-forward push rejection as a signal to integrate first.

  1. Run `git fetch origin`.
  2. Merge or rebase onto the target upstream branch.
  3. Resolve conflicts and re-run the relevant verification.
  4. Push the resulting fast-forward history.

- Do not normalize on `git push --force`.
- If a history rewrite is necessary on your own topic branch, use `git push --force-with-lease origin <branch>`. Never use force-push on shared integration branches.

## Pull Request Workflow

- Open a draft pull request while the branch is still in progress. Use the GitHub UI or `gh pr create --draft`.
- Draft pull requests are the preferred work-in-progress state because they are non-mergeable and avoid premature CODEOWNERS review requests until the PR is marked ready for review.
- Convert the PR to ready for review only after you have:

  - reviewed the diff yourself
  - run the relevant checks for the change
  - updated affected docs
  - removed obvious WIP commits or messages

- Keep PRs small and single-purpose.
- The PR title and body should include:

  - the purpose of the change
  - a concise summary of what changed
  - linked issue(s) when tracking work exists, for example `Closes #123`
  - reviewer guidance when review order or focus areas matter

- Request specific reviewers or teams when the PR is ready. Re-request review after substantial follow-up changes.
- Keep the PR branch in sync with its base branch when required by branch protection or when drift is large enough to hide integration risk.

## GitHub Repository Controls

- Shared integration branches should require pull requests, required reviews, conversation resolution, and required status checks.
- Required status-check job names should be unique across workflows so GitHub can enforce them unambiguously.
- Sensitive paths should be covered by CODEOWNERS when the file exists, including workflows, deployment code, security-critical modules, and the CODEOWNERS file itself.
- If CODEOWNERS is added later, prefer `.github/CODEOWNERS` and protect that path with ownership rules as well.
- Enable secret scanning and push protection on the GitHub repository.
- If a secret or private endpoint is ever staged or pushed, treat it as exposed, rotate or replace it promptly, and clean up the repository history only with an approved remediation plan.
- Require or encourage signed commits and signed release tags where repository policy supports them.
- Review dependency and workflow changes with extra scrutiny before merge.
- Preserve traceability from commit to released artifact wherever the delivery pipeline supports it.

## Command Reference

```bash
git status
git diff
git diff --staged
git add -p
git commit
git fetch origin
git push -u origin <branch>
git push --force-with-lease origin <branch>
gh pr create --draft
```

## References

Research date: 2026-03-07.

- Git: <https://git-scm.com/docs/git-add>
- Git: <https://git-scm.com/docs/git-commit>
- Git: <https://git-scm.com/docs/git-push>
- Git: <https://git-scm.com/docs/giteveryday>
- Git: <https://git-scm.com/docs/gitworkflows>
- Git Book: <https://git-scm.com/book/en/v2/Git-Basics-Recording-Changes-to-the-Repository>
- GitHub Docs: <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests>
- GitHub Docs: <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/getting-started/helping-others-review-your-changes>
- GitHub Docs: <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/requesting-a-pull-request-review>
- GitHub Docs: <https://docs.github.com/en/issues/tracking-your-work-with-issues/using-issues/linking-a-pull-request-to-an-issue>
- GitHub Docs: <https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches/about-protected-branches>
- GitHub Docs: <https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners>
