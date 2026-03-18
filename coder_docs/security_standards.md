# Security Standards

Use this file as the source of truth for repository security expectations when
editing runtime code, benchmark code, packaging logic, prompts that influence
external calls, or developer workflow automation.

## Core Rules

- Treat every external input as untrusted and validate it with allowlists and
  explicit constraints.
- Use subprocess argument arrays instead of interpolated shell commands.
- Set timeouts on network calls and keep TLS verification enabled.
- Never deserialize untrusted data with unsafe mechanisms such as `pickle` or
  unsafe YAML loaders.
- Store secrets in configuration or secret managers, never in code or logs.
- Use the `secrets` module for security-sensitive token generation and
  constant-time comparison for sensitive values.
- Resolve and constrain file paths to approved base directories before reading,
  writing, or extracting files.
- Use secure temporary files and directories instead of predictable temp paths.
- Never log secrets, tokens, raw credentials, or sensitive payloads.
- Pin and audit dependencies, and make security checks part of CI.
- Apply least privilege to runtime identity, filesystem access, network reach,
  and credentials.
- Prefer secure defaults over opt-out security toggles in production paths.

## Repo-Specific Reminders

- Do not stage, commit, or publish private endpoints such as vLLM URLs, API
  keys, tokens, or other sensitive repo-local values.
- Before commit or pull request, review diffs for secrets and other private
  data.
- When working on workflows, release logic, deployment-adjacent code, or
  security-sensitive modules, apply extra review scrutiny.
- Treat `coder_docs/git_github_workflow.md` and
  `coder_docs/github_actions_release_spec.md` as companion docs for
  security-relevant contributor workflow expectations.
