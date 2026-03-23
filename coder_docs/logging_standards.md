# Logging Standards

Use this file as the source of truth for logging expectations in this
repository.

## Core Rules

- Use structured logs by default, preferably JSON or consistent key-value
  output.
- Standardize a minimum event schema with timestamp, severity, message,
  service, environment, logger, event name, outcome, and request or correlation
  identifier.
- Use RFC 3339 or ISO 8601 timestamps with timezone information, preferably
  UTC.
- Use severity levels consistently so logs support alerting and triage.
- Attach request, trace, span, or correlation context to relevant events.
- Log exceptions once at the handling boundary with enough context to diagnose
  the failure.
- Redact or exclude secrets, tokens, credentials, and other sensitive data from
  logs.
- Use parameterized logging rather than eager string interpolation.
- Configure logging centrally and use module-level loggers instead of ad hoc
  root logger usage.
- Use non-blocking logging pipelines for high-throughput services when
  operationally necessary.
- Protect log integrity, retention, and access as part of the security
  posture.
- Do not produce noisy narration logs. Log decisions, state transitions,
  failures, and useful diagnostics.

## Repo-Specific Reminders

- Follow the runtime logging configuration rather than adding ad hoc terminal
  prints.
- Keep operator-facing progress output distinct from diagnostic log payloads.
- When validating or supervising a real run, inspect persisted logs rather than
  treating the live terminal stream as sufficient evidence on its own.
- When changing logger shape or terminal behavior, update the relevant docs in
  `coder_docs/codebase_guide.md` and other operator-facing Markdown files.
