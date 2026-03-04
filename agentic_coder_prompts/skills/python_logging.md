### Core requirements
- Use the standard `logging` module (not `print`) for runtime events.
- Configure logging centrally (single module/function) using `logging.config.dictConfig`.
- Emit structured logs (JSON preferred in non-local environments).
- Use UTC ISO-8601 timestamps.
- Include correlation fields when available:
  - `trace_id`
  - `span_id`
  - `request_id`
  - `job_id`
  - `component`
  - `environment`
  - `version`

### Level and event policy
- `DEBUG`: diagnostic details for development/troubleshooting.
- `INFO`: lifecycle milestones and key business events.
- `WARNING`: recoverable anomalies or degraded behavior.
- `ERROR`: failed operations requiring attention.
- `CRITICAL`: service-impacting failures.
- Log exceptions with stack traces using `logger.exception(...)` or `exc_info=True`.

### Security and compliance
- Never log secrets, tokens, credentials, raw API keys, or sensitive personal data.
- Redact sensitive fields before logging.
- Avoid logging full request/response bodies unless explicitly approved and sanitized.

### Library/application behavior
- Libraries must not configure global logging; attach `NullHandler`.
- Applications own logger configuration and handler setup.
- Avoid duplicate handlers and root logger side effects.
- Ensure log rotation/retention is configured by runtime environment.