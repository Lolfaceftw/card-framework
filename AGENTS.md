# AGENTS.md

## Scope
These instructions apply to all AI agents and contributors in this repository. Follow them for all Python code and prompt design work.

## 1) Prompt Engineering Standard

### Required prompt structure
Every non-trivial prompt must include:
1. `Objective`: One clear outcome.
2. `Context`: Minimal but sufficient domain details and constraints.
3. `Inputs`: Explicit input schema/types.
4. `Output contract`: Required format (JSON/schema/sections), length limits, and quality bar.
5. `Rules`: Hard constraints, safety boundaries, and forbidden behavior.
6. `Examples`: At least one high-quality example when format fidelity matters.
7. `Evaluation`: Acceptance criteria/checklist the model can self-verify against.

### Prompt quality rules
- Be specific, measurable, and testable; avoid vague instructions.
- Separate hard constraints from soft preferences.
- Prefer delimiter-based sections over long prose blocks.
- Ask for cited uncertainty when confidence is low.
- For iterative tasks, require a short plan before execution.
- For extraction/classification, define labels and edge-case handling.

### Prompt anti-patterns (disallowed)
- Hidden goals or conflicting instructions in one block.
- Open-ended style requests without acceptance criteria.
- Ambiguous output formats for machine-consumed results.

## 2) Python Type Hinting Standard

### Baseline
- Target Python `>=3.10`.
- Public functions, methods, class attributes, and module constants must be typed.
- All function signatures must include return types.
- Use modern syntax: `X | Y` (PEP 604), built-in generics (`list[str]`, `dict[str, int]`).
- Use `typing`/`typing_extensions` constructs when needed (`TypedDict`, `Protocol`, `Literal`, `TypeAlias`, `Self`).

### Design rules
- Prefer precise types over `Any`; justify unavoidable `Any` with a short comment.
- Model structured dict payloads with `TypedDict` or Pydantic models.
- Use `Protocol` for behavior-based interfaces.
- Use `dataclass(slots=True)` (or Pydantic models) for structured internal data where appropriate.
- Keep I/O boundaries explicit: parse/validate early, use typed domain objects internally.

## 3) Docstring Standard

### Required coverage
- Public modules, classes, functions, and methods must have docstrings.
- Non-public helpers need docstrings when logic is non-obvious.

### Format
- Follow PEP 257 conventions.
- Use Google-style sections consistently:
  - `Args:`
  - `Returns:`
  - `Raises:`
  - `Yields:` (if generator)
  - `Examples:` (for non-trivial usage)
- First line is a short imperative summary.
- Document units, ranges, defaults, side effects, and failure modes.

## 4) Industry Logging Standard

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

## 5) Engineering Quality Gates

Before merging Python changes:
1. Type checks pass (mypy or pyright).
2. Lint and format pass (ruff + formatter).
3. Tests pass (`pytest`), including failure-path assertions where relevant.
4. New public APIs include type hints and docstrings.
5. Logging added for critical paths and error handling.

## 6) Definition of Done for Agent-Generated Changes
- Prompt text follows Section 1 structure.
- New/changed Python code follows Sections 2-4.
- Observability impact is explicit: what logs were added/changed and why.
- Any intentional deviation is documented in the PR/change notes.
