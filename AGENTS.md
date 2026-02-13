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

## 3) Code Structure and Architecture Standard

### Architecture goals
- Optimize for clean modularity, scalability, maintainability, and reliability.
- Design code so teams can evolve components independently with minimal coupling.
- Keep architecture explicit in code structure, not implicit in conventions alone.

### Module boundaries and ownership
- Organize code by bounded context or domain capability, not by arbitrary technical buckets.
- Each top-level package/module must have a clear responsibility and owner.
- Public APIs must be explicit; internal implementation details must remain private by default.
- Cross-module imports must go through well-defined interfaces, not internal modules.

### Layered design and dependency direction
- Separate concerns into clear layers where applicable:
  - Domain (business rules and core models)
  - Application/service (use cases and orchestration)
  - Infrastructure (DB, network, queues, external providers)
  - Interface/transport (HTTP handlers, CLI, jobs, workers)
- Business logic must not live in transport/controller code.
- Dependencies must point inward toward stable abstractions.
- Use `Protocol`/interfaces and dependency injection at boundaries to reduce coupling.

### Size, complexity, and cohesion
- Keep files, classes, and functions focused on a single reason to change.
- Split "god modules/classes" into smaller components when responsibilities diverge.
- Prefer small composable functions over deeply nested control flow.
- Complexity must be kept within configured linting/type-check thresholds.

### Shared code and naming
- Do not accumulate unrelated helpers in generic `utils` modules.
- Prefer domain-local helper modules before promoting shared abstractions.
- Shared modules must have precise names that reflect business intent.
- Remove dead code and stale abstractions during refactors.

### Extensibility and change safety
- Prefer composition over inheritance unless inheritance provides clear value.
- Add extension points through interfaces/protocols, not flag-heavy branching.
- Minimize breaking changes by versioning external contracts where needed.
- Document non-obvious tradeoffs when introducing new abstractions.

### Reliability and fault tolerance
- Define explicit exception/error categories for recoverable vs non-recoverable failures.
- Apply timeouts for all network and external I/O operations.
- Use retries with bounded backoff only for transient failures.
- Design retryable operations to be idempotent.
- Validate and sanitize inputs at system boundaries.

### Data and contract boundaries
- Define structured request/response payloads with typed schemas (`TypedDict`, dataclass, or Pydantic).
- Validate data at ingress and egress boundaries.
- Preserve backward compatibility for persisted data and public interfaces, or provide migrations.
- Avoid leaking infrastructure-specific payloads into domain models.

### Scalability and performance hygiene
- Identify hot paths and enforce algorithmic complexity awareness.
- Prevent unbounded memory growth and avoid hidden N+1 query patterns.
- Use batching, pagination, and streaming where data volume can grow significantly.
- Define reasonable resource limits and fail safely when limits are exceeded.

### Concurrency and lifecycle safety
- Document async/thread/process safety expectations for shared state.
- Avoid global mutable state unless guarded and justified.
- Ensure startup/shutdown paths are deterministic and observable.

### Architecture decision records
- Record major structural decisions in lightweight ADR-style notes:
  - context
  - decision
  - alternatives considered
  - consequences

## 4) Docstring Standard

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

## 5) Industry Logging Standard

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

## 6) Engineering Quality Gates

Before merging Python changes:
1. Type checks pass (mypy or pyright).
2. Lint and format pass (ruff + formatter).
3. Tests pass (`pytest`), including failure-path assertions where relevant.
4. New public APIs include type hints and docstrings.
5. Logging added for critical paths and error handling.
6. Architectural boundaries are preserved (no prohibited layer/dependency violations).
7. Complexity and maintainability thresholds pass per configured tooling.
8. Contract tests cover critical module/service boundaries when interfaces changed.
9. Reliability behavior is verified for key failure paths (timeouts, retries, fallback, idempotency).
10. Performance regression checks are run for critical paths when change risk warrants it.
11. Quality evidence is generated via `scripts/quality_gate.py` and includes machine-readable artifacts.

## 7) Definition of Done for Agent-Generated Changes
- Prompt text follows Section 1 structure.
- New/changed Python code follows Sections 2-5.
- Code structure changes follow Section 3 architecture rules.
- Observability impact is explicit: what logs were added/changed and why.
- Modularity impact is explicit: which boundaries/interfaces were added, changed, or simplified.
- Scalability implications are explicit: expected load, constraints, and performance-sensitive paths.
- Reliability implications are explicit: failure modes handled and how behavior is validated.
- Maintainability impact is explicit: complexity reduced or tradeoff documented.
- Verification evidence is explicit: include `artifacts/quality/<run_id>/report.json` path and SHA-256 hash.
- Any intentional deviation is documented in the PR/change notes.

## 8) Evidence-Based Verification Standard

### Required evidence for quality claims
- Any claim that lint, type checks, or tests passed must reference `scripts/quality_gate.py` output.
- The claim must include:
  - `artifacts/quality/<run_id>/report.json` path
  - SHA-256 hash of `report.json`
  - pytest JUnit totals (`tests`, `failures`, `errors`, `skipped`)
- A run is valid only if JUnit XML exists and reports `tests > 0`, `failures = 0`, and `errors = 0`.

### Forbidden behavior
- Never claim checks passed without executed command evidence.
- Never fabricate, paraphrase, or infer command outputs from memory.
- Never reuse stale evidence from a different branch/commit without explicitly stating it is stale.

### Prompt contract requirement
- Prompts that request validation must set an output contract requiring artifact references (path + hash), not prose-only summaries.
