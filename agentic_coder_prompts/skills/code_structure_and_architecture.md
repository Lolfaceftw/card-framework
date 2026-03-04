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