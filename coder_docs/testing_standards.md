# Testing Standards

Use this file as the source of truth for testing expectations in this
repository.

## Core Rules

- Prefer realistic tests that reflect the real pipeline rather than toy tests
  detached from production behavior.
- Keep most test value at the integration level, with unit tests for focused
  logic and fewer end-to-end tests.
- Test behavior and contracts rather than private implementation details.
- When an implementation change affects behavior, interfaces, contracts, or
  outputs that tests should cover, update or add the relevant tests in the same
  change.
- Keep tests deterministic by controlling time, randomness, environment, and
  side effects.
- Use fixtures as explicit setup with minimal scope and reliable teardown.
- Mock only true external boundaries such as networks, cloud services, or
  operating-system edges.
- Patch where the object is looked up, and prefer autospec-style mocks when
  mocking is necessary.
- Use realistic schemas, payloads, and pipeline inputs instead of trivial
  placeholder data.
- Add a regression test for every bug fix at the lowest level that reproduces
  the real failure path.
- Enforce strict pytest configuration, explicit markers, and order-independent
  tests.
- Treat flaky tests as defects. Fix them, quarantine them with ownership and a
  deadline, or remove redundant ones.
- Design CI for fast, reliable feedback: quick deterministic checks first,
  broader suites afterward.

## Repo-Specific Reminders

- Most coverage belongs in the existing subsystem test directories under
  `tests/`.
- When runtime flow, prompts, provider behavior, benchmark workflow, or output
  contracts change, update the corresponding tests before considering the work
  complete.
- Use `uv run pytest` for repository test execution and `uv run ruff check .`
  for lint validation.
