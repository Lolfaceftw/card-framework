# Python Engineering Standards

Use this file as the source of truth for Python coding standards in this
repository when you are editing application code, library code, prompts that
shape Python-facing contracts, or tests that depend on public Python behavior.

These rules apply regardless of which coding assistant, IDE, or runtime you
are using.

## Baseline

- Prefer explicit, maintainable Python over clever shortcuts.
- Assume configuration exists; do not hardcode deploy-varying values.
- Write for the next developer who must read, debug, and extend the code.
- Optimize for clarity first. Straightforward code beats clever indirection.
- When principles conflict, choose the option that preserves readability and
  change safety.

## Docstrings

- Use triple double quotes for docstrings.
- Require docstrings on public modules, public classes, and public functions or
  methods.
- Start with a short imperative summary line ending with punctuation.
- For multi-line docstrings, use a summary line, one blank line, then details.
- Use Google-style sections when needed: `Args`, `Returns`, `Raises`,
  `Yields`, and `Attributes`.
- Do not restate the function signature inside the docstring.
- Describe behavior, constraints, side effects, and failure modes rather than
  obvious mechanics.
- Do not duplicate type information in docstrings when type hints already make
  it clear.

## Type Hints

- Use Python `3.12+` typing syntax such as `list[str]`, `dict[str, int]`,
  `A | B`, and type aliases.
- Annotate all public function and method boundaries, including explicit return
  types.
- Use `-> None` explicitly for procedures that do not return a value.
- Keep `Any` as a narrow escape hatch, not a default typing strategy.
- Prefer abstract collection interfaces in public APIs and concrete mutable
  types only where mutation is required.
- Narrow unions before use with explicit checks rather than assuming a variant.
- Use scoped `type: ignore` comments only when unavoidable, and explain them
  briefly.
- Do not leave public code untyped when the contract matters.

## Design And Abstractions

- Prefer composition over inheritance by default.
- Use callable injection, protocols, or small interfaces before introducing
  class-heavy factories or strategies.
- Place adapters at system boundaries where external interfaces do not match
  internal expectations.
- Use decorators or wrappers for orthogonal concerns such as logging, retries,
  caching, metrics, or auth.
- Avoid singleton-style hidden global state; favor explicit wiring and
  controlled lifecycle management.
- Only introduce a pattern when it solves a real duplication, change, or
  testability problem.
- Keep patterns reversible; if the abstraction stops paying for itself,
  simplify it.
- Split code by reason to change, not by arbitrary size or method count.
- Use abstractions only at real volatility seams or true extension points.
- Keep interfaces small and behavior-focused.
- DRY duplicated business knowledge, validation rules, and policy logic before
  DRYing trivial syntax.
- Do not add hooks, flags, or frameworks for hypothetical future needs.
- Prefer simple working code first, then refactor once repetition or variation
  is proven.

## Python Do-Nots

- Do not use bare `except` or broad exception handling that hides unrelated
  failures.
- Do not translate exceptions without preserving the original context with
  exception chaining.
- Do not use mutable default arguments.
- Do not use `eval` or `exec` on external or dynamic input.
- Do not pass untrusted input to subprocess with `shell=True`.
- Do not use `print` for operational diagnostics in application or library
  code; use logging.
- Do not use wildcard imports.
- Do not hardcode deploy-varying constants, URLs, tokens, credentials, or
  environment-specific values; assume configuration exists.
- Do not write comments that feel like you are talking to yourself. Write
  comments for a developer reader, and explain why rather than narrating
  obvious code.
- Do not leave files, resources, or external handles open when a context
  manager is available.

## Repo-Specific Reminders

- Use `uv` as the package manager for this repository.
- Use `ruff` as the linter for this repository.
- Prefer configuration and provider-profile changes over scattered constants.
- When behavior, interfaces, or operator-visible outputs change, update the
  relevant tests and Markdown docs in the same change.
