# AGENTS.md

## Scope
These instructions apply to all AI agents and contributors in this repository. Follow them for all Python code and prompt design work.

## 1) Prompt Engineering Standard

### Required prompt structure
Every non-trivial prompt MUST use delimiter-based sections in this exact order:
1. `Objective`: One clear, testable outcome.
2. `Context`: Minimal domain details, constraints, and assumptions.
3. `Inputs`: Explicit input schema/types with required vs optional fields.
4. `Output contract`: Exact output format (JSON/schema/sections), max length, and quality bar.
5. `Evidence contract`: Required when the prompt requests any execution/verification claim.
6. `Rules`: Hard constraints, safety boundaries, and forbidden behavior.
7. `Examples`: At least one high-quality example when format fidelity matters.
8. `Evaluation`: Acceptance checklist the model can self-verify before finalizing.

### Evidence-aware output contract rules
For prompts that request validation claims (tests, lint, type checks, performance, security, command execution), the `Output contract` MUST require machine-readable evidence with:
- Artifact path + SHA-256 hash for each referenced artifact.
- Command provenance (`producer_command`) for each artifact.
- Run provenance (`run_id`, `git_commit`, `git_branch`, `generated_at_utc`).
- `junit_totals` (`tests`, `failures`, `errors`, `skipped`) for pytest-related claims.
- A fallback state `not_verified` when any required artifact is missing, unreadable, stale, or hashless.

### Prompt quality rules
- Use specific, measurable, testable instructions.
- Separate hard constraints from soft preferences.
- Prefer delimiter-based sections over long prose blocks.
- Require cited uncertainty when confidence is low.
- For iterative tasks, require a short plan before execution.
- For extraction/classification, define labels and edge-case handling.
- Bind validation claims to current-run artifacts, not prior memory.

### Prompt anti-patterns (disallowed)
- Hidden goals or conflicting instructions in one block.
- Open-ended style requests without acceptance criteria.
- Ambiguous output formats for machine-consumed results.
- Any prompt that allows pass/fail claims without artifact path + SHA-256.
- Any prompt that permits inferred/estimated command results instead of executed evidence.

### Compliant Prompt Example (Evidence-Backed Validation)

```text
Objective
Validate repository quality gates and return only artifact-backed verification results.

Context
Repository root: C:\Users\chris\card-framework-merged
Use uv workflows and scripts/quality_gate.py.
This is a validation task; evidence is mandatory.

Inputs
{
  "tier": "fast | full",
  "repo_root": "string (absolute path)"
}

Output contract
Return JSON only with this schema:
{
  "status": "verified | failed | not_verified",
  "claims": [
    {
      "id": "ruff | mypy | pytest",
      "status": "pass | fail | not_verified",
      "statement": "string",
      "evidence_refs": ["report_json", "junit_xml"]
    }
  ],
  "evidence": {
    "report_json": {
      "path": "artifacts/quality/<run_id>/report.json",
      "sha256": "64 lowercase hex",
      "producer_command": "uv run python scripts/quality_gate.py --tier <tier>"
    },
    "junit_xml": {
      "path": "artifacts/quality/<run_id>/pytest.junit.xml",
      "sha256": "64 lowercase hex",
      "producer_command": "uv run --extra dev python -m pytest tests -q --junitxml artifacts/quality/<run_id>/pytest.junit.xml",
      "totals": { "tests": 0, "failures": 0, "errors": 0, "skipped": 0 }
    },
    "provenance": {
      "run_id": "string",
      "git_commit": "40-char sha",
      "git_branch": "string",
      "generated_at_utc": "ISO-8601 UTC timestamp"
    }
  }
}

Rules
- Execute commands; do not infer results from memory.
- Do not claim pass unless required artifacts exist and hashes are present.
- If any evidence element is missing/unreadable, set claim status to `not_verified` and explain in `statement`.
- Never fabricate hashes, paths, totals, or command outputs.

Examples
Example valid claim:
{
  "id": "pytest",
  "status": "pass",
  "statement": "Pytest passed with zero failures/errors.",
  "evidence_refs": ["report_json", "junit_xml"]
}

Evaluation
- All required fields present and machine-parseable.
- `report.json` path exists and hash is 64 lowercase hex.
- JUnit totals satisfy tests > 0, failures = 0, errors = 0 for pass claims.
- Evidence paths and run_id are consistent.
```

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

### Scope
This section applies to every claim about executed checks (lint, type checks, tests, quality gates, performance checks, security checks, or command outcomes).

### Required evidence bundle (hard requirement)
Any verification claim MUST include all of the following:
- `report_json.path`: `artifacts/quality/<run_id>/report.json`
- `report_json.sha256`: SHA-256 of the exact `report.json` bytes
- `junit_xml.path`: pytest JUnit XML path used for totals
- `junit_xml.sha256`: SHA-256 of the exact JUnit XML bytes
- `junit_totals`: `tests`, `failures`, `errors`, `skipped`
- `commands_executed`: exact commands that produced each artifact
- `provenance`: `run_id`, `git_commit`, `git_branch`, `generated_at_utc`

### Validity rules
A claim of "passed" is valid only if:
- Artifacts exist at the cited paths and are readable.
- Hashes are present for every cited artifact.
- `report.json` indicates no failing quality-gate step for the claimed checks.
- JUnit XML exists and reports `tests > 0`, `failures = 0`, and `errors = 0`.
- Artifact paths and `run_id` reference the same run.
- Evidence was produced for the current branch/commit or is explicitly marked stale.

### Anti-fabrication controls (zero tolerance)
- Never claim checks passed without executed command evidence.
- Never fabricate, paraphrase, or infer command outputs from memory.
- Never provide a hash that was not computed from the cited file.
- Never present stale evidence as current evidence.
- If evidence is missing, stale, or inconsistent, return `not_verified` instead of pass/fail.

### Prompt contract requirement
- Any prompt requesting verification MUST require a machine-readable evidence object with artifact `path` + `sha256` fields; prose-only validation summaries are non-compliant.

## 9) Python Package Management Standard

### Required tooling
- Use `uv` as the default and required package manager/runtime wrapper for Python workflows in this repository.
- Use `uv add` and `uv remove` for dependency changes instead of manual dependency edits.
- Run `uv lock` whenever dependencies are added, removed, or changed so the lockfile stays current.
- If dependency fields in `pyproject.toml` are edited manually, run `uv lock` immediately in the same change.
- Validate lock freshness with `uv lock --check` before claiming checks passed.
- Use `uv sync` for local environment setup and dependency installation.
- Use `uv sync --locked` for CI or any reproducible install path.
- Run Python tools via `uv run`, including tests, lint, type checks, and scripts (for example: `uv run pytest`, `uv run ruff check .`, `uv run mypy`).
- For workspaces, run dependency lock/sync from the workspace root; use package-scoped execution when needed (for example, `uv run --package <name> ...`).

### Disallowed by default
- Do not use `pip`, `pip3`, `python -m pip`, `poetry`, or `conda` for project dependency management unless a task explicitly requires it and the deviation is documented in change notes.
- Do not use `uv pip install` to manage project dependencies tracked by `pyproject.toml`/`uv.lock`.

### Verification commands
- `uv lock --check`
- `uv sync --locked`
- `uv run pytest`

## 10) Design Patterns Standard

### Decision Coverage Requirement
For every non-trivial Python change, explicitly choose at least one pattern from each group:
1. Architecture.
2. Code-level design.
3. Reliability.
4. Integration.
5. Verification plan aligned to Sections 6 and 8 (`scripts/quality_gate.py` artifacts required for pass claims).

If a pattern is intentionally not used, document why in change notes or ADR.

### 10.1 Architecture Patterns

1. `Ports and Adapters (Hexagonal Boundary)`
Objective: Keep domain/application logic independent from infrastructure details.
When to use: New external I/O (DB/API/files/SDK) or pluggable providers.
When not to use: Small pure in-memory helpers with no external dependencies.
Minimal Python example:
```python
from pathlib import Path
from typing import Protocol

class SummaryStore(Protocol):
    def save(self, run_id: str, text: str) -> None: ...

class FileSummaryStore:
    def save(self, run_id: str, text: str) -> None:
        Path(f"artifacts/{run_id}.txt").write_text(text, encoding="utf-8")
```
Anti-patterns: Domain code importing SDK clients directly; cross-layer imports into adapter internals.
Verification checks: `uv run mypy` enforces protocol compatibility; contract tests cover adapter behavior; quality evidence recorded via `scripts/quality_gate.py`.

2. `Use-Case Orchestrator`
Objective: Centralize multi-step workflow coordination while keeping business steps isolated.
When to use: Flows with stage ordering, branching, retries, and event emission.
When not to use: Single-step operations with no orchestration logic.
Minimal Python example:
```python
from dataclasses import dataclass

@dataclass(slots=True)
class Orchestrator:
    stage1: Stage1Runner
    stage2: Stage2Runner

    def run(self, request: Stage1Request) -> Stage2Output:
        stage1_out = self.stage1.run(request)
        return self.stage2.run(Stage2Request(transcript_json_path=stage1_out.transcript_json_path))
```
Anti-patterns: Business logic in CLI/controller; stage side effects hidden across utility modules.
Verification checks: Integration tests assert stage order and outputs; logs include stage lifecycle at INFO; failure-path tests cover abort/skip branches.

3. `Composition Root (Builder + DI)`
Objective: Wire concrete implementations in one place and inject dependencies inward.
When to use: Startup/bootstrap of app, jobs, CLI entrypoints.
When not to use: Inside domain services or library modules.
Minimal Python example:
```python
def build_pipeline(config: AppConfig) -> PipelineOrchestrator:
    return (
        PipelineBuilder()
        .with_runtime_context(RuntimeContext.from_config(config))
        .with_stage1(StageProviderFactory.create_stage1(config))
        .with_stage2(Stage2SummarizeRunner(orchestrator=make_summarizer(config)))
        .build()
    )
```
Anti-patterns: Scattered object construction; hidden globals/singletons as implicit dependencies.
Verification checks: `mypy` on constructors/factories; unit tests for composition root wiring; no transport-specific imports in domain/application modules.

### 10.2 Code-Level Patterns

4. `Strategy via Protocol`
Objective: Swap algorithms/behaviors without condition-heavy branching.
When to use: Multiple interchangeable policies (budgeting, ranking, provider behavior).
When not to use: Only one stable implementation with no foreseeable variants.
Minimal Python example:
```python
from typing import Protocol

class BudgetStrategy(Protocol):
    def target_words(self, minutes: float) -> int: ...

class FixedWpmBudget:
    def __init__(self, wpm: float) -> None:
        self._wpm = wpm

    def target_words(self, minutes: float) -> int:
        return max(1, int(round(self._wpm * minutes)))
```
Anti-patterns: `if/elif` ladders for provider modes spread across files.
Verification checks: Protocol conformance under `mypy`; unit tests per strategy; complexity stays within lint thresholds.

5. `Factory Method / Abstract Factory`
Objective: Select concrete implementations from typed config/runtime mode.
When to use: Runtime provider selection for stages, LLMs, adapters.
When not to use: Fixed implementation with no runtime variability.
Minimal Python example:
```python
import os
from typing import Literal

ProviderKind = Literal["openai", "heuristic"]

def make_provider(kind: ProviderKind) -> LLMProvider:
    if kind == "openai":
        return OpenAIProvider(api_key=os.environ["OPENAI_API_KEY"])
    return HeuristicProvider()
```
Anti-patterns: Direct class selection in controllers; factory returning untyped `Any` without justification.
Verification checks: Typed factory signatures; config validation tests for unsupported modes; contract tests for each returned provider.

6. `Typed DTOs and Value Objects`
Objective: Keep internal contracts explicit, validated, and immutable where possible.
When to use: Data crossing layer boundaries or representing policy/config.
When not to use: Ephemeral local variables inside short pure functions.
Minimal Python example:
```python
from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class WordBudget:
    min_words: int
    max_words: int

    def __post_init__(self) -> None:
        if self.min_words <= 0 or self.max_words < self.min_words:
            raise ValueError("invalid budget")
```
Anti-patterns: Raw `dict[str, Any]` passed through core logic; late validation after side effects.
Verification checks: `mypy` passes with strict public typing; unit tests for boundary validation; docstrings include units/ranges/failure modes.

### 10.3 Reliability Patterns

7. `Error Taxonomy and Exception Translation`
Objective: Classify recoverable vs non-recoverable failures at boundaries.
When to use: Any external I/O, parsing, subprocess, model/provider call.
When not to use: Tiny private helpers where built-in errors are already precise.
Minimal Python example:
```python
class RetryableProviderError(RuntimeError):
    pass

class NonRetryableProviderError(RuntimeError):
    pass

def fetch_text(client: ProviderClient) -> str:
    try:
        return client.fetch()
    except TimeoutError as exc:
        raise RetryableProviderError("provider timeout") from exc
```
Anti-patterns: Catch-all `except Exception` with silent ignore; leaking vendor exceptions into domain layer.
Verification checks: Tests assert mapped exception classes; logs on ERROR include stack traces (`logger.exception`/`exc_info=True`); failure-path tests required.

8. `Timeout + Bounded Retry + Jitter`
Objective: Handle transient faults without hanging or retry storms.
When to use: Remote API/network/subprocess calls that are safe to retry.
When not to use: Non-idempotent operations without deduplication or compensation.
Minimal Python example:
```python
import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

def run_with_retry(call: Callable[[], T], attempts: int = 3) -> T:
    for n in range(1, attempts + 1):
        try:
            return call()  # call must enforce timeout internally
        except RetryableProviderError:
            if n == attempts:
                raise
            time.sleep(min(2.0, 0.25 * (2 ** (n - 1)) + random.uniform(0.0, 0.1)))
    raise RuntimeError("unreachable")
```
Anti-patterns: Infinite retries; missing timeout; retrying validation/auth errors.
Verification checks: Unit tests assert max attempts and backoff bounds; integration tests simulate transient failures; logs include attempt count and terminal status.

9. `Idempotent Operations + Graceful Fallback`
Objective: Make retries safe and preserve service continuity with degraded output when needed.
When to use: Artifact writes, repeated job execution, optional subsystems.
When not to use: Safety-critical operations where degraded output is unacceptable.
Minimal Python example:
```python
import json
from pathlib import Path

def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)  # atomic replace

class NullPlanner:
    def propose(self, _: str) -> str | None:
        return None
```
Anti-patterns: Partial file writes; fallback that hides hard failures without logs/metrics.
Verification checks: Re-run tests prove same input -> same persisted state; fallback path emits WARNING with reason; integration tests validate degraded-mode contract.

### 10.4 Integration Patterns

10. `Anti-Corruption Adapter`
Objective: Translate external payloads into internal typed domain contracts.
When to use: Third-party APIs with unstable or vendor-specific schemas.
When not to use: Internal modules already sharing stable typed contracts.
Minimal Python example:
```python
from typing import TypedDict

class VendorSegment(TypedDict):
    id: str
    speaker_name: str
    content: str

def to_domain(raw: VendorSegment) -> TranscriptSegment:
    return TranscriptSegment(
        segment_id=raw["id"],
        speaker=raw["speaker_name"],
        text=raw["content"].strip(),
    )
```
Anti-patterns: Vendor payloads flowing through domain/application layers unchanged.
Verification checks: Contract tests for mapping edge cases; schema validation at ingress; backward-compatibility tests when provider schemas evolve.

11. `Gateway Client Wrapper`
Objective: Isolate SDK/client details behind a narrow typed interface with timeouts.
When to use: Any direct dependency on third-party SDKs or transport protocol.
When not to use: Local in-process libraries with stable pure APIs.
Minimal Python example:
```python
from typing import Protocol

class LLMGateway(Protocol):
    def generate(self, prompt: str, timeout_seconds: float) -> str: ...

class OpenAIGateway(LLMGateway):
    def generate(self, prompt: str, timeout_seconds: float) -> str:
        return (
            self._client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout_seconds,
            ).choices[0].message.content
            or ""
        )
```
Anti-patterns: SDK objects passed across modules; timeout policy duplicated in callers.
Verification checks: Unit tests mock gateway only; integration tests exercise real client in isolated suite; logging includes provider/model/request_id metadata.

12. `Contract and Integration Test Harness`
Objective: Enforce interface behavior across all implementations.
When to use: Multiple adapters/providers implementing one `Protocol`.
When not to use: Single concrete class with no extension point.
Minimal Python example:
```python
def assert_provider_contract(provider: LLMProvider) -> None:
    text = provider.generate(
        system_prompt="JSON only",
        user_prompt="hello",
        max_tokens=32,
        temperature=0.0,
    )
    assert isinstance(text, str)
    assert text.strip()
```
Anti-patterns: Testing only one provider; no shared contract assertions.
Verification checks: `tests/contract` required for each protocol family; `tests/integration` required for critical paths; pass claims must cite `artifacts/quality/<run_id>/report.json` + SHA-256 + JUnit totals.

## 11) Markdown + XML Prompt Delimiter Style Guide

### Heading Hierarchy
- Use one `#` H1 for the document title.
- Use `##` for major sections and `###` for subsections.
- Do not skip levels (for example, `##` to `####`).
- Keep one blank line before and after headings.

### Delimiter Conventions
- Keep policy/process guidance in plain Markdown.
- Use XML-style tags for multi-part prompt payloads with distinct fields.
- Use consistent, descriptive tag names across prompts.
- Nest tags only for true hierarchy, and always close tags.
- For simple instruction/context splits, `###` sections or `"""` delimiters are acceptable.
- Avoid mixing multiple delimiter styles inside one prompt unless required.

### Fenced Block Usage
- Put XML prompt templates inside fenced code blocks using `xml`.
- Add an info string to every fenced block (`xml`, `json`, `bash`, `text`, etc.).
- Use one fence style repo-wide (recommended: backticks).
- Keep one blank line around each fenced block.

### Lintability and Readability Rules
- Keep heading style consistent (ATX `#` headings).
- Keep line length within repo standard (recommended: 100 chars max).
- Keep blank lines around headings and code fences.
- Avoid raw inline HTML/XML in prose; prefer fenced examples.
- If raw tags are required outside fences, allow-list only needed elements in markdownlint `MD033`.
- Use narrow, temporary markdownlint disables with a short reason, then re-enable immediately.

### Markdown vs XML: Decision Rule
| Prefer Markdown | Prefer XML Blocks |
|---|---|
| Human-facing rules, conventions, and checklists | Multi-part prompt payloads with explicit fields |
| Narrative guidance and high-level examples | Inputs requiring strict boundaries between sections |
| Short prompts with 1-2 sections | Reusable templates with nested structure |

### Canonical XML Prompt Template
```xml
<prompt>
  <objective>...</objective>
  <context>...</context>
  <inputs>...</inputs>
  <output_contract>...</output_contract>
  <rules>...</rules>
  <examples>...</examples>
  <evaluation>...</evaluation>
</prompt>
```

### Before / After Example

Before:
````markdown
## prompt
Analyze tickets <instructions>prioritize by severity</instructions>
<context>Use taxonomy v2</context>
Return JSON.
````

After:
````markdown
## Ticket Triage Prompt

### Objective
Prioritize tickets by severity and group by product area.

```xml
<prompt>
  <instructions>Assign P0-P3 severity using policy X.</instructions>
  <context>Use taxonomy v2.</context>
  <inputs>{{tickets_json}}</inputs>
  <output_contract>{"prioritized_tickets":[...],"summary":"..."}</output_contract>
</prompt>
```
````

