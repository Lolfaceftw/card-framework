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