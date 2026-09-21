# Duration-study context and output policy

The study retains the complete source excerpt and accumulated messages within
each summarizer or critic pass. Each outer critic revision still starts a new
task with the existing previous-draft, feedback, and issue-memory inputs. These
changes do not introduce cross-pass conversation replay or change that memory
policy.

## Required run settings

Use the same settings in both duration conditions:

```text
selection.embedding=disabled
orchestrator.prompt_compaction.enabled=false
orchestrator.prompt_compaction.retry_on_overflow=false
orchestrator.prompt_compaction.max_overflow_retries=0
audio.interjector.analysis_max_tokens=null
audio.interjector.repair.prompt_max_tokens=null
```

The selected provider must also omit its default output-token limit and any
request-body prompt truncation setting. Provider errors and output-length
termination must fail the attempt. An omitted client limit does not remove a
server's finite context or generation limits.

## Implemented behavior

- Disabled prompt compaction cannot run through the forced overflow path.
- The summarizer's in-budget review no longer introduces a 256-token limit.
  It retains the normal configured request budget, which is unset in this study.
- Assistant reasoning fields remain in the within-pass message history.
- Explicit null interjector limits remain null through configuration loading.
  Per-turn planning no longer imposes an additional 400-token limit.
- Interjection planning and repair propagate provider exceptions. Failed calls
  cannot turn into default plans or omitted repairs.
- Text planning and repair require complete JSON. The partial-decisions parser
  has been removed; malformed responses raise validation errors.

Stage-4 tool planning still assesses one complete host/next-speaker pair per
request. It does not receive the full source excerpt. Existing audio overlap
fitting and timing decisions are unchanged.

## Scope and remaining limits

The outer-revision issue memory retains its existing maximum of five issue
entries and six remedy entries, with short issue descriptions. Accordingly, the
study must describe **full per-pass source and message history**, not complete
conversation retention across all outer revisions.

Tool-call argument normalization is still the existing application behavior:
malformed arguments may be normalized before replay. This is separate from
prompt-length clipping. The configured maximum number of tool turns and critic
passes remains a termination policy, not an output-token limit.

The critic's separately configured repetition guard can interrupt a stream and
use its existing retry/verdict behavior. A run requiring uninterrupted responses
must disable that guard explicitly. Provider context/length failures must not be
classified as repetition-guard events.

## Verification

Run targeted Ruff and Python compilation checks for the edited agent and audio
files. Validate the JSON parsers against retained real interjection records,
including deliberately incomplete versions of those same records as rejection
checks. A live smoke run must additionally verify complete source text in the
submitted requests, absence of output limits during the in-budget review and
stage-4 calls, and honest failure on provider overflow or length termination.

On 2026-09-20, targeted Ruff, Python compilation, and imports passed. The strict
parsers accepted six rendered decisions and two complete logged repair responses
from retained run `interjector_prompt_perspective_ep2_5m_20260325_204520`.
Removing the final JSON delimiter from those same real records produced the
expected validation errors. Those local checks did not call an LLM or establish
that the institutional endpoint preserves context; the live smoke run remains
necessary.
