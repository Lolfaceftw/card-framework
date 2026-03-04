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