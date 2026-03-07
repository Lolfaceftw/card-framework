# Academic Standards Guide

This repository is used for research-oriented work. When code introduces a non-trivial formula, algorithm, heuristic, threshold, scoring rule, statistical procedure, or other robustness-sensitive method, contributors must ground that implementation in academically defensible sources instead of inventing a "magic formula."

This document defines the default standard for methodology-sensitive implementations in this repo.

## Policy

- Prefer academically established methods before designing a new one.
- Treat peer-reviewed journals as the first search target for methodological backing.
- If journals do not provide a suitable method, use the next best authoritative source such as peer-reviewed conference papers, IEEE or similar standards, or well-established textbooks.
- Use blogs, forum posts, vendor marketing pages, and model-generated formulas only as orientation, not as the primary methodological basis for production code.
- Use preprints only when peer-reviewed support is unavailable, and say so explicitly in the implementation notes.
- Do not present an internal heuristic as if it were an academically established method.

## When This Applies

Apply this standard whenever the implementation meaningfully depends on:

- a formula or mathematical transformation,
- an algorithm or procedure with non-obvious behavior,
- a threshold or weighting rule that affects outputs,
- ranking, scoring, clustering, or deduplication logic,
- statistical estimation, aggregation, or normalization,
- a robustness claim, especially when the code may be used in research analysis or evaluation.

If the method is important enough that a reviewer could reasonably ask, "Why this formula?" or "Why should we trust this algorithm?", this standard applies.

## Required Backing For Implementations

Before implementing a robustness-sensitive method:

- Search academic journals first for an established method that fits the problem.
- Prefer the original or canonical source when it is practical to identify.
- Confirm that the selected method actually matches the repo's problem, assumptions, and data shape instead of copying a formula because it looks plausible.

When implementing a backed method in code:

- Add an inline IEEE-style citation comment near the implementation site using bracketed numbering such as `[1]`.
- Add a fuller reference in the same file whenever practical. For Python modules, prefer a module docstring `References:` section.
- If the implementation is adapted rather than copied directly, say that it is adapted and briefly note the relevant deviation.
- If a threshold, coefficient, or weighting is source-derived, cite that source in the nearby comment.
- If a threshold remains project-specific, say that explicitly instead of implying the paper endorsed it.

Minimum inline comment expectation:

```python
# IEEE citation: Adapted similarity scoring procedure from [1].
score = numerator / denominator
```

Preferred same-file reference note for Python modules:

```python
"""Compute question deduplication scores.

References:
    [1] A. Author and B. Author, "Paper title," Journal Name, vol. 10,
    no. 2, pp. 10-20, 2024, doi: 10.0000/example.
"""
```

The fuller reference should include enough information for another researcher to locate the source. Prefer authors, title, venue, year, and DOI when available.

## Custom Methods And Literature Gaps

If no academically supported method fits the problem, a custom implementation is allowed only after documenting a short `Current Literature and Gaps:` note.

That note must capture:

- what literature or source categories were searched,
- which candidate methods were considered,
- why the existing literature did not solve the repo's actual problem,
- what gap the custom method is intended to fill,
- the assumptions, tradeoffs, and known limitations of the custom method.

When a custom method is implemented:

- Mark it clearly as an internal method, heuristic, or gap-driven design.
- Cite the literature that motivated the design or showed the gap, even if no paper provided the final method directly.
- Do not describe the method as academically validated unless that validation actually exists.

## Compliant And Non-Compliant Patterns

Compliant:

- Implementing a published algorithm and citing it inline with an IEEE-style comment and fuller same-file reference.
- Adapting a published method to this repo's data while documenting the adaptation and limits.
- Introducing a custom heuristic only after documenting `Current Literature and Gaps:` and citing the sources that show why the gap exists.

Non-compliant:

- Adding a formula because it "seems to work" without academic support or a gap analysis.
- Copying a threshold, weighting, or scoring rule from a blog post without tracing it to an authoritative source.
- Presenting a project-specific heuristic as if it were a standard published method.
- Leaving a robustness-sensitive implementation without an inline citation comment.

## Reviewer Expectations

- Ask for the source when a non-trivial method appears without clear academic grounding.
- Treat missing inline citations on robustness-sensitive logic as a documentation defect.
- Treat unexplained formulas, thresholds, and heuristics as suspect until they are backed or explicitly justified through `Current Literature and Gaps:`.
- Prefer removing or simplifying weakly justified methodological complexity over preserving a clever but unsupported implementation.

## When To Update This Document

Update this file whenever the repository changes its expectations for:

- acceptable source hierarchy for methodological backing,
- citation style or placement requirements,
- gap-analysis requirements for custom methods,
- reviewer expectations for research-grade implementations.
