<?xml version="1.0" encoding="UTF-8"?>
<repoEngineeringStandards repo="qa-eval" language="Python" pythonVersion="3.12+">
  <overview>
    <purpose>Define the engineering standards for contributors working in this repository.</purpose>
    <principles>
      <principle>Prefer explicit, maintainable Python over clever shortcuts.</principle>
      <principle>Assume configuration exists; do not hardcode deploy-varying values.</principle>
      <principle>Prefer realistic pipeline confidence over toy examples.</principle>
      <principle>Write for the next developer who must read, debug, and extend the code.</principle>
    </principles>
  </overview>

  <section id="python-google-docstrings" title="Python Google Docstrings">
    <rule>Use triple double quotes for docstrings.</rule>
    <rule>Require docstrings on public modules, public classes, and public functions or methods.</rule>
    <rule>Start with a short imperative summary line ending with punctuation.</rule>
    <rule>For multi-line docstrings, use a summary line, one blank line, then details.</rule>
    <rule>Use Google-style sections when needed: Args, Returns, Raises, Yields, and Attributes.</rule>
    <rule>Do not restate the function signature inside the docstring.</rule>
    <rule>Describe behavior, constraints, side effects, and failure modes rather than obvious mechanics.</rule>
    <rule>Do not duplicate type information in docstrings when type hints already make it clear.</rule>
  </section>

  <section id="python-type-hinting" title="Python Type Hinting">
    <rule>Use Python 3.12 typing syntax such as list[str], dict[str, int], A | B, and type aliases.</rule>
    <rule>Annotate all public function and method boundaries, including explicit return types.</rule>
    <rule>Use -&gt; None explicitly for procedures that do not return a value.</rule>
    <rule>Keep Any as a narrow escape hatch, not a default typing strategy.</rule>
    <rule>Prefer abstract collection interfaces in public APIs and concrete mutable types only where mutation is required.</rule>
    <rule>Narrow unions before use with explicit checks rather than assuming a variant.</rule>
    <rule>Use scoped type ignore comments only when unavoidable, and explain them briefly.</rule>
    <rule>Do not leave public code untyped when the contract matters.</rule>
  </section>

  <section id="python-design-patterns" title="Python Design Patterns">
    <rule>Prefer composition over inheritance by default.</rule>
    <rule>Use callable injection, protocols, or small interfaces before introducing class-heavy factories or strategies.</rule>
    <rule>Place adapters at system boundaries where external interfaces do not match internal expectations.</rule>
    <rule>Use decorators or wrappers for orthogonal concerns such as logging, retries, caching, metrics, or auth.</rule>
    <rule>Avoid singleton-style hidden global state; favor explicit wiring and controlled lifecycle management.</rule>
    <rule>Only introduce a pattern when it solves a real duplication, change, or testability problem.</rule>
    <rule>Keep patterns reversible; if the abstraction stops paying for itself, simplify it.</rule>
  </section>

  <section id="python-solid-kiss-yagni-dry" title="Python SOLID, KISS, YAGNI, and DRY">
    <rule>Optimize for clarity first. Straightforward code beats clever indirection.</rule>
    <rule>Split code by reason to change, not by arbitrary size or method count.</rule>
    <rule>Use abstractions only at real volatility seams or true extension points.</rule>
    <rule>Keep interfaces small and behavior-focused.</rule>
    <rule>DRY duplicated business knowledge, validation rules, and policy logic before DRYing trivial syntax.</rule>
    <rule>Do not add hooks, flags, or frameworks for hypothetical future needs.</rule>
    <rule>Prefer simple working code first, then refactor once repetition or variation is proven.</rule>
    <rule>When principles conflict, choose the option that preserves readability and change safety.</rule>
  </section>

  <section id="python-donts" title="Python Don&apos;ts">
    <rule>Do not use bare except or broad exception handling that hides unrelated failures.</rule>
    <rule>Do not translate exceptions without preserving the original context with exception chaining.</rule>
    <rule>Do not use mutable default arguments.</rule>
    <rule>Do not use eval or exec on external or dynamic input.</rule>
    <rule>Do not pass untrusted input to subprocess with shell=True.</rule>
    <rule>Do not use print for operational diagnostics in application or library code; use logging.</rule>
    <rule>Do not use wildcard imports.</rule>
    <rule>Do not hardcode deploy-varying constants, URLs, tokens, credentials, or environment-specific values; assume configuration exists.</rule>
    <rule>Do not write comments that feel like you are talking to yourself. Write comments for a developer reader, and explain why rather than narrating obvious code.</rule>
    <rule>Do not leave files, resources, or external handles open when a context manager is available.</rule>
  </section>

  <section id="python-cybersecurity-practices" title="Python Cybersecurity Practices">
    <rule>Treat every external input as untrusted and validate it with allowlists and explicit constraints.</rule>
    <rule>Use subprocess argument arrays instead of interpolated shell commands.</rule>
    <rule>Set timeouts on network calls and keep TLS verification enabled.</rule>
    <rule>Never deserialize untrusted data with unsafe mechanisms such as pickle or unsafe YAML loaders.</rule>
    <rule>Store secrets in configuration or secret managers, never in code or logs.</rule>
    <rule>Use the secrets module for security-sensitive token generation and constant-time comparison for sensitive values.</rule>
    <rule>Resolve and constrain file paths to approved base directories before reading, writing, or extracting files.</rule>
    <rule>Use secure temporary files and directories instead of predictable temp paths.</rule>
    <rule>Never log secrets, tokens, raw credentials, or sensitive payloads.</rule>
    <rule>Pin and audit dependencies, and make security checks part of CI.</rule>
    <rule>Apply least privilege to runtime identity, filesystem access, network reach, and credentials.</rule>
    <rule>Prefer secure defaults over opt-out security toggles in production paths.</rule>
  </section>

  <section id="git-and-github-workflow" title="Git and GitHub Workflow">
    <rule>Use a pull-request-first workflow for integration branches. Do not push directly to shared protected branches.</rule>
    <rule>Protect integration branches with rulesets or branch protection, required reviews, and required status checks.</rule>
    <rule>Require CODEOWNERS review on sensitive paths such as workflows, deployment code, security-critical modules, and the CODEOWNERS file itself when present.</rule>
    <rule>Treat CI checks as merge gates, not advisory signals.</rule>
    <rule>Create a dedicated semver-named release-preparation branch such as `release/vX.Y.Z` before attempting any PyPI publish, and push the release tag only after that branch has been reviewed and merged.</rule>
    <rule>After pushing a release tag or other workflow-sensitive change, verify the resulting GitHub Actions run to completion instead of assuming the push succeeded end to end.</rule>
    <rule>Enable secret scanning and push protection, and rotate exposed credentials immediately if a secret is detected.</rule>
    <rule>Never stage, commit, or open a pull request with files that contain API keys, access tokens, secrets, private service URLs such as vLLM endpoints, or other private information.</rule>
    <rule>Sign commits and release tags where repository policy supports it.</rule>
    <rule>Use force push only on your own topic branches when necessary, and prefer force-with-lease over force.</rule>
    <rule>Review dependency and workflow changes with extra scrutiny before merging.</rule>
    <rule>Prefer git revert on shared branches rather than rewriting published history.</rule>
    <rule>Preserve traceability from commit to release artifact wherever possible.</rule>
    <rule>If a tagged release workflow fails, fix forward with a new version and tag instead of reusing the failed release version ambiguously.</rule>
  </section>

  <section id="logging-industry-standard-practices" title="Logging Industry Standard Practices">
    <rule>Use structured logs by default, preferably JSON or consistent key-value output.</rule>
    <rule>Standardize a minimum event schema with timestamp, severity, message, service, environment, logger, event name, outcome, and request or correlation identifier.</rule>
    <rule>Use RFC 3339 or ISO 8601 timestamps with timezone information, preferably UTC.</rule>
    <rule>Use severity levels consistently so logs support alerting and triage.</rule>
    <rule>Attach request, trace, span, or correlation context to relevant events.</rule>
    <rule>Log exceptions once at the handling boundary with enough context to diagnose the failure.</rule>
    <rule>Redact or exclude secrets, tokens, credentials, and other sensitive data from logs.</rule>
    <rule>Use parameterized logging rather than eager string interpolation.</rule>
    <rule>Configure logging centrally and use module-level loggers instead of ad hoc root logger usage.</rule>
    <rule>Use non-blocking logging pipelines for high-throughput services when operationally necessary.</rule>
    <rule>Protect log integrity, retention, and access as part of the security posture.</rule>
    <rule>Do not produce noisy narration logs. Log decisions, state transitions, failures, and useful diagnostics.</rule>
  </section>

  <section id="python-testing-standards" title="Python Testing Standards">
    <rule>Prefer realistic tests that reflect the real pipeline rather than toy tests detached from production behavior.</rule>
    <rule>Keep most test value at the integration level, with unit tests for focused logic and fewer end-to-end tests.</rule>
    <rule>Test behavior and contracts rather than private implementation details.</rule>
    <rule>When an implementation change affects behavior, interfaces, contracts, or outputs that tests should cover, update or add the relevant tests in the same change.</rule>
    <rule>Keep tests deterministic by controlling time, randomness, environment, and side effects.</rule>
    <rule>Use fixtures as explicit setup with minimal scope and reliable teardown.</rule>
    <rule>Mock only true external boundaries such as networks, cloud services, or operating-system edges.</rule>
    <rule>Patch where the object is looked up, and prefer autospec-style mocks when mocking is necessary.</rule>
    <rule>Use realistic schemas, payloads, and pipeline inputs instead of trivial placeholder data.</rule>
    <rule>Add a regression test for every bug fix at the lowest level that reproduces the real failure path.</rule>
    <rule>Enforce strict pytest configuration, explicit markers, and order-independent tests.</rule>
    <rule>Treat flaky tests as defects. Fix them, quarantine them with ownership and a deadline, or remove redundant ones.</rule>
    <rule>Design CI for fast, reliable feedback: quick deterministic checks first, broader suites afterward.</rule>
  </section>

  <section id="project-specific-guidelines" title="Project Specific Guidelines">
    <rule>Treat @coder_docs/* as the authoritative project-local guidance set for this repository.</rule>
    <rule>Read coder_docs/codebase_guide.md first at the start of every new coding-agent session before making assumptions about architecture, flow, or file ownership.</rule>
    <rule>Use coder_docs/memory/errors_and_notes.md as the running repository memory for captured errors, regressions, and notable pitfalls that future sessions should avoid repeating.</rule>
    <rule>When you encounter a meaningful error or fix a mistake worth preserving, prepend a brief entry to the top of coder_docs/memory/errors_and_notes.md with the date and time, a Problem field, and the Solution that resolved it.</rule>
    <rule>Use coder_docs/academic_standards.md as the source of truth for formulas, algorithms, heuristics, scoring rules, or other robustness-sensitive implementations that require academic backing.</rule>
    <rule>For methodology-sensitive implementations, search academic journals first, avoid unsupported magic formulas, and add inline IEEE-style citation comments at the implementation site with a fuller same-file reference whenever practical.</rule>
    <rule>Use coder_docs/git_github_workflow.md as the source of truth for local Git staging, commit construction, push safety, and GitHub pull request workflow in this repository.</rule>
    <rule>Use coder_docs/github_actions_release_spec.md as the source of truth for GitHub Actions release workflow behavior, build standards, and post-tag verification in this repository.</rule>
    <rule>Use coder_docs/ruff.md as the source of truth for linting workflow and coder_docs/uv_package_manager.md as the source of truth for dependency and environment management.</rule>
    <rule>Use uv as the package manager for this repository and use ruff as the linter for this repository.</rule>
    <rule>When implementation changes affect tested behavior, contracts, or outputs, update the relevant tests in the same change before considering the work complete.</rule>
    <rule>When code changes affect behavior, interfaces, setup, outputs, workflows, or operator expectations, update the relevant Markdown documentation in the same change, including files such as README.md, docs/*.md, and applicable coder_docs/*.md.</rule>
    <rule>Treat Markdown documentation review as part of every implementation, review, and verification pass so documentation updates are considered before concluding the task.</rule>
    <rule>When code, configuration, prompts, runtime flow, tooling, or developer workflow changes materially, update coder_docs/codebase_guide.md in the same change so it stays current.</rule>
    <rule>When GitHub Actions release behavior, release verification, or build standards change materially, update coder_docs/github_actions_release_spec.md, coder_docs/git_github_workflow.md, and coder_docs/codebase_guide.md in the same change.</rule>
    <rule>When linting, dependency-management, Git and GitHub workflow, or academic-methodology policy changes materially, update the corresponding file in coder_docs in the same change.</rule>
    <rule>Keep coder_docs content specific to this repository's actual behavior, current commands, and constraints. Do not let it drift into generic Python advice.</rule>
    <rule>If AGENTS.md conflicts with a repo-specific operational detail documented in coder_docs, reconcile the documents immediately rather than ignoring the mismatch.</rule>
  </section>
  <section id="scrapling-mcp-web-research" title="Scrapling MCP Web Research">
    <rule>Use coder_docs/scrapling.md as the source of truth for external web retrieval and page extraction workflow.</rule>
    <rule>Use Scrapling MCP as the required tool for external web search, page retrieval, and browsing in this repository whenever open-web information is needed.</rule>
    <rule>Treat generic non-Scrapling web-search and browsing tools as disallowed when Scrapling MCP can satisfy the task.</rule>
    <rule>If the user already provided a URL or the authoritative source URL is known, go directly to Scrapling MCP instead of another web tool.</rule>
    <rule>Prefer the cheapest Scrapling tier that fits the target: get or bulk_get for simple static pages, fetch or bulk_fetch for JavaScript or explicit waits, stealthy_fetch or bulk_stealthy_fetch for Cloudflare, anti-bot protections, or lower-tier failures.</rule>
    <rule>Use bulk Scrapling tools for independent multi-URL retrieval instead of serial single-page requests.</rule>
    <rule>Use Scrapling features aggressively to reduce noise and token use: main_content_only, css_selector, extraction_type selection, disable_resources where safe, network_idle, and wait_selector.</rule>
    <rule>Prefer authoritative direct pages over search-result pages, and prefer targeted page extraction over whole-page dumps.</rule>
    <rule>If a URL must be discovered first or a hard requirement cannot be satisfied by Scrapling alone, use only the smallest possible non-Scrapling fallback for that discovery step, then return to Scrapling immediately for actual retrieval and extraction.</rule>
    <rule>Do not keep using non-Scrapling web tools for discovery, page retrieval, or ongoing browsing after the target URL is known and Scrapling MCP can complete the task.</rule>
  </section>
</repoEngineeringStandards>
