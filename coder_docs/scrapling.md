# Scrapling MCP Guide

This repository uses Scrapling MCP as the default toolset for external web retrieval and page extraction.

Use this document as the source of truth for how an agent should choose and operate Scrapling MCP tools in this repo. This is a workflow document, not a general Scrapling tutorial.

Authoritative upstream references:

- Scrapling repository: <https://github.com/D4Vinci/Scrapling/>
- Scrapling MCP guide: <https://scrapling.readthedocs.io/en/latest/ai/mcp-server.html>
- Scrapling fetcher selection guide: <https://scrapling.readthedocs.io/en/latest/fetching/choosing.html>

## Policy

- Use Scrapling MCP for external page retrieval and extraction by default.
- Prefer direct authoritative URLs over generic search-result pages whenever the destination is already known.
- If the user gives a URL or names a known primary source, go straight to Scrapling instead of using another web tool first.
- Use Scrapling to reduce noise before reasoning: prefer `main_content_only=true`, targeted `css_selector`, and the smallest useful extraction scope.
- Use bulk Scrapling tools when multiple URLs can be fetched independently.
- If a URL must be discovered first, minimal non-Scrapling fallback is allowed only for URL discovery or another hard requirement. Once a target URL is known, return to Scrapling immediately for retrieval and extraction.
- Do not weaken security defaults just to force a lower-tier request through. If a lower tier fails, escalate to a browser-backed Scrapling tool before considering any looser network behavior.

## Tool Inventory

Scrapling MCP exposes six web-retrieval tools in this environment.

### `get`

Use for fast HTTP retrieval of simple pages.

- Best for static pages, documentation pages, blog posts, raw files, and pages that render enough content without JavaScript.
- Supports extraction shaping such as `extraction_type`, `css_selector`, and `main_content_only`.
- Supports request tuning such as `impersonate`, `headers`, `cookies`, `params`, `auth`, `http3`, redirect behavior, retry behavior, TLS verification, and proxy settings.

### `bulk_get`

Use when several static pages can be fetched independently.

- Best for multiple documentation pages, article pages, or a small batch of source URLs.
- Prefer this over repeated single `get` calls when there is no dependency between URLs.

### `fetch`

Use for browser-backed retrieval with Playwright.

- Best for JavaScript-rendered pages, SPAs, sites that need waiting, or pages where DOM state matters.
- Supports `network_idle`, `wait_selector`, `wait_selector_state`, `disable_resources`, `headless`, `real_chrome`, `cdp_url`, locale, timezone, and browser headers/cookies.

### `bulk_fetch`

Use when multiple dynamic pages can be fetched in parallel.

- Best for batches of product pages, documentation pages with JS rendering, or multi-page extraction where each page can be loaded independently.

### `stealthy_fetch`

Use for protected or suspicious targets where the regular browser tier is likely to fail.

- Best for Cloudflare, anti-bot systems, browser-fingerprint-sensitive sites, or sites that fail on `get` and `fetch`.
- Supports advanced anti-detection controls such as `solve_cloudflare`, `block_webrtc`, `hide_canvas`, `allow_webgl`, `real_chrome`, and `google_search`.

### `bulk_stealthy_fetch`

Use when multiple protected pages must be fetched in parallel.

- Best for batches of protected product pages or paginated targets behind the same protection layer.

## Selection Ladder

Always choose the cheapest Scrapling tier that is likely to work, then escalate only when needed.

1. Start with `get` or `bulk_get` for simple static content.
2. Escalate to `fetch` or `bulk_fetch` when the page depends on JavaScript, browser state, or explicit waits.
3. Escalate to `stealthy_fetch` or `bulk_stealthy_fetch` when the site is protected, Cloudflare-backed, fingerprint-sensitive, or lower tiers fail.

### Quick Decision Rules

- Use `get` for docs, raw markdown, plain article pages, API references, and other stable text pages.
- Use `fetch` for SPAs, interactive docs, lazy-rendered content, and pages where a selector must appear before extraction.
- Use `stealthy_fetch` when the user mentions protection, you expect Cloudflare, or a normal request/browser attempt fails for anti-bot reasons.
- Use a bulk variant whenever multiple URLs can be processed in parallel without depending on earlier outputs.

## Default Parameter Policy

These defaults should be the starting point unless the target page or task needs something else.

### Extraction defaults

- Prefer `extraction_type="markdown"` for readable, token-efficient page content.
- Use `extraction_type="text"` when only plain text matters and page structure is unnecessary.
- Use `extraction_type="html"` when selectors are not enough and raw markup or attributes matter.
- Prefer `main_content_only=true` unless navigation, headers, sidebars, or footer links are part of the task.
- Add `css_selector` whenever the relevant region is knowable. This is one of Scrapling's biggest advantages and should be used aggressively to reduce noise and token use.

### Browser defaults

- Prefer `headless=true` unless visible browsing is explicitly useful.
- Prefer `disable_resources=true` on `fetch`, `bulk_fetch`, `stealthy_fetch`, and `bulk_stealthy_fetch` unless images, fonts, media, or stylesheets are required for the target content to appear.
- Use `network_idle=true` for SPAs or pages that hydrate after first paint.
- Use `wait_selector` when a specific element signals readiness more reliably than generic network idle.
- Keep timeouts explicit on slow or protected sites instead of relying on defaults.

### Multi-URL defaults

- Prefer `bulk_get` over repeated `get` calls.
- Prefer `bulk_fetch` over repeated `fetch` calls.
- Prefer `bulk_stealthy_fetch` over repeated `stealthy_fetch` calls.
- Keep the batch limited to the URLs that actually matter for the answer.

## Feature-Maximizing Usage

The goal is to use Scrapling's full value, not just treat it as a plain HTTP client.

### Targeted extraction

- Use `css_selector` to isolate the smallest relevant region before passing content onward.
- If the selector is uncertain, use one exploratory fetch to inspect the page structure, then retry with a narrower selector.
- For lists, target the repeated item container instead of scraping the whole page.

### Browser impersonation and request shaping

- Use `impersonate` on `get` and `bulk_get` when a site behaves differently by client fingerprint.
- Leave `stealthy_headers` enabled on static requests unless a task requires exact manual headers.
- Use `http3` only when a target benefits from it and the request path supports it.
- Use `headers`, `cookies`, `params`, and `auth` only when they are necessary for the page or endpoint.
- Use `follow_redirects` and `max_redirects` deliberately when redirect chains matter to the task.
- Use `retries` and `retry_delay` for unstable targets instead of manually repeating the same request pattern.

### Dynamic-content handling

- Use `network_idle=true` for content that loads after scripts run.
- Use `wait_selector` with a specific readiness element for slow or staged pages.
- Use `wait_selector_state="visible"` when the element must be rendered, not just attached.
- Use `wait` when the page needs a short post-load settling period after the main readiness signal.
- Use `real_chrome=true` or `cdp_url` only when browser realism or an existing browser session materially helps.

### Stealth features

- Use `solve_cloudflare=true` when Cloudflare Turnstile or interstitial protection is likely.
- Increase the timeout materially when using the Cloudflare solver.
- Use `block_webrtc=true` when proxy integrity matters.
- Use `hide_canvas=true` when canvas fingerprinting risk is relevant.
- Leave `allow_webgl=true` unless a very specific debugging case requires disabling it.
- Use `google_search=true` only where the default search-style referer behavior is helpful for stealth.

### Locale, geography, and identity

- Use `locale` and `timezone_id` when the site content is region-sensitive or formatting-sensitive.
- Use `proxy` or proxy authentication only when geo-targeting, rate separation, or access constraints require it.
- Prefer explicit locale and timezone when the page content changes by region and the answer depends on that difference.

### Reliability controls

- Use explicit `timeout` values for slow pages, large pages, or protected flows.
- Use retry controls for flaky static endpoints before escalating to a browser tier, but do not keep retrying a clearly wrong tier forever.
- Keep TLS verification enabled by default. If a static path fails in this environment, prefer `fetch` or `stealthy_fetch` before considering looser verification.

## Standard Workflows

### Known authoritative page

1. Fetch the page directly with Scrapling.
2. Use `main_content_only=true`.
3. Add `css_selector` if the answer lives in a specific region.
4. Escalate only if the lower tier misses content.

### Multi-page documentation lookup

1. Identify the exact doc pages or canonical URLs.
2. Use `bulk_get` for plain docs or `bulk_fetch` for JS-heavy docs.
3. Extract markdown or scoped sections only.
4. Refetch narrowly with selectors if the first pass is too noisy.

### Protected site

1. Start with `stealthy_fetch` if protection is expected.
2. Use `solve_cloudflare=true` for likely Cloudflare challenges.
3. Set a longer timeout.
4. Add `wait_selector` for the post-challenge content region.

### URL discovery then retrieval

1. Use minimal non-Scrapling fallback only to discover the destination URL when it is not otherwise knowable.
2. Move back to Scrapling immediately once the target URL is known.
3. Perform the actual page retrieval and extraction with Scrapling.

## Failure Handling

- If `get` fails because the site is dynamic, escalate to `fetch`.
- If `get` or `fetch` fails due to anti-bot or browser-fingerprint issues, escalate to `stealthy_fetch`.
- If a selector returns nothing, fetch the broader content once, inspect structure, then retry with a corrected selector.
- If `network_idle` causes long waits on chatty sites, switch to `wait_selector`.
- If browser-backed tools are slow, keep `disable_resources=true` and narrow extraction with `css_selector`.
- If certificate or TLS issues affect static requests in this environment, prefer browser-backed Scrapling tools before changing verification behavior.

## Output Shaping Expectations

- Prefer returning the smallest relevant extracted content, not a whole-page dump.
- Prefer Markdown for narrative pages and documentation.
- Prefer text for lightweight fact extraction.
- Prefer HTML only when markup structure, links, or attributes are part of the task.
- When comparing multiple pages, keep extraction consistent across pages so downstream reasoning is simpler.

## Do Not Do This

- Do not start with `stealthy_fetch` for every normal documentation page.
- Do not scrape an entire page when a selector can isolate the needed content.
- Do not loop serially over multiple URLs when a bulk tool would do.
- Do not keep using a lower-tier tool after clear evidence that the page needs a browser or stealth.
- Do not use non-Scrapling web tooling for the actual page retrieval when Scrapling can do the job.

## Current Repo Expectation

- Treat this file as the authoritative Scrapling workflow for the repository.
- If Scrapling MCP usage policy changes materially, update this file and `coder_docs/codebase_guide.md` in the same change.
- If `AGENTS.md` and this file drift, reconcile them immediately.
