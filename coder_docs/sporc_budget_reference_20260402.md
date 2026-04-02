# SPoRC Benchmark Budget Reference (2026-04-02)

This note records the under-$50 benchmark plan discussed on 2026-04-02.

## Recommended study design

- Dataset: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_max3_manifest.json`
- Sample selection: first 10 samples from deterministic seeded ranking with `sample_seed=42`, using `sha256(f"42:{sample_id}")`
- Preset: `5m` only
- Max iterations: `1`
- Contestants:
  - `gpt-5.4-mini` via OpenAI
  - `claude-haiku-4.5` via Anthropic
  - `deepseek-chat` = DeepSeek V3.2 non-thinking mode
  - one local open/open-weight model, preferably `Qwen3-32B`, `Qwen3-14B`, or `Llama 4 Scout`
- Judge: `gemini-3.1-pro-preview`

## Repo provider-profile ids for this plan

Use these SPoRC benchmark provider ids from
`src/card_framework/benchmark/provider_profiles.yaml`:

- OpenAI contestant: `openai_gpt_5_4_mini`
- Anthropic contestant: `anthropic_claude_haiku_4_5`
- DeepSeek contestant: `deepseek_chat`
- Local open/open-weight contestant: `vllm_default`
- Google judge: `google_gemini_3_1_pro_preview_judge`

For the local `vllm_default` contestant, also set `CARD_FRAMEWORK_VLLM_MODEL`
or `VLLM_MODEL` to the exact served model id you want benchmarked. Without
that pin, the local vLLM adapter falls back to the first model exposed by the
server's `/models` endpoint.

Budget-safety defaults in the benchmark path:

- Gemini `gemini-3.1-pro-preview` judge keeps thinking enabled.
- Other budget-plan contestants default to thinking disabled; custom Gemini 3 Pro contestant profiles may still fall back to provider defaults because Gemini does not support thinking-off there.

Why this design:

- It still covers both proprietary and open/open-weight models.
- It keeps the API bill near the undergraduate-project budget target.
- It avoids benchmarking Gemini as both contestant and judge.

## Estimated API spend from the local simulation

These figures come from the actual local prompt and tool-loop simulation described below. They are estimated token-cost totals, not provider-billed ground truth.

| Provider | Purpose | Expected spend (USD) |
| --- | --- | ---: |
| OpenAI | `gpt-5.4-mini` contestant | 12.1376 |
| Anthropic | `claude-haiku-4.5` contestant | 16.1352 |
| DeepSeek | `deepseek-chat` contestant | 4.4707 |
| Google | `gemini-3.1-pro-preview` judge across all 4 contestants | 4.8766 |
| **Total** |  | **37.6201** |

The local open/open-weight contestant contributes no API spend in this plan, but it still incurs Google judge cost because it is judged like the other contestants.

## Suggested wallet top-up

Recommended under-$50 split:

- OpenAI: `$14`
- Anthropic: `$20`
- DeepSeek: `$6`
- Google: `$8`
- Total: `$48`

This split intentionally leaves more headroom on Anthropic and Google, because repo-default provider settings can enable thinking or reasoning there. It is still only a practical buffer, not a guarantee against higher billed token usage.

## Methodology used to estimate cost

### 1. Real dataset slice

The estimate used the first 10 seed-42 samples from the curated long-form SPoRC max-3-speaker manifest. Their transcript word counts were:

`[17203, 19736, 20635, 18742, 17402, 17610, 28460, 17603, 31820, 18092]`

Summary statistics:

- mean: `20,730.3`
- median: `18,417`
- min: `17,203`
- max: `31,820`

### 2. Real prompt templates

The estimate rendered the repo's actual templates:

- `src/card_framework/prompts/templates/summarizer_system.jinja2`
- `src/card_framework/prompts/templates/summarizer_user.jinja2`
- `src/card_framework/prompts/templates/critic_system.jinja2`
- `src/card_framework/prompts/templates/critic_user.jinja2`
- `src/card_framework/prompts/templates/conversation_summary_judge_system.jinja2`
- `src/card_framework/prompts/templates/conversation_summary_judge_user.jinja2`

### 3. Tool-loop simulation

For each selected sample, the estimate simulated the actual transcript-only summarizer and critic structure used by the repo:

- summarizer target: `5m`
- synthetic final draft target: `750` words across `30` speaker lines
- summarizer loop includes:
  - `add_speaker_message(...)`
  - auto `estimate_duration`
  - auto `count_words`
  - auto `save_draft` once in budget
  - final `finalize_draft()`
- critic loop for `max_iterations=1` includes:
  - `run_deterministic_checks(...)`
  - `submit_verdict(...)`
- judge loop includes one rendered judge request against the full transcript and final summary XML

### 4. Token estimation rule

Requests and assistant/tool payloads were serialized to compact JSON and approximated with:

`estimated_tokens ~= len(serialized_payload) / 4`

This was applied consistently to:

- full prompt requests
- tool schemas
- assistant tool-call payloads
- tool-result payloads
- judge requests

### 5. Aggregated token totals over 10 samples

Per contestant model, across all 10 samples:

- summarizer+critic input tokens: `15,894,397.5`
- summarizer+critic output tokens: `48,167.5`

Per contestant evaluation, across all 10 samples:

- judge input tokens: `608,753.0`
- judge output tokens: `137.5`

Because there are four contestants in the recommended design, Google judge spend is effectively four copies of the per-contestant judge total.

### 6. Billing formula

For each provider:

`cost = (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate`

Google judge cost was added once per contestant, including the local open/open-weight model.

## Pricing sources used on 2026-04-02

- OpenAI pricing: `https://openai.com/api/pricing/`
  - `GPT-5.4 mini`: `$0.750 / 1M input`, `$4.500 / 1M output`
- Anthropic pricing: `https://platform.claude.com/docs/en/about-claude/pricing`
  - `Claude Haiku 4.5`: `$1 / MTok input`, `$5 / MTok output`
- Google pricing: `https://ai.google.dev/gemini-api/docs/pricing`
  - `gemini-3.1-pro-preview`: `$2.00 / 1M input` and `$12.00 / 1M output` at `<=200k` prompt size
- DeepSeek pricing: `https://api-docs.deepseek.com/quick_start/pricing`
  - `deepseek-chat` / DeepSeek V3.2: `$0.28 / 1M input (cache miss)`, `$0.42 / 1M output`

## Important caveats

1. These numbers are best treated as a **visible-token lower-bound estimate**.
2. The repo defaults enable thinking/reasoning in several providers, including Anthropic and Google, so real billed usage can exceed this estimate.
3. This plan assumes the open/open-weight contestant runs locally. If it is moved to a paid API, the budget changes materially.
4. This reference is for the `5m`-only, `max_iterations=1`, `10 samples` plan. It does not apply to the earlier multi-preset or multi-iteration scenarios.
