# Configuration Guide

`conf/config.yaml` is the canonical runtime config for source checkouts.
`src/card_framework/config/config.yaml` is the bundled fallback used by packaged
installs when the repo-root config is not available.

## Quick Start

1. Copy `.env.example` to `.env`.
2. Fill only the provider secrets you actually use.
3. Edit the `selection` block in `conf/config.yaml`.
4. Leave the concrete provider definitions in `profiles` alone unless you are
   intentionally changing a model, base URL, or local embedding choice.

## What To Edit

Most day-to-day changes happen in this block:

```yaml
selection:
  llm: "vllm_local"
  embedding: "disabled"
  stage_llm:
    summarizer: "default"
    critic: "default"
    interjector: "default"
```

- `selection.llm`: chooses the shared LLM profile used by the runtime.
- `selection.embedding`: chooses the retrieval embedding profile.
- `selection.stage_llm.*`: optionally overrides one stage with another LLM
  profile. Use `default` to inherit the shared LLM.

## Built-In LLM Profiles

- `vllm_local`: local OpenAI-compatible vLLM endpoint.
- `transformers_local`: local Transformers provider.
- `deepseek_chat`: DeepSeek chat model.
- `deepseek_reasoner`: DeepSeek reasoning model.
- `google_gemini`: Gemini provider.
- `glm_46`: GLM provider.
- `huggingface_toolace`: Hugging Face ToolACE through the router endpoint.
- `nanbeige_thinking`: Nanbeige through the Hugging Face router endpoint.

## Built-In Embedding Profiles

- `disabled`: disables retrieval embeddings and uses the no-op provider.
- `qwen3_4b`: Qwen embedding model on CUDA.
- `minilm_cpu`: lightweight CPU embedding fallback.

## `.env` Variables

Use canonical variable names in `.env`:

```dotenv
CARD_FRAMEWORK_VLLM_API_KEY=EMPTY
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
ZAI_API_KEY=
HF_TOKEN=
```

The runtime also accepts older aliases and mirrors them automatically. That
means existing values such as `VLLM_API_KEY`, `GOOGLE_API_KEY`, `GLM_API_KEY`,
`HF_ACCESS_TOKEN`, and `HF_API_TOKEN` still work.

## Common Examples

Use local vLLM with no retrieval:

```yaml
selection:
  llm: "vllm_local"
  embedding: "disabled"
  stage_llm:
    summarizer: "default"
    critic: "default"
    interjector: "default"
```

Use DeepSeek for the critic only:

```yaml
selection:
  llm: "vllm_local"
  embedding: "disabled"
  stage_llm:
    summarizer: "default"
    critic: "deepseek_reasoner"
    interjector: "default"
```

Enable retrieval with the Qwen embedding model:

```yaml
selection:
  llm: "vllm_local"
  embedding: "qwen3_4b"
  stage_llm:
    summarizer: "default"
    critic: "default"
    interjector: "default"
```

## QA Benchmark Config

`src/card_framework/benchmark/qa_config.yaml` now resolves its `vllm.api_key`
from `.env` too. Keep the benchmark base URL and timeout in YAML, but keep the
actual key in `.env`.

## Notes

- Do not commit `.env`.
- Keep secrets out of CLI overrides when possible.
- If you need a packaged install to use a custom full config file, point
  `CARD_FRAMEWORK_CONFIG` at that YAML and keep the secrets in environment
  variables.
