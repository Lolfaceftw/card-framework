# Summarizer-Critic Benchmarks

These benchmarks run CARD's summarizer and critic loop on real datasets and
check how well each summary hits its target duration. There are two:

| Dataset | Content | Guide |
| --- | --- | --- |
| QMSum | Meeting transcripts with human reference summaries | [qmsum/README.md](qmsum/README.md) |
| SPoRC | Long multi-speaker podcast episodes | [sporc/README.md](sporc/README.md) |

## Which command to use

| Command | Runs |
| --- | --- |
| `uv run python -m card_framework.benchmark.summarizer_critic.qmsum` | QMSum |
| `uv run python -m card_framework.benchmark.summarizer_critic` | QMSum (older alias for the line above) |
| `uv run python -m card_framework.benchmark.summarizer_critic.sporc` | SPoRC |

## Shared defaults

- Provider profiles come from `src/card_framework/benchmark/provider_profiles.yaml`.
- The `vllm_default` profile points at `http://127.0.0.1:8000/v1`. To change it
  without editing YAML, set `VLLM_URL`. `CARD_FRAMEWORK_VLLM_URL`,
  `CARD_FRAMEWORK_VLLM_BASE_URL`, and `VLLM_BASE_URL` also work.
- `--max-samples 0` runs every prepared sample.
- `--sample-seed <n>` picks a reproducible random subset. Without it, samples
  run in manifest order.

## Where outputs go

Everything is written under `artifacts/summarizer_critic_benchmark/`, which git
ignores:

| Folder | Contents |
| --- | --- |
| `datasets/qmsum_general_test` | Prepared QMSum manifest and transcripts |
| `datasets/sporc/prepared_sample` | Prepared SPoRC sample split |
| `datasets/sporc/prepared_full` | Prepared SPoRC full split and curated manifests |
| `runs` | QMSum runs |
| `supervisor` | QMSum supervised sessions |
| `sporc_runs` | SPoRC runs |
| `sporc_supervisor` | SPoRC supervised sessions |
| `sweeps` | SPoRC duration sweeps |
| `sporc_human_sampling` | Human transcript-sampling sessions |
| `sporc_llm_sampling` | LLM transcript-sampling runs |
| `embedding_cache` | Transcript embeddings, reused across runs when the sample, transcript, and embedding model match |
