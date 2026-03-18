# Summarizer-Critic Benchmarks

This package contains the repository's real-dataset benchmark CLIs for the
duration-first summarizer/critic runtime. It currently has two benchmark
tracks with different datasets and operator workflows.

## Which CLI To Use

- `uv run python -m card_framework.benchmark.summarizer_critic`
  Backward-compatible root CLI for the QMSum benchmark path.
- `uv run python -m card_framework.benchmark.summarizer_critic.qmsum`
  Explicit QMSum CLI. This is the same benchmark surface as the package root.
- `uv run python -m card_framework.benchmark.summarizer_critic.sporc`
  SPoRC CLI for dataset preparation, transcript sampling, execute runs,
  duration sweeps, and supervised runs.

## Shared Defaults

- Provider profiles default to
  `src/card_framework/benchmark/provider_profiles.yaml`.
- The tracked `vllm_default` profile points at `http://127.0.0.1:8000/v1`.
- To override that tracked vLLM base URL at runtime without editing YAML, set
  one of `VLLM_URL`, `CARD_FRAMEWORK_VLLM_URL`,
  `CARD_FRAMEWORK_VLLM_BASE_URL`, or `VLLM_BASE_URL`.
- `--sample-seed <int>` enables deterministic seeded-random sample selection.
  Omitting the flag preserves manifest order.
- `--max-samples 0` means "all prepared samples" for both benchmark families.

## Generated Output Roots

Generated benchmark artifacts default to the gitignored
`artifacts/summarizer_critic_benchmark` tree.

Common subtrees:

- `datasets/qmsum_general_test`: prepared QMSum manifest and transcript files
- `runs`: direct QMSum benchmark runs
- `supervisor`: QMSum supervised sessions
- `datasets/sporc/prepared_sample`: prepared SPoRC sample split
- `datasets/sporc/prepared_full`: prepared SPoRC full split and curated manifests
- `sporc_runs`: direct SPoRC execute runs
- `sporc_supervisor`: SPoRC supervised sessions
- `sweeps`: SPoRC duration-sweep aggregates
- `sporc_human_sampling`: human transcript-sampling sessions
- `sporc_llm_sampling`: Anthropic-backed transcript-sampling runs

## Detailed Guides

- [`qmsum/README.md`](./qmsum/README.md): QMSum prepare, execute, supervise,
  concurrency, and output details
- [`sporc/README.md`](./sporc/README.md): SPoRC preparation, sampling,
  transcript export, execute runs, sweeps, and troubleshooting
