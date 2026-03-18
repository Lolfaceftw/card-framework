# QMSum Summarizer-Critic Benchmark

This benchmark runs the repository's real summarizer/critic stack against the
official QMSum general-summary test split. It derives `target_seconds` from the
human reference summaries, executes the same A2A-style summarizer and critic
agents used by the runtime, and writes JSON plus Markdown artifacts under
`artifacts/summarizer_critic_benchmark`.

For the package-level benchmark split and shared defaults, see
[`../README.md`](../README.md).

## Entry Points

Both commands below run the same QMSum benchmark CLI:

```powershell
uv run python -m card_framework.benchmark.summarizer_critic --help
uv run python -m card_framework.benchmark.summarizer_critic.qmsum --help
```

## Defaults That Matter

- Provider profiles default to
  `src/card_framework/benchmark/provider_profiles.yaml`.
- The default QMSum dataset manifest path is
  `artifacts/summarizer_critic_benchmark/datasets/qmsum_general_test/manifest.json`.
- The default official QMSum checkout path is
  `artifacts/summarizer_critic_benchmark/datasets/qmsum_official_repo`.
- The tracked `vllm_default` provider profile points at
  `http://127.0.0.1:8000/v1`.
- To override that vLLM endpoint without editing YAML, set one of
  `VLLM_URL`, `CARD_FRAMEWORK_VLLM_URL`, or `VLLM_BASE_URL`.

Example PowerShell override:

```powershell
$env:VLLM_URL = "http://host:8000/v1"
```

## Prepare The Dataset

This clones or reuses the official `Yale-LILY/QMSum` repository and prepares the
reproducible benchmark manifest:

```powershell
uv run python -m card_framework.benchmark.summarizer_critic prepare-dataset
```

Override the repo checkout or prepared-output roots only when you need a custom
location:

```powershell
uv run python -m card_framework.benchmark.summarizer_critic prepare-dataset `
  --qmsum-repo-dir "C:\benchmarks\QMSum" `
  --prepared-root "C:\benchmarks\qmsum_general_test"
```

## Smoke Run

Use one supervised sample first to verify your provider, live streaming, judge
scoring, and telemetry:

```powershell
$env:VLLM_URL = "http://host:8000/v1"
uv run python -m card_framework.benchmark.summarizer_critic supervise `
  --prepare-if-missing `
  --summarizer-provider vllm_default `
  --critic-provider vllm_default `
  --judge-provider vllm_default `
  --max-samples 1 `
  --max-iterations 1 `
  --max-runs 1 `
  --sample-interval-seconds 5
```

## Full Supervised Run

Use the supervisor when you want repeated benchmark runs with RAM/GPU guardrails,
session notes, and child logs:

```powershell
$env:VLLM_URL = "http://host:8000/v1"
uv run python -m card_framework.benchmark.summarizer_critic supervise `
  --prepare-if-missing `
  --summarizer-provider vllm_default `
  --critic-provider vllm_default `
  --judge-provider vllm_default `
  --max-iterations 4 `
  --critic-timeout-seconds 900 `
  --max-runs 1 `
  --sample-interval-seconds 15
```

Default supervisor guardrails:

- total system RAM threshold: `80%`
- per-GPU VRAM threshold: `12 GiB`

## Full Direct Run

Use `execute` when you want one benchmark run without the outer supervisor:

```powershell
$env:VLLM_URL = "http://host:8000/v1"
uv run python -m card_framework.benchmark.summarizer_critic execute `
  --prepare-if-missing `
  --summarizer-provider vllm_default `
  --critic-provider vllm_default `
  --judge-provider vllm_default `
  --max-iterations 4 `
  --critic-timeout-seconds 900
```

Useful execution flags:

- `--max-samples <int>`: `0` means all prepared examples
- `--sample-seed <int>`: deterministic seeded-random sample selection
- `--embedding-mode disabled|enabled`
- `--summarizer-timeout-seconds <float>`
- `--critic-timeout-seconds <float>`
- `--retrieval-timeout-seconds <float>`
- `--index-timeout-seconds <float>`

## Concurrent Direct Runs

When you launch multiple `execute` processes in parallel, give each process its
own internal service ports so the retrieval, summarizer, and critic A2A servers
do not collide:

```powershell
$env:VLLM_URL = "http://host:8000/v1"
uv run python -m card_framework.benchmark.summarizer_critic execute `
  --prepare-if-missing `
  --summarizer-provider vllm_default `
  --critic-provider vllm_default `
  --judge-provider vllm_default `
  --max-iterations 1 `
  --retrieval-port 31000 `
  --summarizer-port 31001 `
  --critic-port 31002
```

Each concurrent run must use a distinct trio of
`--retrieval-port`, `--summarizer-port`, and `--critic-port` values.

## Outputs

Main artifact roots:

- prepared dataset: `artifacts/summarizer_critic_benchmark/datasets/qmsum_general_test`
- direct runs: `artifacts/summarizer_critic_benchmark/runs/<run_id>`
- supervised sessions: `artifacts/summarizer_critic_benchmark/supervisor/<session_id>`

Each direct run writes:

- `report.json`
- `notes.md`
- `results.md`

The report records the active sample-selection mode as either a concrete
`sample_seed` or `manifest_order`.

## Notes

- `judge-provider` defaults to the critic provider when omitted, and the critic
  provider defaults to the summarizer provider when omitted.
- If you keep the tracked `vllm_default` profile on `127.0.0.1:8000/v1`, you do
  not need to set a vLLM URL environment variable at all.
- Use `src/card_framework/benchmark/provider_profiles.yaml` when you want tracked,
  reviewable benchmark provider definitions. Use env overrides only for
  deploy-varying endpoint details.
