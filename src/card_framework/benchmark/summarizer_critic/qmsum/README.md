# Summarizer-Critic Benchmark

This package runs the duration-first real-dataset benchmark for the repository's
multi-agent summarizer-critic workflow.

It uses the official QMSum test general-summary split, derives `target_seconds`
 from the human references, runs the real summarizer/critic A2A stack, and
 writes JSON plus Markdown artifacts under
 `artifacts/summarizer_critic_benchmark`.

## Prerequisites

- Run commands from the repository root.
- Use `uv` for Python execution.
- Set `CARD_FRAMEWORK_VLLM_BASE_URL` in the shell instead of hardcoding the
  endpoint into files or command history snippets.
- The packaged provider profiles for this benchmark live in
  `artifacts/runtime/provider_profiles_env_vllm.yaml`.

## Prepare The Dataset

PowerShell:

```powershell
uv run python -m card_framework.benchmark.summarizer_critic prepare-dataset --qmsum-repo-dir "artifacts/external/QMSum" --prepared-root "artifacts/summarizer_critic_benchmark/datasets/qmsum_general_test"
```

This prepares a reproducible manifest at:

- `artifacts/summarizer_critic_benchmark/datasets/qmsum_general_test/manifest.json`

## Smoke Run

Use this first to verify live agent output, scoring, and telemetry on a single
example.

```powershell
$env:CARD_FRAMEWORK_VLLM_BASE_URL='<secret-vllm-base-url>'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'; uv run python -m card_framework.benchmark.summarizer_critic supervise --prepare-if-missing --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/qmsum_general_test/manifest.json" --qmsum-repo-dir "artifacts/external/QMSum" --provider-profiles "artifacts/runtime/provider_profiles_env_vllm.yaml" --summarizer-provider vllm_secret_agent --critic-provider vllm_secret_agent --judge-provider vllm_secret_judge --max-samples 1 --max-iterations 1 --max-runs 1 --sample-interval-seconds 5
```

## Full Supervised Run

Use this for the full dataset with telemetry guardrails, live output, and
 session notes/results.

```powershell
$env:CARD_FRAMEWORK_VLLM_BASE_URL='<secret-vllm-base-url>'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'; uv run python -m card_framework.benchmark.summarizer_critic supervise --prepare-if-missing --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/qmsum_general_test/manifest.json" --qmsum-repo-dir "artifacts/external/QMSum" --provider-profiles "artifacts/runtime/provider_profiles_env_vllm.yaml" --summarizer-provider vllm_secret_agent --critic-provider vllm_secret_agent --judge-provider vllm_secret_judge --max-iterations 4 --critic-timeout-seconds 900 --max-runs 1 --sample-interval-seconds 15
```

Default supervisor guardrails:

- kill the child benchmark if total system RAM reaches `80%`
- kill the child benchmark if any GPU reaches `12 GiB` used VRAM

Live terminal output includes:

- summarizer / critic streamed content when the provider emits it
- benchmark sample progress
- remaining examples
- rolling average examples per minute
- ETA
- supervisor RAM / GPU heartbeat lines

## Full Direct Run

Use this if you want the benchmark without the outer supervisor wrapper.

```powershell
$env:CARD_FRAMEWORK_VLLM_BASE_URL='<secret-vllm-base-url>'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'; uv run python -m card_framework.benchmark.summarizer_critic execute --prepare-if-missing --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/qmsum_general_test/manifest.json" --qmsum-repo-dir "artifacts/external/QMSum" --provider-profiles "artifacts/runtime/provider_profiles_env_vllm.yaml" --summarizer-provider vllm_secret_agent --critic-provider vllm_secret_agent --judge-provider vllm_secret_judge --max-iterations 4 --critic-timeout-seconds 900
```

## Concurrent Direct Runs

When you launch multiple `execute` processes in parallel, assign explicit internal
service ports per run so the retrieval, summarizer, and critic A2A servers do not
race on ephemeral port selection.

```powershell
$env:CARD_FRAMEWORK_VLLM_BASE_URL='<secret-vllm-base-url>'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING='1'; uv run python -m card_framework.benchmark.summarizer_critic execute --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/qmsum_general_test/manifest.json" --qmsum-repo-dir "artifacts/external/QMSum" --provider-profiles "artifacts/runtime/provider_profiles_env_vllm.yaml" --summarizer-provider vllm_secret_agent --critic-provider vllm_secret_agent --judge-provider vllm_secret_judge --max-iterations 1 --retrieval-port 31000 --summarizer-port 31001 --critic-port 31002
```

Each concurrent run must use a distinct trio of `--retrieval-port`,
`--summarizer-port`, and `--critic-port` values.

Use `--sample-seed <int>` when you want deterministic seeded-random sample
selection instead of the default manifest-order prefix. Omitting the flag keeps
the existing manifest order.

## Main Outputs

- run artifacts: `artifacts/summarizer_critic_benchmark/runs/<run_id>`
- supervisor session artifacts: `artifacts/summarizer_critic_benchmark/supervisor/<session_id>`
- per-run machine-readable report: `report.json`
- per-run notes: `notes.md`
- per-run human-readable results: `results.md` with native Markdown tables plus 95% CI values for aggregate metric means
- per-run notes now record the active sample-selection mode as either a concrete `sample_seed` or `manifest_order`

## Notes

- `vllm_secret_agent` is the main summarizer/critic profile.
- `vllm_secret_judge` is the judge profile with the explicit thinking override
  needed for judge-style structured outputs.
- The benchmark scores duration control, factualness, naturalness, and speaker
  attribution, then computes a composite score.
