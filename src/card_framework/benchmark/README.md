# Benchmark Runner

The benchmark source, manifests, rubrics, and QA settings now live under
`src/card_framework/benchmark`. The CLIs resolve those packaged resources by
default, so explicit path flags are usually optional.

## Summarizer-Critic Real Dataset Benchmark

```bash
uv run python -m card_framework.benchmark.summarizer_critic prepare-dataset
uv run python -m card_framework.benchmark.summarizer_critic execute --prepare-if-missing --summarizer-provider vllm_default
uv run python -m card_framework.benchmark.summarizer_critic supervise --prepare-if-missing --max-runs 3
```
This package lives under `src/card_framework/benchmark/summarizer_critic` and is
separate from the older matrix benchmark. It:
- fetches or reuses the official `Yale-LILY/QMSum` repository
- prepares the full `data/ALL/test` general-summary slice into a reproducible manifest
- derives `target_seconds` from the human reference summary length
- runs the actual summarizer, critic, retrieval, and orchestrator stack with the duration-first contracts
- scores duration control, factualness, naturalness, and speaker attribution
- writes machine-readable reports plus `notes.md` and `results.md` under `artifacts/summarizer_critic_benchmark`

See `src/card_framework/benchmark/summarizer_critic/README.md` for the full
PowerShell commands, provider-profile IDs, smoke-run recipe, and full supervised
run instructions.

## Summarization Benchmark

```bash
uv run python -m card_framework.benchmark.run execute --preset hourly
```

Common options:
- `--preset smoke|hourly|full`
- `--manifest src/card_framework/benchmark/manifests/benchmark_v1.json`
- `--provider-profiles src/card_framework/benchmark/provider_profiles.yaml`
- `--providers <comma-separated ids>`
- `--judge-provider <provider id>`
- `--judge-rubric src/card_framework/benchmark/rubrics/default_summarization_rubric.json`
- `--judge-repeats <int>`
- `--disable-order-swap`
- `--alignscore-model <model name>`
- `--output-dir artifacts/benchmark`

Outputs:
- `artifacts/benchmark/<run_id>/benchmark_report.json`
- `artifacts/benchmark/<run_id>/verification.json`
- `artifacts/quality/<run_id>/report.json`

If every benchmark cell is skipped before any sample runs, the CLI exits
non-zero and keeps the generated artifacts so the failure can be diagnosed.

Prepare a manifest:

```bash
uv run python -m card_framework.benchmark.run prepare-manifest --sources local
```

For the local source, the CLI now auto-discovers a reusable transcript in this
order unless you pass `--local-transcript-path` explicitly:
- repo-root `transcript.json`
- repo-root `*.transcript.json`
- `artifacts/transcripts/*.transcript.json`

## Watchdog Runner

Use the watchdog when you need repeated summarization benchmark runs with periodic RAM and GPU telemetry plus automatic threshold kills.

```bash
uv run python -m card_framework.benchmark.watchdog --max-runs 3 --sample-interval-seconds 5
```

Defaults:
- loops the summarization smoke benchmark (`benchmark.run execute --preset smoke`)
- kills the child run at `80%` total system RAM or `12 GiB` used GPU VRAM
- writes markdown notes, markdown results, per-run stdout/stderr logs, and telemetry under `artifacts/benchmark_watchdog`
- uses the repo-local `.uv-cache` directory for child runs

You can replace the child command after `--`.

```bash
uv run python -m card_framework.benchmark.watchdog -- python -m card_framework.benchmark.qa --summary-xml summary.xml --source-transcript transcript.json
```

## Diarization Benchmark

```bash
uv run python -m card_framework.benchmark.diarization prepare-manifest
uv run python -m card_framework.benchmark.diarization execute
```

Common options:
- `--manifest src/card_framework/benchmark/manifests/diarization_ami_test.json`
- `--config src/card_framework/config/config.yaml`
- `--providers <comma-separated provider ids>`
- `--device auto|cpu|cuda`
- `--max-samples <int>`
- `--collar <seconds>`
- `--skip-overlap`
- `--output-dir artifacts/diarization_benchmark`

## QA Benchmark

```bash
uv run python -m card_framework.benchmark.qa
```

The QA CLI uses:
- `src/card_framework/benchmark/provider_profiles.yaml`
- `src/card_framework/benchmark/qa_config.yaml`
- `src/card_framework/config/config.yaml`

## QA Supervisor

```bash
uv run python -m card_framework.benchmark.qa_supervisor --summary-xml <path-to-summary.xml> --source-transcript <path-to-transcript>
```

The supervisor wraps the real QA CLI in a child process, runs a telemetry smoke test first, samples host RAM plus NVIDIA VRAM periodically, kills the child when RAM reaches `80%` or any GPU reaches `12288 MiB`, and writes session notes, markdown results, logs, and telemetry under `artifacts/qa_supervisor/<session_id>`.

## MRCR Helper

```bash
uv run python -m card_framework.benchmark.mrcr --skip-model-metadata
```

This helper resolves MRCR vLLM client settings from the packaged benchmark and
runtime configs without downloading datasets at import time.
