# Benchmark Runner

The benchmark source, manifests, rubrics, and QA settings now live under
`src/card_framework/benchmark`. The CLIs resolve those packaged resources by
default, so explicit path flags are usually optional.

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

## MRCR Helper

```bash
uv run python -m card_framework.benchmark.mrcr --skip-model-metadata
```

This helper resolves MRCR vLLM client settings from the packaged benchmark and
runtime configs without downloading datasets at import time.
