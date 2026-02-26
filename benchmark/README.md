# Benchmark Runner

## Execute

```bash
uv run python -m benchmark.run execute --preset hourly
```

Key options:
- `--preset smoke|hourly|full`
- `--manifest benchmark/manifests/benchmark_v1.json`
- `--provider-profiles benchmark/provider_profiles.yaml`
- `--providers <comma-separated ids>`
- `--judge-provider <provider id>`
- `--judge-rubric benchmark/rubrics/default_summarization_rubric.json`
- `--judge-repeats <int>`
- `--disable-order-swap`
- `--alignscore-model <model name>`
- `--output-dir artifacts/benchmark`

Outputs:
- `artifacts/benchmark/<run_id>/benchmark_report.json`
- `artifacts/benchmark/<run_id>/verification.json`
- `artifacts/quality/<run_id>/report.json`

Reference-free metrics in each sample result:
- `alignscore` + `alignscore_backend`
- `judge_scores` (`factuality`, `relevance`, `coherence`, `overall`)
- `judge_pairwise_winner`, `judge_order_consistent`, `judge_repeat_delta`

## Prepare Frozen Manifest

```bash
uv run python -m benchmark.run prepare-manifest --sources local --output benchmark/manifests/benchmark_v1.json
```

Supported sources:
- `local`
- `qmsum`
- `ami`
