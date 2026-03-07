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

## Diarization Benchmark

```bash
uv run python -m benchmark.diarization prepare-manifest
uv run python -m benchmark.diarization
```

Key options:
- `prepare-manifest`
- `execute`
- `--manifest <json path>`
- `--config conf/config.yaml`
- `--providers <comma-separated provider ids>`
- `--device auto|cpu|cuda`
- `--max-samples <int>`
- `--collar <seconds>`
- `--skip-overlap`
- `--output-dir artifacts/diarization_benchmark`

AMI prep options:
- `--output benchmark/manifests/diarization_ami_test.json`
- `--data-root artifacts/diarization_datasets/ami`
- `--subset train|dev|test`
- `--num-samples <int>`
- `--force-download`

Default comparison set:
- configured `audio.diarization.provider`
- `pyannote_community1`
- `nemo_sortformer_streaming`
- `nemo_sortformer_offline`

Default AMI prep behavior:
- Downloads public `Mix-Headset.wav` audio from the AMI corpus.
- Downloads `only_words` RTTM/UEM/list files from `BUTSpeechFIT/AMI-diarization-setup`.
- Writes the default diarization manifest to `benchmark/manifests/diarization_ami_test.json`.

Manifest fields:
- `sample_id`
- `dataset`
- `subset` (optional)
- `audio_filepath` or `audio_path`
- `rttm_filepath` or `reference_rttm_path`
- `uem_filepath` or `uem_path` (optional, required for JER)
- `num_speakers` (optional)

Outputs:
- `artifacts/diarization_benchmark/<run_id>/diarization_report.json`
- `artifacts/diarization_benchmark/<run_id>/verification.json`
- `artifacts/diarization_benchmark/<run_id>/predictions/<provider>/<sample>.rttm`

Notes:
- `uv run python -m benchmark.diarization --manifest <path>` still works as the legacy execute-style invocation.
- The CLI scores DER with `pyannote.metrics` for every sample and computes JER when a per-sample UEM file is provided.
- The benchmark uses the same `audio_pipeline.factory.build_speaker_diarizer(...)` wiring as stage 1, so provider behavior matches runtime behavior.
- The default AMI prep path currently supports only the public `Mix-Headset` stream. Use a custom manifest for CALLHOME, DIHARD, or other local datasets. The manual template is `benchmark/manifests/diarization_manifest.example.json`.
- `pyannote_community1` requires `pyannote.audio` plus accepted Hugging Face model terms and a token available through `audio.diarization.pyannote.auth_token` or `audio.diarization.pyannote.auth_token_env`.
- The current Sortformer adapters benchmark the pretrained model API directly. Vendor-published post-processing YAML tuning is not reproduced automatically inside this repo.

## MRCR

```bash
uv run python -m benchmark.mrcr
```

`benchmark/mrcr.py` resolves the `vllm_default` provider from
`benchmark/provider_profiles.yaml`, then applies endpoint overrides from
`benchmark/qa_config.yaml`. You can override connection settings with
`VLLM_BASE_URL`, `VLLM_API_KEY`, and `VLLM_TIMEOUT_SECONDS`.
