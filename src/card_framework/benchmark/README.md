# Benchmarks

This folder holds every benchmark that ships with CARD. Each one is a module
you run with `uv run python -m ...` from a source checkout. Manifests, rubrics,
and provider profiles are bundled with the package, so most path flags are
optional.

All outputs go under `artifacts/`, which git ignores.

| Benchmark | What it measures | Command |
| --- | --- | --- |
| [Summarizer-critic](#summarizer-critic-benchmarks) | Summary quality and duration control on QMSum and SPoRC | `card_framework.benchmark.summarizer_critic` |
| [Summarization matrix](#summarization-matrix) | Several providers on the same transcripts, scored by a judge | `card_framework.benchmark.run` |
| [Diarization](#diarization) | Speaker diarization error rate on AMI-style data | `card_framework.benchmark.diarization` |
| [QA](#qa) | Whether a summary answers questions about its source | `card_framework.benchmark.qa` |

Most benchmarks expect an OpenAI-compatible vLLM server at
`http://127.0.0.1:8000/v1`. To use a different address, set `VLLM_URL`.

## Summarizer-critic benchmarks

These run the same summarizer, critic, and retrieval agents the main pipeline
uses, against real datasets with human reference summaries.

```bash
uv run python -m card_framework.benchmark.summarizer_critic prepare-dataset
uv run python -m card_framework.benchmark.summarizer_critic execute --prepare-if-missing --summarizer-provider vllm_default
```

A run scores duration control, factualness, naturalness, and speaker
attribution, then writes `report.json`, `notes.md`, and `results.md` under
`artifacts/summarizer_critic_benchmark`.

Detailed guides:

- [summarizer_critic/README.md](summarizer_critic/README.md): which CLI to use and shared defaults
- [summarizer_critic/qmsum/README.md](summarizer_critic/qmsum/README.md): the QMSum meeting benchmark
- [summarizer_critic/sporc/README.md](summarizer_critic/sporc/README.md): the SPoRC podcast benchmark

The judge prompt is shared by every benchmark and lives in
`src/card_framework/prompts/templates/conversation_summary_judge_*.jinja2`. Each
scored dimension uses the same 1–5 Likert items, so the same wording works for
human raters and for the LLM judge.

## Summarization matrix

Runs several providers over the same transcripts and scores each one with a
reference-free judge that sees the full source transcript.

```bash
# Build a manifest from a local transcript
uv run python -m card_framework.benchmark.run prepare-manifest --sources local

# Run the benchmark
uv run python -m card_framework.benchmark.run execute --preset hourly
```

`prepare-manifest --sources local` uses the first transcript it finds, checking
in this order (or pass `--local-transcript-path`):

1. `transcript.json` in the repository root
2. `*.transcript.json` in the repository root
3. `artifacts/transcripts/*.transcript.json`

Common `execute` options:

| Option | Default |
| --- | --- |
| `--preset smoke\|hourly\|full` | |
| `--manifest <path>` | `manifests/benchmark_v1.json` |
| `--provider-profiles <path>` | `provider_profiles.yaml` |
| `--providers <ids>` | all profiles, comma-separated |
| `--judge-provider <id>` | |
| `--judge-rubric <path>` | `rubrics/default_summarization_rubric.json` |
| `--judge-repeats <n>` | |
| `--disable-order-swap` | order swap on |
| `--alignscore-model <name>` | |
| `--output-dir <path>` | `artifacts/benchmark` |

Outputs:

- `artifacts/benchmark/<run_id>/benchmark_report.json`
- `artifacts/benchmark/<run_id>/verification.json`
- `artifacts/quality/<run_id>/report.json`

If every sample is skipped, the command exits with a non-zero code and keeps its
output folder so you can see why.

### Running with memory limits

`watchdog` runs a benchmark repeatedly and kills it if system RAM reaches 80% or
GPU memory reaches 12 GiB. It records RAM and GPU usage while it runs.

```bash
uv run python -m card_framework.benchmark.watchdog --max-runs 3 --sample-interval-seconds 5
```

By default it runs `benchmark.run execute --preset smoke`. To watch a different
command, put it after `--`:

```bash
uv run python -m card_framework.benchmark.watchdog -- python -m card_framework.benchmark.qa --summary-xml summary.xml --source-transcript transcript.json
```

Logs, telemetry, and notes go to `artifacts/benchmark_watchdog`.

## Diarization

```bash
uv run python -m card_framework.benchmark.diarization prepare-manifest
uv run python -m card_framework.benchmark.diarization execute
```

Common options:

| Option | Default |
| --- | --- |
| `--manifest <path>` | `manifests/diarization_ami_test.json` |
| `--config <path>` | the bundled `config/config.yaml` |
| `--providers <ids>` | |
| `--device auto\|cpu\|cuda` | `auto` |
| `--max-samples <n>` | all |
| `--collar <seconds>` | `0` (strict) |
| `--skip-overlap` | overlap scored |
| `--output-dir <path>` | `artifacts/diarization_benchmark` |

The bundled AMI manifest expects the audio under
`artifacts/diarization_datasets/ami/`. To describe your own data, see
`manifests/diarization_manifest.example.json`.

## QA

```bash
uv run python -m card_framework.benchmark.qa
```

Settings come from `provider_profiles.yaml`, `qa_config.yaml`, and the bundled
`config/config.yaml`.

To run it with the same RAM and GPU limits as `watchdog`, use the supervisor:

```bash
uv run python -m card_framework.benchmark.qa_supervisor --summary-xml <summary.xml> --source-transcript <transcript.json>
```

It runs a short telemetry check first, then writes notes, logs, and telemetry
under `artifacts/qa_supervisor/<session_id>`.

## MRCR helper

```bash
uv run python -m card_framework.benchmark.mrcr --skip-model-metadata
```

Prints the vLLM client settings that MRCR runs would use, based on the bundled
configs. It doesn't download any datasets.
