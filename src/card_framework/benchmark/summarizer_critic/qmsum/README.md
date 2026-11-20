# QMSum Benchmark

Runs CARD's summarizer and critic on the
[QMSum](https://github.com/Yale-LILY/QMSum) general-summary test split. Each
sample's target duration comes from the length of its human reference summary,
and an LLM judge scores the result.

For defaults shared with the SPoRC benchmark, see [../README.md](../README.md).

Commands below use bash syntax. In PowerShell, set variables with
`$env:VLLM_URL = "..."` and replace the trailing `\` with a backtick.

## 1. Point it at your LLM

The `vllm_default` profile expects a vLLM server at `http://127.0.0.1:8000/v1`.
If yours runs there, skip this step. Otherwise:

```bash
export VLLM_URL="http://your-host:8000/v1"
```

## 2. Prepare the dataset

```bash
uv run python -m card_framework.benchmark.summarizer_critic prepare-dataset
```

This clones the QMSum repository (or reuses an existing clone) and writes a
manifest to
`artifacts/summarizer_critic_benchmark/datasets/qmsum_general_test/manifest.json`.
To use other folders, pass `--qmsum-repo-dir <path>` and `--prepared-root <path>`.

You can skip this step: the commands below take `--prepare-if-missing`.

## 3. Run a smoke test

Run a single sample first to check that your provider, streaming, judge, and
telemetry all work:

```bash
uv run python -m card_framework.benchmark.summarizer_critic supervise \
  --prepare-if-missing \
  --summarizer-provider vllm_default \
  --max-samples 1 \
  --max-iterations 1 \
  --max-runs 1 \
  --sample-interval-seconds 5
```

The critic defaults to the summarizer's provider, and the judge to the critic's.
Pass `--critic-provider` or `--judge-provider` to use different models.

## 4. Run the full benchmark

There are two ways to run it:

| Command | Use it when |
| --- | --- |
| `supervise` | You want the run killed if RAM hits 80% or a GPU hits 12 GiB, plus session notes and child logs |
| `execute` | You want one plain run with no supervisor |

```bash
uv run python -m card_framework.benchmark.summarizer_critic supervise \
  --prepare-if-missing \
  --summarizer-provider vllm_default \
  --max-iterations 4 \
  --critic-timeout-seconds 900 \
  --max-runs 1 \
  --sample-interval-seconds 15
```

To run without the supervisor, replace `supervise` with `execute` and drop
`--max-runs` and `--sample-interval-seconds`.

### Useful flags

| Flag | Meaning |
| --- | --- |
| `--max-samples <n>` | Number of samples to run; `0` runs all |
| `--sample-seed <n>` | Pick a reproducible random subset instead of manifest order |
| `--max-iterations <n>` | Maximum summarizer/critic rounds per sample |
| `--embedding-mode disabled\|enabled` | Turn retrieval on or off |
| `--summarizer-timeout-seconds <s>` | Per-request timeout for the summarizer |
| `--critic-timeout-seconds <s>` | Per-request timeout for the critic |
| `--retrieval-timeout-seconds <s>` | Per-request timeout for retrieval |
| `--index-timeout-seconds <s>` | Per-request timeout for indexing |

With `--embedding-mode enabled`, transcript embeddings are cached in
`artifacts/summarizer_critic_benchmark/embedding_cache` and reused when the
sample, transcript, and embedding model are unchanged.

## Running several benchmarks at once

Each `execute` process starts three local services (retrieval, summarizer,
critic) on fixed ports. To run processes in parallel, give each one its own
three ports:

```bash
uv run python -m card_framework.benchmark.summarizer_critic execute \
  --prepare-if-missing \
  --summarizer-provider vllm_default \
  --retrieval-port 31000 \
  --summarizer-port 31001 \
  --critic-port 31002
```

## Outputs

| Folder | Contents |
| --- | --- |
| `artifacts/summarizer_critic_benchmark/runs/<run_id>/` | `report.json`, `notes.md`, `results.md` |
| `artifacts/summarizer_critic_benchmark/supervisor/<session_id>/` | Supervisor notes, logs, and telemetry |

Start with `results.md`. `report.json` has per-sample detail and records
whether samples were chosen by seed or by manifest order.
