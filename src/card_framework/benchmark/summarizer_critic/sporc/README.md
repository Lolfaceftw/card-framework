# SPoRC Summarizer-Critic Benchmark

This folder contains the SPoRC-specific benchmark runner for the summarizer-critic pipeline.

The benchmark can run in two modes:

- `transcript-only`: benchmark the summarizer-critic loop from transcript text only.
- `voice-clone`: run the faithful combined path that also prepares speaker samples and live-draft audio artifacts.

For summarizer-quality benchmarking, `transcript-only` is the recommended mode.

## What This Benchmark Measures

Each run evaluates generated summaries with an LLM-as-a-judge rubric modeled after G-Eval.

The judge scores three dimensions from `0.0` to `1.0`:

- `Factualness`: whether the summary stays faithful to the transcript and avoids important information loss.
- `Naturalness`: whether the summary flows naturally, chronologically, and reads like a coherent spoken segment.
- `Speaker Grammar Similarity`: whether the summary preserves speaker-specific grammar, cadence, filler words, and personality where appropriate.

The benchmark also records:

- target duration,
- candidate duration,
- convergence status,
- iteration count,
- tool invocations,
- per-sample artifacts and aggregate markdown/JSON reports.

## Prerequisites

1. Install dependencies with `uv`.
2. Accept access to the gated Hugging Face dataset `blitt/SPoRC` if you will download it directly.
3. Set environment variables before running:

```bash
HF_ACCESS_TOKEN=...
VLLM_URL=http://host:8000/v1
```

The repository now auto-loads the root `.env` file on package import when it exists, without overriding variables that are already set in the shell.

It also mirrors common aliases automatically, including:

- `HF_ACCESS_TOKEN` -> `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`
- `VLLM_URL` -> `CARD_FRAMEWORK_VLLM_URL`, `VLLM_BASE_URL`

So for SPoRC benchmark runs, a root `.env` containing `HF_ACCESS_TOKEN` and `VLLM_URL` is enough.

The benchmark runner also accepts the Hugging Face token through `HF_TOKEN`, so a common shell pattern is:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN"
```

If you use a local `.env`, make sure it contains at least:

```bash
HF_ACCESS_TOKEN=hf_xxx
VLLM_URL=http://host:8000/v1
```

## Main Entry Point

Run everything through:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc
```

Available subcommands:

- `prepare-dataset`
- `curate-longform`
- `execute`
- `duration-sweep`
- `supervise`

## Choose The Right Command

Use `execute` when you want one benchmark configuration:

- one transcript-only or voice-clone run,
- one optional `--duration-preset` value or one `--target-seconds-override`,
- one sample cap controlled by `--max-samples`.

Use `duration-sweep` when you want multiple duration targets in one invocation:

- `--mode single` runs one preset and uses `--duration-preset`,
- `--mode batch-all` runs multiple presets and uses `--duration-presets`,
- `--max-samples` controls how many prepared samples run per preset by default,
- `--sample-seed` switches sample selection from manifest-order prefixing to deterministic seeded-random ordering,
- `--examples` remains available as a sweep-only override for `--max-samples`,
- `--parallelism 0` runs every selected preset concurrently,
- stalled preset children now heartbeat every `30s`, write per-preset checkpoints, and auto-restart from checkpoint when the active sample stops making progress,
- `duration-sweep` is transcript-only by default unless you add `--enable-voice-clone`.

Common flag mix-ups:

- `execute` does not accept a comma-separated list in `--duration-preset`.
- `duration-sweep --mode batch-all` should use `--duration-presets`, not `--duration-preset`.
- `duration-sweep` now honors `--max-samples`; use `--examples` only when you want a sweep-only override.

## Artifact Locations

Default roots:

- sample prepared dataset root: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_sample`
- sample manifest: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_sample/manifest.json`
- full prepared dataset root: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full`
- full manifest: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/manifest.json`
- full samples sidecar: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/samples.jsonl.gz`
- curated long-form manifest: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json`
- curated long-form audit: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_audit.json`
- execute runs: `artifacts/summarizer_critic_benchmark/sporc_runs`
- sweep runs: `artifacts/summarizer_critic_benchmark/sweeps`
- supervised sessions: `artifacts/summarizer_critic_benchmark/sporc_supervisor`

Each `execute` run writes:

- `report.json`
- `results.md`
- `notes.md`

Each `duration-sweep` run writes:

- `report.json`
- `results.md`
- `logs/<preset>.log`
- `checkpoints/<preset>.checkpoint.json`

Each preset inside a `duration-sweep` still launches a normal `execute` run, so
the aggregate sweep artifacts live under `artifacts/summarizer_critic_benchmark/sweeps`
while the detailed per-preset run artifacts still appear under
`artifacts/summarizer_critic_benchmark/sporc_runs`. Sweep children suppress token
stream rendering, report sample progress back to the parent, and persist
checkpoint state after every terminal sample, so the sweep terminal view stays
on aggregate sample positions and combined ETA while stalled children can be
restarted without rerunning completed samples.

## Step 1: Prepare The Dataset

Prepare the default sample split:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" uv run python -m card_framework.benchmark.summarizer_critic.sporc prepare-dataset --dataset-variant sample
```

Prepare the full split:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" uv run python -m card_framework.benchmark.summarizer_critic.sporc prepare-dataset --dataset-variant full
```

The full split now writes to `prepared_full` by default so it does not collide with
sample-manifest artifacts. Full prepares now keep `manifest.json` as a compact
header and stream sample rows through `samples.jsonl.gz`, so the post-download
prepare step does not have to hold the entire episode table and turn table in RAM.
The CLI now also prints timed stderr status lines such as `Loading ...` and
`Loaded ... in X.XX seconds.` while it imports the SPoRC helpers, downloads
source files, loads or rebuilds the episode cache, and streams the speaker-turn
table, so long full prepares no longer look idle after the Hugging Face
download bars disappear.

If Hugging Face prints the `hf_xet` fallback warning, installing `hf_xet` only
improves download throughput. It does not change how the local prepare step
parses and joins the downloaded SPoRC files.

Curate a strict long-form multi-speaker manifest from the prepared full data:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc curate-longform \
  --source-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/manifest.json"
```

`curate-longform` uses the released SPoRC speaker-turn table as the diarized
transcript source and keeps only episodes that satisfy all of these gates:

- `15,000+` spoken words after removing music/noise placeholder tokens
- `120+` transcript segments
- `2+` speakers total
- `2+` substantial speakers with `1,000+` spoken words and `20+` segments each
- dominant-speaker spoken-word share `<= 0.90`
- non-lexical token ratio `<= 0.10`

The command now also prints timed stderr status lines while it loads the source
manifest, evaluates stored curation stats, and writes the curated manifest plus
audit JSON, so long curation passes no longer look idle. Fresh
`prepare-dataset` outputs now persist reusable curation stats in sample
metadata, so `curate-longform` only falls back to reopening transcript JSON for
older manifests that were prepared before that optimization.

Use local gated files instead of downloading:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc prepare-dataset \
  --episode-data-path "C:/path/to/episodeLevelData.jsonl.gz" \
  --turn-data-path "C:/path/to/speakerTurnData.jsonl.gz"
```

Useful flags:

- `--prepared-root`: custom output directory
- `--dataset-variant sample|full`
- `--reference-wpm`: target-duration derivation reference rate
- `--force-download`: redownload source dataset files
- `curate-longform --source-manifest <path>`: prepared manifest to filter
- `curate-longform --output-manifest <path>`: curated long-form manifest path
- `curate-longform --audit-path <path>`: inclusion/exclusion audit JSON path

## Step 2: Run One Benchmark

Recommended transcript-only run:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc execute \
  --prepare-if-missing \
  --summarizer-provider vllm_default \
  --critic-provider vllm_default \
  --judge-provider vllm_default \
  --embedding-mode disabled \
  --transcript-only
```

Run with a fixed duration preset:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc execute \
  --prepare-if-missing \
  --summarizer-provider vllm_default \
  --critic-provider vllm_default \
  --judge-provider vllm_default \
  --embedding-mode disabled \
  --transcript-only \
  --duration-preset 1m \
  --max-samples 10
```

Run with an explicit target instead of a preset:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc execute \
  --prepare-if-missing \
  --transcript-only \
  --target-seconds-override 90 \
  --max-samples 10
```

Run the faithful audio path instead of transcript-only:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc execute \
  --prepare-if-missing \
  --summarizer-provider vllm_default \
  --critic-provider vllm_default \
  --judge-provider vllm_default
```

Important behavior:

- `execute` defaults to the voice-clone path.
- add `--transcript-only` to skip source-audio download, speaker-sample generation, and live-draft voice cloning.
- use `--dataset-manifest` with the curated long-form manifest for `5m` and `15m` benchmark runs.
- direct `execute` runs launched from a real terminal now stream live markdown-rendered panels for the Summarizer, Critic, and Judge, and each panel appears as soon as that agent request starts so long-prefill calls no longer look idle; piped or supervised runs still fall back to plain streamed `[THINKING]` and `[CONTENT]` text.
- `--duration-preset` and `--target-seconds-override` are mutually exclusive.

## Step 3: Run A Duration Sweep

Show the non-interactive menu/help text:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep
```

Run one preset for `10` examples:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
  --mode single \
  --duration-preset 5m \
  --max-samples 10 \
  --summarizer-provider vllm_default \
  --critic-provider vllm_default \
  --judge-provider vllm_default \
  --embedding-mode disabled \
  --max-iterations 50
```

Run all built-in presets for `10` examples each:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
  --mode batch-all \
  --max-samples 10 \
  --parallelism 0 \
  --summarizer-provider vllm_default \
  --critic-provider vllm_default \
  --judge-provider vllm_default \
  --embedding-mode disabled \
  --max-iterations 50
```

Run a custom subset such as `30s`, `1m`, and `5m` for `10` examples each:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
  --mode batch-all \
  --duration-presets "30s,1m,5m" \
  --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --max-samples 10 \
  --parallelism 0 \
  --summarizer-provider vllm_default \
  --critic-provider vllm_default \
  --judge-provider vllm_default \
  --embedding-mode disabled \
  --max-iterations 50
```

Built-in presets:

- `30s`
- `1m`
- `5m`
- `15m`

Sweep-specific flags:

- `--mode menu|single|batch-all`
- `--max-samples <int>` where `0` means all prepared examples per preset
- `--examples <int>` to override `--max-samples` only for the sweep
- `--parallelism <int>` where `0` means run all selected presets concurrently
- `--duration-presets 30s,1m,5m,15m`
- `--sweep-root <path>`
- `--enable-voice-clone`
- `--child-stall-seconds <float>` to restart a child when its active sample reports no new activity for too long
- `--max-child-restarts <int>` to cap parent-driven checkpoint resumes per preset
- `--progress-heartbeat-seconds <float>` to control sweep-child heartbeat cadence

Important behavior:

- `duration-sweep` is transcript-only by default.
- add `--enable-voice-clone` to switch the sweep to the faithful audio path.
- `--mode single` expects one `--duration-preset` such as `5m`.
- `--mode batch-all` expects `--duration-presets "30s,1m,5m"` when you want a subset.
- quote the comma-separated `--duration-presets` value in PowerShell to keep it as one argument.
- `duration-sweep` defaults to `--max-samples` for per-preset sample count, and `--examples` overrides it only for the sweep.
- `--sample-seed <int>` makes `execute`, `supervise`, and `duration-sweep` choose a reproducible seeded-random subset; omitting it preserves manifest order.
- `duration-sweep` defaults to aggregate progress output with combined ETA instead of child token streams.
- `duration-sweep` also applies sweep-only request timeout defaults of `900s` for summarizer/critic and `300s` for retrieval/index so a wedged backend becomes a failed sample instead of a permanently hung preset.

## Running A Custom Manifest

If you already prepared or curated a subset manifest, pass it directly:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
  --mode batch-all \
  --duration-presets "30s,1m,5m" \
  --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --max-samples 10 \
  --parallelism 0 \
  --summarizer-provider vllm_default \
  --critic-provider vllm_default \
  --judge-provider vllm_default \
  --embedding-mode disabled \
  --max-iterations 50
```

## Important Flags

Shared execution flags users usually need:

- `--dataset-manifest <path>`
- `--prepare-if-missing`
- `--force-download-dataset`
- `--force-download-audio`
- `--summarizer-provider <provider_id>`
- `--critic-provider <provider_id>`
- `--judge-provider <provider_id>`
- `--embedding-mode disabled|enabled`
- `--max-iterations <int>`
- `--output-root <path>`
- `execute`: `--max-samples <int>` where `0` means all prepared examples
- `execute`: `--sample-seed <int>` for deterministic seeded-random sample selection
- `execute`: `--duration-preset <30s|1m|5m|15m>`
- `execute`: `--target-seconds-override <int>`
- `duration-sweep`: `--max-samples <int>` where `0` means all prepared examples per preset
- `duration-sweep`: `--sample-seed <int>` for deterministic seeded-random per-preset sample selection
- `duration-sweep`: `--examples <int>` as a sweep-only override
- `duration-sweep`: `--parallelism <int>`
- `duration-sweep`: `--duration-presets 30s,1m,5m,15m`

## Timeouts

Stage timeout flags are per-request network timeouts, not overall benchmark limits.

Relevant flags:

- `--summarizer-timeout-seconds`
- `--critic-timeout-seconds`
- `--retrieval-timeout-seconds`
- `--index-timeout-seconds`

If omitted, the stage timeout is unbounded.

Sweep exception:

- `duration-sweep` injects default request timeouts of `900s` for summarizer/critic and `300s` for retrieval/index unless you override them explicitly.
- standalone `execute` keeps the historical unbounded defaults unless you pass timeout flags yourself.

Practical guidance:

- use shorter timeouts for quick failure detection,
- use longer timeouts when the backend is slow,
- remember that a timeout does not mean the whole benchmark budget expired; it means that stage request did not complete within the configured window.

## Multi-Endpoint Runs

One benchmark process uses one provider URL at a time through `VLLM_URL`.

To fan out across multiple vLLM ports such as `8000`, `8001`, and `8002`, start separate benchmark processes and override `VLLM_URL` per process.

Example pattern:

```bash
VLLM_URL=http://host:8000/v1 <command A>
VLLM_URL=http://host:8001/v1 <command B>
VLLM_URL=http://host:8002/v1 <command C>
```

## Recommended Benchmark Methodology

For summarizer-only evaluation:

1. prepare the dataset,
2. run `transcript-only`,
3. keep `--embedding-mode disabled` unless retrieval is part of the experiment,
4. fix the duration with `--duration-preset`,
5. use a fixed manifest or fixed sample count for comparability,
6. compare `results.md` and `report.json` across runs.

For the duration-constraint study described in this repository, the most direct command shape is:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
  --mode batch-all \
  --max-samples 10 \
  --parallelism 0 \
  --summarizer-provider vllm_default \
  --critic-provider vllm_default \
  --judge-provider vllm_default \
  --embedding-mode disabled \
  --transcript-only \
  --max-iterations 50
```

## Reading The Outputs

Start with `results.md` for native Markdown tables. Run reports now include
95% confidence intervals for aggregate metric means, and duration-sweep reports
also include a composite-score CI column per preset.

Use `report.json` when you need:

- full per-sample metrics,
- judge scores,
- convergence status,
- transcript and summary per-speaker messages,
- artifact paths.

## Troubleshooting

- missing Hugging Face access: confirm the dataset license/terms are accepted and `HF_TOKEN` is set.
- missing manifest: run `prepare-dataset` first or add `--prepare-if-missing`.
- wrong mode: remember `execute` defaults to voice-clone, while `duration-sweep` defaults to transcript-only.
- empty or partial results: inspect the run's `notes.md`, `results.md`, and `report.json` under `artifacts/summarizer_critic_benchmark/sporc_runs`.
- slow backend: reduce sample count first, then tune stage timeouts and/or distribute separate runs across multiple `VLLM_URL` endpoints.
