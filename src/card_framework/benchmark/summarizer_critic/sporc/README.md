# SPoRC Summarizer-Critic Benchmark

This folder contains the SPoRC-specific benchmark runner for the summarizer-critic pipeline.

The benchmark can run in two modes:

- `transcript-only`: benchmark the summarizer-critic loop from transcript text only.
- `voice-clone`: run the faithful combined path that also prepares speaker samples and live-draft audio artifacts.

For summarizer-quality benchmarking, `transcript-only` is the recommended mode.
In transcript-only mode, the runner now normalizes the copied working transcript
before scoring: it removes punctuation-only fragments, merges adjacent
same-speaker spans, keeps the raw dataset speaker lists in segment extras for
auditing, and marks ambiguous turns with speaker-candidate metadata instead of
guessing a neighboring speaker so the working transcript remains faithful to
the dataset's primary-speaker assignment.

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
ANTHROPIC_API_KEY=
```

## Main Entry Point

Run everything through:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc
```

Available subcommands:

- `prepare-dataset`
- `curate-longform`
- `dedupe-manifest`
- `export-transcripts`
- `human-sample`
- `llm-sample`
- `execute`
- `duration-sweep`
- `supervise`

## Choose The Right Command

Use `execute` when you want one benchmark configuration:

- one transcript-only or voice-clone run,
- one optional `--duration-preset` value or one `--target-seconds-override`,
- one retained-sample target controlled by `--max-samples`.

Use `duration-sweep` when you want multiple duration targets in one invocation:

- `--mode single` runs one preset and uses `--duration-preset`,
- `--mode batch-all` runs multiple presets and uses `--duration-presets`,
- `--max-samples` controls how many retained samples each preset should finish with by default,
- `--sample-seed` switches sample selection from manifest-order prefixing to deterministic seeded-random ordering,
- `--examples` remains available as a sweep-only override for the same retained-sample target,
- `--parallelism 0` runs every selected preset concurrently,
- `--concurrency <int>` seeds that many worker children per active preset and donates those workers forward when earlier presets finish,
- stalled preset children now heartbeat every `30s`, write per-preset checkpoints, and auto-restart from checkpoint when the active sample stops making progress,
- `duration-sweep` is transcript-only by default unless you add `--enable-voice-clone`.

Common flag mix-ups:

- `execute` does not accept a comma-separated list in `--duration-preset`.
- `duration-sweep --mode batch-all` should use `--duration-presets`, not `--duration-preset`.
- `duration-sweep` now honors `--max-samples`; use `--examples` only when you want a sweep-only override.
- `--max-samples 100` means "retain 100 benchmarked samples," not "stop after the first 100 candidates touched."

Use `human-sample` when you want a human rater to choose transcript segments from
the curated long-form manifest:

- it defaults to `benchmark_longform_15k_manifest.json` and takes the first `N`
  episodes in manifest order,
- it computes a per-episode sampled-word budget from
  `WPM * (preset_seconds / 60) * factor`, with interactive defaults of
  `170 WPM` and factor `5`,
- it estimates a target sample count from transcript line length, splits each
  transcript into word-balanced windows, and shows a transcript-global line
  number column for each window,
- each window accepts up to its displayed quota via comma-separated line
  numbers or `/override <segment>`,
- `Ctrl+C` pauses the session, writes the live SQLite and export artifacts, and
  lets the operator resume later with `--resume-session <uuid>`.

Use `llm-sample` when you want Anthropic to choose transcript-grounded sample
segments automatically:

- it defaults to the same curated long-form manifest prefix and per-episode
  budget formula used by `human-sample`,
- it sends the full transcript in one request per episode instead of windowing,
- it streams summarized thinking plus the structured sample payload to the
  terminal through the shared Rich callback path,
- it persists per-episode `request.json`, `stream.jsonl`, and `response.json`
  traces plus run-level JSONL/CSV exports under `sporc_llm_sampling`,
- every saved sample now includes a direct `speaker_label`, and each episode
  also writes a `selected_segments.txt` export for quick inspection,
- it currently reads `ANTHROPIC_API_KEY` from `.env` and defaults to
  `claude-opus-4-6` with adaptive thinking enabled,
- the structured tool payload is the authoritative output, but the sampler does
  not enforce exact verbatim transcript matching on every returned segment,
  explicitly favors readable reviewer-facing excerpts that preserve what the
  speaker said, including disfluencies, while lightly cleaning up only
  transcript formatting noise such as fragmentation, spacing, and punctuation,
  and does not reject a run just because Anthropic decides to include or skip
  prose around that tool call,
- if you pass `--samples <N>`, it ignores the WPM-based target-word formula and
  returns exactly `N` samples from each full transcript instead,
- if you pass `--select-episode "[7,10]"`, it runs only manifest episodes `7`
  through `10` inclusive and overrides the default manifest-prefix selection,
- if you pass `--disable-thinking`, it disables Anthropic thinking and runs
  with `temperature=0.0`.

## Artifact Locations

Default roots:

- sample prepared dataset root: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_sample`
- sample manifest: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_sample/manifest.json`
- full prepared dataset root: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full`
- full manifest: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/manifest.json`
- full samples sidecar: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/samples.jsonl.gz`
- curated long-form manifest: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json`
- curated long-form audit: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_audit.json`
- human sampling sessions: `artifacts/summarizer_critic_benchmark/sporc_human_sampling`
- automatic LLM sampling runs: `artifacts/summarizer_critic_benchmark/sporc_llm_sampling`
- execute runs: `artifacts/summarizer_critic_benchmark/sporc_runs`
- sweep runs: `artifacts/summarizer_critic_benchmark/sweeps`
- supervised sessions: `artifacts/summarizer_critic_benchmark/sporc_supervisor`

Each `human-sample` session writes:

- `<session_id>/session.sqlite3`
- `<session_id>/session_summary.json`
- `<session_id>/selected_samples.jsonl`
- `<session_id>/selected_samples.csv`

The human-sampling exports update immediately after each completed window, so a
researcher can inspect saved selections without waiting for the whole session
to finish.

Each `llm-sample` run writes:

- `<run_id>/run_summary.json`
- `<run_id>/selected_samples.jsonl`
- `<run_id>/selected_samples.csv`
- `<run_id>/episodes/<sample_id>/request.json`
- `<run_id>/episodes/<sample_id>/stream.jsonl`
- `<run_id>/episodes/<sample_id>/response.json`
- `<run_id>/episodes/<sample_id>/selected_segments.txt`

Each `execute` run writes:

- `report.json`
- `results.md`
- `notes.md`
- `samples/index.json`
- `samples/README.md`
- `samples/<sample_id>/manifest.json`
- `samples/<sample_id>/final/summary.xml`
- `samples/<sample_id>/final/critic_feedback.txt` when loop diagnostics include final critic feedback
- `samples/<sample_id>/final/loop_diagnostics.json`
- `samples/<sample_id>/final/sample_result.json`
- `samples/<sample_id>/llm/summarizer/call_XXX.json`
- `samples/<sample_id>/llm/critic/call_XXX.json`
- `samples/<sample_id>/llm/judge/call_XXX.json`

The `samples/` subtree is the developer-facing entrypoint for raw per-sample artifacts:

- `samples/index.json` lists every executed sample folder for the run.
- `samples/README.md` explains the directory layout.
- each `call_XXX.json` file keeps the normalized request payload, streamed reasoning tokens, streamed content tokens, and the final response or error for that model call.
- each sample-local `manifest.json` points at that sample's final outputs plus all recorded call files.

Each `duration-sweep` run writes:

- `report.json`
- `results.md`
- `logs/<preset>.log` or `logs/<preset>.shard-<i>-of-<n>.log`
- `checkpoints/<preset>.checkpoint.json` or `checkpoints/<preset>.shard-<i>-of-<n>.checkpoint.json`

Each preset inside a `duration-sweep` still launches one or more normal `execute`
runs, so the aggregate sweep artifacts live under
`artifacts/summarizer_critic_benchmark/sweeps` while the detailed per-child run
artifacts still appear under
`artifacts/summarizer_critic_benchmark/sporc_runs`. Sweep children suppress token
stream rendering, report sample progress back to the parent, and persist
checkpoint state after every terminal sample, so the sweep terminal view stays
on aggregate sample positions and a donation-aware combined ETA while stalled children can be
restarted without rerunning completed samples. On real terminals the parent now
shows colorized status lines plus a footer `tqdm` progress bar with ETA instead
of repeatedly printing raw `Sweep progress: ...` lines. When a child stalls, the resumed
`execute` run records only the orphaned active sample failed with
`sweep_watchdog_stalled` and then continues the rest of that preset from the
checkpoint instead of aborting the whole preset. When `--concurrency > 1`, each
seeded preset gets multiple worker children that coordinate through a shared
preset queue, so new workers can join a preset mid-run without overlapping
samples and finished presets can donate their worker slots to the next
unfinished preset in order. Pressing `Ctrl+C` now stops every active child,
closes the footer bar cleanly, and exits the sweep with `Cancelled.` after the
active child connections are torn down.

To inspect one sweep shard end to end:

1. Open `sweeps/<sweep_id>/logs/<preset>.shard-<i>-of-<n>.log`.
2. Find the `run_start` event and note its child `run_id`.
3. Open `sporc_runs/<run_id>/report.json` or `sporc_runs/<run_id>/notes.md`.
4. Open `sporc_runs/<run_id>/samples/README.md` or `sporc_runs/<run_id>/samples/index.json`.
5. Drill into `sporc_runs/<run_id>/samples/<sample_id>/` for that shard's final outputs and raw LLM traces.

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
- `dedupe-manifest --manifest-path <path>`: inline manifest to rewrite in place
- `dedupe-manifest --audit-path <path>`: optional dedupe audit JSON path
- `export-transcripts --manifest-path <path>`: manifest to read transcript paths from
- `export-transcripts --episodes <spec>`: episode positions or ranges to export
- `export-transcripts --output-path <path>`: optional combined transcript TXT path
- `export-transcripts --backwards-compat`: interpret `--episodes` as curated longform positions and map them onto the target manifest by `episode_key`
- `export-transcripts --raw`: write exact matching JSONL rows from the dataset with no formatting from us

## Step 1.45: Dedupe A Curated Manifest In Place

Rewrite the curated inline manifest in place and emit a sibling audit JSON:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc dedupe-manifest
```

Or target a custom inline manifest explicitly:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc dedupe-manifest \
  --manifest-path "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --audit-path "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.dedupe_audit.json"
```

Important behavior:

- `dedupe-manifest` rewrites the same manifest path atomically; it does not write a second manifest copy.
- it keeps the first occurrence of each confirmed duplicate and preserves the kept-sample order.
- it is optimized for large manifests: it groups rows by cheap metadata fingerprints first and only opens transcript JSON for small collision groups that need confirmation.
- it currently supports manifests with inline `samples` arrays, which matches the curated long-form manifest produced by `curate-longform`.

## Step 1.46: Export Combined Raw Transcript Text

Write one combined `.txt` file for selected manifest episodes:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc export-transcripts \
  --episodes "7,10-12"
```

Or target a custom manifest and output path:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc export-transcripts \
  --manifest-path "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --episodes "[7,10]" \
  --output-path "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/episodes_7_10.txt"
```

Or write the exact matching raw dataset rows instead of the human-formatted transcript view:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc export-transcripts \
  --episodes "7,10-12" \
  --raw
```

Important behavior:

- `export-transcripts` writes one combined raw transcript text file for the selected manifest positions by default.
- `--episodes` accepts comma-separated positions, ranges like `7-10`, and bracket form like `[7,10]` for an inclusive range.
- `--backwards-compat` keeps `--episodes` on curated longform positions even if `--manifest-path` points at the original prepared manifest, and maps those selections onto the target manifest by `episode_key`.
- it uses the manifest's actual raw `turn_data_path`, not the prepared transcript JSON files.
- `--raw` writes the exact matching JSONL rows from that dataset file with no headers, no timestamp rendering, no speaker-label formatting, and no normalization from us.
- it is optimized for speed: it only reads the selected manifest rows plus a streamed pass over the raw turn file that stops as soon as the final requested episode block has been collected.

## Step 1.5: Run Human Transcript Sampling

Launch the interactive sampler with the curated long-form manifest:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc human-sample
```

Or provide the session settings up front:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc human-sample \
  --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --duration-preset 5m \
  --episodes 20 \
  --words-per-minute 170 \
  --sample-factor 5 \
  --max-windows-per-episode 10
```

Resume a paused session:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc human-sample \
  --resume-session "<session-uuid>"
```

Window input examples:

- `12`
- `12,13`
- `12,/override [SPEAKER_04] Replacement segment text`
- `"/override [SPEAKER_04] Replacement, with comma"`
- `/skip`

## Step 1.6: Run Automatic Anthropic Sampling

Launch the automatic sampler with the curated long-form manifest:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc llm-sample
```

Override the main defaults up front:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc llm-sample \
  --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --duration-preset 5m \
  --episodes 20 \
  --words-per-minute 170 \
  --sample-factor 5 \
  --model claude-opus-4-6 \
  --thinking-effort medium
```

Use a fixed sample count instead of the WPM formula:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc llm-sample \
  --episodes 20 \
  --samples 20
```

Rerun only a specific inclusive manifest span:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc llm-sample \
  --select-episode "[7,10]"
```

Important behavior:

- `llm-sample` is non-interactive and does not support resume yet.
- it fails fast when `ANTHROPIC_API_KEY` is missing.
- `--samples <N>` overrides the WPM-based target-word mode.
- `--select-episode "[start,end]"` uses 1-based inclusive manifest positions and overrides `--episodes`.
- `--disable-thinking` disables Anthropic thinking and uses `temperature=0.0`.
- it retries one invalid out-of-budget or malformed structured submission per
  episode, then records the episode as failed and continues.

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
- transcript-only runs still start from the prepared SPoRC transcript, but they now write a normalized working copy under `sporc_runs/<run_id>/working_transcripts/` before the summarizer-critic loop starts; that working copy only applies lossless cleanup and uncertainty tagging, not speaker reassignment.
- use `--dataset-manifest` with the curated long-form manifest for `5m` and `15m` benchmark runs.
- `--max-samples` is an exact retained-sample target. If one candidate is excluded for infra-only reasons, the runner keeps going until it retains the requested count or exhausts the manifest.
- replacement-eligible outcomes are infra-only: `preflight_failed`, `startup_unavailable`, `connection_unavailable`, `stage_timeout`, and resume/watchdog orphan failures.
- startup or connection-unavailable failures retry the same sample up to two additional times before that sample is excluded and replaced by the next deterministic candidate.
- normal benchmark failures after the sample actually ran still count toward the retained target.
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

Run all presets concurrently with `3` worker children per seeded preset:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
  --mode batch-all \
  --max-samples 30 \
  --parallelism 0 \
  --concurrency 3 \
  --summarizer-provider vllm_default \
  --critic-provider vllm_default \
  --judge-provider vllm_default \
  --embedding-mode disabled \
  --max-iterations 50
```

Run a custom subset such as `30s`, `1m`, and `5m` for `100` retained examples each:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
  --mode batch-all \
  --duration-presets "30s,1m,5m" \
  --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --max-samples 100 \
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
- `--max-samples <int>` where `0` means all prepared examples per preset and positive values mean retained samples per preset
- `--examples <int>` to override `--max-samples` only for the sweep using the same retained-sample semantics
- `--parallelism <int>` where `0` means run all selected presets concurrently
- `--concurrency <int>` to seed multiple worker children per active preset
- `--duration-presets 30s,1m,5m,15m`
- `--sweep-root <path>`
- `--enable-voice-clone`
- `--child-stall-seconds <float>` to restart a child when its active sample reports no new activity for too long
- `--max-child-restarts <int>` as the warning threshold for repeated parent-driven checkpoint resumes; stalled samples are still failed individually and later samples continue
- `--progress-heartbeat-seconds <float>` to control sweep-child heartbeat cadence

Important behavior:

- `duration-sweep` is transcript-only by default.
- add `--enable-voice-clone` to switch the sweep to the faithful audio path.
- `--mode single` expects one `--duration-preset` such as `5m`.
- `--mode batch-all` expects `--duration-presets "30s,1m,5m"` when you want a subset.
- quote the comma-separated `--duration-presets` value in PowerShell to keep it as one argument.
- `duration-sweep` defaults to `--max-samples` for the per-preset retained target, and `--examples` overrides it only for the sweep.
- `--parallelism` controls how many presets are seeded at once, while `--concurrency` controls how many worker children each seeded preset starts with.
- finished presets donate their worker slots forward to the next unfinished preset in duration order, so later presets can speed up as earlier presets complete.
- `--sample-seed <int>` makes `execute`, `supervise`, and `duration-sweep` choose a reproducible seeded-random subset; omitting it preserves manifest order.
- `duration-sweep` defaults to aggregate progress output with a donation-aware combined ETA instead of child token streams, and interactive terminals now render that aggregate view as a footer `tqdm` bar.
- `duration-sweep` also applies sweep-only request timeout defaults of `900s` for summarizer/critic and `300s` for retrieval/index so a wedged backend becomes a failed sample instead of a permanently hung preset.
- `Ctrl+C` during `duration-sweep` now shuts down every active child before the parent exits, so local A2A ports, pipes, and log handles do not linger.
- run notes/results now expose retained sample count, attempted candidate count, and replacement-excluded candidates separately so backfill behavior stays auditable.

## Running A Custom Manifest

If you already prepared or curated a subset manifest, pass it directly:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
  --mode batch-all \
  --duration-presets "30s,1m,5m" \
  --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --max-samples 100 \
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
- `execute`: `--max-samples <int>` where `0` means all prepared examples and positive values mean retained samples
- `execute`: `--sample-seed <int>` for deterministic seeded-random sample selection
- `execute`: `--duration-preset <30s|1m|5m|15m>`
- `execute`: `--target-seconds-override <int>`
- `duration-sweep`: `--max-samples <int>` where `0` means all prepared examples per preset and positive values mean retained samples per preset
- `duration-sweep`: `--sample-seed <int>` for deterministic seeded-random per-preset sample selection
- `duration-sweep`: `--examples <int>` as a sweep-only override
- `duration-sweep`: `--parallelism <int>`
- `duration-sweep`: `--concurrency <int>`
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
also include a composite-score CI column per preset. SPoRC execute reports now
separate retained sample results from replacement-excluded candidates, while
`notes.md` records the retained target, attempted candidate count, and
replacement count explicitly.

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
