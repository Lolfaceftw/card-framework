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
The judge now receives the full formatted transcript by default rather than a
truncated source-text prefix, and the reusable conversation-summary judge
wording now lives in shared prompt templates under
`src/card_framework/prompts/templates/conversation_summary_judge_system.jinja2`
and `src/card_framework/prompts/templates/conversation_summary_judge_user.jinja2`
instead of inline SPoRC-only strings.
Judge calls also stop setting an explicit output-token cap from the application
side, so the provider receives no `max_tokens` override unless a different
caller adds one.

The judge scores three dimensions from `0.0` to `1.0`:

- `Factualness`: whether the summary stays faithful to the transcript and avoids important information loss.
- `Naturalness`: whether the summary flows naturally, chronologically, and reads like a coherent spoken segment.
- `Speaker Grammar Similarity`: whether the summary preserves speaker-specific grammar, cadence, filler words, and personality where appropriate.

The shared judge prompt now phrases each dimension as three parallel
Likert-friendly guide items so the same rubric wording can be reused for human
participant surveys and for the LLM judge. The prompt maps item responses from
`1-5` onto `0.00-1.00` with `1->0.00`, `2->0.25`, `3->0.50`, `4->0.75`, and
`5->1.00`, then averages the three item scores within each dimension.

The benchmark also records:

- target duration,
- candidate duration,
- convergence status,
- iteration count,
- tool invocations,
- per-sample artifacts and aggregate markdown/JSON reports.

When a provider returns an explicit `429` cooldown window, `execute` now keeps
retrying the same sample after that wait instead of burning replacement slots on
repeated `startup_unavailable` outcomes. If the process is interrupted while the
sample is waiting on provider quota, the checkpoint marks that sample as safe to
retry in place on resume rather than converting it into a
`resume_interrupted_sample` replacement. The runner now also mirrors the
incremental draft mutations into checkpoint state so a quota-bound OpenAI sample
resumes from its saved partial draft on the next retry or resume, instead of
restarting the summarizer loop from an empty registry every time.

## Prerequisites

1. Install dependencies with `uv`.
2. Accept access to the gated Hugging Face dataset `blitt/SPoRC` if you will download it directly.
3. Set environment variables before running:

```bash
HF_ACCESS_TOKEN=...
VLLM_URL=http://host:8000/v1
CARD_FRAMEWORK_VLLM_MODEL=Qwen/Qwen3.5-27B
```

The repository now auto-loads the root `.env` file on package import when it exists, without overriding variables that are already set in the shell.

It also mirrors common aliases automatically, including:

- `HF_ACCESS_TOKEN` -> `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`, `HUGGINGFACE_TOKEN`
- `VLLM_URL` -> `CARD_FRAMEWORK_VLLM_URL`, `CARD_FRAMEWORK_VLLM_BASE_URL`,
  `VLLM_BASE_URL`

So for SPoRC benchmark runs, a root `.env` containing `HF_ACCESS_TOKEN`,
`VLLM_URL`, and `CARD_FRAMEWORK_VLLM_MODEL` is enough.

The benchmark runner also accepts the Hugging Face token through `HF_TOKEN`, so a common shell pattern is:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN"
```

If you use a local `.env`, make sure it contains at least:

```bash
HF_ACCESS_TOKEN=hf_xxx
VLLM_URL=http://host:8000/v1
CARD_FRAMEWORK_VLLM_MODEL=Qwen/Qwen3.5-27B
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
- `filter-audio-urls`
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
- one optional `--judge-only-source <path>` reuse pass that skips summarizer and
  critic, loads persisted `summary.xml` artifacts from prior execute runs or
  sweep logs, and reruns only the judge over those saved summaries.

Use `duration-sweep` when you want multiple duration targets in one invocation:

- `--mode single` runs one preset and uses `--duration-preset`,
- `--mode batch-all` runs multiple presets and uses `--duration-presets`,
- `--max-samples` controls how many retained samples each preset should finish with by default,
- `--sample-seed` switches sample selection from manifest-order prefixing to deterministic seeded-random ordering,
- `--examples` remains available as a sweep-only override for the same retained-sample target,
- `--parallelism 0` runs every selected preset concurrently,
- `--concurrency <int>` seeds that many worker children per active preset and donates those workers forward when earlier presets finish,
- when `--embedding-mode enabled`, `duration-sweep` now starts one parent-owned retrieval service for the whole sweep, namespaces each worker, and serializes embedding/index/retrieve requests so the configured embedding model loads once per sweep instead of once per child,
- embedding-enabled `execute` runs and `duration-sweep` children now reuse a persistent transcript-embedding cache under `artifacts/summarizer_critic_benchmark/embedding_cache`, keyed by sample ID, transcript content hash, and embedding-provider identity,
- stalled preset children now heartbeat every `30s`, write per-preset checkpoints, and auto-restart from checkpoint when the active sample stops making progress,
- checkpoint writes now use per-path serialization plus retry/backoff around the final atomic replace so transient Windows file locks do not abort a long sweep as easily,
- `duration-sweep` is transcript-only by default unless you add `--enable-voice-clone`.

Common flag mix-ups:

- `execute` does not accept a comma-separated list in `--duration-preset`.
- `--judge-only-source` belongs to `execute`; it is not a `duration-sweep`
  flag, and it reuses existing summary artifacts instead of launching new
  summarizer/critic work.
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

- it now supports `--provider anthropic` and `--provider vllm`; the vLLM path
  reuses the repo's OpenAI-compatible client and reads `VLLM_URL` from `.env`,
- it defaults to the same curated long-form manifest prefix, uses the duration
  preset as the primary local selection budget, and now accepts `30s`, `1m`,
  `5m`, `10m`, and `15m` presets,
- it sends the full transcript in one request per episode instead of windowing,
- it streams summarized thinking plus the structured sample payload to the
  terminal through the shared Rich callback path,
- it persists per-episode `request.json`, `stream.jsonl`, and `response.json`
  traces plus run-level JSONL/CSV exports under `sporc_llm_sampling`,
- every saved sample now includes bounded source anchors (`source_start_index`,
  `source_end_index`, timestamps, and `speaker_mode`), and each episode also
  writes a `selected_segments.txt` export for quick inspection,
- it currently reads `ANTHROPIC_API_KEY` from `.env` and defaults to
  `claude-opus-4-6` with adaptive thinking enabled,
- the structured tool payload is the authoritative output, but the sampler does
  not require exact verbatim transcript matching; instead it enforces bounded
  grounded reconstruction inside one cited transcript span per sample, preserves
  disfluencies and flow where possible, allows only local repair of transcript
  noise such as fragmentation, spacing, punctuation damage, and diarization
  seams, rejects false single-speaker flattening of cited multi-speaker
  spans, and now gives Anthropic two tools: `measure_selection_duration` for
  exact local duration accounting over already-chosen spans plus automatic
  `budget` feedback (`remaining_seconds_to_target`, `in_budget`, and
  `recommended_action`) and `submit_samples` for the final answer. The duration
  preset here is now the primary locally measured sampled-source budget, and
  the returned source metadata is still treated as best-effort and not
  transcript-cross-checked locally,
- if the last allowed tool round returns a locally valid measured selection
  that is already in-budget, the sampler now auto-promotes that last measured
  span set instead of failing purely because `submit_samples` did not arrive in
  time,
- if you pass `--samples <N>`, it ignores the WPM-based target-word formula and
  returns exactly `N` samples from each full transcript instead,
- if you pass `--text-only --samples <N>`, it ignores the duration preset for
  selection, reads the raw SPoRC speaker-turn data instead of the prepared
  transcript view, asks the model to return exact raw-turn spans only, locally
  merges adjacent same-speaker fragments inside a selected span into one
  coherent speaker line, and writes a per-episode `combined_segments.txt`
  artifact with the combined text-only output,
- if you pass `--transcribe-enabled`, `llm-sample` first localizes the source
  episode audio and runs the repo's stage-1 pipeline on it. That means source
  separation, Granite ASR by default, and the configured diarizer, then the
  resulting stage-1 transcript becomes the LLM's source instead of the dataset
  transcript. In this mode overlap-aware transcript JSON is preserved with
  single-speaker `segments` plus optional `overlap_speakers` metadata, and the
  sampler prompt now shows that overlap metadata to the model. This mode also
  fails fast if the local runtime downgrades diarization to `single_speaker`,
  because that would lose the overlap-preservation behavior,
- if `--dataset-manifest` points at an `episode_assets/prepared_episodes_manifest.json`
  summary produced by `prepare_episode_assets.py`, `llm-sample` now prefers the
  referenced full `source_transcript_path` / `prepared_full.transcript.json`
  input, only falls back to `transcript_path` or `sampled_transcript_path` when
  needed, and still skips stage-1 unless you also add `--transcribe-enabled`,
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
- auto-generated compact execution manifest sibling: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.compact.json`
- curated long-form audit: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_audit.json`
- budget-plan max-3-speaker manifest: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_max3_manifest.json`
- budget-plan max-3-speaker audit: `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_max3_audit.json`
- human sampling sessions: `artifacts/summarizer_critic_benchmark/sporc_human_sampling`
- automatic LLM sampling runs: `artifacts/summarizer_critic_benchmark/sporc_llm_sampling`
- shared embedding cache: `artifacts/summarizer_critic_benchmark/embedding_cache`
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
- `<run_id>/episodes/<sample_id>/combined_segments.txt` in `--text-only` mode
- `<run_id>/episodes/<sample_id>/stage1_source/transcript.json` plus localized
  source audio and audio-stage artifacts when `--transcribe-enabled` is used

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
active child connections are torn down. With embeddings enabled, those child
workers now share one retrieval service and one embedding model for the sweep;
worker namespaces prevent transcript collisions, and the shared service
serializes requests so only one embedding/index/retrieve operation runs at a
time.

To inspect one sweep shard end to end:

1. Open `sweeps/<sweep_id>/logs/<preset>.shard-<i>-of-<n>.log`.
2. Find the `run_start` event and note its child `run_id`.
3. Open `sporc_runs/<run_id>/report.json` or `sporc_runs/<run_id>/notes.md`.
4. Open `sporc_runs/<run_id>/samples/README.md` or `sporc_runs/<run_id>/samples/index.json`.
5. Drill into `sporc_runs/<run_id>/samples/<sample_id>/` for that shard's final outputs and raw LLM traces.

To rerun only the judge against a prior execute run or sweep:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc execute \
  --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_max3_manifest.compact.json" \
  --judge-provider deepseek_judge \
  --judge-only-source "artifacts/summarizer_critic_benchmark/sweeps/20260320T180920Z" \
  --output-root "artifacts/summarizer_critic_benchmark/sporc_runs" \
  --disable-stream-output
```

`--judge-only-source` accepts:

- an execute run root such as `artifacts/.../sporc_runs/<run_id>`
- an execute `report.json`
- a sweep root such as `artifacts/.../sweeps/<sweep_id>`
- a sweep `report.json`

Repeat `--judge-only-source` to combine multiple execute runs or sweeps into one
judge-only report. When the same `sample_id` appears multiple times across the
reused artifacts, the judge-only run appends the duration label to keep the new
sample artifact directories unique.

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
  --source-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/manifest.json" \
  --output-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_max3_manifest.json" \
  --audit-path "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_max3_audit.json" \
  --max-speakers 3
```

Those explicit output paths match the budget wrapper and the April 2026
under-`$50` benchmark plan.

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
older manifests that were prepared before that optimization. If you already
have a prepared or previously filtered manifest, point `--source-manifest` at
that manifest to stay on the metadata-first fast path rather than rescanning
raw SPoRC source files. Filtered manifest and audit outputs now also record the
selected manifest's `selected_max_unique_speaker_count` so later tooling can
read the current cap directly from the filtered dataset. Rerunning
`curate-longform` with the same source manifest, output paths, and speaker cap
reuses the existing filtered output instead of rescanning the source manifest.

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
- `curate-longform --max-speakers <N>`: keep only episodes with at most `N` unique speakers
- `dedupe-manifest --manifest-path <path>`: inline manifest to rewrite in place
- `dedupe-manifest --audit-path <path>`: optional dedupe audit JSON path
- `filter-audio-urls --manifest-path <path>`: manifest to validate against source-audio reachability
- `filter-audio-urls --output-manifest <path>`: optional filtered manifest output path; omit to rewrite in place
- `filter-audio-urls --audit-path <path>`: optional validation audit JSON path
- `filter-audio-urls --timeout-seconds <float>`: per-request timeout for remote HEAD/GET probes
- `filter-audio-urls --max-workers <int>`: maximum concurrent URL probes
- `filter-audio-urls --max-workers-per-host <int>`: host-level concurrency cap
- `filter-audio-urls --max-retries <int>`: transient retry count for remote probes
- `filter-audio-urls --retry-backoff-seconds <float>`: base exponential retry backoff
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

## Step 1.46: Filter A Manifest To Reachable Audio URLs

Keep only episodes whose `metadata.source_audio_url` or fallback `episode_key`
is still reachable. Inline manifests stay inline, compact manifests keep their
`samples_path` sidecar shape, and the command emits a sibling audit JSON by
default.

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc filter-audio-urls \
  --manifest-path "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --max-workers 8 \
  --max-workers-per-host 2
```

Or write the filtered result to a separate manifest explicitly:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc filter-audio-urls \
  --manifest-path "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.compact.json" \
  --output-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.filtered.compact.json" \
  --audit-path "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.filtered.audio_url_audit.json"
```

Important behavior:

- `filter-audio-urls` prefers `metadata.source_audio_url` and falls back to `episode_key`.
- it treats existing local files as valid and probes remote URLs with `HEAD`, then falls back to a tiny streamed `GET` when hosts reject `HEAD`.
- it validates remote URLs with bounded concurrency and per-host caps instead of one fully serial probe stream.
- it retries transient probe failures with backoff and caches duplicate URL results within the same run.
- it emits periodic validation progress with checked row counts, kept/removed counts, a compact progress bar, rolling rows-per-minute, and ETA, and those updates now continue across long stretches of valid rows.
- for compact manifests it now streams sidecar input and output rows instead of materializing the full sidecar in memory before filtering.
- it preserves the original sample order while removing failing rows.
- it rewrites manifest JSON and compact sidecars atomically.

## Step 1.47: Export Combined Raw Transcript Text

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

Use the raw text-only mode with a local vLLM server before spending Anthropic tokens:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc llm-sample \
  --provider vllm \
  --text-only \
  --samples 20 \
  --select-episode "[7,10]"
```

Rerun only a specific inclusive manifest span:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc llm-sample \
  --select-episode "[7,10]"
```

Important behavior:

- `llm-sample` is non-interactive and does not support resume yet.
- it fails fast when `ANTHROPIC_API_KEY` is missing for `--provider anthropic`
  or when `VLLM_URL` is missing for `--provider vllm`.
- `--samples <N>` overrides the WPM-based target-word mode.
- `--text-only` requires `--samples <N>`.
- `--select-episode "[start,end]"` uses 1-based inclusive manifest positions and overrides `--episodes`.
- `--disable-thinking` disables Anthropic thinking and uses `temperature=0.0`.
- returned source indices, timestamps, and speaker metadata are treated as best-effort model output.
- it is one-shot per episode: one invalid or malformed structured submission is
  recorded as failed immediately instead of triggering a correction retry.

## Step 1.7: Reuse Sampled Segments As Stage-2 Transcripts

Build one sampled transcript per selected episode and optionally hand each one
to the normal `setup_and_run` wrapper starting at stage-2:

```bash
uv run python scripts/run_sampled_transcript_pipeline.py \
  --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --llm-sample-run-root "artifacts/summarizer_critic_benchmark/sporc_llm_sampling/<run_id>"
```

Or skip the interactive menu and prepare only one explicit manifest slice:

```bash
uv run python scripts/run_sampled_transcript_pipeline.py \
  --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --selected-samples-path "artifacts/summarizer_critic_benchmark/sporc_llm_sampling/<run_id>/selected_samples.jsonl" \
  --select-episodes "7-10" \
  --prepare-only
```

Important behavior:

- the helper shows `10` sampled episodes per page by default and truncates long titles in the menu.
- selections use manifest positions, so the numbering stays aligned with the prepared manifest and the existing `llm-sample` range semantics.
- it writes one sampled transcript JSON per selected episode under `artifacts/sampled_transcript_pipeline/<run-id>/episodes/.../inputs/`.
- sampled transcript timings are synthetic sequential timings derived from sampled word counts, not original transcript timestamps.
- it preserves useful metadata from the source transcript, including `source_audio_path` and `speaker_samples_manifest_path` when those already exist.
- when the source transcript does not already carry `metadata.speaker_samples_manifest_path`, the helper still produces the stage-2 transcript, but the downstream runtime may need speaker-sample bootstrap work if voice cloning or the interjector stay enabled.
- `--prepare-only` is the safest mode when you want to inspect the sampled transcripts first or when the downstream audio-stage prerequisites are not already present.

## Step 1.8: Prepare Source Audio And Speaker Samples For Sampled Segments

When the sampled-segment `.txt` files already exist under
`src/card_framework/benchmark/summarizer_critic/sporc/sampled_segments/`, you can
prepare one per-episode folder with:

- a copied sampled-segment text file,
- a local source-audio copy or download,
- a prepared transcript whose metadata points at that local audio and the new
  speaker-sample manifest,
- speaker samples bootstrapped from the localized episode audio via fresh source
  separation and NeMo diarization, then remapped onto the reusable transcript
  speaker IDs.

Run:

```bash
uv run python scripts/run_sampled_segment_speaker_prep.py \
  --dataset-manifest /abs/path/to/benchmark_longform_15k_manifest.json
```

Notes:

- the helper matches each sampled-segment file by its `Sample ID:` header first
  and falls back to normalized episode-title matching only when needed,
- output defaults to
  `src/card_framework/benchmark/summarizer_critic/sporc/sampled_segments/episodes/`,
- it now bootstraps fresh speaker samples from the localized episode audio so
  the sample timings come from new diarization output instead of the prepared
  SPoRC transcript timings, without rerunning ASR for this prep path,
- the prep flow still preserves the reusable transcript speaker IDs by remapping
  the bootstrap manifest labels back onto the prepared transcript labels before
  later voice-clone stages consume them,
- speaker clipping respects `audio.speaker_samples.source_audio`, so this prep
  flow can still force clipping against the localized source MP3 when desired,
- when the configured diarizer is NeMo-derived but the local host cannot import
  or support NeMo, this bootstrap path now auto-delegates to the repo's Docker
  Compose runtime instead of failing after local separation work.

## Step 1.75: Prepare Episode Audio And Speaker Samples For Local Sampled Segments

If you already have plain-text sampled segment files under
`src/card_framework/benchmark/summarizer_critic/sporc/sampled_segments/`, you
can match them back to the prepared SPoRC manifest, download each episode's
source MP3, and generate the speaker-sample manifest the main pipeline expects:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets
```

Useful overrides:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets \
  --segments-dir "src/card_framework/benchmark/summarizer_critic/sporc/sampled_segments" \
  --dataset-manifest "artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full/benchmark_longform_15k_manifest.json" \
  --output-root "src/card_framework/benchmark/summarizer_critic/sporc/sampled_segments/episode_assets" \
  --force
```

Important behavior:

- the helper reads the `Episode:` header from each sampled-segment `.txt` file and uses `Sample ID:` when present for an exact manifest match.
- duplicate sampled-segment files for the same SPoRC sample are grouped into one episode asset folder.
- each prepared episode folder contains copied sampled-segment files, downloaded source audio, a local full transcript annotated with `source_audio_path` and `speaker_samples_manifest_path`, plus the generated speaker-sample manifest and WAV clips.
- the helper now bootstraps fresh speaker samples from the downloaded episode audio via source separation and NeMo diarization, then remaps the resulting speaker labels back onto the reusable SPoRC transcript speaker IDs.
- overlap-heavy speaker regions are skipped for sampling by subtracting any simultaneous speaker activity before the final sample slices are selected, so samples can end up shorter than `30s` when that is the cleanest option.
- if the configured diarizer is NeMo-derived but the local runtime cannot import NeMo, the helper auto-delegates only this diarization-first speaker-sample bootstrap to `docker compose` using the repo's `setup-and-run` or `setup-and-run-gpu` service. Keep Docker Desktop and the repo Compose stack available on Windows hosts.

## Step 1.76: Build Final Human-Eval Audio From Episode Assets

When `sampled_segments/episode_assets/prepared_episodes_manifest.json` already
exists, you can run one end-to-end workflow that:

- reads each episode's full `prepared_full.transcript.json`,
- asks the LLM to choose relevant contiguous source spans under a `5m`, `10m`,
  or `15m` duration target,
- tracks estimated Anthropic spend from reported token usage because Anthropic
  does not expose remaining prepaid credits through the standard API, and
- cuts one final concatenated source-audio file per episode from the localized
  `source_audio_path`.

Recommended command for five local episode-assets bundles:

```bash
uv run python scripts/run_episode_audio_slicing.py \
  --dataset-manifest "src/card_framework/benchmark/summarizer_critic/sporc/sampled_segments/episode_assets/prepared_episodes_manifest.json" \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --duration-preset 10m \
  --episodes 5 \
  --audio-format mp3 \
  --disable-thinking \
  --max-estimated-cost-usd 20
```

Important behavior:

- unlike the older episode-assets compatibility path, the audio-slicing helper now prefers `source_transcript_path` over `sampled_transcript_path`, so the LLM always sees the full episode transcript instead of a pre-sliced transcript.
- no transcript truncation is applied in this workflow; the LLM receives the full prepared transcript for each selected episode.
- each run writes live `run_summary.json`, `selected_samples.jsonl`, `selected_samples.csv`, and an `experiments.md` log under `artifacts/summarizer_critic_benchmark/sporc_episode_audio_slices/<run_id>/`.
- each episode folder under that run root keeps the raw sampler request and response artifacts plus the final rendered audio file and a `selected_audio_manifest.json` sidecar.
- the estimated Anthropic spend is best-effort and comes from the response `usage` fields plus the current model-pricing snapshot; Anthropic's Admin reporting APIs are separate and do not provide a real-time remaining-balance endpoint for a normal API key.

## Step 2: Run One Benchmark

### Budget-plan wrapper

For the April 2026 under-`$50` plan, use the repo wrapper instead of typing the
four `execute` commands by hand.

For non-validate runs, the wrapper now starts provider steps concurrently by
default, streams verbose child output plus progress/heartbeat events to the
terminal, keeps retrying retryable connection/startup failures until the
provider comes back, restarts a provider from scratch when retryable infra
replacements would otherwise change the retained sample cohort, and persists a
wrapper-session UUID so interrupted runs can resume unfinished providers from
their execute checkpoints. Resume also rehydrates missing provider report
metadata from the execute output roots, so healthy partial providers continue
from the cancelled checkpoint while providers whose discovered report only shows
retryable startup/connection failures are restarted from a fresh checkpoint.
When a provider returns a `429` with an explicit `try again in ...` hint, the
execute retry loop now waits for that cooldown before retrying the same sample
so funded runs do not waste retries against the same TPM window. The wrapper
also refreshes `_sessions/<uuid>/budget_profile.json` and prints a compact
session budget line from the persisted `call_XXX.json` traces so you can profile
visible-token lower-bound spend across contestants and the shared judge during a
live run.

Validate prerequisites and print the exact commands without spending money:

This pre-funding validate mode still exits cleanly before the wallets are
funded: missing live-run API keys and the later `vllm_default` model pin are
reported as later requirements instead of blocking the validation pass.

```powershell
uv run python scripts/run_sporc_budget_sweep.py --mode validate --skip-vllm-server-check
```

Run a local vLLM dry run before the funded sweep:

```powershell
$env:VLLM_URL="http://127.0.0.1:8000/v1"
$env:CARD_FRAMEWORK_VLLM_MODEL="Qwen/Qwen3.5-27B"
uv run python scripts/run_sporc_budget_sweep.py --mode dry-run
```

Resume an interrupted wrapper session from its printed UUID:

```powershell
uv run python scripts/run_sporc_budget_sweep.py --resume-session-id <uuid>
```

Force provider runs back to serial mode when needed:

```powershell
uv run python scripts/run_sporc_budget_sweep.py --mode live --max-concurrent-providers 1
```

Run the full funded sweep after the provider wallets are ready:

```powershell
uv run python scripts/run_sporc_budget_sweep.py --mode live
```

The wrapper enforces the documented budget defaults:

- `5m` preset only
- `10` retained samples
- `sample_seed=42`
- `max_iterations=0`
- transcript-only path
- contestants: `openai_gpt_5_4_mini`, `anthropic_claude_haiku_4_5`, `deepseek_chat`, `vllm_default`
- judge: `google_gemini_3_1_pro_preview_judge`

### Ablation-study wrapper

Use `scripts/run_sporc_ablation_study.py` for the IEEE ablation table. It writes
a run-local provider profile file, cell-local configs, per-cell execute logs,
the normal SPoRC sample artifacts, an aggregate JSON/Markdown report, and an
optional LaTeX section for `research/EEE_199_IEEE`.

Default protocol:

- `5m` preset, `20` retained seeded-random samples per cell, `sample_seed=42`
- transcript-only path with embeddings disabled
- ablations: full loop, no persisted loop memory, and no critic
- Qwen/Qwen3.6-27B through vLLM with explicit thinking and non-thinking matched
  summarizer/critic conditions
- no vLLM `max_tokens` cap unless you pass a positive `--vllm-max-tokens`
- DeepSeek V4 Pro with thinking disabled as the judge
- factualness as the primary automatic endpoint; naturalness, speaker-grammar
  similarity, and composite as exploratory diagnostics
- live wrapper progress with a sample/cell progress bar, heartbeat updates, and
  ETA smoothed by `--eta-learning-rate`

Plan without running:

```powershell
uv run python scripts/run_sporc_ablation_study.py --mode plan
```

Run the ablation study and refresh the paper section:

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:VLLM_URL="http://127.0.0.1:8000/v1"
$env:CARD_FRAMEWORK_VLLM_MODEL="Qwen/Qwen3.6-27B"
uv run python scripts/run_sporc_ablation_study.py --mode run --update-paper
```

Use `--eta-learning-rate 0.25` to keep the ETA stable, or increase it when the
latest sample duration should affect the estimate more strongly.

Use `--thinking-grid factorial` if the paper needs the cross-role interaction
cells (`summarizer thinking` with `critic non-thinking`, and vice versa).

### Max-iteration sweep wrapper

Use `scripts/run_sporc_max_iter_sweep.py` when you want a quick curve showing
how the judge scores change as the critic-feedback budget increases. The wrapper
defaults to `--max-iterations-list 1,2,3,4,5` and `--max-samples 1`, so a smoke
run executes one retained seeded-random sample per max-iteration value.

```powershell
$env:DEEPSEEK_API_KEY="..."
$env:VLLM_URL="http://127.0.0.1:8000/v1"
$env:CARD_FRAMEWORK_VLLM_MODEL="Qwen/Qwen3.6-27B"
uv run python scripts/run_sporc_max_iter_sweep.py --mode run
```

Each run writes `max_iter_sweep_report.json`, `max_iter_sweep_report.md`,
`max_iter_sweep.csv`, and `max_iter_sweep.svg` under
`artifacts/summarizer_critic_benchmark/sporc_max_iter_sweeps/<timestamp>/`.
Use `--max-samples` to increase the per-cell retained sample count after the
shape of the curve looks worth funding.

The wrapper now treats `--max-iterations` as a critic-feedback budget instead of
the old summarize+critic pair count:

- `0` = summarizer only
- `1` = summarizer -> critic -> summarizer
- `2` = summarizer -> critic -> summarizer -> critic -> summarizer

The wrapper accepts `0` directly and keeps negative values invalid so the
budget-safe wrapper never turns a typo into an unbounded execute run.

If a funded wrapper run already stopped at `max_iterations_reached`, you can now
salvage the missing extra summarizer pass plus judge rerun in place without
relaunching the whole execute loop:

```powershell
uv run python scripts/run_sporc_post_critic_revision.py --session-id <uuid>
```

That helper reuses the saved `summary.xml`, `critic_feedback.txt`, and sample
artifacts from the existing execute report, runs exactly one more
critic-guided summarizer pass with `max_iterations=0`, reruns only the judge,
and rewrites the existing report artifacts in place.

When `vllm_default` is pinned to `Qwen/Qwen3.5-27B`, the benchmark prep keeps
vLLM thinking enabled for that local contestant instead of forcing it off.

It also fails fast if you try to use `deepseek-reasoner` as the SPoRC
summarizer or critic, because DeepSeek's docs mark that model as not supporting
function calling.

Recommended transcript-only run:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" CARD_FRAMEWORK_VLLM_MODEL="$CARD_FRAMEWORK_VLLM_MODEL" uv run python -m card_framework.benchmark.summarizer_critic.sporc execute \
  --prepare-if-missing \
  --summarizer-provider vllm_default \
  --critic-provider vllm_default \
  --judge-provider vllm_default \
  --embedding-mode disabled \
  --transcript-only
```

Run with a fixed duration preset:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" CARD_FRAMEWORK_VLLM_MODEL="$CARD_FRAMEWORK_VLLM_MODEL" uv run python -m card_framework.benchmark.summarizer_critic.sporc execute \
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
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" CARD_FRAMEWORK_VLLM_MODEL="$CARD_FRAMEWORK_VLLM_MODEL" uv run python -m card_framework.benchmark.summarizer_critic.sporc execute \
  --prepare-if-missing \
  --transcript-only \
  --target-seconds-override 90 \
  --max-samples 10
```

Run the faithful audio path instead of transcript-only:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" CARD_FRAMEWORK_VLLM_MODEL="$CARD_FRAMEWORK_VLLM_MODEL" uv run python -m card_framework.benchmark.summarizer_critic.sporc execute \
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
- `execute` and `duration-sweep` now auto-compact large inline manifests into sibling `*.compact.json` plus `*.compact.samples.jsonl.gz` files before heavy worker startup, so repeated benchmark children do not all reparse the original inline `samples` array.
- replacement-eligible outcomes are infra-only: `preflight_failed`, `startup_unavailable`, `connection_unavailable`, `stage_timeout`, and resume/watchdog orphan failures.
- startup or connection-unavailable failures retry the same sample up to two additional times before that sample is excluded and replaced by the next deterministic candidate.
- normal benchmark failures after the sample actually ran still count toward the retained target.
- sweep workers coordinated through the shared SQLite queue now load the claimed sample payload from SQLite on demand instead of materializing the full prepared dataset in every child process.
- direct `execute` runs launched from a real terminal now stream live markdown-rendered panels for the Summarizer, Critic, and Judge, and each panel appears as soon as that agent request starts so long-prefill calls no longer look idle; piped or supervised runs still fall back to plain streamed `[THINKING]` and `[CONTENT]` text.
- `--duration-preset` and `--target-seconds-override` are mutually exclusive.

## Step 3: Run A Duration Sweep

Show the non-interactive menu/help text:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep
```

Run one preset for `10` examples:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" CARD_FRAMEWORK_VLLM_MODEL="$CARD_FRAMEWORK_VLLM_MODEL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
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
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" CARD_FRAMEWORK_VLLM_MODEL="$CARD_FRAMEWORK_VLLM_MODEL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
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
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" CARD_FRAMEWORK_VLLM_MODEL="$CARD_FRAMEWORK_VLLM_MODEL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
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
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" CARD_FRAMEWORK_VLLM_MODEL="$CARD_FRAMEWORK_VLLM_MODEL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
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
- `--duration-presets 30s,1m,5m,10m,15m`
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
- a preset reports `completed` only after it fills the requested retained target; underfilled presets now stay `partial`, and a nonzero child exit no longer reuses a stale `completed` report snapshot.

## Running A Custom Manifest

If you already prepared or curated a subset manifest, pass it directly:

```bash
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" CARD_FRAMEWORK_VLLM_MODEL="$CARD_FRAMEWORK_VLLM_MODEL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
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
- `execute`: `--duration-preset <30s|1m|5m|10m|15m>`
- `execute`: `--target-seconds-override <int>`
- `duration-sweep`: `--max-samples <int>` where `0` means all prepared examples per preset and positive values mean retained samples per preset
- `duration-sweep`: `--sample-seed <int>` for deterministic seeded-random per-preset sample selection
- `duration-sweep`: `--examples <int>` as a sweep-only override
- `duration-sweep`: `--parallelism <int>`
- `duration-sweep`: `--concurrency <int>`
- `duration-sweep`: `--duration-presets 30s,1m,5m,10m,15m`

`--embedding-mode enabled` uses the active embedding profile from config rather
than a hard-coded benchmark-only model. In the repository defaults that profile
is `qwen3_4b`, so enabling embeddings now points SPoRC sweeps at
`Qwen/Qwen3-Embedding-4B` unless you change `selection.embedding`. The default
Qwen profile is tuned for the measured Windows RTX 3060 sweet spot:
`batch_size=1`, `max_seq_length=384`, and `attention_implementation=eager`.

`--max-iterations` semantics now mean critic-feedback budget:

- `0` keeps one summarizer pass and skips critic calls entirely
- `1` runs one critic pass and then one final critic-guided summarizer revision
- `2+` keeps alternating critic and summarizer until the budget is spent
- negative values still map to unbounded only on the lower-level `execute`
  contract; the budget wrapper does not allow them

## Sweep Analysis

To analyze an in-flight or completed duration sweep, run:

```powershell
uv run python scripts/analyze_sporc_sweep.py `
  --sweep-root artifacts/summarizer_critic_benchmark/sweeps/<sweep_id>
```

The helper emits five PNG graphs plus CSV, Markdown, and JSON summaries under
`<sweep>/analysis/<timestamp>/` so you can inspect worker timelines, stage
concurrency, observed worker-time shares, failure waste, and transcript
complexity versus wall time from the same sweep artifact set. It also prints
timed `Loading ...` / `Loaded ... in X.XX seconds.` phase updates while it
parses logs, loads checkpoints, computes transcript complexity, renders graphs,
and writes summaries, so long analyses do not leave the terminal blank.

## Timeouts

Stage timeout flags are per-request network timeouts, not overall benchmark limits.
Explicit values above the default `30s` request timeout are honored, and the
shared retrieval service now keeps its health endpoint responsive while slow
index or retrieve work is running in the background thread pool.

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
HF_TOKEN="$HF_ACCESS_TOKEN" VLLM_URL="$VLLM_URL" CARD_FRAMEWORK_VLLM_MODEL="$CARD_FRAMEWORK_VLLM_MODEL" uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
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
- child exits with connection refused or reset errors: verify the effective `VLLM_URL` or `CARD_FRAMEWORK_VLLM_BASE_URL`; SPoRC benchmarks honor repo `.env` overrides, so a dead remote endpoint can replace the tracked local provider URL.
- slow backend: reduce sample count first, then tune stage timeouts and/or distribute separate runs across multiple `VLLM_URL` endpoints.
