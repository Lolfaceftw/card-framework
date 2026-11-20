# SPoRC Benchmark

Runs CARD's summarizer and critic loop on long, multi-speaker podcast episodes
from [SPoRC](https://huggingface.co/datasets/blitt/SPoRC), then has an LLM judge
score each summary.

There are two modes:

| Mode | What runs | Use it for |
| --- | --- | --- |
| `transcript-only` | Summarizer and critic on the transcript text | Measuring summary quality (recommended) |
| `voice-clone` | The above plus audio download, speaker samples, and voice-cloned drafts | Testing the full audio path |

`execute` defaults to `voice-clone`; `duration-sweep` defaults to
`transcript-only`.

Commands below use bash syntax. In PowerShell, set variables with
`$env:NAME = "..."`, replace the trailing `\` with a backtick, and quote
comma-separated values such as `"30s,1m,5m"`.

**Contents**

- [Before you start](#before-you-start)
- [Workflow](#workflow): prepare, curate, run, sweep, read results
- [Building human-evaluation material](#building-human-evaluation-material)
- [Reference](#reference): flags, timeouts, output layout, sweep internals
- [Troubleshooting](#troubleshooting)

## What gets scored

The judge rates each summary from `0.0` to `1.0` on three dimensions:

| Dimension | Question |
| --- | --- |
| Factualness | Does the summary stay faithful to the transcript without losing important information? |
| Naturalness | Does it flow in order and read like a coherent spoken segment? |
| Speaker Grammar Similarity | Does each speaker keep their own grammar, cadence, filler words, and personality? |

Each dimension is the average of three 1–5 Likert items, mapped to
`1→0.00, 2→0.25, 3→0.50, 4→0.75, 5→1.00`. Human raters can answer the same
items. The judge sees the full transcript. The prompt is in
`src/card_framework/prompts/templates/conversation_summary_judge_*.jinja2`.

Each run also records target and actual duration, whether the loop converged,
the number of iterations, and tool calls.

In `transcript-only` mode the runner first cleans a copy of the transcript:
it drops punctuation-only fragments and merges consecutive turns by the same
speaker. Where the speaker is unclear, it marks the turn rather than guessing,
so speaker labels stay as the dataset has them.

## Before you start

1. Install dependencies: `uv sync --dev`.
2. On Hugging Face, accept the terms for the gated dataset
   [`blitt/SPoRC`](https://huggingface.co/datasets/blitt/SPoRC).
3. Put these in `.env` at the repository root:

   ```dotenv
   HF_TOKEN=hf_...                      # Hugging Face token with SPoRC access
   VLLM_URL=http://your-host:8000/v1    # your OpenAI-compatible LLM server
   ANTHROPIC_API_KEY=                   # only needed for llm-sample
   ```

CARD loads `.env` automatically, and variables already set in your shell take
priority. `HF_ACCESS_TOKEN` also works in place of `HF_TOKEN`, and
`CARD_FRAMEWORK_VLLM_URL`, `CARD_FRAMEWORK_VLLM_BASE_URL`, or `VLLM_BASE_URL` in
place of `VLLM_URL`.

All commands go through one entry point:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc <subcommand>
```

| Subcommand | Purpose |
| --- | --- |
| `prepare-dataset` | Download SPoRC and build a manifest |
| `curate-longform` | Keep only long, multi-speaker episodes |
| `dedupe-manifest` | Remove duplicate episodes from a manifest |
| `filter-audio-urls` | Remove episodes whose audio is no longer reachable |
| `execute` | Run one benchmark configuration |
| `duration-sweep` | Run several target durations in one go |
| `supervise` | Run `execute` with RAM and GPU limits |
| `export-transcripts` | Write selected episodes' transcripts to one text file |
| `human-sample` | Have a person pick transcript segments |
| `llm-sample` | Have an LLM pick transcript segments |

In the commands below, `$PREPARED` stands for
`artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full`.

## Workflow

### 1. Prepare the dataset

```bash
# Small sample split, good for trying things out
uv run python -m card_framework.benchmark.summarizer_critic.sporc prepare-dataset --dataset-variant sample

# Full split
uv run python -m card_framework.benchmark.summarizer_critic.sporc prepare-dataset --dataset-variant full
```

The two splits write to separate folders (`prepared_sample` and
`prepared_full`), so they don't overwrite each other. The full split stores its
rows in `samples.jsonl.gz` beside `manifest.json` and streams them, so it
doesn't need the whole dataset in memory. Status lines such as
`Loaded ... in X.XX seconds.` show progress after the download bars finish.

Already have the gated files locally? Skip the download:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc prepare-dataset \
  --episode-data-path /path/to/episodeLevelData.jsonl.gz \
  --turn-data-path /path/to/speakerTurnData.jsonl.gz
```

| Flag | Meaning |
| --- | --- |
| `--dataset-variant sample\|full` | Which split to prepare |
| `--prepared-root <path>` | Custom output folder |
| `--reference-wpm <n>` | Speaking rate used to derive target durations |
| `--force-download` | Download the source files again |

If Hugging Face warns about `hf_xet`, installing it only speeds up downloads. It
doesn't change the results.

### 2. Curate long-form episodes

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc curate-longform \
  --source-manifest "$PREPARED/manifest.json" \
  --max-speakers 3
```

An episode is kept only if it passes every check:

- 15,000 or more spoken words, not counting music and noise markers
- 120 or more transcript segments
- at least 2 speakers, and at least 2 who each speak 1,000+ words over 20+ segments
- no single speaker has more than 90% of the words
- no more than 10% non-word tokens
- no more speakers than `--max-speakers`, if given

Output goes to `$PREPARED/benchmark_longform_15k_manifest.json` and a matching
`..._audit.json` that says why each episode was kept or dropped. Use
`--output-manifest` and `--audit-path` to change these paths.

Rerunning with the same inputs reuses the existing output. Curation reads
statistics saved during `prepare-dataset`. It only reopens transcripts for
manifests prepared by older versions.

### 3. Clean up the manifest (optional)

**Remove duplicates.** This rewrites the manifest in place and keeps the first
copy of each episode, in the original order:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc dedupe-manifest
```

By default it targets the curated manifest. Pass `--manifest-path` and
`--audit-path` to target another. It only works on manifests that list samples
inline, which is what `curate-longform` produces.

**Remove episodes whose audio is gone.** Useful before `voice-clone` runs, since
those download the episode audio:

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc filter-audio-urls \
  --manifest-path "$PREPARED/benchmark_longform_15k_manifest.json" \
  --max-workers 8 \
  --max-workers-per-host 2
```

It checks `metadata.source_audio_url` (or `episode_key` if that's missing) with
a `HEAD` request, falling back to a small `GET`. Local files count as reachable.
It rewrites the manifest in place unless you pass `--output-manifest`, and keeps
the original order.

| Flag | Meaning |
| --- | --- |
| `--output-manifest <path>` | Write to a new file instead of in place |
| `--audit-path <path>` | Where to write the audit JSON |
| `--timeout-seconds <s>` | Timeout per request |
| `--max-workers <n>` | Total concurrent checks |
| `--max-workers-per-host <n>` | Concurrent checks per host |
| `--max-retries <n>` | Retries for temporary failures |
| `--retry-backoff-seconds <s>` | Base delay between retries |

### 4. Run one benchmark

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc execute \
  --prepare-if-missing \
  --transcript-only \
  --summarizer-provider vllm_default \
  --embedding-mode disabled \
  --duration-preset 1m \
  --max-samples 10
```

The critic defaults to the summarizer's provider, and the judge to the critic's.

Variations:

- **Exact target length:** replace `--duration-preset 1m` with
  `--target-seconds-override 90`. You can't use both.
- **Full audio path:** drop `--transcript-only`. The run then also downloads the
  episode audio, builds speaker samples, and voice-clones live drafts.
- **Long presets (`5m`, `15m`):** add
  `--dataset-manifest "$PREPARED/benchmark_longform_15k_manifest.json"` so
  episodes are long enough.
- **RAM and GPU limits:** use `supervise` instead of `execute`.

How `--max-samples` counts: it is the number of samples that must finish. If a
sample fails because of infrastructure, the runner retries it up to twice and
then replaces it with the next one. Infrastructure failures are
`preflight_failed`, `startup_unavailable`, `connection_unavailable`,
`stage_timeout`, and watchdog or resume orphans. A sample that runs and then
scores badly still counts.

### 5. Run a duration sweep

A sweep runs the same benchmark at several target lengths. Built-in presets
are `30s`, `1m`, `5m`, `10m`, and `15m`.

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep \
  --mode batch-all \
  --duration-presets "30s,1m,5m" \
  --dataset-manifest "$PREPARED/benchmark_longform_15k_manifest.json" \
  --max-samples 100 \
  --parallelism 0 \
  --summarizer-provider vllm_default \
  --embedding-mode disabled \
  --max-iterations 50
```

| Flag | Meaning |
| --- | --- |
| `--mode single` + `--duration-preset 5m` | Run one preset |
| `--mode batch-all` + `--duration-presets "..."` | Run several presets (all built-in ones if omitted) |
| `--max-samples <n>` | Samples to finish per preset; `0` means all |
| `--examples <n>` | Overrides `--max-samples` for this sweep only |
| `--parallelism <n>` | Presets running at once; `0` means all |
| `--concurrency <n>` | Worker processes per running preset |
| `--sample-seed <n>` | Reproducible random sample order |
| `--enable-voice-clone` | Use the full audio path instead of transcript-only |
| `--sweep-root <path>` | Custom output folder |
| `--child-stall-seconds <s>` | Restart a worker after this long without progress |
| `--max-child-restarts <n>` | Warn after this many restarts |
| `--progress-heartbeat-seconds <s>` | How often workers report progress |

Running `duration-sweep` with no arguments prints its help.

When a preset finishes, its workers move on to the next unfinished preset, so
later presets speed up. A preset is marked `completed` only once it reaches
`--max-samples`; otherwise it stays `partial`. `Ctrl+C` stops every worker
cleanly before exiting.

### 6. Read the results

Every `execute` run writes to `artifacts/summarizer_critic_benchmark/sporc_runs/<run_id>/`:

- `results.md`: start here. Markdown tables with 95% confidence intervals,
  plus a composite-score column per preset for sweeps.
- `notes.md`: the sample target, how many samples were tried, and how many were
  replaced.
- `report.json`: per-sample metrics, judge scores, convergence, per-speaker
  messages, and artifact paths.
- `samples/`: raw per-sample output. `samples/README.md` explains the layout,
  and every LLM call is saved under `samples/<sample_id>/llm/` with its request,
  streamed tokens, and response.

To graph a sweep (in progress or finished):

```bash
uv run python scripts/analyze_sporc_sweep.py \
  --sweep-root artifacts/summarizer_critic_benchmark/sweeps/<sweep_id>
```

This writes five PNG charts plus CSV, Markdown, and JSON summaries to
`<sweep>/analysis/<timestamp>/`. They cover worker timelines, stage concurrency,
time spent on failures, and transcript complexity against wall time.

### Recommended setup for comparing summarizers

1. Prepare and curate the dataset.
2. Use `transcript-only`.
3. Keep `--embedding-mode disabled` unless you're testing retrieval.
4. Fix the duration with `--duration-preset`.
5. Use the same manifest and `--max-samples` (and `--sample-seed`, if used)
   for every run.
6. Compare `results.md` and `report.json` across runs.

## Building human-evaluation material

These optional tools pick transcript excerpts and prepare matching audio for
listening studies. They aren't needed to run the benchmark.

### Export transcripts as text

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc export-transcripts --episodes "7,10-12"
```

`--episodes` takes positions in the manifest: `7,10-12`, or `[7,10]` for an
inclusive range. Add `--raw` to write the dataset's original JSONL rows with no
formatting. Add `--output-path` to choose the file, or `--manifest-path` to use
a different manifest. `--backwards-compat` reads positions from the curated
manifest and matches them by `episode_key`, so old position numbers still work
after you switch manifests.

### Pick segments by hand: `human-sample`

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc human-sample \
  --duration-preset 5m \
  --episodes 20
```

It takes the first `--episodes` episodes from the curated manifest and splits
each transcript into windows with line numbers. In each window, type the line
numbers you want:

| Input | Effect |
| --- | --- |
| `12` or `12,13` | Select those lines |
| `12,/override [SPEAKER_04] New text` | Select line 12 and add a rewritten segment |
| `"/override [SPEAKER_04] Text, with comma"` | Quote overrides that contain commas |
| `/skip` | Skip this window |

How many words to pick per episode is
`words-per-minute × preset minutes × sample-factor` (default 170 WPM, factor 5).
Change it with `--words-per-minute`, `--sample-factor`, and
`--max-windows-per-episode`.

Results are saved after every window. Press `Ctrl+C` to pause, and continue
later with `--resume-session <session-id>`. Each session writes
`session.sqlite3`, `session_summary.json`, and `selected_samples.jsonl`/`.csv`
under `sporc_human_sampling/<session_id>/`.

### Pick segments with an LLM: `llm-sample`

```bash
uv run python -m card_framework.benchmark.summarizer_critic.sporc llm-sample \
  --duration-preset 5m \
  --episodes 20
```

The model reads each full transcript and returns spans that fit the preset
length. It has a tool to measure the duration of its picks before submitting.
The defaults are Anthropic and `claude-opus-4-6` with thinking on, which needs
`ANTHROPIC_API_KEY`. That model is deprecated in CARD. Newer Claude models work
with thinking on, but not with `--disable-thinking` (see
[Model status](../../../../../CONFIG.MD#model-status)).

| Flag | Effect |
| --- | --- |
| `--provider anthropic\|vllm` | Model provider. `vllm` uses `VLLM_URL`. |
| `--model <name>` / `--thinking-effort <level>` | Anthropic model and thinking effort |
| `--disable-thinking` | Turn thinking off and use temperature 0 |
| `--samples <n>` | Return exactly `n` samples per episode instead of using the word budget |
| `--text-only --samples <n>` | Pick exact turns from the raw speaker data and merge consecutive turns by the same speaker |
| `--select-episode "[7,10]"` | Only episodes 7 to 10 (1-based, inclusive) |
| `--transcribe-enabled` | Transcribe the episode audio with CARD first and sample from that transcript, which keeps overlapping-speech information |

Things to know:

- Each episode gets one attempt. An invalid answer is recorded as a failure,
  not retried. There is no resume.
- If the last tool round has a valid in-budget selection but the model never
  submitted, that selection is used.
- The model may lightly repair transcript noise such as broken punctuation and
  fragments, but every sample must stay within one cited transcript span.
  Source indices and timestamps are best-effort.
- With `--transcribe-enabled`, the run stops if diarization falls back to a
  single speaker.

Each run writes `run_summary.json` and `selected_samples.jsonl`/`.csv` under
`sporc_llm_sampling/<run_id>/`. Each episode gets an `episodes/<sample_id>/`
folder with the request, the stream, the response, and `selected_segments.txt`
(plus `combined_segments.txt` in `--text-only` mode).

### Turn picked segments into pipeline inputs

These helpers take sampled segments and prepare what the main pipeline needs:
a transcript, the source audio, and speaker voice samples.

**Sampled transcripts for stage 2.** Builds one transcript per episode from an
`llm-sample` run, then optionally runs `setup_and_run` from the summarization
stage:

```bash
uv run python scripts/run_sampled_transcript_pipeline.py \
  --dataset-manifest "$PREPARED/benchmark_longform_15k_manifest.json" \
  --llm-sample-run-root artifacts/summarizer_critic_benchmark/sporc_llm_sampling/<run_id> \
  --select-episodes "7-10" \
  --prepare-only
```

Without `--select-episodes`, an interactive menu lists 10 episodes per page.
`--prepare-only` writes the transcripts to
`artifacts/sampled_transcript_pipeline/<run-id>/` without running the pipeline.
Timings in these transcripts are synthetic, based on word counts. If the source
transcript has no speaker samples, voice cloning and interjection will need
them generated first.

**Audio and speaker samples from segment files.** Both helpers below read
plain-text segment files from
`src/card_framework/benchmark/summarizer_critic/sporc/sampled_segments/`. The
repository doesn't include any, so put yours there first. Each file needs an
`Episode:` header, and a `Sample ID:` header gives an exact match.

The helpers download each episode's audio, run source separation and NeMo
diarization to cut fresh voice samples, and map the speaker labels back to the
SPoRC transcript. Segments with overlapping speech are skipped, so samples can
be shorter than 30 seconds. If NeMo can't load on your machine, they run that
step in the Docker Compose service instead, so keep Docker available.

```bash
# Writes to sampled_segments/episodes/
uv run python scripts/run_sampled_segment_speaker_prep.py \
  --dataset-manifest /abs/path/to/benchmark_longform_15k_manifest.json

# Writes to sampled_segments/episode_assets/, grouping duplicate files per episode
uv run python -m card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets \
  --dataset-manifest "$PREPARED/benchmark_longform_15k_manifest.json" \
  --force
```

**Final listening clips.** Once
`sampled_segments/episode_assets/prepared_episodes_manifest.json` exists, this
asks an LLM to pick spans from each full transcript and cuts one audio file per
episode:

```bash
uv run python scripts/run_episode_audio_slicing.py \
  --dataset-manifest src/card_framework/benchmark/summarizer_critic/sporc/sampled_segments/episode_assets/prepared_episodes_manifest.json \
  --provider anthropic \
  --model claude-sonnet-4-6 \
  --duration-preset 10m \
  --episodes 5 \
  --audio-format mp3 \
  --disable-thinking \
  --max-estimated-cost-usd 20
```

It stops once the estimated Anthropic cost reaches `--max-estimated-cost-usd`.
The estimate comes from each response's token usage, because the API doesn't
report your remaining balance. **The price table only covers Claude 4.x models**
(`claude-sonnet-4-6`, the deprecated default, among them). With any other model
the estimate stays at 0 and the limit never triggers. Results go to
`artifacts/summarizer_critic_benchmark/sporc_episode_audio_slices/<run_id>/`.

## Reference

### Shared flags

| Flag | Meaning |
| --- | --- |
| `--dataset-manifest <path>` | Manifest to run on |
| `--prepare-if-missing` | Prepare the dataset first if needed |
| `--force-download-dataset` / `--force-download-audio` | Download again |
| `--summarizer-provider` / `--critic-provider` / `--judge-provider <id>` | Profiles from `provider_profiles.yaml` |
| `--embedding-mode disabled\|enabled` | Turn retrieval on or off |
| `--max-iterations <n>` | Maximum summarizer/critic rounds per sample |
| `--max-samples <n>` | Samples to finish; `0` means all |
| `--sample-seed <n>` | Reproducible random sample order |
| `--duration-preset <preset>` | `30s`, `1m`, `5m`, `10m`, or `15m` (`execute`) |
| `--target-seconds-override <n>` | Exact target length (`execute`) |
| `--output-root <path>` | Custom output folder |

`--embedding-mode enabled` uses the embedding profile set in
`selection.embedding` in the config. With `qwen3_4b`, it's tuned for an
RTX 3060: `batch_size=1`, `max_seq_length=384`, `attention_implementation=eager`.
Embeddings are cached in `artifacts/summarizer_critic_benchmark/embedding_cache`.

### Timeouts

| Flag | Applies to |
| --- | --- |
| `--summarizer-timeout-seconds` | Each summarizer request |
| `--critic-timeout-seconds` | Each critic request |
| `--retrieval-timeout-seconds` | Each retrieval request |
| `--index-timeout-seconds` | Each indexing request |

These limit a single request, not the whole run. `execute` has no timeout
unless you set one. `duration-sweep` defaults to 900 seconds for the summarizer
and critic and 300 seconds for retrieval and indexing, so one stuck request
fails a sample instead of hanging a preset.

### Several LLM servers

One benchmark process talks to one `VLLM_URL`. To spread work across servers,
start one process per server:

```bash
VLLM_URL=http://host:8000/v1 <command>
VLLM_URL=http://host:8001/v1 <command>
```

### Output layout

Under `artifacts/summarizer_critic_benchmark/`:

| Path | Contents |
| --- | --- |
| `datasets/sporc/prepared_sample/` | Sample split |
| `datasets/sporc/prepared_full/` | Full split, curated manifests, and audits |
| `sporc_runs/<run_id>/` | One `execute` run (see [Read the results](#6-read-the-results)) |
| `sweeps/<sweep_id>/` | Sweep `report.json`, `results.md`, `logs/`, and `checkpoints/` |
| `sporc_supervisor/` | Supervised sessions |
| `sporc_human_sampling/` | `human-sample` sessions |
| `sporc_llm_sampling/` | `llm-sample` runs |
| `embedding_cache/` | Cached transcript embeddings |

`execute` and `duration-sweep` convert large manifests to a compact form
(`*.compact.json` plus `*.compact.samples.jsonl.gz`) before starting workers.

### How sweeps work

Each preset runs one or more ordinary `execute` workers. Their detailed output
lands in `sporc_runs/`, and the sweep's combined report lands in `sweeps/`.

- Workers save a checkpoint after every sample and send a heartbeat every 30
  seconds. A stalled worker is restarted from its checkpoint, and only the
  sample it was on is marked `sweep_watchdog_stalled`.
- Workers in the same preset share a SQLite queue, so they never run the same
  sample twice.
- With embeddings on, the sweep starts one shared retrieval service, so the
  embedding model loads once rather than once per worker.

To follow one worker's run:

1. Open `sweeps/<sweep_id>/logs/<preset>.shard-<i>-of-<n>.log`.
2. Find the `run_start` event and note its `run_id`.
3. Open `sporc_runs/<run_id>/report.json`, then `samples/` for raw output.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Hugging Face access denied | Accept the SPoRC terms on Hugging Face and check `HF_TOKEN` |
| Manifest not found | Run `prepare-dataset`, or add `--prepare-if-missing` |
| Unexpected audio downloads | `execute` defaults to `voice-clone`; add `--transcript-only` |
| Connection refused or reset | Check the `VLLM_URL` that is actually in effect, since a value in `.env` overrides the default local URL |
| Empty or partial results | Read `notes.md`, `results.md`, and `report.json` in the run folder |
| Slow LLM server | Lower `--max-samples`, raise the timeouts, or spread runs across several servers |
| `execute` rejects `--duration-preset 1m,5m` | `execute` takes one preset; use `duration-sweep --duration-presets` for several |
