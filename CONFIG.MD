# Configuration

CARD reads its settings from one YAML file:

| You are running | Config file used |
| --- | --- |
| A source checkout | `conf/config.yaml` |
| A pip install | `src/card_framework/config/config.yaml` (bundled with the package) |
| Either, with a custom file | The path in `CARD_FRAMEWORK_CONFIG` |

Secrets never go in YAML. They go in a `.env` file or in environment variables.

## Quick start

1. Copy `.env.example` to `.env`.
2. Fill in the keys for the providers you use. Leave the rest empty.
3. Pick your models in the `selection` block of `conf/config.yaml`.

Leave the `profiles` block alone unless you are adding a model or changing a
base URL.

## Choosing models

Most changes happen in `selection`:

```yaml
selection:
  llm: "vllm_local"        # LLM profile used by every stage
  embedding: "disabled"    # retrieval embedding profile
  stage_llm:               # optional per-stage override
    summarizer: "default"  # "default" means: use selection.llm
    critic: "default"
    interjector: "default"
```

For example, to keep a local model for summarizing but use DeepSeek as the
critic, set `critic: "deepseek_reasoner"`.

### LLM profiles

| Profile | Model | Key in `.env` | Status |
| --- | --- | --- | --- |
| `vllm_local` | Any model served by an OpenAI-compatible vLLM server | `CARD_FRAMEWORK_VLLM_API_KEY` (only if the server needs one) | Current |
| `anthropic_claude_opus` | `claude-opus-4-6` | `ANTHROPIC_API_KEY` | **Deprecated** |
| `deepseek_chat` | `deepseek-chat` | `DEEPSEEK_API_KEY` | **Deprecated, likely not working** |
| `deepseek_reasoner` | `deepseek-reasoner` | `DEEPSEEK_API_KEY` | **Deprecated, likely not working** |
| `google_gemini` | `gemini-2.5-pro` | `GEMINI_API_KEY` | **Deprecated** |
| `glm_46` | `glm-4.6` | `ZAI_API_KEY` | **Deprecated** |
| `huggingface_toolace` | `Team-ACE/ToolACE-8B` via the Hugging Face router | `HF_TOKEN` | Not checked |
| `nanbeige_thinking` | `Nanbeige/Nanbeige4-3B-Thinking-2511` via the Hugging Face router | `HF_TOKEN` | Not checked |
| `transformers_local` | `Salesforce/Llama-xLAM-2-3b-fc-r`, run in-process | none | Not checked |

`vllm_local` reads its address from `CARD_FRAMEWORK_VLLM_BASE_URL`. For what
"Deprecated" means here, see [Model status](#model-status).

### Embedding profiles

Embeddings are only used for retrieval, which is off by default.

| Profile | Model | Device |
| --- | --- | --- |
| `disabled` | none | none |
| `qwen3_4b` | `Qwen/Qwen3-Embedding-4B` | CUDA |
| `minilm_cpu` | `sentence-transformers/all-MiniLM-L6-v2` | CPU |

### Model status

*Checked 2026-11-20.* "Deprecated" means CARD still ships the model as a
default, but a newer one exists and CARD will switch to it in a future
release. Every such line in the code and config is tagged `DEPRECATED(model)`,
so `git grep "DEPRECATED(model)"` finds them all.

| Model | Used by | Status | Replacement |
| --- | --- | --- | --- |
| `claude-opus-4-6` | `anthropic_claude_opus`, SPoRC `llm-sample` default | Still served by Anthropic; superseded | `claude-opus-5-5` |
| `claude-sonnet-4-6` | SPoRC audio-slicing default | Still served by Anthropic; superseded | `claude-sonnet-5-5` |
| `deepseek-chat` | `deepseek_chat`, summary-matrix default | DeepSeek announced this name would be discontinued on 2026-07-24 | `deepseek-v4-flash` |
| `deepseek-reasoner` | `deepseek_reasoner`, benchmark judge | Same as above | `deepseek-v4-flash` or `deepseek-v4-pro`, in thinking mode |
| `gemini-2.5-pro` | `google_gemini` | Closed to new Gemini API users; shutdown dates conflict | A Gemini 3 model |
| `glm-4.6` | `glm_46` | Still listed; superseded | GLM-5.x |

The open-weight models (ToolACE, Nanbeige, xLAM, Qwen, Granite) run locally or
through the Hugging Face router and were not checked.

#### Known issues with newer models

Changing the model name works for most profiles: CARD already passes back
Gemini thought signatures and DeepSeek/GLM `reasoning_content` in tool loops.
Fix these first:

- **Newer Claude models, thinking off.** `--disable-thinking` (in `llm-sample`
  and audio slicing) sends `temperature: 0`, and `llm-sample --text-only
  --disable-thinking` also forces the `submit_samples` tool.
  `claude-opus-5-5` and `claude-sonnet-5-5` reject both with a 400 error, and on
  `claude-opus-5-5` thinking can't be turned off at all. With thinking on (the
  default), CARD builds a valid request for these models.
- **Newer Claude models, cost limit.** The audio-slicing price table only
  lists 4.x models. For any other model the cost estimate is 0, so
  `--max-estimated-cost-usd` never stops the run.
- **Newer Claude models, effort.** `--thinking-effort` accepts
  `low|medium|high|max` but not `xhigh`.
- **DeepSeek V4.** Thinking is turned on with a request parameter rather than
  a separate model name, which `DeepSeekProvider` doesn't send yet. Switching
  `deepseek-reasoner` to `deepseek-v4-flash` would run without reasoning.

## `.env` variables

```dotenv
CARD_FRAMEWORK_VLLM_API_KEY=EMPTY
ANTHROPIC_API_KEY=
DEEPSEEK_API_KEY=
GEMINI_API_KEY=
ZAI_API_KEY=
HF_TOKEN=
```

CARD loads `.env` from the folder you run in. Variables already set in your
shell take priority. Older names still work and are copied to the names above:
`VLLM_API_KEY`, `GOOGLE_API_KEY`, `GLM_API_KEY`, `HF_ACCESS_TOKEN`, and
`HF_API_TOKEN`.

The benchmark QA config (`src/card_framework/benchmark/qa_config.yaml`) also
reads its vLLM key from `.env`.

## Running with Docker

Docker Compose is the recommended way to run from a source checkout. The
services mount your checkout, so the container uses your `conf/config.yaml`,
`.env`, and input files directly.

```bash
docker compose build setup-and-run

# One recording, CPU
docker compose run --rm setup-and-run --audio-path audio.wav

# One recording, NVIDIA GPU
docker compose --profile gpu run --rm setup-and-run-gpu --override audio.device=cuda --audio-path audio.wav

# Interactive menu (pick mode, models, and inputs)
docker compose run --rm setup-and-run

# Several recordings, one after another
docker compose run --rm setup-and-run --audio-path a.mp3 --audio-path b.mp3

# Each recording shortened to 30s, 1m, and 5m
docker compose run --rm setup-and-run --audio-path a.mp3 --audio-path b.mp3 --duration-sweep
```

Things to know:

- Each service is capped at about 8 GB of RAM plus 2 GB of swap
  (`mem_limit: 8g`, `memswap_limit: 10g` in `compose.yml`).
- The container keeps its Python environments in Docker volumes, so it never
  writes Linux virtualenvs into your checkout.
- Without input flags, the wrapper opens the interactive menu. Add `--no-menu`
  to skip it.
- With several `--audio-path` flags, each recording gets its own output folder
  and the batch root gets a `batch_manifest.json`.
- `--duration-sweep` transcribes each recording once and reuses that
  transcript for every length. Finished lengths are skipped on rerun. Progress
  is recorded in `duration_sweep_manifest.json`.

## Resuming a run

Every run started by `card_framework.cli.setup_and_run` gets a session ID,
printed when the run ends. To continue an interrupted run:

```bash
docker compose run --rm setup-and-run --resume-session <session-id>
```

Resume works per stage. If a stage (transcription, summarization, voice
cloning, or interjection) was interrupted, that whole stage runs again.

Sessions are stored in `artifacts/pipeline_sessions/sessions.sqlite3`, and each
session's final summary is saved to
`artifacts/pipeline_sessions/<session-id>/summary.xml`. The `session.*` keys in
the config are managed by the wrapper; you only need them if you call the
runtime without `setup_and_run`.

## Audio stage settings

These live under `audio` in `conf/config.yaml`. The defaults suit podcasts and
other long multi-speaker recordings.

### Speech separation (`audio.separation`)

```yaml
provider: "demucs"
model: "htdemucs"
timeout_seconds: 1800
window_length_seconds: 600   # process long files in 10-minute windows
window_context_seconds: 5    # with 5 seconds of overlap on each side
```

### Transcription (`audio.asr`)

```yaml
provider: "granite_speech"   # or "faster_whisper"
model: "ibm-granite/granite-4.0-1b-speech"
chunk_length_seconds: 25
chunk_overlap_seconds: 1
max_new_tokens: 96
repetition_penalty: 1.25
language: "en"
forced_alignment:
  enabled: true
  required: true
  batch_size: 8
```

The overlap and repetition penalty were tuned to stop words repeating where
chunks meet, such as "index index". A larger overlap or a lower penalty brings
those repeats back. Lowering `max_new_tokens` drops words without reducing the
repeats further.

### Speaker diarization (`audio.diarization`)

```yaml
provider: "nemo"
nemo_model: "diar_msdd_telephonic"
vad_model: "vad_multilingual_marblenet"
speaker_embedding_model: "titanet_large"
min_speakers:                # blank = let NeMo estimate
max_speakers: 10
allow_single_speaker_fallback: false
vad_onset: 0.8
vad_offset: 0.6
vad_pad_onset: 0.1
vad_pad_offset: 0.0
vad_min_duration_on: 0.2
vad_min_duration_off: 0.6
clustering_max_rp_threshold: 0.4
clustering_sparse_search_volume: 60
clustering_maj_vote_spk_count: true
```

How to adjust it:

- **Unknown number of speakers** (for example, shows with guests): leave
  `min_speakers` blank. Raise `max_speakers` only if a recording can really
  have more than 10 voices.
- **Known number of speakers:** set `min_speakers` and `max_speakers` to the
  same value. NeMo then uses exactly that count.
- **Too many tiny speaker switches:** the `vad_*` keys control how short a
  speech burst or pause can be before it counts. Raising
  `vad_min_duration_on` and `vad_min_duration_off` merges short fragments.
- **Too few speakers found:** raise `clustering_max_rp_threshold` and
  `clustering_sparse_search_volume`.

CARD also fixes one- to three-word speaker switches in the transcript when the
segments on both sides agree on the speaker.

## Security notes

- Never commit `.env`. It is in `.gitignore`.
- Avoid passing secrets as `--override` flags, since they end up in your shell
  history.
