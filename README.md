# CARD: Constraint-aware Audio Resynthesis and Distillation

CARD (Constraint-aware Audio Resynthesis and Distillation) provides an automated
pipeline that turns a single input audio file into:

- diarization and transcript artifacts,
- per-speaker voice samples,
- a structured summary JSON for downstream synthesis.

### Undergraduate Student Project | University of the Philippines, Diliman

This root README focuses on running the automated pipeline end-to-end while
preserving the core project framing from `main`.

<div align="center">
  <a href="#abstract">Abstract</a> |
  <a href="#system-architecture">Architecture</a> |
  <a href="#automated-pipeline-quickstart-powershell">Quickstart</a> |
  <a href="#pipeline-argument-reference">Arguments</a> |
  <a href="#project-structure">Project Structure</a> |
  <a href="#related-workflows">Workflows</a> |
  <a href="#development-notes">Development</a>
</div>

<br/>

**Authors:**
- **Rei Dennis Agustin** (BS Electronics Engineering)
- **Sean Luigi P. Caranzo** (BS Computer Engineering)
- **Johnbell R. De Leon** (BS Computer Engineering)
- **Christian Klein C. Ramos** (BS Electronics Engineering)

**Adviser:**
- **Rowel D. Atienza**

## Abstract

The exponential growth of long-form podcasting creates a consumption bottleneck, as listeners lack efficient means to digest multi-speaker content within limited timeframes. Current summarization approaches, whether text-based or extractive audio clipping, fail to preserve the immersive, prosodic nature of conversational audio.

**CARD (Constraint-aware Audio Resynthesis and Distillation)** is a generative framework designed to resolve the trade-off between consumption efficiency and audio fidelity. We propose a pipeline that addresses three critical challenges in audio generation:

1. **Temporal Control:** Utilizing forced alignment to calculate speaker-specific speaking rates, enabling an LLM to compress dialogue into a structured representation that strictly adheres to a user-defined time budget.
2. **Spectral Control:** Utilizing diarization timestamps to harvest reference samples directly from raw input, driving zero-shot voice cloning via **IndexTTS2**.
3. **Conversational Control:** A refinement module using a 4-bit quantized **Mistral 7B** model to predict semantic interjection points, generating syntactically-aware asynchronous overlaps.

The outcome is a functional prototype that validates the feasibility of duration-controlled, high-fidelity conversational resynthesis.

## System Architecture

The CARD paradigm shifts from extraction to resynthesis through a four-stage pipeline:

### 1) Audio2Script (Ingestion)
- **Models:** OpenAI Whisper (ASR) + NVIDIA NeMo (Forced Alignment).
- **Function:** Ingests raw podcast audio to generate a timestamped, speaker-attributed transcript. It calculates the original Words-Per-Minute (WPM) to derive the word budget for the summary.

### 2) Speaker Audio Extraction
- **Models:** Meta Demucs (Source Separation) + SepFormer (Targeted Overlap Separation).
- **Function:** Extracts clean, speaker-pure reference audio tracks. It uses a "Targeted Separation" strategy:
  - Non-overlapping segments are extracted directly.
  - Overlapping segments are processed via SepFormer to disentangle speakers.
  - Outputs enrollment embeddings for the Voice Cloner.

### 3) Script Summarizer
- **Models:** GitHub Copilot / GPT-5 Class LLM.
- **Function:** Compresses the transcript into a JSON structure strictly adhering to the time budget. It injects `emo_text` (affective prompts) and maintains speaker identity.

### 4) Voice Cloning and Backchanneling
- **Models:** **IndexTTS2** (Synthesis) + **Mistral 7B Quantized** (Conversational Supervisor).
- **Function:** Synthesizes the audio using zero-shot cloning. The Mistral module analyzes the text flow to inject asynchronous interjections (e.g., "Right," "No way") and overlaps to restore the "parasocial" vibe of human conversation.

## Automated Pipeline Quickstart (PowerShell)

### 1) Install dependencies

```powershell
pip install -U uv
uv sync --extra audio2script --extra dev
```

Notes:
- `--extra audio2script` is required for the automated pipeline.
- `--extra dev` is optional but recommended for lint/test tooling.

### 2) Configure credentials with `.env` (recommended)

Create `.env` in repo root:

```dotenv
HF_TOKEN=hf_...
DEEPSEEK_API_KEY=sk-...
OPENAI_API_KEY=sk-...
```

`run_pipeline.py` and `run_pipeline_deepseek.py` auto-load `.env` at startup.

### 3) Run the automated pipeline (DeepSeek primary)

```powershell
uv run .\audio2script_and_summarizer\run_pipeline_deepseek.py `
  --input .\data\samples\audio_30s.wav `
  --config .\config\card.yaml
```

### 4) OpenAI alternate path

```powershell
uv run .\audio2script_and_summarizer\run_pipeline.py `
  --input .\data\samples\audio_30s.wav `
  --config .\config\card.yaml
```

### 5) Verify output artifacts

With `--config .\config\card.yaml`, output is routed to:

- `artifacts/diarization/<input_stem>.json`
- `artifacts/transcripts/<input_stem>.txt`
- `artifacts/transcripts/<input_stem>.srt`
- `artifacts/voices/<input_stem>/`
- `artifacts/summaries/<input_stem>_summary.json`
- `artifacts/temp/<input_stem>/`

## How The Automated Pipeline Runs

Both pipeline scripts orchestrate the same core stages:

1. Stage 1: Diarization + transcript generation.
2. Stage 1.5: Speaker sample extraction to WAV files.
3. Stage 2: LLM summarization to final JSON.

Use these scripts directly:

- DeepSeek: `uv run .\audio2script_and_summarizer\run_pipeline_deepseek.py`
- OpenAI: `uv run .\audio2script_and_summarizer\run_pipeline.py`

## Pipeline Argument Reference

### Common arguments (both scripts)

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input` | Yes | None | Path to input audio file. |
| `--device` | No | `cuda` if available, else `cpu` | Runtime device for diarization/transcription stages. |
| `--voice-dir` | No | Derived from input/config | Directory for speaker sample WAV outputs. |
| `--no-stem` | No | `False` | Skip Demucs source-separation pre-step in diarization stage. |
| `--show-deprecation-warnings` | No | `False` | Show third-party deprecation warnings normally filtered in CLI output. |
| `--no-progress` | No | `False` | Disable child-stage progress bars. |
| `--plain-ui` | No | `False` | Disable rich terminal dashboard UI. |
| `--config` | No | None | Runtime YAML config path (same as `CARD_CONFIG_PATH`). |

### DeepSeek script arguments

Script: `.\audio2script_and_summarizer\run_pipeline_deepseek.py`

| Argument | Required | Default | Description |
|---|---|---|---|
| `--deepseek-key` | No* | `$DEEPSEEK_API_KEY` | DeepSeek API key. |
| `--model` | No | `deepseek-chat` | DeepSeek model passed to summarizer. |
| `--whisper-model` | No | `medium.en` | Whisper model for Stage 1 transcription. |
| `--language` | No | auto | Explicit language hint for transcription. |
| `--max-completion-tokens` | No | `8192` | Max output tokens for summarization response. |
| `--request-timeout-seconds` | No | `120.0` | HTTP timeout for summarizer requests. |
| `--http-retries` | No | `1` | Request retry count for summarizer calls. |
| `--temperature` | No | `0.2` | Generation temperature for summarizer. |

\* Required unless `DEEPSEEK_API_KEY` is set in environment or `.env`.

Example with explicit key and model:

```powershell
uv run .\audio2script_and_summarizer\run_pipeline_deepseek.py `
  --input .\data\samples\audio_30s.wav `
  --config .\config\card.yaml `
  --deepseek-key "sk-..." `
  --model deepseek-chat `
  --max-completion-tokens 8192 `
  --request-timeout-seconds 120 `
  --http-retries 1 `
  --temperature 0.2
```

### OpenAI script arguments

Script: `.\audio2script_and_summarizer\run_pipeline.py`

| Argument | Required | Default | Description |
|---|---|---|---|
| `--openai-key` / `--api-key` | No* | `$OPENAI_API_KEY` | OpenAI API key for Stage 2 summarization. |

\* Required unless `OPENAI_API_KEY` is set in environment or `.env`.

Example with explicit key:

```powershell
uv run .\audio2script_and_summarizer\run_pipeline.py `
  --input .\data\samples\audio_30s.wav `
  --config .\config\card.yaml `
  --openai-key "sk-..."
```

## Runtime Config and Output Routing

Canonical config file:

- `config/card.yaml`

Optional schema:

- `config/card.schema.yaml`

Config resolution order in the pipeline scripts:

1. `--config <path>`
2. `CARD_CONFIG_PATH` environment variable
3. no config (legacy side-by-side output behavior)

When config is not provided and `CARD_CONFIG_PATH` is unset, outputs are written
beside the input file using the input path without extension (for example
`podcast.wav` -> `podcast.json`):

- `<input_without_extension>.json`
- `<input_without_extension>.txt`
- `<input_without_extension>.srt`
- `<input_without_extension>_summary.json`
- `<input_without_extension>_voices/`

## Troubleshooting (Automated Pipeline)

- Error: `No Deepseek API Key found`
  - Set `DEEPSEEK_API_KEY` in `.env` or pass `--deepseek-key`.
- Error: `No OpenAI API key found`
  - Set `OPENAI_API_KEY` in `.env` or pass `--openai-key`.
- HF token issues during diarization:
  - Set `HF_TOKEN` in `.env` before running to avoid interactive prompts.
- Error: `Input file not found`
  - Confirm the `--input` path is correct from repo root.
- CUDA requested but unavailable:
  - Pipeline falls back to CPU automatically; use `--device cpu` to force CPU.
- Rich dashboard/UI problems:
  - Add `--plain-ui` for plain console logging.

## Related Workflows

This root guide intentionally prioritizes the automated pipeline.

- Wrapper entrypoints with auto-config injection:
  - [`scripts/run_audio2script.py` (main branch)](https://github.com/Lolfaceftw/card-framework/blob/main/scripts/run_audio2script.py)
  - [`scripts/run_audio2script_deepseek.py` (main branch)](https://github.com/Lolfaceftw/card-framework/blob/main/scripts/run_audio2script_deepseek.py)
- Speaker separation workflow:
  - [`scripts/run_separation.py` (main branch)](https://github.com/Lolfaceftw/card-framework/blob/main/scripts/run_separation.py)
  - [`CARD-SpeakerAudioExtraction/src/separation/main.py`](CARD-SpeakerAudioExtraction/src/separation/main.py)
- Additional docs:
  - [`audio2script_and_summarizer/README.md`](audio2script_and_summarizer/README.md)
  - [`docs/CONFIGURATION.md` (main branch)](https://github.com/Lolfaceftw/card-framework/blob/main/docs/CONFIGURATION.md)
  - [`docs/ARCHITECTURE.md` (main branch)](https://github.com/Lolfaceftw/card-framework/blob/main/docs/ARCHITECTURE.md)
  - [`docs/STRUCTURE.md` (main branch)](https://github.com/Lolfaceftw/card-framework/blob/main/docs/STRUCTURE.md)

## Project Structure

- [`audio2script_and_summarizer/`](audio2script_and_summarizer/) - Runtime pipeline orchestration, diarization, summarization, and Stage 3 integration.
- [`CARD-SpeakerAudioExtraction/`](CARD-SpeakerAudioExtraction/) - Speaker extraction and separation subsystem.
- [`voice-cloner-and-interjector/`](voice-cloner-and-interjector/) - IndexTTS2 engine and related voice-cloning assets/workflows.
- [`benchmarks/`](benchmarks/) - Voice cloning benchmark tooling and run artifacts.
- [`scripts/quality_gate.py`](scripts/quality_gate.py) - Evidence-based lint/type/test quality gate runner.
- [`tests/`](tests/) - Automated test suite.
- [`FLOW.MD`](FLOW.MD) - Comprehensive sequence-diagram flow documentation.
- [`AGENTS.md`](AGENTS.md) - Repository-wide agent/contributor standards.

## Development Notes

- Run commands from repository root.
- Use `uv run` consistently for project commands.
- Keep generated artifacts out of git (already covered by `.gitignore`).
