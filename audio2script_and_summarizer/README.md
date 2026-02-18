# Audio2Script + Summarizer Integration Prototype

An isolated, end-to-end pipeline that ingests raw podcast audio, performs speaker diarization, and outputs a duration-constrained, speaker-aware JSON summary.

## 🛠️ Prerequisites

1.  **Python 3.10+** (3.12 recommended)
2.  **FFmpeg** (Required for audio processing)
    * *Windows:* [Download](https://ffmpeg.org/download.html) and add `bin` to PATH.
    * *Ubuntu/WSL:* `sudo apt install ffmpeg`
    * *Mac:* `brew install ffmpeg`
3.  **NVIDIA GPU** (Optional): Requires CUDA 11.8+ for faster processing.
4.  **uv** (Recommended): Install with `pip install uv` or see [uv docs](https://docs.astral.sh/uv/)
5.  **HuggingFace Token**: Required for pyannote.audio diarization models.

## 📦 Installation

### Using uv (Recommended)

From the root `card-framework` directory:

```bash
# Install dependencies with CUDA support
uv sync --extra audio2script

# Or install all extras including audio2script
uv sync --all-extras
```

> [!NOTE]
> The `pyproject.toml` is configured to automatically use CUDA-enabled PyTorch on Windows/Linux via the `pytorch-cuda` index.

### Dependency Management Policy

Use `uv` only for dependency resolution and execution in this repository.
Do not install `audio2script_and_summarizer` dependencies with `pip -r requirements.txt`.

## 🚀 Usage

You must provide:
- **OpenAI API Key** for the summarization stage
- **HuggingFace Token** for pyannote diarization models
- **Target summary duration** in minutes (prompted if not passed)

By default, the pipeline runs an emotion-aware IndexTTS2 preflight calibration
pass on extracted speaker samples to estimate WPM, then converts your target
minutes into a strict word budget for the summarizer.
Stage 1.75 now reuses calibration results from an input-keyed cache by default
to avoid repeating identical preflight work across runs.
You can optionally derive WPM directly from transcript timestamps with
`--wpm-source transcript`.

### Using uv (Recommended)

From the root `card-framework` directory:

```bash
# Set required environment variables
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."

# Run the pipeline (you will be prompted for target minutes and LLM provider if omitted)
uv run --extra audio2script python -m audio2script_and_summarizer \
    --input "path/to/podcast.wav"

# Or pass the target duration explicitly (minutes)
uv run --extra audio2script python -m audio2script_and_summarizer \
    --input "path/to/podcast.wav" \
    --target-minutes 5

# Or pass API key directly (OpenAI)
uv run --extra audio2script python -m audio2script_and_summarizer \
    --input "path/to/podcast.wav" \
    --openai-key "sk-..."

# Use DeepSeek instead of OpenAI
export DEEPSEEK_API_KEY="ds-..."
uv run --extra audio2script python -m audio2script_and_summarizer \
    --input "path/to/podcast.wav" \
    --llm-provider deepseek \
    --target-minutes 8

# DeepSeek defaults to model deepseek-reasoner with a larger output token budget.
# Override with --model only if you need a specific model behavior.

# Use transcript-derived WPM instead of TTS preflight calibration
uv run --extra audio2script python -m audio2script_and_summarizer \
    --input "path/to/podcast.wav" \
    --wpm-source transcript

# One-shot emotion-aware WPM calibration export (per emotion, per speaker)
uv run --extra audio2script python -m audio2script_and_summarizer.calibrate_wpm \
    --voice-dir "path/to/<audio>_voices" \
    --transcript-json "path/to/<audio>.json" \
    --output "calibrated_wpm.json"
```

### Alternate Entry Point (still via uv)

```bash
uv run --extra audio2script python -m audio2script_and_summarizer.run_pipeline \
    --input "path/to/podcast.wav" --openai-key "sk-..."
```

## ⚙️ Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input` | Path to `.wav`, `.mp3`, or `.m4a` file | Required |
| `--openai-key` | Your OpenAI API key | `$OPENAI_API_KEY` |
| `--device` | Processing device | `cuda` (if available) |
| `--voice-dir` | Directory for speaker samples | `stage2_voices` |
| `--wpm-source` | WPM source (`tts_preflight`, `transcript`, or alias `indextts`) | `tts_preflight` |
| `--target-minutes` | Target summary duration in minutes | Prompted |
| `--duration-tolerance-seconds` | Allowed Stage 3 duration delta in seconds | `3.0` |
| `--max-duration-correction-passes` | Number of closed-loop Stage 2/3 correction passes | `1` |
| `--calibration-presets-path` | Emotion preset config JSON for TTS preflight | `audio2script_and_summarizer/emotion_pacing_presets.json` |
| `--wpm-calibration-cache-mode` | Stage 1.75 cache policy (`auto`, `refresh`, `off`) | `auto` |
| `--wpm-calibration-cache-dir` | Stage 1.75 calibration cache directory | `artifacts/cache/wpm_calibration` |
| `--llm-provider` | `openai` or `deepseek` | Prompted |
| `--word-budget-tolerance` | Allowed deviation ratio (e.g. 0.05 = +/-5%) | `0.05` |
| `--skip-a2s` | Skip Stage 1/1.5 and choose an existing transcript JSON for direct DeepSeek summarization | `false` |
| `--skip-a2s-search-root` | Root folder scanned for transcript JSON when `--skip-a2s` is used | `.` |
| `--deepseek-max-completion-tokens` | Hard output token ceiling passed to DeepSeek summarizer | `64000` |

## 🔊 Diarizer Options

### pyannote (Default, Recommended)
Uses [pyannote.audio](https://github.com/pyannote/pyannote-audio) for speaker diarization.
- ✅ Compatible with all card-framework dependencies
- ✅ State-of-the-art diarization quality
- ⚠️ Requires HuggingFace token (`HF_TOKEN` env var)

### msdd (NeMo)
Uses NVIDIA NeMo's Multi-Scale Diarization Decoder.
- ⚠️ **Incompatible** with main card-framework dependencies (protobuf conflict)
- Requires separate virtual environment with `nemo_toolkit[asr]`
- Use only if pyannote doesn't meet your needs

## 🔧 CUDA Support

The pipeline automatically uses CUDA if available. When installed via `uv sync --extra audio2script`:

- **PyTorch** is installed with CUDA 12.8 support on Windows/Linux
- **Pyannote.audio** uses GPU-accelerated diarization
- **Faster Whisper** leverages CuDNN for transcription

To verify CUDA is working:

```bash
uv run --extra audio2script python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```
