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

### Using uv (Recommended)

From the root `card-framework` directory:

```bash
# Set required environment variables
export OPENAI_API_KEY="sk-..."
export HF_TOKEN="hf_..."

# Run the pipeline
uv run --extra audio2script python -m audio2script_and_summarizer \
    --input "path/to/podcast.wav"

# Or pass API key directly
uv run --extra audio2script python -m audio2script_and_summarizer \
    --input "path/to/podcast.wav" \
    --openai-key "sk-..."
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
| `--diarizer` | Diarization model (`pyannote` or `msdd`) | `pyannote` |

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
