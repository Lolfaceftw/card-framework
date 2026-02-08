# Audio2Script + Summarizer Integration Prototype

An isolated, end-to-end pipeline that ingests raw podcast audio, performs speaker diarization, splits audio by speaker, and outputs a duration-constrained, speaker-aware JSON summary.

## 🛠️ Prerequisites

1.  **Python 3.10+** (3.12 recommended)
2.  **FFmpeg** (Required for audio processing)
    * *Windows:* [Download](https://ffmpeg.org/download.html) and add `bin` to PATH.
    * *Ubuntu/WSL:* `sudo apt install ffmpeg`
    * *Mac:* `brew install ffmpeg`
3.  **NVIDIA GPU** (Optional): Requires CUDA 11.8+ for faster processing.

## 📦 Installation

1.  **Clone & Switch Branch**
    ```bash
    git clone [https://github.com/Lolfaceftw/card-framework.git](https://github.com/Lolfaceftw/card-framework.git)
    cd card-framework
    git checkout feature/audio-pipeline
    cd audio2script_and_summarizer
    ```

2.  **Setup Virtual Environment**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\Activate
    # Linux/Mac:
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

You must provide a Generative AI API Key (Gemini or OpenAI) for the summarization stage.

**Run the Pipeline:**
```bash
python run_pipeline.py --input "path/to/podcast.wav" --api-key "AIzaSy..."

```

**What Happens:**

1. **Diarization:** Creates `podcast.json` (Speaker timestamps).
2. **Splitting:** Creates `podcast_voices/` folder with 30s samples for each speaker.
3. **Summarization:** Creates `podcast_summary.json` (The final script).

**Options:**

* `--input`: Path to `.wav`, `.mp3`, or `.m4a` file.
* `--api-key`: Your API key (can also be set via `LLM_API_KEY` env var).
* `--device`: `cuda` (default) or `cpu`.
* `--voice-dir`: (Optional) Custom folder output for voice samples.