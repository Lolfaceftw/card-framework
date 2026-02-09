# Audio2Script + Audio Splitter + Summarizer Integration Prototype

An isolated, end-to-end pipeline that ingests raw podcast audio, performs speaker diarization, splits audio by speaker, and outputs a duration-constrained, speaker-aware JSON summary.

## Windows Setup (WSL 2)

If you are on Windows, you must run this pipeline inside WSL 2 (Windows Subsystem for Linux) to avoid compatibility issues with audio libraries.

1. Open PowerShell or Command Prompt as Administrator.
2. Run this command:
```powershell
wsl --install

```


3. Restart your computer.
4. After restarting, a terminal window will open to set up your Ubuntu username and password.

## Memory Configuration (Prevents Crashing)

By default, WSL 2 caps memory usage at 50% of your system RAM. To prevent crashes during processing, you must increase this limit and enable swap memory.

1. Press **Windows Key + R**, type `%UserProfile%`, and press Enter.
2. Create a file named `.wslconfig` (ensure it has no .txt extension).
3. Paste the following configuration:
```ini
[wsl2]
memory=12GB
swap=16GB

```


> `swap` acts as "emergency RAM" stored on the disk. It’s much slower than real RAM, but it prevents WSL from crashing when memory runs out.


*Note: Set `memory` to roughly 70-80% of your total system RAM. Leave at least 4GB for Windows.*

4. Restart WSL by running `wsl --shutdown` in PowerShell.

## Prerequisites

1. **Python 3.10+** (3.12 recommended)
2. **FFmpeg** (Required for audio processing)
Run this inside your WSL/Ubuntu terminal:
```bash
sudo apt update && sudo apt install ffmpeg -y

```


3. **NVIDIA GPU** (Optional): Requires CUDA 11.8+ for faster processing.

## Installation

**Note: Run all the following commands inside your WSL 2 (Ubuntu) terminal.**

1. **Clone & Switch Branch**
```bash
git clone https://github.com/Lolfaceftw/card-framework.git
cd card-framework
git checkout feature/audio-pipeline
cd audio2script_and_summarizer

```


2. **Setup Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate

```


3. **Install Dependencies**
```bash
pip install -r requirements.txt

```



## Usage

You must provide a Generative AI API Key (Gemini or OpenAI) for the summarization stage.

**Run the Pipeline:**

```bash
python run_pipeline.py --input "path/to/podcast.wav" --api-key "AIzaSy..." --no-stem

```

**Options:**

* `--input`: Path to input audio file.
* `--api-key`: Your API key (can also be set via `LLM_API_KEY` env var).
* `--device`: `cuda` (default) or `cpu`.
* `--voice-dir`: (Optional) Custom folder output for voice samples.
* `--no-stem`: **Skip Demucs vocal separation.** Use this to save time if your audio has no background music. It still converts input to WAV format to ensure compatibility.

## File Transfer Guide

To easily copy your input files into the Linux environment:

1. Open **File Explorer** in Windows.
2. In the address bar, type `\\wsl$\Ubuntu` and press Enter.
3. Navigate to your project folder (usually `home/your_username/card-framework/...`).
4. Drag and drop your audio files directly into the `inputs` folder.

```

```
