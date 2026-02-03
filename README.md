<div align="center">
<h1>CARD: Constraint-aware Audio Resynthesis and Distillation</h1>
<h3>Undergraduate Student Project | University of the Philippines, Diliman</h3>
</div>

<div align="center">
  <a href="#-abstract">Abstract</a> | 
  <a href="#-system-architecture">Architecture</a> | 
  <a href="#-installation">Installation</a> | 
  <a href="#-usage-workflow">Usage</a>
</div>

<br/>

**Authors:**
*   **Rei Dennis Agustin** (BS Electronics Engineering)
*   **Sean Luigi P. Caranzo** (BS Computer Engineering)
*   **Johnbell R. De Leon** (BS Computer Engineering)
*   **Christian Klein C. Ramos** (BS Electronics Engineering)

**Adviser:**
*   **Rowel D. Atienza**

---

## 📖 Abstract

The exponential growth of long-form podcasting creates a consumption bottleneck, as listeners lack efficient means to digest multi-speaker content within limited timeframes. Current summarization approaches, whether text-based or extractive audio clipping, fail to preserve the immersive, prosodic nature of conversational audio.

**CARD (Constraint-aware Audio Resynthesis and Distillation)** is a generative framework designed to resolve the trade-off between consumption efficiency and audio fidelity. We propose a pipeline that addresses three critical challenges in audio generation:

1.  **Temporal Control:** Utilizing forced alignment to calculate speaker-specific speaking rates, enabling an LLM to compress dialogue into a structured representation that strictly adheres to a user-defined time budget.
2.  **Spectral Control:** Utilizing diarization timestamps to harvest reference samples directly from raw input, driving zero-shot voice cloning via **IndexTTS2**.
3.  **Conversational Control:** A refinement module using a 4-bit quantized **Mistral 7B** model to predict semantic interjection points, generating syntactically-aware asynchronous overlaps.

The outcome is a functional prototype that validates the feasibility of duration-controlled, high-fidelity conversational resynthesis.

---

## 🏗️ System Architecture

The CARD paradigm shifts from extraction to resynthesis through a four-stage pipeline:

### 1. Audio2Script (Ingestion)
*   **Models:** OpenAI Whisper (ASR) + NVIDIA NeMo (Forced Alignment).
*   **Function:** Ingests raw podcast audio to generate a timestamped, speaker-attributed transcript. It calculates the original Words-Per-Minute (WPM) to derive the word budget for the summary.

### 2. Speaker Audio Extraction
*   **Models:** Meta Demucs (Source Separation) + SepFormer (Targeted Overlap Separation).
*   **Function:** Extracts clean, speaker-pure reference audio tracks. It uses a "Targeted Separation" strategy:
    *   Non-overlapping segments are extracted directly.
    *   Overlapping segments are processed via SepFormer to disentangle speakers.
    *   Outputs enrollment embeddings for the Voice Cloner.

### 3. Script Summarizer
*   **Models:** GitHub Copilot / GPT-5 Class LLM.
*   **Function:** Compresses the transcript into a JSON structure strictly adhering to the time budget. It injects `emo_text` (affective prompts) and maintains speaker identity.

### 4. Voice Cloning & Backchanneling
*   **Models:** **IndexTTS2** (Synthesis) + **Mistral 7B Quantized** (Conversational Supervisor).
*   **Function:** Synthesizes the audio using zero-shot cloning. The Mistral module analyzes the text flow to inject asynchronous interjections (e.g., "Right," "No way") and overlaps to restore the "parasocial" vibe of human conversation.

---

## ⚙️ Installation

We use `uv` to manage the project's dependency environment.

1.  **Install `uv`:**
    ```bash
    pip install -U uv
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Lolfaceftw/card-framework.git && cd card-framework
    git lfs pull
    ```

3.  **Install Dependencies:**
    This command installs the environments for the Voice Cloner (IndexTTS2) and the CARD Speaker Extraction module.
    ```bash
    uv sync --extra webui --extra speaker-extraction
    ```

4.  **Download Models:**
    ```bash
    uv tool install "huggingface-hub[cli,hf_xet]"
    cd voice-cloner-and-interjector
    hf download IndexTeam/IndexTTS-2 --local-dir=checkpoints
    ```

---

## 🚀 Usage Workflow

### Phase 1: Speaker Audio Extraction
Extract clean reference audio and speaker embeddings from your raw podcast file. This implements the **Targeted Separation** methodology described in Chapter 4.2.

```bash
uv run CARD-SpeakerAudioExtraction/src/separation/main.py \
    --audio inputs/my_podcast.wav \
    --diarization inputs/my_podcast_transcript.json \
    --mode targeted \
    --output-dir outputs/extracted_voices/
```

### Phase 2: Script Summarization (JSON Generation)
The summarizer LLM must output a JSON structure compatible with the CARD pipeline. Below is the required schema (as defined in Section 4.3.4):

```json
[
  {
    "speaker": "Host",
    "voice_sample": "outputs/extracted_voices/SPEAKER_00.wav",
    "use_emo_text": true,
    "emo_text": "Warm welcoming slightly thoughtful with genuine fascination.",
    "emo_alpha": 0.6,
    "text": "Welcome back to the show. Today we are discussing the future of AI."
  },
  {
    "speaker": "Guest",
    "voice_sample": "outputs/extracted_voices/SPEAKER_01.wav",
    "use_emo_text": true,
    "emo_text": "Excited and fast-paced.",
    "emo_alpha": 0.7,
    "text": "It's great to be here! The pace of innovation is just exploding."
  }
]
```

### Phase 3: Resynthesis & Backchanneling
This stage combines IndexTTS2 for identity preservation and Mistral 7B for conversational dynamics (interjections).

**Run the Advanced Conversational Pipeline:**
This script corresponds to Section 4.4, handling "Post-Hoc Acoustic Alignment" and "Context-Aware Response Generation."

```bash
# Ensure you have your summarized JSON ready
uv run tools/podcast/llm_context_vibe.py
```

*Note: This script uses the quantized Mistral model to detect trigger words (e.g., conflict, surprise) and inserts asynchronous overlaps.*

---

## 📂 Project Structure

*   `indextts/` - Core IndexTTS2 inference engine (Voice Cloner).
*   `CARD-SpeakerAudioExtraction/` - The Audio2Script and Separation modules (SepFormer/Demucs integration).
*   `tools/podcast/` - The CARD integration scripts (Mistral Backchanneling + Pipeline Orchestration).
*   `checkpoints/` - Model weights.

---

## 📚 Citation

If you use the CARD framework or the IndexTTS2 engine in your research, please cite:

**The CARD Project:**
```bibtex
@techreport{agustin2025card,
  title={CARD: Constraint-aware Audio Resynthesis and Distillation},
  author={Agustin, Rei Dennis and Caranzo, Sean Luigi P. and De Leon, Johnbell R. and Ramos, Christian Klein C.},
  institution={University of the Philippines, Diliman},
  year={2025},
  month={December},
  type={Undergraduate Student Project}
}
```

**IndexTTS2 Engine:**
```bibtex
@article{zhou2025indextts2,
  title={IndexTTS2: A Breakthrough in Emotionally Expressive and Duration-Controlled Auto-Regressive Zero-Shot Text-to-Speech},
  author={Siyi Zhou, Yiquan Zhou, Yi He, Xun Zhou, Jinchao Wang, Wei Deng, Jingchen Shu},
  journal={arXiv preprint arXiv:2506.21619},
  year={2025}
}
```