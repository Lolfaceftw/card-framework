# IndexTTS2 Voice Clone Benchmark

This folder contains a modular benchmark script for assessing IndexTTS2 voice-cloning similarity in the CARD pipeline.

## Research Basis (Cutoff: February 12, 2026)

Selected metrics reflect commonly used voice-cloning evaluation practice as of `2026-02-12`:

- **Speaker similarity cosine (SS)**: embedding cosine between generated and holdout target reference.
- **ASV EER**: verification-style equal error rate across same-speaker and different-speaker pairs.
- **Top-1 speaker retrieval accuracy**: nearest-centroid speaker identity correctness.
- **SMOS/CMOS subjective kit**: randomized A/B package for human speaker-similarity ratings.

Primary references:

- IndexTTS2 paper: https://arxiv.org/abs/2506.21619
- Seed-TTS eval toolkit: https://github.com/BytedanceSpeech/seed-tts-eval
- WavLM speaker verification model: https://huggingface.co/microsoft/wavlm-base-plus-sv
- SpeechBrain speaker verification docs: https://speechbrain.readthedocs.io/en/latest/API/speechbrain.inference.speaker.html
- ITU-T P.808 recommendation page: https://www.itu.int/rec/T-REC-P.808

## Manifest Format

Create a manifest JSON with rows shaped like:

```json
[
  {
    "speaker_id": "SPEAKER_00",
    "prompt_wav": "outputs/extracted_voices/SPEAKER_00_prompt.wav",
    "reference_wav": "outputs/extracted_voices/SPEAKER_00_holdout.wav",
    "text": "Example text to synthesize.",
    "use_emo_text": true,
    "emo_text": "Neutral and clear delivery.",
    "emo_alpha": 0.6
  }
]
```

Notes:

- `prompt_wav` should be the clip used for cloning.
- `reference_wav` should be a **different holdout clip** from the same speaker for fair scoring.

## Run

From repository root:

```bash
uv run python benchmarks/benchmark_indextts2_voice_clone.py \
  --manifest benchmarks/examples/voice_clone_manifest.json \
  --cfg-path checkpoints/config.yaml \
  --model-dir checkpoints \
  --device cuda
```

## Easiest Setup (Wizard)

If you do not want to hand-edit manifest JSON, run the interactive wizard:

```bash
uv run python benchmarks/setup_voice_clone_benchmark.py
```

The wizard will:

- discover your latest `*_summary.json`,
- explain `prompt_wav` and `reference_wav`,
- generate a manifest automatically,
- optionally run the benchmark immediately.

Quick smoke run:

```bash
uv run python benchmarks/benchmark_indextts2_voice_clone.py \
  --manifest <your_manifest.json> \
  --max-items 5 \
  --device cpu
```

Objective-only (skip MOS package):

```bash
uv run python benchmarks/benchmark_indextts2_voice_clone.py \
  --manifest <your_manifest.json> \
  --no-prepare-mos-kit
```

## Outputs

Each run writes under `benchmarks/runs/<timestamp>/`:

- `metrics_summary.json`: aggregate metrics and confidence intervals.
- `pair_scores.csv`: per-item objective scores.
- `generated/`: synthesized benchmark utterances.
- `run.log`: UTC structured logs.
- `mos_pack/` (if enabled): randomized A/B audio plus rating templates.
