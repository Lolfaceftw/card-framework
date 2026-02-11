# Turnover

## Objective
Start a fresh conversation with clear context, scope, and expectations.

## Context
- Repo: card-framework
- Branch: feature/audio-pipeline
- Date: 2026-02-11
- Environment: Windows / PowerShell

## Current Status
- Task: review changes on `feature/audio-pipeline` and test audio pipeline
- Tests run: `uv run --extra audio2script python -m audio2script_and_summarizer --help`
- Full pipeline: not run yet (requires local API keys and write access)

## Key Findings (Review)
- HF token prompt occurs at import time in `audio2script_and_summarizer/diarize.py` (blocks non-interactive runs).
- Prompt format in `summarizer_deepseek.py` does not follow repo prompt standard.
- `summarizer_deepseek.py` configures logging globally (violates library logging policy).
- Non-ASCII / mojibake characters present in multiple files.
- `.gitignore` ignores `temp_outputs` but not `temp_outputs_*`.
- Missing type hints / return types in new files.

## Next Steps
1. Rotate any leaked API keys and export fresh keys in the local env.
2. Run full CUDA tests for both OpenAI and Deepseek pipelines on `audio_samples/audio.wav`.
3. Fix review findings and re-run pipeline.

## Notes
- Audio sample: `audio_samples/audio.wav` (~1.05 GB).
- CUDA preferred; fall back to CPU if unavailable.
