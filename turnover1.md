# Turnover 1

## Objective
Provide a complete handoff of testing activity for the `audio2script_and_summarizer` pipeline after API key setup, including what was run, what failed, why it failed, and what to do next.

## Repository Context
- Repo: `card-framework`
- Branch: `feature/audio-pipeline` (verified during this session)
- Date: 2026-02-11
- Environment: Windows + PowerShell
- Input under test: `audio_samples/audio.wav` (~1.05 GB)

## User Requests Covered
1. Proceed with testing after API keys were set.
2. Confirm branch is `feature/audio-pipeline`.
3. Clarify expected `.env` format.
4. Resume tests after `.env` changes.

## Branch and Environment Verification
- Confirmed active branch: `feature/audio-pipeline`.
- Confirmed expected env var names from code and `.env`:
  - `OPENAI_API_KEY`
  - `DEEPSEEK_API_KEY`
  - `HF_TOKEN`

## .env Guidance Provided
Expected root `.env` format:

```dotenv
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=...
HF_TOKEN=hf_...
```

Notes:
- Use plain `KEY=value` lines.
- No `export`.
- No spaces around `=`.

## What Was Executed
### Discovery / validation commands
- Read `TURNOVER.md`.
- Verified project tree and relevant module files.
- Read `audio2script_and_summarizer/README.md`.
- Confirmed CLI args from:
  - `audio2script_and_summarizer/run_pipeline.py`
  - `audio2script_and_summarizer/run_pipeline_deepseek.py`
- Confirmed branch with `git branch --show-current`.
- Confirmed env variable usage via `rg` across pipeline files.

### Test runs attempted
1. OpenAI pipeline (initial):

```powershell
uv run --extra audio2script python -m audio2script_and_summarizer.run_pipeline --input audio_samples/audio.wav --device cuda
```

- First failure: `[ERROR] No OpenAI API Key found. Use --openai-key`
- Cause: `.env` values were not in the current process environment for that invocation.

2. OpenAI pipeline (with env load + UTF-8 settings):

```powershell
Get-Content .env | ForEach-Object { ...Set-Item Env:... }
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
uv run --extra audio2script python -m audio2script_and_summarizer.run_pipeline --input audio_samples/audio.wav --device cuda
```

- Progressed through demucs separation and diarization startup.
- Failed in Stage 1 with traceback (details in Findings).

3. Deepseek pipeline (with env load + UTF-8 settings):

```powershell
Get-Content .env | ForEach-Object { ...Set-Item Env:... }
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
uv run --extra audio2script python -m audio2script_and_summarizer.run_pipeline_deepseek --input audio_samples/audio.wav --device cuda
```

- Same Stage 1 failure as OpenAI path.

## Key Findings
### 1) Common hard blocker in Stage 1 (both OpenAI + Deepseek)
Both pipelines fail before summarization during pyannote model load with:

- `_pickle.UnpicklingError: Weights only load failed`
- Message indicates PyTorch 2.6+ changed default `torch.load(..., weights_only=True)` behavior.
- Failure path from traceback:
  - `audio2script_and_summarizer/diarize.py:265`
  - `audio2script_and_summarizer/diarization/pyannote.py:49`
  - `pyannote.audio -> lightning_fabric -> torch.load`

Impact:
- Stage 2 (OpenAI/Deepseek summarization) never runs.
- API key validity for LLM summarizers is not fully exercised in this run because execution never reaches summarizer calls.

### 2) Windows console encoding issue (encountered and worked around)
OpenAI runner initially crashed at startup print due to emoji output under cp1252:

- `UnicodeEncodeError: 'charmap' codec can't encode character '\U0001f680'`
- Location: `audio2script_and_summarizer/run_pipeline.py:70`

Workaround used during testing:
- `PYTHONUTF8=1`
- `PYTHONIOENCODING=utf-8`

### 3) Additional warnings seen (non-blocking for this run)
- HuggingFace symlink warnings on Windows cache.
- Torchaudio deprecation warnings.
- `hf_xet` performance warning (fallback to HTTP).

## Partial Artifacts / Runtime Behavior
- Demucs separation started and progressed in both runs.
- Temporary outputs were created under `temp_outputs_*` directories.
- No successful final transcript-summary output was produced due to Stage 1 failure.

## What Was Not Changed
- No repository code changes were made in this session.
- No commits were created.

## Recommended Next Steps
1. Fix pyannote + torch compatibility in Stage 1:
- Option A: Pin to dependency versions known to work together (preferred for stability).
- Option B: Add a controlled compatibility shim around checkpoint loading (only if trusted-source assumptions are accepted).

2. Remove Windows console fragility:
- Replace emoji prints with ASCII-safe output in:
  - `audio2script_and_summarizer/run_pipeline.py`
  - `audio2script_and_summarizer/run_pipeline_deepseek.py`

3. Re-run end-to-end tests after fix:

```powershell
# OpenAI
uv run --extra audio2script python -m audio2script_and_summarizer.run_pipeline --input audio_samples/audio.wav --device cuda

# Deepseek
uv run --extra audio2script python -m audio2script_and_summarizer.run_pipeline_deepseek --input audio_samples/audio.wav --device cuda
```

4. Optional hardening:
- Add a deterministic preflight check that fails fast with actionable messaging when incompatible torch/pyannote versions are detected.

## Handoff Summary
- Branch is correct.
- `.env` structure is correct.
- Both full pipelines were executed with API/env loading and reached Stage 1 processing.
- Current blocker is a pyannote checkpoint load failure caused by torch serialization behavior change.
- Pipeline remains blocked until Stage 1 dependency compatibility is fixed.
