# Errors And Notes

Use this file as the repository memory for problems worth avoiding in future sessions and the fixes that resolved them.

Add new entries at the top so the newest lessons are visible first.
Record each entry with both the date and time, preferably in ISO 8601 format with timezone information.

### 2026-03-07T16:02:08+08:00 - ETA Should Stay Silent Until The Repo Has Learned History
- Problem: The pipeline was showing large bootstrap ETA values on a first-ever run and only flushing learned throughput to disk at coarse end-of-run boundaries, so a completed separation stage could still leave no saved ETA profile for later stages or interrupted runs.
- Solution: Gate ETA display on persisted or learned per-stage history instead of bootstrap defaults, and save the ETA profile immediately after each completed audio, speaker-sample, and voice-clone learning update.

### 2026-03-07T15:37:05+08:00 - Stage-4 Imports Must Stay Lazy Around Optional Dependencies
- Problem: Adding `audio_pipeline/interjector.py` to the default import path initially pulled in optional runtime dependencies such as `numpy` and `jinja2` during test collection, which broke non-audio test environments before the fallback timing path could even run.
- Solution: Keep Stage-4 optional dependencies lazy at module boundaries: defer `PromptManager` and waveform-decoding imports until the code path actually needs them, and keep `llm_provider.py` type-only `numpy` imports behind postponed annotations so the runtime can import without the embedding stack installed.

### 2026-03-07T15:11:52+08:00 - Scrapling Policy Drifted Back To Generic Web Search
- Problem: The repo-level guidance said Scrapling was the default web-retrieval path, but the prompt wording was still soft enough that longer sessions could drift back to generic web-search and browsing behavior.
- Solution: Strengthen `AGENTS.md` and `coder_docs/scrapling.md` so Scrapling is the required open-web workflow except for narrow discovery-only fallback, and update `coder_docs/codebase_guide.md` to point at `AGENTS.md` as the active repo-local prompt surface.

### 2026-03-07T15:01:58+08:00 - Startup Cold Path Pulled In Unrelated Provider Stacks
- Problem: `providers.__init__` eagerly imported every provider backend, so importing `main.py` or `providers.logging_provider` also imported `sentence_transformers` and `transformers` even when the runtime used DeepSeek with `NoOpEmbeddingProvider`. That made the startup critical path pay for unrelated local-model stacks before the summarizer could start.
- Solution: Replace package-level provider exports with lazy `__getattr__` resolution, keep startup waits on a shared parallel A2A health poll instead of a fixed sleep, and use `audio.speaker_samples.defer_until_voice_clone` for fast-start runs that can postpone speaker-sample generation until stage-3.

## Entry Template

### YYYY-MM-DDTHH:MM:SS+TZ:TZ - Short Title
- Problem: Describe the error, regression, or pitfall that occurred.
- Solution: Describe the fix, workaround, or decision that resolved it.
