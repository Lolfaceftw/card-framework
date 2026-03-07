# Errors And Notes

Use this file as the repository memory for problems worth avoiding in future sessions and the fixes that resolved them.

Add new entries at the top so the newest lessons are visible first.
Record each entry with both the date and time, preferably in ISO 8601 format with timezone information.

### 2026-03-07T15:01:58+08:00 - Startup Cold Path Pulled In Unrelated Provider Stacks
- Problem: `providers.__init__` eagerly imported every provider backend, so importing `main.py` or `providers.logging_provider` also imported `sentence_transformers` and `transformers` even when the runtime used DeepSeek with `NoOpEmbeddingProvider`. That made the startup critical path pay for unrelated local-model stacks before the summarizer could start.
- Solution: Replace package-level provider exports with lazy `__getattr__` resolution, keep startup waits on a shared parallel A2A health poll instead of a fixed sleep, and use `audio.speaker_samples.defer_until_voice_clone` for fast-start runs that can postpone speaker-sample generation until stage-3.

## Entry Template

### YYYY-MM-DDTHH:MM:SS+TZ:TZ - Short Title
- Problem: Describe the error, regression, or pitfall that occurred.
- Solution: Describe the fix, workaround, or decision that resolved it.
