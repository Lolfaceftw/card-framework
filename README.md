# llm-test

## Summary Matrix Runner

Use `scripts/run_summary_matrix.py` when you want the repository's normal
`setup_and_run.py` summarizer and critic workflow across the built-in Qwen
summarizer/critic ordered-pair matrix, with optional DeepSeek model coverage,
while saving one summary XML per model pair.

```bash
uv run python scripts/run_summary_matrix.py \
  --vllm-host <host> \
  --transcript-path <path-to-transcript.json> \
  --deepseek-api-key <deepseek-api-key>
```

Notes:

- Output files are written under `artifacts/summary_matrix/<timestamp>` by
  default.
- Each summary file is named
  `<summarizer>_<critic>-summary.xml`.
- The helper preserves the repo-default merged stage-2/stage-3 live-draft
  voice-clone path and disables only stage-4 interjector output.
- Prefer `--transcript-path` for repeated matrix runs so the helper can stay on
  the faster stage-2 path. Use `--audio-path` when you need a fresh stage-1
  transcript first.
- Each child `setup_and_run.py` invocation now streams its live token/output
  back to the parent terminal while also being captured into that pair's log
  file.
- Pass additional Hydra overrides with repeated
  `--override KEY=VALUE` flags. The helper owns the shared summarizer `llm.*`,
  `stage_llm.critic.*`, summary-only synthesis toggles, and per-pair loop
  memory artifact path.
