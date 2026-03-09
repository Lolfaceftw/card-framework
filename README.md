# card-framework

This repository now uses a standard `src` layout. Maintained Python source
lives under `src/card_framework`, and vendored third-party runtime code lives
under `src/card_framework/_vendor/index_tts`.

## Layout

```text
src/card_framework/
  cli/
  runtime/
  shared/
  retrieval/
  agents/
  audio_pipeline/
  benchmark/
  orchestration/
  providers/
  prompts/templates/
  config/
  _vendor/index_tts/
```

Runtime checkpoints stay outside the package under `checkpoints/index_tts`, and
generated artifacts stay under `artifacts/`.

## Common Commands

```bash
uv sync --dev
uv run python -m card_framework.cli.main --help
uv run python -m card_framework.cli.setup_and_run --help
uv run python -m card_framework.cli.calibrate --help
uv run python -m card_framework.cli.run_summary_matrix --help
uv run pytest tests/real -q
```

## Documentation

- `coder_docs/codebase_guide.md` is the source of truth for repo architecture,
  workflows, commands, and maintenance expectations.
- `coder_docs/fault_localization_workflow.md` is the source of truth for
  agent-driven fault localization and failing-test triage in this repository.

## Summary Matrix Runner

Use `card_framework.cli.run_summary_matrix` when you want the repository's
normal `card_framework.cli.setup_and_run` summarizer and critic workflow across
the built-in Qwen ordered-pair matrix, with optional DeepSeek coverage, while
saving one summary XML per model pair.

```bash
uv run python -m card_framework.cli.run_summary_matrix \
  --vllm-host <host> \
  --transcript-path <path-to-transcript.json> \
  --deepseek-api-key <deepseek-api-key>
```

Notes:

- Output files are written under `artifacts/summary_matrix/<timestamp>` by
  default.
- Each summary file is named `<summarizer>_<critic>-summary.xml`.
- The helper preserves the repo-default merged stage-2/stage-3 live-draft
  voice-clone path and disables only stage-4 interjector output.
- Prefer `--transcript-path` for repeated matrix runs so the helper can stay on
  the faster stage-2 path. Use `--audio-path` when you need a fresh stage-1
  transcript first.
