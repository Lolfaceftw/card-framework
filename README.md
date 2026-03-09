# CARD Framework

This repository is the current implementation of **CARD: Constraint-aware Audio
Resynthesis and Distillation**, the project described in
[`EEE_196_CARD_UCL.md`](./EEE_196_CARD_UCL.md).

The paper is the conceptual and academic baseline. The codebase, however, has
already moved beyond parts of the manuscript's original implementation plan.
This README therefore prioritizes **what the repository actually does now**.
When the paper and the current code diverge, treat the code, config, and
`coder_docs` as the source of truth for day-to-day development.

## Paper Metadata

**Authors**

- Rei Dennis Agustin, 2022-03027, BS Electronics Engineering
- Sean Luigi P. Caranzo, 2022-05398, BS Computer Engineering
- Johnbell R. De Leon, 2021-01437, BS Computer Engineering
- Christian Klein C. Ramos, 2022-03126, BS Electronics Engineering

**Research Adviser**

- Rowel D. Atienza

**Affiliation**

- University of the Philippines Diliman
- December 2025

## Abstract

CARD addresses the long-form podcast consumption bottleneck by generating a
shorter conversational audio output that retains speaker identity and
prosodic character instead of collapsing everything into plain text. The
project combines transcript generation, speaker-aware summarization,
voice-cloned resynthesis, and conversational overlap handling so a
multi-speaker recording can be compressed toward a user-defined duration
without discarding the listening experience that makes the original medium
valuable.

## High-Level Architecture

```mermaid
flowchart LR
    A[Source Audio] --> B[Stage 1<br/>Audio Ingestion]
    B --> C[Transcript JSON<br/>Speaker Metadata]
    C --> D[Stage 2<br/>Summarizer + Critic Loop]
    D --> E[Summary XML<br/>Speaker-Tagged Turns]
    E --> F[Stage 3<br/>Voice Clone Resynthesis]
    F --> G[Cloned Summary Audio]
    G --> H[Stage 4<br/>Interjector / Backchannels]
    H --> I[Final Conversational Audio]

    C -. Optional evaluation input .-> J[Benchmarks]
    E -. Optional evaluation input .-> J

    K[Hydra Config + Provider Adapters] -. controls .-> B
    K -. controls .-> D
    K -. controls .-> F
    K -. controls .-> H
```

## What CARD Does

CARD is a multi-stage pipeline for converting long-form multi-speaker audio into
a shorter, speaker-aware, resynthesized conversational output.

At a high level, the repository currently supports:

- **Stage 1: Audio ingestion and transcript generation**
  - Source separation
  - ASR, diarization, and alignment
  - Transcript JSON generation with speaker metadata
- **Stage 2: Constraint-aware summarization**
  - Summarizer and critic agent loop
  - Duration-first summary generation with speaker-tagged XML output
  - Retrieval-backed or full-transcript summarization paths
- **Stage 3: Voice cloning and resynthesis**
  - Speaker sample generation
  - Voice-cloned rendering of summary turns
  - Live-draft voice cloning during summarizer edits
- **Stage 4: Conversational interjection**
  - Optional overlap and backchannel synthesis on top of the cloned summary
- **Benchmarking and evaluation**
  - Summarization benchmark workflows
  - Source-grounded QA benchmark workflows
  - Diarization benchmark workflows

## Paper vs. Current Repository

[`EEE_196_CARD_UCL.md`](./EEE_196_CARD_UCL.md) explains the original CARD paper,
problem framing, and proposed module design. The repository now reflects a more
developed engineering system than that initial write-up.

Important differences from the manuscript-level description include:

- The repo is now **configuration-driven** through Hydra instead of being tied
  to one fixed experimental path.
- The runtime is now **duration-first**, centered on `target_seconds` and
  tolerance checks, rather than a simple word-budget-only workflow.
- The summary output contract is now **speaker-tagged XML**, which feeds the
  downstream voice-clone and interjector stages.
- The default stage-2/stage-3 flow can use **live-draft voice cloning**, where
  turn audio is rendered during summary editing instead of only after the final
  draft is approved.
- The repository includes substantial **benchmarking, evaluation, and operator
  tooling** that goes beyond the initial paper narrative.
- Provider support has expanded: the codebase is organized around adapters and
  config-selected backends rather than a single hardcoded model stack.

In short: the paper explains **why CARD exists**; this repository captures
**how CARD currently works**.

## Repository Layout

```text
src/card_framework/
  agents/           A2A executors, DTOs, tool loops, client transport
  audio_pipeline/   Audio ingestion, speaker samples, voice cloning, interjector
  benchmark/        Summarization, QA, and diarization benchmarks
  cli/              Runtime, setup, calibration, matrix, and eval entrypoints
  config/           Hydra configuration
  orchestration/    Transcript DTOs and stage orchestration
  prompts/          Jinja2 prompt templates
  providers/        LLM and embedding provider adapters
  retrieval/        Transcript indexing and retrieval
  runtime/          Runtime planning and execution support
  shared/           Shared utilities, events, and logging
  _vendor/index_tts/
```

Other important locations:

- `artifacts/`: generated transcripts, cloned audio, benchmark outputs, and
  other runtime artifacts
- `checkpoints/`: local model/runtime checkpoints
- `coder_docs/`: repository-specific architecture, workflow, and maintenance
  guidance

## Common Commands

```bash
uv sync --dev
uv run python -m card_framework.cli.main --help
uv run python -m card_framework.cli.setup_and_run --help
uv run python -m card_framework.cli.calibrate --help
uv run python -m card_framework.cli.run_summary_matrix --help
uv run python -m card_framework.benchmark.run --help
uv run python -m card_framework.benchmark.diarization --help
uv run python -m card_framework.benchmark.qa --help
uv run ruff check .
uv run pytest
```

Common execution entrypoints:

```bash
uv run python -m card_framework.cli.setup_and_run --audio-path <path-to-audio>
uv run python -m card_framework.cli.main
uv run python -m card_framework.cli.calibrate
```

## Package Usage

The repository now exposes a library entrypoint for installed-package use:

```bash
pip install card-framework
```

```python
from card_framework import infer

result = infer(
    "audio.wav",
    "outputs/run_001",
    300,
    device="cpu",
    vllm_url="http://localhost:8000/v1",
)
print(result.summary_xml_path)
print(result.final_audio_path)
```

`infer(audio_wav, output_dir, target_duration_seconds, *, device, ...)` runs
the full stage-1 to stage-4 pipeline and returns an `InferenceResult` with the
main emitted artifact paths. `target_duration_seconds` is required for every
call and overrides any duration target declared in the loaded config file.
`device` is also required and must be either `cpu` or `cuda`. `vllm_url` is the
first-class packaged-runtime override for OpenAI-compatible endpoints, and it
forces the shared summarizer, critic, and interjector LLM path onto the
provided vLLM-compatible server for that call. The call writes into `output_dir`
using this high-level layout:

```text
outputs/run_001/
  transcript.json
  summary.xml
  agent_interactions.log
  audio_stage/
    voice_clone/
    interjector/
```

Installed-package runtime notes:

- Supported public packaged-runtime platform as of March 9, 2026: Windows only.
  macOS and Linux are not yet validated for the public `pip install
  card-framework` whole-pipeline path, and `infer(...)` now fails fast on those
  platforms instead of attempting a partial run.
- `CARD_FRAMEWORK_CONFIG`: optional path to a full YAML config file when you
  need to override the default packaged provider/runtime config for `infer(...)`.
- `CARD_FRAMEWORK_HOME`: optional writable runtime home used for extracted
  IndexTTS assets, checkpoints, and bootstrap state. If unset, the package uses
  the platform-appropriate user data directory.
- `CARD_FRAMEWORK_VLLM_URL`: optional environment-variable equivalent of the
  `vllm_url=` argument.
- `CARD_FRAMEWORK_VLLM_API_KEY`: optional environment-variable equivalent of
  the `vllm_api_key=` argument. If omitted for vLLM, the packaged runtime uses
  `EMPTY`, which matches the common local keyless vLLM setup.
- If you choose `device="cuda"`, the packaged runtime currently supports only
  CUDA 12.6. `infer(...)` now validates that the installed PyTorch build reports
  CUDA 12.6 before it proceeds.
- The packaged default is now vLLM-first. If the effective config selects
  another provider, `infer(...)` resolves required credentials before it starts
  the subprocess runtime:
  - interactive terminals: `infer(...)` securely prompts for missing API keys
    or access tokens without echoing them and without placing them on the
    subprocess command line
  - non-interactive runs: `infer(...)` fails fast with an actionable error that
    names the missing config field and the supported environment variable
- Supported credential environment variables for the packaged path include
  `DEEPSEEK_API_KEY`, `GEMINI_API_KEY` or `GOOGLE_API_KEY`, `ZAI_API_KEY`,
  `HUGGINGFACE_TOKEN` or `HF_TOKEN`, and the configured
  `audio.diarization.pyannote.auth_token_env` value.
- Whole-pipeline inference still requires external tools such as `ffmpeg`.
  When voice cloning or calibration paths are active, the package also expects
  `uv` so it can bootstrap the vendored IndexTTS runtime in the writable
  runtime home on first use.

## Public PyPI Release

This repository now includes a GitHub Actions trusted-publishing workflow at
`.github/workflows/publish-pypi.yml` that publishes tags matching `v*` to PyPI.

For the first public release of `card-framework`, use PyPI's **pending
publisher** flow because the project does not exist on PyPI yet. Configure:

- PyPI project name: `card-framework`
- GitHub owner: `Lolfaceftw`
- GitHub repository: `card-framework`
- Workflow filename: `publish-pypi.yml`
- Environment name: `pypi`

Repository-side release steps:

1. Merge the publishing workflow to `main`.
2. In GitHub repository settings, create the `pypi` environment.
3. In PyPI account settings, add the pending trusted publisher with the fields
   above.
4. Tag the release from `main` and push it, for example:

   ```bash
   git tag -a v0.1.0 -m v0.1.0
   git push origin v0.1.0
   ```

5. After the workflow succeeds, verify the public release:

   ```bash
   python -m pip install --no-cache-dir card-framework
   python -c "from card_framework import infer; print(infer)"
   ```

## Documentation

- [`EEE_196_CARD_UCL.md`](./EEE_196_CARD_UCL.md): the CARD paper and project
  manuscript
- [`coder_docs/codebase_guide.md`](./coder_docs/codebase_guide.md): current
  architecture, runtime flow, commands, and maintenance expectations
- [`coder_docs/memory/errors_and_notes.md`](./coder_docs/memory/errors_and_notes.md):
  repository memory for recurring pitfalls and prior fixes
- [`coder_docs/fault_localization_workflow.md`](./coder_docs/fault_localization_workflow.md):
  bug triage and failing-test workflow

If you are changing behavior, prompts, workflows, or commands, start with
`coder_docs/codebase_guide.md`.

## License

This repository is source-available under
[`LICENSE.md`](./LICENSE.md), using the **PolyForm Noncommercial 1.0.0**
license. Noncommercial use is allowed; commercial use requires separate
permission from the licensors.
