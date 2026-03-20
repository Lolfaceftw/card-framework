# Docker Workflow

Use this file as the source of truth for the repository's containerized runtime
workflow.

The Docker path is centered on `card_framework.cli.setup_and_run`. It is the
recommended way to run the full pipeline on a Windows host when you need a
Linux execution environment for NeMo-backed diarization.

## Why This Exists

- Native Windows NeMo is not upstream-supported.
- Docker Desktop with Linux containers gives the repo a consistent Linux
  runtime on Windows, Linux, and macOS hosts.
- Keeping `setup_and_run` as the main entrypoint preserves the current
  bootstrap behavior: dependency sync, vendored IndexTTS setup, model
  provisioning, stage selection, calibration decisions, and full pipeline
  execution.

## Current Files

- `Dockerfile`: base runtime image with Python 3.12, `uv`, `ffmpeg`, and build
  tools.
- `compose.yml`: operator-facing services for `setup_and_run` CPU and GPU runs.
- `docker/setup_and_run_entrypoint.py`: small wrapper that executes
  `uv run python -m card_framework.cli.setup_and_run` from the mounted repo
  checkout.

## Design

- The container bind-mounts the repository checkout into
  `/workspace/card-framework`.
- The repo-root `conf/config.yaml` and `.env` stay the operator-facing config
  surface.
- The root `.venv` and the vendored
  `src/card_framework/_vendor/index_tts/.venv` are mounted to named Docker
  volumes so Linux virtual environments do not pollute the host checkout.
- Hugging Face and uv caches are also stored in named volumes so repeated runs
  reuse downloaded assets.
- The Compose services force `linux/amd64` so the runtime matches the Linux
  x86_64 path that NeMo supports upstream. On Apple Silicon that means Docker
  runs under emulation, so expect slower startup.

## Main Commands

Build the runtime image:

```bash
docker compose build setup-and-run
```

Show the wrapper help:

```bash
docker compose run --rm setup-and-run --help
```

Run setup only:

```bash
docker compose run --rm setup-and-run --setup-only
```

Run the full pipeline from source audio:

```bash
docker compose run --rm setup-and-run --audio-path audio.wav
```

Run the GPU profile for Windows or Linux NVIDIA hosts:

```bash
docker compose --profile gpu run --rm setup-and-run-gpu --override audio.device=cuda --audio-path audio.wav
```

Run only the stage-1 audio pipeline in the GPU container and stop after the
transcript is written:

```bash
docker compose --profile gpu run --rm --entrypoint bash setup-and-run-gpu -lc "uv run python scripts/run_audio_stage_only.py --audio-path audio.wav --device cuda"
```

Run only Granite ASR on an existing `vocals.wav` artifact and write raw chunked
transcription output plus seam-oriented metrics:

```bash
docker compose --profile gpu run --rm --entrypoint bash setup-and-run-gpu -lc "uv run python scripts/run_transcription_experiment.py --audio-path artifacts/audio_stage/runs/<run-id>/separation/htdemucs/audio/vocals.wav --device cuda --override audio.asr.forced_alignment.enabled=false --override audio.asr.forced_alignment.required=false"
```

Rerun only diarization against a fixed separated `vocals.wav` artifact and
write a quick speaker-preview transcript from an existing timed transcript:

```bash
docker compose --profile gpu run --rm --entrypoint bash setup-and-run-gpu -lc "uv run python scripts/run_diarization_experiment.py --audio-path artifacts/audio_stage/runs/<run-id>/separation/htdemucs/audio/vocals.wav --base-transcript-path artifacts/transcripts/<run-id>.transcript.json --device cuda"
```

Run stage-3 voice cloning directly from an existing summary:

```bash
docker compose run --rm setup-and-run --voiceclone-from-summary summary.xml
```

## Windows + NeMo Guidance

For Windows hosts, the supported path is:

1. Install Docker Desktop with the WSL 2 backend enabled.
2. Install an NVIDIA Windows driver that supports CUDA on WSL 2.
3. Keep `audio.diarization.provider=nemo` in `conf/config.yaml`.
4. Run the pipeline with `setup-and-run-gpu`.

This is a Linux container fallback on a Windows host. It is not native Windows
NeMo support.

The sampled-segment speaker prep helpers now auto-use this same Compose runtime
for their diarization-only speaker-sample bootstrap when:

- `audio.diarization.provider` is NeMo-derived, and
- the current host does not support NeMo or does not have the local NeMo
  runtime installed.

That means commands such as
`python -m card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets`
can keep using your repo-local config while still handing NeMo work off to the
Linux container automatically.

## CPU-Only Guidance

- `setup-and-run` is the generic CPU-oriented service.
- CPU-only hosts can still use the full wrapper flow, but NeMo itself remains a
  Linux-oriented dependency and may not be the right diarizer on every host.
- For broader host compatibility without NVIDIA GPU pass-through, use
  `audio.diarization.provider=pyannote_community1` or accept the repo's
  `single_speaker` fallback path where appropriate.

## Operator Notes

- Run Compose commands from the repository root.
- Copy `.env.example` to `.env` before the first containerized run.
- `setup_and_run` still auto-selects stage-2 when it discovers a reusable
  transcript in the mounted workspace.
- Relative paths such as `audio.wav`, `transcript.json`, and `summary.xml`
  resolve relative to the mounted repo checkout because the container runs from
  `/workspace/card-framework`.
- The GPU profile is intentionally separate so CPU hosts do not fail on `gpus:
  all`.
- For transcript tuning, prefer fixing one stage-1 run ID first and then reuse
  its `separation/.../vocals.wav` with `scripts/run_diarization_experiment.py`
  when you want to isolate NeMo changes from run-to-run Demucs variation.

## Near-Term Plan

1. Keep Docker centered on `setup_and_run` until the operator path is stable.
2. Add optional sidecar services later for local vLLM and related model
   backends when the runtime contract for those services is settled.
3. Keep expanding the same host-side Docker delegation pattern only where the
   local runtime truly cannot satisfy a configured backend, without silently
   moving unrelated stages into containers.
