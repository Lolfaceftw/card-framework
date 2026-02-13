# FLOW: CARD Runtime and Automation Pipelines

This document is the implementation-aligned flow map for the CARD automated pipeline in this repository.  
All sequences below are based on current code paths under `audio2script_and_summarizer/`, `scripts/`, and `benchmarks/`.

## 1) System Boundary and Components

| Component | Responsibility | Primary Path |
| --- | --- | --- |
| Operator / CLI | Starts pipeline runs and sets flags/env | `python -m audio2script_and_summarizer` |
| Runtime Orchestrator | Controls Stage 1 -> 3, branch modes, duration loop, UI streaming | `audio2script_and_summarizer/run_pipeline.py` |
| DeepSeek Wrapper | Forces `--llm-provider deepseek` then delegates to runtime orchestrator | `audio2script_and_summarizer/run_pipeline_deepseek.py` |
| Stage 1: Diarization + Transcript | Optional Demucs, Whisper transcription, alignment, diarization, transcript outputs | `audio2script_and_summarizer/diarize.py` |
| Stage 1.5: Speaker Splitter | Extracts per-speaker prompt WAV clips from diarization JSON | `audio2script_and_summarizer/audio_splitter.py` |
| Stage 1.75: WPM Calibration | Derives speaking rate from TTS preflight or transcript timing | `audio2script_and_summarizer/tts_pacing_calibration.py`, `audio2script_and_summarizer/transcript_wpm.py` |
| Stage 2: OpenAI Summarizer | Produces summary JSON with source/speaker validation | `audio2script_and_summarizer/summarizer.py` |
| Stage 2: DeepSeek Summarizer | Produces summary JSON with agentic/local tool loop and report sidecar | `audio2script_and_summarizer/deepseek/core.py` via `audio2script_and_summarizer/summarizer_deepseek.py` |
| Stage 3: Voice Resynthesis | IndexTTS2 synthesis, optional Mistral interjections, merge/export | `audio2script_and_summarizer/stage3_voice.py` |
| DeepSeek Stream Router | Parses `[DEEPSEEK_STREAM]` markers and updates dashboard | `audio2script_and_summarizer/deepseek/stream_events.py` |
| Quality Gate Automation | Runs `ruff`, `mypy`, `pytest`, validates JUnit, writes evidence artifacts | `scripts/quality_gate.py` |
| Voice Clone Benchmark | Runs objective/subjective evaluation and writes benchmark artifacts | `benchmarks/voice_clone/runner.py` |

## 2) Primary End-to-End Runtime Flow

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator
    participant RP as run_pipeline.py
    participant S1 as Stage 1 diarize.py
    participant S15 as Stage 1.5 audio_splitter.py
    participant S175 as Stage 1.75 WPM calibration
    participant S2 as Stage 2 summarizer
    participant S3 as Stage 3 stage3_voice.py
    participant FS as Filesystem

    U->>RP: Start run (input audio, provider, duration settings)
    RP->>S1: Subprocess: python -m audio2script_and_summarizer.diarize
    S1->>FS: Write <base>.json, <base>.txt, <base>.srt
    S1-->>RP: Stage 1 complete

    RP->>S15: Subprocess: python -m audio2script_and_summarizer.audio_splitter
    S15->>FS: Write <base>_voices/*.wav
    S15-->>RP: Stage 1.5 complete

    RP->>S175: In-process WPM derivation (tts_preflight or transcript)
    S175-->>RP: avg_wpm + per_speaker_wpm
    RP->>RP: Compute initial word_budget = round(avg_wpm * target_minutes)

    RP->>S2: Run summarizer pass with word budget
    S2->>FS: Write <base>_summary.json
    S2->>FS: Write <base>_summary.json.report.json
    S2-->>RP: Stage 2 pass complete

    RP->>S3: Run Stage 3 on summary JSON
    S3->>FS: Write <base>_summary_resynth.wav
    S3->>FS: Write <base>_summary_resynth_interjections.json
    S3-->>RP: Measured output duration + interjection metrics

    RP->>FS: Patch summary report with duration metrics
    RP-->>U: Pipeline complete
```

Entry condition: full mode (no `--skip-a2s`, no `--skip-a2s-summary`).  
Exit artifacts: transcript artifacts, speaker WAVs, summary JSON/report, final resynthesized WAV, interjection log.

## 3) Runtime Branches and Modes

### 3.1 Provider Branch (OpenAI vs DeepSeek)

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator
    participant DSW as run_pipeline_deepseek.py
    participant RP as run_pipeline.py
    participant OAI as summarizer.py
    participant DSK as deepseek/core.py
    participant FS as Filesystem

    alt DeepSeek wrapper entrypoint
        U->>DSW: python -m audio2script_and_summarizer.run_pipeline_deepseek ...
        DSW->>RP: Inject --llm-provider deepseek, call main()
    else Direct runtime entrypoint
        U->>RP: python -m audio2script_and_summarizer.run_pipeline --llm-provider ...
    end

    alt llm_provider = openai
        RP->>OAI: Stage 2 subprocess with --api-key, --word-budget, --target-minutes
        OAI->>FS: Write <base>_summary.json and .report.json
    else llm_provider = deepseek
        RP->>DSK: Stage 2 subprocess with DeepSeek key and agent flags
        DSK->>FS: Write <base>_summary.json and .report.json
    end
```

### 3.2 `--skip-a2s` Branch

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator
    participant RP as run_pipeline.py
    participant DISC as transcript discovery
    participant S175 as Stage 1.75
    participant S2 as DeepSeek summarizer
    participant S3 as Stage 3
    participant FS as Filesystem

    U->>RP: Run with --skip-a2s
    RP->>DISC: Scan --skip-a2s-search-root for transcript JSON candidates
    DISC-->>RP: Selected transcript JSON path
    RP->>FS: Verify <base>_voices directory has WAV files
    RP->>S175: Derive avg_wpm/per_speaker_wpm
    RP->>S2: Run Stage 2 (forced DeepSeek provider in this mode)
    S2->>FS: Write <base>_summary.json + .report.json
    RP->>S3: Run Stage 3 unless --skip-stage3
    S3->>FS: Write resynth WAV + interjection log
```

### 3.3 `--skip-a2s-summary` Branch

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator
    participant RP as run_pipeline.py
    participant DISC as summary discovery
    participant S3 as Stage 3
    participant FS as Filesystem

    U->>RP: Run with --skip-a2s-summary
    alt --summary-json provided
        RP->>FS: Resolve and validate provided summary path
    else discover latest summary
        RP->>DISC: Find newest *_summary.json under --skip-a2s-search-root
        DISC-->>RP: Selected summary path
    end
    RP->>S3: Run Stage 3 directly from summary JSON
    S3->>FS: Write <summary_stem>_resynth.wav
    S3->>FS: Write <summary_stem>_resynth_interjections.json
```

### 3.4 `--skip-stage3` Branch

```mermaid
sequenceDiagram
    autonumber
    participant RP as run_pipeline.py
    participant S2 as Stage 2 summarizer
    participant CAL as tts_pacing estimate
    participant FS as Filesystem

    RP->>S2: Run summarizer pass
    S2->>FS: Write <base>_summary.json + .report.json
    alt tts_preflight calibration exists
        RP->>CAL: Estimate duration from summary text
        CAL-->>RP: Estimated seconds (informational)
    end
    RP->>FS: Patch summary report with measured_duration_seconds = null
    RP-->>RP: Finish with Stage 3 skipped
```

## 4) Stage-Level Internal Sequences

### 4.1 Stage 1 Internal (Diarization + Transcript Generation)

```mermaid
sequenceDiagram
    autonumber
    participant RP as run_pipeline.py
    participant D as diarize.py
    participant DEM as Demucs htdemucs
    participant WH as Whisper (faster-whisper)
    participant FA as Forced Aligner
    participant DIA as Diarizer (pyannote/msdd)
    participant FS as Filesystem

    RP->>D: Launch subprocess with input audio/device flags
    alt stemming enabled (default)
        D->>DEM: Source separation
        DEM-->>D: vocals.wav (or failure fallback)
    else --no-stem
        D-->>D: Use original audio directly
    end

    D->>WH: Transcribe audio
    D->>FA: Forced alignment + sentence mapping
    D->>DIA: Speaker diarization
    D->>FS: Write speaker-aware transcript <base>.txt
    D->>FS: Write subtitle file <base>.srt
    D->>FS: Write grouped segment JSON <base>.json
    D-->>RP: Exit 0
```

Entry condition: `--input` path is available in full mode.  
Primary Stage 1 outputs consumed downstream: `<base>.json` (by Stage 1.5/1.75/2), `<base>.txt`, `<base>.srt`.

### 4.2 Stage 1.5 Internal (Speaker Sample Extraction)

```mermaid
sequenceDiagram
    autonumber
    participant RP as run_pipeline.py
    participant SPL as audio_splitter.py
    participant FS as Filesystem

    RP->>SPL: Launch subprocess with --audio, --json, --output-dir
    SPL->>FS: Load diarization segments from <base>.json
    SPL->>FS: Extract speaker intervals and clip WAV prompts
    SPL->>FS: Write <base>_voices/<speaker>.wav
    SPL-->>RP: Exit 0
```

### 4.3 Stage 1.75 Internal (WPM Source Branch)

```mermaid
sequenceDiagram
    autonumber
    participant RP as run_pipeline.py
    participant TTS as tts_pacing_calibration
    participant TR as transcript_wpm
    participant IDX as IndexTTS2 checkpoints

    alt --wpm-source tts_preflight (default)
        RP->>TTS: calibrate_tts_pacing_profiles(voice_dir, cfg_path, model_dir)
        TTS->>IDX: Load IndexTTS2 checkpoints
        TTS-->>RP: calibration profile
        RP->>TTS: estimate_weighted_wpm_from_transcript(...)
        TTS-->>RP: avg_wpm + per_speaker_wpm
    else --wpm-source transcript
        RP->>TR: compute_wpm_from_transcript(<base>.json)
        TR-->>RP: avg_wpm + per_speaker_wpm
    end
```

### 4.4 Stage 2 + Stage 3 Duration Correction Loop

```mermaid
sequenceDiagram
    autonumber
    participant RP as run_pipeline.py
    participant S2 as Stage 2 summarizer
    participant S3 as Stage 3
    participant FS as Filesystem

    RP->>RP: Initialize current_word_budget and pass counter
    loop Until tolerance met or max correction passes reached
        RP->>S2: Run Stage 2 with current_word_budget
        S2->>FS: Write <base>_summary.json + .report.json
        alt --skip-stage3
            RP-->>RP: Break loop (no measured duration)
        else Stage 3 enabled
            RP->>S3: Run Stage 3 from summary
            S3-->>RP: measured_duration_seconds
            RP->>RP: Compute delta = measured - target
            alt abs(delta) <= duration_tolerance
                RP-->>RP: Break loop (target met)
            else correction budget remains
                RP->>RP: corrected_budget = round(current * target/measured)
                RP->>RP: Continue next pass with corrected budget
            else no correction budget left
                RP-->>RP: Break loop (outside tolerance, max passes reached)
            end
        end
    end
    RP->>FS: Update .report.json with final duration metrics
```

### 4.5 DeepSeek Agentic Tool Loop (Stage 2 Internal)

```mermaid
sequenceDiagram
    autonumber
    participant RP as run_pipeline.py
    participant DS as deepseek/core.py
    participant API as DeepSeek API
    participant TOOL as Local tool handlers
    participant DASH as Dashboard stream router
    participant FS as Filesystem

    RP->>DS: Launch Stage 2 DeepSeek subprocess
    DS->>FS: Initialize staged output buffer (.agent_buffer)
    DS->>API: Start streamed completion with tools=auto
    DS->>DASH: Emit stream start/token/context_usage events

    loop Up to agent_max_tool_rounds
        API-->>DS: Tool call request
        DS->>TOOL: Execute one tool call
        alt read_transcript_lines
            TOOL-->>DS: Transcript slice by index range
        else count_words
            TOOL-->>DS: Word counts
        else evaluate_script_constraints
            TOOL-->>DS: pass/fail + hints + repaired dialogue
        else write_output_segment
            TOOL->>FS: Overwrite/append staged JSON buffer
            TOOL-->>DS: Segment write status
        end
        DS->>API: Return tool outputs
    end

    alt loop exhausted and auto_salvage enabled
        DS->>FS: Read staged output buffer
        DS->>TOOL: Local final constraint validation
        alt staged output valid
            DS-->>DS: Accept salvaged staged output
        else invalid/missing staged output
            DS-->>DS: Raise tool_loop_exhausted error path
        end
    end

    DS->>FS: Write final summary JSON
    DS->>FS: Write summary report JSON sidecar
    DS->>DASH: Emit summary_json_ready + done stream events
    DS-->>RP: Exit with success/failure status
```

### 4.6 OpenAI Summarizer Validation/Retry Flow

```mermaid
sequenceDiagram
    autonumber
    participant RP as run_pipeline.py
    participant OA as summarizer.py
    participant API as OpenAI API
    participant VAL as speaker_validation
    participant FS as Filesystem

    RP->>OA: Launch Stage 2 OpenAI subprocess
    loop Up to max_retries + 1 attempts
        OA->>API: Request structured summary JSON
        API-->>OA: Candidate response
        OA->>VAL: Validate schema/speaker/source IDs
        alt valid and within budget
            OA->>FS: Write <base>_summary.json
            OA->>FS: Write <base>_summary.json.report.json
            OA-->>RP: Success
        else invalid/truncated/budget failure
            OA-->>OA: Build retry instruction and attempt again
        end
    end
```

### 4.7 Stage 3 Internal (IndexTTS2 + Optional Interjections)

```mermaid
sequenceDiagram
    autonumber
    participant RP as run_pipeline.py
    participant S3 as stage3_voice.py
    participant TTS as IndexTTS2Engine
    participant M as MistralInterjectionPlanner
    participant FS as Filesystem

    RP->>S3: run_stage3_pipeline(summary_json_path, output_path?, device, planner settings)
    S3->>FS: Load and validate summary entries
    S3->>FS: Resolve each voice_sample path

    loop For each summary entry
        S3->>TTS: infer(main segment audio)
        TTS->>FS: Write temp main segment WAV
    end

    S3->>M: ensure_available()
    alt planner available
        loop For eligible segments
            S3->>M: propose interjection(candidate speaker/context)
            M-->>S3: candidate or none
        end
        S3-->>S3: Select top candidates by confidence and ratio cap
        loop For selected candidates
            S3->>TTS: infer(interjection audio)
            TTS->>FS: Write temp interjection WAV
            S3-->>S3: Overlay interjection at computed anchor+style delay
        end
    else planner unavailable
        S3-->>S3: Degrade gracefully (no interjections)
    end

    S3-->>S3: Merge segments with pauses/crossfades
    S3->>FS: Export final WAV (<summary_stem>_resynth.wav)
    S3->>FS: Export interjection log JSON (<summary_stem>_resynth_interjections.json)
    S3-->>RP: Return Stage3Result(duration/interjection_count/mistral_enabled)
```

## 5) Artifact Lineage and Contracts

| Artifact | Producer | Consumer(s) | Notes |
| --- | --- | --- | --- |
| `<base>.json` | Stage 1 `diarize.py` | Stage 1.5, Stage 1.75, Stage 2 | Grouped diarized transcript segments |
| `<base>.txt` | Stage 1 | Human/operator | Speaker-aware transcript text |
| `<base>.srt` | Stage 1 | Human/operator/media tools | Subtitle transcript |
| `<base>_voices/*.wav` | Stage 1.5 `audio_splitter.py` | Stage 2, Stage 3 | Per-speaker prompt clips |
| `<base>_summary.json` | Stage 2 summarizer | Stage 3 | Structured dialogue lines with speaker + voice_sample + emotion controls |
| `<base>_summary.json.report.json` | Stage 2 summarizer, then patched by runtime | Operator, audit scripts | Includes validation/retry/tool-loop metrics; runtime adds duration delta/tolerance fields |
| `<summary_stem>_resynth.wav` | Stage 3 | Operator/downstream media | Final synthesized audio |
| `<summary_stem>_resynth_interjections.json` | Stage 3 | Operator/debugging | Interjection placement metadata |
| `artifacts/quality/<run_id>/report.json` | `scripts/quality_gate.py` | Merge/review verification | Canonical machine-readable quality evidence |
| `artifacts/quality/<run_id>/pytest.junit.xml` | `scripts/quality_gate.py` | Evidence validation | Must report tests > 0 and failures/errors = 0 for valid pass claim |
| `benchmarks/runs/<timestamp>/metrics_summary.json` | Benchmark runner | Analysts | Aggregate benchmark metrics + artifact pointers |

## 6) Auxiliary Automated Pipelines

### 6.1 Quality Gate Evidence Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator/CI
    participant Q as scripts/quality_gate.py
    participant UV as uv run
    participant RF as ruff
    participant MY as mypy
    participant PY as pytest
    participant FS as Filesystem

    U->>Q: python scripts/quality_gate.py
    Q->>FS: Create artifacts/quality/<run_id>/

    Q->>UV: Run ruff check on core scope
    UV->>RF: Execute lint
    RF-->>Q: Exit code + stdout/stderr
    Q->>FS: Write ruff stdout/stderr logs

    Q->>UV: Run mypy on core scope
    UV->>MY: Execute type check
    MY-->>Q: Exit code + stdout/stderr
    Q->>FS: Write mypy stdout/stderr logs

    Q->>UV: Run pytest with --junitxml
    UV->>PY: Execute tests
    PY-->>Q: Exit code + JUnit XML + logs
    Q->>FS: Write pytest stdout/stderr logs + junit XML

    Q->>Q: Parse JUnit totals and validate evidence rules
    Q->>FS: Write report.json + report.md + SHA256 in report.md
    Q-->>U: Exit 0 if all checks pass and evidence valid, else exit 1
```

### 6.2 Voice Clone Benchmark Pipeline

```mermaid
sequenceDiagram
    autonumber
    participant U as Operator
    participant B as benchmarks/voice_clone/runner.py
    participant M as Manifest loader
    participant TTS as IndexTTS2Engine
    participant EMB as Speaker embedder
    participant MET as Metrics engine
    participant ASR as Text metrics (optional)
    participant MOS as MOS kit writer (optional)
    participant FS as Filesystem

    U->>B: Run benchmark CLI (--manifest, --cfg-path, --model-dir, ...)
    B->>M: Load and validate manifest rows
    M-->>B: Validated benchmark items

    loop For each manifest item
        B->>TTS: Generate cloned utterance
        TTS->>FS: Write generated/<item>.wav
    end

    B->>EMB: Embed generated + reference clips
    EMB-->>B: Embedding vectors
    B->>MET: Compute pair scores, retrieval accuracy, ASV-EER inputs
    MET->>FS: Write pair_scores.csv

    opt ASR metrics enabled
        B->>ASR: Transcribe generated audio
        ASR->>FS: Write text_scores.csv (WER/CER)
    end

    opt MOS pack enabled
        B->>MOS: Build blinded A/B audio package
        MOS->>FS: Write mos_pack/* (pairs, templates, instructions, key)
    end

    B->>FS: Write metrics_summary.json + run.log
    B-->>U: Benchmark run complete
```

## 7) Failure, Retry, and Degradation Paths

| Area | Failure Mode | Runtime Behavior |
| --- | --- | --- |
| Stage subprocesses | Non-zero exit from Stage 1/1.5/2 | Runtime stops and returns failure |
| Stage 1 source separation | Demucs failure | Falls back to original audio for transcription |
| Stage 1 environment | `HF_TOKEN` missing | Interactive prompt in `diarize.py` import/main path; may block non-interactive execution |
| Stage 1.5 outputs | No WAV clips produced | Runtime fails with explicit error |
| Stage 2 credentials | Missing OpenAI/DeepSeek API key | Runtime fails before summarizer run |
| Stage 2 schema/constraints | Invalid model output | Summarizer retry loop (OpenAI and DeepSeek paths) |
| DeepSeek tool loop | Max rounds exhausted | Auto-salvage staged output if configured; otherwise fail with diagnostics |
| Stage 3 planner | Mistral unavailable | Degraded success with zero interjections; synthesis still completes |
| Duration control | Output outside tolerance | Re-run Stage 2/3 with corrected word budget until max correction passes reached |
| Quality gate evidence | Missing/invalid JUnit or failing checks | `report.json` status `fail`; script exits non-zero |

## 8) Observability and Logs

| Channel | Source | What it Captures |
| --- | --- | --- |
| Runtime orchestrator logs | `run_pipeline.py` via `logging_utils.configure_logging` | Stage transitions, subprocess command lifecycle, duration checks |
| Dashboard stream pane | `run_pipeline.py` + `deepseek/stream_events.py` | DeepSeek token stream, context usage, completion markers |
| Stage-specific logs | stage modules using `logging` | Stage internals and warnings |
| DeepSeek chat traces | `deepseek/core.py` | NDJSON trace events and run metadata under `deepseek_chat_logs/` |
| Quality evidence logs | `scripts/quality_gate.py` | Per-step stdout/stderr and consolidated report |
| Benchmark logs | `benchmarks/voice_clone/runner.py` | Run lifecycle, metric summary pointers, and `run.log` |

## 9) Reality Notes (Current Repo vs Stale Docs)

1. `run_pipeline.py` currently has no `--config` or `CARD_CONFIG_PATH` support in code, even though root `README.md` references config-based output routing.
2. Root `README.md` references `scripts/run_audio2script.py`, `scripts/run_audio2script_deepseek.py`, and `scripts/run_separation.py`; these files are not present in the current repository tree.
3. Root `README.md` references `docs/CONFIGURATION.md`, `docs/ARCHITECTURE.md`, and `docs/STRUCTURE.md`; a `docs/` directory is not present in the current repository tree.
4. The canonical runtime orchestrator in this repo is `audio2script_and_summarizer/run_pipeline.py`, with `run_pipeline_deepseek.py` as the DeepSeek-only shim.
