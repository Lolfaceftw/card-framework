# Errors And Notes

### 2026-03-07T18:44:23+08:00 - IndexTTS Emotion Presets And Beam Search Needed A Repo-Level Low-Latency Path
- Problem: Even after the warm IndexTTS runtime fix, repeated stage-3 and stage-4 turns still paid unnecessary latency because preset `emo_text` guidance re-ran Qwen emotion analysis on every synth call, and the vendored IndexTTS defaults still used `num_beams=3` for mel-token generation.
- Solution: Cache `emo_text -> emo_vector` inside the warm runtime boundary, pass cached vectors directly into IndexTTS inference so repeated preset prompts skip Qwen analysis, and expose generation knobs in repo config with `audio.voice_clone.num_beams=1` as the repo default for faster steady-state synthesis.

### 2026-03-07T18:35:35+08:00 - IndexTTS Subprocess Backend Must Reuse One Warm Runtime Until Explicit Offload
- Problem: The default `audio.voice_clone.execution_backend=subprocess` path launched a fresh `uv run` process for every single IndexTTS synth call. Calibration phrases, stage-3 voice cloning, and stage-4 interjections all reloaded the full GPT, semantic codec, s2mel, campplus, BigVGAN, and text-normalization stack instead of reusing one warm runtime.
- Solution: Replace the per-call subprocess launch with a shared persistent worker keyed by the effective IndexTTS config, share the same warm runtime across matching gateway instances, add explicit provider `close()` and `release_all_cached_resources()` hooks for real voice-clone-module offload paths, and keep the runtime loaded until that explicit release or process exit.

### 2026-03-07T18:34:00+08:00 - Live Rich Panels Must Collapse To One Stable Message After Streaming
- Problem: The terminal UI left the final streamed `Summarizer` or `Critic` Rich live panel resident after token streaming finished. When later status lines arrived from stage 3, some terminals made the last panel look like it had relaunched, creating a false impression that stage 2 and stage 3 were running simultaneously.
- Solution: End live agent panels with `transient=True`, then print one final static panel after teardown so later stage messages cannot replay the previous live frame.

### 2026-03-07T18:25:00+08:00 - setup_and_run Must Respect Repo Toggle Defaults Instead Of Overriding Stage-4 Off
- Problem: `setup_and_run.py` resolved synthesis toggles from CLI overrides only and then injected wrapper-local fallback values into the runtime. That meant a normal full or stage-2 run could silently override `conf/config.yaml` and force `audio.interjector.enabled=false`, preventing stage 4 even when the repo config enabled it.
- Solution: Read the nested boolean defaults directly from `conf/config.yaml` with a stdlib-only bootstrap parser, apply CLI overrides on top of those repo defaults, and add regression tests covering repo-default interjector enablement.

### 2026-03-07T18:07:47+08:00 - Stage-2 Must Reuse Existing Speaker Manifests And Rebuild Stale Calibration
- Problem: Reusable stage-2 transcripts could already carry a valid `metadata.speaker_samples_manifest_path`, but `main.py` still regenerated speaker samples from fallback `audio.audio_path`, which could shift perceived speaker identity. At the same time, `audio_pipeline.calibration.ensure_voice_clone_calibration` reused an old artifact even when the current transcript pointed at a different speaker-sample manifest, causing duration estimates and final stage-3 runtime to drift.
- Solution: Reuse transcript-linked speaker manifests for stage-2 instead of regenerating them, treat an existing speaker manifest as satisfying the stage-2 bootstrap audio fallback check, and rebuild calibration whenever the active speaker-sample manifest or calibrated speaker coverage no longer matches the current run.

### 2026-03-07T17:58:14+08:00 - setup_and_run Must Explain Auto-Selected Stage-2 And Calibration Inference Logs
- Problem: Reusable `transcript.json` artifacts made plain `setup_and_run.py` runs auto-start at stage-2, but the wrapper did not say that clearly. Operators could then mistake calibration-time IndexTTS inference logs for a simultaneous stage-3 voice-clone run while the summarizer was starting.
- Solution: Print the selected effective `pipeline.start_stage` with the reason, add a hint when stage-2 will rerun summarization before voice cloning even though `summary.xml` already exists, and warn that calibration may emit temporary IndexTTS inference logs before stage-2 or stage-3 work begins.

### 2026-03-07T17:51:13+08:00 - Loop Memory Must Remember Repeated Failed Remedies, Not Just Open Issues
- Problem: The summarizer already carried unresolved issue memory between critic passes, but it could still oscillate back to a previously attempted fix pattern when later critic feedback reintroduced the same remedy in slightly different words.
- Solution: Track normalized remedy-attempt signatures alongside unresolved issues, surface repeated-remedy alerts in the revise prompt, and persist the loop-memory artifact by transcript hash and duration target so later passes can avoid replaying the same failed fix verbatim.

### 2026-03-07T17:32:51+08:00 - A2A Startup Probe Must Prefer The Non-Deprecated Agent Card Path
- Problem: `agents/health.py` still probed `/.well-known/agent.json`, which now emits a deprecation warning because newer A2A servers expect `/.well-known/agent-card.json`.
- Solution: Probe `/.well-known/agent-card.json` first during startup health checks and fall back to the legacy path only when an older server does not expose the new endpoint.

Use this file as the repository memory for problems worth avoiding in future sessions and the fixes that resolved them.

Add new entries at the top so the newest lessons are visible first.
Record each entry with both the date and time, preferably in ISO 8601 format with timezone information.

### 2026-03-07T17:29:41+08:00 - UI Console Output Must Sanitize Unicode For Windows cp1252 Terminals
- Problem: A real `setup_and_run.py` stage-2 run reached the pipeline, but Rich UI subscribers started throwing `UnicodeEncodeError` on Windows because system messages included characters such as `→` that the active `cp1252` console could not encode.
- Solution: Sanitize operator-facing console text before printing, replacing common Unicode punctuation with ASCII and falling back to encoding-safe replacement for the active terminal encoding.

### 2026-03-07T17:27:11+08:00 - Bootstrap Transcript Metadata Readers Must Tolerate UTF-8 BOM Files
- Problem: The repo-root `transcript.json` loaded fine in the main runtime, but the new bootstrap metadata inspection in `setup_and_run.py` rejected the same file because it was saved with a UTF-8 BOM.
- Solution: Read bootstrap transcript metadata with `utf-8-sig` so Windows/BOM-authored transcript artifacts remain valid stage-2 inputs.

### 2026-03-07T17:19:32+08:00 - setup_and_run Must Reuse Existing Transcripts And Backfill Stage-2 Audio
- Problem: `setup_and_run.py` defaulted to stage-1 even when a reusable transcript already existed at the repo root, it tried to validate the future stage-1 transcript output during calibration before the audio stage had produced that file, and stage-2 reusable transcripts without `metadata.vocals_audio_path` still failed speaker-sample generation unless `audio.audio_path` was supplied manually.
- Solution: Resolve the bootstrap start stage from explicit overrides first, auto-promote to stage-2 when a reusable transcript is discoverable (preferring repo-root `transcript.json` or `*.transcript.json`), skip transcript-path validation during calibration for stages that have not generated one yet, and auto-detect or prompt for reusable source audio so stage-2 speaker-sample extraction and calibration can fall back to `audio.audio_path`.

### 2026-03-07T16:45:00+08:00 - Duration Migration Must Update Tool Telemetry And Prompt Caps Together
- Problem: Converting the summarizer and critic loop from word budgets to duration budgets changed the tool contracts, but old tests and helper assumptions still expected `min_words` and `max_words`, and the loop-context truncation helper exceeded its own cap once the new duration issue text got longer.
- Solution: Migrate test fixtures and prompt/tool expectations to `target_seconds`, `duration_tolerance_ratio`, `estimate_duration`, and `emo_preset`, and enforce the loop-context character cap after appending the truncation marker so prompt-safe bounds remain real.

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
