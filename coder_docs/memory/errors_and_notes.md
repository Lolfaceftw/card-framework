### 2026-03-18T23:59:00+08:00 - Benchmark vLLM Env Overrides Must Recognize CARD_FRAMEWORK_VLLM_BASE_URL
- Problem: The main runtime config had moved to the prefixed `CARD_FRAMEWORK_VLLM_BASE_URL` env name, but benchmark provider-profile overrides and dotenv alias propagation still only recognized `VLLM_URL`, `CARD_FRAMEWORK_VLLM_URL`, and `VLLM_BASE_URL`. SPoRC and QMSum benchmark runs could therefore ignore a valid `.env` override, fall back to the tracked `http://127.0.0.1:8000/v1` profile URL, and fail at startup before any sample work began.
- Solution: Treat `CARD_FRAMEWORK_VLLM_BASE_URL` as a first-class alias in shared dotenv propagation and benchmark vLLM override resolution, and keep regression tests around both the direct benchmark override path and the `.env` alias-loading path.

### 2026-03-18T22:30:12.0551133+08:00 - Granite Seam Tuning Must Be Isolated On Fixed Vocals And Overlap Size Matters More Than Token Cap
- Problem: After the worst Granite `yeah` loops were removed, the remaining transcript defects looked like seam leftovers, but full stage-1 Docker reruns could still change Demucs separation enough to confound ASR tuning. On a fixed `vocals.wav` artifact, the old Granite default of `25s` chunks with `2s` overlap still produced many seam-style duplicate candidates such as `curve and then just ride the wave`, `index index`, and `hidden hidden`.
- Solution: Tune Granite on a saved `vocals.wav` with `scripts/run_transcription_experiment.py`, drop `chunk_overlap_seconds` from `2` to `1` before pushing decode limits harder, and then raise `repetition_penalty` to `1.25` while keeping `max_new_tokens=96`. On the fixed-vocals experiment set that cut seam-oriented duplicate candidates from `32` to `7` and repeated-token segments from `34` to `14`.

### 2026-03-18T22:12:00+08:00 - NeMo Exact Speaker Count Needed A Real Oracle Path Instead Of Reusing min_speakers
- Problem: The audio gateway exposed `min_speakers` and `max_speakers`, but NeMo's runtime only accepts an exact `num_speakers` oracle count. The old wiring passed `min_speakers` straight into that exact-count slot while still writing `num_speakers: null` into the NeMo manifest, so exact-count experiments crashed and unequal bounds risked taking the wrong path conceptually.
- Solution: Treat only `min_speakers == max_speakers` as an exact speaker-count request, write that exact value into the NeMo manifest, keep unequal bounds on the clustering side only, and use exact-count overrides sparingly because they can over-fragment long podcast audio.

### 2026-03-18T20:20:28.4313688+08:00 - Repeated GPU Stage-1 Runs Can Change Demucs Vocals Enough To Confound Diarization Tuning
- Problem: During Docker GPU transcript tuning, repeated full stage-1 runs on the same `audio.wav` produced different `separation/.../vocals.wav` hashes. That means later RTTM or transcript differences can come from upstream separation variance, not just NeMo or Granite parameter changes.
- Solution: When tuning diarization, pin one stage-1 run first, reuse its saved `vocals.wav` for NeMo experiments, and compare diarization changes on that fixed artifact before drawing conclusions about the diarizer settings.

### 2026-03-18T17:18:30.1981470+08:00 - Session Resume Must Fail Loudly On Missing Checkpoints And Skip Stage-3 When Voice Cloning Is Off
- Problem: The first pass of pipeline-session resume could still leave two sharp edges. A stage-2-only run with `audio.voice_clone.enabled=false` still entered the stage-3 bookkeeping path and dereferenced a missing voice-clone result, and the child runtime could mark a session as failed in SQLite but still exit `0`, letting the wrapper report success even though required checkpoints were missing.
- Solution: Make the stage orchestrator treat stage-3 as optional when voice cloning is disabled, have `card_framework.cli.main` raise a non-zero failure when the saved session still lacks required stage checkpoints after the run, translate `Ctrl+C` into a clean exit code `130`, and clean stale per-stage artifacts before `--resume-session` reruns the interrupted stage boundary.

### 2026-03-18T19:35:07.1429713+08:00 - Some Tiny Transcript Speaker Flips Can Be Alignment Artifacts Even When Raw RTTM Is Stable
- Problem: The latest podcast transcript still showed zero-duration or one-word speaker flips at specific lines, but the matching raw RTTM windows were single-speaker stretches. Continuing to push NeMo VAD thresholds would not fix those artifacts and risked degrading diarization elsewhere.
- Solution: Verify the raw RTTM before treating a visible transcript flip as a diarization problem, keep NeMo settings stable when RTTM is already clean in that window, and smooth tiny sandwiched transcript segments after grouping when both neighboring segments agree on the speaker.

### 2026-03-18T18:49:49.3877351+08:00 - Stronger NeMo VAD Padding And Duration Thresholds Can Regress Podcast Diarization
- Problem: After lowering Granite token generation reduced the worst repeated-token loop, a stronger NeMo pass using `vad_pad_offset=0.05`, `vad_min_duration_on=0.3`, and `vad_min_duration_off=0.8` made the same podcast episode produce more rapid speaker flips and many one-word transcript intrusions.
- Solution: Treat NeMo VAD tuning as evidence-driven and reversible, prefer comparing raw RTTM micro-turn counts against the previous best run before pushing thresholds further, and rely on transcript-side speaker-flip smoothing for tiny mid-sentence intrusions once NeMo is already near the right regime.

### 2026-03-18T16:45:27.6806535+08:00 - NeMo Micro-Turn Fixes Should Start With VAD Postprocessing Knobs, Not Speaker-Count Bounds
- Problem: After Granite transcript text improved, the latest NeMo-backed podcast run still produced many abrupt speaker flips because the raw RTTM already contained hundreds of sub-0.5-second turns. Tightening `min_speakers` and `max_speakers` would reduce flexibility for guest-heavy episodes and was not the right first lever for a general podcast pipeline.
- Solution: Expose NeMo VAD boundary controls in config, keep speaker-count bounds unset by default, and start the first podcast-oriented tuning pass by removing the negative `vad_pad_offset` while raising `vad_min_duration_on` and `vad_min_duration_off` to suppress micro-turns without hard-coding the number of speakers.

### 2026-03-18T17:10:29+08:00 - Pipeline Resume Must Trust Stage Checkpoints, Not Mid-Stage Scratch State
- Problem: The main `setup_and_run` workflow had unique per-run work and transcript paths, but no durable session store. A crash or `Ctrl+C` left the wrapper without a stable UUID to resume, and the only final summary artifact lived at workspace-root `summary.xml`, which is unsafe when multiple sessions exist. Trying to resume from the middle of transcription or voice cloning would also have depended on partial stage scratch output that is not reliably checkpointed.
- Solution: Add a SQLite-backed pipeline session store keyed by UUID, record stage-1 through stage-4 start/completion boundaries, checkpoint summaries under a session-scoped path as well as the workspace mirror, and resume only from the next incomplete stage with `--resume-session <uuid>` so interrupted runs restart the current stage cleanly instead of trusting partial in-flight artifacts.

### 2026-03-18T15:15:16+08:00 - CUDA Cleanup Logging Should Reuse The Shared Timed Loading Pattern
- Problem: Explicit GPU-memory cleanup existed in a few paths, but it was inconsistent and mostly silent. That made it hard to tell from logs whether CUDA cleanup was actually running after stage transitions or runtime offload steps.
- Solution: Route the shared CUDA cleanup helper and the remaining explicit cleanup call sites through the same timed status pattern used elsewhere, so `device=cuda` runs log `Loading ...` and `Loaded ... in X.XX seconds.` around cleanup while CPU runs skip both the cleanup work and the cleanup log noise.

### 2026-03-18T15:23:40.9917115+08:00 - ETA History Must Live In One Shared Store Instead Of Per-Run Work Dirs
- Problem: `setup_and_run.py` was still assigning a fresh per-run `audio.work_dir`, and the ETA learner was persisting its profile inside that run-local folder. That made each wrapper run start from a mostly empty ETA profile, so sample counts stayed at `1` and later runs could not reuse the latest learned separation, transcription, speaker-sample, or voice-clone timings.
- Solution: Move ETA persistence onto a stable shared SQLite-backed store path, keep recording timestamped stage observations there across runs, and let the runtime keep run-local artifact work dirs without fragmenting ETA history.

### 2026-03-18T14:58:16+08:00 - Stage-1 Needed Explicit ASR And Diarizer Resource Release To Avoid Carrying VRAM Forward
- Problem: Stage-1 kept heavyweight transcription and diarization model objects alive in-process after their outputs were already materialized. On CUDA runs that could leave Granite or diarization allocations resident into the next stage, raising peak VRAM and making Windows report spillover into shared GPU memory even after diarization finished.
- Solution: Add best-effort `release_cached_resources(...)` hooks for the GPU-backed ASR and diarization adapters, release the transcriber before NeMo diarization begins, release the diarizer right after it completes, and clear the CUDA allocator slack after the references are dropped.

### 2026-03-18T15:04:40.6760656+08:00 - Granite Speech Prompt Echoes Must Be Stripped Using The Rendered Chat Prompt
- Problem: Granite Speech chunk decodes could echo the chat-templated transcription prompt as plain `USER: ... ASSISTANT:` text instead of the raw `<|audio|>...` prompt string. The existing cleanup only stripped a leading raw prompt prefix, so prompt echoes could survive inside chunk text, pollute the transcript JSON, and break overlap continuity once chunk dedupe and diarization split the text into segments.
- Solution: Keep the fix Granite-specific, pass the rendered Granite prompt into chunk cleanup, strip rendered role-labeled prompt echoes with whitespace-tolerant matching before chunk dedupe/final transcript assembly, and keep regression tests around both direct cleanup and `_transcribe_chunk(...)` so Faster-Whisper behavior stays untouched.

### 2026-03-18T14:43:09+08:00 - NeMo Diarization Startup Needed The Shared Loading Reporter For Wrapper Runs
- Problem: `card_framework.cli.setup_and_run` already surfaced timed `Loading ...` / `Loaded ... in X.XX seconds.` lines for its own bootstrap phases, but stage-1 NeMo diarization still only showed the generic `running NeMo diarization` stage log. That left a long silent-feeling gap during NeMo model startup in wrapper-driven runs.
- Solution: Wrap NeMo MSDD diarizer startup with the shared loading-duration reporter inside the gateway itself, and add a gateway regression test so wrapper and direct runtime runs both emit the same timed operator-facing loading messages as Demucs and Granite Speech.

### 2026-03-18T11:05:00+08:00 - Post-Release Repo Metadata Must Advance Beyond The Latest Published PyPI Version
- Problem: The repository continued to report `version = "1.1.0"` and still had release guidance claiming `1.0.4` was the current public version even though PyPI had already published `1.1.0` and the branch contained additional unreleased commits. That made it easy to prepare an ambiguous re-release from post-`1.1.0` code while still wearing the published `1.1.0` version number.
- Solution: Bump the in-repo package version to the next unreleased line (`1.2.0`) once post-`1.1.0` work accumulates, and update the release docs in the same change so examples and current-public-version notes match the actual PyPI state.

### 2026-03-18T09:23:00.9723511+08:00 - Embeddings-Disabled Summarization Must Use The Full Transcript When Runtime And Benchmark Parity Matter
- Problem: The shared `SummarizerExecutor` had been changed to replace the full transcript with a bounded coverage-sampled excerpt whenever embeddings were disabled. That kept prompts smaller, but it also meant SPoRC transcript-only benchmark runs and normal runtime runs no longer exposed the full transcript to the summarizer, which could materially distort long-duration outputs and break expectations when evaluating embeddings-disabled behavior.
- Solution: Restore full-transcript prompt injection for embeddings-disabled summarization in the shared executor so runtime and benchmark paths stay aligned, while continuing to rely on the existing full-transcript timeout floors to absorb the larger prompt payloads.

### 2026-03-18T09:15:20.9858960+08:00 - AGENTS.md Must Stay A Pointer Layer Into coder_docs
- Problem: `AGENTS.md` had grown into a second copy of the repository standards instead of acting as the intended routing layer into `coder_docs`. That made the file harder to use across different developer runtimes and increased the risk of drift because several rules only existed in `AGENTS.md`.
- Solution: Move the embedded Python, security, logging, and testing standards into dedicated `coder_docs/*.md` files, rewrite `AGENTS.md` as a compact topic index with capability-gated guidance, and keep future detail in the topic-specific docs instead of re-expanding the pointer file.

### 2026-03-18T09:15:20.9858960+08:00 - Repo Guidance Must Not Pretend `researcher` Is The Only Allowed Sub-Agent
- Problem: Repo-local delegation docs hardcoded `researcher` as the only supported sub-agent and explicitly discouraged non-research delegated roles. In environments where `../.codex/config.toml` also enables `explorer`, `worker`, or `monitor`, that wording can cause coding agents to refuse legitimate execution-style delegation even though multi-agent support is enabled.
- Solution: Rewrite `AGENTS.md`, `coder_docs/research_agents.md`, `coder_docs/codebase_guide.md`, `coder_docs/scrapling.md`, and `README.md` so they treat `../.codex/config.toml` as the source of truth for available sub-agents, allow scoped delegation to any configured role, and keep Scrapling rules focused on how delegated web work is performed rather than on a single hardcoded agent name.

### 2026-03-17T23:15:00+08:00 - SPoRC Sweep Workers Must Not Reparse Or Rematerialize The Full Manifest In Every Child
- Problem: `duration-sweep` was coordinating retained-sample claiming correctly through SQLite, but each child still independently parsed the same large inline prepared manifest and materialized the full prepared sample set before claiming just one sample. On the long-form 15k manifest that turned startup into a RAM/commit spike, so higher worker concurrency could fail fast with `MemoryError` during manifest loading before any benchmark sample ran.
- Solution: Add an execution-time compact-manifest path that rewrites large inline manifests into sibling `*.compact.json` plus `*.compact.samples.jsonl.gz` files once before heavy worker startup, then teach shared-queue workers to initialize SQLite from that compact manifest and cache serialized prepared-sample payloads in the queue rows. After that change, workers still share deterministic order and retained-target accounting, but each child only loads the claimed sample payload from SQLite instead of rematerializing the full candidate catalog.

### 2026-03-17T22:10:00+08:00 - SPoRC Retained Sample Backfill Must Stay Local To SPoRC And Preserve Repo-Wide A2A Defaults
- Problem: The SPoRC exact-retained-sampling work correctly needed explicit HTTP connection-pool limits, bounded pool waits, infra-only replacement, and backfill semantics, but the first transport patch widened those network defaults into the shared `AgentClient` constructor. That would have silently changed non-SPoRC A2A callers such as QMSum, QA, and the main CLI by forcing `trust_env=False`, a shorter keepalive policy, and an early pool timeout everywhere.
- Solution: Keep the new transport controls as opt-in `AgentClient` constructor parameters, preserve the historical repo-wide defaults unless a caller explicitly overrides them, and let SPoRC pass its stricter `httpx.Limits`, `pool_timeout_seconds`, and `trust_env=False` settings only from the benchmark-managed runtime. In the same change, make SPoRC `execute` and `duration-sweep` treat positive `--max-samples` values as exact retained-sample targets, retry startup/connection failures on the same sample before replacement, and backfill later deterministic candidates only for infra-style exclusions so reports can separate retained, attempted, and replacement-excluded rows cleanly.

### 2026-03-17T20:52:11.6716255+08:00 - SPoRC Transcript Export Needed A True Raw Dataset Pass-Through Mode
- Problem: `export-transcripts` was sourcing rows from the raw SPoRC `turn_data_path`, but it still reformatted them into `[timestamp] [speaker] text` lines. That preserved the raw row boundaries and noisy transcript fragments, yet it was not the literal dataset payload a user needs when they explicitly want the original unprocessed JSONL rows.
- Solution: Add `export-transcripts --raw` to write the exact matching dataset JSONL rows for the selected episodes with no headers, timestamp rendering, speaker-label formatting, or normalization from repo code.

### 2026-03-17T20:47:31.0451870+08:00 - SPoRC Export Backwards-Compat Must Preserve Curated Episode Positions
- Problem: The first `export-transcripts --backwards-compat` implementation treated `--episodes` as original dataset episode numbers. That broke the established longform workflow immediately because the default curated manifest starts at sparse original episode indices such as `77`, so a normal `--episodes 1-10 --backwards-compat` call failed instead of preserving the curated selection semantics.
- Solution: Define `--backwards-compat` as a mapping mode, not a renumbering mode. Keep `--episodes` as curated longform manifest positions, load that compatibility manifest when the target manifest is the original prepared sidecar, and resolve the target rows by stable `episode_key`.

### 2026-03-17T20:40:53.2061858+08:00 - SPoRC Transcript Export Must Not Scan Past The Final Requested Episode Block
- Problem: `sporc export-transcripts` always streamed the entire raw `speakerTurnData.jsonl.gz` file even when the requested manifest positions were a small early slice such as `--episodes 1-10`. On the full SPoRC dataset that meant rereading a 10.7 GB gzip after the command had already collected every requested episode.
- Solution: Stream the raw turn file by contiguous episode group, keep only the current selected episode block in memory, and stop as soon as the last requested episode group finishes instead of scanning the rest of the dataset.

### 2026-03-17T19:48:24+08:00 - Curated SPoRC Longform Manifests Can Contain Exact Duplicate Episodes
- Problem: The curated `benchmark_longform_15k_manifest.json` can contain exact duplicate episodes that share the same title, summary, and transcript content under different source URLs. When `llm-sample` reads the manifest prefix or a selected range, those duplicate rows look like sampler-level duplication even though the duplication is already present in the manifest.
- Solution: Add a dedicated `dedupe-manifest` command that rewrites the same inline manifest in place. Use a metadata-first candidate grouping pass and only open transcript JSON for collision groups that need exact confirmation, so large manifests can be deduped quickly without hashing every transcript file.

### 2026-03-17T18:31:25.3284262+08:00 - SPoRC LLM Sampling Exact-Match Validation Was Too Restrictive For Anthropic Selections
- Problem: The SPoRC `llm-sample` validator required every returned `segment_text` to match one transcript segment exactly or a normalized concatenation of adjacent same-speaker fragments. In practice this rejected otherwise useful Anthropic selections and turned the sampler into a brittle substring-matcher instead of an automatic segment selector.
- Solution: Remove the exact-match guard and the prompt language that demanded verbatim transcript extraction. Keep the structured-output contract, `speaker_label` requirement, and sample-count or budget validation, but let the model choose grounded segments without forcing exact substring matches.

### 2026-03-17T18:25:53+08:00 - Anthropic Tool-Only Sampler Replies Must Not Be Rejected For Missing Prose
- Problem: The SPoRC `llm-sample` prompt was tightened to demand verbatim transcript-only samples, but the validator still required a short assistant paragraph before the `submit_samples` tool call. Anthropic often goes straight from thinking into structured tool use in both thinking and non-thinking modes, so valid tool payloads were being rejected with `Model did not emit the required short answer paragraph before the tool call.`
- Solution: Treat the structured `submit_samples` payload as the authoritative answer, remove the pre-tool prose requirement from the prompt and validator, and stream the incremental tool JSON on the content channel so operators can still watch the selected segments arrive live.

### 2026-03-17T18:03:55.3629826+08:00 - Anthropic Thinking Rejects Explicit Temperature Values Other Than 1
- Problem: After setting the shared `AnthropicProvider` default to `temperature=0.0`, adaptive-thinking SPoRC sampler calls started failing with `invalid_request_error: 'temperature' may only be set to 1 when thinking is enabled or in adaptive mode.`
- Solution: When Anthropic thinking is enabled, do not send a `temperature` field by default. Let adaptive thinking run with Anthropic's own default behavior unless a caller deliberately opts into a compatible value.

### 2026-03-17T17:44:54.3656749+08:00 - Anthropic Thinking Cannot Be Combined With Forced Tool Choice
- Problem: The new SPoRC `llm-sample` workflow enabled Anthropic adaptive thinking and also forced `tool_choice` onto the `submit_samples` client tool. Anthropic rejected those requests with `invalid_request_error: Thinking may not be enabled when tool_choice forces tool use.`
- Solution: Keep thinking enabled, but when the shared `AnthropicProvider` sees a forced tool-choice mode such as a specific tool or `required/any`, downgrade that request to `tool_choice={"type":"auto"}` and let the prompt plus post-response validation enforce the tool call instead of the API-level force flag.

### 2026-03-17T00:05:56+08:00 - SPoRC Transcript-Only Working Transcripts Must Preserve Raw Speaker Attribution And Mark Uncertainty
- Problem: The first transcript-only SPoRC normalization pass improved readability by collapsing short speaker flips toward nearby dominant speakers, but that changed the canonical working transcript speaker labels. On noisy SPoRC episodes this could turn a dataset labeling error into a confident-looking wrong attribution, which is worse than leaving the turn ambiguous when transcript fidelity matters.
- Solution: Keep transcript-only normalization lossless with respect to speaker ownership: drop punctuation-only fragments, merge adjacent same-speaker spans, preserve `raw_primary_speaker` plus `raw_speakers`, and add `speaker_candidates` plus `speaker_uncertain` metadata instead of reassigning a turn to a neighbor.

### 2026-03-16T23:43:17.8203647+08:00 - SPoRC Transcript-Only Runs Must Normalize Raw Dataset Speaker Turns Before Judging
- Problem: The SPoRC transcript-only benchmark copied prepared transcripts straight into each run's `working_transcripts/` folder and judged them as-is. The prepared transcripts faithfully preserved the raw SPoRC `speaker[0]` turn labels and every micro-turn from the dataset, so noisy episodes could surface dozens of incidental `SPEAKER_XX` labels, hundreds of sub-second fragments, and machine-hostile duplicate alias-map keys such as `SPEAKER_00` plus `speaker_00`.
- Solution: Keep the prepared transcript as the raw dataset artifact, but normalize the transcript-only working copy before the summarizer-critic loop runs: drop punctuation-only fragments, merge adjacent same-speaker spans, preserve raw dataset speaker lists in per-segment extras, collapse short non-core speaker flips toward dominant or named speakers, and serialize only canonical alias-map keys while keeping case-insensitive lookup in code.

### 2026-03-16T22:39:00+08:00 - SPoRC Shard Artifacts Needed A Run-Local Sample Index And Raw LLM Trace Persistence
- Problem: SPoRC reports already kept judge scores and speaker-style comparison payloads, but they did not persist the raw summarizer, critic, or judge request/response content in a developer-friendly folder tree. Duration-sweep shard logs exposed `run_id`, yet a developer still had to infer where the child run's final sample outputs and raw LLM content lived.
- Solution: Treat each `sporc_runs/<run_id>/` directory as the shard-level root, add a hierarchical `samples/` subtree with `samples/index.json`, `samples/README.md`, per-sample `final/` outputs, and `llm/{summarizer,critic,judge}/call_XXX.json` traces, and record those resolved artifact paths back into each sample result in `report.json`.

### 2026-03-16T10:06:34.5332544+08:00 - SPoRC Sweep Worker Pooling Must Not Forward Old Sample-Shard Flags Into Shared-Queue Children
- Problem: The new duration-sweep worker-donation scheduler moved in-preset concurrency onto a shared preset queue so finished presets could donate worker slots forward. But the parent was still forwarding per-worker `--sample-shard-index/--sample-shard-count` values into each child command, so every pooled worker exited immediately with `--shared-sample-queue-root cannot be combined with sample-shard selection.` before any sample ran.
- Solution: Keep the parent-side worker-slot labels for logs, checkpoints, and restart tracking, but force pooled execute children to run with `--sample-shard-index 1 --sample-shard-count 1` whenever `--shared-sample-queue-root` is set. Add a regression test on `_launch_duration_sweep_child(...)` so shared-queue children always launch with full-sample scope internally.

### 2026-03-16T09:39:31+08:00 - SPoRC Sweep UX Needed Footer Progress, Colored Status, And Graceful Ctrl+C Teardown
- Problem: After adding in-preset sweep fan-out, the parent still printed plain repeated `Sweep progress: ...` lines, which buried important restart/finish messages during long runs. Interrupting a sweep also relied on raw process-tree termination paths that did not explicitly close the sweep UI or child resources before bubbling `KeyboardInterrupt`.
- Solution: Keep non-interactive stderr plain, but on real terminals render colorized SPoRC CLI status lines and a footer `tqdm` aggregate progress bar with ETA. Make the duration-sweep interrupt path stop every active child, join readers, close handles, tear down the footer bar cleanly, and let the shared Windows process-tree helper try `CTRL_BREAK_EVENT` before falling back to forced `taskkill`.

### 2026-03-16T09:22:23+08:00 - SPoRC Duration Sweeps Need In-Preset Fan-Out Instead Of One Serial Child Per Preset
- Problem: `sporc duration-sweep` could already run multiple presets at once via `--parallelism`, but each preset still had only one serial `execute` child. Asking for something like `--parallelism 0 --concurrency 3` had no effect because the sweep parent tracked only one child per preset, so large preset runs could not use multiple child processes without reworking the single-active-sample execute runtime.
- Solution: Keep `execute` single-sample serial semantics intact, add deterministic `--sample-shard-index/--sample-shard-count` filtering to the execute path, and let the duration-sweep parent fan each active preset out into multiple shard children through `--concurrency`. Aggregate those shard reports back into one preset summary, keep shard logs/checkpoints separate, and preserve per-child watchdog recovery.

### 2026-03-15T16:50:00+08:00 - SPoRC Duration Sweeps Must Not Abort A Whole Preset After Several Individual Sample Stalls
- Problem: The duration-sweep watchdog already restarts a stalled child via `--resume-from-checkpoint`, and the resumed execute run records only the orphaned active sample failed with `sweep_watchdog_stalled`. But the parent was still counting those restarts at the preset level and aborting the entire preset once `--max-child-restarts` was exhausted, which silently dropped later untried samples from the denominator instead of continuing them.
- Solution: Keep using checkpoint resume to fail only the stalled sample, but let the preset continue after repeated stalled samples instead of converting the threshold into a whole-preset failure. Update the CLI help and SPoRC docs so `--max-child-restarts` is treated as a warning threshold rather than a preset-abort gate.

### 2026-03-15T16:32:00+08:00 - Main Can Merge Release Work Without The Matching Package Version Bump
- Problem: PR `#16` merged the `v1.1.0` feature branch into `main`, but `main` still carried `version = "1.0.7"` in `pyproject.toml` and the editable package record in `uv.lock`. Tagging that merge commit as `v1.1.0` would have retriggered the PyPI publish workflow with stale `1.0.7` metadata and likely failed or produced an ambiguous release attempt.
- Solution: Cut a dedicated `release/v1.1.0` branch from the merged `main`, bump both package-version metadata locations to `1.1.0`, run the documented release preflight on that branch, merge the branch, and only then push the matching `v1.1.0` tag from the merged `main` commit.

### 2026-03-15T16:35:00+08:00 - Runtime A2A Readiness Failures Must Propagate Their Actual Exception
- Problem: `card_framework.cli.main._wait_for_agent_servers()` caught its own `RuntimeError` and immediately called `sys.exit(1)`, which hid the readiness failure from tests and any caller that needed the real exception details.
- Solution: Remove the local `sys.exit(1)` conversion so the timeout `RuntimeError` propagates naturally, and add a regression test that asserts the helper raises when A2A servers never become healthy.

### 2026-03-15T16:20:00+08:00 - PR Follow-Ups Needed Benchmark-Compatibility, Safe QMSum Prep, And Callback-Swap Serialization
- Problem: The `v1.1.0` PR left several review regressions behind: `benchmark.run` still required removed `orchestrator.min_words`/`max_words`, vendored JudgeArena modules used package-fragile imports and eager optional dependencies, QMSum preparation could delete or extract outside an approved root, the QMSum runner still used racy ephemeral-port reservation, and `BaseA2AExecutor` temporarily swapped shared provider callbacks without per-call serialization.
- Solution: Add a legacy word-budget fallback in `benchmark.run`, convert JudgeArena imports to package-relative fallbacks with lazy optional dependency errors, constrain QMSum repo preparation under `artifacts/` and validate ZIP members before extraction, keep reserved sockets open until uvicorn consumes them, and move temporary callback swaps into the worker-thread chat call under a per-provider lock so concurrent critic streams cannot cross-wire.

### 2026-03-15T13:32:40.3674365+08:00 - SPoRC Duration Sweeps Need Checkpointed Child Recovery For Hung vLLM Calls
- Problem: A `sporc duration-sweep` child could start a long `5m` sample, enter the summarizer revise path, and then stay alive indefinitely with an open vLLM socket but no further progress events. The parent sweep only knew about `sample_start` and `sample_done`, so one wedged child froze that preset forever at the same sample index.
- Solution: Add sweep-only child heartbeats plus per-preset execute checkpoints, write partial execute reports after every terminal sample, and let the parent watchdog kill and relaunch a stalled child with `--resume-from-checkpoint`. On resume, mark the orphaned active sample failed with `sweep_watchdog_stalled`, continue with later samples, rebuild the local A2A runtime after failed samples, and keep standalone `sporc execute` unbounded unless explicit timeout flags are passed.

### 2026-03-15T12:55:00+08:00 - ETA Must Abandon A Bad Cold-Start Prior As Soon As Chunk Throughput Is Real
- Problem: The dynamic ETA tracker blended the initial profile-based estimate with live chunk progress using the same smoothing in both directions. On long-stage separation runs that meant a pessimistic first-run prior could linger for multiple chunks, so logs showed unrealistic drops like `58m -> 44m -> 33m -> 25m` even though each chunk runtime already implied a much shorter finish time.
- Solution: Make the tracker contract immediately to the observed total when real progress says the stage is faster than the prior, and keep smoothing only for slower upward corrections so ETA stabilizes quickly after the first real chunk without becoming noisy when later chunks slow down.

### 2026-03-15T12:45:37.0083186+08:00 - Audio-Stage ETA Should Not Wait For Learned History
- Problem: Separation and transcription already had configured/default throughput multipliers, but the orchestrator suppressed ETA entirely until a learned or persisted stage-history sample existed. That made first-run stage-1 logs look inconsistent with the new chunk-progress reporting even when the runtime knew the input audio duration.
- Solution: Let the orchestrator estimate audio-stage ETA whenever duration is known, using the strategy's configured/default throughput on first run and refining future runs from learned or persisted history.

### 2026-03-15T12:40:00+08:00 - Long Demucs Separations Need Outer Windowing Or RAM Scales With Full Track Length
- Problem: The stage-1 separator was invoking `python -m demucs.separate` on the entire source file. For multi-hour stereo WAV inputs that made Demucs allocate full-length float32 mixture/output tensors, so one `python.exe` child could spike into double-digit gigabytes before Granite ASR even started.
- Solution: Keep `htdemucs` as the separator, but wrap it in repo-level outer windowing with bounded `audio.separation.window_length_seconds` plus `window_context_seconds`, trim each chunk back to its base window after separation, stitch the vocals windows back together, and emit timed loading plus `chunk i/N` progress so operators can see that the RAM-heavy step is bounded work rather than a hung process.

### 2026-03-15T11:49:00+08:00 - Granite Speech Requires The Model-Card Chat Prompt And Positional Processor Inputs
- Problem: `ibm-granite/granite-4.0-1b-speech` does not behave like the generic Hugging Face ASR pipeline path. Using ad hoc `text=` or `audio=` keyword arguments, or skipping the model-card chat template, risks processor/input-shape failures and mismatched decoding behavior.
- Solution: Follow the Granite model-card contract directly: build the prompt with `tokenizer.apply_chat_template(...)`, pass the prompt and mono `16 kHz` waveform positionally into `processor(prompt, wav, device=..., return_tensors="pt")`, and keep chunking/progress handling in the repo adapter instead of relying on the generic pipeline abstraction.

### 2026-03-15T11:08:55+08:00 - Packaged Infer Imports Must Stay Dependency-Light For Wheel Smoke
- Problem: The clean-wheel smoke test installs `card-framework` with `--no-deps` before importing `from card_framework import infer`. That import path was still eager enough to fail immediately on optional runtime modules such as `python-dotenv`, which meant the built wheel could not prove its advertised public export in a minimal environment.
- Solution: Make repository-root dotenv loading a graceful no-op when `python-dotenv` is unavailable and keep `card_framework.api` dependency-light by lazy-loading runtime config normalization inside the packaged `infer(...)` call path instead of at module import time.

### 2026-03-15T10:54:38.7152585+08:00 - Runtime Config Drift And Inline Secrets Needed One Canonical Selector-Based Workflow
- Problem: The repo had two different runtime config entrypoints, a root `conf/config.yaml` full of commented provider blocks, and inline API-key fields spread across tracked YAML. That made normal configuration noisy, let repo-local and packaged defaults drift, and encouraged users to keep secrets in config files instead of `.env`.
- Solution: Make `conf/config.yaml` the canonical source-checkout config, keep `src/card_framework/config/config.yaml` as a bundled mirror, normalize the new `selection` plus `profiles` schema into explicit runtime sections in shared code, and move secret values to `.env`-backed fields with documented alias support.

### 2026-03-15T10:06:06+08:00 - Summarizer-Critic Benchmarks Need Deterministic Seeded Sample Selection
- Problem: The QMSum and SPoRC real benchmark runners both selected the first `N` prepared samples in manifest order. That kept runs reproducible, but it baked manifest-order bias into every limited benchmark and made it impossible to request a deterministic randomized subset for more robust comparisons.
- Solution: Add a shared seeded selector keyed by `(sample_seed, sample_id)`, thread `--sample-seed` through QMSum `execute` and `supervise` plus SPoRC `execute`, `supervise`, and `duration-sweep`, and record the active seed or `manifest_order` mode in benchmark artifacts so runs stay auditable.

### 2026-03-15T00:02:10+08:00 - SPoRC Duration Sweeps Should Fan Out Presets Concurrently And Show Aggregate Progress
- Problem: `sporc duration-sweep` still ran presets serially, ignored `--max-samples` unless `--examples` was also set, and mirrored child token streams into the terminal. That made sweeps unnecessarily slow on vLLM-backed runs and buried the useful operator signal under per-token output.
- Solution: Make duration sweeps honor `--max-samples` by default with `--examples` as a sweep-only override, launch preset `execute` children concurrently with explicit retrieval/summarizer/critic ports, suppress child token streaming during sweeps, emit machine-readable sample progress from child runs, and show only aggregate sample positions plus a combined ETA in the parent sweep process.

### 2026-03-14T21:11:11+08:00 - Benchmark Results Tables Should Be Native Markdown And Show Aggregate 95% CIs
- Problem: The summarizer-critic benchmark reports wrote fixed-width ASCII tables inside fenced code blocks, which made `results.md` harder to reuse in GitHub or docs views. They also showed only point estimates for aggregate means, so sweep and run summaries gave no quick sense of sampling uncertainty.
- Solution: Switch the QMSum and SPoRC report writers to native Markdown tables, add a shared reporting helper for Markdown rendering, and compute two-sided 95% Student-`t` confidence intervals for aggregate metric means when `n >= 2` while leaving the CI undefined for `n < 2`.

### 2026-03-14T21:11:11+08:00 - Critic Loop Guards Must Stop Repetition During Streaming, Not Only After A Full Turn
- Problem: The initial critic reasoning-loop guard only evaluated completed assistant turns. When the critic started streaming a repetitive "let me re-read the transcript" loop, operators still had to watch the bad text stream until the provider finished the turn or stalled.
- Solution: Add a stream-time repetition callback that aborts the critic response as soon as repeated long spans emerge, retry the critic once with an explicit "do not repeat that mistake" instruction, and fall back to the deterministic `CRITIC LOOP GUARD:` verdict path only if the retried stream loops again.

### 2026-03-14T21:22:44.0732130+08:00 - Critic Semantic-Alignment Checks Need A No-Tool Repetition Guard
- Problem: The critic could get stuck narrating the same transcript re-check over and over during semantic-alignment evaluation without emitting another tool call or a final verdict. That let repetitive "let me re-read the transcript" content burn critic turns until the generic loop cap, producing noisy failures and hiding the actual fallback cause.
- Solution: Add a shared opt-in no-tool reasoning-loop guard in `BaseA2AExecutor`, enable it for `Critic` through `agents.critic.reasoning_loop_guard`, stop single-turn or cross-turn repetitive no-tool responses early, and fall back through the deterministic critic verdict path with a `CRITIC LOOP GUARD:` feedback prefix.

### 2026-03-14T20:21:37.6627553+08:00 - Runtime Entry Points Must Surface Timed Loading Status Instead Of Looking Idle
- Problem: The real pipeline entrypoints could still sit silently during cold startup while `cli.main` imported the A2A and executor stack, reusable transcripts loaded, calibration bootstrapped, providers initialized, embeddings indexed, or packaged runtime bootstrap ran. Operators only saw progress again after those waits completed, so `setup_and_run`, direct `cli.main`, and packaged `infer(...)` could all look frozen even when they were working normally.
- Solution: Reuse the shared loading-duration context manager across the actual runtime path, lazy-load the heaviest `cli.main` startup modules behind timed first-use helpers, and emit operator-facing `Loading ...` / `Loaded ... in X.XX seconds.` lines from `cli.main`, `cli.setup_and_run`, packaged `infer(...)`, retrieval indexing, speaker-sample bootstrap, calibration, provider setup, and A2A readiness checks.

### 2026-03-14T19:49:24.3386858+08:00 - SPoRC Long-Form Curation Must Reuse Prepared Stats Instead Of Reopening Every Transcript
- Problem: `curate-longform` originally reloaded every prepared transcript JSON and accumulated the full audit row list in memory, so the pass stayed linear but got slower in practice as filesystem reads, Python object growth, and final audit serialization piled up across the full SPoRC manifest.
- Solution: Persist reusable long-form curation stats in each prepared sample during `prepare-dataset`, have `curate-longform` filter those stored stats directly with transcript-read fallback only for older manifests, and stream audit rows through a temporary JSONL file before writing the final audit JSON so curation stays mostly manifest-bound and memory-bounded.

### 2026-03-14T19:01:35.9825862+08:00 - Full SPoRC Prepare Must Show Timed Progress And Handle Ctrl+C Cleanly
- Problem: After the Hugging Face `episodeLevelData.jsonl.gz` and `speakerTurnData.jsonl.gz` download bars finished, `uv run python -m card_framework.benchmark.summarizer_critic.sporc prepare-dataset --dataset-variant full` could sit silently for a long time while importing the heavy benchmark runner and streaming tens of millions of turn rows. Operators had no indication whether the process was still working, so cancelling it surfaced a raw `KeyboardInterrupt` traceback from `json.loads(...)` inside `_iter_jsonl_gz(...)`.
- Solution: Lazy-load the heavy SPoRC execute runner only for execute-style commands, add a shared perf-timing context manager that emits `Loading ...` and `Loaded ... in X.XX seconds.` lines to stderr around slow CLI imports and prepare-time loading phases, emit periodic status lines while indexing episode metadata and streaming turn groups, and catch `KeyboardInterrupt` in the CLI so `Ctrl+C` exits cleanly with `Cancelled.` instead of a traceback.

### 2026-03-14T18:47:42.6396613+08:00 - Full SPoRC Prepare Must Stream And Spill To Disk Instead Of Materializing Both Gzip Tables
- Problem: `benchmark/summarizer_critic/sporc` full-dataset preparation read `episodeLevelData.jsonl.gz` and `speakerTurnData.jsonl.gz` into Python lists and then duplicated the full turn table again in an in-memory `turns_by_episode` map. On the March 14, 2026 full SPoRC files that meant trying to hold roughly `12.3G + 10.7G` of compressed source plus Python object overhead in RAM, which exhausted a 32 GB machine immediately after download.
- Solution: Replace the eager loads with a disk-backed SQLite episode metadata index, stream `speakerTurnData.jsonl.gz` one contiguous episode block at a time, spool prepared sample rows on disk so final output can be reordered back into episode-file order, and write the full prepared manifest as a compact `manifest.json` plus `samples.jsonl.gz` sidecar so `execute` and `curate-longform` can iterate bounded prefixes without loading every prepared sample row into memory.

### 2026-03-14T17:38:55.6184697+08:00 - SPoRC Long-Form Benchmarks Must Curate From Full Speaker-Turn Data, Not The Sample Manifest
- Problem: The default SPoRC prepared sample manifest mixes teaser-length, music-heavy, and otherwise benchmark-incompatible episodes with genuinely long ones. Treating that sample manifest as the source for `5m` or `15m` runs causes retrieval and discovery to chase content that is not present in the released diarized transcript, especially for intro-plus-music episodes.
- Solution: Keep full-dataset preparation separate under `artifacts/summarizer_critic_benchmark/datasets/sporc/prepared_full`, add a dedicated `curate-longform` step that filters the prepared full manifest using the released Hugging Face speaker-turn table as the diarized transcript source, and require strict long-form gates such as `15,000+` spoken words, multiple substantial speakers, strong turn counts, low dominant-speaker share, and low music/noise-token ratios before writing the curated benchmark manifest.

### 2026-03-14T07:41:21+00:00 - Benchmark Embedding Enablement Could Ignore Legacy Repo Config And Miss The Intended Qwen Model
- Problem: The benchmark matrix helper built its `enabled` embedding profile from `src/card_framework/config/config.yaml`, but during the src-layout migration some local runs still kept their embedding edits in the legacy repo-root `conf/config.yaml`. When the packaged config still pointed at `NoOpEmbeddingProvider`, `--embedding-mode enabled` silently fell back to MiniLM instead of the user's intended Qwen embedding model.
- Solution: Teach benchmark embedding-profile resolution to check the legacy repo-root embedding section before falling back to MiniLM, normalizing old `providers.*` targets into current `card_framework.providers.*` paths so SPoRC and other benchmark runs can honor legacy Qwen embedding edits during the migration window.

### 2026-03-14T07:20:00+00:00 - Research-Agent-Only Docs Must Not Reintroduce Non-Research Delegation Roles
- Problem: Repo-local Markdown guidance still described multiple non-research delegated roles even though the intended documentation model is research-agent-only. That left `AGENTS.md`, `README.md`, and `coder_docs/*` with conflicting guidance about how delegation should work.
- Solution: Replace the old multi-role guidance with `coder_docs/research_agents.md`, rewrite the docs around the lone `researcher` agent, and keep local repository reads, commands, and implementation in the main thread.

### 2026-03-14T15:00:00+00:00 - Completed A2A Task History Can Corrupt SPoRC Critic JSON Results
- Problem: After fixing the stuck-at-summarizer issue, transcript-only SPoRC runs could still loop forever because `AgentClient` extracted text from a completed task by concatenating the terminal `status.message` JSON with `history` text that still contained the original user task payload. The critic therefore returned strings that looked like `{"status": ...}\n{"draft": ...}`, which failed `CriticTaskResponse.model_validate_json(...)` with `trailing characters at line 2`.
- Solution: Change `AgentClient._extract_text_result(...)` to prefer artifact text, top-level parts, nested result payloads, and terminal `status.message` text before falling back to `history`, and add a regression test that exercises a completed-task response carrying both a final message and historical request text.

### 2026-03-14T14:05:00+00:00 - Non-Blocking A2A Agent Replies Must Complete The Task, Not Just Emit A Final Message
- Problem: Transcript-only SPoRC `execute` runs could appear frozen for minutes immediately after `[Summarizer] Result: ...` because the agent executor enqueued a final raw `Message` but never emitted a terminal `completed` task-status event. The local `AgentClient` polls `tasks/get`, so it kept seeing the summarizer task stuck in `working` and never advanced to the critic.
- Solution: Change `BaseA2AExecutor.send_response(...)` to complete the current A2A task with a terminal status update carrying the final agent message whenever `task_id` and `context_id` are present, while preserving the raw-message fallback only for contexts without task metadata.

### 2026-03-13T15:10:02+00:00 - SPoRC Transcript-Only Judge Runs On Qwen vLLM Need The Explicit Chat-Template Thinking Override
- Problem: The new SPoRC transcript-only duration benchmark could converge successfully, but the G-Eval judge still returned empty content on the remote Qwen3.5 vLLM endpoint when the default vLLM provider config was used. That zeroed the SPoRC judge metrics even though the summarizer-critic loop itself succeeded.
- Solution: When instantiating the SPoRC judge on `card_framework.providers.vllm_provider.VLLMProvider`, keep `enable_thinking=True` so `extra_body` is sent, but override `thinking_extra_body` to `{"chat_template_kwargs": {"enable_thinking": false}}` so the server emits normal JSON content for judge prompts.

### 2026-03-13T11:54:04+00:00 - DeepSeek Reasoning History Must Exclude `reasoning_content`
- Problem: DeepSeek's reasoning-model docs state that sending prior `reasoning_content` back in message history causes a `400` error, but the provider normalization path preserved assistant `reasoning_content` fields when rebuilding chat history.
- Solution: Drop `reasoning_content` from normalized outbound history, keep reasoning only for local display/inspection, and use `deepseek-reasoner` with omitted `max_tokens` plus untruncated source text for judge-only sweeps.

### 2026-03-13T08:27:19+00:00 - Concurrent QMSum Sweeps Need Explicit Internal Service Ports
- Problem: Running multiple `summarizer_critic execute` processes in parallel relied on a bind-and-release ephemeral-port probe for retrieval, summarizer, and critic A2A servers. Under concurrent sweeps that could race and make separate runs contend for the same local ports.
- Solution: Add optional `--retrieval-port`, `--summarizer-port`, and `--critic-port` overrides to the QMSum execute parser, validate that the three ports are distinct per run, and use explicit unique port trios from sweep harnesses.

### 2026-03-13T07:46:13+00:00 - SPoRC Voice-Clone Benchmarks Need Transcript Milliseconds, Source-Audio Downloads, And Live-Draft Archiving
- Problem: The first SPoRC benchmark path only fed transcript text into the summarizer-critic loop, so it never exercised the repo's faithful combined stage-2/stage-3 voice-clone flow. SPoRC transcript segments were also stored in raw seconds, which breaks speaker-sample extraction because the audio pipeline expects transcript `start_time` and `end_time` values in milliseconds.
- Solution: Normalize SPoRC transcript segment timings to milliseconds during dataset preparation, persist each sample's source-audio URL in transcript metadata, download source episode audio on demand during execution, generate transcript-derived speaker-sample manifests before the loop, run live-draft voice cloning during summarization, and archive per-sample voice-clone outputs plus paths in `report.json`.

### 2026-03-13T07:46:13+00:00 - SPoRC Judge Benchmarks Must Score Only Judge Dimensions And Persist Speaker Comparison Payloads
- Problem: The first SPoRC benchmark implementation still carried a QMSum-style `duration_control` metric into the composite score even though the requested evaluation dimensions were only `Naturalness`, `Factualness`, and `Speaker Replication`. It also built transcript-versus-summary per-speaker message lists for the judge prompt without persisting those lists in the sample report, which weakened traceability for speaker-replication audits.
- Solution: Keep duration telemetry separate from judge scoring, compute the SPoRC composite from only the three judge dimensions, and persist transcript per-speaker messages, summary per-speaker messages, alias maps, and speaker notes in each sample result written to `report.json`.

### 2026-03-13T07:21:49+00:00 - SPoRC Benchmark Wiring Needed Root QMSum Shims And Unbounded Loop Support
- Problem: The new SPoRC benchmark path needed its own package under `benchmark/summarizer_critic/sporc`, but the existing QMSum package still imported nonexistent root modules such as `card_framework.benchmark.summarizer_critic.runner`. That broke imports during test collection and also blocked a clean SPoRC implementation. The SPoRC benchmark also needed true unbounded summarizer-critic loops and no per-stage request timeouts, which the runtime previously modeled only as positive numeric values.
- Solution: Add backward-compatible root shim modules that forward to the QMSum implementation, point the QMSum package at its own CLI to avoid circular imports, and teach the runtime orchestrator plus agent client to accept `None` for unbounded `max_iterations` and no-timeout stage requests.

### 2026-03-13T04:55:00+00:00 - Remote vLLM Streaming Can Fail Mid-Benchmark But Non-Streaming Often Recovers
- Problem: During the full QMSum run, some remote OpenAI-compatible stream requests failed inside the OpenAI client with `list index out of range`, which surfaced as hard summarizer failures even though the endpoint itself could still answer the same prompt.
- Solution: Add a `VLLMProvider` fallback that retries failed streaming `chat(...)` and `generate(...)` requests in non-streaming mode before surfacing an error. Keep the streamed path as the default fast path, but use non-streaming as a resilience path for unstable remote servers.

### 2026-03-13T01:10:00+00:00 - Constrained Benchmark Caps Must Still Score The Last Draft
- Problem: The duration-constrained benchmark can intentionally cap `max_iterations`, but `run_loop_with_diagnostics(...)` discarded the latest draft and zeroed the final estimate whenever the cap was reached. That made constrained runs look like hard failures instead of scoring the best draft produced within budget.
- Solution: Preserve the latest draft, word count, and estimated seconds in the `max_iterations_reached` diagnostics payload so benchmark code can evaluate the capped run fairly while still marking it as non-converged.

### 2026-03-12T23:47:00+00:00 - Qwen vLLM Judge Runs Need An Explicit `enable_thinking: false` Chat Template Override
- Problem: The first live benchmark smoke run reached the reference-free judge, but the remote Qwen vLLM endpoint either streamed empty content or emitted long reasoning traces instead of contract JSON when the provider merely set `enable_thinking=False`. That zeroed naturalness and reduced factualness coverage.
- Solution: For this vLLM/Qwen endpoint, keep `enable_thinking=True` on the provider and pass `thinking_extra_body.chat_template_kwargs.enable_thinking=false` so the server applies the explicit chat-template override and returns usable JSON for judge-style prompts.

### 2026-03-12T23:36:00+00:00 - Malformed Summarizer Drafts Must Not Crash Benchmark Revise Iterations
- Problem: The first live QMSum smoke run reached critic feedback, but the summarizer's revise path tried to parse the previous draft as strict XML and crashed on malformed model output instead of recovering. That turned one bad draft into a hard benchmark failure before the loop could self-correct.
- Solution: Make `SummarizerExecutor._load_draft_into_registry(...)` ignore malformed XML drafts and continue with an empty registry so subsequent iterations can recover from critic feedback instead of aborting the run.

### 2026-03-13T12:30:00+00:00 - Real Summarizer-Critic Benchmarks Need A Stub Calibration And Duration-First Runner Contract
- Problem: The older summarization benchmark path still assumed word-budget orchestrator calls and did not pass the now-required `calibration` dependency into the actual Summarizer and Critic executors. That made it a poor fit for a real duration-first QMSum benchmark even before dataset work started.
- Solution: Add a dedicated `benchmark/summarizer_critic` package that prepares QMSum separately, derives `target_seconds` from human references, runs the real A2A agents with a synthetic benchmark calibration artifact, and keeps the duration-first integration fix scoped to the new package.

### 2026-03-13T01:25:21+08:00 - QA Supervisor Must Preflight Child Imports And Preserve Repo-Local Source Paths
- Problem: The new telemetry supervisor could loop on the QA benchmark even when the child process was dead on arrival from environment issues. In repo-local source runs, the child import also needed src forwarded into PYTHONPATH; otherwise the preflight could misdiagnose a missing package as ModuleNotFoundError: card_framework instead of the real dependency failure.
- Solution: Add a lightweight QA import preflight before benchmark loops begin, forward repo-local src into child-process PYTHONPATH, and stop after the smoke test with recorded notes/results when the benchmark environment is not importable.

### 2026-03-13T01:25:21+08:00 - Malformed Script Entry Newlines In pyproject.toml Can Break The Entire Benchmark Toolchain
- Problem: The new benchmark watchdog and QA supervisor script entries were introduced with literal PowerShell newline text plus a duplicate [project.scripts] header, which made uv reject pyproject.toml before any benchmark or test command could start.
- Solution: Normalize the script entries onto separate lines, remove the duplicate header, and verify repo command discovery again before debugging benchmark behavior.

### 2026-03-13T01:24:00+08:00 - QA Supervisor Threshold Kills Need A Direct Process Fallback After Windows `taskkill`
- Problem: The new QA supervisor correctly detected RAM threshold breaches, but on Windows the child process could survive long enough after `taskkill /T /F` for `process.wait(timeout=5.0)` to raise `TimeoutExpired`. That left the threshold-kill verification hanging onto an open child and log handle instead of completing cleanly.
- Solution: After issuing `taskkill`, wait briefly for the child and fall back to `process.kill()` plus a final `wait()` when the process is still alive. This keeps threshold-killed runs deterministic and lets telemetry/result artifacts close cleanly.

ï»¿### 2026-03-13T01:22:43+08:00 - Benchmark Supervisors Need A Synced Runtime Environment And Clean Script Metadata
- Problem: The supervised benchmark tooling could not be exercised reliably because `pyproject.toml` had a malformed `[project.scripts]` block with literal newline text and a duplicate header, and the first real benchmark attempts in this offline workspace showed two separate runtime blockers: `benchmark.qa` failed immediately with `ModuleNotFoundError: hydra`, while `benchmark.run` failed under `uv run` before execution because Git-backed dependencies could not be fetched without network access.
- Solution: Normalize the `[project.scripts]` block so repo tooling parses again, keep the QA and generic benchmark supervisors as the resource-watch entrypoints, and treat real supervised benchmark runs as requiring a pre-synced environment with the runtime dependencies already installed or cached locally before looping the actual benchmark commands.


### 2026-03-13T01:20:00+08:00 - QA Benchmark Runs Need A Supervisor To Preserve Telemetry And Logs When The Local Env Is Unsynced
- Problem: In this workspace the real `card_framework.benchmark.qa` command can fail before benchmark execution starts when the active Python environment is missing repo dependencies such as `hydra-core`. A plain failing run loses the resource-usage context and forces the next session to rediscover whether the crash came from RAM/VRAM pressure or from an import-time environment problem.
- Solution: Add a supervised QA benchmark wrapper that performs a telemetry smoke test first, samples host RAM and NVIDIA VRAM during child-process execution, kills threshold breaches, and always writes session notes, markdown results, logs, and telemetry artifacts even when the child dies on an import-time dependency error.
# Errors And Notes

### 2026-03-13T01:20:00+08:00 - Benchmark Smoke Runs Need Repo-Local uv Cache And Preseeded Offline Dependencies
- Problem: The first monitored smoke benchmark failed before runtime startup because the default user-profile uv cache path raised an access-denied error in this environment, and switching to a fresh repo-local uv cache then failed again because outbound network was disabled and the child environment could not fetch git or CUDA wheel dependencies on demand.
- Solution: Make the benchmark watchdog force a repo-local `UV_CACHE_DIR` and offline uv mode for reproducible child runs, and treat a hydrated `.uv-cache` or prebuilt environment as a prerequisite for full benchmark execution in sandboxed/offline sessions.

### 2026-03-13T01:18:14.0985768+08:00 - Broken Benchmark Supervision Entry Points Must Not Block Nightly Runs
- Problem: The repo's benchmark supervision path had two separate blockers: `pyproject.toml` could invalidate `uv` script discovery when the QA and QA-supervisor script declarations were malformed, and `src/card_framework/benchmark/watchdog.py` contained a split string literal that raised a syntax error on import. That combination broke the operator path for guarded repeated benchmark runs before any real telemetry loop could start.
- Solution: Keep the QA and QA-supervisor script declarations valid TOML, repair the newline join in `watchdog.py`, and add regression coverage that parses `pyproject.toml` with `tomllib` so the benchmark scripts stay importable and runnable.

### 2026-03-13T00:08:40+08:00 - Codex Docs Must Match The Configured Research-Agent Workflow
- Problem: The repository documented Scrapling and other contributor workflows, but it did not reflect the configured research-agent guidance. That drift encouraged inconsistent delegation for external research and left repo-local docs out of sync.
- Solution: Add `coder_docs/research_agents.md` as the authoritative research-agent workflow, align `AGENTS.md`, `coder_docs/codebase_guide.md`, `coder_docs/scrapling.md`, and `README.md` around `researcher`, and keep repository reads, commands, and implementation in the main thread.

### 2026-03-10T17:44:52+08:00 - Packaged `setup_and_run` Must Separate Caller Workspace Paths From Package Resources And Infer Output Root
- Problem: Switching pip-installed `infer(...)` from a direct `card_framework.cli.main` subprocess to `card_framework.cli.setup_and_run` restored the desired live prints, but the wrapper still resolved mutable paths from `shared.paths.REPO_ROOT`. Inside a wheel install that made reusable transcripts, `audio.wav`, calibration discovery, and artifact summaries point into `.venv\\Lib`, while `infer(output_dir=...)` also needed to keep its flat output contract instead of falling back to repo-style `artifacts/*` paths.
- Solution: Give `setup_and_run` and `calibrate` explicit caller-workspace, output-root, and config-file controls. Resolve packaged resources from package paths, keep packaged IndexTTS runtime assets on `resolve_runtime_layout()`, run the child pipeline from the explicit infer output root with `hydra.run.dir=.`/`hydra.output_subdir=null`, and have `infer(...)` delegate through the wrapper so live streaming returns without breaking the public artifact bundle contract.

### 2026-03-10T06:17:51+08:00 - Packaged CTC Aligner Bootstrap Must Use uv-Compatible Archive URLs And UTF-8-Safe Subprocess Capture
- Problem: A fresh packaged `infer(device="cuda")` run inside a repo-local `uv venv` no longer had `pip`, so the pinned `ctc-forced-aligner` bootstrap fell back to approximate timing. The first `uv pip` fix still failed because the old GitHub codeload URL did not end in a file extension that `uv` accepts, and the parent packaged subprocess capture could also throw `UnicodeDecodeError` on Windows while reading child output with the locale codec.
- Solution: Prefer `uv pip install --python <active-interpreter>` when `uv` is available, switch the pinned aligner archive URL to the GitHub archive form that ends in `.tar.gz`, and force packaged bootstrap and pipeline subprocess capture to use `encoding="utf-8", errors="replace"` so Windows output cannot crash the parent process.

### 2026-03-10T06:17:51+08:00 - Embeddings-Disabled Summarizer Fallback Must Use A Bounded Coverage Sample, Not The Raw Full Transcript
- Problem: The packaged summarizer path with embeddings disabled was injecting the entire transcript directly into the first tool-loop prompt. On the real `audio.wav` smoke input that grew to an 88k-character prompt and the first summarizer request timed out after 30 minutes without returning.
- Solution: Build a bounded transcript excerpt for embeddings-disabled summarization that preserves start, middle, and end coverage with omission markers and per-line truncation. Keep the critic on the full transcript, but keep the summarizer prompt small enough to respond in real packaged CUDA runs.

### 2026-03-09T23:53:20.1028388+08:00 - Packaged CUDA Repair Must Prefer `uv pip` Inside uv-Managed Projects
- Problem: The packaged `infer(device="cuda")` path could now self-repair a CPU-only PyTorch install to CUDA 12.6, but the first implementation always shelled out to `python -m pip`. That ignored the repository's uv-first packaging policy and could bypass a caller's uv-managed environment workflow.
- Solution: Inspect the caller's working directory and parent directories for uv project signals such as `uv.lock` or `pyproject.toml` uv metadata, then prefer `uv pip uninstall/install --python <active-interpreter> --torch-backend cu126` for the PyTorch repair path. Keep `python -m pip` only as the fallback when no uv-managed project can be confirmed.

### 2026-03-09T23:27:36+08:00 - Release Publishes Need A Dedicated Semver-Named Branch Before Tag Push
- Problem: The repo's PyPI workflow is tag-triggered, which made it easy to mentally collapse "branch" and "publish" into one step and skip an explicit release-branch rule. That ambiguity encourages ad hoc tagging from whatever local branch happens to contain the version bump, which weakens review flow and makes it less obvious which commit was intentionally prepared for release.
- Solution: Document one release path across `AGENTS.md`, `coder_docs/git_github_workflow.md`, `coder_docs/github_actions_release_spec.md`, and `coder_docs/codebase_guide.md`: create a dedicated semver-named release-preparation branch such as `release/vX.Y.Z`, run preflight there, merge it, then create and push the matching `vX.Y.Z` tag from the merged integration-branch commit.

### 2026-03-09T23:55:00+08:00 - Packaged `infer(...)` Must Self-Resolve `uv` And `ffmpeg` And Degrade Cleanly On Aligner Bootstrap Failures
- Problem: The PyPI wheel could publish and import, but the packaged `infer(...)` path still depended on PATH luck for `uv` and `ffmpeg`, and the published docs claimed the pinned `ctc-forced-aligner` would bootstrap automatically even though the checked-in code did not. That left downstream `pip install` users one missing executable or one failed aligner build away from a brittle first run.
- Solution: Add runtime dependency coverage for `uv` and `imageio-ffmpeg`, resolve packaged `uv` from the active interpreter scripts directory and `ffmpeg` from PATH or the bundled `imageio-ffmpeg` executable, keep the wheel metadata free of `ctc-forced-aligner`, and make packaged `infer(...)` fall back to approximate timing when the pinned aligner bootstrap cannot complete instead of aborting the whole run.

### 2026-03-09T21:40:00+08:00 - PyPI Publish Workflow Smoke Test Must Match The Real Public `infer(...)` Signature
- Problem: The first tagged PyPI publish attempt built the `card-framework` distributions successfully but the GitHub Actions smoke test still asserted that `card_framework.infer(...)` exposed only three parameters. The real public API had already grown keyword-only `device`, `vllm_url`, and `vllm_api_key` parameters, so the build job failed before `uv publish`, leaving the pending trusted publisher unused and PyPI with no released version.
- Solution: Update `.github/workflows/publish-pypi.yml` so the smoke test asserts the full public `infer(...)` signature, then bump the package version for the next release tag instead of retrying the broken tag unchanged.

### 2026-03-09T17:05:00+08:00 - Packaged `infer(...)` Must Require An Explicit Device And Gate CUDA To 12.6
- Problem: The public `infer(...)` contract still relied on config-default `audio.device=auto`, so packaged callers could not declare CPU versus CUDA intent at the API boundary, and nothing stopped a user from attempting unsupported CUDA runtime combinations.
- Solution: Make the packaged API require `device=\"cpu\"` or `device=\"cuda\"`, inject that choice into the effective runtime config, and reject `device=\"cuda\"` unless the installed PyTorch build reports CUDA 12.6 and CUDA is actually available on the machine.

### 2026-03-09T16:42:00+08:00 - Frozen `RuntimeBootstrapError` Broke Traceback Handling During Temp-Config Cleanup
- Problem: The installed-package `infer(...)` path now uses a temporary merged config whenever it injects prompted credentials or vLLM-first overrides. When bootstrap failed inside that context, `contextlib` tried to reattach `__traceback__`, but the frozen dataclass `RuntimeBootstrapError` rejected that mutation and raised a misleading `TypeError` instead of preserving the original bootstrap failure.
- Solution: Remove `frozen=True` from `RuntimeBootstrapError` and keep a regression test that forces a bootstrap failure while temp-config cleanup is active, so real bootstrap errors surface as themselves.

### 2026-03-09T16:25:00+08:00 - Packaged `infer(...)` Must Fail Fast On Unsupported Platforms And Resolve Secrets Before Launch
- Problem: The public pip-installed `card_framework.infer(...)` path still trusted the loaded config blindly. That left packaged runs on macOS or Linux failing later and less clearly, and provider credentials such as API keys or Hugging Face access tokens were only discovered deep inside the runtime after the subprocess had already started.
- Solution: Make `infer(...)` explicitly Windows-only for the packaged whole-pipeline contract, add a vLLM-first `vllm_url` and `vllm_api_key` override path, and resolve missing provider credentials from config, environment variables, or secure interactive prompts before subprocess launch so secrets do not need to travel through CLI argv.

### 2026-03-09T15:32:00+08:00 - First Public PyPI Release Needs A Pending Trusted Publisher, Not Just `uv publish`
- Problem: Local `uv publish --dry-run` validated the `card-framework` artifacts, but the release still could not become public because PyPI had no project and no trusted publisher configured yet. Relying on a local token-based publish would bypass the repo's long-term release path and leave the public release process undocumented.
- Solution: Add a top-level GitHub Actions trusted-publishing workflow (`.github/workflows/publish-pypi.yml`) that builds and smoke-checks the tagged distributions before `uv publish`, and document that the first public release must use PyPI's pending-publisher flow for project `card-framework` with owner `Lolfaceftw`, repository `card-framework`, workflow filename `publish-pypi.yml`, and environment `pypi`.

### 2026-03-09T15:07:42+08:00 - Installed `infer(...)` Calls Must Carry An Explicit Duration Target
- Problem: The new packaged `card_framework.infer(...)` entrypoint initially only required `(audio_wav, output_dir)` even though the runtime is duration-first and the summarizer budget is a first-class contract. That left installed-package callers implicitly depending on config-default duration instead of declaring the intended output length at the API boundary.
- Solution: Require `target_duration_seconds` on every `infer(audio_wav, output_dir, target_duration_seconds)` call, validate it before bootstrap or subprocess work begins, and pass it into the Hydra runtime as an explicit duration override.

### 2026-03-09T14:58:54+08:00 - Installed Wheel Runtime Must Not Treat Site-Packages Like The Repository Root
- Problem: The built `card-framework` wheel already imported and packaged correctly, but installed-package runtime code still derived writable paths from `shared.paths.REPO_ROOT`. In a clean virtualenv that made `REPO_ROOT` resolve to the environment `Lib` directory, which pointed mutable runtime defaults such as `checkpoints/index_tts` and vendored runner paths into site-packages-shaped locations instead of a real writable CARD runtime home.
- Solution: Add an install-safe runtime layout for the public `card_framework.infer(audio_wav, output_dir, target_duration_seconds)` API, bootstrap the vendored IndexTTS project and checkpoints into a writable runtime home (`CARD_FRAMEWORK_HOME` or the platform user-data directory), and run the existing Hydra pipeline with explicit installed-package overrides rather than assuming a checkout-shaped filesystem.

### 2026-03-09T13:33:25+08:00 - Local Benchmark Defaults Must Discover Real Transcripts And Fail Loudly On Zero Execution
- Problem: The packaged summarization benchmark still assumed a stale `summary.json` local transcript default, so `card-framework-benchmark prepare-manifest --sources local` and the legacy `card-framework-eval` entrypoint broke on a fresh repo state. After fixing transcript discovery, the benchmark still exited successfully even when every cell was skipped before any sample ran, which made a missing vLLM endpoint look like a green run.
- Solution: Reuse repo-aware transcript auto-discovery for local benchmark manifests, switch packaged local manifest defaults to `auto`, align the main config transcript default with `audio.output_transcript_path`, and make the benchmark CLI exit non-zero with a direct operator-facing error when zero cells or zero samples execute while still preserving the report artifacts.

### 2026-03-09T13:27:24+08:00 - Packaged Eval Help And Repo Ruff Gate Must Reflect Maintained Code
- Problem: The legacy `card_framework.cli.eval` entrypoint ignored CLI arguments, so `card-framework-eval --help` ran the smoke benchmark path instead of showing help, and the documented repo-level `uv run ruff check .` command regressed by linting the vendored `src/card_framework/_vendor` tree as if it were maintained first-party code.
- Solution: Make the eval wrapper forward real CLI arguments to `card_framework.benchmark.run`, preserve the smoke preset only as the no-argument default, add a real packaged help smoke test, and exclude `src/card_framework/_vendor` from the repo Ruff gate while documenting that vendor exclusion in `coder_docs/ruff.md`.

### 2026-03-09T12:15:00+08:00 - Vendored IndexTTS GPT Constructors Must Stay Valid Python For Live-Draft Runs
- Problem: A real `card_framework.cli.setup_and_run --audio-path audio.wav` stage-1 run completed separation, transcription, diarization, and speaker-sample generation, then crashed on the first live-draft voice-clone call because the warm IndexTTS subprocess could not import vendored `indextts/gpt/model.py` and `model_v2.py`; both constructors had a malformed parameter named `card_framework.retrieval.embeddings`.
- Solution: Restore the constructor argument name to `embeddings` in both vendored GPT modules and add a regression test that `py_compile`s the vendored GPT files so syntax regressions fail in pytest before a long runtime run.

### 2026-03-08T23:58:00+08:00 - src Migration Tests Must Use Real A2A Message Payloads And OmegaConf Lists
- Problem: After preloading the real `a2a` package to stop cross-test stub pollution, several executor tests still parsed `str(queue.events[0])` as if A2A responses were raw JSON strings, while `benchmark.mrcr` still rejected valid `provider_profiles.yaml` data because OmegaConf exposes `providers` as `ListConfig`, not a plain `list`.
- Solution: Add a shared test helper that extracts the first text part from either stubbed or real A2A messages, update the stale executor assertions and packaged CLI import path, and make `benchmark.mrcr` accept OmegaConf `ListConfig` provider lists so the MRCR config resolver works with the repository's real YAML loader.

### 2026-03-08T23:20:00+08:00 - Removing Root Wrappers Requires Module-Based Subprocess Calls Everywhere
- Problem: After the `src/card_framework` migration, bootstrap subprocesses, summary-matrix subprocesses, tests, and docs still called `main.py`, `calibrate.py`, and `setup_and_run.py`, so deleting the root wrappers would have broken the actual runtime and verification flow.
- Solution: Move all subprocess invocations, test expectations, and operator docs to `python -m card_framework.cli.*`, then remove the root wrapper files and shim packages so the repository is package-first end to end.

### 2026-03-08T22:36:30+08:00 - src Layout Refactor Must Rebase Default Paths And Import-Time Benchmarks
- Problem: Moving maintained code under `src/card_framework` and vendoring IndexTTS under `src/card_framework/_vendor` left several defaults, docs, and tests still pointing at root `conf/`, `prompts/`, `benchmark/`, and `third_party/index_tts`, while `benchmark.mrcr` still performed heavyweight downloads and endpoint calls during import.
- Solution: Rebase runtime defaults and docs onto the packaged `src/card_framework` layout, keep bootstrap path resolution tolerant of the legacy tree during the migration window, and rewrite `benchmark.mrcr` into a lazy config-resolution helper so test collection no longer triggers network or dataset side effects.

### 2026-03-08T00:25:00+08:00 - Summary-Matrix Streaming Must Ignore Blank Sections And Cover DeepSeek Too
- Problem: Piped summary-matrix runs could still show empty `[CONTENT]` headers from whitespace-only tool-call preambles, and DeepSeek stage overrides bypassed the shared streaming callback entirely, so reasoning/content streaming stayed inconsistent across model pairs.
- Solution: Ignore leading whitespace-only chunks before opening plain `[THINKING]` or `[CONTENT]` sections, and route DeepSeek streaming through the shared response-callback path so both terminal and piped matrix runs use the same fallback behavior.

### 2026-03-08T00:12:38+08:00 - Piped Summary-Matrix Runs Need A Plain Streaming Fallback Instead Of Rich Live Panels
- Problem: `scripts/run_summary_matrix.py` captures each child `setup_and_run.py` stdout through a pipe. Rich live panels in the child process are designed for a real TTY, so once the noisy raw JSON logger output was removed, matrix runs appeared to stop streaming because the live token renderer had no pipe-friendly fallback.
- Solution: Make `RichConsoleResponseCallback` detect non-interactive stdout and fall back to plain-text streamed `[THINKING]` and `[CONTENT]` chunks, so child runs still stream through the matrix runner pipe without relying on Rich live terminal behavior.

### 2026-03-08T00:07:59+08:00 - Tool Results Must Render Structured JSON With Real Newlines In The Terminal
- Problem: The Rich terminal UI printed `tool_result` payloads as a single truncated string, so JSON dict payloads showed escaped `\\n` sequences instead of readable multiline transcript excerpts, and the console logger echoed the same payload as one long noisy blob.
- Solution: Parse JSON-shaped tool results before terminal rendering, format dict/list payloads as structured multiline output that preserves newline-bearing string values, and reduce the logger-side `tool_result` console line to a concise summary.

### 2026-03-08T00:05:01+08:00 - Terminal Logging Must Summarize LLM Payloads Instead Of Dumping Full Messages And Tool Schemas
- Problem: Once `setup_and_run.py` stopped suppressing the live Summarizer and Critic stream, `LoggingLLMProvider` started printing full `Messages`, full `Tools` schema lists, and full response JSON at `INFO`, which buried the useful live stream under huge terminal dumps.
- Solution: Keep terminal logs at concise summaries such as message counts, tool names, and response lengths/previews, and reserve full payload dumps for `DEBUG` logging only.

### 2026-03-08T00:01:32+08:00 - setup_and_run Must Not Suppress Live Summarizer And Critic Terminal Output
- Problem: `setup_and_run.py` enabled `logging.print_to_terminal=true` for operator feedback but also hard-overrode `logging.summarizer_critic_print_to_terminal=false`, so the live Summarizer and Critic reasoning/content stream stayed hidden even when callers expected the child run to stream to the terminal.
- Solution: Stop injecting the suppressing `logging.summarizer_critic_print_to_terminal=false` override in `build_run_overrides(...)` so the repo default once again follows `logging.print_to_terminal` unless a caller explicitly disables the agent stream.

### 2026-03-07T23:54:33+08:00 - Live-Draft Summarizer Timeout Must Cover Inline Voice Cloning
- Problem: Stage-2 live-draft runs synthesize real IndexTTS audio inside the summarizer tool loop, but the orchestrator still capped full-transcript summarizer requests at the generic 900-second timeout floor. Longer runs timed out the localhost summarizer call first, then the process began unwinding while the agent was still voice cloning, which surfaced secondary IndexTTS worker-exit and interpreter-shutdown noise.
- Solution: Raise the live-draft summarizer timeout floor to `max(900s, target_seconds * 6)` whenever a draft-audio state path is active, and keep the vendored IndexTTS `infer_v2` path from forwarding `length_penalty` when `num_beams=1` so the repo-default low-latency generation path stops emitting invalid-generation warnings.

### 2026-03-07T23:17:13+08:00 - Hydra Stage Critic Overrides Must Use Append Syntax In The Summary Matrix Runner
- Problem: `scripts/run_summary_matrix.py` injected `stage_llm.critic._target_=` and sibling keys as plain Hydra overrides, but `stage_llm.critic` starts as an empty structured dict in `conf/config.yaml`, so Hydra rejected those keys with `Key '_target_' is not in struct`.
- Solution: Emit the matrix runner's `stage_llm.critic.*` overrides with Hydra append syntax such as `+stage_llm.critic._target_=...`, `+stage_llm.critic.base_url=...`, and `+stage_llm.critic.api_key=...` so the critic config can be created from the empty repo default.

### 2026-03-07T23:14:59+08:00 - Summary Matrix Must Not Override The Repo-Default Live Stage-2/Stage-3 Path
- Problem: The summary-matrix runner forced `audio.voice_clone.enabled=false` and `audio.voice_clone.live_drafting.enabled=false`, which silently pushed stage-2 runs onto the legacy calibration-backed duration path instead of the repository's default merged live-draft stage-2/stage-3 flow.
- Solution: Preserve the repo defaults for voice cloning and live drafting inside `scripts/run_summary_matrix.py`, disable only stage-4 interjector output, and keep tests that assert the runner no longer injects the legacy calibration-triggering overrides.

### 2026-03-07T23:13:07+08:00 - Summary Matrix Pairs Intentionally Include Self-Pairs And Must Stream Child Output
- Problem: The summary-matrix helper had been pushed toward strict permutation semantics, but the actual operator requirement is to run every ordered summarizer/critic pair from one model pool, including self-pairs such as `qwen3_5_27b x qwen3_5_27b`, and to show live child token/output instead of buffering everything until the pair finishes.
- Solution: Build pair cells with `itertools.product(model_profiles, repeat=2)`, keep both directions for mixed-model pairs, print a start line for each cell, and tee the child `setup_and_run.py` stdout/stderr stream live to the parent terminal while also persisting the raw UTF-8 log file per pair.

### 2026-03-07T23:08:31+08:00 - Summary Matrix Must Use One Model Pool Permutation Set, Not A 3x4 Cross Product
- Problem: `scripts/run_summary_matrix.py` was described as running summarizer/critic permutations, but the implementation still iterated a separate 3-summarizer by 4-critic cross product and allowed same-model self-pairs such as `qwen3_5_27b x qwen3_5_27b`.
- Solution: Build one shared model pool, add the optional DeepSeek model to that pool, and generate ordered distinct summarizer/critic pairs with `itertools.permutations(..., 2)` so the reported counts and executed runs match real permutation semantics.

### 2026-03-07T23:08:31+08:00 - IndexTTS Worker Log Streaming Must Sanitize Unicode Before Printing On Windows
- Problem: The warm IndexTTS subprocess worker streamed raw log lines back through `print(...)`, and a Windows `cp1252` stdout encoding crashed on characters such as `ÃƒÂ°Ã…Â¸Ã¢â‚¬ËœÃ¢â‚¬Â°` before the parent run could finish.
- Solution: Sanitize streamed worker log lines for the active stdout encoding before printing so unsupported Unicode is replaced safely instead of raising `UnicodeEncodeError`.

### 2026-03-07T22:40:46+08:00 - Summarizer Tool Handlers Must Reject Empty Required Args Without Crashing
- Problem: A stage-2 live-drafting run hit `add_speaker_message {}` from the summarizer, and `AddSpeakerMessageHandler.execute()` indexed `arguments["speaker_id"]` directly. That raised `KeyError('speaker_id')`, crashed the A2A worker, and aborted the whole pipeline instead of returning a normal tool error the loop could recover from.
- Solution: Validate required string arguments in the summarizer tool handlers before use, return structured `missing_*` or `empty_*` tool errors for malformed calls, and keep regression tests around empty `add_speaker_message` and `query_transcript` payloads plus dispatcher behavior so bad tool args stay inside the tool loop.

### 2026-03-07T21:36:42+08:00 - Stage-4 Planner Output Must Stay Sparse And Survive Truncated JSON
- Problem: The interjector prompt asked for one decision object per eligible host turn, including explicit false entries. On longer summaries that bloated the planner response enough for the model to stop mid-JSON, after which stage-4 parsed nothing and silently fell back to all-false decisions with `artifact_count=0`.
- Solution: Make the planner return only positive interjection decisions, keep missing turns implicitly false, and salvage any fully formed decision objects from a truncated `decisions` array so stage-4 can still render usable overlaps when the model response stops early.

### 2026-03-07T21:32:09+08:00 - Stage-4 Planner Anchors Must Respect The Runtime Overlap Window
- Problem: The interjector planner only saw raw host tokens, so it could choose a syntactically valid anchor very early in the turn. Stage-4 then accepted the planner decision, but the runtime overlap gate rejected it later because the configured host-progress window starts at 35% of the host turn, producing `should_interject=true` decisions with `artifact_count=0`.
- Solution: Include a per-turn preferred anchor token window in the stage-4 planner prompt, pass the configured host-progress ratios into planner validation, and reject anchors outside that approximate window before synthesis so planner output matches what the runtime can actually render.

### 2026-03-07T20:50:06+08:00 - In-Budget Auto-Saved Summaries Must Not Wait Indefinitely For A Finalize Tool Call
- Problem: During live drafting, the summarizer could hit the target duration, auto-save a solid draft, and then spend the next DeepSeek turn narrating its review in plain text instead of calling `finalize_draft()`. Because that review turn was unbounded, the localhost A2A request could time out after 900 seconds even though the draft itself was already ready.
- Solution: Enter a bounded draft-review mode after auto-save: inject an explicit review-only instruction, cap the next review turn token budget, and if the model still replies with plain text instead of a tool call after the draft is in budget, submit the current saved draft instead of waiting indefinitely.

### 2026-03-07T20:02:49+08:00 - Stage-2 Bootstrap Audio Must Match The Reusable Transcript Before Live Drafting Starts
- Problem: Stage-2 speaker-sample bootstrap could accept a fallback `audio.audio_path` that was unrelated to the reusable `transcript.json`, generate a manifest with missing or misaligned speaker labels, and only fail later inside live draft voice cloning with a missing speaker-sample error.
- Solution: Validate the inferred bootstrap transcript against the reusable transcript before accepting its speaker samples, remap bootstrap speaker labels back onto the reusable transcript labels when the audio matches but label IDs differ, and fail early with a direct mismatch error when the bootstrap audio or speaker coverage is incompatible.

### 2026-03-07T19:41:15+08:00 - Stage-2 Missing Speaker Samples Must Rebuild From Audio, Not Reclip The Transcript
- Problem: `setup_and_run.py` auto-promoted reusable `transcript.json` inputs to stage-2 correctly, but when `metadata.speaker_samples_manifest_path` was missing the runtime only re-clipped samples from transcript timings plus fallback `audio.audio_path`. That path was very fast, but it skipped fresh separation and diarization, so coarse or mixed reusable transcripts produced contaminated voice references.
- Solution: Treat stage-2 missing speaker samples as an audio-bootstrap requirement: resolve real source audio, run a fresh audio inference pass to produce a vocals stem and aligned speaker transcript, generate samples from that inferred vocals audio under the current run work directory, and write only the new speaker-sample metadata back to the reusable transcript for later reuse.

### 2026-03-07T19:21:54+08:00 - Live Stage-2 Audio Needs Stable Turn IDs And A Persisted Draft-Audio Sidecar
- Problem: The old stage-2 flow estimated duration from one upfront calibration artifact, then ran stage-3 as a second full render pass after critic approval. That meant actual cloned durations could drift away from stage-2 decisions, revise mode could not safely reuse already-rendered turn audio, and deleted or re-added lines had no stable identity across retries.
- Solution: Give every registry line a stable `turn_id`, persist live draft audio state keyed by those IDs, synthesize changed turns immediately during stage-2 when `audio.voice_clone.live_drafting.enabled=true`, feed critic checks from the persisted actual segment durations, and finalize the approved live turn cache directly into the stage-3 manifest instead of re-rendering the whole summary.

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
- Problem: A real `setup_and_run.py` stage-2 run reached the pipeline, but Rich UI subscribers started throwing `UnicodeEncodeError` on Windows because system messages included characters such as `ÃƒÂ¢Ã¢â‚¬Â Ã¢â‚¬â„¢` that the active `cp1252` console could not encode.
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
