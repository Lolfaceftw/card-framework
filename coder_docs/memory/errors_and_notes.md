# Errors And Notes

### 2026-03-09T22:00:07+08:00 - Secret-History Remediation Must Expect `git-filter-repo` To Drop `origin`
- Problem: Purging leaked secrets and private endpoints from published Git history required a full `git-filter-repo --replace-text` rewrite across all refs. The rewrite succeeded, but `git-filter-repo` intentionally removed the `origin` remote as a safety guard, which can look like unrelated repository damage if the operator expects the remote to remain configured.
- Solution: Before rewriting, create an out-of-repo backup bundle and capture the exact secret and private-endpoint replacements to redact. After the rewrite, rescan both `HEAD` and `git log --all` for the exact leaked values plus broad secret and private-IP patterns, then restore `origin` explicitly before preparing the required `git push --force-with-lease --all` and tag follow-up.

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
- Problem: The warm IndexTTS subprocess worker streamed raw log lines back through `print(...)`, and a Windows `cp1252` stdout encoding crashed on characters such as `👉` before the parent run could finish.
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
