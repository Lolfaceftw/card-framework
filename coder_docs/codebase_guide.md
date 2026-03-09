<?xml version="1.0" encoding="UTF-8"?>
<codebaseGuide workspace="llm_test" package="card-framework" language="Python" pythonVersion="3.12+" packageManager="uv" linter="ruff">
  <overview>
    <purpose>Orient a new coding agent to the actual architecture, workflows, contracts, and maintenance expectations of this repository before code changes are made.</purpose>
    <repositoryIdentity>This workspace combines a Hydra-configured summarization runtime, an audio-to-transcript and voice-clone pipeline, multiple LLM and embedding provider adapters, and two benchmark systems: a matrix-style summarization benchmark and a source-grounded QA benchmark.</repositoryIdentity>
    <sessionStartChecklist>
      <step>Read this file first to understand subsystem boundaries, entrypoints, and data flow.</step>
      <step>Review coder_docs/memory/errors_and_notes.md for previously captured pitfalls before repeating work in an area.</step>
      <step>Use coder_docs/academic_standards.md for formulas, thresholds, scoring logic, or methodology-sensitive implementations.</step>
      <step>Use coder_docs/ruff.md for lint workflow, coder_docs/uv_package_manager.md for dependency and environment workflow, coder_docs/git_github_workflow.md for contributor Git and GitHub workflow, and coder_docs/scrapling.md for external web retrieval workflow.</step>
      <step>Use coder_docs/fault_localization_workflow.md as the source of truth when a task is primarily bug triage, failing-test analysis, or fault localization.</step>
      <step>Inspect the exact config and prompt files for the subsystem you are changing before editing behavior.</step>
      <step>When behavior, configuration, commands, or workflow changes materially, update this guide in the same change.</step>
    </sessionStartChecklist>
    <primaryWorkflow>
      <stage id="stage-1">Audio stage: source separation, ASR, diarization, alignment, and transcript JSON emission.</stage>
      <stage id="stage-2">Summarization stage: local A2A retrieval, summarizer, and critic agents cooperate until convergence or iteration cap. When `audio.voice_clone.live_drafting.enabled=true` and voice cloning is enabled, each mutating draft step also renders per-turn audio immediately and uses actual measured segment duration instead of WPM estimates.</stage>
      <stage id="stage-3">Voice-clone stage: summary XML plus generated speaker samples are rendered into per-turn cloned audio and optional merged output. In live-draft mode this stage usually finalizes already-rendered turn audio into the persisted manifest instead of re-synthesizing every turn.</stage>
      <stage id="stage-4">Interjector stage: the next speaker can add short overlapping backchannels or echo-agreements onto the stage-3 merged audio using summary-aware token anchors.</stage>
    </primaryWorkflow>
    <agentOperatingPrinciples>
      <principle>Treat configuration as authoritative; prefer overrides and config edits over hardcoded values.</principle>
      <principle>Preserve the separation between orchestration, provider adapters, prompts, and data-contract modules.</principle>
      <principle>Favor behavior-level understanding first: runtime pipeline, audio pipeline, and benchmark workflows are all first-class parts of this repo.</principle>
    </agentOperatingPrinciples>
  </overview>

  <section id="repo-layout" title="Repository Layout">
    <directory path="src/card_framework/agents">A2A executor implementations, task DTOs, tool dispatch, loop control, parsing, correction guidance, and client transport helpers.</directory>
    <directory path="src/card_framework/audio_pipeline">Stage-1 and stage-3 use-case orchestration, contracts, ETA tracking, runtime helpers, gateways, runners, speaker-sample generation, voice-clone orchestration, and the live draft voice-clone session used by merged stage-2/stage-3 runs.</directory>
    <directory path="src/card_framework/benchmark">Summarization benchmark CLI, reference-free evaluation pipeline, QA benchmark CLI, manifests, rubrics, metrics, matrix construction, and report artifact helpers.</directory>
    <directory path="src/card_framework/cli">Operator-facing CLI modules for the runtime, calibration helper, setup bootstrap, summary-matrix runner, evaluation entrypoint, and terminal UI.</directory>
    <directory path="src/card_framework/providers">Concrete LLM and embedding adapters such as vLLM, Transformers, DeepSeek, GLM, Google GenAI, Hugging Face, Nanbeige, sentence-transformer embeddings, and logging wrappers.</directory>
    <directory path="src/card_framework/prompts/templates">Jinja2 prompt templates for summarizer, critic, QA ground-truth generation, QA evaluator, and corrector flows.</directory>
    <directory path="src/card_framework/orchestration">Typed transcript DTOs and the stage orchestrator that bridges stage plans with runtime execution.</directory>
    <directory path="src/card_framework/config">Hydra application configuration, including provider selection, stage controls, audio settings, ports, orchestrator limits, and logging.</directory>
    <directory path="src/card_framework/_vendor/index_tts">Vendored third-party IndexTTS runtime source required by the voice-clone pipeline.</directory>
    <directory path="tests">Pytest coverage grouped by subsystem: agents, audio_pipeline, benchmark, bootstrap, orchestration, providers, and top-level runtime behavior.</directory>
    <directory path="coder_docs">Project-local policy and workflow documentation. This file is the architecture/session guide; sibling docs cover methodology, fault localization, linting, package management, Git and GitHub workflow, and web retrieval workflow.</directory>
    <file path="AGENTS.md">Repo-local coding-agent instructions, including the required Scrapling web-research policy and the rule to keep coder_docs aligned with prompt changes.</file>
    <entrypoints>
      <entrypoint path="src/card_framework/cli/main.py">Primary Hydra runtime for the summarization, audio, and voice-clone pipeline.</entrypoint>
      <entrypoint path="src/card_framework/cli/calibrate.py">One-time voice-clone calibration helper that discovers or bootstraps speaker samples, renders punctuation-rich calibration phrases, and prints the persisted preset/WPM mapping.</entrypoint>
      <entrypoint path="src/card_framework/cli/setup_and_run.py">Bootstrap and convenience runner for dependency checks, stage-aware vendored-runtime sync/model provisioning, and full pipeline execution.</entrypoint>
      <entrypoint path="src/card_framework/cli/run_summary_matrix.py">Batch helper that runs the standard setup-and-run summarizer workflow across ordered summarizer/critic model pairs, optionally adds a DeepSeek model to the pair pool, preserves the repo-default merged live-draft stage-2/stage-3 voice-clone flow, isolates loop-memory artifacts per model pair, streams each child run's live output back to the parent terminal, disables only stage-4 interjector output, and copies each resulting summary into `artifacts/summary_matrix` as `<summarizer>_<critic>-summary.xml`.</entrypoint>
      <entrypoint path="src/card_framework/benchmark/run.py">CLI for summarization benchmark matrix execution and manifest preparation.</entrypoint>
      <entrypoint path="src/card_framework/benchmark/qa.py">CLI for source-grounded QA benchmark execution against an existing summary and source transcript.</entrypoint>
      <entrypoint path="src/card_framework/cli/eval.py">Benchmark smoke entrypoint.</entrypoint>
      <entrypoint path="src/card_framework/cli/ui.py">Rich terminal presentation layer that subscribes to the shared event bus.</entrypoint>
    </entrypoints>
  </section>

  <section id="runtime-pipeline" title="Runtime Pipeline">
    <compositionRoot>The runtime composes from card_framework.cli.main using Hydra config from src/card_framework/config/config.yaml and resolves paths relative to the original working directory.</compositionRoot>
    <stagePlanning>
      <rule>pipeline.start_stage controls execution mode and is validated by pipeline_plan.build_pipeline_stage_plan.</rule>
      <mode id="stage-1">Run audio stage, then summarizer/critic loop, then optional voice clone and optional stage-4 interjector.</mode>
      <mode id="stage-2">Skip audio generation and start from an existing transcript JSON, then run summarizer/critic loop and optional voice clone and interjector. With live drafting enabled, stage-2 owns per-turn stage-3 rendering while the draft is being built.</mode>
      <mode id="stage-3">Skip to voice clone using an existing summary XML and transcript metadata that already points to a speaker-sample manifest. With live drafting enabled, this path still routes through the incremental live renderer in one batch so it emits the same manifest shape as stage-2 finalization.</mode>
      <mode id="stage-4">Skip directly to the interjector using an existing summary XML plus an existing stage-3 voice-clone manifest.</mode>
    </stagePlanning>
    <runtimeFlow>
      <step>Resolve logging, project root, pipeline config, audio config, and runtime device.</step>
      <step>If stage-1 is active, run audio_pipeline.AudioToScriptOrchestrator to emit transcript JSON and stage artifacts.</step>
      <step>Load transcript JSON into orchestration.transcript.Transcript unless pipeline.start_stage=stage-4, which runs from summary XML plus voice-clone manifest only.</step>
      <step>Generate speaker samples after transcript availability when configured, or defer them until just before the voice-clone work begins when `audio.speaker_samples.defer_until_voice_clone=true`. In live-draft mode the deferred path is still pulled forward before the stage-2 summarizer loop starts because actual audio rendering needs the manifest immediately. When stage-2 reuses a transcript that does not already carry a valid `metadata.speaker_samples_manifest_path`, the runtime now bootstraps a fresh audio inference pass from source audio, validates that inferred transcript against the reusable transcript before accepting it, remaps bootstrap speaker IDs back onto the reusable transcript speaker labels when possible, and writes only the new speaker-sample metadata back onto the reusable transcript.</step>
      <step>When live drafting is enabled and voice cloning is active, skip calibration entirely and let the summarizer and critic consume actual rendered turn durations from the persisted live-draft sidecar. When live drafting is disabled, ensure the project-level voice-clone calibration artifact exists before duration-aware summarization or critic evaluation. The calibration path can reuse an existing speaker-sample manifest or bootstrap one from transcript or audio context, but it must be regenerated when the active manifest or calibrated speaker coverage changes.</step>
      <step>When `orchestrator.loop_memory.artifact_path` is configured, the stage orchestrator scopes summarizer loop-memory artifacts by transcript hash and duration target so repeated failed critic remedies can be surfaced on later revise passes instead of being retried verbatim.</step>
      <step>Instantiate shared or per-stage LLM providers, including the optional stage_llm.interjector override, embedding provider, transcript index, local A2A apps, and the orchestrator stack.</step>
      <step>Run the stage orchestrator, persist final summary XML to summary.xml, optionally finalize or render stage-3 voice cloning, and optionally run the stage-4 interjector.</step>
    </runtimeFlow>
    <retrievalPolicy>
      <rule>When a real embedding provider is configured, transcript segments are indexed into embeddings.TranscriptIndex and retrieved through the retrieval agent.</rule>
      <rule>When embedding is disabled or configured as the no-op provider, the summarizer and critic receive the full transcript text directly instead of retrieval results.</rule>
      <rule>Timeout floors are increased automatically for full-transcript mode because prompt payloads become much larger.</rule>
      <rule>When live-draft stage-2 or merged stage-2/stage-3 voice cloning is active, the summarizer timeout floor is raised further to `max(900s, target_seconds * 6)` because tool mutations can block on real IndexTTS renders before the localhost A2A request returns.</rule>
    </retrievalPolicy>
    <localAgentTopology>
      <agent>Summarizer agent runs on the configured summarizer localhost port.</agent>
      <agent>Critic agent runs on the configured critic localhost port.</agent>
      <agent>Retrieval agent runs on the configured retrieval localhost port.</agent>
      <transport>agents.client.AgentClient is the internal transport used by the orchestrator and executor layers.</transport>
      <startup>card_framework.cli.main now waits for local A2A servers with a shared parallel health-poll loop instead of a fixed preflight sleep followed by serial checks.</startup>
    </localAgentTopology>
  </section>

  <section id="agents-and-prompts" title="Agents And Prompts">
    <executors>
      <executor name="SummarizerExecutor">Tool-loop summarizer that mutates a message registry, supports revise mode, loop-context carryover, optional discovery guardrails, and retrieval-backed query tools. The summarizer is duration-first: each line carries an `emo_preset`, the tool loop auto-runs `estimate_duration`, and convergence is measured in seconds rather than target word count. Once a mutation lands in budget, the loop auto-saves the draft, injects a bounded review-only follow-up, and falls back to submitting the current draft if the model answers that review turn with plain text instead of another tool call. In live-draft mode, add/edit/remove mutations synthesize or delete turn audio immediately through `audio_pipeline/live_draft_voice_clone.py`.</executor>
      <executor name="CriticExecutor">LLM critic that combines deterministic duration, truncation, and XML checks with optional transcript verification and emits pass/fail verdict JSON including `estimated_seconds`. When a matching live-draft audio sidecar exists, critic duration checks prefer actual rendered seconds and mark the payload with `duration_source=actual_audio`.</executor>
      <executor name="InfoRetrievalExecutor">Indexes transcript segments and serves MMR-based retrieval results from embeddings.TranscriptIndex.</executor>
      <executor name="GroundTruthCreatorExecutor">Generates contract-bound QA questions from source text for the QA benchmark.</executor>
      <executor name="QAEvaluatorExecutor">Scores a candidate summary against generated QA questions, validates quote grounding, and aggregates correctness metrics.</executor>
      <executor name="LLMCorrectorAgent">Produces retry guidance for failed ground-truth generation or evaluator turns.</executor>
    </executors>
    <agentInfrastructure>
      <component path="agents/base.py">Base A2A executor loop, tool-call sanitization, replay de-duplication, and LLM tool-loop handling.</component>
      <component path="agents/dtos.py">Pydantic request and response contracts shared across runtime and benchmark workflows.</component>
      <component path="agents/client.py">Internal client for localhost A2A task submission with retry and timeout behavior.</component>
      <component path="agents/summarizer_tool_dispatcher.py">Summarizer tool execution, mutation gating, and finalize semantics.</component>
      <component path="agents/loop_context.py">Iteration memory and compact feedback carry-forward for summarizer retries.</component>
    </agentInfrastructure>
    <prompts>
      <prompt path="src/card_framework/prompts/templates/summarizer_system.jinja2">Primary summarizer system instructions.</prompt>
      <prompt path="src/card_framework/prompts/templates/summarizer_revise.jinja2">Revision-mode summarizer instructions.</prompt>
      <prompt path="src/card_framework/prompts/templates/critic_system.jinja2">Critic system instructions.</prompt>
      <prompt path="src/card_framework/prompts/templates/interjector_system.jinja2">Stage-4 planner instructions for overlap decisions and anchor spans.</prompt>
      <prompt path="src/card_framework/prompts/templates/interjector_user.jinja2">Stage-4 planner user payload describing eligible host turns and token indices.</prompt>
      <prompt path="src/card_framework/prompts/templates/qa_ground_truth_system.jinja2">QA ground-truth generation instructions.</prompt>
      <prompt path="src/card_framework/prompts/templates/qa_evaluator_system.jinja2">QA evaluator instructions.</prompt>
      <prompt path="src/card_framework/prompts/templates/corrector_system.jinja2">Corrector retry-guidance instructions.</prompt>
      <manager path="src/card_framework/shared/prompt_manager.py">Jinja2 loading and rendering is centralized in PromptManager.</manager>
    </prompts>
    <editingGuidance>
      <rule>If you change tool schemas, task DTOs, or prompt expectations, inspect both the executor implementation and the matching template files.</rule>
      <rule>Keep prompt variables aligned with PromptManager usage and the calling executor.</rule>
      <rule>Update tests for prompt-facing contract changes, especially around QA evaluator and summarizer tool-loop behavior.</rule>
    </editingGuidance>
  </section>

  <section id="data-contracts" title="Data Contracts">
    <transcriptContract>
      <shape>Transcript JSON is centered on a top-level segments array and optional metadata object.</shape>
      <segmentFields>Each normalized segment carries speaker, start_time, end_time, and text. Extra fields may exist and should be preserved when practical.</segmentFields>
      <metadataFields>Metadata may include source paths, device, warnings, alignment information, speaker-sample manifest path, speaker-sample directory, and generation timestamps.</metadataFields>
      <typedBoundary>orchestration.transcript.Transcript and TranscriptSegment are the typed domain DTOs used by the stage orchestrator.</typedBoundary>
    </transcriptContract>
    <summaryContract>
      <shape>The final summary is XML text persisted to workspace-root summary.xml through summary_output.write_summary_xml_to_workspace.</shape>
      <turnContract>Each speaker turn is stored as XML with an optional `emo_preset` attribute. Missing attributes remain backward-compatible and default to `neutral`.</turnContract>
      <consumer>Stage-3 voice cloning uses summary XML speaker turns to decide how many voice-clone artifacts to generate and which speaker samples to match; stage-4 interjection reuses the same summary XML plus the stage-3 manifest to place overlaps against synthesized audio timing.</consumer>
      <assumption>Critic logic and downstream processing expect speaker-tagged XML blocks rather than plain prose summaries.</assumption>
    </summaryContract>
    <agentTaskContracts>
      <contract>SummarizerTaskRequest carries `target_seconds`, `duration_tolerance_ratio`, retrieval port, feedback, previous draft, loop context, optional full transcript, and live-draft wiring fields such as `speaker_samples_manifest_path` and `draft_audio_state_path`. Legacy `min_words` and `max_words` fields still exist only as compatibility fallback, while loop context now includes repeated-remedy warnings derived from persisted critic-failure history.</contract>
      <contract>CriticTaskRequest carries the candidate draft, `target_seconds`, `duration_tolerance_ratio`, optional full transcript, and optional `draft_audio_state_path` so deterministic checks can reuse actual rendered duration telemetry. Critic responses now include both `word_count` telemetry and `estimated_seconds`.</contract>
      <contract>RetrieveTaskRequest and IndexTaskRequest define the retrieval agent boundary.</contract>
      <contract>QA benchmark DTOs in agents/dtos.py define creator, evaluator, and corrector payload shapes.</contract>
    </agentTaskContracts>
    <benchmarkContracts>
      <summarizationBenchmark>src/card_framework/benchmark/manifests/benchmark_v1.json defines transcript samples; src/card_framework/benchmark/provider_profiles.yaml defines provider configs; src/card_framework/benchmark/rubrics/default_summarization_rubric.json defines LLM-judge rubric inputs.</summarizationBenchmark>
      <qaBenchmark>benchmark.qa_contracts.GroundTruthSet enforces a 100-question contract with exactly 50 factualness and 50 naturalness questions.</qaBenchmark>
      <reporting>src/card_framework/benchmark/types.py and src/card_framework/benchmark/qa_contracts.py hold the serializable report and scoring structures that downstream tooling should preserve.</reporting>
    </benchmarkContracts>
  </section>

  <section id="audio-pipeline" title="Audio Pipeline">
    <responsibility>The audio subsystem converts source audio into a structured transcript and optional cloned-audio output while keeping heavy model integrations behind protocol-style adapters.</responsibility>
    <stageOneFlow>
      <step>Source separation produces vocals-first audio.</step>
      <step>ASR produces segment timing and optional word-level timestamps.</step>
      <step>Diarization produces speaker turns.</step>
      <step>Alignment merges ASR and diarization into normalized transcript segments.</step>
      <step>Transcript JSON is emitted atomically for later summarization and benchmark use.</step>
    </stageOneFlow>
    <postTranscriptFlow>
      <step>SpeakerSampleGenerator extracts per-speaker reference audio clips from the selected source audio.</step>
      <step>A manifest is written and its path is stored in transcript metadata.</step>
      <step>`audio_pipeline/live_draft_voice_clone.py` can keep a persisted live-draft sidecar keyed by stable `turn_id`, synthesize only changed turns, report actual segment durations back into stage-2 tooling, and finalize the live turn set into the normal stage-3 manifest without a second full render pass.</step>
      <step>Voice-clone calibration remains the legacy fallback path for stage-2 duration estimation when live drafting is disabled. It measures per-speaker WPM for the fixed emotion preset catalog by synthesizing punctuation-rich temporary phrases against the discovered speaker samples and persisting a single project artifact under `artifacts/calibration`.</step>
      <step>VoiceCloneOrchestrator consumes summary XML plus the manifest to create per-turn cloned segments and optional merged output. Live-draft finalization now emits the same manifest shape plus extra fields such as `turn_id`, `duration_ms`, `word_count`, and `actual_wpm` for downstream consumers. Each turn resolves its stored `emo_preset` to repo-configured `emo_text` before calling IndexTTS.</step>
      <step>The default IndexTTS subprocess backend now keeps one warm nested worker per matching voice-clone config, so calibration, stage-3 cloning, and stage-4 interjection reuse the same loaded weights until the provider is explicitly closed or the parent process exits.</step>
      <step>Within that warm runtime, repeated `emo_text` preset prompts are cached as resolved emotion vectors and forwarded back into IndexTTS directly, avoiding repeated Qwen emotion-analysis passes for common presets such as `neutral`, `warm`, or `engaged`.</step>
      <step>The repo-default low-latency IndexTTS path forwards `length_penalty` only when beam search is actually enabled, so the default `num_beams=1` configuration does not emit invalid-generation-flag warnings on every live-draft render.</step>
      <step>InterjectorOrchestrator can then analyze eligible host turns, align stage-3 audio back to summary text, synthesize short overlaps from the next speaker, and emit a second merged WAV plus its own manifest. The stage-4 planner prompt now includes a per-turn preferred anchor token window derived from the configured host-progress ratios, requests only positive overlap decisions so long runs do not waste tokens on explicit false entries, and the parser can salvage fully formed decisions from a truncated JSON array before synthesis.</step>
    </postTranscriptFlow>
    <adapterMap>
      <adapter>Demucs handles source separation when separation.provider=demucs.</adapter>
      <adapter>Faster-Whisper handles transcription and optional forced alignment.</adapter>
      <adapter>NeMo MSDD remains the default diarization backend, with optional single-speaker fallback behavior controlled in config.</adapter>
      <adapter>Alternative diarization backends now include pyannote Community-1 plus NeMo Sortformer offline and streaming checkpoints, all wired through `audio_pipeline.factory.build_speaker_diarizer` and selected by `audio.diarization.provider`.</adapter>
      <adapter>FFmpeg-backed export is used for speaker sample generation.</adapter>
      <adapter>IndexTTS and passthrough voice-clone providers are composed through the voice-clone gateway layer, and the IndexTTS gateway exposes explicit runtime-release hooks for module offload paths plus repo-level generation tuning such as the default low-latency `num_beams=1` setting.</adapter>
    </adapterMap>
    <etaAndObservability>
      <rule>ETA estimation and learning live in audio_pipeline/eta.py and are reused across audio and voice-clone stages.</rule>
      <rule>Operator-facing ETA messages are shown only after the matching stage has learned or loaded persisted throughput history; first-run stages stay silent until history exists.</rule>
      <rule>Learned ETA throughput is persisted immediately after each completed audio, speaker-sample, and voice-clone stage so interrupted pipelines retain finished-stage history.</rule>
      <rule>Windows-only dedicated GPU heartbeat monitoring can run during voice cloning when enabled and when the resolved provider device is CUDA.</rule>
      <rule>src/card_framework/shared/events.py provides the shared event bus used by UI and logging subscribers.</rule>
    </etaAndObservability>
    <artifacts>
      <artifact>Default transcript output path: artifacts/transcripts/latest.transcript.json</artifact>
      <artifact>Default audio working directory: artifacts/audio_stage</artifact>
      <artifact>Default calibration artifact path: artifacts/calibration/voice_clone_calibration.json</artifact>
      <artifact>Speaker sample artifacts live under the configured speaker sample output directory inside the audio work directory.</artifact>
      <artifact>Voice clone artifacts live under the configured voice clone output directory inside the audio work directory.</artifact>
      <artifact>Live-draft turn audio is cached under `voice_clone/live_draft_turns`, and the corresponding sidecar state file is written alongside other voice-clone artifacts inside the voice clone output directory.</artifact>
      <artifact>Interjector artifacts live under the configured interjector output directory inside the audio work directory and include an `interjector_manifest.json` plus a merged overlap WAV.</artifact>
    </artifacts>
  </section>

  <section id="benchmarking" title="Benchmarking">
    <summarizationBenchmark>
      <workflow>benchmark.run executes provider and embedding matrix cells over a manifest of transcript samples, then aggregates runtime and reference-free metrics.</workflow>
      <subcommand name="execute">Runs the benchmark matrix with a preset, provider profiles, optional judge provider, and reference-free scoring controls.</subcommand>
      <subcommand name="prepare-manifest">Builds a frozen manifest from supported sources such as local, QMSum, or AMI.</subcommand>
      <referenceFreeMetrics>Reference-free evaluation combines AlignScore-like scoring and LLM-as-judge scoring, including order-swap and repeat diagnostics.</referenceFreeMetrics>
    </summarizationBenchmark>
    <diarizationBenchmark>
      <workflow>benchmark.diarization executes the repo's actual speaker-diarizer providers against a manifest of audio files plus reference RTTM and optional UEM files, writes predicted RTTMs per provider, and aggregates DER, optional JER, runtime, real-time factor, and peak GPU memory. The same module now also prepares a default public AMI manifest by downloading `Mix-Headset.wav` audio plus RTTM/UEM/list files from the AMI corpus and `BUTSpeechFIT/AMI-diarization-setup`.</workflow>
      <subcommand name="execute">Runs diarization provider comparisons such as NeMo MSDD, pyannote Community-1, and NeMo Sortformer variants using the same adapter factory used by stage-1 audio inference.</subcommand>
      <subcommand name="prepare-manifest">Builds the default AMI diarization manifest at `src/card_framework/benchmark/manifests/diarization_ami_test.json` and caches downloaded public assets under `artifacts/diarization_datasets/ami`.</subcommand>
      <manifestShape>The diarization manifest expects per-sample audio, reference RTTM, optional UEM, dataset, subset, and optional speaker-count metadata. Use `src/card_framework/benchmark/manifests/diarization_manifest.example.json` as the manual template for CALLHOME, DIHARD, or other local datasets.</manifestShape>
      <scoringPolicy>Diarization scoring uses `pyannote.metrics` DER for every sample and computes JER only when a UEM file is available. The default CLI policy is strict scoring with zero collar and overlapping speech included.</scoringPolicy>
    </diarizationBenchmark>
    <qaBenchmark>
      <workflow>src/card_framework/benchmark/qa.py evaluates an existing summary.xml against a source transcript by generating QA ground truth, then asking an evaluator agent to answer from the summary.</workflow>
      <providerSelection>Creator and evaluator providers may share one provider profile or be split through explicit CLI overrides.</providerSelection>
      <runtimeConfig>QA-specific limits, quote-relevance behavior, timeouts, vLLM settings, and corrector settings live in src/card_framework/benchmark/qa_config.yaml and src/card_framework/benchmark/qa_settings.py.</runtimeConfig>
      <compatibilityGuard>The QA input guard checks that the supplied summary and source transcript appear compatible before expensive evaluation begins.</compatibilityGuard>
    </qaBenchmark>
    <outputs>
      <artifact>Summarization benchmark outputs are written under artifacts/benchmark and artifacts/quality.</artifact>
      <artifact>QA benchmark outputs are written under artifacts/qa_benchmark.</artifact>
      <artifact>Verification payloads capture command provenance, git info, and report hashes for benchmark runs.</artifact>
    </outputs>
  </section>

  <section id="providers-config-and-commands" title="Providers, Config, And Commands">
    <configurationFiles>
      <file path="src/card_framework/config/config.yaml">Main runtime configuration for pipeline stages, provider selection, stage_llm overrides, audio settings, live draft voice-clone toggles, voice-clone emotion presets, calibration artifact path, duration targets, loop-memory artifact path, loop guardrails, and logging.</file>
      <file path="src/card_framework/benchmark/provider_profiles.yaml">Named provider profiles used by benchmark.run and benchmark.qa.</file>
      <file path="src/card_framework/benchmark/qa_config.yaml">QA benchmark settings for vLLM connection details, input guard, corrector, evaluator runtime, and timeouts.</file>
      <file path="pyproject.toml">Python version floor, dependencies, uv source/index policy, and pytest configuration.</file>
      <file path="uv.lock">Committed lockfile for the uv-managed environment.</file>
    </configurationFiles>
    <providerPolicy>
      <rule>Do not copy deploy-specific credentials, tokens, or endpoint secrets into docs or code comments.</rule>
      <rule>Prefer provider profile changes and Hydra config changes over scattered per-module constants.</rule>
      <rule>When changing provider request or response normalization, inspect llm_provider.py and the affected provider adapter tests together.</rule>
      <rule>Preserve the lazy-export behavior in providers/__init__.py so importing one provider module does not eagerly import unrelated heavy backends.</rule>
    </providerPolicy>
    <canonicalCommands>
      <command>uv sync --dev</command>
      <command>uv run python -m card_framework.cli.calibrate</command>
      <command>uv run python -m card_framework.cli.main</command>
      <command>uv run python -m card_framework.cli.setup_and_run --audio-path &lt;path-to-audio&gt;</command>
      <command>uv run python -m card_framework.cli.run_summary_matrix --vllm-host &lt;host&gt; --transcript-path &lt;path-to-transcript.json&gt;</command>
      <command>uv run python -m card_framework.benchmark.run execute --preset hourly</command>
      <command>uv run python -m card_framework.benchmark.run prepare-manifest --sources local</command>
      <command>uv run python -m card_framework.benchmark.diarization prepare-manifest</command>
      <command>uv run python -m card_framework.benchmark.diarization execute</command>
      <command>uv run python -m card_framework.benchmark.qa --summary-xml &lt;path-to-summary.xml&gt; --source-transcript &lt;path-to-transcript&gt;</command>
      <command>uv run ruff check .</command>
      <command>uv run pytest</command>
    </canonicalCommands>
    <commandNotes>
      <note>Use uv run for repo commands so execution stays inside the locked environment.</note>
      <note>Benchmark, diarization benchmark, QA, runtime, calibration, and setup helpers are all package module entrypoints under `card_framework.*`.</note>
      <note>The diarization benchmark uses the same `audio_pipeline.factory.build_speaker_diarizer(...)` path as the runtime so provider benchmark results stay aligned with stage-1 behavior.</note>
      <note>The default AMI prep path currently targets the public `Mix-Headset` stream and the `only_words` RTTM/UEM setup from `BUTSpeechFIT/AMI-diarization-setup`.</note>
      <note>`audio.diarization.provider` now supports the default NeMo MSDD path, `pyannote_community1`, `nemo_sortformer_offline`, `nemo_sortformer_streaming`, and the existing `single_speaker` fallback backend.</note>
      <note>`card_framework.cli.calibrate` is the project-level one-time calibration helper. It prints the persisted preset/WPM mapping and can bootstrap speaker samples from a manifest, transcript, or raw audio path.</note>
      <note>`card_framework.cli.setup_and_run` is the preferred convenience wrapper when the task involves IndexTTS setup and full end-to-end stage execution.</note>
      <note>When `card_framework.cli.setup_and_run` enables terminal logging, it now leaves `logging.summarizer_critic_print_to_terminal` unset so the live Summarizer and Critic reasoning/content stream follows the repo logging config unless the caller explicitly disables it.</note>
      <note>The `LoggingLLMProvider` terminal path now logs concise message/tool/response summaries at `INFO` and reserves full prompt, tool-schema, and response payload dumps for `DEBUG`, so streamed runs stay readable while deeper payload inspection remains opt-in.</note>
      <note>The Rich UI now parses JSON-shaped `tool_result` payloads before printing them, so dict and list outputs render as structured multiline blocks and transcript excerpts keep their real line breaks instead of showing escaped `\n` sequences.</note>
      <note>When stdout is not a real terminal, such as child runs launched through `card_framework.cli.run_summary_matrix`, the shared streamed-response callback path used by the repo's vLLM and DeepSeek providers falls back from Rich live panels to plain streamed `[THINKING]` and `[CONTENT]` text, while ignoring whitespace-only preambles so the parent pipe receives meaningful incremental model output instead of blank section headers.</note>
      <note>`card_framework.cli.run_summary_matrix` is the operator helper for ordered summarizer/critic pair comparisons, including self-pairs. It still drives `card_framework.cli.setup_and_run` so stage selection and summarizer/critic orchestration stay aligned with the repo default workflow, preserves the merged live-draft stage-2/stage-3 path, disables only stage-4 interjector output, streams live child output to the parent terminal, and writes one copied summary file per summarizer/critic pair under `artifacts/summary_matrix`.</note>
      <note>When no explicit start stage is provided, `card_framework.cli.setup_and_run` now auto-selects stage-2 if it finds a reusable transcript, preferring repo-root `transcript.json` or `*.transcript.json` before `artifacts/transcripts/*.transcript.json`.</note>
      <note>When stage-2 is auto-selected while a repo-root `summary.xml` already exists, `card_framework.cli.setup_and_run` prints that it will still rerun summarization before voice cloning and points operators at `--voiceclone-from-summary` for stage-3 clone-only runs.</note>
      <note>For stage-2 reusable transcripts that lack a valid `metadata.speaker_samples_manifest_path`, `card_framework.cli.setup_and_run` now resolves source audio from transcript metadata first, then from a repo-root `audio.wav`-style file, and otherwise prompts so the runtime can bootstrap fresh separation, transcription, diarization, and speaker-sample generation. The runtime will reject that bootstrap result early when the inferred transcript or speaker coverage does not match the reusable transcript.</note>
      <note>When a stage-2 reusable transcript already carries a valid `metadata.speaker_samples_manifest_path`, the runtime reuses that manifest and skips the speaker-sample bootstrap audio pass.</note>
      <note>`audio.voice_clone.live_drafting.enabled=true` is now the default merged stage-2/stage-3 path when voice cloning is enabled. In that mode `card_framework.cli.setup_and_run` skips calibration and the runtime uses actual rendered turn durations instead of WPM estimates.</note>
      <note>`card_framework.cli.setup_and_run` pre-runs calibration only for the legacy stage-2 estimate path, skips calibration for stage-3 clone-only runs, and explicitly defers calibration to `card_framework.cli.main` for fresh stage-1 runs when the legacy estimate path is active so it does not duplicate the audio stage.</note>
      <note>Stage-2 calibration logs only appear in the legacy estimate path now. Live drafting should emit actual turn renders during the summarizer loop instead of temporary calibration syntheses.</note>
      <note>When CLI overrides do not specify `audio.voice_clone.enabled`, `audio.interjector.enabled`, or `audio.speaker_samples.enabled`, `card_framework.cli.setup_and_run` now respects the defaults declared in `src/card_framework/config/config.yaml` instead of forcing wrapper-local fallback values.</note>
      <note>`card_framework.cli.setup_and_run` now skips IndexTTS repo sync, nested uv sync, and model provisioning only when neither final synthesis nor the legacy calibration-backed estimate path needs the voice-clone runtime.</note>
      <note>Use audio.speaker_samples.defer_until_voice_clone=true when time-to-first-summary matters more than precomputing speaker sample artifacts ahead of live drafting or stage-3 rendering.</note>
    </commandNotes>
  </section>

  <section id="testing-and-maintenance" title="Testing And Maintenance">
    <testLayout>
      <group path="tests/agents">Agent loop behavior, DTO interactions, correction logic, parser fallback behavior, and QA evaluator behavior.</group>
      <group path="tests/audio_pipeline">Alignment, ETA, gateways, sample generation, voice clone orchestration, and audio-stage behavior.</group>
      <group path="tests/benchmark">Manifest, matrix, QA config, QA contracts, reference-free evaluation, and benchmark runner behavior.</group>
      <group path="tests/cli">Operator-facing CLI and terminal UI behavior.</group>
      <group path="tests/orchestration">Stage orchestrator and transcript DTO behavior.</group>
      <group path="tests/providers">Provider normalization, logging wrapper, null provider, and vLLM reasoning behavior.</group>
      <group path="tests/runtime">Loop orchestrator and pipeline-stage planning behavior.</group>
      <group path="tests/shared">Shared utilities such as summary persistence and logging helpers.</group>
      <group path="tests/real">Packaged-resource and CLI smoke coverage that exercises the installed package shape.</group>
      <group path="tests/support">Reusable test-only helpers for localhost servers and A2A message extraction. Keep them small and behavior-focused so tests reuse infrastructure without growing a parallel framework.</group>
    </testLayout>
    <pytestPolicy>
      <marker>unit: deterministic fast unit tests</marker>
      <marker>integration: component integration tests using local dependencies</marker>
      <marker>contract: shared interface or adapter behavior tests</marker>
      <marker>slow: intentionally slow tests</marker>
      <marker>gpu: tests that require GPU acceleration</marker>
      <marker>network: tests that require external network access</marker>
    </pytestPolicy>
    <maintenanceRules>
      <rule>When changing runtime flow, prompts, provider behavior, contracts, benchmark workflow, or operator-facing commands, update this guide in the same change.</rule>
      <rule>When a change affects tested behavior, add or update tests in the corresponding subsystem.</rule>
      <rule>When you fix a meaningful error, recurring pitfall, or debugging trap, prepend a short dated and timed Problem and Solution note to the top of coder_docs/memory/errors_and_notes.md.</rule>
      <rule>When contributor Git or GitHub workflow expectations change materially, update coder_docs/git_github_workflow.md and this guide in the same change.</rule>
      <rule>When the repository's bug-triage or fault-localization workflow changes materially, update coder_docs/fault_localization_workflow.md and this guide in the same change.</rule>
      <rule>Before commit or pull request, review staged diffs for secrets, private endpoints such as vLLM URLs, credentials, tokens, and other private data. Do not let them enter repo history.</rule>
      <rule>When methodology-sensitive benchmark logic changes, update coder_docs/academic_standards.md usage at the implementation site and keep citations or gap notes current.</rule>
      <rule>Keep README.md, src/card_framework/benchmark/README.md, and coder_docs files aligned when commands or workflows move.</rule>
      <rule>Prefer architecture-preserving edits: orchestration logic in orchestration or main, protocol and adapter logic in audio_pipeline or providers, benchmark-specific behavior in benchmark, and A2A task behavior in agents.</rule>
    </maintenanceRules>
  </section>
</codebaseGuide>
