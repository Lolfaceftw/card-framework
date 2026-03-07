<?xml version="1.0" encoding="UTF-8"?>
<codebaseGuide workspace="llm_test" package="llm-test" language="Python" pythonVersion="3.12+" packageManager="uv" linter="ruff">
  <overview>
    <purpose>Orient a new coding agent to the actual architecture, workflows, contracts, and maintenance expectations of this repository before code changes are made.</purpose>
    <repositoryIdentity>This workspace combines a Hydra-configured summarization runtime, an audio-to-transcript and voice-clone pipeline, multiple LLM and embedding provider adapters, and two benchmark systems: a matrix-style summarization benchmark and a source-grounded QA benchmark.</repositoryIdentity>
    <sessionStartChecklist>
      <step>Read this file first to understand subsystem boundaries, entrypoints, and data flow.</step>
      <step>Review coder_docs/memory/errors_and_notes.md for previously captured pitfalls before repeating work in an area.</step>
      <step>Use coder_docs/academic_standards.md for formulas, thresholds, scoring logic, or methodology-sensitive implementations.</step>
      <step>Use coder_docs/ruff.md for lint workflow, coder_docs/uv_package_manager.md for dependency and environment workflow, coder_docs/git_github_workflow.md for contributor Git and GitHub workflow, and coder_docs/scrapling.md for external web retrieval workflow.</step>
      <step>Inspect the exact config and prompt files for the subsystem you are changing before editing behavior.</step>
      <step>When behavior, configuration, commands, or workflow changes materially, update this guide in the same change.</step>
    </sessionStartChecklist>
    <primaryWorkflow>
      <stage id="stage-1">Audio stage: source separation, ASR, diarization, alignment, and transcript JSON emission.</stage>
      <stage id="stage-2">Summarization stage: local A2A retrieval, summarizer, and critic agents cooperate until convergence or iteration cap.</stage>
      <stage id="stage-3">Voice-clone stage: summary XML plus generated speaker samples are rendered into per-turn cloned audio and optional merged output.</stage>
      <stage id="stage-4">Interjector stage: the next speaker can add short overlapping backchannels or echo-agreements onto the stage-3 merged audio using summary-aware token anchors.</stage>
    </primaryWorkflow>
    <agentOperatingPrinciples>
      <principle>Treat configuration as authoritative; prefer overrides and config edits over hardcoded values.</principle>
      <principle>Preserve the separation between orchestration, provider adapters, prompts, and data-contract modules.</principle>
      <principle>Favor behavior-level understanding first: runtime pipeline, audio pipeline, and benchmark workflows are all first-class parts of this repo.</principle>
    </agentOperatingPrinciples>
  </overview>

  <section id="repo-layout" title="Repository Layout">
    <directory path="agents">A2A executor implementations, task DTOs, tool dispatch, loop control, parsing, correction guidance, and client transport helpers.</directory>
    <directory path="audio_pipeline">Stage-1 and stage-3 use-case orchestration, contracts, ETA tracking, runtime helpers, gateways, runners, speaker-sample generation, and voice-clone orchestration.</directory>
    <directory path="benchmark">Summarization benchmark CLI, reference-free evaluation pipeline, QA benchmark CLI, manifests, rubrics, metrics, matrix construction, and report artifact helpers.</directory>
    <directory path="providers">Concrete LLM and embedding adapters such as vLLM, Transformers, DeepSeek, GLM, Google GenAI, Hugging Face, Nanbeige, sentence-transformer embeddings, and logging wrappers.</directory>
    <directory path="prompts">Jinja2 prompt templates for summarizer, critic, QA ground-truth generation, QA evaluator, and corrector flows.</directory>
    <directory path="orchestration">Typed transcript DTOs and the stage orchestrator that bridges stage plans with runtime execution.</directory>
    <directory path="conf">Hydra application configuration, including provider selection, stage controls, audio settings, ports, orchestrator limits, and logging.</directory>
    <directory path="tests">Pytest coverage grouped by subsystem: agents, audio_pipeline, benchmark, bootstrap, orchestration, providers, and top-level runtime behavior.</directory>
    <directory path="coder_docs">Project-local policy and workflow documentation. This file is the architecture/session guide; sibling docs cover methodology, linting, package management, Git and GitHub workflow, and web retrieval workflow.</directory>
    <file path="AGENTS.md">Repo-local coding-agent instructions, including the required Scrapling web-research policy and the rule to keep coder_docs aligned with prompt changes.</file>
    <entrypoints>
      <entrypoint path="main.py">Primary Hydra runtime for the summarization, audio, and voice-clone pipeline.</entrypoint>
      <entrypoint path="calibrate.py">One-time voice-clone calibration helper that discovers or bootstraps speaker samples, renders punctuation-rich calibration phrases, and prints the persisted preset/WPM mapping.</entrypoint>
      <entrypoint path="setup_and_run.py">Bootstrap and convenience runner for dependency checks, stage-aware third-party sync/model provisioning, and full pipeline execution.</entrypoint>
      <entrypoint path="benchmark/run.py">CLI for summarization benchmark matrix execution and manifest preparation.</entrypoint>
      <entrypoint path="benchmark/qa.py">CLI for source-grounded QA benchmark execution against an existing summary and source transcript.</entrypoint>
      <entrypoint path="eval.py">Compatibility wrapper that forwards to the benchmark smoke preset.</entrypoint>
      <entrypoint path="ui.py">Rich terminal presentation layer that subscribes to the shared event bus.</entrypoint>
    </entrypoints>
  </section>

  <section id="runtime-pipeline" title="Runtime Pipeline">
    <compositionRoot>The runtime composes from main.py using Hydra config from conf/config.yaml and resolves paths relative to the original working directory.</compositionRoot>
    <stagePlanning>
      <rule>pipeline.start_stage controls execution mode and is validated by pipeline_plan.build_pipeline_stage_plan.</rule>
      <mode id="stage-1">Run audio stage, then summarizer/critic loop, then optional voice clone and optional stage-4 interjector.</mode>
      <mode id="stage-2">Skip audio generation and start from an existing transcript JSON, then run summarizer/critic loop and optional voice clone and interjector.</mode>
      <mode id="stage-3">Skip to voice clone using an existing summary XML and transcript metadata that already points to a speaker-sample manifest.</mode>
      <mode id="stage-4">Skip directly to the interjector using an existing summary XML plus an existing stage-3 voice-clone manifest.</mode>
    </stagePlanning>
    <runtimeFlow>
      <step>Resolve logging, project root, pipeline config, audio config, and runtime device.</step>
      <step>If stage-1 is active, run audio_pipeline.AudioToScriptOrchestrator to emit transcript JSON and stage artifacts.</step>
      <step>Load transcript JSON into orchestration.transcript.Transcript unless pipeline.start_stage=stage-4, which runs from summary XML plus voice-clone manifest only.</step>
      <step>Generate speaker samples after transcript availability when configured, or defer them until just before stage-3 voice cloning when audio.speaker_samples.defer_until_voice_clone=true. Speaker-sample extraction prefers transcript metadata `vocals_audio_path` and falls back to configured `audio.audio_path` when reusable transcripts do not carry stem metadata.</step>
      <step>Ensure the project-level voice-clone calibration artifact exists before duration-aware summarization, critic evaluation, or stage-3 voice cloning. The calibration path can reuse an existing speaker-sample manifest or bootstrap one from transcript or audio context, but it must be regenerated when the active manifest or calibrated speaker coverage changes.</step>
      <step>When `orchestrator.loop_memory.artifact_path` is configured, the stage orchestrator scopes summarizer loop-memory artifacts by transcript hash and duration target so repeated failed critic remedies can be surfaced on later revise passes instead of being retried verbatim.</step>
      <step>Instantiate shared or per-stage LLM providers, including the optional stage_llm.interjector override, embedding provider, transcript index, local A2A apps, and the orchestrator stack.</step>
      <step>Run the stage orchestrator, persist final summary XML to summary.xml, optionally run stage-3 voice cloning, and optionally run the stage-4 interjector.</step>
    </runtimeFlow>
    <retrievalPolicy>
      <rule>When a real embedding provider is configured, transcript segments are indexed into embeddings.TranscriptIndex and retrieved through the retrieval agent.</rule>
      <rule>When embedding is disabled or configured as the no-op provider, the summarizer and critic receive the full transcript text directly instead of retrieval results.</rule>
      <rule>Timeout floors are increased automatically for full-transcript mode because prompt payloads become much larger.</rule>
    </retrievalPolicy>
    <localAgentTopology>
      <agent>Summarizer agent runs on the configured summarizer localhost port.</agent>
      <agent>Critic agent runs on the configured critic localhost port.</agent>
      <agent>Retrieval agent runs on the configured retrieval localhost port.</agent>
      <transport>agents.client.AgentClient is the internal transport used by the orchestrator and executor layers.</transport>
      <startup>main.py now waits for local A2A servers with a shared parallel health-poll loop instead of a fixed preflight sleep followed by serial checks.</startup>
    </localAgentTopology>
  </section>

  <section id="agents-and-prompts" title="Agents And Prompts">
    <executors>
      <executor name="SummarizerExecutor">Tool-loop summarizer that mutates a message registry, supports revise mode, loop-context carryover, optional discovery guardrails, and retrieval-backed query tools. The summarizer is now duration-first: each line carries an `emo_preset`, the tool loop auto-runs `estimate_duration`, and convergence is measured in estimated seconds rather than target word count.</executor>
      <executor name="CriticExecutor">LLM critic that combines deterministic duration, truncation, and XML checks with optional transcript verification and emits pass/fail verdict JSON including `estimated_seconds`.</executor>
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
      <prompt path="prompts/summarizer_system.jinja2">Primary summarizer system instructions.</prompt>
      <prompt path="prompts/summarizer_revise.jinja2">Revision-mode summarizer instructions.</prompt>
      <prompt path="prompts/critic_system.jinja2">Critic system instructions.</prompt>
      <prompt path="prompts/interjector_system.jinja2">Stage-4 planner instructions for overlap decisions and anchor spans.</prompt>
      <prompt path="prompts/interjector_user.jinja2">Stage-4 planner user payload describing eligible host turns and token indices.</prompt>
      <prompt path="prompts/qa_ground_truth_system.jinja2">QA ground-truth generation instructions.</prompt>
      <prompt path="prompts/qa_evaluator_system.jinja2">QA evaluator instructions.</prompt>
      <prompt path="prompts/corrector_system.jinja2">Corrector retry-guidance instructions.</prompt>
      <manager path="prompt_manager.py">Jinja2 loading and rendering is centralized in PromptManager.</manager>
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
      <contract>SummarizerTaskRequest carries `target_seconds`, `duration_tolerance_ratio`, retrieval port, feedback, previous draft, loop context, and optional full transcript. Legacy `min_words` and `max_words` fields still exist only as compatibility fallback, while loop context now includes repeated-remedy warnings derived from persisted critic-failure history.</contract>
      <contract>CriticTaskRequest carries the candidate draft, `target_seconds`, `duration_tolerance_ratio`, and optional full transcript. Critic responses now include both `word_count` telemetry and `estimated_seconds`.</contract>
      <contract>RetrieveTaskRequest and IndexTaskRequest define the retrieval agent boundary.</contract>
      <contract>QA benchmark DTOs in agents/dtos.py define creator, evaluator, and corrector payload shapes.</contract>
    </agentTaskContracts>
    <benchmarkContracts>
      <summarizationBenchmark>benchmark/manifests/benchmark_v1.json defines transcript samples; benchmark/provider_profiles.yaml defines provider configs; benchmark/rubrics/default_summarization_rubric.json defines LLM-judge rubric inputs.</summarizationBenchmark>
      <qaBenchmark>benchmark.qa_contracts.GroundTruthSet enforces a 100-question contract with exactly 50 factualness and 50 naturalness questions.</qaBenchmark>
      <reporting>benchmark/types.py and benchmark/qa_contracts.py hold the serializable report and scoring structures that downstream tooling should preserve.</reporting>
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
      <step>Voice-clone calibration measures per-speaker WPM for the fixed emotion preset catalog by synthesizing punctuation-rich temporary phrases against the discovered speaker samples and persisting a single project artifact under `artifacts/calibration`.</step>
      <step>VoiceCloneOrchestrator consumes summary XML plus the manifest to create per-turn cloned segments and optional merged output. Each turn resolves its stored `emo_preset` to repo-configured `emo_text` before calling IndexTTS.</step>
      <step>The default IndexTTS subprocess backend now keeps one warm nested worker per matching voice-clone config, so calibration, stage-3 cloning, and stage-4 interjection reuse the same loaded weights until the provider is explicitly closed or the parent process exits.</step>
      <step>Within that warm runtime, repeated `emo_text` preset prompts are cached as resolved emotion vectors and forwarded back into IndexTTS directly, avoiding repeated Qwen emotion-analysis passes for common presets such as `neutral`, `warm`, or `engaged`.</step>
      <step>InterjectorOrchestrator can then analyze eligible host turns, align stage-3 audio back to summary text, synthesize short overlaps from the next speaker, and emit a second merged WAV plus its own manifest.</step>
    </postTranscriptFlow>
    <adapterMap>
      <adapter>Demucs handles source separation when separation.provider=demucs.</adapter>
      <adapter>Faster-Whisper handles transcription and optional forced alignment.</adapter>
      <adapter>NeMo handles diarization, with optional single-speaker fallback behavior controlled in config.</adapter>
      <adapter>FFmpeg-backed export is used for speaker sample generation.</adapter>
      <adapter>IndexTTS and passthrough voice-clone providers are composed through the voice-clone gateway layer, and the IndexTTS gateway exposes explicit runtime-release hooks for module offload paths plus repo-level generation tuning such as the default low-latency `num_beams=1` setting.</adapter>
    </adapterMap>
    <etaAndObservability>
      <rule>ETA estimation and learning live in audio_pipeline/eta.py and are reused across audio and voice-clone stages.</rule>
      <rule>Operator-facing ETA messages are shown only after the matching stage has learned or loaded persisted throughput history; first-run stages stay silent until history exists.</rule>
      <rule>Learned ETA throughput is persisted immediately after each completed audio, speaker-sample, and voice-clone stage so interrupted pipelines retain finished-stage history.</rule>
      <rule>Windows-only dedicated GPU heartbeat monitoring can run during voice cloning when enabled and when the resolved provider device is CUDA.</rule>
      <rule>events.py provides the shared event bus used by UI and logging subscribers.</rule>
    </etaAndObservability>
    <artifacts>
      <artifact>Default transcript output path: artifacts/transcripts/latest.transcript.json</artifact>
      <artifact>Default audio working directory: artifacts/audio_stage</artifact>
      <artifact>Default calibration artifact path: artifacts/calibration/voice_clone_calibration.json</artifact>
      <artifact>Speaker sample artifacts live under the configured speaker sample output directory inside the audio work directory.</artifact>
      <artifact>Voice clone artifacts live under the configured voice clone output directory inside the audio work directory.</artifact>
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
    <qaBenchmark>
      <workflow>benchmark/qa.py evaluates an existing summary.xml against a source transcript by generating QA ground truth, then asking an evaluator agent to answer from the summary.</workflow>
      <providerSelection>Creator and evaluator providers may share one provider profile or be split through explicit CLI overrides.</providerSelection>
      <runtimeConfig>QA-specific limits, quote-relevance behavior, timeouts, vLLM settings, and corrector settings live in benchmark/qa_config.yaml and benchmark/qa_settings.py.</runtimeConfig>
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
      <file path="conf/config.yaml">Main runtime configuration for pipeline stages, provider selection, stage_llm overrides, audio settings, voice-clone emotion presets, calibration artifact path, duration targets, loop-memory artifact path, loop guardrails, and logging.</file>
      <file path="benchmark/provider_profiles.yaml">Named provider profiles used by benchmark.run and benchmark.qa.</file>
      <file path="benchmark/qa_config.yaml">QA benchmark settings for vLLM connection details, input guard, corrector, evaluator runtime, and timeouts.</file>
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
      <command>uv run python calibrate.py</command>
      <command>uv run python main.py</command>
      <command>uv run python setup_and_run.py --audio-path &lt;path-to-audio&gt;</command>
      <command>uv run python -m benchmark.run execute --preset hourly</command>
      <command>uv run python -m benchmark.run prepare-manifest --sources local --output benchmark/manifests/benchmark_v1.json</command>
      <command>uv run python -m benchmark.qa --summary-xml &lt;path-to-summary.xml&gt; --source-transcript &lt;path-to-transcript&gt;</command>
      <command>uv run ruff check .</command>
      <command>uv run pytest</command>
    </canonicalCommands>
    <commandNotes>
      <note>Use uv run for repo commands so execution stays inside the locked environment.</note>
      <note>Benchmark and QA CLIs are module entrypoints; main.py is a direct script entrypoint under Hydra.</note>
      <note>`calibrate.py` is the project-level one-time calibration helper. It prints the persisted preset/WPM mapping and can bootstrap speaker samples from a manifest, transcript, or raw audio path.</note>
      <note>setup_and_run.py is the preferred convenience wrapper when the task involves third-party IndexTTS setup and full end-to-end stage execution.</note>
      <note>When no explicit start stage is provided, setup_and_run.py now auto-selects stage-2 if it finds a reusable transcript, preferring repo-root `transcript.json` or `*.transcript.json` before `artifacts/transcripts/*.transcript.json`.</note>
      <note>When stage-2 is auto-selected while a repo-root `summary.xml` already exists, setup_and_run.py prints that it will still rerun summarization before voice cloning and points operators at `--voiceclone-from-summary` for stage-3 clone-only runs.</note>
      <note>For stage-2 reusable transcripts that lack `metadata.vocals_audio_path`, setup_and_run.py auto-uses a repo-root `audio.wav`-style source file when present and otherwise prompts for source audio so speaker-sample generation and calibration can still run.</note>
      <note>When a stage-2 reusable transcript already carries a valid `metadata.speaker_samples_manifest_path`, the runtime now reuses that manifest instead of regenerating speaker samples from fallback source audio.</note>
      <note>setup_and_run.py pre-runs calibration for stage-2 and stage-3 flows, and explicitly defers calibration to `main.py` for fresh stage-1 runs so it does not duplicate the audio stage.</note>
      <note>Stage-2 and stage-3 calibration can emit IndexTTS inference logs before summarization or the final voice-clone pass starts; those syntheses are temporary calibration phrases, not the persisted stage-3 output.</note>
      <note>When CLI overrides do not specify `audio.voice_clone.enabled`, `audio.interjector.enabled`, or `audio.speaker_samples.enabled`, setup_and_run.py now respects the defaults declared in `conf/config.yaml` instead of forcing wrapper-local fallback values.</note>
      <note>setup_and_run.py now skips IndexTTS repo sync, nested uv sync, and model provisioning only when both audio.voice_clone.enabled=false and audio.interjector.enabled=false.</note>
      <note>Use audio.speaker_samples.defer_until_voice_clone=true when time-to-first-summary matters more than precomputing speaker sample artifacts ahead of stage-3.</note>
    </commandNotes>
  </section>

  <section id="testing-and-maintenance" title="Testing And Maintenance">
    <testLayout>
      <group path="tests/agents">Agent loop behavior, DTO interactions, correction logic, parser fallback behavior, and QA evaluator behavior.</group>
      <group path="tests/audio_pipeline">Alignment, ETA, gateways, sample generation, voice clone orchestration, and audio-stage behavior.</group>
      <group path="tests/benchmark">Manifest, matrix, QA config, QA contracts, reference-free evaluation, and benchmark runner behavior.</group>
      <group path="tests/bootstrap">setup_and_run.py bootstrap and override logic.</group>
      <group path="tests/orchestration">Stage orchestrator and transcript DTO behavior.</group>
      <group path="tests/providers">Provider normalization, logging wrapper, null provider, and vLLM reasoning behavior.</group>
      <group path="tests/*.py">Top-level runtime behavior such as pipeline planning, loop context, summary persistence, and UI filtering.</group>
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
      <rule>Before commit or pull request, review staged diffs for secrets, private endpoints such as vLLM URLs, credentials, tokens, and other private data. Do not let them enter repo history.</rule>
      <rule>When methodology-sensitive benchmark logic changes, update coder_docs/academic_standards.md usage at the implementation site and keep citations or gap notes current.</rule>
      <rule>Keep README.md, benchmark/README.md, and coder_docs files aligned when commands or workflows move.</rule>
      <rule>Prefer architecture-preserving edits: orchestration logic in orchestration or main, protocol and adapter logic in audio_pipeline or providers, benchmark-specific behavior in benchmark, and A2A task behavior in agents.</rule>
    </maintenanceRules>
  </section>
</codebaseGuide>
