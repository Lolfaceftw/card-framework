from collections.abc import Callable, Sequence
import asyncio
import logging
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, cast

import hydra
import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import InMemoryQueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from omegaconf import DictConfig, OmegaConf

from agents.client import AgentClient
from agents.critic import CriticExecutor
from agents.health import AgentHealthChecker
from agents.retrieval import InfoRetrievalExecutor
from agents.summarizer import SummarizerExecutor
from agents.utils import load_transcript
from audio_pipeline import (
    build_audio_to_script_orchestrator,
    build_speaker_sample_generator,
    build_voice_clone_orchestrator,
)
from audio_pipeline.contracts import TranscriptPayload
from audio_pipeline.eta import (
    DynamicEtaTracker,
    StageEtaStrategy,
    StageProgressCallback,
    StageProgressUpdate,
    UnitStageEtaLearner,
    UnitStageEtaStrategy,
    format_eta_seconds,
)
from audio_pipeline.gpu_heartbeat import (
    VoiceCloneGpuHeartbeatService,
    WindowsNvidiaDedicatedGpuProbe,
    parse_voice_clone_gpu_heartbeat_config,
)
from audio_pipeline.io import write_transcript_atomic
from audio_pipeline.runtime import resolve_device, resolve_path
from audio_pipeline.speaker_samples import resolve_sample_source_audio_path
from embeddings import TranscriptIndex
from events import event_bus
from llm_provider import EmbeddingProvider, LLMProvider
from logger_utils import configure_logger
from orchestration import StageOrchestrator
from orchestrator import Orchestrator
from pipeline_plan import build_pipeline_stage_plan
from providers.logging_provider import LoggingLLMProvider
from providers.response_callbacks import RichConsoleResponseCallback
from orchestration.transcript import Transcript


def _suppress_chatty_logger_propagation() -> None:
    """Prevent noisy third-party logs from leaking to terminal handlers."""
    for logger_name in ["google", "google_genai", "httpx", "a2a", "uvicorn"]:
        logging.getLogger(logger_name).propagate = False


def _build_a2a_app(name: str, description: str, port: int, executor: AgentExecutor):
    """Build a Starlette A2A application from an executor."""
    agent_card = AgentCard(
        name=name,
        description=description,
        url=f"http://127.0.0.1:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            AgentSkill(
                id="default",
                name=name,
                description=description,
                tags=["summarization"],
            )
        ],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        queue_manager=InMemoryQueueManager(),
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    return a2a_app.build()


def _run_server_in_thread(name: str, app, port: int) -> threading.Thread:
    """Run a uvicorn server in a daemon thread."""
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    def _serve() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    thread = threading.Thread(target=_serve, name=name, daemon=True)
    thread.start()
    return thread


def _to_plain_dict(value: DictConfig | dict | None) -> dict:
    """Convert Hydra config nodes into plain dictionaries."""
    if isinstance(value, DictConfig):
        resolved = OmegaConf.to_container(value, resolve=True)
        if isinstance(resolved, dict):
            return resolved
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {}


def _has_target_config(value: dict | None) -> bool:
    """Return whether a config mapping includes a non-empty Hydra target."""
    if not isinstance(value, dict):
        return False
    target = value.get("_target_")
    return isinstance(target, str) and bool(target.strip())


def _instantiate_llm_provider(
    llm_cfg: DictConfig | dict,
    *,
    enable_logging: bool,
) -> LLMProvider:
    """Instantiate an LLM provider and apply logging wrapper when enabled."""
    llm: LLMProvider = hydra.utils.instantiate(llm_cfg)
    _attach_stream_callback(llm)
    if enable_logging:
        return LoggingLLMProvider(inner_provider=llm)
    return llm


def _attach_stream_callback(llm: LLMProvider) -> None:
    """
    Attach a UI stream callback when a provider supports callback injection.

    This keeps UI dependencies in the composition root rather than provider
    implementation modules.
    """
    callback_setter = getattr(llm, "set_response_callback", None)
    if callable(callback_setter):
        callback_setter(RichConsoleResponseCallback())
        return

    inner_provider = getattr(llm, "inner_provider", None)
    if isinstance(inner_provider, LLMProvider):
        _attach_stream_callback(inner_provider)


def _resolve_stage_llm(
    override_cfg: dict | None,
    *,
    shared_llm: LLMProvider,
    enable_logging: bool,
) -> tuple[LLMProvider, str]:
    """Resolve stage LLM provider, falling back to shared provider when unset."""
    if _has_target_config(override_cfg):
        return (
            _instantiate_llm_provider(override_cfg, enable_logging=enable_logging),
            "override",
        )
    return shared_llm, "shared"


def _should_defer_speaker_samples(audio_cfg_dict: dict[str, Any]) -> bool:
    """Return whether speaker-sample generation should move off the summary critical path."""
    speaker_samples_cfg = _to_plain_dict(audio_cfg_dict.get("speaker_samples", {}))
    return bool(speaker_samples_cfg.get("enabled", True)) and bool(
        speaker_samples_cfg.get("defer_until_voice_clone", False)
    )


def _build_speaker_sample_preparer(
    *,
    stage_start: str,
    audio_cfg_dict: dict[str, Any],
    project_root: Path,
    transcript_path: str,
    eta_strategy: StageEtaStrategy | None = None,
    eta_update_interval_seconds: float = 10.0,
    eta_progress_smoothing: float = 0.25,
    eta_overrun_factor: float = 1.15,
    eta_headroom_seconds: float = 1.0,
) -> Callable[[Transcript], Transcript]:
    """Build a callback that can prepare speaker samples immediately before voice cloning."""

    def _prepare(transcript: Transcript) -> Transcript:
        return _run_post_transcript_speaker_sample_step(
            stage_start=stage_start,
            audio_cfg_dict=audio_cfg_dict,
            project_root=project_root,
            transcript_path=transcript_path,
            transcript=transcript,
            eta_strategy=eta_strategy,
            eta_update_interval_seconds=eta_update_interval_seconds,
            eta_progress_smoothing=eta_progress_smoothing,
            eta_overrun_factor=eta_overrun_factor,
            eta_headroom_seconds=eta_headroom_seconds,
        )

    return _prepare


def _run_post_transcript_speaker_sample_step(
    *,
    stage_start: str,
    audio_cfg_dict: dict[str, Any],
    project_root: Path,
    transcript_path: str,
    transcript: Transcript,
    eta_strategy: StageEtaStrategy | None = None,
    eta_update_interval_seconds: float = 10.0,
    eta_progress_smoothing: float = 0.25,
    eta_overrun_factor: float = 1.15,
    eta_headroom_seconds: float = 1.0,
) -> Transcript:
    """
    Generate per-speaker voice samples after transcript availability.

    Args:
        stage_start: Active start stage from pipeline plan.
        audio_cfg_dict: Resolved audio config mapping.
        project_root: Repository root for relative path resolution.
        transcript_path: Path to transcript JSON to update.
        transcript: Loaded transcript domain DTO.

    Returns:
        Transcript with updated speaker-sample metadata when generation runs;
        otherwise the original transcript.
    """
    if stage_start not in {"stage-1", "stage-2"}:
        return transcript

    speaker_samples_cfg = _to_plain_dict(audio_cfg_dict.get("speaker_samples", {}))
    if not bool(speaker_samples_cfg.get("enabled", True)):
        return transcript

    work_dir = resolve_path(
        str(audio_cfg_dict.get("work_dir", "artifacts/audio_stage")),
        base_dir=project_root,
    )
    output_dir_name = str(speaker_samples_cfg.get("output_dir_name", "speaker_samples"))
    if not output_dir_name.strip():
        raise ValueError("audio.speaker_samples.output_dir_name must be non-empty.")
    samples_output_dir = resolve_path(output_dir_name, base_dir=work_dir)
    # Voice sample extraction is constrained to the vocals stem to maximize
    # speaker separation quality and avoid background bleed from full-mix audio.
    source_audio_path = resolve_sample_source_audio_path(
        source_mode="vocals",
        transcript_metadata=transcript.metadata,
        configured_audio_path=str(audio_cfg_dict.get("audio_path", "")),
        base_dir=project_root,
    )
    sample_generator = build_speaker_sample_generator(audio_cfg_dict)
    expected_sample_count = len(
        {
            segment.speaker
            for segment in transcript.segments
            if isinstance(segment.speaker, str) and segment.speaker.strip()
        }
    )
    eta_tracker: DynamicEtaTracker | None = None
    if (
        expected_sample_count > 0
        and isinstance(eta_strategy, UnitStageEtaStrategy)
    ):
        estimated_total_seconds = eta_strategy.estimate_unit_stage_total_seconds(
            stage="speaker_samples",
            total_units=expected_sample_count,
        )
        if estimated_total_seconds is not None:
            eta_tracker = DynamicEtaTracker(
                initial_total_seconds=estimated_total_seconds,
                progress_smoothing=eta_progress_smoothing,
                overrun_factor=eta_overrun_factor,
                headroom_seconds=eta_headroom_seconds,
            )

    if eta_tracker is None:
        event_bus.publish(
            "system_message",
            (
                "Generating speaker voice samples from "
                f"{source_audio_path} into {samples_output_dir}"
            ),
        )
    else:
        event_bus.publish(
            "system_message",
            (
                "Generating speaker voice samples from "
                f"{source_audio_path} into {samples_output_dir} "
                f"(estimated time left {format_eta_seconds(eta_tracker.initial_total_seconds)})"
            ),
        )

    started_at = time.monotonic()
    tracker_lock = threading.Lock()
    stop_event = threading.Event()
    ticker_thread: threading.Thread | None = None

    progress_callback: StageProgressCallback | None = None
    if eta_tracker is not None:
        def _on_progress(update: StageProgressUpdate) -> None:
            elapsed_seconds = max(0.0, time.monotonic() - started_at)
            with tracker_lock:
                eta_tracker.observe_progress(
                    elapsed_seconds=elapsed_seconds,
                    update=update,
                )

        progress_callback = _on_progress

        if eta_update_interval_seconds > 0:
            def _ticker() -> None:
                while not stop_event.wait(eta_update_interval_seconds):
                    elapsed_seconds = max(0.0, time.monotonic() - started_at)
                    with tracker_lock:
                        remaining_seconds = eta_tracker.estimate_signed_remaining_seconds(
                            elapsed_seconds=elapsed_seconds
                        )
                    if remaining_seconds >= 0:
                        event_bus.publish(
                            "status_message",
                            (
                                "Speaker sample stage: estimated time left "
                                f"{format_eta_seconds(remaining_seconds)}"
                            ),
                            inline=True,
                        )
                        continue
                    event_bus.publish(
                        "status_message",
                        (
                            "Speaker sample stage: running longer than estimate by "
                            f"{format_eta_seconds(abs(remaining_seconds))}"
                        ),
                        inline=True,
                    )

            ticker_thread = threading.Thread(target=_ticker, daemon=True)
            ticker_thread.start()

    try:
        sample_result = sample_generator.generate(
            transcript_payload=transcript.to_payload(),
            source_audio_path=source_audio_path,
            output_dir=samples_output_dir,
            progress_callback=progress_callback,
        )
    finally:
        stop_event.set()
        if ticker_thread is not None:
            ticker_thread.join(timeout=0.2)

    elapsed_seconds = max(0.0, time.monotonic() - started_at)
    if (
        isinstance(eta_strategy, UnitStageEtaLearner)
        and len(sample_result.artifacts) > 0
    ):
        try:
            eta_strategy.observe_unit_stage_duration(
                stage="speaker_samples",
                total_units=len(sample_result.artifacts),
                elapsed_seconds=elapsed_seconds,
            )
        except Exception:
            event_bus.publish(
                "system_message",
                "Speaker sample stage: ETA learning update skipped.",
            )

    event_bus.publish(
        "status_message",
        (
            f"Generated {len(sample_result.artifacts)} speaker samples "
            f"at {sample_result.output_dir} in {format_eta_seconds(elapsed_seconds)}"
        ),
    )

    metadata = dict(transcript.metadata)
    metadata.update(
        {
            "speaker_samples_manifest_path": str(sample_result.manifest_path),
            "speaker_samples_dir": str(sample_result.output_dir),
            "speaker_sample_count": len(sample_result.artifacts),
            "speaker_samples_generated_at_utc": sample_result.generated_at_utc,
        }
    )
    updated_transcript = transcript.with_metadata(metadata)

    resolved_transcript_path = resolve_path(transcript_path, base_dir=project_root)
    write_transcript_atomic(
        cast(TranscriptPayload, updated_transcript.to_payload()),
        resolved_transcript_path,
    )
    event_bus.publish(
        "status_message",
        f"Speaker sample manifest written to {sample_result.manifest_path}",
    )
    return updated_transcript


def _wait_for_agent_servers(
    *,
    checker: AgentHealthChecker,
    servers: Sequence[tuple[str, int]],
    overall_timeout_seconds: float = 10.0,
    poll_interval_seconds: float = 0.2,
    request_timeout_seconds: float = 1.0,
) -> None:
    """Block until required A2A servers are healthy or terminate the process."""
    if not servers:
        return
    event_bus.publish("system_message", "Waiting for A2A servers to start...")
    if checker.wait_for_many(
        servers,
        overall_timeout_seconds=overall_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        request_timeout_seconds=request_timeout_seconds,
    ):
        return
    sys.exit(1)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the configured summarization pipeline."""
    logging_cfg = _to_plain_dict(cfg.get("logging", {}))
    configure_logger(logging_cfg)

    if not bool(logging_cfg.get("print_to_terminal", False)):
        _suppress_chatty_logger_propagation()

    project_root = Path(hydra.utils.get_original_cwd())
    audio_cfg_dict = _to_plain_dict(cfg.get("audio", {}))
    pipeline_cfg_dict = _to_plain_dict(cfg.get("pipeline", {}))
    stage_plan = build_pipeline_stage_plan(pipeline_cfg_dict, project_root=project_root)
    runtime_device = resolve_device(str(audio_cfg_dict.get("device", "auto")))
    work_dir = resolve_path(
        str(audio_cfg_dict.get("work_dir", "artifacts/audio_stage")),
        base_dir=project_root,
    )
    audio_orchestrator = build_audio_to_script_orchestrator(audio_cfg_dict)
    eta_profile_path = work_dir / audio_orchestrator.eta_profile_filename
    eta_profile_context = audio_orchestrator._build_eta_profile_context(device=runtime_device)
    if not stage_plan.run_audio_stage:
        audio_orchestrator._load_eta_profile(
            profile_path=eta_profile_path,
            context=eta_profile_context,
        )

    event_bus.publish(
        "system_message",
        (
            "Pipeline plan: "
            f"start_stage={stage_plan.start_stage}"
        ),
    )

    transcript_path = str(cfg.transcript_path)
    if stage_plan.run_audio_stage:
        audio_path_value = str(audio_cfg_dict.get("audio_path", "")).strip()
        if not audio_path_value:
            raise ValueError(
                "audio.audio_path is required when pipeline.start_stage=stage-1. "
                "To skip stage-1, set pipeline.start_stage=stage-2."
            )

        input_audio_path = resolve_path(audio_path_value, base_dir=project_root)
        output_transcript_path = resolve_path(
            str(
                audio_cfg_dict.get(
                    "output_transcript_path",
                    "artifacts/transcripts/latest.transcript.json",
                )
            ),
            base_dir=project_root,
        )
        audio_stage_output = audio_orchestrator.run(
            input_audio_path=input_audio_path,
            output_transcript_path=output_transcript_path,
            work_dir=work_dir,
            device=runtime_device,
            metadata_overrides={
                "separator_model": str(
                    audio_cfg_dict.get("separation", {}).get("model", "htdemucs")
                ),
                "transcriber_model": str(
                    audio_cfg_dict.get("asr", {}).get("model", "large-v3")
                ),
                "diarizer_backend": str(
                    audio_cfg_dict.get("diarization", {}).get("provider", "nemo")
                ),
            },
        )
        transcript_path = str(audio_stage_output.transcript_path)
    else:
        event_bus.publish(
            "system_message",
            f"Skipping audio stage (start_stage={stage_plan.start_stage}).",
        )

    resolved_transcript_path = resolve_path(transcript_path, base_dir=project_root)
    if not resolved_transcript_path.exists():
        if stage_plan.start_stage == "stage-3":
            raise ValueError(
                "transcript_path must point to an existing transcript JSON with "
                "metadata.speaker_samples_manifest_path when "
                "pipeline.start_stage=stage-3."
            )
        raise FileNotFoundError(f"Transcript file not found: {resolved_transcript_path}")

    transcript_path = str(resolved_transcript_path)
    event_bus.publish("system_message", f"Loading transcript from {transcript_path}...")
    transcript_payload = load_transcript(transcript_path)
    transcript = Transcript.from_mapping(transcript_payload)
    event_bus.publish(
        "system_message",
        f"Loaded {len(transcript.segments)} segments",
    )
    speaker_sample_preparer: Callable[[Transcript], Transcript] | None = None
    if _should_defer_speaker_samples(audio_cfg_dict):
        event_bus.publish(
            "system_message",
            "Speaker sample generation deferred until voice clone stage.",
        )
        speaker_sample_preparer = _build_speaker_sample_preparer(
            stage_start=stage_plan.start_stage,
            audio_cfg_dict=audio_cfg_dict,
            project_root=project_root,
            transcript_path=transcript_path,
            eta_strategy=audio_orchestrator.eta_strategy,
            eta_update_interval_seconds=audio_orchestrator.eta_update_interval_seconds,
            eta_progress_smoothing=audio_orchestrator.eta_progress_smoothing,
            eta_overrun_factor=audio_orchestrator.eta_overrun_factor,
            eta_headroom_seconds=audio_orchestrator.eta_headroom_seconds,
        )
    else:
        transcript = _run_post_transcript_speaker_sample_step(
            stage_start=stage_plan.start_stage,
            audio_cfg_dict=audio_cfg_dict,
            project_root=project_root,
            transcript_path=transcript_path,
            transcript=transcript,
            eta_strategy=audio_orchestrator.eta_strategy,
            eta_update_interval_seconds=audio_orchestrator.eta_update_interval_seconds,
            eta_progress_smoothing=audio_orchestrator.eta_progress_smoothing,
            eta_overrun_factor=audio_orchestrator.eta_overrun_factor,
            eta_headroom_seconds=audio_orchestrator.eta_headroom_seconds,
        )
    if stage_plan.run_summarizer_stage or stage_plan.run_critic_stage:
        stage_terminal_logging = bool(
            logging_cfg.get(
                "summarizer_critic_print_to_terminal",
                logging_cfg.get("print_to_terminal", False),
            )
        )
        current_terminal_logging = bool(logging_cfg.get("print_to_terminal", False))
        if stage_terminal_logging != current_terminal_logging:
            logging_cfg["print_to_terminal"] = stage_terminal_logging
            configure_logger(logging_cfg)
            if not stage_terminal_logging:
                _suppress_chatty_logger_propagation()

    voice_clone_orchestrator = build_voice_clone_orchestrator(
        audio_cfg_dict,
        project_root=project_root,
    )
    voice_clone_gpu_heartbeat: VoiceCloneGpuHeartbeatService | None = None
    if voice_clone_orchestrator is None:
        event_bus.publish("system_message", "Voice clone stage disabled.")
    else:
        event_bus.publish(
            "system_message",
            (
                "Voice clone stage enabled with output dir "
                f"{voice_clone_orchestrator.output_dir}"
            ),
        )
        voice_clone_cfg = _to_plain_dict(audio_cfg_dict.get("voice_clone", {}))
        heartbeat_cfg = parse_voice_clone_gpu_heartbeat_config(
            _to_plain_dict(voice_clone_cfg.get("gpu_heartbeat", {}))
        )
        provider_device = str(
            getattr(voice_clone_orchestrator.provider, "device", "")
        ).strip().lower()
        if not heartbeat_cfg.enabled:
            event_bus.publish(
                "system_message",
                "Voice clone stage: dedicated GPU heartbeat disabled by config.",
            )
        elif os.name != "nt":
            event_bus.publish(
                "system_message",
                "Voice clone stage: dedicated GPU heartbeat is Windows-only; skipping.",
            )
        elif provider_device != "cuda":
            event_bus.publish(
                "system_message",
                (
                    "Voice clone stage: dedicated GPU heartbeat skipped because "
                    f"voice clone provider device is '{provider_device or 'unknown'}'."
                ),
            )
        else:
            voice_clone_gpu_heartbeat = VoiceCloneGpuHeartbeatService(
                probe=WindowsNvidiaDedicatedGpuProbe(
                    command_timeout_seconds=heartbeat_cfg.command_timeout_seconds,
                ),
                emit_status=lambda message: event_bus.publish("status_message", message),
                emit_system=lambda message: event_bus.publish("system_message", message),
                interval_seconds=heartbeat_cfg.interval_seconds,
                threshold_ratio=heartbeat_cfg.dedicated_usage_threshold_ratio,
                top_other_processes=heartbeat_cfg.top_other_processes,
            )
            event_bus.publish(
                "system_message",
                (
                    "Voice clone stage: dedicated GPU heartbeat configured "
                    f"(interval={heartbeat_cfg.interval_seconds}s, "
                    f"threshold={heartbeat_cfg.dedicated_usage_threshold_ratio:.0%})."
                ),
            )
    event_bus.publish("system_message", "Instantiating LLM providers...")
    shared_agent_client = AgentClient(event_bus=event_bus)
    logging_enabled = bool(cfg.get("logging", {}).get("enabled", False))
    shared_llm = _instantiate_llm_provider(cfg.llm, enable_logging=logging_enabled)

    stage_llm_cfg = _to_plain_dict(cfg.get("stage_llm", {}))
    summarizer_llm, summarizer_llm_source = _resolve_stage_llm(
        stage_llm_cfg.get("summarizer"),
        shared_llm=shared_llm,
        enable_logging=logging_enabled,
    )
    critic_llm, critic_llm_source = _resolve_stage_llm(
        stage_llm_cfg.get("critic"),
        shared_llm=shared_llm,
        enable_logging=logging_enabled,
    )

    event_bus.publish(
        "system_message", f"Default LLM provider: {type(shared_llm).__name__}"
    )
    event_bus.publish(
        "system_message",
        (
            "Summarizer LLM provider: "
            f"{type(summarizer_llm).__name__} (source={summarizer_llm_source})"
        ),
    )
    event_bus.publish(
        "system_message",
        f"Critic LLM provider: {type(critic_llm).__name__} (source={critic_llm_source})",
    )

    is_embedding_enabled = "NoOpEmbeddingProvider" not in cfg.embedding.get(
        "_target_", ""
    )

    if is_embedding_enabled and cfg.embedding.get("device", "") == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                event_bus.publish(
                    "error_message",
                    "CUDA is requested but not available on this device.",
                )
                print(
                    "\n[WARNING] CUDA is not available on this device, but it was requested for embeddings."
                )
                print("Options:")
                print("  [1] Fallback to CPU (slow)")
                print("  [2] Disable embeddings completely (use NoOpProvider)")

                choice = ""
                while choice not in ["1", "2"]:
                    choice = input("Please select an option [1/2]: ").strip()

                if choice == "1":
                    cfg.embedding.device = "cpu"
                    event_bus.publish(
                        "system_message", "Falling back to CPU for embeddings."
                    )
                elif choice == "2":
                    cfg.embedding._target_ = (
                        "providers.null_provider.NoOpEmbeddingProvider"
                    )
                    is_embedding_enabled = False
                    event_bus.publish("system_message", "Embeddings disabled by user.")
        except ImportError:
            pass

    transcript_index = None
    if is_embedding_enabled and stage_plan.requires_retrieval_tools:
        event_bus.publish("system_message", "Instantiating Embedding provider...")
        embedding: EmbeddingProvider = hydra.utils.instantiate(cfg.embedding)
        event_bus.publish(
            "system_message", f"Embedding provider: {type(embedding).__name__}"
        )
        transcript_index = TranscriptIndex(embedding_provider=embedding)
    else:
        event_bus.publish(
            "system_message",
            "Embedding provider disabled or not required for selected pipeline plan.",
        )

    retrieval_port = cfg.ports.retrieval
    summarizer_port = cfg.ports.summarizer
    critic_port = cfg.ports.critic
    agents_cfg = _to_plain_dict(cfg.get("agents", {}))
    summarizer_cfg = _to_plain_dict(agents_cfg.get("summarizer", {}))
    critic_cfg = _to_plain_dict(agents_cfg.get("critic", {}))

    if transcript_index is not None:
        event_bus.publish(
            "system_message",
            f"Starting Info Retrieval A2A server on port {retrieval_port}...",
        )
        retrieval_app = _build_a2a_app(
            name="InfoRetrieval",
            description="Indexes transcript segments and retrieves relevant ones.",
            port=retrieval_port,
            executor=InfoRetrievalExecutor(transcript_index),
        )
        _run_server_in_thread("retrieval-a2a", retrieval_app, retrieval_port)
    else:
        event_bus.publish("system_message", "Info Retrieval Agent disabled.")

    if stage_plan.run_summarizer_stage:
        event_bus.publish(
            "system_message",
            f"Starting Summarizer A2A server on port {summarizer_port}...",
        )
        summarizer_app = _build_a2a_app(
            name="Summarizer",
            description="Produces abstractive summaries.",
            port=summarizer_port,
            executor=SummarizerExecutor(
                llm=summarizer_llm,
                retrieval_port=retrieval_port,
                max_tool_turns=int(summarizer_cfg.get("max_tool_turns", 3)),
                is_embedding_enabled=is_embedding_enabled,
                loop_guardrails=_to_plain_dict(summarizer_cfg.get("loop_guardrails")),
                agent_client=shared_agent_client,
                event_bus=event_bus,
            ),
        )
        _run_server_in_thread("summarizer-a2a", summarizer_app, summarizer_port)
    else:
        event_bus.publish("system_message", "Summarizer stage disabled by pipeline plan.")

    if stage_plan.run_critic_stage:
        event_bus.publish(
            "system_message",
            f"Starting Critic A2A server on port {critic_port}...",
        )
        critic_app = _build_a2a_app(
            name="Critic",
            description="Evaluates summaries.",
            port=critic_port,
            executor=CriticExecutor(
                llm=critic_llm,
                max_tool_turns=int(critic_cfg.get("max_tool_turns", 5)),
                retrieval_port=retrieval_port,
                is_embedding_enabled=is_embedding_enabled,
                agent_client=shared_agent_client,
                event_bus=event_bus,
            ),
        )
        _run_server_in_thread("critic-a2a", critic_app, critic_port)
    else:
        event_bus.publish("system_message", "Critic stage disabled by pipeline plan.")

    servers_to_check: list[tuple[str, int]] = []
    if transcript_index is not None:
        servers_to_check.append(("InfoRetrieval", retrieval_port))
    if stage_plan.run_summarizer_stage:
        servers_to_check.append(("Summarizer", summarizer_port))
    if stage_plan.run_critic_stage:
        servers_to_check.append(("Critic", critic_port))

    checker = AgentHealthChecker(max_retries=5, base_delay=1.0)
    _wait_for_agent_servers(
        checker=checker,
        servers=servers_to_check,
        overall_timeout_seconds=10.0,
        poll_interval_seconds=0.2,
        request_timeout_seconds=1.0,
    )

    event_bus.publish("system_message", "Starting orchestration loop...")
    orchestrator = Orchestrator(
        retrieval_port=retrieval_port,
        summarizer_port=summarizer_port,
        critic_port=critic_port,
        timeouts=dict(cfg.orchestrator.get("timeouts", {})),
        agent_client=shared_agent_client,
        event_bus=event_bus,
    )
    stage_orchestrator = StageOrchestrator(
        orchestrator=orchestrator,
        stage_plan=stage_plan,
        project_root=project_root,
        min_words=int(cfg.orchestrator.min_words),
        max_words=int(cfg.orchestrator.max_words),
        max_iterations=int(cfg.orchestrator.max_iterations),
        voice_clone_orchestrator=voice_clone_orchestrator,
        speaker_sample_preparer=speaker_sample_preparer,
        eta_strategy=audio_orchestrator.eta_strategy,
        eta_update_interval_seconds=audio_orchestrator.eta_update_interval_seconds,
        eta_progress_smoothing=audio_orchestrator.eta_progress_smoothing,
        eta_overrun_factor=audio_orchestrator.eta_overrun_factor,
        eta_headroom_seconds=audio_orchestrator.eta_headroom_seconds,
        voice_clone_gpu_heartbeat=voice_clone_gpu_heartbeat,
    )
    try:
        asyncio.run(
            stage_orchestrator.run(
                transcript=transcript,
                retrieval_enabled=transcript_index is not None,
            )
        )
    finally:
        audio_orchestrator._save_eta_profile(
            profile_path=eta_profile_path,
            context=eta_profile_context,
        )


if __name__ == "__main__":
    main()
