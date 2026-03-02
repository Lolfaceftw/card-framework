import asyncio
import logging
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
)
from audio_pipeline.contracts import TranscriptPayload
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


def _run_post_transcript_speaker_sample_step(
    *,
    stage_start: str,
    audio_cfg_dict: dict[str, Any],
    project_root: Path,
    transcript_path: str,
    transcript: Transcript,
) -> None:
    """
    Generate per-speaker voice samples after transcript availability.

    Args:
        stage_start: Active start stage from pipeline plan.
        audio_cfg_dict: Resolved audio config mapping.
        project_root: Repository root for relative path resolution.
        transcript_path: Path to transcript JSON to update.
        transcript: Loaded transcript domain DTO.
    """
    if stage_start not in {"audio", "transcript"}:
        return

    speaker_samples_cfg = _to_plain_dict(audio_cfg_dict.get("speaker_samples", {}))
    if not bool(speaker_samples_cfg.get("enabled", True)):
        return

    work_dir = resolve_path(
        str(audio_cfg_dict.get("work_dir", "artifacts/audio_stage")),
        base_dir=project_root,
    )
    output_dir_name = str(speaker_samples_cfg.get("output_dir_name", "speaker_samples"))
    if not output_dir_name.strip():
        raise ValueError("audio.speaker_samples.output_dir_name must be non-empty.")
    samples_output_dir = resolve_path(output_dir_name, base_dir=work_dir)
    source_audio_path = resolve_sample_source_audio_path(
        source_mode=str(speaker_samples_cfg.get("source_audio", "vocals")),
        transcript_metadata=transcript.metadata,
        configured_audio_path=str(audio_cfg_dict.get("audio_path", "")),
        base_dir=project_root,
    )
    sample_generator = build_speaker_sample_generator(audio_cfg_dict)
    event_bus.publish(
        "system_message",
        (
            "Generating speaker voice samples from "
            f"{source_audio_path} into {samples_output_dir}"
        ),
    )
    sample_result = sample_generator.generate(
        transcript_payload=transcript.to_payload(),
        source_audio_path=source_audio_path,
        output_dir=samples_output_dir,
    )
    event_bus.publish(
        "status_message",
        (
            f"Generated {len(sample_result.artifacts)} speaker samples "
            f"at {sample_result.output_dir}"
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


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run the configured summarization pipeline."""
    configure_logger(cfg.logging)

    if not cfg.logging.get("print_to_terminal", False):
        for logger_name in ["google", "google_genai", "httpx", "a2a", "uvicorn"]:
            logging.getLogger(logger_name).propagate = False

    project_root = Path(hydra.utils.get_original_cwd())
    audio_cfg_dict = _to_plain_dict(cfg.get("audio", {}))
    pipeline_cfg_dict = _to_plain_dict(cfg.get("pipeline", {}))
    stage_plan = build_pipeline_stage_plan(pipeline_cfg_dict, project_root=project_root)

    event_bus.publish(
        "system_message",
        (
            "Pipeline plan: "
            f"start_stage={stage_plan.start_stage}, stop_stage={stage_plan.stop_stage}"
        ),
    )

    transcript_path = str(cfg.transcript_path)
    if stage_plan.run_audio_stage:
        audio_path_value = str(audio_cfg_dict.get("audio_path", "")).strip()
        if not audio_path_value:
            raise ValueError(
                "audio.audio_path is required when pipeline.start_stage=audio. "
                "To skip stage 1, set pipeline.start_stage=transcript."
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
        work_dir = resolve_path(
            str(audio_cfg_dict.get("work_dir", "artifacts/audio_stage")),
            base_dir=project_root,
        )
        runtime_device = resolve_device(str(audio_cfg_dict.get("device", "auto")))
        audio_orchestrator = build_audio_to_script_orchestrator(audio_cfg_dict)
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

    event_bus.publish("system_message", f"Loading transcript from {transcript_path}...")
    transcript_payload = load_transcript(transcript_path)
    transcript = Transcript.from_mapping(transcript_payload)
    event_bus.publish(
        "system_message",
        f"Loaded {len(transcript.segments)} segments",
    )
    _run_post_transcript_speaker_sample_step(
        stage_start=stage_plan.start_stage,
        audio_cfg_dict=audio_cfg_dict,
        project_root=project_root,
        transcript_path=transcript_path,
        transcript=transcript,
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

    event_bus.publish("system_message", "Waiting for A2A servers to start...")
    time.sleep(2)

    servers_to_check: list[tuple[str, int]] = []
    if transcript_index is not None:
        servers_to_check.append(("InfoRetrieval", retrieval_port))
    if stage_plan.run_summarizer_stage:
        servers_to_check.append(("Summarizer", summarizer_port))
    if stage_plan.run_critic_stage:
        servers_to_check.append(("Critic", critic_port))

    checker = AgentHealthChecker(max_retries=5, base_delay=1.0)
    for name, port in servers_to_check:
        if not checker.check(name, port):
            event_bus.publish(
                "error_message",
                f"[ERR] {name} server failed to start after multiple attempts.",
            )
            sys.exit(1)

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
    )
    asyncio.run(
        stage_orchestrator.run(
            transcript=transcript,
            retrieval_enabled=transcript_index is not None,
        )
    )


if __name__ == "__main__":
    main()
