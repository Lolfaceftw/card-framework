import asyncio
import logging
import sys
import threading
import time

import hydra
import uvicorn
from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import InMemoryQueueManager
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from omegaconf import DictConfig

from agents.critic import CriticExecutor
from agents.retrieval import InfoRetrievalExecutor
from agents.summarizer import SummarizerExecutor
from agents.utils import load_transcript
from agents.health import AgentHealthChecker
from embeddings import TranscriptIndex
from llm_provider import EmbeddingProvider, LLMProvider
from logger_utils import configure_logger
from orchestrator import Orchestrator
from providers.logging_provider import LoggingLLMProvider
from events import event_bus


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
        agent_card=agent_card, http_handler=request_handler
    )
    return a2a_app.build()


def _run_server_in_thread(name: str, app, port: int) -> threading.Thread:
    """Run a uvicorn server in a daemon thread."""
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)

    def _serve():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve())

    t = threading.Thread(target=_serve, name=name, daemon=True)
    t.start()
    return t


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ── 1. Configure Logging ──
    configure_logger(cfg.logging)

    # Silence third-party loggers if terminal output is disabled
    if not cfg.logging.get("print_to_terminal", False):
        for logger_name in ["google", "google_genai", "httpx", "a2a", "uvicorn"]:
            logging.getLogger(logger_name).propagate = False

    # ── 2. Load transcript ──
    event_bus.publish(
        "system_message", f"Loading transcript from {cfg.transcript_path}..."
    )
    transcript = load_transcript(cfg.transcript_path)
    event_bus.publish(
        "system_message", f"Loaded {len(transcript.get('segments', []))} segments"
    )

    # ── 3. Instantiate providers ──
    event_bus.publish("system_message", "Instantiating LLM provider...")
    llm: LLMProvider = hydra.utils.instantiate(cfg.llm)

    if cfg.get("logging", {}).get("enabled", False):
        llm = LoggingLLMProvider(inner_provider=llm)

    event_bus.publish("system_message", f"LLM provider: {type(llm).__name__}")

    # Check CUDA and embedding config
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
            pass  # If torch isn't installed, evaluating the provider will fail later anyway.

    transcript_index = None
    if is_embedding_enabled:
        event_bus.publish("system_message", "Instantiating Embedding provider...")
        embedding: EmbeddingProvider = hydra.utils.instantiate(cfg.embedding)
        event_bus.publish(
            "system_message", f"Embedding provider: {type(embedding).__name__}"
        )

        # ── 3. Build the transcript index ──
        transcript_index = TranscriptIndex(embedding_provider=embedding)

    # ── 4. Start A2A servers ──
    retrieval_port = cfg.ports.retrieval
    summarizer_port = cfg.ports.summarizer
    critic_port = cfg.ports.critic

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

    event_bus.publish(
        "system_message", f"Starting Summarizer A2A server on port {summarizer_port}..."
    )
    summarizer_app = _build_a2a_app(
        name="Summarizer",
        description="Produces abstractive summaries.",
        port=summarizer_port,
        executor=SummarizerExecutor(
            llm=llm,
            retrieval_port=retrieval_port,
            max_tool_turns=cfg.get("agents", {})
            .get("summarizer", {})
            .get("max_tool_turns", 3),
            is_embedding_enabled=is_embedding_enabled,
        ),
    )
    _run_server_in_thread("summarizer-a2a", summarizer_app, summarizer_port)

    event_bus.publish(
        "system_message", f"Starting Critic A2A server on port {critic_port}..."
    )
    critic_app = _build_a2a_app(
        name="Critic",
        description="Evaluates summaries.",
        port=critic_port,
        executor=CriticExecutor(
            llm=llm,
            max_tool_turns=cfg.get("agents", {})
            .get("critic", {})
            .get("max_tool_turns", 5),
            retrieval_port=retrieval_port,
            is_embedding_enabled=is_embedding_enabled,
        ),
    )
    _run_server_in_thread("critic-a2a", critic_app, critic_port)

    event_bus.publish("system_message", "Waiting for A2A servers to start...")
    time.sleep(2)

    # ── 5. Verify servers are up ──
    servers_to_check = [("Summarizer", summarizer_port), ("Critic", critic_port)]
    if transcript_index is not None:
        servers_to_check.insert(0, ("InfoRetrieval", retrieval_port))

    checker = AgentHealthChecker(max_retries=5, base_delay=1.0)

    for name, port in servers_to_check:
        is_up = checker.check(name, port)
        if not is_up:
            event_bus.publish(
                "error_message",
                f"[ERR] {name} server failed to start after multiple attempts.",
            )
            sys.exit(1)

    # ── 6. Run the orchestrator ──
    event_bus.publish("system_message", "Starting orchestration loop...")
    orchestrator = Orchestrator(
        retrieval_port=retrieval_port,
        summarizer_port=summarizer_port,
        critic_port=critic_port,
        timeouts=dict(cfg.orchestrator.get("timeouts", {})),
    )

    async def run():
        full_text = ""
        if is_embedding_enabled:
            await orchestrator.index_transcript(transcript)
        else:
            segments = transcript.get("segments", [])
            full_text = "".join(
                f"[{s.get('speaker', 'UNKNOWN')}]: {s.get('text', '')}\n"
                for s in segments
            )

        result = await orchestrator.run_loop(
            min_words=cfg.orchestrator.min_words,
            max_words=cfg.orchestrator.max_words,
            max_iterations=cfg.orchestrator.max_iterations,
            full_transcript_text=full_text,
        )
        if result:
            event_bus.publish(
                "agent_message", "Orchestrator", f"Final Summary:\n\n{result}"
            )

    asyncio.run(run())


if __name__ == "__main__":
    main()
