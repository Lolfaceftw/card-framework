"""Execution engine for the duration-first summarizer-critic benchmark."""

from __future__ import annotations

from collections.abc import Mapping
import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf
import uvicorn

from card_framework.agents.client import AgentClient
from card_framework.agents.critic import CriticExecutor
from card_framework.agents.health import AgentHealthChecker
from card_framework.agents.retrieval import InfoRetrievalExecutor
from card_framework.agents.summarizer import SummarizerExecutor
from card_framework.audio_pipeline.calibration import (
    VoiceCloneCalibration,
    resolve_emo_preset_catalog,
)
from card_framework.benchmark.artifacts import git_info, utc_now_iso
from card_framework.benchmark.instrumentation import EventCapture
from card_framework.benchmark.matrix import (
    MatrixConfigError,
    build_embedding_profiles,
    load_provider_profiles,
    missing_required_env,
    resolve_provider_config,
)
from card_framework.benchmark.reference_free.alignscore_runner import (
    AlignScoreRunner,
    AlignScoreRunnerConfig,
)
from card_framework.benchmark.reference_free.contracts import (
    ReferenceFreeContractError,
    load_judge_rubric,
)
from card_framework.benchmark.reference_free.judge_runner import (
    LLMJudgeRunner,
    LLMJudgeRunnerConfig,
)
from card_framework.benchmark.reference_free.pipeline import (
    ReferenceFreePipeline,
    ReferenceFreePipelineConfig,
)
from card_framework.benchmark.summarizer_critic.sampling import (
    select_benchmark_samples,
)
from card_framework.benchmark.summarizer_critic.config import (
    BenchmarkExecutionReport,
    BenchmarkSummary,
    DEFAULT_DATASET_MANIFEST_PATH,
    DEFAULT_REFERENCE_WPM,
    DEFAULT_RUNS_ROOT,
    PreparedBenchmarkSample,
    SampleBenchmarkResult,
    SampleMetricBundle,
)
from card_framework.benchmark.summarizer_critic.dataset import (
    DatasetPreparationError,
    load_prepared_dataset_manifest,
    prepare_official_qmsum_benchmark,
)
from card_framework.benchmark.summarizer_critic.metrics import (
    combine_factualness,
    combine_naturalness,
    compute_composite_score,
    score_duration_control,
    score_speaker_attribution,
)
from card_framework.benchmark.summarizer_critic.reporting import (
    write_execution_artifacts,
)
from card_framework.benchmark.datasets import load_transcript
from card_framework.benchmark.types import (
    BenchmarkSample,
    EmbeddingProfile,
    ProviderProfile,
)
from card_framework.cli.main import _build_a2a_app
from card_framework.providers.logging_provider import LoggingLLMProvider
from card_framework.providers.response_callbacks import RichConsoleResponseCallback
from card_framework.retrieval.embeddings import TranscriptIndex
from card_framework.runtime.loop_orchestrator import Orchestrator
from card_framework.shared.events import event_bus
from card_framework.shared.config_loading import load_runtime_config
from card_framework.shared.llm_provider import EmbeddingProvider, LLMProvider
from card_framework.shared.logger_utils import configure_logger
from card_framework.shared.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_JUDGE_RUBRIC_PATH,
    DEFAULT_PROVIDER_PROFILES_PATH,
    REPO_ROOT,
)
from card_framework.shared.summary_xml import DEFAULT_EMO_PRESET


class BenchmarkExecutionError(RuntimeError):
    """Raise when the summarizer-critic benchmark cannot proceed."""


class ManagedServer:
    """Uvicorn server and backing thread handle for managed lifecycle."""

    def __init__(
        self, *, name: str, port: int, server: uvicorn.Server, thread: threading.Thread
    ) -> None:
        self.name = name
        self.port = port
        self.server = server
        self.thread = thread


_RESERVED_PORT_SOCKETS: dict[int, socket.socket] = {}
_RESERVED_PORTS_LOCK = threading.Lock()


def _reserve_port() -> int:
    handle = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    handle.bind(("127.0.0.1", 0))
    handle.listen(2048)
    port = int(handle.getsockname()[1])
    with _RESERVED_PORTS_LOCK:
        _RESERVED_PORT_SOCKETS[port] = handle
    return port


def _consume_reserved_port_socket(port: int) -> socket.socket | None:
    """Return and remove a reserved socket for the requested port."""
    with _RESERVED_PORTS_LOCK:
        return _RESERVED_PORT_SOCKETS.pop(port, None)


def _release_reserved_port(port: int) -> None:
    """Close any still-reserved socket for the requested port."""
    handle = _consume_reserved_port_socket(port)
    if handle is None:
        return
    try:
        handle.close()
    except OSError:
        return


def _resolve_requested_port(name: str, value: int | None) -> int:
    """Return an explicit port override or reserve a free one."""
    if value is None:
        return _reserve_port()

    port = int(value)
    if not 1 <= port <= 65535:
        raise BenchmarkExecutionError(
            f"{name} port must be between 1 and 65535; received {port}."
        )
    return port


def _resolve_benchmark_ports(args: argparse.Namespace) -> tuple[int, int, int]:
    """Resolve distinct retrieval, summarizer, and critic ports for one run."""
    retrieval_port = _resolve_requested_port("retrieval", args.retrieval_port)
    summarizer_port = _resolve_requested_port("summarizer", args.summarizer_port)
    critic_port = _resolve_requested_port("critic", args.critic_port)
    resolved_ports = {
        "retrieval": retrieval_port,
        "summarizer": summarizer_port,
        "critic": critic_port,
    }
    if len(set(resolved_ports.values())) != len(resolved_ports):
        raise BenchmarkExecutionError(
            "Retrieval, summarizer, and critic ports must be distinct for one run."
        )
    return retrieval_port, summarizer_port, critic_port


def _run_server_in_thread(name: str, app: Any, port: int) -> ManagedServer:
    reserved_socket = _consume_reserved_port_socket(port)
    config = uvicorn.Config(
        app, host="127.0.0.1", port=port, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)

    def _serve() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if reserved_socket is None:
                loop.run_until_complete(server.serve())
            else:
                loop.run_until_complete(server.serve(sockets=[reserved_socket]))
        finally:
            if reserved_socket is not None:
                try:
                    reserved_socket.close()
                except OSError:
                    pass

    thread = threading.Thread(target=_serve, name=name, daemon=True)
    thread.start()
    return ManagedServer(name=name, port=port, server=server, thread=thread)


def _stop_server(server: ManagedServer) -> None:
    server.server.should_exit = True
    server.thread.join(timeout=10.0)


def _load_base_config(config_path: Path) -> DictConfig:
    if not config_path.exists():
        raise BenchmarkExecutionError(f"Config path does not exist: {config_path}")
    return load_runtime_config(config_path)


def _resolve_orchestrator_timeouts(
    base_cfg: DictConfig,
    args: argparse.Namespace,
) -> dict[str, float | None]:
    """Merge benchmark CLI timeout overrides onto the base orchestrator config."""
    raw_timeouts = OmegaConf.to_container(
        base_cfg.orchestrator.get("timeouts", {}),
        resolve=True,
    )
    timeout_items = raw_timeouts if isinstance(raw_timeouts, Mapping) else {}
    merged: dict[str, float | None] = {
        str(key): float(value)
        for key, value in timeout_items.items()
        if value is not None
    }
    overrides = {
        "summarizer": float(args.summarizer_timeout_seconds),
        "critic": float(args.critic_timeout_seconds),
        "retrieval": float(args.retrieval_timeout_seconds),
        "index": float(args.index_timeout_seconds),
    }
    for key, value in overrides.items():
        if value > 0:
            merged[key] = value
    return merged


def _make_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_pid{os.getpid()}"


def _format_eta(seconds: float | None) -> str:
    """Format an ETA in a compact human-readable form."""
    if seconds is None or not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    rounded_seconds = int(round(seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_snapshot(
    *,
    completed_samples: int,
    total_samples: int,
    elapsed_seconds: float,
) -> tuple[int, str, str]:
    """Return remaining count, rolling throughput, and ETA strings."""
    remaining_samples = max(total_samples - completed_samples, 0)
    if completed_samples <= 0 or elapsed_seconds <= 0:
        average_examples_per_minute = None
    else:
        average_examples_per_minute = completed_samples / (elapsed_seconds / 60.0)
    average_text = (
        f"{average_examples_per_minute:.2f} ex/min"
        if average_examples_per_minute is not None
        else "n/a"
    )
    if remaining_samples <= 0:
        eta_text = "0s"
    elif average_examples_per_minute is None or average_examples_per_minute <= 0:
        eta_text = "n/a"
    else:
        eta_seconds = (remaining_samples / average_examples_per_minute) * 60.0
        eta_text = _format_eta(eta_seconds)
    return remaining_samples, average_text, eta_text


def _select_samples(
    samples: tuple[PreparedBenchmarkSample, ...],
    *,
    max_samples: int,
    sample_seed: int | None,
) -> list[PreparedBenchmarkSample]:
    """Select benchmark samples in manifest or seeded-random order."""
    return select_benchmark_samples(
        samples,
        max_samples=max_samples,
        sample_seed=sample_seed,
        sample_id_getter=lambda sample: sample.sample_id,
    )


def _format_transcript(transcript: dict[str, Any]) -> str:
    raw_segments = transcript.get("segments", [])
    if not isinstance(raw_segments, list):
        return ""
    return "".join(
        f"[{segment.get('speaker', 'UNKNOWN')}]: {segment.get('text', '')}\n"
        for segment in raw_segments
        if isinstance(segment, dict)
    )


def _find_provider_profile(
    profiles: list[ProviderProfile], provider_id: str
) -> ProviderProfile:
    for profile in profiles:
        if profile.provider_id == provider_id:
            return profile
    raise BenchmarkExecutionError(f"Unknown provider profile: {provider_id}")


def _resolve_embedding_profile(
    base_cfg: DictConfig, embedding_mode: str
) -> EmbeddingProfile:
    profiles = build_embedding_profiles(base_cfg)
    try:
        return profiles[embedding_mode]
    except KeyError as exc:
        raise BenchmarkExecutionError(
            f"Unknown embedding mode: {embedding_mode}"
        ) from exc


def _configure_benchmark_logging(
    base_cfg: DictConfig,
    *,
    log_file: str | None = None,
    print_to_terminal: bool | None = None,
) -> None:
    """Enable benchmark logging with optional file and terminal overrides."""
    logging_cfg = base_cfg.get("logging")
    if not isinstance(logging_cfg, DictConfig):
        return
    if print_to_terminal is None:
        print_to_terminal = True
    logging_cfg["print_to_terminal"] = bool(print_to_terminal)
    logging_cfg["summarizer_critic_print_to_terminal"] = bool(print_to_terminal)
    if log_file is not None:
        logging_cfg["log_file"] = log_file
    configure_logger(logging_cfg)


def _attach_stream_callback(llm: LLMProvider) -> None:
    """Attach live stdout streaming when the provider supports callbacks."""
    callback_setter = getattr(llm, "set_response_callback", None)
    if callable(callback_setter):
        callback_setter(RichConsoleResponseCallback())
        return

    inner_provider = getattr(llm, "inner_provider", None)
    if isinstance(inner_provider, LLMProvider):
        _attach_stream_callback(inner_provider)


def _coerce_mapping(value: Any) -> dict[str, Any]:
    """Coerce OmegaConf/plain mapping-like values into a plain dictionary."""
    if isinstance(value, Mapping):
        return {str(key): raw_value for key, raw_value in value.items()}
    return {}


def _build_stub_calibration(
    *, run_root: Path, audio_cfg: dict[str, Any]
) -> VoiceCloneCalibration:
    dummy_manifest_path = run_root / "stub_speaker_samples_manifest.json"
    dummy_manifest_path.write_text(
        json.dumps({"samples": []}, indent=2), encoding="utf-8"
    )
    emo_presets = resolve_emo_preset_catalog(audio_cfg)
    speaker_preset_wpm = {"": {name: DEFAULT_REFERENCE_WPM for name in emo_presets}}
    preset_default_wpm = {name: DEFAULT_REFERENCE_WPM for name in emo_presets}
    return VoiceCloneCalibration(
        artifact_path=run_root / "stub_calibration.json",
        generated_at_utc=utc_now_iso(),
        speaker_samples_manifest_path=dummy_manifest_path,
        preset_emo_texts=emo_presets,
        calibration_phrases=("benchmark duration baseline",),
        speaker_preset_wpm=speaker_preset_wpm,
        preset_default_wpm=preset_default_wpm,
    )


async def _execute_sample(
    *,
    sample: PreparedBenchmarkSample,
    orchestrator: Orchestrator,
    event_capture: EventCapture,
    retrieval_enabled: bool,
    duration_tolerance_ratio: float,
    max_iterations: int,
    reference_free_pipeline: ReferenceFreePipeline,
    calibration: VoiceCloneCalibration,
) -> SampleBenchmarkResult:
    sample_started = time.perf_counter()
    retrieval_before = event_capture.count("retrieval_stats")
    tool_before = event_capture.count("tool_invocation")
    try:
        transcript = load_transcript(
            BenchmarkSample(
                sample_id=sample.sample_id,
                dataset=sample.dataset,
                transcript_path=sample.transcript_path,
                metadata=sample.metadata,
            )
        )
        source_text = _format_transcript(transcript)
        full_transcript_text = source_text if not retrieval_enabled else ""
        if retrieval_enabled:
            await orchestrator.index_transcript(transcript)
        diagnostics = await orchestrator.run_loop_with_diagnostics(
            target_seconds=sample.target_seconds,
            duration_tolerance_ratio=duration_tolerance_ratio,
            max_iterations=max_iterations,
            full_transcript_text=full_transcript_text,
        )
        candidate_summary = str(diagnostics.get("draft", "")).strip()
        candidate_seconds = float(
            diagnostics.get("final_estimated_seconds", 0.0) or 0.0
        )
        if candidate_seconds <= 0 and candidate_summary:
            try:
                from card_framework.shared.summary_xml import parse_summary_xml

                candidate_seconds = calibration.estimate_turns_seconds(
                    parse_summary_xml(candidate_summary)
                )
            except Exception:
                stripped_summary = re.sub(r"<[^>]+>", " ", candidate_summary)
                candidate_seconds = calibration.estimate_text_seconds(
                    speaker="",
                    emo_preset=DEFAULT_EMO_PRESET,
                    text=stripped_summary,
                )

        reference_free_result = None
        if candidate_summary:
            reference_free_result = reference_free_pipeline.evaluate_sample(
                source_text=source_text,
                summary_text=candidate_summary,
            )
        duration_control = score_duration_control(
            target_seconds=sample.target_seconds,
            candidate_seconds=candidate_seconds,
            tolerance_ratio=duration_tolerance_ratio,
        )
        factualness = combine_factualness(
            alignscore=(
                reference_free_result.alignscore if reference_free_result else None
            ),
            judge_scores=(
                reference_free_result.judge_scores if reference_free_result else None
            ),
        )
        naturalness = combine_naturalness(
            reference_free_result.judge_scores if reference_free_result else None,
            summary_xml=candidate_summary,
        )
        speaker_attribution = score_speaker_attribution(
            candidate_summary,
            transcript,
            reference_summary=sample.reference_summary,
        )
        metrics = SampleMetricBundle(
            duration_control=duration_control,
            factualness=factualness,
            naturalness=naturalness,
            speaker_attribution=speaker_attribution,
            composite=compute_composite_score(
                duration_control=duration_control,
                factualness=factualness,
                naturalness=naturalness,
                speaker_attribution=speaker_attribution,
            ),
        )
        duration_seconds = round(time.perf_counter() - sample_started, 3)
        retrieval_after = event_capture.count("retrieval_stats")
        tool_after = event_capture.count("tool_invocation")
        return SampleBenchmarkResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            meeting_id=sample.meeting_id,
            status="completed" if candidate_summary else "failed",
            target_seconds=sample.target_seconds,
            candidate_seconds=round(candidate_seconds, 3),
            converged=bool(diagnostics.get("converged", False)),
            iterations_run=int(diagnostics.get("iterations_run", 0)),
            final_status=str(diagnostics.get("final_status", "unknown")),
            duration_seconds=duration_seconds,
            retrieval_events=max(0, retrieval_after - retrieval_before),
            tool_invocations=max(0, tool_after - tool_before),
            metrics=metrics,
            alignscore=(
                reference_free_result.alignscore if reference_free_result else None
            ),
            alignscore_backend=(
                reference_free_result.alignscore_backend
                if reference_free_result
                else None
            ),
            judge_scores=(
                reference_free_result.judge_scores if reference_free_result else None
            ),
            reference_free_status=(
                reference_free_result.status if reference_free_result else "not_scored"
            ),
            reference_free_error=(
                reference_free_result.error_message if reference_free_result else None
            ),
        )
    except Exception as exc:
        duration_seconds = round(time.perf_counter() - sample_started, 3)
        retrieval_after = event_capture.count("retrieval_stats")
        tool_after = event_capture.count("tool_invocation")
        return SampleBenchmarkResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            meeting_id=sample.meeting_id,
            status="failed",
            target_seconds=sample.target_seconds,
            candidate_seconds=0.0,
            converged=False,
            iterations_run=0,
            final_status="execution_error",
            duration_seconds=duration_seconds,
            retrieval_events=max(0, retrieval_after - retrieval_before),
            tool_invocations=max(0, tool_after - tool_before),
            metrics=SampleMetricBundle(0.0, 0.0, 0.0, 0.0, 0.0),
            error_message=str(exc),
        )


async def _run_benchmark_async(args: argparse.Namespace) -> BenchmarkExecutionReport:
    dataset_manifest_path = Path(args.dataset_manifest).resolve()
    if args.prepare_if_missing and not dataset_manifest_path.exists():
        prepared = prepare_official_qmsum_benchmark(
            qmsum_repo_dir=Path(args.qmsum_repo_dir).resolve(),
            prepared_root=dataset_manifest_path.parent,
            reference_wpm=float(args.reference_wpm),
            force_download=bool(args.force_download_dataset),
        )
        dataset_manifest = prepared
    else:
        dataset_manifest = load_prepared_dataset_manifest(dataset_manifest_path)

    run_id = _make_run_id()
    run_root = Path(args.output_root).resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_base_config(Path(args.config).resolve())
    _configure_benchmark_logging(base_cfg)
    profiles = load_provider_profiles(Path(args.provider_profiles).resolve())
    summarizer_profile = _find_provider_profile(profiles, args.summarizer_provider)
    critic_profile = _find_provider_profile(
        profiles,
        args.critic_provider or args.summarizer_provider,
    )
    judge_profile = _find_provider_profile(
        profiles,
        args.judge_provider or args.critic_provider or args.summarizer_provider,
    )
    for profile in (summarizer_profile, critic_profile, judge_profile):
        missing_env = missing_required_env(profile)
        if missing_env:
            raise BenchmarkExecutionError(
                f"Provider '{profile.provider_id}' is missing required environment variables: {missing_env}"
            )

    embedding_profile = _resolve_embedding_profile(base_cfg, args.embedding_mode)
    logging_enabled = bool(base_cfg.get("logging", {}).get("enabled", False))
    calibration = _build_stub_calibration(
        run_root=run_root,
        audio_cfg=_coerce_mapping(OmegaConf.to_container(base_cfg.audio, resolve=True)),
    )
    llm_summarizer: LLMProvider = hydra.utils.instantiate(
        OmegaConf.create(resolve_provider_config(summarizer_profile))
    )
    llm_critic: LLMProvider = hydra.utils.instantiate(
        OmegaConf.create(resolve_provider_config(critic_profile))
    )
    llm_judge: LLMProvider = hydra.utils.instantiate(
        OmegaConf.create(resolve_provider_config(judge_profile))
    )
    _attach_stream_callback(llm_summarizer)
    _attach_stream_callback(llm_critic)
    _attach_stream_callback(llm_judge)
    if logging_enabled:
        llm_summarizer = LoggingLLMProvider(inner_provider=llm_summarizer)
        llm_critic = LoggingLLMProvider(inner_provider=llm_critic)
        llm_judge = LoggingLLMProvider(inner_provider=llm_judge)

    embedding: EmbeddingProvider = hydra.utils.instantiate(
        OmegaConf.create(embedding_profile.embedding_config)
    )
    retrieval_enabled = embedding_profile.embedding_id != "disabled"
    transcript_index = TranscriptIndex(embedding_provider=embedding)

    event_capture = EventCapture(
        [
            "tool_invocation",
            "retrieval_stats",
            "orchestrator_iteration_completed",
        ]
    )
    event_capture.start()
    shared_agent_client = AgentClient(event_bus=event_bus)
    servers: list[ManagedServer] = []
    reserved_ports: tuple[int, int, int] | None = None
    try:
        retrieval_port, summarizer_port, critic_port = _resolve_benchmark_ports(args)
        reserved_ports = (retrieval_port, summarizer_port, critic_port)

        if not retrieval_enabled:
            _release_reserved_port(retrieval_port)

        if retrieval_enabled:
            retrieval_app = _build_a2a_app(
                name="InfoRetrieval",
                description="Indexes transcript segments and retrieves relevant ones.",
                port=retrieval_port,
                executor=InfoRetrievalExecutor(transcript_index),
            )
            servers.append(
                _run_server_in_thread("sc-retrieval", retrieval_app, retrieval_port)
            )

        summarizer_app = _build_a2a_app(
            name="Summarizer",
            description="Produces abstractive summaries.",
            port=summarizer_port,
            executor=SummarizerExecutor(
                llm=llm_summarizer,
                retrieval_port=retrieval_port,
                max_tool_turns=int(base_cfg.agents.summarizer.max_tool_turns),
                calibration=calibration,
                is_embedding_enabled=retrieval_enabled,
                loop_guardrails=_coerce_mapping(
                    OmegaConf.to_container(
                        base_cfg.agents.summarizer.get("loop_guardrails", {}),
                        resolve=True,
                    )
                ),
                agent_client=shared_agent_client,
                event_bus=event_bus,
            ),
        )
        servers.append(
            _run_server_in_thread("sc-summarizer", summarizer_app, summarizer_port)
        )

        critic_app = _build_a2a_app(
            name="Critic",
            description="Evaluates summaries.",
            port=critic_port,
            executor=CriticExecutor(
                llm=llm_critic,
                calibration=calibration,
                max_tool_turns=int(base_cfg.agents.critic.max_tool_turns),
                retrieval_port=retrieval_port,
                is_embedding_enabled=retrieval_enabled,
                reasoning_loop_guard=_coerce_mapping(
                    OmegaConf.to_container(
                        base_cfg.agents.critic.get("reasoning_loop_guard", {}),
                        resolve=True,
                    )
                ),
                agent_client=shared_agent_client,
                event_bus=event_bus,
            ),
        )
        servers.append(_run_server_in_thread("sc-critic", critic_app, critic_port))

        checker = AgentHealthChecker(max_retries=5, base_delay=1.0)
        targets = [("Summarizer", summarizer_port), ("Critic", critic_port)]
        if retrieval_enabled:
            targets.insert(0, ("InfoRetrieval", retrieval_port))
        for name, port in targets:
            if not checker.check(name, port):
                raise BenchmarkExecutionError(f"Health check failed for {name}:{port}")

        orchestrator = Orchestrator(
            retrieval_port=retrieval_port,
            summarizer_port=summarizer_port,
            critic_port=critic_port,
            timeouts=_resolve_orchestrator_timeouts(base_cfg, args),
            agent_client=shared_agent_client,
            event_bus=event_bus,
        )

        rubric = load_judge_rubric(Path(args.judge_rubric).resolve())
        reference_free_pipeline = ReferenceFreePipeline(
            alignscore_runner=AlignScoreRunner(
                config=AlignScoreRunnerConfig(
                    model_name=str(args.alignscore_model),
                    source_chunk_words=int(args.alignscore_source_chunk_words),
                )
            ),
            judge_runner=LLMJudgeRunner(
                judge_llm=llm_judge,
                rubric=rubric,
                config=LLMJudgeRunnerConfig(
                    source_char_limit=int(args.judge_source_char_limit)
                ),
            ),
            config=ReferenceFreePipelineConfig(
                enable_order_swap=not bool(args.disable_order_swap),
                judge_repeats=int(args.judge_repeats),
            ),
        )

        sample_seed = (
            int(args.sample_seed)
            if getattr(args, "sample_seed", None) is not None
            else None
        )
        samples_to_run = _select_samples(
            dataset_manifest.samples,
            max_samples=int(args.max_samples),
            sample_seed=sample_seed,
        )
        total_samples = len(samples_to_run)
        sample_selection_label = (
            f"seeded_random({sample_seed})"
            if sample_seed is not None
            else "manifest_order"
        )

        event_bus.publish(
            "status_message",
            (
                f"Starting summarizer-critic benchmark run {run_id} with "
                f"{total_samples} sample(s); summarizer={summarizer_profile.provider_id}; "
                f"critic={critic_profile.provider_id}; judge={judge_profile.provider_id}; "
                f"sample_selection={sample_selection_label}."
            ),
        )

        results: list[SampleBenchmarkResult] = []
        benchmark_started = time.perf_counter()
        duration_tolerance_ratio = float(base_cfg.orchestrator.duration_tolerance_ratio)
        max_iterations = (
            int(args.max_iterations)
            if int(args.max_iterations) > 0
            else int(base_cfg.orchestrator.max_iterations)
        )
        for sample_index, sample in enumerate(samples_to_run, start=1):
            remaining_samples, average_text, eta_text = _progress_snapshot(
                completed_samples=sample_index - 1,
                total_samples=total_samples,
                elapsed_seconds=time.perf_counter() - benchmark_started,
            )
            event_bus.publish(
                "status_message",
                (
                    f"Sample {sample_index}/{len(samples_to_run)} start: "
                    f"{sample.sample_id} target={sample.target_seconds}s "
                    f"remaining={remaining_samples} rolling_avg={average_text} eta={eta_text}"
                ),
            )
            result = await _execute_sample(
                sample=sample,
                orchestrator=orchestrator,
                event_capture=event_capture,
                retrieval_enabled=retrieval_enabled,
                duration_tolerance_ratio=duration_tolerance_ratio,
                max_iterations=max_iterations,
                reference_free_pipeline=reference_free_pipeline,
                calibration=calibration,
            )
            results.append(result)
            remaining_samples, average_text, eta_text = _progress_snapshot(
                completed_samples=sample_index,
                total_samples=total_samples,
                elapsed_seconds=time.perf_counter() - benchmark_started,
            )
            event_bus.publish(
                "status_message",
                (
                    f"Sample {sample_index}/{len(samples_to_run)} done: "
                    f"{sample.sample_id} status={result.final_status} "
                    f"remaining={remaining_samples} rolling_avg={average_text} eta={eta_text} "
                    f"composite={result.metrics.composite:.3f} "
                    f"elapsed={result.duration_seconds:.1f}s"
                ),
            )
        successful_results = [
            result for result in results if result.status == "completed"
        ]
        summary = BenchmarkSummary(
            sample_count=len(results),
            success_count=len(successful_results),
            failure_count=len(results) - len(successful_results),
            mean_duration_control=(
                sum(result.metrics.duration_control for result in results)
                / len(results)
                if results
                else 0.0
            ),
            mean_factualness=(
                sum(result.metrics.factualness for result in results) / len(results)
                if results
                else 0.0
            ),
            mean_naturalness=(
                sum(result.metrics.naturalness for result in results) / len(results)
                if results
                else 0.0
            ),
            mean_speaker_attribution=(
                sum(result.metrics.speaker_attribution for result in results)
                / len(results)
                if results
                else 0.0
            ),
            mean_composite=(
                sum(result.metrics.composite for result in results) / len(results)
                if results
                else 0.0
            ),
        )
        git_commit, git_branch = git_info(REPO_ROOT)
        provisional_report = BenchmarkExecutionReport(
            run_id=run_id,
            generated_at_utc=utc_now_iso(),
            status="completed" if summary.failure_count == 0 else "partial",
            dataset_manifest_path=str(dataset_manifest.manifest_path),
            qmsum_repo_dir=str(dataset_manifest.qmsum_repo_dir),
            summarizer_provider=summarizer_profile.provider_id,
            critic_provider=critic_profile.provider_id,
            judge_provider=judge_profile.provider_id,
            embedding_mode=embedding_profile.embedding_id,
            commands_executed=[" ".join(["python", *sys.argv])],
            git_commit=git_commit,
            git_branch=git_branch,
            report_path="",
            notes_path="",
            results_path="",
            samples=results,
            summary=summary,
            sample_seed=sample_seed,
        )
        report_path, notes_path, results_path = write_execution_artifacts(
            run_root=run_root,
            report=provisional_report,
        )
        final_report = replace(
            provisional_report,
            report_path=str(report_path),
            notes_path=str(notes_path),
            results_path=str(results_path),
        )
        write_execution_artifacts(run_root=run_root, report=final_report)
        return final_report
    finally:
        event_capture.stop()
        for server in reversed(servers):
            _stop_server(server)
        if reserved_ports is not None:
            for port in reserved_ports:
                _release_reserved_port(port)
        await shared_agent_client.close()


def execute_benchmark(args: argparse.Namespace) -> BenchmarkExecutionReport:
    """Run the summarizer-critic benchmark and return the report."""
    try:
        return asyncio.run(_run_benchmark_async(args))
    except (
        BenchmarkExecutionError,
        DatasetPreparationError,
        MatrixConfigError,
        ReferenceFreeContractError,
    ) as exc:
        raise BenchmarkExecutionError(str(exc)) from exc


def build_execute_parser(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for the execute subcommand."""
    parser.add_argument(
        "--dataset-manifest", default=str(DEFAULT_DATASET_MANIFEST_PATH)
    )
    parser.add_argument("--prepare-if-missing", action="store_true")
    parser.add_argument("--force-download-dataset", action="store_true")
    parser.add_argument(
        "--qmsum-repo-dir",
        default=str(
            DEFAULT_DATASET_MANIFEST_PATH.parent.parent / "qmsum_official_repo"
        ),
    )
    parser.add_argument("--reference-wpm", type=float, default=DEFAULT_REFERENCE_WPM)
    parser.add_argument("--output-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument(
        "--provider-profiles", default=str(DEFAULT_PROVIDER_PROFILES_PATH)
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--summarizer-provider", default="vllm_default")
    parser.add_argument("--critic-provider", default="")
    parser.add_argument("--judge-provider", default="")
    parser.add_argument(
        "--embedding-mode", choices=("disabled", "enabled"), default="disabled"
    )
    parser.add_argument("--judge-rubric", default=str(DEFAULT_JUDGE_RUBRIC_PATH))
    parser.add_argument("--judge-repeats", type=int, default=1)
    parser.add_argument("--disable-order-swap", action="store_true")
    parser.add_argument(
        "--alignscore-model", default="sentence-transformers/all-MiniLM-L6-v2"
    )
    parser.add_argument("--alignscore-source-chunk-words", type=int, default=256)
    parser.add_argument("--judge-source-char-limit", type=int, default=30000)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help=(
            "Deterministically randomize sample ordering and prefix selection by "
            "seed. Omit to preserve manifest order."
        ),
    )
    parser.add_argument("--max-iterations", type=int, default=0)
    parser.add_argument("--summarizer-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--critic-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--retrieval-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--index-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--retrieval-port", type=int, default=None)
    parser.add_argument("--summarizer-port", type=int, default=None)
    parser.add_argument("--critic-port", type=int, default=None)
