"""CLI entrypoint for reproducible summarizer benchmark runs."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
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
from card_framework.benchmark.artifacts import (
    build_verification_payload,
    git_info,
    sha256_file,
    utc_now_iso,
    write_json_with_hash,
)
from card_framework.benchmark.datasets import ManifestError, load_manifest, load_transcript, prepare_manifest
from card_framework.benchmark.instrumentation import EventCapture
from card_framework.benchmark.matrix import (
    MatrixConfigError,
    build_cells,
    build_embedding_profiles,
    cells_to_dict,
    default_presets,
    load_provider_profiles,
    missing_required_env,
    resolve_provider_config,
)
from card_framework.benchmark.metrics import aggregate_results, parse_junit_totals
from card_framework.benchmark.reference_free.alignscore_runner import (
    AlignScoreRunner,
    AlignScoreRunnerConfig,
)
from card_framework.benchmark.reference_free.contracts import (
    JudgeRubric,
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
from card_framework.benchmark.types import (
    BenchmarkAggregate,
    BenchmarkReport,
    CellRunResult,
    ProviderProfile,
    SampleRunResult,
)
from card_framework.retrieval.embeddings import TranscriptIndex
from card_framework.shared.events import event_bus
from card_framework.shared.llm_provider import EmbeddingProvider, LLMProvider
from card_framework.shared.logger_utils import configure_logger
from card_framework.runtime.loop_orchestrator import Orchestrator
from card_framework.providers.logging_provider import LoggingLLMProvider
from card_framework.cli.main import _build_a2a_app
from card_framework.shared.paths import (
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    DEFAULT_CONFIG_PATH,
    DEFAULT_JUDGE_RUBRIC_PATH,
    DEFAULT_PROVIDER_PROFILES_PATH,
    REPO_ROOT,
)


class BenchmarkRuntimeError(RuntimeError):
    """Raised when benchmark execution cannot proceed."""


class ManagedServer:
    """Uvicorn server and backing thread handle for managed lifecycle."""

    def __init__(self, *, name: str, port: int, server: uvicorn.Server, thread: threading.Thread) -> None:
        self.name = name
        self.port = port
        self.server = server
        self.thread = thread


def _reserve_port() -> int:
    """Reserve and return a currently available localhost TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _reserve_ports(count: int) -> list[int]:
    """Reserve ``count`` localhost ports."""
    return [_reserve_port() for _ in range(count)]


def _run_server_in_thread(name: str, app: Any, port: int) -> ManagedServer:
    """Start a uvicorn server in a daemon thread and return lifecycle handle."""
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
    return ManagedServer(name=name, port=port, server=server, thread=thread)


def _stop_server(server: ManagedServer, timeout_seconds: float = 10.0) -> None:
    """Stop managed uvicorn server gracefully."""
    server.server.should_exit = True
    server.thread.join(timeout=timeout_seconds)


def _format_transcript(transcript: dict[str, Any]) -> str:
    """Format transcript segments for non-retrieval summarization mode."""
    return "".join(
        f"[{segment.get('speaker', 'UNKNOWN')}]: {segment.get('text', '')}\n"
        for segment in transcript.get("segments", [])
    )


def _classify_error(exc: Exception) -> str:
    """Classify execution errors into stable categories."""
    text = str(exc).lower()
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if "health" in text or "not responding" in text:
        return "server_start_error"
    if "index" in text:
        return "index_error"
    if "critic" in text and "json" in text:
        return "critic_parse_error"
    if "failed after" in text or "http" in text:
        return "transport_error"
    if "dataset" in text or "manifest" in text or "transcript" in text:
        return "dataset_load_error"
    return "unknown_error"


def _make_run_id() -> str:
    """Generate a run identifier using UTC timestamp and PID."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_pid{os.getpid()}"


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for benchmark commands."""
    parser = argparse.ArgumentParser(description="Summarizer benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    execute = subparsers.add_parser("execute", help="Execute benchmark matrix")
    execute.add_argument(
        "--manifest",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH),
        help="Path to benchmark manifest JSON",
    )
    execute.add_argument(
        "--preset",
        default="hourly",
        choices=sorted(default_presets().keys()),
        help="Matrix preset",
    )
    execute.add_argument(
        "--output-dir",
        default="artifacts/benchmark",
        help="Output directory root",
    )
    execute.add_argument(
        "--provider-profiles",
        default=str(DEFAULT_PROVIDER_PROFILES_PATH),
        help="Provider profile YAML",
    )
    execute.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Base application config YAML",
    )
    execute.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional sample cap override (0 means preset default)",
    )
    execute.add_argument(
        "--providers",
        default="",
        help="Comma-separated provider IDs to include (default: all profiles)",
    )
    execute.add_argument(
        "--judge-provider",
        default="",
        help=(
            "Provider profile ID used for LLM-as-judge metrics. "
            "If omitted, each candidate model self-judges."
        ),
    )
    execute.add_argument(
        "--judge-rubric",
        default=str(DEFAULT_JUDGE_RUBRIC_PATH),
        help="Path to JSON rubric contract for LLM-as-judge scoring.",
    )
    execute.add_argument(
        "--judge-repeats",
        type=int,
        default=0,
        help=(
            "Direct judge repeats per sample. "
            "0 uses preset default (smoke=1, hourly=2, full=3)."
        ),
    )
    execute.add_argument(
        "--disable-order-swap",
        action="store_true",
        help="Disable order-swapped pairwise bias diagnostics for judge metrics.",
    )
    execute.add_argument(
        "--alignscore-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model used by alignscore semantic fallback backend.",
    )
    execute.add_argument(
        "--alignscore-source-chunk-words",
        type=int,
        default=256,
        help="Source chunk size in words for alignscore fallback backend.",
    )
    execute.add_argument(
        "--judge-source-char-limit",
        type=int,
        default=30000,
        help="Maximum source transcript characters sent to judge prompts.",
    )

    prepare = subparsers.add_parser(
        "prepare-manifest", help="Build a frozen benchmark manifest"
    )
    prepare.add_argument(
        "--output",
        default=str(DEFAULT_BENCHMARK_MANIFEST_PATH),
        help="Target manifest path",
    )
    prepare.add_argument(
        "--sources",
        default="local",
        help="Comma-separated sources: local,qmsum,ami",
    )
    prepare.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Samples per remote source",
    )
    prepare.add_argument(
        "--local-transcript-path",
        default="auto",
        help=(
            "Local transcript path for local source. Defaults to auto-discovery "
            "from repo-root transcript.json, repo-root *.transcript.json, then "
            "artifacts/transcripts/*.transcript.json."
        ),
    )

    return parser


def _sample_cap_for_preset(preset_name: str, override: int) -> int:
    preset = default_presets()[preset_name]
    if override > 0:
        return min(override, preset.max_samples)
    return preset.max_samples


def _select_samples(samples: list[Any], cap: int) -> list[Any]:
    return samples[:cap]


def _default_judge_repeats_for_preset(preset_name: str) -> int:
    """Return default direct-judge repeats for a preset."""
    if preset_name == "smoke":
        return 1
    if preset_name == "hourly":
        return 2
    return 3


def _find_provider_profile(
    profiles: list[ProviderProfile],
    provider_id: str,
) -> ProviderProfile | None:
    """Find a provider profile by ID."""
    for profile in profiles:
        if profile.provider_id == provider_id:
            return profile
    return None


async def _run_cell(
    *,
    cell: Any,
    samples: list[Any],
    base_cfg: DictConfig,
    judge_profile: ProviderProfile | None,
    judge_rubric: JudgeRubric,
    judge_repeats: int,
    enable_order_swap: bool,
    alignscore_model: str,
    alignscore_source_chunk_words: int,
    judge_source_char_limit: int,
) -> CellRunResult:
    """Execute one matrix cell against all selected samples."""
    servers: list[ManagedServer] = []
    event_capture = EventCapture(
        [
            "tool_invocation",
            "retrieval_stats",
            "a2a_call_started",
            "a2a_call_succeeded",
            "a2a_call_retry",
            "a2a_call_failed",
            "llm_call_completed",
            "orchestrator_iteration_completed",
            "reference_free_started",
            "reference_free_completed",
            "reference_free_failed",
        ]
    )
    event_capture.start()
    shared_event_bus = event_bus
    shared_agent_client = AgentClient(event_bus=shared_event_bus)

    try:
        llm_cfg = OmegaConf.create(cell.llm_config)
        llm: LLMProvider = hydra.utils.instantiate(llm_cfg)
        if bool(base_cfg.get("logging", {}).get("enabled", False)):
            llm = LoggingLLMProvider(inner_provider=llm)

        judge_llm: LLMProvider = llm
        if judge_profile is not None:
            judge_cfg = OmegaConf.create(resolve_provider_config(judge_profile))
            judge_llm = hydra.utils.instantiate(judge_cfg)
            if bool(base_cfg.get("logging", {}).get("enabled", False)):
                judge_llm = LoggingLLMProvider(inner_provider=judge_llm)

        alignscore_runner = AlignScoreRunner(
            config=AlignScoreRunnerConfig(
                model_name=alignscore_model,
                source_chunk_words=alignscore_source_chunk_words,
            )
        )
        judge_runner = LLMJudgeRunner(
            judge_llm=judge_llm,
            rubric=judge_rubric,
            config=LLMJudgeRunnerConfig(source_char_limit=judge_source_char_limit),
        )
        reference_free_pipeline = ReferenceFreePipeline(
            alignscore_runner=alignscore_runner,
            judge_runner=judge_runner,
            config=ReferenceFreePipelineConfig(
                enable_order_swap=enable_order_swap,
                judge_repeats=judge_repeats,
            ),
        )

        embedding_cfg = OmegaConf.create(cell.embedding_config)
        embedding: EmbeddingProvider = hydra.utils.instantiate(embedding_cfg)
        is_embedding_enabled = "NoOpEmbeddingProvider" not in str(
            cell.embedding_config.get("_target_", "")
        )

        transcript_index = TranscriptIndex(embedding_provider=embedding)

        retrieval_port, summarizer_port, critic_port = _reserve_ports(3)

        if is_embedding_enabled:
            retrieval_app = _build_a2a_app(
                name="InfoRetrieval",
                description="Indexes transcript segments and retrieves relevant ones.",
                port=retrieval_port,
                executor=InfoRetrievalExecutor(transcript_index),
            )
            servers.append(
                _run_server_in_thread("benchmark-retrieval-a2a", retrieval_app, retrieval_port)
            )

        summarizer_app = _build_a2a_app(
            name="Summarizer",
            description="Produces abstractive summaries.",
            port=summarizer_port,
            executor=SummarizerExecutor(
                llm=llm,
                retrieval_port=retrieval_port,
                max_tool_turns=base_cfg.get("agents", {})
                .get("summarizer", {})
                .get("max_tool_turns", 3),
                is_embedding_enabled=is_embedding_enabled,
                agent_client=shared_agent_client,
                event_bus=shared_event_bus,
            ),
        )
        servers.append(
            _run_server_in_thread("benchmark-summarizer-a2a", summarizer_app, summarizer_port)
        )

        critic_app = _build_a2a_app(
            name="Critic",
            description="Evaluates summaries.",
            port=critic_port,
            executor=CriticExecutor(
                llm=llm,
                max_tool_turns=base_cfg.get("agents", {})
                .get("critic", {})
                .get("max_tool_turns", 5),
                retrieval_port=retrieval_port,
                is_embedding_enabled=is_embedding_enabled,
                agent_client=shared_agent_client,
                event_bus=shared_event_bus,
            ),
        )
        servers.append(_run_server_in_thread("benchmark-critic-a2a", critic_app, critic_port))

        checker = AgentHealthChecker(max_retries=5, base_delay=1.0)
        health_targets = [("Summarizer", summarizer_port), ("Critic", critic_port)]
        if is_embedding_enabled:
            health_targets.insert(0, ("InfoRetrieval", retrieval_port))

        for name, port in health_targets:
            if not checker.check(name, port):
                raise BenchmarkRuntimeError(f"Health check failed for {name}:{port}")

        orchestrator = Orchestrator(
            retrieval_port=retrieval_port,
            summarizer_port=summarizer_port,
            critic_port=critic_port,
            timeouts=dict(base_cfg.orchestrator.get("timeouts", {})),
            agent_client=shared_agent_client,
            event_bus=shared_event_bus,
        )

        min_words = int(base_cfg.orchestrator.min_words)
        max_words = int(base_cfg.orchestrator.max_words)
        max_iterations = int(base_cfg.orchestrator.max_iterations)

        sample_results: list[SampleRunResult] = []

        for sample in samples:
            sample_started = time.perf_counter()
            retrieval_before = event_capture.count("retrieval_stats")
            tool_before = event_capture.count("tool_invocation")
            try:
                transcript = load_transcript(sample)
                source_text = _format_transcript(transcript)
                full_transcript_text = ""
                if is_embedding_enabled:
                    await orchestrator.index_transcript(transcript)
                else:
                    full_transcript_text = source_text

                diagnostics = await orchestrator.run_loop_with_diagnostics(
                    min_words=min_words,
                    max_words=max_words,
                    max_iterations=max_iterations,
                    full_transcript_text=full_transcript_text,
                )

                final_word_count = int(diagnostics.get("final_word_count", 0))
                if final_word_count <= 0 and diagnostics.get("draft"):
                    final_word_count = len(str(diagnostics["draft"]).split())

                fallback_verdict_used = any(
                    detail.get("critic_status") == "invalid_json"
                    for detail in diagnostics.get("iteration_details", [])
                )

                status = "pass" if diagnostics.get("converged", False) else "fail"
                duration_seconds = round(time.perf_counter() - sample_started, 3)
                retrieval_after = event_capture.count("retrieval_stats")
                tool_after = event_capture.count("tool_invocation")
                candidate_summary = str(diagnostics.get("draft", ""))

                event_capture_metadata = {
                    "sample_id": sample.sample_id,
                    "provider_id": cell.provider_id,
                    "judge_provider_id": (
                        judge_profile.provider_id if judge_profile else cell.provider_id
                    ),
                }
                event_capture_metadata = {
                    key: value for key, value in event_capture_metadata.items() if value
                }

                reference_free_result = None
                try:
                    shared_event_bus.publish(
                        "reference_free_started",
                        **event_capture_metadata,
                    )
                    reference_free_result = reference_free_pipeline.evaluate_sample(
                        source_text=source_text,
                        summary_text=candidate_summary,
                    )
                    shared_event_bus.publish(
                        "reference_free_completed",
                        status=reference_free_result.status,
                        **event_capture_metadata,
                    )
                except Exception as reference_free_exc:
                    shared_event_bus.publish(
                        "reference_free_failed",
                        error=str(reference_free_exc),
                        **event_capture_metadata,
                    )
                    reference_free_result = None

                sample_results.append(
                    SampleRunResult(
                        sample_id=sample.sample_id,
                        dataset=sample.dataset,
                        status=status,
                        duration_seconds=duration_seconds,
                        converged=bool(diagnostics.get("converged", False)),
                        iterations_run=int(diagnostics.get("iterations_run", 0)),
                        final_status=str(diagnostics.get("final_status", "unknown")),
                        final_word_count=final_word_count,
                        word_budget_in_range=min_words <= final_word_count <= max_words,
                        fallback_verdict_used=fallback_verdict_used,
                        retrieval_events=max(0, retrieval_after - retrieval_before),
                        tool_invocations=max(0, tool_after - tool_before),
                        alignscore=(
                            reference_free_result.alignscore
                            if reference_free_result
                            else None
                        ),
                        alignscore_backend=(
                            reference_free_result.alignscore_backend
                            if reference_free_result
                            else None
                        ),
                        judge_scores=(
                            reference_free_result.judge_scores
                            if reference_free_result
                            else None
                        ),
                        judge_pairwise_winner=(
                            reference_free_result.judge_pairwise_winner
                            if reference_free_result
                            else None
                        ),
                        judge_order_consistent=(
                            reference_free_result.judge_order_consistent
                            if reference_free_result
                            else None
                        ),
                        judge_repeat_delta=(
                            reference_free_result.judge_repeat_delta
                            if reference_free_result
                            else None
                        ),
                        reference_free_status=(
                            reference_free_result.status
                            if reference_free_result
                            else "error"
                        ),
                        reference_free_error=(
                            reference_free_result.error_message
                            if reference_free_result
                            else "Reference-free pipeline execution failed"
                        ),
                    )
                )
            except Exception as exc:
                duration_seconds = round(time.perf_counter() - sample_started, 3)
                retrieval_after = event_capture.count("retrieval_stats")
                tool_after = event_capture.count("tool_invocation")
                sample_results.append(
                    SampleRunResult(
                        sample_id=sample.sample_id,
                        dataset=sample.dataset,
                        status="fail",
                        duration_seconds=duration_seconds,
                        converged=False,
                        iterations_run=0,
                        final_status="execution_error",
                        final_word_count=0,
                        word_budget_in_range=False,
                        fallback_verdict_used=False,
                        retrieval_events=max(0, retrieval_after - retrieval_before),
                        tool_invocations=max(0, tool_after - tool_before),
                        reference_free_status="not_scored",
                        failure_category=_classify_error(exc),
                        error_message=str(exc),
                    )
                )

        overall_status = (
            "pass" if all(result.status == "pass" for result in sample_results) else "fail"
        )
        return CellRunResult(
            cell_id=cell.cell_id,
            provider_id=cell.provider_id,
            embedding_id=cell.embedding_id,
            repeat_index=cell.repeat_index,
            status=overall_status,
            sample_results=sample_results,
        )

    finally:
        event_capture.stop()
        for server in reversed(servers):
            _stop_server(server)
        await shared_agent_client.close()


def _commands_executed() -> list[str]:
    """Capture command provenance for report artifacts."""
    return [" ".join(["python", *sys.argv])]


def _load_base_config(config_path: Path) -> DictConfig:
    if not config_path.exists():
        raise BenchmarkRuntimeError(f"Config path does not exist: {config_path}")
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise BenchmarkRuntimeError("Failed to load benchmark base config")
    return cfg


def _provider_filter(raw: str) -> set[str] | None:
    cleaned = [item.strip() for item in raw.split(",") if item.strip()]
    if not cleaned:
        return None
    return set(cleaned)


def _resolve_report_outcome(
    *,
    aggregates: BenchmarkAggregate,
    failures: list[dict[str, Any]],
) -> tuple[str, str | None]:
    """Resolve top-level benchmark status and optional terminal error.

    Args:
        aggregates: Aggregate execution counters for the benchmark run.
        failures: Collected cell-level failure payloads.

    Returns:
        Tuple of report status and optional user-facing error summary.
    """
    if aggregates.executed_cells <= 0:
        first_failure = failures[0].get("error") if failures else None
        detail = (
            f" First failure: {first_failure}"
            if isinstance(first_failure, str) and first_failure.strip()
            else ""
        )
        return (
            "failed",
            (
                "Benchmark did not execute any cells. Check provider reachability, "
                f"credentials, and runtime prerequisites.{detail}"
            ),
        )
    if aggregates.total_samples <= 0:
        return (
            "failed",
            "Benchmark executed zero samples. Check the manifest and sample selection.",
        )
    return "completed", None


def execute_command(args: argparse.Namespace) -> int:
    """Execute benchmark command and write report artifacts."""
    repo_root = REPO_ROOT
    run_id = _make_run_id()

    config_path = (repo_root / args.config).resolve()
    base_cfg = _load_base_config(config_path)
    configure_logger(base_cfg.logging)

    manifest_path = (repo_root / args.manifest).resolve()
    provider_profiles_path = (repo_root / args.provider_profiles).resolve()

    try:
        samples = load_manifest(manifest_path, repo_root)
    except ManifestError as exc:
        raise BenchmarkRuntimeError(str(exc)) from exc

    presets = default_presets()
    preset = presets[args.preset]
    sample_cap = _sample_cap_for_preset(args.preset, int(args.max_samples or 0))
    selected_samples = _select_samples(samples, sample_cap)
    if not selected_samples:
        raise BenchmarkRuntimeError("No samples selected after applying cap")

    all_profiles = load_provider_profiles(provider_profiles_path)
    profiles = list(all_profiles)
    provider_filter = _provider_filter(args.providers)

    if provider_filter is not None:
        profiles = [profile for profile in profiles if profile.provider_id in provider_filter]
    elif args.preset == "smoke":
        profiles = [
            ProviderProfile(
                provider_id="configured_default",
                description="Configured default provider from packaged config.yaml",
                llm_config=OmegaConf.to_container(base_cfg.llm, resolve=False),
                required_env=[],
            )
        ]

    if not profiles:
        raise BenchmarkRuntimeError("No provider profiles selected for execution")

    judge_profile: ProviderProfile | None = None
    judge_provider_id = str(args.judge_provider).strip()
    if judge_provider_id:
        judge_profile = _find_provider_profile(all_profiles, judge_provider_id)
        if judge_profile is None:
            raise BenchmarkRuntimeError(
                f"Judge provider '{judge_provider_id}' was not found in provider profiles"
            )
        missing_judge_env = missing_required_env(judge_profile)
        if missing_judge_env:
            raise BenchmarkRuntimeError(
                "Judge provider is missing required environment variables: "
                + ", ".join(missing_judge_env)
            )

    rubric_path = (repo_root / args.judge_rubric).resolve()
    try:
        judge_rubric = load_judge_rubric(rubric_path)
    except ReferenceFreeContractError as exc:
        raise BenchmarkRuntimeError(str(exc)) from exc

    judge_repeats = (
        int(args.judge_repeats)
        if int(args.judge_repeats) > 0
        else _default_judge_repeats_for_preset(args.preset)
    )
    enable_order_swap = not bool(args.disable_order_swap)

    embedding_profiles = build_embedding_profiles(base_cfg)

    unavailable_providers: list[dict[str, Any]] = []
    available_profiles = []
    for profile in profiles:
        missing_env = missing_required_env(profile)
        if missing_env:
            unavailable_providers.append(
                {
                    "provider_id": profile.provider_id,
                    "reason": "missing_environment",
                    "missing_env": missing_env,
                }
            )
            continue
        available_profiles.append(profile)

    if not available_profiles:
        raise BenchmarkRuntimeError(
            "All selected providers are unavailable due to missing required environment variables"
        )

    cells = build_cells(
        profiles=available_profiles,
        embedding_profiles=embedding_profiles,
        preset=preset,
        sample_count=len(selected_samples),
    )

    results: list[CellRunResult] = []
    failures: list[dict[str, Any]] = list(unavailable_providers)

    for cell in cells:
        try:
            result = asyncio.run(
                _run_cell(
                    cell=cell,
                    samples=selected_samples,
                    base_cfg=base_cfg,
                    judge_profile=judge_profile,
                    judge_rubric=judge_rubric,
                    judge_repeats=judge_repeats,
                    enable_order_swap=enable_order_swap,
                    alignscore_model=str(args.alignscore_model),
                    alignscore_source_chunk_words=int(
                        args.alignscore_source_chunk_words
                    ),
                    judge_source_char_limit=int(args.judge_source_char_limit),
                )
            )
            results.append(result)
        except Exception as exc:
            results.append(
                CellRunResult(
                    cell_id=cell.cell_id,
                    provider_id=cell.provider_id,
                    embedding_id=cell.embedding_id,
                    repeat_index=cell.repeat_index,
                    status="skipped",
                    sample_results=[],
                    skipped_reason=str(exc),
                )
            )
            failures.append(
                {
                    "cell_id": cell.cell_id,
                    "provider_id": cell.provider_id,
                    "embedding_id": cell.embedding_id,
                    "failure_category": _classify_error(exc),
                    "error": str(exc),
                }
            )

    aggregates = aggregate_results(results)
    report_status, terminal_error = _resolve_report_outcome(
        aggregates=aggregates,
        failures=failures,
    )
    commit, branch = git_info(repo_root)

    report = BenchmarkReport(
        run_id=run_id,
        status=report_status,
        generated_at_utc=utc_now_iso(),
        git_commit=commit,
        git_branch=branch,
        preset=preset.name,
        manifest_path=str(manifest_path),
        provider_profiles_path=str(provider_profiles_path),
        commands_executed=_commands_executed(),
        matrix=cells_to_dict(cells),
        results=results,
        aggregates=aggregates,
        failures=failures,
    )

    output_root = (repo_root / args.output_dir / run_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    report_path = output_root / "benchmark_report.json"
    write_json_with_hash(report_path, asdict(report))

    quality_root = (repo_root / "artifacts" / "quality" / run_id).resolve()
    quality_root.mkdir(parents=True, exist_ok=True)
    quality_report_path = quality_root / "report.json"
    report_sha256 = write_json_with_hash(quality_report_path, asdict(report))

    junit_path = quality_root / "pytest.junit.xml"
    junit_sha256 = sha256_file(junit_path) if junit_path.exists() else ""
    junit_totals = parse_junit_totals(junit_path)

    verification = build_verification_payload(
        run_id=run_id,
        report_path=quality_report_path,
        report_sha256=report_sha256,
        junit_path=junit_path,
        junit_sha256=junit_sha256,
        junit_totals=junit_totals,
        commands_executed=_commands_executed(),
        git_commit=commit,
        git_branch=branch,
    )

    if junit_path.exists() and junit_sha256 and junit_totals["tests"] > 0:
        verification["status"] = "verified"

    verification_path = output_root / "verification.json"
    write_json_with_hash(verification_path, verification)

    summary = {
        "status": report_status,
        "run_id": run_id,
        "report_path": str(report_path),
        "quality_report_path": str(quality_report_path),
        "verification_path": str(verification_path),
        "judge_provider": judge_profile.provider_id if judge_profile else "self_judge",
        "judge_rubric": str(rubric_path),
        "executed_cells": aggregates.executed_cells,
        "skipped_cells": aggregates.skipped_cells,
        "total_samples": aggregates.total_samples,
        "critic_pass_rate": aggregates.critic_pass_rate,
        "alignscore_mean": aggregates.alignscore_mean,
        "judge_overall_mean": aggregates.judge_overall_mean,
    }
    if terminal_error is not None:
        summary["error"] = terminal_error
    print(json.dumps(summary, indent=2))
    return 0 if report_status == "completed" else 1


def prepare_manifest_command(args: argparse.Namespace) -> int:
    """Generate a frozen benchmark manifest."""
    repo_root = REPO_ROOT
    output_path = (repo_root / args.output).resolve()

    sources = [entry.strip() for entry in args.sources.split(",") if entry.strip()]
    if not sources:
        raise BenchmarkRuntimeError("At least one source must be provided")

    manifest = prepare_manifest(
        repo_root=repo_root,
        output_path=output_path,
        sources=sources,
        num_samples=args.num_samples,
        local_transcript_path=args.local_transcript_path,
    )
    print(json.dumps({"manifest_path": str(output_path), "samples": len(manifest["samples"])}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Program entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "execute":
            return execute_command(args)
        if args.command == "prepare-manifest":
            return prepare_manifest_command(args)
        parser.error("Unknown command")
        return 2
    except (
        BenchmarkRuntimeError,
        MatrixConfigError,
        ManifestError,
        ReferenceFreeContractError,
    ) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

