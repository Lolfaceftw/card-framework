"""Standalone QA benchmark runner for source-grounded summary scoring."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import sys
import threading
from typing import Any, Literal, cast

import hydra
from omegaconf import DictConfig, OmegaConf
import uvicorn

from card_framework.agents.corrector import LLMCorrectorAgent
from card_framework.agents.client import AgentClient
from card_framework.agents.dtos import GroundTruthCreatorTaskRequest
from card_framework.agents.ground_truth_creator import GroundTruthCreatorExecutor
from card_framework.agents.health import AgentHealthChecker
from card_framework.agents.qa_evaluator import QAEvaluatorExecutor
from card_framework.benchmark.artifacts import (
    build_verification_payload,
    git_info,
    write_json_with_hash,
)
from card_framework.benchmark.matrix import (
    MatrixConfigError,
    load_provider_profiles,
    missing_required_env,
    resolve_provider_config,
)
from card_framework.benchmark.qa_input_guard import evaluate_input_compatibility
from card_framework.benchmark.qa_contracts import (
    EvaluatorAnswerRecord,
    EvaluatorScore,
    GroundTruthSet,
    QABenchmarkReport,
    build_score,
    validate_answer_coverage,
)
from card_framework.benchmark.qa_evaluator_runner import evaluate_questions_with_fresh_contexts
from card_framework.benchmark.qa_settings import (
    CorrectorRuntimeConfig,
    EvaluatorRuntimeConfig,
    QAConfigError,
    as_positive_float,
    as_positive_int,
    resolve_corrector_runtime_config,
    resolve_evaluator_runtime_config,
    resolve_input_guard_config,
    resolve_workflow_timeouts,
)
from card_framework.benchmark.types import ProviderProfile
from card_framework.shared.events import event_bus
from card_framework.shared.llm_provider import EmbeddingProvider, LLMProvider
from card_framework.shared.logger_utils import configure_logger, logger
from card_framework.providers.logging_provider import LoggingLLMProvider
from card_framework.cli.main import _build_a2a_app
from card_framework.shared.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROVIDER_PROFILES_PATH,
    DEFAULT_QA_CONFIG_PATH,
    REPO_ROOT,
)


class QABenchmarkRuntimeError(RuntimeError):
    """Raised when the QA benchmark cannot proceed."""


class ManagedServer:
    """Uvicorn server and backing thread handle for managed lifecycle."""

    def __init__(
        self,
        *,
        name: str,
        port: int,
        server: uvicorn.Server,
        thread: threading.Thread,
    ) -> None:
        self.name = name
        self.port = port
        self.server = server
        self.thread = thread


def _reserve_port() -> int:
    """Reserve and return an available localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _reserve_ports(count: int) -> list[int]:
    """Reserve a fixed number of localhost ports."""
    return [_reserve_port() for _ in range(count)]


def _run_server_in_thread(name: str, app: Any, port: int) -> ManagedServer:
    """Start a uvicorn application in a daemon thread."""
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


def _make_run_id() -> str:
    """Generate QA benchmark run id using UTC timestamp and pid."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_pid{os.getpid()}"


def _commands_executed() -> list[str]:
    """Capture command provenance list for report artifacts."""
    return [" ".join(["python", *sys.argv])]


def _supports_ansi_colors() -> bool:
    """Return whether ANSI terminal colors should be used."""
    if os.getenv("NO_COLOR", "").strip():
        return False
    if os.getenv("FORCE_COLOR", "").strip():
        return True
    if not sys.stdout.isatty():
        return False
    if os.getenv("TERM", "").lower() == "dumb":
        return False
    if os.name != "nt":
        return True
    return bool(
        os.getenv("WT_SESSION")
        or os.getenv("ANSICON")
        or os.getenv("ConEmuANSI", "").upper() == "ON"
        or os.getenv("TERM_PROGRAM", "").lower() == "vscode"
    )


def _style_text(
    text: str,
    *,
    color_code: str = "",
    bold: bool = False,
    use_color: bool,
) -> str:
    """Apply ANSI style when colors are enabled."""
    if not use_color:
        return text
    parts: list[str] = []
    if bold:
        parts.append("\033[1m")
    if color_code:
        parts.append(color_code)
    return "".join(parts) + text + "\033[0m"


def _score_color(score_out_of_100: int) -> str:
    """Return ANSI color code representing score quality."""
    if score_out_of_100 >= 70:
        return "\033[32m"
    if score_out_of_100 >= 40:
        return "\033[33m"
    return "\033[31m"


def _tool_status_color(status: str) -> str:
    """Return ANSI color code for one tool status label."""
    normalized = status.strip().lower()
    if normalized == "accepted":
        return "\033[32m"
    if any(
        token in normalized
        for token in ("error", "failed", "invalid", "missing", "retry_exceeded")
    ):
        return "\033[31m"
    return "\033[36m"


def _format_terminal_success_summary(
    *,
    run_id: str,
    score: EvaluatorScore,
    answers: list[EvaluatorAnswerRecord],
    report_path: Path,
    trace_path: Path,
    questions_path: Path,
    verification_path: Path,
    use_color: bool,
) -> str:
    """Build a human-readable QA benchmark success summary for terminal output."""
    tool_status_counts = Counter(answer.tool_status for answer in answers)
    sorted_statuses = sorted(
        tool_status_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    accepted_count = tool_status_counts.get("accepted", 0)
    failure_count = sum(
        count for status, count in tool_status_counts.items() if status != "accepted"
    )

    lines: list[str] = [
        "",
        _style_text(
            "QA Evaluation Completed",
            color_code="\033[36m",
            bold=True,
            use_color=use_color,
        ),
        _style_text("=" * 28, color_code="\033[36m", use_color=use_color),
        f"Run ID: {run_id}",
        (
            "Score: "
            + _style_text(
                f"{score.score_out_of_100}/100",
                color_code=_score_color(score.score_out_of_100),
                bold=True,
                use_color=use_color,
            )
        ),
        f"Factualness: {score.factualness_correct}/50",
        f"Naturalness: {score.naturalness_correct}/50",
        f"Grounding: {score.summary_grounding_pass_count}/{score.total_questions} ({score.summary_grounding_pass_rate * 100.0:.1f}%)",
        f"Tool-accepted answers: {accepted_count}/{score.total_questions}",
        f"Non-accepted answers: {failure_count}",
        "",
        _style_text(
            "Tool Status Breakdown",
            color_code="\033[36m",
            bold=True,
            use_color=use_color,
        ),
    ]
    for status, count in sorted_statuses:
        lines.append(
            f"  - {_style_text(status, color_code=_tool_status_color(status), use_color=use_color)}: {count}"
        )

    lines.extend(
        [
            "",
            _style_text(
                "Artifacts", color_code="\033[36m", bold=True, use_color=use_color
            ),
            f"  - Report: {report_path}",
            f"  - Trace: {trace_path}",
            f"  - Questions: {questions_path}",
            f"  - Verification: {verification_path}",
            "",
        ]
    )
    return "\n".join(lines)


def _format_terminal_failure_summary(*, error: str, use_color: bool) -> str:
    """Build a human-readable QA benchmark failure summary for terminal output."""
    return "\n".join(
        [
            "",
            _style_text(
                "QA Evaluation Failed",
                color_code="\033[31m",
                bold=True,
                use_color=use_color,
            ),
            _style_text("=" * 20, color_code="\033[31m", use_color=use_color),
            f"Reason: {error}",
            "",
        ]
    )


OutputFormat = Literal["auto", "human", "json", "both"]
ColorMode = Literal["auto", "always", "never"]


def _resolve_output_format(
    output_format: OutputFormat,
) -> Literal["human", "json", "both"]:
    """Resolve output format when ``auto`` is requested."""
    if output_format != "auto":
        return output_format
    return "human" if sys.stdout.isatty() else "json"


def _resolve_use_color(color_mode: ColorMode) -> bool:
    """Resolve whether colored output should be enabled."""
    if color_mode == "always":
        return True
    if color_mode == "never":
        return False
    return _supports_ansi_colors()


def _build_success_stdout_payload(
    *,
    run_id: str,
    score: EvaluatorScore,
    report_path: Path,
    verification_path: Path,
) -> dict[str, Any]:
    """Build backward-compatible machine-readable success stdout payload."""
    return {
        "run_id": run_id,
        "score_out_of_100": score.score_out_of_100,
        "factualness_correct": score.factualness_correct,
        "naturalness_correct": score.naturalness_correct,
        "summary_grounding_pass_count": score.summary_grounding_pass_count,
        "summary_grounding_pass_rate": score.summary_grounding_pass_rate,
        "report_path": str(report_path),
        "verification_path": str(verification_path),
    }


def _emit_success_output(
    *,
    output_format: OutputFormat,
    color_mode: ColorMode,
    run_id: str,
    score: EvaluatorScore,
    answers: list[EvaluatorAnswerRecord],
    report_path: Path,
    trace_path: Path,
    questions_path: Path,
    verification_path: Path,
) -> None:
    """Emit success output in human/json/both formats."""
    resolved_format = _resolve_output_format(output_format)
    use_color = _resolve_use_color(color_mode)
    human_summary = _format_terminal_success_summary(
        run_id=run_id,
        score=score,
        answers=answers,
        report_path=report_path,
        trace_path=trace_path,
        questions_path=questions_path,
        verification_path=verification_path,
        use_color=use_color,
    )
    json_payload = json.dumps(
        _build_success_stdout_payload(
            run_id=run_id,
            score=score,
            report_path=report_path,
            verification_path=verification_path,
        ),
        indent=2,
    )

    if resolved_format == "human":
        print(human_summary)
        return
    if resolved_format == "json":
        print(json_payload)
        return
    print(human_summary)
    print(json_payload)


def _emit_failure_output(
    *,
    output_format: OutputFormat,
    color_mode: ColorMode,
    error: str,
) -> None:
    """Emit failure output in human/json/both formats."""
    resolved_format = _resolve_output_format(output_format)
    use_color = _resolve_use_color(color_mode)
    human_summary = _format_terminal_failure_summary(error=error, use_color=use_color)
    json_payload = json.dumps({"status": "failed", "error": error}, indent=2)

    if resolved_format == "human":
        print(human_summary)
        return
    if resolved_format == "json":
        print(json_payload)
        return
    print(human_summary)
    print(json_payload)


def _load_base_config(config_path: Path) -> DictConfig:
    """Load base application config for provider and logging settings."""
    if not config_path.exists():
        raise QABenchmarkRuntimeError(f"Config path does not exist: {config_path}")
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise QABenchmarkRuntimeError("Failed to load benchmark base config")
    return cfg


def _load_qa_config(qa_config_path: Path) -> DictConfig:
    """Load QA benchmark-specific configuration."""
    if not qa_config_path.exists():
        raise QABenchmarkRuntimeError(
            f"QA config path does not exist: {qa_config_path}"
        )
    cfg = OmegaConf.load(qa_config_path)
    if not isinstance(cfg, DictConfig):
        raise QABenchmarkRuntimeError("Failed to load QA benchmark config")
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for QA benchmark execution."""
    parser = argparse.ArgumentParser(description="QA benchmark runner")
    parser.add_argument(
        "--summary-xml",
        required=True,
        help="Path to candidate summary.xml file",
    )
    parser.add_argument(
        "--source-transcript",
        required=True,
        help="Path to source transcript/document file",
    )
    parser.add_argument(
        "--provider-profiles",
        default=str(DEFAULT_PROVIDER_PROFILES_PATH),
        help="Provider profiles yaml path",
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Base application config yaml path",
    )
    parser.add_argument(
        "--provider",
        default="vllm_default",
        help="Default provider profile id for both agents",
    )
    parser.add_argument(
        "--qa-config",
        default=str(DEFAULT_QA_CONFIG_PATH),
        help="QA benchmark-specific config path",
    )
    parser.add_argument(
        "--creator-provider",
        default="",
        help="Optional provider profile id override for GroundTruthCreator",
    )
    parser.add_argument(
        "--evaluator-provider",
        default="",
        help="Optional provider profile id override for Evaluator",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/qa_benchmark",
        help="Output root for QA benchmark artifacts",
    )
    parser.add_argument(
        "--output-format",
        choices=["auto", "human", "json", "both"],
        default="auto",
        help="Terminal output format. 'auto' uses human on TTY and json otherwise.",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Color mode for human terminal output.",
    )
    return parser


def _find_provider_profile(
    profiles: list[ProviderProfile], provider_id: str
) -> ProviderProfile | None:
    """Find provider profile by provider_id."""
    for profile in profiles:
        if profile.provider_id == provider_id:
            return profile
    return None


def _resolve_provider_selection(
    *,
    default_provider_id: str,
    creator_provider_id: str,
    evaluator_provider_id: str,
) -> tuple[str, str, str]:
    """Resolve final provider IDs for creator and evaluator."""
    base_provider = default_provider_id.strip()
    if not base_provider:
        raise QABenchmarkRuntimeError("--provider must be non-empty")
    creator_final = creator_provider_id.strip() or base_provider
    evaluator_final = evaluator_provider_id.strip() or base_provider
    return base_provider, creator_final, evaluator_final


def _instantiate_llm(
    *,
    profile: ProviderProfile,
    base_cfg: DictConfig,
    qa_cfg: DictConfig,
) -> LLMProvider:
    """Instantiate an LLM provider from selected profile."""
    llm_config_payload = resolve_provider_config(profile)
    if (
        str(llm_config_payload.get("_target_", "")).strip()
        == "card_framework.providers.vllm_provider.VLLMProvider"
    ):
        vllm_cfg = qa_cfg.get("vllm", {})
        configured_base_url = str(vllm_cfg.get("base_url", "")).strip()
        configured_api_key = str(vllm_cfg.get("api_key", "")).strip()
        chat_template_kwargs = vllm_cfg.get("chat_template_kwargs")
        if configured_base_url:
            llm_config_payload["base_url"] = configured_base_url
        if configured_api_key:
            llm_config_payload["api_key"] = configured_api_key
        configured_timeout = vllm_cfg.get("timeout_seconds")
        llm_config_payload["request_timeout_seconds"] = as_positive_float(
            raw_value=configured_timeout,
            field_name="vllm.timeout_seconds",
            default_value=30.0,
        )
        if chat_template_kwargs is not None:
            normalized_chat_kwargs = OmegaConf.to_container(
                chat_template_kwargs, resolve=True
            )
            if isinstance(normalized_chat_kwargs, dict):
                # VLLMProvider sends thinking_extra_body only when enable_thinking=True.
                llm_config_payload["enable_thinking"] = True
                llm_config_payload["thinking_extra_body"] = {
                    "chat_template_kwargs": normalized_chat_kwargs
                }

    llm_cfg = OmegaConf.create(llm_config_payload)
    llm: LLMProvider = hydra.utils.instantiate(llm_cfg, _convert_="all")
    if bool(base_cfg.get("logging", {}).get("enabled", False)):
        llm = LoggingLLMProvider(inner_provider=llm)
    return llm


def _instantiate_embedding_provider(base_cfg: DictConfig) -> EmbeddingProvider | None:
    """Instantiate configured embedding provider unless no-op provider is selected."""
    embedding_cfg = base_cfg.get("embedding")
    if embedding_cfg is None:
        return None
    embedding_provider = hydra.utils.instantiate(embedding_cfg, _convert_="all")
    if not isinstance(embedding_provider, EmbeddingProvider):
        raise QABenchmarkRuntimeError(
            "Configured embedding provider does not implement EmbeddingProvider."
        )
    if type(embedding_provider).__name__ == "NoOpEmbeddingProvider":
        return None
    return embedding_provider


def _resolve_evaluator_embedding_provider(
    *,
    base_cfg: DictConfig,
    evaluator_runtime_config: EvaluatorRuntimeConfig,
) -> EmbeddingProvider | None:
    """Resolve optional evaluator embedding provider for semantic quote modes."""
    quote_mode = evaluator_runtime_config.quote_relevance.mode
    if quote_mode not in {"semantic_similarity", "hybrid"}:
        return None
    embedding_provider = _instantiate_embedding_provider(base_cfg)
    if embedding_provider is not None:
        return embedding_provider
    if quote_mode == "semantic_similarity":
        raise QABenchmarkRuntimeError(
            "qa.evaluator.quote_relevance.mode=semantic_similarity requires a "
            "non-noop embedding provider in base config."
        )
    logger.warning(
        "[QABenchmark] hybrid quote relevance mode is configured without a non-noop embedding provider; falling back to lexical checks when semantic fallback is needed."
    )
    return None


def _read_summary_xml(path: Path) -> str:
    """Read candidate summary XML from disk."""
    if not path.exists():
        raise QABenchmarkRuntimeError(f"summary.xml path does not exist: {path}")
    summary_xml = path.read_text(encoding="utf-8").strip()
    if not summary_xml:
        raise QABenchmarkRuntimeError("summary.xml is empty")
    return summary_xml


def _load_source_text(path: Path) -> str:
    """Load source transcript as evidence-indexed plain text."""
    if not path.exists():
        raise QABenchmarkRuntimeError(f"source transcript path does not exist: {path}")
    raw = path.read_text(encoding="utf-8").lstrip("\ufeff").strip()
    if not raw:
        raise QABenchmarkRuntimeError("source transcript is empty")

    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("segments"), list):
            lines: list[str] = []
            evidence_index = 1
            for segment in payload["segments"]:
                if not isinstance(segment, dict):
                    continue
                text = str(segment.get("text", "")).strip()
                if not text:
                    continue
                speaker = str(segment.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
                evidence_id = f"E{evidence_index:04d}"
                lines.append(f"[{evidence_id}] [{speaker}] {text}")
                evidence_index += 1
            if lines:
                return "\n".join(lines)

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        raise QABenchmarkRuntimeError("source transcript has no non-empty lines")
    indexed = [f"[E{index:04d}] {line}" for index, line in enumerate(lines, start=1)]
    return "\n".join(indexed)


async def _run_qa_workflow(
    *,
    creator_llm: LLMProvider,
    evaluator_llm: LLMProvider,
    summary_xml: str,
    source_text: str,
    evaluator_runtime_config: EvaluatorRuntimeConfig,
    evaluator_embedding_provider: EmbeddingProvider | None,
    creator_max_generation_attempts: int,
    creator_request_retries: int,
    evaluator_request_retries: int,
    creator_request_timeout_seconds: float,
    evaluator_request_timeout_seconds: float,
    server_stop_timeout_seconds: float,
    corrector_runtime_config: CorrectorRuntimeConfig | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run creator and evaluator A2A tasks and return raw payloads."""
    servers: list[ManagedServer] = []
    client = AgentClient(event_bus=event_bus)

    resolved_corrector_runtime_config = (
        corrector_runtime_config
        if corrector_runtime_config is not None
        else CorrectorRuntimeConfig(enabled=False, max_tokens=700, max_examples=2)
    )
    creator_port, evaluator_port = _reserve_ports(2)
    creator_corrector_agent: LLMCorrectorAgent | None = None
    evaluator_corrector_agent: LLMCorrectorAgent | None = None
    if resolved_corrector_runtime_config.enabled:
        creator_corrector_agent = LLMCorrectorAgent(
            llm=creator_llm,
            event_bus=event_bus,
            max_tokens=resolved_corrector_runtime_config.max_tokens,
            max_examples=resolved_corrector_runtime_config.max_examples,
        )
        evaluator_corrector_agent = LLMCorrectorAgent(
            llm=evaluator_llm,
            event_bus=event_bus,
            max_tokens=resolved_corrector_runtime_config.max_tokens,
            max_examples=resolved_corrector_runtime_config.max_examples,
        )
    try:
        creator_app = _build_a2a_app(
            name="GroundTruthCreator",
            description="Generates source-grounded QA benchmark questions.",
            port=creator_port,
            executor=GroundTruthCreatorExecutor(
                llm=creator_llm,
                max_generation_attempts=creator_max_generation_attempts,
                event_bus=event_bus,
                corrector_agent=creator_corrector_agent,
            ),
        )
        servers.append(
            _run_server_in_thread("qa-ground-truth-creator", creator_app, creator_port)
        )

        evaluator_app = _build_a2a_app(
            name="Evaluator",
            description="Evaluates candidate summary using QA ground truth.",
            port=evaluator_port,
            executor=QAEvaluatorExecutor(
                llm=evaluator_llm,
                evaluator_runtime_config=evaluator_runtime_config,
                event_bus=event_bus,
                embedding_provider=evaluator_embedding_provider,
                corrector_agent=evaluator_corrector_agent,
            ),
        )
        servers.append(
            _run_server_in_thread("qa-evaluator", evaluator_app, evaluator_port)
        )

        checker = AgentHealthChecker(max_retries=5, base_delay=1.0)
        if not checker.check("GroundTruthCreator", creator_port):
            raise QABenchmarkRuntimeError(
                f"Health check failed for GroundTruthCreator:{creator_port}"
            )
        if not checker.check("Evaluator", evaluator_port):
            raise QABenchmarkRuntimeError(
                f"Health check failed for Evaluator:{evaluator_port}"
            )

        creator_task = GroundTruthCreatorTaskRequest(source_text=source_text)
        logger.info(
            "[QABenchmark] Starting GroundTruthCreator request with timeout=%.1fs retries=%s",
            creator_request_timeout_seconds,
            creator_request_retries,
        )
        creator_payload: dict[str, Any] | None = None
        last_creator_error: Exception | None = None
        for attempt in range(1, creator_request_retries + 1):
            try:
                creator_response_raw = await client.send_task(
                    creator_port,
                    creator_task,
                    timeout=creator_request_timeout_seconds,
                    metadata={"component": "qa_benchmark", "stage": "creator"},
                )
                creator_payload_raw = json.loads(creator_response_raw)
                if not isinstance(creator_payload_raw, dict):
                    raise QABenchmarkRuntimeError(
                        "GroundTruthCreator returned non-object JSON payload."
                    )
                creator_payload = creator_payload_raw
                break
            except Exception as exc:
                last_creator_error = exc
                if attempt >= creator_request_retries:
                    break
                logger.warning(
                    "[QABenchmark] GroundTruthCreator attempt %s/%s failed: %s. Retrying.",
                    attempt,
                    creator_request_retries,
                    exc,
                )
                await asyncio.sleep(float(min(5, attempt)))
        if creator_payload is None:
            raise QABenchmarkRuntimeError(
                "GroundTruthCreator request failed after "
                f"{creator_request_retries} attempts: {last_creator_error}"
            ) from last_creator_error

        question_set = GroundTruthSet.model_validate(creator_payload)
        logger.info(
            "[QABenchmark] Starting per-question evaluator workflow questions=%s concurrency=%s timeout_per_question=%.1fs retries=%s",
            len(question_set.questions),
            evaluator_runtime_config.per_question_concurrency,
            evaluator_request_timeout_seconds,
            evaluator_request_retries,
        )
        evaluator_payload = await evaluate_questions_with_fresh_contexts(
            client=client,
            evaluator_port=evaluator_port,
            summary_xml=summary_xml,
            source_text=source_text,
            question_set=question_set,
            evaluator_request_timeout_seconds=evaluator_request_timeout_seconds,
            per_question_concurrency=evaluator_runtime_config.per_question_concurrency,
            evaluator_request_retries=evaluator_request_retries,
        )

        return creator_payload, evaluator_payload
    finally:
        for server in reversed(servers):
            _stop_server(server, timeout_seconds=server_stop_timeout_seconds)
        await client.close()


def execute_command(args: argparse.Namespace) -> int:
    """Execute QA benchmark and write report artifacts."""
    repo_root = REPO_ROOT
    run_id = _make_run_id()

    config_path = (repo_root / args.config).resolve()
    base_cfg = _load_base_config(config_path)
    qa_config_path = (repo_root / args.qa_config).resolve()
    qa_cfg = _load_qa_config(qa_config_path)
    configure_logger(base_cfg.logging)

    summary_xml_path = (repo_root / args.summary_xml).resolve()
    source_transcript_path = (repo_root / args.source_transcript).resolve()
    provider_profiles_path = (repo_root / args.provider_profiles).resolve()

    summary_xml = _read_summary_xml(summary_xml_path)
    source_text = _load_source_text(source_transcript_path)
    (
        min_overlap_ratio,
        min_shared_tokens,
        min_shared_distinctive_tokens,
        min_shared_name_phrases,
        input_guard_mode,
    ) = resolve_input_guard_config(qa_cfg)
    if input_guard_mode != "off":
        compatibility = evaluate_input_compatibility(
            summary_xml=summary_xml,
            source_text=source_text,
            min_overlap_ratio=min_overlap_ratio,
            min_shared_tokens=min_shared_tokens,
            min_shared_distinctive_tokens=min_shared_distinctive_tokens,
            min_shared_name_phrases=min_shared_name_phrases,
        )
        compatibility_message = (
            f"[QABenchmark] Input guard overlap={compatibility.token_overlap_ratio:.4f} "
            f"shared={compatibility.shared_token_count} "
            f"summary_tokens={compatibility.summary_token_count} "
            f"source_tokens={compatibility.source_token_count} "
            f"distinctive_overlap={compatibility.distinctive_overlap_ratio:.4f} "
            f"shared_distinctive={compatibility.shared_distinctive_count} "
            f"shared_name_phrases={compatibility.shared_name_phrase_count} "
            f"preview={list(compatibility.shared_tokens_preview)[:6]} "
            f"distinctive_preview={list(compatibility.shared_distinctive_preview)[:6]} "
            f"name_phrase_preview={list(compatibility.shared_name_phrase_preview)[:4]}"
        )
        logger.info(compatibility_message)
        if not compatibility.is_compatible:
            if input_guard_mode == "warn":
                event_bus.publish(
                    "status_message",
                    "WARNING: input guard failed, continuing because qa.input_guard.mode=warn.",
                )
            else:
                raise QABenchmarkRuntimeError(
                    compatibility.reason
                    + " Set qa.input_guard.mode=warn or off to bypass."
                )

    all_profiles = load_provider_profiles(provider_profiles_path)
    default_provider, creator_provider_id, evaluator_provider_id = (
        _resolve_provider_selection(
            default_provider_id=str(args.provider),
            creator_provider_id=str(args.creator_provider),
            evaluator_provider_id=str(args.evaluator_provider),
        )
    )
    del default_provider

    creator_profile = _find_provider_profile(all_profiles, creator_provider_id)
    evaluator_profile = _find_provider_profile(all_profiles, evaluator_provider_id)
    if creator_profile is None:
        raise QABenchmarkRuntimeError(
            f"Creator provider '{creator_provider_id}' was not found."
        )
    if evaluator_profile is None:
        raise QABenchmarkRuntimeError(
            f"Evaluator provider '{evaluator_provider_id}' was not found."
        )

    missing_creator_env = missing_required_env(creator_profile)
    if missing_creator_env:
        raise QABenchmarkRuntimeError(
            "Creator provider missing required env vars: "
            + ", ".join(missing_creator_env)
        )
    missing_evaluator_env = missing_required_env(evaluator_profile)
    if missing_evaluator_env:
        raise QABenchmarkRuntimeError(
            "Evaluator provider missing required env vars: "
            + ", ".join(missing_evaluator_env)
        )

    creator_llm = _instantiate_llm(
        profile=creator_profile, base_cfg=base_cfg, qa_cfg=qa_cfg
    )
    evaluator_llm = _instantiate_llm(
        profile=evaluator_profile, base_cfg=base_cfg, qa_cfg=qa_cfg
    )

    evaluator_runtime_config = resolve_evaluator_runtime_config(qa_cfg)
    corrector_runtime_config = resolve_corrector_runtime_config(qa_cfg)
    evaluator_embedding_provider = _resolve_evaluator_embedding_provider(
        base_cfg=base_cfg,
        evaluator_runtime_config=evaluator_runtime_config,
    )
    logger.info(
        "[QABenchmark] Evaluator runtime config max_tool_turns=%s max_attempts=%s chat_max_tokens=%s no_tool_call_patience=%s max_tool_calls_per_turn=%s per_question_concurrency=%s quote_relevance_mode=%s semantic_threshold=%s auto_repair=%s repair_min_score=%.3f min_candidate_chars=%s embedding_provider_enabled=%s",
        evaluator_runtime_config.max_tool_turns,
        evaluator_runtime_config.max_attempts_per_question,
        evaluator_runtime_config.chat_max_tokens,
        evaluator_runtime_config.no_tool_call_patience,
        evaluator_runtime_config.max_tool_calls_per_turn,
        evaluator_runtime_config.per_question_concurrency,
        evaluator_runtime_config.quote_relevance.mode,
        evaluator_runtime_config.quote_relevance.semantic_threshold,
        evaluator_runtime_config.quote_relevance.auto_repair,
        evaluator_runtime_config.quote_relevance.repair_min_score,
        evaluator_runtime_config.quote_relevance.min_candidate_chars,
        evaluator_embedding_provider is not None,
    )
    logger.info(
        "[QABenchmark] Corrector runtime config enabled=%s max_tokens=%s max_examples=%s",
        corrector_runtime_config.enabled,
        corrector_runtime_config.max_tokens,
        corrector_runtime_config.max_examples,
    )
    creator_max_generation_attempts = int(
        qa_cfg.get("qa", {}).get("creator", {}).get("max_generation_attempts", 3)
    )
    creator_request_retries = as_positive_int(
        raw_value=qa_cfg.get("qa", {}).get("creator", {}).get("request_retries"),
        field_name="qa.creator.request_retries",
        default_value=3,
    )
    evaluator_request_retries = as_positive_int(
        raw_value=qa_cfg.get("qa", {}).get("evaluator", {}).get("request_retries"),
        field_name="qa.evaluator.request_retries",
        default_value=3,
    )
    (
        creator_request_timeout_seconds,
        evaluator_request_timeout_seconds,
        server_stop_timeout_seconds,
    ) = resolve_workflow_timeouts(qa_cfg)

    creator_payload, evaluator_payload = asyncio.run(
        _run_qa_workflow(
            creator_llm=creator_llm,
            evaluator_llm=evaluator_llm,
            summary_xml=summary_xml,
            source_text=source_text,
            evaluator_runtime_config=evaluator_runtime_config,
            evaluator_embedding_provider=evaluator_embedding_provider,
            creator_max_generation_attempts=max(1, creator_max_generation_attempts),
            creator_request_retries=creator_request_retries,
            evaluator_request_retries=evaluator_request_retries,
            creator_request_timeout_seconds=creator_request_timeout_seconds,
            evaluator_request_timeout_seconds=evaluator_request_timeout_seconds,
            server_stop_timeout_seconds=server_stop_timeout_seconds,
            corrector_runtime_config=corrector_runtime_config,
        )
    )

    question_set = GroundTruthSet.model_validate(creator_payload)
    answers_raw = evaluator_payload.get("answers", [])
    if not isinstance(answers_raw, list):
        raise QABenchmarkRuntimeError("Evaluator payload missing 'answers' list.")
    answers = [EvaluatorAnswerRecord.model_validate(item) for item in answers_raw]
    try:
        validate_answer_coverage(question_set=question_set, answers=answers)
    except ValueError as exc:
        raise QABenchmarkRuntimeError(
            f"Evaluator answer coverage validation failed: {exc}"
        ) from exc
    score = build_score(answers)

    commit, branch = git_info(repo_root)
    report = QABenchmarkReport(
        run_id=run_id,
        status="completed",
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        git_commit=commit,
        git_branch=branch,
        summary_xml_path=str(summary_xml_path),
        source_transcript_path=str(source_transcript_path),
        provider_id=str(args.provider),
        creator_provider_id=creator_provider_id,
        evaluator_provider_id=evaluator_provider_id,
        score=score,
        answers=answers,
    )

    output_root = (repo_root / args.output_dir / run_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    questions_path = output_root / "qa_question_set.json"
    trace_path = output_root / "qa_trace.json"
    report_path = output_root / "qa_report.json"

    write_json_with_hash(
        questions_path,
        {"questions": [question.model_dump() for question in question_set.questions]},
    )
    write_json_with_hash(
        trace_path, {"answers": [answer.model_dump() for answer in answers]}
    )
    report_sha256 = write_json_with_hash(report_path, report.model_dump())

    verification_payload = build_verification_payload(
        run_id=run_id,
        report_path=report_path,
        report_sha256=report_sha256,
        junit_path=output_root / "pytest.junit.xml",
        junit_sha256="",
        junit_totals={"tests": 0, "failures": 0, "errors": 0, "skipped": 0},
        commands_executed=_commands_executed(),
        git_commit=commit,
        git_branch=branch,
    )
    verification_path = output_root / "verification.json"
    write_json_with_hash(verification_path, verification_payload)

    _emit_success_output(
        output_format=cast(OutputFormat, args.output_format),
        color_mode=cast(ColorMode, args.color),
        run_id=run_id,
        score=score,
        answers=answers,
        report_path=report_path,
        trace_path=trace_path,
        questions_path=questions_path,
        verification_path=verification_path,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Program entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return execute_command(args)
    except (QABenchmarkRuntimeError, MatrixConfigError, QAConfigError) as exc:
        _emit_failure_output(
            output_format=cast(OutputFormat, args.output_format),
            color_mode=cast(ColorMode, args.color),
            error=str(exc),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

