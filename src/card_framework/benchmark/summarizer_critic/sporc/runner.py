"""Execution engine for the SPoRC summarizer-critic benchmark."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import argparse
import asyncio
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import threading
import time
from typing import Any, Callable
from urllib.parse import urlparse

import hydra
import httpx
from omegaconf import OmegaConf
import requests

from card_framework.agents.client import AgentClient
from card_framework.agents.critic import CriticExecutor
from card_framework.agents.health import AgentHealthChecker
from card_framework.agents.retrieval import InfoRetrievalExecutor
from card_framework.agents.summarizer import SummarizerExecutor
from card_framework.agents.utils import count_words
from card_framework.benchmark.artifacts import git_info, utc_now_iso
from card_framework.benchmark.instrumentation import EventCapture
from card_framework.benchmark.matrix import (
    MatrixConfigError,
    load_provider_profiles,
    missing_required_env,
    resolve_provider_config,
)
from card_framework.benchmark.summarizer_critic.qmsum.runner import (
    BenchmarkExecutionError,
    ManagedServer,
    _build_stub_calibration,
    _coerce_mapping,
    _configure_benchmark_logging,
    _find_provider_profile,
    _load_base_config,
    _make_run_id,
    _progress_snapshot,
    _release_reserved_port,
    _resolve_embedding_profile,
    _resolve_benchmark_ports,
    _run_server_in_thread,
    _stop_server,
)
from card_framework.benchmark.summarizer_critic.sampling import (
    select_benchmark_samples,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    BenchmarkExecutionReport,
    BenchmarkSummary,
    DEFAULT_DATASET_MANIFEST_PATH,
    DEFAULT_DATASET_VARIANT,
    DEFAULT_REFERENCE_WPM,
    DURATION_PRESET_SECONDS,
    DEFAULT_RUNS_ROOT,
    PreparedBenchmarkSample,
    SampleBenchmarkResult,
    SampleMetricBundle,
)
from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    DatasetPreparationError,
    ensure_compact_prepared_dataset_manifest,
    iter_prepared_dataset_samples,
    load_prepared_dataset_manifest,
    prepare_sporc_benchmark,
)
from card_framework.benchmark.summarizer_critic.sporc.judge import (
    DIMENSION_NAMES,
    SPoRCJudge,
    SPoRCJudgeConfig,
)
from card_framework.benchmark.summarizer_critic.sporc.reporting import (
    write_execution_artifacts,
)
from card_framework.benchmark.types import (
    EmbeddingProfile,
)
from card_framework.audio_pipeline.factory import (
    build_speaker_sample_generator,
    build_voice_clone_orchestrator,
)
from card_framework.audio_pipeline.live_draft_voice_clone import (
    LiveDraftVoiceCloneSession,
)
from card_framework.audio_pipeline.speaker_samples import (
    resolve_sample_source_audio_path,
)
from card_framework.cli.main import _build_a2a_app
from card_framework.orchestration.transcript import Transcript, TranscriptSegment
from card_framework.providers.logging_provider import LoggingLLMProvider
from card_framework.providers.response_callbacks import (
    CompositeResponseCallback,
    RichConsoleResponseCallback,
)
from card_framework.retrieval.embeddings import TranscriptIndex
from card_framework.runtime.loop_orchestrator import Orchestrator
from card_framework.shared.events import event_bus
from card_framework.shared.llm_provider import (
    EmbeddingProvider,
    LLMProvider,
    LLMResponseCallback,
    normalize_messages,
    normalize_tools,
)
from card_framework.shared.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROVIDER_PROFILES_PATH,
    REPO_ROOT,
)
from card_framework.shared.summary_xml import (
    DEFAULT_EMO_PRESET,
    SummaryTurn,
    parse_summary_xml,
)

_SOURCE_AUDIO_TIMEOUT = (10.0, 300.0)
_SOURCE_AUDIO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "audio/*,*/*;q=0.8",
    "Referer": "https://www.buzzsprout.com/",
}
_SHARED_SAMPLE_QUEUE_DB_NAME = "sample_queue.sqlite3"
_VLLM_PROVIDER_TARGET = "card_framework.providers.vllm_provider.VLLMProvider"
_BENCHMARK_VLLM_REQUEST_TIMEOUT_SECONDS = 60.0
_TRANSCRIPT_ONLY_CRITIC_VLLM_REQUEST_TIMEOUT_SECONDS = 300.0
_PROGRESS_EVENT_KEY = "sporc_progress_event"
_STAGE_TIMEOUT_FINAL_STATUS = "stage_timeout"
_STARTUP_UNAVAILABLE_FINAL_STATUS = "startup_unavailable"
_CONNECTION_UNAVAILABLE_FINAL_STATUS = "connection_unavailable"
_RESUME_FAILURE_DEFAULT_STATUS = "resume_interrupted_sample"
_SWEEP_STALL_FINAL_STATUS = "sweep_watchdog_stalled"
_CHECKPOINT_VERSION = 1
_TRANSCRIPT_ONLY_SPEAKER_MERGE_GAP_MS = 250
_TRANSCRIPT_ONLY_ALNUM_PATTERN = re.compile(r"[A-Za-z0-9]")
_TRANSCRIPT_ONLY_SPEAKER_LABEL_PATTERN = re.compile(r"^SPEAKER_\d+$", re.IGNORECASE)
_REPLACEMENT_ELIGIBLE_FINAL_STATUSES = frozenset(
    {
        "preflight_failed",
        _STAGE_TIMEOUT_FINAL_STATUS,
        _STARTUP_UNAVAILABLE_FINAL_STATUS,
        _CONNECTION_UNAVAILABLE_FINAL_STATUS,
        _RESUME_FAILURE_DEFAULT_STATUS,
        _SWEEP_STALL_FINAL_STATUS,
    }
)
_RETRYABLE_INFRA_FINAL_STATUSES = frozenset(
    {
        _STARTUP_UNAVAILABLE_FINAL_STATUS,
        _CONNECTION_UNAVAILABLE_FINAL_STATUS,
    }
)
_SAMPLE_INFRA_RETRY_LIMIT = 2
_AGENT_CLIENT_MAX_KEEPALIVE_CONNECTIONS = 20
_AGENT_CLIENT_MAX_CONNECTIONS = 100
_AGENT_CLIENT_KEEPALIVE_EXPIRY_SECONDS = 30.0
_AGENT_CLIENT_POOL_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class _ManagedBenchmarkRuntime:
    """Own one SPoRC execute runtime instance and its local A2A servers."""

    orchestrator: Orchestrator
    judge: SPoRCJudge
    event_capture: EventCapture
    artifact_recorder: "_SampleArtifactRecorder"
    retrieval_enabled: bool
    shared_agent_client: AgentClient
    servers: list[ManagedServer]
    retrieval_port: int
    summarizer_port: int
    critic_port: int


@dataclass(slots=True)
class _PendingTraceRecord:
    """Track one in-flight streamed response before it is persisted."""

    role: str
    call_index: int
    agent_name: str
    started_at_utc: str
    completed_at_utc: str | None = None
    reasoning_content: str = ""
    content: str = ""


@dataclass(slots=True)
class _SampleArtifactState:
    """Track one active sample's artifact directories and recorded call files."""

    sample_id: str
    sample_root: Path
    manifest_path: Path
    final_dir: Path
    llm_root: Path
    llm_dirs: dict[str, Path]
    target_seconds: int
    dataset: str
    episode_key: str
    call_counts: dict[str, int]
    recorded_call_paths: dict[str, list[Path]]
    pending_calls: dict[str, _PendingTraceRecord]


@dataclass(slots=True, frozen=True)
class _SampleArtifactPaths:
    """Expose persisted sample artifact locations back to report payloads."""

    sample_artifact_dir: str
    artifact_manifest_path: str
    summarizer_trace_dir: str
    critic_trace_dir: str
    judge_trace_dir: str
    final_summary_path: str | None
    final_critic_feedback_path: str | None
    loop_diagnostics_path: str | None
    sample_result_artifact_path: str


def _json_default(value: object) -> object:
    """Serialize model-like objects safely for artifact JSON dumps."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return str(value)


def _serialize_provider_response(response_message: Any) -> dict[str, Any]:
    """Return a stable JSON-friendly response payload for trace artifacts."""
    if isinstance(response_message, dict):
        return deepcopy(response_message)
    serialized: dict[str, Any] = {
        "role": getattr(response_message, "role", "assistant"),
        "content": getattr(response_message, "content", ""),
    }
    reasoning_content = getattr(response_message, "reasoning_content", None)
    if reasoning_content is None:
        reasoning_content = getattr(response_message, "reasoning", None)
    if reasoning_content is not None:
        serialized["reasoning_content"] = reasoning_content
    tool_calls = getattr(response_message, "tool_calls", None)
    if tool_calls:
        serialized["tool_calls"] = json.loads(
            json.dumps(tool_calls, default=_json_default)
        )
    return serialized


def _attach_response_callback(
    llm: LLMProvider,
    callback: LLMResponseCallback,
) -> bool:
    """Attach a streaming callback to a provider or wrapped inner provider."""
    callback_setter = getattr(llm, "set_response_callback", None)
    if callable(callback_setter):
        callback_setter(callback)
        return True

    inner_provider = getattr(llm, "inner_provider", None)
    if isinstance(inner_provider, LLMProvider):
        return _attach_response_callback(inner_provider, callback)
    return False


class _SampleArtifactRecorder:
    """Persist per-sample trace and final-output artifacts for one execute run."""

    def __init__(self, *, run_root: Path) -> None:
        self._run_root = run_root
        self._samples_root = (run_root / "samples").resolve()
        self._lock = threading.RLock()
        self._active_sample: _SampleArtifactState | None = None
        self._samples_root.mkdir(parents=True, exist_ok=True)

    @property
    def samples_root(self) -> Path:
        """Return the root directory that contains per-sample artifacts."""
        return self._samples_root

    def activate_sample(
        self,
        *,
        sample: PreparedBenchmarkSample,
        target_seconds: int,
    ) -> None:
        """Create and register the per-sample artifact tree for a run sample."""
        sample_root = (self._samples_root / sample.sample_id).resolve()
        final_dir = (sample_root / "final").resolve()
        llm_root = (sample_root / "llm").resolve()
        llm_dirs = {
            role: (llm_root / role).resolve()
            for role in ("summarizer", "critic", "judge")
        }
        for directory in (sample_root, final_dir, llm_root, *llm_dirs.values()):
            directory.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._active_sample = _SampleArtifactState(
                sample_id=sample.sample_id,
                sample_root=sample_root,
                manifest_path=(sample_root / "manifest.json").resolve(),
                final_dir=final_dir,
                llm_root=llm_root,
                llm_dirs=llm_dirs,
                target_seconds=target_seconds,
                dataset=sample.dataset,
                episode_key=sample.episode_key,
                call_counts=defaultdict(int),
                recorded_call_paths=defaultdict(list),
                pending_calls={},
            )

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        """Write one JSON artifact with stable formatting."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )

    def begin_stream(self, *, role: str, agent_name: str) -> None:
        """Start buffering one streamed response for the active sample."""
        with self._lock:
            if self._active_sample is None:
                return
            state = self._active_sample
            state.call_counts[role] += 1
            state.pending_calls[role] = _PendingTraceRecord(
                role=role,
                call_index=state.call_counts[role],
                agent_name=agent_name,
                started_at_utc=utc_now_iso(),
            )

    def append_reasoning(self, *, role: str, token: str) -> None:
        """Append one streamed reasoning token for the active sample."""
        if not token:
            return
        with self._lock:
            if self._active_sample is None:
                return
            pending = self._active_sample.pending_calls.get(role)
            if pending is not None:
                pending.reasoning_content += token

    def append_content(self, *, role: str, token: str) -> None:
        """Append one streamed content token for the active sample."""
        if not token:
            return
        with self._lock:
            if self._active_sample is None:
                return
            pending = self._active_sample.pending_calls.get(role)
            if pending is not None:
                pending.content += token

    def complete_stream(self, *, role: str) -> None:
        """Mark one streamed response complete for the active sample."""
        with self._lock:
            if self._active_sample is None:
                return
            pending = self._active_sample.pending_calls.get(role)
            if pending is not None:
                pending.completed_at_utc = utc_now_iso()

    def record_call(self, *, role: str, payload: dict[str, Any]) -> None:
        """Persist one finalized provider request/response trace."""
        with self._lock:
            if self._active_sample is None:
                return
            state = self._active_sample
            pending = state.pending_calls.pop(role, None)
            if pending is None:
                state.call_counts[role] += 1
                pending = _PendingTraceRecord(
                    role=role,
                    call_index=state.call_counts[role],
                    agent_name=str(payload.get("agent_name") or role.title()),
                    started_at_utc=utc_now_iso(),
                    completed_at_utc=utc_now_iso(),
                )
            elif pending.completed_at_utc is None:
                pending.completed_at_utc = utc_now_iso()

            call_path = (
                state.llm_dirs[role] / f"call_{pending.call_index:03d}.json"
            ).resolve()
            self._write_json(
                call_path,
                {
                    "sample_id": state.sample_id,
                    "dataset": state.dataset,
                    "episode_key": state.episode_key,
                    "target_seconds": state.target_seconds,
                    "role": role,
                    "call_index": pending.call_index,
                    "agent_name": pending.agent_name,
                    "stream_started_at_utc": pending.started_at_utc,
                    "stream_completed_at_utc": pending.completed_at_utc,
                    "streamed_reasoning_content": pending.reasoning_content,
                    "streamed_content": pending.content,
                    **payload,
                },
            )
            state.recorded_call_paths[role].append(call_path)

    def _active_paths(self) -> _SampleArtifactPaths | None:
        """Return current sample artifact paths without mutating state."""
        with self._lock:
            if self._active_sample is None:
                return None
            state = self._active_sample
            return _SampleArtifactPaths(
                sample_artifact_dir=str(state.sample_root),
                artifact_manifest_path=str(state.manifest_path),
                summarizer_trace_dir=str(state.llm_dirs["summarizer"]),
                critic_trace_dir=str(state.llm_dirs["critic"]),
                judge_trace_dir=str(state.llm_dirs["judge"]),
                final_summary_path=None,
                final_critic_feedback_path=None,
                loop_diagnostics_path=None,
                sample_result_artifact_path=str(
                    (state.final_dir / "sample_result.json").resolve()
                ),
            )

    def finalize_sample(
        self,
        *,
        result: SampleBenchmarkResult,
        diagnostics: dict[str, Any] | None,
        candidate_summary: str,
    ) -> _SampleArtifactPaths | None:
        """Write final sample artifacts and return their resolved locations."""
        with self._lock:
            if self._active_sample is None:
                return None
            state = self._active_sample
            final_summary_path: Path | None = None
            critic_feedback_path: Path | None = None
            loop_diagnostics_path: Path | None = None
            sample_result_path = (state.final_dir / "sample_result.json").resolve()

            if candidate_summary.strip():
                final_summary_path = (state.final_dir / "summary.xml").resolve()
                final_summary_path.write_text(candidate_summary, encoding="utf-8")

            if diagnostics is not None:
                loop_diagnostics_path = (
                    state.final_dir / "loop_diagnostics.json"
                ).resolve()
                self._write_json(loop_diagnostics_path, diagnostics)
                critic_feedback = str(diagnostics.get("final_feedback", "") or "").strip()
                if critic_feedback:
                    critic_feedback_path = (
                        state.final_dir / "critic_feedback.txt"
                    ).resolve()
                    critic_feedback_path.write_text(critic_feedback, encoding="utf-8")

            artifact_paths = _SampleArtifactPaths(
                sample_artifact_dir=str(state.sample_root),
                artifact_manifest_path=str(state.manifest_path),
                summarizer_trace_dir=str(state.llm_dirs["summarizer"]),
                critic_trace_dir=str(state.llm_dirs["critic"]),
                judge_trace_dir=str(state.llm_dirs["judge"]),
                final_summary_path=(
                    str(final_summary_path) if final_summary_path is not None else None
                ),
                final_critic_feedback_path=(
                    str(critic_feedback_path)
                    if critic_feedback_path is not None
                    else None
                ),
                loop_diagnostics_path=(
                    str(loop_diagnostics_path)
                    if loop_diagnostics_path is not None
                    else None
                ),
                sample_result_artifact_path=str(sample_result_path),
            )
            result_with_artifacts = replace(
                result,
                sample_artifact_dir=artifact_paths.sample_artifact_dir,
                artifact_manifest_path=artifact_paths.artifact_manifest_path,
                summarizer_trace_dir=artifact_paths.summarizer_trace_dir,
                critic_trace_dir=artifact_paths.critic_trace_dir,
                judge_trace_dir=artifact_paths.judge_trace_dir,
                final_summary_path=artifact_paths.final_summary_path,
                final_critic_feedback_path=artifact_paths.final_critic_feedback_path,
                loop_diagnostics_path=artifact_paths.loop_diagnostics_path,
                sample_result_artifact_path=artifact_paths.sample_result_artifact_path,
            )
            self._write_json(sample_result_path, asdict(result_with_artifacts))
            self._write_json(
                state.manifest_path,
                {
                    "sample_id": state.sample_id,
                    "dataset": state.dataset,
                    "episode_key": state.episode_key,
                    "target_seconds": state.target_seconds,
                    "sample_artifact_dir": str(state.sample_root),
                    "final": {
                        "summary_xml": (
                            str(final_summary_path) if final_summary_path is not None else ""
                        ),
                        "critic_feedback": (
                            str(critic_feedback_path)
                            if critic_feedback_path is not None
                            else ""
                        ),
                        "loop_diagnostics": (
                            str(loop_diagnostics_path)
                            if loop_diagnostics_path is not None
                            else ""
                        ),
                        "sample_result": str(sample_result_path),
                    },
                    "llm_traces": {
                        role: {
                            "directory": str(state.llm_dirs[role]),
                            "calls": [str(path) for path in state.recorded_call_paths[role]],
                        }
                        for role in ("summarizer", "critic", "judge")
                    },
                },
            )
            return artifact_paths

    def clear_active_sample(self) -> None:
        """Drop any active sample context after sample completion."""
        with self._lock:
            self._active_sample = None


class _ArtifactCaptureResponseCallback(LLMResponseCallback):
    """Buffer streamed reasoning/content so per-call artifacts keep raw text."""

    def __init__(self, *, recorder: _SampleArtifactRecorder, role: str) -> None:
        self._recorder = recorder
        self._role = role

    def on_start(self, agent_name: str) -> None:
        self._recorder.begin_stream(role=self._role, agent_name=agent_name)

    def on_thought_token(self, token: str) -> None:
        self._recorder.append_reasoning(role=self._role, token=token)

    def on_content_token(self, token: str) -> None:
        self._recorder.append_content(role=self._role, token=token)

    def on_complete(self) -> None:
        self._recorder.complete_stream(role=self._role)


class _ArtifactCaptureLLMProvider(LLMProvider):
    """Wrap an LLM provider and persist structured request/response trace payloads."""

    def __init__(
        self,
        *,
        inner_provider: LLMProvider,
        recorder: _SampleArtifactRecorder,
        role: str,
    ) -> None:
        self.inner_provider = inner_provider
        self._recorder = recorder
        self._role = role

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        started = time.perf_counter()
        try:
            response = self.inner_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            self._recorder.record_call(
                role=self._role,
                payload={
                    "operation": "generate",
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "max_tokens": max_tokens,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "error": str(exc),
                },
            )
            raise

        self._recorder.record_call(
            role=self._role,
            payload={
                "operation": "generate",
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "response_text": response,
            },
        )
        return response

    def chat(
        self,
        messages: list[dict[str, Any]] | tuple[dict[str, Any], ...] | Any,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> Any:
        started = time.perf_counter()
        normalized_messages = normalize_messages(messages)
        normalized_tools = normalize_tools(tools)
        try:
            response = self.inner_provider.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            self._recorder.record_call(
                role=self._role,
                payload={
                    "operation": "chat",
                    "messages": normalized_messages,
                    "tools": normalized_tools,
                    "tool_choice": tool_choice,
                    "max_tokens": max_tokens,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "error": str(exc),
                },
            )
            raise

        self._recorder.record_call(
            role=self._role,
            payload={
                "operation": "chat",
                "messages": normalized_messages,
                "tools": normalized_tools,
                "tool_choice": tool_choice,
                "max_tokens": max_tokens,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "response": _serialize_provider_response(response),
            },
        )
        return response


class _BenchmarkProgressTracker:
    """Track active-sample liveness for heartbeats and checkpoints."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        duration_preset: str,
        persist_callback: Callable[[], None] | None,
    ) -> None:
        self._args = args
        self._duration_preset = duration_preset
        self._persist_callback = persist_callback
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._active_sample: dict[str, Any] | None = None
        self._subscriptions: list[tuple[str, Callable[..., Any]]] = []

    def subscribe(self) -> None:
        """Subscribe to runtime events that indicate benchmark activity."""
        subscriptions = [
            ("a2a_call_started", self._on_a2a_call_started),
            ("a2a_call_retry", self._on_a2a_activity),
            ("a2a_call_succeeded", self._on_a2a_activity),
            ("orchestrator_iteration_started", self._on_iteration_started),
            ("tool_invocation", self._on_activity),
            ("status_message", self._on_activity),
        ]
        for event_name, callback in subscriptions:
            event_bus.subscribe(event_name, callback)
            self._subscriptions.append((event_name, callback))

    def unsubscribe(self) -> None:
        """Unsubscribe from any registered runtime events."""
        for event_name, callback in self._subscriptions:
            event_bus.unsubscribe(event_name, callback)
        self._subscriptions.clear()

    def start(self) -> None:
        """Start periodic heartbeat emission when enabled."""
        if not bool(getattr(self._args, "emit_progress_heartbeats", False)):
            return
        if self._heartbeat_thread is not None:
            return
        interval_seconds = max(
            1.0,
            float(getattr(self._args, "progress_heartbeat_seconds", 30.0)),
        )

        def _pump() -> None:
            while not self._stop_event.wait(interval_seconds):
                payload = self.snapshot_heartbeat_payload()
                if payload is None:
                    continue
                _emit_progress_event(
                    self._args,
                    event="sample_heartbeat",
                    duration_preset=self._duration_preset,
                    payload=payload,
                )
                self._persist()

        self._heartbeat_thread = threading.Thread(
            target=_pump,
            name=f"sporc-progress-heartbeat-{self._duration_preset}",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def stop(self) -> None:
        """Stop heartbeat emission."""
        self._stop_event.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None

    def activate_sample(
        self,
        *,
        sample_index: int,
        sample_count: int,
        sample_id: str,
        target_seconds: int,
    ) -> None:
        """Begin tracking one active sample."""
        now_monotonic = time.monotonic()
        now_utc = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._active_sample = {
                "sample_index": sample_index,
                "sample_count": sample_count,
                "sample_id": sample_id,
                "target_seconds": target_seconds,
                "stage": "sample_start",
                "iteration": None,
                "started_at_utc": now_utc,
                "started_monotonic": now_monotonic,
                "last_activity_utc": now_utc,
                "last_activity_monotonic": now_monotonic,
                "elapsed_sample_seconds": 0.0,
                "last_activity_age_seconds": 0.0,
            }
        self._persist()

    def clear_active_sample(self) -> None:
        """Clear the currently active sample after a terminal result."""
        with self._lock:
            self._active_sample = None
        self._persist()

    def active_sample_snapshot(self) -> dict[str, Any] | None:
        """Return a JSON-serializable active-sample snapshot for checkpoints."""
        with self._lock:
            if self._active_sample is None:
                return None
            return self._snapshot_locked()

    def snapshot_heartbeat_payload(self) -> dict[str, Any] | None:
        """Return the payload for one machine-readable heartbeat event."""
        with self._lock:
            if self._active_sample is None:
                return None
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict[str, Any]:
        assert self._active_sample is not None
        active_sample = dict(self._active_sample)
        now_monotonic = time.monotonic()
        started_monotonic = float(active_sample.pop("started_monotonic", now_monotonic))
        last_activity_monotonic = float(
            active_sample.pop("last_activity_monotonic", now_monotonic)
        )
        elapsed_sample_seconds = max(0.0, now_monotonic - started_monotonic)
        last_activity_age_seconds = max(0.0, now_monotonic - last_activity_monotonic)
        active_sample["elapsed_sample_seconds"] = round(elapsed_sample_seconds, 3)
        active_sample["last_activity_age_seconds"] = round(
            last_activity_age_seconds,
            3,
        )
        return active_sample

    def _mark_activity(
        self,
        *,
        stage: str | None = None,
        iteration: int | None = None,
    ) -> None:
        with self._lock:
            if self._active_sample is None:
                return
            now_monotonic = time.monotonic()
            self._active_sample["last_activity_monotonic"] = now_monotonic
            self._active_sample["last_activity_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            if stage is not None:
                self._active_sample["stage"] = stage
            if iteration is not None:
                self._active_sample["iteration"] = int(iteration)
        self._persist()

    def _persist(self) -> None:
        if self._persist_callback is None:
            return
        self._persist_callback()

    def _on_a2a_call_started(
        self,
        call_id: str,
        port: int,
        timeout: float,
        max_retries: int,
        **metadata: Any,
    ) -> None:
        del call_id, port, timeout, max_retries
        raw_stage = metadata.get("stage")
        raw_iteration = metadata.get("iteration")
        stage = str(raw_stage) if isinstance(raw_stage, str) and raw_stage else None
        iteration = int(raw_iteration) if isinstance(raw_iteration, int) else None
        self._mark_activity(stage=stage, iteration=iteration)

    def _on_iteration_started(
        self,
        iteration: int,
        max_iterations: int | None = None,
    ) -> None:
        del max_iterations
        self._mark_activity(stage="summarizer", iteration=iteration)

    def _on_a2a_activity(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._mark_activity()

    def _on_activity(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._mark_activity()


def _resolve_orchestrator_timeouts(args: argparse.Namespace) -> dict[str, float | None]:
    """Resolve SPoRC timeout overrides with no-timeout defaults."""
    resolved: dict[str, float | None] = {}
    for stage in ("summarizer", "critic", "retrieval", "index"):
        resolved[stage] = None
    overrides = {
        "summarizer": args.summarizer_timeout_seconds,
        "critic": args.critic_timeout_seconds,
        "retrieval": args.retrieval_timeout_seconds,
        "index": args.index_timeout_seconds,
    }
    for stage, raw_value in overrides.items():
        if raw_value is None:
            continue
        if float(raw_value) > 0:
            resolved[stage] = float(raw_value)
    return resolved


def _normalize_max_iterations(raw_value: int | None) -> int | None:
    if raw_value is None:
        return None
    if int(raw_value) <= 0:
        return None
    return int(raw_value)


def _emit_progress_event(
    args: argparse.Namespace,
    *,
    event: str,
    duration_preset: str,
    payload: dict[str, Any],
) -> None:
    """Emit one machine-readable progress event for sweep supervisors."""
    if not bool(getattr(args, "emit_progress_events", False)):
        return
    event_payload = {
        _PROGRESS_EVENT_KEY: event,
        "duration_preset": duration_preset,
        **payload,
    }
    print(json.dumps(event_payload), flush=True)


def _deserialize_sample_result(raw_payload: dict[str, Any]) -> SampleBenchmarkResult:
    """Rebuild one sample result from JSON-compatible checkpoint payload."""
    sample_payload = dict(raw_payload)
    raw_metrics = sample_payload.pop("metrics", None) or {}
    if not isinstance(raw_metrics, dict):
        raw_metrics = {}
    result = SampleBenchmarkResult(
        metrics=SampleMetricBundle(
            factualness=float(raw_metrics.get("factualness", 0.0) or 0.0),
            naturalness=float(raw_metrics.get("naturalness", 0.0) or 0.0),
            speaker_grammar_similarity=float(
                raw_metrics.get("speaker_grammar_similarity", 0.0) or 0.0
            ),
            composite=float(raw_metrics.get("composite", 0.0) or 0.0),
        ),
        **sample_payload,
    )
    if "counts_toward_target" in raw_payload:
        return result
    return _finalize_result_target_membership(result)


def _load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    """Load a persisted execute checkpoint from disk."""
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BenchmarkExecutionError(
            f"Invalid SPoRC checkpoint payload in {checkpoint_path}."
        )
    checkpoint_version = int(payload.get("checkpoint_version", 0) or 0)
    if checkpoint_version != _CHECKPOINT_VERSION:
        raise BenchmarkExecutionError(
            f"Unsupported SPoRC checkpoint version {checkpoint_version} in {checkpoint_path}."
        )
    return payload


def _write_checkpoint_payload(
    checkpoint_path: Path,
    payload: dict[str, Any],
) -> None:
    """Atomically persist one execute checkpoint payload."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix(f"{checkpoint_path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(checkpoint_path)


def _classify_sample_failure(exc: Exception) -> tuple[str, str]:
    """Map one sample exception to a stable terminal status and message."""
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if (
        "health check failed" in lowered
        or "server not responding" in lowered
        or "state=failed" in lowered
    ):
        return _STARTUP_UNAVAILABLE_FINAL_STATUS, message
    if (
        "pooltimeout" in lowered
        or "connecterror" in lowered
        or "connecttimeout" in lowered
        or "connection refused" in lowered
        or "all connection attempts failed" in lowered
        or "nodename nor servname provided" in lowered
    ):
        return _CONNECTION_UNAVAILABLE_FINAL_STATUS, message
    if "timeout=" in lowered or "readtimeout" in lowered or "writetimeout" in lowered:
        return _STAGE_TIMEOUT_FINAL_STATUS, message
    if "timed out" in lowered or "timeout" in lowered:
        return _STAGE_TIMEOUT_FINAL_STATUS, message
    return "execution_error", message


def _build_failed_result(
    *,
    sample: PreparedBenchmarkSample,
    target_seconds: int,
    final_status: str,
    error_message: str,
    duration_seconds: float,
    retrieval_events: int = 0,
    tool_invocations: int = 0,
) -> SampleBenchmarkResult:
    """Build one failed result outside the normal success path."""
    counts_toward_target = _counts_toward_target_for_status(final_status)
    return SampleBenchmarkResult(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        episode_key=sample.episode_key,
        status="failed",
        target_seconds=target_seconds,
        candidate_seconds=0.0,
        converged=False,
        iterations_run=0,
        final_status=final_status,
        duration_seconds=duration_seconds,
        retrieval_events=retrieval_events,
        tool_invocations=tool_invocations,
        metrics=SampleMetricBundle(0.0, 0.0, 0.0, 0.0),
        error_message=error_message,
        counts_toward_target=counts_toward_target,
        replacement_reason=(
            final_status if not counts_toward_target else None
        ),
    )


def _counts_toward_target_for_status(final_status: str) -> bool:
    """Return whether one terminal status should consume a retained sample slot."""
    return final_status not in _REPLACEMENT_ELIGIBLE_FINAL_STATUSES


def _count_target_results(
    results_by_sample_id: dict[str, SampleBenchmarkResult],
) -> int:
    """Count results that consume the retained benchmark target."""
    return sum(
        1
        for result in results_by_sample_id.values()
        if bool(result.counts_toward_target)
    )


def _count_attempted_candidates(
    results_by_sample_id: dict[str, SampleBenchmarkResult],
) -> int:
    """Count all candidates that reached a terminal recorded result."""
    return len(results_by_sample_id)


def _build_resume_failure_result(
    *,
    sample: PreparedBenchmarkSample,
    target_seconds: int,
    reason: str,
    active_sample: dict[str, Any],
) -> SampleBenchmarkResult:
    """Build a failed result for an orphaned active sample during resume."""
    stage = str(active_sample.get("stage") or "unknown")
    raw_iteration = active_sample.get("iteration")
    iteration_text = (
        f" iteration={int(raw_iteration)}" if isinstance(raw_iteration, int) else ""
    )
    elapsed_seconds = float(active_sample.get("elapsed_sample_seconds", 0.0) or 0.0)
    return SampleBenchmarkResult(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        episode_key=sample.episode_key,
        status="failed",
        target_seconds=target_seconds,
        candidate_seconds=0.0,
        converged=False,
        iterations_run=int(raw_iteration) if isinstance(raw_iteration, int) else 0,
        final_status=reason,
        duration_seconds=round(elapsed_seconds, 3),
        retrieval_events=0,
        tool_invocations=0,
        metrics=SampleMetricBundle(0.0, 0.0, 0.0, 0.0),
        error_message=(
            "Marked failed while resuming from checkpoint after the previous child "
            f"stopped during stage={stage}{iteration_text}."
        ),
        counts_toward_target=False,
        replacement_reason=reason,
    )


def _ordered_results(
    sample_order: list[str],
    results_by_sample_id: dict[str, SampleBenchmarkResult],
) -> list[SampleBenchmarkResult]:
    """Return results in deterministic sample-order sequence."""
    return [
        results_by_sample_id[sample_id]
        for sample_id in sample_order
        if sample_id in results_by_sample_id
    ]


def _retained_results(
    results: list[SampleBenchmarkResult],
) -> list[SampleBenchmarkResult]:
    """Return the sample results that count toward the retained cohort."""
    return [result for result in results if result.counts_toward_target]


def _replacement_excluded_results(
    results: list[SampleBenchmarkResult],
) -> list[SampleBenchmarkResult]:
    """Return attempted candidates that were excluded and backfilled."""
    return [result for result in results if not result.counts_toward_target]


def _retained_result_count(
    results_by_sample_id: dict[str, SampleBenchmarkResult],
) -> int:
    """Count results that consume one retained benchmark slot."""
    return sum(
        1 for result in results_by_sample_id.values() if result.counts_toward_target
    )


def _counts_toward_target(result: SampleBenchmarkResult) -> bool:
    """Return whether one result belongs in the retained benchmark cohort."""
    return _counts_toward_target_for_status(result.final_status)


def _finalize_result_target_membership(
    result: SampleBenchmarkResult,
    *,
    infra_retry_attempts: int = 0,
) -> SampleBenchmarkResult:
    """Apply retained-cohort membership and retry metadata to one result."""
    counts_toward_target = _counts_toward_target(result)
    replacement_reason = None if counts_toward_target else result.final_status
    return replace(
        result,
        counts_toward_target=counts_toward_target,
        replacement_reason=replacement_reason,
        infra_retry_attempts=infra_retry_attempts,
    )


def _is_retryable_infra_result(result: SampleBenchmarkResult) -> bool:
    """Return whether one failed result deserves same-sample infra retry."""
    return result.final_status in _RETRYABLE_INFRA_FINAL_STATUSES


def _prepare_provider_config(
    profile: Any,
    *,
    role: str = "",
    transcript_only: bool = False,
) -> dict[str, Any]:
    """Resolve provider config, applying SPoRC benchmark-safe vLLM overrides."""
    resolved = resolve_provider_config(profile)
    if str(resolved.get("_target_", "")).strip() != _VLLM_PROVIDER_TARGET:
        return resolved
    prepared = dict(resolved)
    prepared["request_timeout_seconds"] = max(
        float(prepared.get("request_timeout_seconds", 0.0) or 0.0),
        _BENCHMARK_VLLM_REQUEST_TIMEOUT_SECONDS,
    )
    if role == "critic" and transcript_only:
        prepared["request_timeout_seconds"] = max(
            float(prepared.get("request_timeout_seconds", 0.0) or 0.0),
            _TRANSCRIPT_ONLY_CRITIC_VLLM_REQUEST_TIMEOUT_SECONDS,
        )
        # Qwen-backed transcript-only critic prompts inline the full transcript,
        # which can make thinking-mode tool selection stall long enough to hit
        # request timeouts before the first tool call arrives. Keep the vLLM
        # extra_body path but disable chat-template thinking explicitly.
        prepared["enable_thinking"] = True
        prepared["thinking_extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": False}
        }
        return prepared
    if role != "judge":
        return prepared
    # Some Qwen-backed vLLM judge prompts return empty content unless the chat
    # template disables thinking explicitly while still sending extra_body.
    # See repo memory notes dated 2026-03-12 and 2026-03-13.
    prepared["enable_thinking"] = True
    prepared["thinking_extra_body"] = {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    return prepared


def _resolve_override_target_seconds(args: argparse.Namespace) -> int | None:
    """Resolve a CLI-level target override for every sample in the run."""
    raw_preset = str(getattr(args, "duration_preset", "") or "").strip()
    raw_override = getattr(args, "target_seconds_override", None)
    if raw_preset and raw_override is not None:
        raise BenchmarkExecutionError(
            "Use either --duration-preset or --target-seconds-override, not both."
        )
    if raw_preset:
        resolved = DURATION_PRESET_SECONDS.get(raw_preset)
        if resolved is None:
            raise BenchmarkExecutionError(f"Unsupported duration preset: {raw_preset}")
        return int(resolved)
    if raw_override is None:
        return None
    if int(raw_override) <= 0:
        raise BenchmarkExecutionError("--target-seconds-override must be positive.")
    return int(raw_override)


def _resolve_sample_target_seconds(
    sample: PreparedBenchmarkSample,
    args: argparse.Namespace,
) -> int:
    """Resolve the effective target duration for one sample."""
    override_target_seconds = _resolve_override_target_seconds(args)
    if override_target_seconds is not None:
        return override_target_seconds
    return int(sample.target_seconds)


def _select_samples(
    samples: tuple[PreparedBenchmarkSample, ...],
    *,
    max_samples: int,
    sample_seed: int | None,
) -> list[PreparedBenchmarkSample]:
    """Select SPoRC samples in manifest or seeded-random order."""
    return select_benchmark_samples(
        samples,
        max_samples=max_samples,
        sample_seed=sample_seed,
        sample_id_getter=lambda sample: sample.sample_id,
    )


def _transcript_word_count(transcript: Transcript) -> int:
    """Count transcript words from normalized transcript segments."""
    return sum(count_words(segment.text) for segment in transcript.segments)


def _timestamp_span_seconds(transcript: Transcript) -> float:
    """Compute transcript duration from segment timestamps when available."""
    if not transcript.segments:
        return 0.0
    start_time = min(segment.start_time for segment in transcript.segments)
    end_time = max(segment.end_time for segment in transcript.segments)
    if end_time <= start_time:
        return 0.0
    return max(0.0, (end_time - start_time) / 1000.0)


def _estimated_source_seconds_from_words(
    *,
    transcript_word_count: int,
    reference_wpm: float,
) -> float:
    """Estimate source duration from transcript length at the reference speech rate."""
    effective_reference_wpm = (
        float(reference_wpm) if float(reference_wpm) > 0 else DEFAULT_REFERENCE_WPM
    )
    if transcript_word_count <= 0:
        return 0.0
    return (float(transcript_word_count) / effective_reference_wpm) * 60.0


def _evaluate_sample_preflight(
    *,
    sample: PreparedBenchmarkSample,
    target_seconds: int,
    reference_wpm: float,
) -> tuple[Transcript, dict[str, int | float | str], str | None]:
    """Evaluate whether a prepared sample can support the requested target."""
    source_payload = json.loads(
        Path(sample.transcript_path).read_text(encoding="utf-8")
    )
    transcript = Transcript.from_mapping(source_payload)
    transcript_word_count = _transcript_word_count(transcript)
    timestamp_span_seconds = _timestamp_span_seconds(transcript)
    estimated_source_seconds_from_words = _estimated_source_seconds_from_words(
        transcript_word_count=transcript_word_count,
        reference_wpm=reference_wpm,
    )
    max_feasible_summary_seconds = max(
        timestamp_span_seconds,
        estimated_source_seconds_from_words,
    )
    stats: dict[str, int | float | str] = {
        "segment_count": len(transcript.segments),
        "transcript_word_count": transcript_word_count,
        "timestamp_span_seconds": round(timestamp_span_seconds, 3),
        "estimated_source_seconds_from_words": round(
            estimated_source_seconds_from_words,
            3,
        ),
        "max_feasible_summary_seconds": round(max_feasible_summary_seconds, 3),
    }
    if not transcript.segments:
        return transcript, stats, "Transcript has no usable segments."
    if target_seconds > max_feasible_summary_seconds:
        return (
            transcript,
            stats,
            (
                f"Requested target {target_seconds}s exceeds transcript capacity "
                f"({max_feasible_summary_seconds:.1f}s max feasible from "
                f"{len(transcript.segments)} segments / {transcript_word_count} words). "
                "Use a shorter target or a curated long-form manifest."
            ),
        )
    return transcript, stats, None


def _build_preflight_failed_result(
    *,
    sample: PreparedBenchmarkSample,
    target_seconds: int,
    reason: str,
    stats: dict[str, int | float | str],
) -> SampleBenchmarkResult:
    """Build a skipped result for a sample that fails execution preflight."""
    return SampleBenchmarkResult(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        episode_key=sample.episode_key,
        status="skipped",
        target_seconds=target_seconds,
        candidate_seconds=0.0,
        converged=False,
        iterations_run=0,
        final_status="preflight_failed",
        duration_seconds=0.0,
        retrieval_events=0,
        tool_invocations=0,
        metrics=SampleMetricBundle(0.0, 0.0, 0.0, 0.0),
        error_message=reason,
        preflight_reason=reason,
        preflight_stats=dict(stats),
        counts_toward_target=False,
        replacement_reason="preflight_failed",
    )


def _summarize_results(
    results: list[SampleBenchmarkResult],
    *,
    target_sample_count: int,
) -> BenchmarkSummary:
    """Summarize retained results while keeping attempted candidates auditable."""
    retained_results = _retained_results(results)
    executed_results = [
        result for result in retained_results if result.status != "skipped"
    ]
    successful_results = [
        result for result in executed_results if result.status == "completed"
    ]
    failed_results = [
        result for result in executed_results if result.status == "failed"
    ]
    skipped_results = [
        result for result in retained_results if result.status == "skipped"
    ]
    replacement_results = _replacement_excluded_results(results)
    return BenchmarkSummary(
        target_sample_count=target_sample_count,
        sample_count=len(retained_results),
        attempted_candidate_count=len(results),
        replacement_count=len(replacement_results),
        executed_count=len(executed_results),
        skipped_count=len(skipped_results),
        success_count=len(successful_results),
        failure_count=len(failed_results),
        mean_factualness=(
            sum(result.metrics.factualness for result in executed_results)
            / len(executed_results)
            if executed_results
            else 0.0
        ),
        mean_naturalness=(
            sum(result.metrics.naturalness for result in executed_results)
            / len(executed_results)
            if executed_results
            else 0.0
        ),
        mean_speaker_grammar_similarity=(
            sum(
                result.metrics.speaker_grammar_similarity for result in executed_results
            )
            / len(executed_results)
            if executed_results
            else 0.0
        ),
        mean_composite=(
            sum(result.metrics.composite for result in executed_results)
            / len(executed_results)
            if executed_results
            else 0.0
        ),
    )


def _report_status_from_summary(summary: BenchmarkSummary) -> str:
    """Resolve the top-level report status from summary counts."""
    if summary.executed_count == 0:
        return "failed"
    if summary.sample_count < summary.target_sample_count:
        return "partial"
    if summary.failure_count == 0 and summary.skipped_count == 0:
        return "completed"
    return "partial"


def _resolve_sample_shard_args(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve and validate optional sample-shard selection arguments."""
    shard_index = int(getattr(args, "sample_shard_index", 1) or 1)
    shard_count = int(getattr(args, "sample_shard_count", 1) or 1)
    if shard_count <= 0:
        raise BenchmarkExecutionError("--sample-shard-count must be positive.")
    if shard_index <= 0 or shard_index > shard_count:
        raise BenchmarkExecutionError(
            "--sample-shard-index must be between 1 and --sample-shard-count."
        )
    return shard_index, shard_count


def _select_sample_shard(
    samples: list[PreparedBenchmarkSample],
    *,
    shard_index: int,
    shard_count: int,
) -> list[PreparedBenchmarkSample]:
    """Filter a selected sample list down to one deterministic shard."""
    if shard_count == 1:
        return list(samples)
    zero_based_index = shard_index - 1
    return [
        sample
        for sample_position, sample in enumerate(samples)
        if sample_position % shard_count == zero_based_index
    ]


def _resolve_shared_sample_queue_root(args: argparse.Namespace) -> Path | None:
    """Resolve an optional shared sample-queue root for pooled sweep workers."""
    raw_value = str(getattr(args, "shared_sample_queue_root", "") or "").strip()
    if not raw_value:
        return None
    return Path(raw_value).resolve()


def _open_shared_sample_queue_connection(queue_root: Path) -> sqlite3.Connection:
    """Open the preset-level SQLite queue used by pooled sweep workers."""
    queue_root.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        queue_root / _SHARED_SAMPLE_QUEUE_DB_NAME,
        timeout=30.0,
        isolation_level=None,
    )
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout = 30000")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS queue_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS sample_queue (
            sample_id TEXT PRIMARY KEY,
            sample_position INTEGER NOT NULL,
            sample_payload_json TEXT,
            status TEXT NOT NULL,
            owner_label TEXT,
            owner_pid INTEGER,
            final_status TEXT,
            counts_toward_target INTEGER NOT NULL DEFAULT 0,
            replacement_reason TEXT,
            updated_at_utc TEXT NOT NULL
        )
        """
    )
    sample_queue_columns = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(sample_queue)")
    }
    if "sample_payload_json" not in sample_queue_columns:
        connection.execute(
            "ALTER TABLE sample_queue ADD COLUMN sample_payload_json TEXT"
        )
    return connection


def _initialize_shared_sample_queue(
    connection: sqlite3.Connection,
    *,
    sample_order: list[str],
    target_sample_count: int,
) -> None:
    """Create or validate the shared preset queue against the selected sample order."""
    serialized_order = json.dumps(sample_order, ensure_ascii=True)
    serialized_target = str(target_sample_count)
    now_iso = datetime.now(timezone.utc).isoformat()
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            "SELECT value FROM queue_metadata WHERE key = 'sample_order_json'"
        ).fetchone()
        target_row = connection.execute(
            "SELECT value FROM queue_metadata WHERE key = 'target_sample_count'"
        ).fetchone()
        if row is None:
            connection.execute(
                "INSERT INTO queue_metadata(key, value) VALUES (?, ?)",
                ("sample_order_json", serialized_order),
            )
            connection.execute(
                "INSERT INTO queue_metadata(key, value) VALUES (?, ?)",
                ("target_sample_count", serialized_target),
            )
            connection.executemany(
                """
                INSERT INTO sample_queue(
                    sample_id,
                    sample_position,
                    sample_payload_json,
                    status,
                    owner_label,
                    owner_pid,
                    final_status,
                    counts_toward_target,
                    replacement_reason,
                    updated_at_utc
                ) VALUES (?, ?, NULL, 'pending', NULL, NULL, NULL, 0, NULL, ?)
                """,
                [
                    (sample_id, index, now_iso)
                    for index, sample_id in enumerate(sample_order, start=1)
                ],
            )
        elif str(row[0]) != serialized_order or target_row is None or str(target_row[0]) != serialized_target:
            raise BenchmarkExecutionError(
                "Shared sample queue does not match the selected benchmark sample order or retained target."
            )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


def _shared_sample_sort_key(
    *,
    sample_id: str,
    sample_seed: int | None,
    manifest_position: int,
) -> str:
    """Build the deterministic queue ordering key for one candidate sample."""
    if sample_seed is None:
        return f"{manifest_position:08d}"
    return hashlib.sha256(f"{sample_seed}:{sample_id}".encode("utf-8")).hexdigest()


def _serialize_prepared_sample(sample: PreparedBenchmarkSample) -> str:
    """Serialize one prepared sample for queue-backed worker lookup."""
    return json.dumps(
        asdict(sample),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _deserialize_prepared_sample(raw_payload: str) -> PreparedBenchmarkSample:
    """Deserialize one queued prepared sample payload."""
    payload = json.loads(raw_payload)
    if not isinstance(payload, dict):
        raise BenchmarkExecutionError(
            "Shared sample queue payload must decode to a JSON object."
        )
    return PreparedBenchmarkSample(
        sample_id=str(payload.get("sample_id") or ""),
        dataset=str(payload.get("dataset") or ""),
        episode_key=str(payload.get("episode_key") or ""),
        transcript_path=str(payload.get("transcript_path") or ""),
        reference_summary=str(payload.get("reference_summary") or ""),
        target_seconds=int(payload.get("target_seconds") or 0),
        metadata=(
            dict(payload.get("metadata"))
            if isinstance(payload.get("metadata"), dict)
            else {}
        ),
    )


def _initialize_shared_sample_queue_from_manifest(
    connection: sqlite3.Connection,
    *,
    manifest_path: Path,
    requested_target_sample_count: int,
    sample_seed: int | None,
) -> tuple[list[str], int]:
    """Create or validate a shared queue directly from the prepared manifest."""
    connection.execute("BEGIN IMMEDIATE")
    try:
        row = connection.execute(
            "SELECT value FROM queue_metadata WHERE key = 'sample_order_json'"
        ).fetchone()
        target_row = connection.execute(
            "SELECT value FROM queue_metadata WHERE key = 'target_sample_count'"
        ).fetchone()
        if row is not None and target_row is not None:
            sample_order = json.loads(str(row[0]))
            if not isinstance(sample_order, list) or not all(
                isinstance(sample_id, str) for sample_id in sample_order
            ):
                raise BenchmarkExecutionError(
                    "Shared sample queue sample order metadata is invalid."
                )
            target_sample_count = int(target_row[0])
            connection.execute("COMMIT")
            return [str(sample_id) for sample_id in sample_order], target_sample_count

        connection.execute("DELETE FROM queue_metadata")
        connection.execute("DELETE FROM sample_queue")
        connection.execute("DROP TABLE IF EXISTS temp.shared_sample_catalog_stage")
        connection.execute(
            """
            CREATE TEMP TABLE shared_sample_catalog_stage (
                sort_key TEXT NOT NULL,
                sample_id TEXT PRIMARY KEY,
                sample_payload_json TEXT NOT NULL
            )
            """
        )
        total_sample_count = 0
        for total_sample_count, sample in enumerate(
            iter_prepared_dataset_samples(manifest_path, max_samples=None),
            start=1,
        ):
            connection.execute(
                """
                INSERT INTO shared_sample_catalog_stage(
                    sort_key,
                    sample_id,
                    sample_payload_json
                ) VALUES (?, ?, ?)
                """,
                (
                    _shared_sample_sort_key(
                        sample_id=sample.sample_id,
                        sample_seed=sample_seed,
                        manifest_position=total_sample_count,
                    ),
                    sample.sample_id,
                    _serialize_prepared_sample(sample),
                ),
            )
        if total_sample_count <= 0:
            raise BenchmarkExecutionError(
                "Shared sample queue initialization found no benchmark candidates."
            )

        target_sample_count = (
            total_sample_count
            if requested_target_sample_count <= 0
            else min(requested_target_sample_count, total_sample_count)
        )
        sample_order: list[str] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        for sample_position, row in enumerate(
            connection.execute(
                """
                SELECT sample_id, sample_payload_json
                FROM shared_sample_catalog_stage
                ORDER BY sort_key ASC, sample_id ASC
                """
            ),
            start=1,
        ):
            sample_id = str(row[0])
            sample_order.append(sample_id)
            connection.execute(
                """
                INSERT INTO sample_queue(
                    sample_id,
                    sample_position,
                    sample_payload_json,
                    status,
                    owner_label,
                    owner_pid,
                    final_status,
                    counts_toward_target,
                    replacement_reason,
                    updated_at_utc
                ) VALUES (?, ?, ?, 'pending', NULL, NULL, NULL, 0, NULL, ?)
                """,
                (
                    sample_id,
                    sample_position,
                    str(row[1]),
                    now_iso,
                ),
            )
        connection.execute("DROP TABLE IF EXISTS temp.shared_sample_catalog_stage")
        connection.execute(
            "INSERT INTO queue_metadata(key, value) VALUES (?, ?)",
            ("sample_order_json", json.dumps(sample_order, ensure_ascii=True)),
        )
        connection.execute(
            "INSERT INTO queue_metadata(key, value) VALUES (?, ?)",
            ("target_sample_count", str(target_sample_count)),
        )
        connection.execute("COMMIT")
        return sample_order, target_sample_count
    except Exception:
        connection.execute("ROLLBACK")
        raise


def _load_shared_sample(
    connection: sqlite3.Connection,
    *,
    sample_id: str,
) -> PreparedBenchmarkSample:
    """Load one claimed sample payload from the shared queue catalog."""
    row = connection.execute(
        """
        SELECT sample_payload_json
        FROM sample_queue
        WHERE sample_id = ?
        """,
        (sample_id,),
    ).fetchone()
    if row is None or not str(row[0] or "").strip():
        raise BenchmarkExecutionError(
            f"Shared sample queue is missing payload for sample '{sample_id}'."
        )
    return _deserialize_prepared_sample(str(row[0]))


def _claim_next_shared_sample_id(
    connection: sqlite3.Connection,
    *,
    worker_label: str,
    worker_pid: int,
) -> str | None:
    """Claim the next pending sample from a preset-level shared queue."""
    claimed_sample_id: str | None = None
    now_iso = datetime.now(timezone.utc).isoformat()
    connection.execute("BEGIN IMMEDIATE")
    try:
        counted_row = connection.execute(
            """
            SELECT
                COALESCE(SUM(CASE WHEN status = 'completed' AND counts_toward_target = 1 THEN 1 ELSE 0 END), 0),
                COALESCE(SUM(CASE WHEN status = 'claimed' THEN 1 ELSE 0 END), 0)
            FROM sample_queue
            """
        ).fetchone()
        target_row = connection.execute(
            "SELECT value FROM queue_metadata WHERE key = 'target_sample_count'"
        ).fetchone()
        counted_results = int(counted_row[0]) if counted_row is not None else 0
        claimed_results = int(counted_row[1]) if counted_row is not None else 0
        target_sample_count = int(target_row[0]) if target_row is not None else 0
        if counted_results + claimed_results >= target_sample_count:
            connection.execute("COMMIT")
            return None
        row = connection.execute(
            """
            SELECT sample_id
            FROM sample_queue
            WHERE status = 'pending'
            ORDER BY sample_position ASC
            LIMIT 1
            """
        ).fetchone()
        if row is not None:
            claimed_sample_id = str(row[0])
            connection.execute(
                """
                UPDATE sample_queue
                SET status = 'claimed',
                    owner_label = ?,
                    owner_pid = ?,
                    updated_at_utc = ?
                WHERE sample_id = ?
                """,
                (worker_label, worker_pid, now_iso, claimed_sample_id),
            )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return claimed_sample_id


def _mark_shared_sample_completed(
    connection: sqlite3.Connection,
    *,
    sample_id: str,
    final_status: str,
    counts_toward_target: bool,
    replacement_reason: str | None = None,
) -> None:
    """Mark one shared-queue sample terminal so other workers skip it."""
    resolved_status = "completed" if counts_toward_target else "replaced"
    connection.execute("BEGIN IMMEDIATE")
    try:
        connection.execute(
            """
            UPDATE sample_queue
            SET status = ?,
                owner_label = NULL,
                owner_pid = NULL,
                final_status = ?,
                counts_toward_target = ?,
                replacement_reason = ?,
                updated_at_utc = ?
            WHERE sample_id = ?
            """,
            (
                resolved_status,
                final_status,
                1 if counts_toward_target else 0,
                replacement_reason,
                datetime.now(timezone.utc).isoformat(),
                sample_id,
            ),
        )
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise


def _build_execution_report(
    *,
    run_id: str,
    dataset_manifest: Any,
    summarizer_provider: str,
    critic_provider: str,
    judge_provider: str,
    embedding_mode: str,
    results: list[SampleBenchmarkResult],
    duration_tolerance_ratio: float,
    transcript_only: bool,
    duration_preset: str,
    override_target_seconds: int | None,
    sample_seed: int | None,
    target_sample_count: int,
    checkpoint_path: Path | None,
    resumed_from_checkpoint: bool,
    resume_count: int,
    sample_shard_index: int,
    sample_shard_count: int,
) -> BenchmarkExecutionReport:
    """Build the SPoRC execution report before artifact paths are known."""
    summary = _summarize_results(results, target_sample_count=target_sample_count)
    git_commit, git_branch = git_info(REPO_ROOT)
    status = _report_status_from_summary(summary)
    if summary.sample_count == 0 and sample_shard_count > 1:
        status = "completed"
    return BenchmarkExecutionReport(
        run_id=run_id,
        generated_at_utc=utc_now_iso(),
        status=status,
        dataset_manifest_path=str(dataset_manifest.manifest_path),
        sporc_dataset_id=dataset_manifest.sporc_dataset_id,
        dataset_variant=dataset_manifest.dataset_variant,
        summarizer_provider=summarizer_provider,
        critic_provider=critic_provider,
        judge_provider=judge_provider,
        embedding_mode=embedding_mode,
        commands_executed=[" ".join(["python", *sys.argv])],
        git_commit=git_commit,
        git_branch=git_branch,
        report_path="",
        notes_path="",
        results_path="",
        samples=results,
        summary=summary,
        sample_artifacts_root=None,
        duration_tolerance_ratio=duration_tolerance_ratio,
        transcript_only=transcript_only,
        duration_preset=duration_preset,
        override_target_seconds=override_target_seconds,
        sample_seed=sample_seed,
        checkpoint_path=(str(checkpoint_path) if checkpoint_path is not None else None),
        resumed_from_checkpoint=resumed_from_checkpoint,
        resume_count=resume_count,
        sample_shard_index=sample_shard_index,
        sample_shard_count=sample_shard_count,
    )


def _write_execution_report_snapshot(
    *,
    run_root: Path,
    run_id: str,
    dataset_manifest: Any,
    summarizer_provider: str,
    critic_provider: str,
    judge_provider: str,
    embedding_mode: str,
    sample_order: list[str],
    results_by_sample_id: dict[str, SampleBenchmarkResult],
    duration_tolerance_ratio: float,
    transcript_only: bool,
    duration_preset: str,
    override_target_seconds: int | None,
    sample_seed: int | None,
    target_sample_count: int,
    checkpoint_path: Path | None,
    resumed_from_checkpoint: bool,
    resume_count: int,
    sample_shard_index: int,
    sample_shard_count: int,
) -> BenchmarkExecutionReport:
    """Write one partial or final execution report snapshot."""
    samples_root = (run_root / "samples").resolve()
    sample_artifacts_root = str(samples_root) if samples_root.exists() else None
    provisional_report = replace(
        _build_execution_report(
            run_id=run_id,
            dataset_manifest=dataset_manifest,
            summarizer_provider=summarizer_provider,
            critic_provider=critic_provider,
            judge_provider=judge_provider,
            embedding_mode=embedding_mode,
            results=_ordered_results(sample_order, results_by_sample_id),
            duration_tolerance_ratio=duration_tolerance_ratio,
            transcript_only=transcript_only,
            duration_preset=duration_preset,
            override_target_seconds=override_target_seconds,
            sample_seed=sample_seed,
            target_sample_count=target_sample_count,
            checkpoint_path=checkpoint_path,
            resumed_from_checkpoint=resumed_from_checkpoint,
            resume_count=resume_count,
            sample_shard_index=sample_shard_index,
            sample_shard_count=sample_shard_count,
        ),
        sample_artifacts_root=sample_artifacts_root,
    )
    report_path, notes_path, results_path = write_execution_artifacts(
        run_root=run_root,
        report=provisional_report,
    )
    finalized_report = replace(
        provisional_report,
        report_path=str(report_path),
        notes_path=str(notes_path),
        results_path=str(results_path),
    )
    write_execution_artifacts(run_root=run_root, report=finalized_report)
    return finalized_report


def _build_runtime_port_args(
    args: argparse.Namespace,
    *,
    force_dynamic_ports: bool,
) -> argparse.Namespace:
    """Return port settings for one runtime build."""
    if not force_dynamic_ports:
        return args
    runtime_args = argparse.Namespace(**vars(args))
    runtime_args.retrieval_port = None
    runtime_args.summarizer_port = None
    runtime_args.critic_port = None
    return runtime_args


async def _start_managed_runtime(
    *,
    args: argparse.Namespace,
    run_root: Path,
    base_cfg: Any,
    summarizer_provider_config: dict[str, Any],
    critic_provider_config: dict[str, Any],
    judge_provider_config: dict[str, Any],
    embedding_profile: EmbeddingProfile,
    transcript_only: bool,
    calibration: Any,
    voice_clone_orchestrator: Any | None,
    logging_enabled: bool,
    force_dynamic_ports: bool,
) -> _ManagedBenchmarkRuntime:
    """Instantiate providers, local agents, and the orchestrator for one run phase."""
    llm_summarizer: LLMProvider = hydra.utils.instantiate(
        OmegaConf.create(summarizer_provider_config)
    )
    llm_critic: LLMProvider = hydra.utils.instantiate(
        OmegaConf.create(critic_provider_config)
    )
    llm_judge: LLMProvider = hydra.utils.instantiate(
        OmegaConf.create(judge_provider_config)
    )
    artifact_recorder = _SampleArtifactRecorder(run_root=run_root)
    summarizer_callback: LLMResponseCallback = _ArtifactCaptureResponseCallback(
        recorder=artifact_recorder,
        role="summarizer",
    )
    critic_callback: LLMResponseCallback = _ArtifactCaptureResponseCallback(
        recorder=artifact_recorder,
        role="critic",
    )
    judge_callback: LLMResponseCallback = _ArtifactCaptureResponseCallback(
        recorder=artifact_recorder,
        role="judge",
    )
    if not bool(args.disable_stream_output):
        summarizer_callback = CompositeResponseCallback(
            summarizer_callback,
            RichConsoleResponseCallback(),
        )
        critic_callback = CompositeResponseCallback(
            critic_callback,
            RichConsoleResponseCallback(),
        )
        judge_callback = CompositeResponseCallback(
            judge_callback,
            RichConsoleResponseCallback(),
        )
    _attach_response_callback(llm_summarizer, summarizer_callback)
    _attach_response_callback(llm_critic, critic_callback)
    _attach_response_callback(llm_judge, judge_callback)
    if logging_enabled:
        llm_summarizer = LoggingLLMProvider(inner_provider=llm_summarizer)
        llm_critic = LoggingLLMProvider(inner_provider=llm_critic)
        llm_judge = LoggingLLMProvider(inner_provider=llm_judge)
    llm_summarizer = _ArtifactCaptureLLMProvider(
        inner_provider=llm_summarizer,
        recorder=artifact_recorder,
        role="summarizer",
    )
    llm_critic = _ArtifactCaptureLLMProvider(
        inner_provider=llm_critic,
        recorder=artifact_recorder,
        role="critic",
    )
    llm_judge = _ArtifactCaptureLLMProvider(
        inner_provider=llm_judge,
        recorder=artifact_recorder,
        role="judge",
    )

    embedding: EmbeddingProvider = hydra.utils.instantiate(
        OmegaConf.create(embedding_profile.embedding_config)
    )
    retrieval_enabled = embedding_profile.embedding_id != "disabled"
    transcript_index = TranscriptIndex(embedding_provider=embedding)
    judge = SPoRCJudge(
        judge_llm=llm_judge,
        config=SPoRCJudgeConfig(source_char_limit=int(args.judge_source_char_limit)),
    )
    event_capture = EventCapture(
        ["tool_invocation", "retrieval_stats", "orchestrator_iteration_completed"]
    )
    event_capture.start()
    shared_agent_client = AgentClient(
        event_bus=event_bus,
        limits=httpx.Limits(
            max_keepalive_connections=_AGENT_CLIENT_MAX_KEEPALIVE_CONNECTIONS,
            max_connections=_AGENT_CLIENT_MAX_CONNECTIONS,
            keepalive_expiry=_AGENT_CLIENT_KEEPALIVE_EXPIRY_SECONDS,
        ),
        pool_timeout_seconds=_AGENT_CLIENT_POOL_TIMEOUT_SECONDS,
        trust_env=False,
    )
    servers: list[ManagedServer] = []
    reserved_ports: tuple[int, int, int] | None = None
    try:
        runtime_args = _build_runtime_port_args(
            args,
            force_dynamic_ports=force_dynamic_ports,
        )
        retrieval_port, summarizer_port, critic_port = _resolve_benchmark_ports(
            runtime_args
        )
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
                _run_server_in_thread("sporc-retrieval", retrieval_app, retrieval_port)
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
                voice_clone_orchestrator=voice_clone_orchestrator,
                live_draft_audio_enabled=not transcript_only,
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
            _run_server_in_thread("sporc-summarizer", summarizer_app, summarizer_port)
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
        servers.append(_run_server_in_thread("sporc-critic", critic_app, critic_port))

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
            timeouts=_resolve_orchestrator_timeouts(args),
            agent_client=shared_agent_client,
            event_bus=event_bus,
        )
        return _ManagedBenchmarkRuntime(
            orchestrator=orchestrator,
            judge=judge,
            event_capture=event_capture,
            artifact_recorder=artifact_recorder,
            retrieval_enabled=retrieval_enabled,
            shared_agent_client=shared_agent_client,
            servers=servers,
            retrieval_port=retrieval_port,
            summarizer_port=summarizer_port,
            critic_port=critic_port,
        )
    except Exception:
        event_capture.stop()
        for server in reversed(servers):
            _stop_server(server)
        if reserved_ports is not None:
            for port in reserved_ports:
                _release_reserved_port(port)
        await shared_agent_client.close()
        raise


async def _stop_managed_runtime(runtime: _ManagedBenchmarkRuntime | None) -> None:
    """Tear down one managed runtime instance."""
    if runtime is None:
        return
    runtime.event_capture.stop()
    for server in reversed(runtime.servers):
        _stop_server(server)
    await runtime.shared_agent_client.close()


def _format_transcript(transcript: dict[str, Any]) -> str:
    raw_segments = transcript.get("segments", [])
    if not isinstance(raw_segments, list):
        return ""
    return "".join(
        f"[{segment.get('speaker', 'UNKNOWN')}]: {segment.get('text', '')}\n"
        for segment in raw_segments
        if isinstance(segment, dict)
    )


def _coerce_alias_map(
    transcript: dict[str, Any], sample: PreparedBenchmarkSample
) -> dict[str, str]:
    def _normalize_alias_map(raw_alias_map: dict[str, Any]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        seen_casefold: set[str] = set()
        for key, value in raw_alias_map.items():
            cleaned_key = str(key).strip()
            cleaned_value = str(value).strip()
            if not cleaned_key or not cleaned_value:
                continue
            canonical_key = (
                cleaned_key.upper()
                if _TRANSCRIPT_ONLY_SPEAKER_LABEL_PATTERN.match(cleaned_key)
                else cleaned_key
            )
            casefold_key = canonical_key.casefold()
            if casefold_key in seen_casefold:
                continue
            normalized[canonical_key] = cleaned_value
            seen_casefold.add(casefold_key)
        return normalized

    metadata = transcript.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(
        metadata.get("speaker_alias_map"), dict
    ):
        return _normalize_alias_map(metadata["speaker_alias_map"])
    raw_alias_map = sample.metadata.get("speaker_alias_map", {})
    if isinstance(raw_alias_map, dict):
        return _normalize_alias_map(raw_alias_map)
    return {}


def _coerce_speaker_notes(
    transcript: dict[str, Any], sample: PreparedBenchmarkSample
) -> list[str]:
    metadata = transcript.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("speaker_notes"), list):
        return [str(item) for item in metadata["speaker_notes"] if str(item).strip()]
    raw_notes = sample.metadata.get("speaker_notes", [])
    if isinstance(raw_notes, list):
        return [str(item) for item in raw_notes if str(item).strip()]
    return []


def _canonical_speaker_label(label: str, alias_map: dict[str, str]) -> str:
    cleaned = label.strip()
    if not cleaned:
        return "UNKNOWN"
    direct_match = alias_map.get(cleaned)
    if direct_match:
        return direct_match
    cleaned_casefold = cleaned.casefold()
    for alias_key, canonical_label in alias_map.items():
        if alias_key.casefold() == cleaned_casefold:
            return canonical_label
    return cleaned


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [text]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        deduped.append(cleaned)
        seen.add(cleaned)
    return deduped


def _join_segment_text(left_text: str, right_text: str) -> str:
    left = left_text.rstrip()
    right = right_text.lstrip()
    if not left:
        return right
    if not right:
        return left
    if right[:1] in ",.;:?!)]}" or left.endswith(("(", "/", "-", "'")):
        return f"{left}{right}"
    return f"{left} {right}"


def _segment_duration_ms(segment: TranscriptSegment) -> int:
    return max(0, int(segment.end_time) - int(segment.start_time))


def _segment_word_count(segment: TranscriptSegment) -> int:
    return count_words(segment.text)


def _merge_segment_extras(
    left_extras: dict[str, Any],
    right_extras: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(left_extras)
    for key, value in right_extras.items():
        merged.setdefault(key, value)
    left_raw_primary = str(
        left_extras.get("raw_primary_speaker", merged.get("raw_primary_speaker", ""))
    ).strip()
    right_raw_primary = str(
        right_extras.get("raw_primary_speaker", merged.get("raw_primary_speaker", ""))
    ).strip()
    raw_overlap_speakers = _dedupe_preserve_order(
        _coerce_string_list(left_extras.get("raw_overlap_speakers"))
        + _coerce_string_list(right_extras.get("raw_overlap_speakers"))
    )
    raw_speakers = _dedupe_preserve_order(
        _coerce_string_list(left_extras.get("raw_speakers"))
        + _coerce_string_list(right_extras.get("raw_speakers"))
        + [left_raw_primary, right_raw_primary]
        + raw_overlap_speakers
    )
    speaker_candidates = _dedupe_preserve_order(
        _coerce_string_list(left_extras.get("speaker_candidates"))
        + _coerce_string_list(right_extras.get("speaker_candidates"))
        + raw_speakers
    )
    merged["raw_primary_speaker"] = left_raw_primary or right_raw_primary
    merged["raw_overlap_speakers"] = raw_overlap_speakers
    merged["raw_speakers"] = raw_speakers
    merged["speaker_candidates"] = speaker_candidates
    merged["speaker_uncertain"] = (
        bool(left_extras.get("speaker_uncertain"))
        or bool(right_extras.get("speaker_uncertain"))
        or len(speaker_candidates) > 1
    )
    return merged


def _merge_adjacent_same_speaker_segments(
    segments: list[TranscriptSegment],
    *,
    merge_gap_ms: int,
) -> list[TranscriptSegment]:
    if not segments:
        return []
    merged: list[TranscriptSegment] = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        gap_ms = max(0, int(segment.start_time) - int(previous.end_time))
        if previous.speaker == segment.speaker and gap_ms <= merge_gap_ms:
            merged[-1] = TranscriptSegment(
                speaker=previous.speaker,
                text=_join_segment_text(previous.text, segment.text),
                start_time=previous.start_time,
                end_time=max(previous.end_time, segment.end_time),
                extras=_merge_segment_extras(previous.extras, segment.extras),
            )
            continue
        merged.append(segment)
    return merged


def _extract_named_speaker_labels(
    *,
    metadata: dict[str, Any],
    sample: PreparedBenchmarkSample,
    alias_map: dict[str, str],
) -> list[str]:
    metadata_labels = metadata.get("named_speaker_labels")
    if isinstance(metadata_labels, list):
        return _dedupe_preserve_order(
            [
                _canonical_speaker_label(str(label), alias_map)
                for label in metadata_labels
                if str(label).strip()
            ]
        )
    sample_labels = sample.metadata.get("named_speaker_labels")
    if isinstance(sample_labels, list):
        return _dedupe_preserve_order(
            [
                _canonical_speaker_label(str(label), alias_map)
                for label in sample_labels
                if str(label).strip()
            ]
        )
    return _dedupe_preserve_order(
        [
            canonical_label
            for alias_key, canonical_label in alias_map.items()
            if not _TRANSCRIPT_ONLY_SPEAKER_LABEL_PATTERN.match(alias_key)
        ]
    )


def _extract_main_speaker_labels(
    *,
    metadata: dict[str, Any],
    sample: PreparedBenchmarkSample,
    alias_map: dict[str, str],
) -> list[str]:
    for raw_value in (
        metadata.get("main_speaker_labels"),
        sample.metadata.get("main_speaker_labels"),
    ):
        if isinstance(raw_value, list):
            return _dedupe_preserve_order(
                [
                    _canonical_speaker_label(str(label), alias_map)
                    for label in raw_value
                    if str(label).strip()
                ]
            )
    speaker_notes = metadata.get("speaker_notes")
    if not isinstance(speaker_notes, list):
        speaker_notes = sample.metadata.get("speaker_notes", [])
    if isinstance(speaker_notes, list):
        for note in speaker_notes:
            note_text = str(note)
            if not note_text.startswith("mainEpSpeakers:"):
                continue
            raw_labels = [
                part.strip()
                for part in note_text.split(":", 1)[1].split(",")
                if part.strip()
            ]
            return _dedupe_preserve_order(
                [
                    _canonical_speaker_label(label, alias_map)
                    for label in raw_labels
                ]
            )
    return []


def _normalize_transcript_for_transcript_only_benchmark(
    *,
    transcript: Transcript,
    sample: PreparedBenchmarkSample,
) -> Transcript:
    alias_map = _coerce_alias_map(transcript.to_payload(), sample)
    prepared_segments: list[TranscriptSegment] = []
    raw_speakers_seen: list[str] = []
    for segment in sorted(
        transcript.segments,
        key=lambda item: (int(item.start_time), int(item.end_time)),
    ):
        cleaned_text = str(segment.text).strip()
        if not cleaned_text or not _TRANSCRIPT_ONLY_ALNUM_PATTERN.search(cleaned_text):
            continue
        raw_primary_speaker = str(
            segment.extras.get("raw_primary_speaker", segment.speaker)
        ).strip() or str(segment.speaker).strip() or "UNKNOWN"
        raw_overlap_speakers = _dedupe_preserve_order(
            _coerce_string_list(
                segment.extras.get(
                    "raw_overlap_speakers",
                    segment.extras.get("overlap_speakers"),
                )
            )
        )
        raw_speakers = _dedupe_preserve_order(
            _coerce_string_list(segment.extras.get("raw_speakers"))
            + [raw_primary_speaker]
            + raw_overlap_speakers
        )
        canonical_speaker = _canonical_speaker_label(str(segment.speaker), alias_map)
        speaker_candidates = _dedupe_preserve_order(
            [
                _canonical_speaker_label(candidate, alias_map)
                for candidate in raw_speakers
                if candidate
            ]
        )
        speaker_uncertain = len(speaker_candidates) > 1
        raw_speakers_seen.extend(raw_speakers)
        extras = dict(segment.extras)
        extras["raw_primary_speaker"] = raw_primary_speaker
        extras["raw_overlap_speakers"] = raw_overlap_speakers
        extras["raw_speakers"] = raw_speakers
        extras["speaker_candidates"] = speaker_candidates
        extras["speaker_uncertain"] = speaker_uncertain
        prepared_segments.append(
            TranscriptSegment(
                speaker=canonical_speaker,
                text=cleaned_text,
                start_time=int(segment.start_time),
                end_time=int(segment.end_time),
                extras=extras,
            )
        )

    normalized_segments = _merge_adjacent_same_speaker_segments(
        prepared_segments,
        merge_gap_ms=_TRANSCRIPT_ONLY_SPEAKER_MERGE_GAP_MS,
    )
    named_speaker_labels = _extract_named_speaker_labels(
        metadata=transcript.metadata,
        sample=sample,
        alias_map=alias_map,
    )
    main_speaker_labels = _extract_main_speaker_labels(
        metadata=transcript.metadata,
        sample=sample,
        alias_map=alias_map,
    )
    updated_metadata = dict(transcript.metadata)
    updated_metadata["speaker_alias_map"] = alias_map
    updated_metadata["named_speaker_labels"] = named_speaker_labels
    updated_metadata["main_speaker_labels"] = main_speaker_labels
    raw_normalization = updated_metadata.get("speaker_normalization", {})
    normalization_metadata = (
        dict(raw_normalization)
        if isinstance(raw_normalization, dict)
        else {}
    )
    normalization_metadata.pop("core_speaker_min_share", None)
    normalization_metadata.pop("core_speaker_labels", None)
    normalization_metadata.update(
        {
            "mode": "transcript_only_benchmark",
            "speaker_assignment_mode": "raw_with_uncertainty",
            "speaker_reassignment_enabled": False,
            "merge_gap_ms": _TRANSCRIPT_ONLY_SPEAKER_MERGE_GAP_MS,
            "raw_segment_count": len(transcript.segments),
            "raw_distinct_speaker_count": len(_dedupe_preserve_order(raw_speakers_seen)),
            "normalized_segment_count": len(normalized_segments),
            "normalized_distinct_speaker_count": len(
                {segment.speaker for segment in normalized_segments}
            ),
            "uncertain_segment_count": sum(
                1
                for segment in normalized_segments
                if bool(segment.extras.get("speaker_uncertain"))
            ),
        }
    )
    updated_metadata["speaker_normalization"] = normalization_metadata
    return Transcript(
        segments=tuple(normalized_segments),
        metadata=updated_metadata,
        extras=dict(transcript.extras),
    )


def _speaker_message_lists_from_transcript(
    transcript: dict[str, Any],
    alias_map: dict[str, str],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    raw_segments = transcript.get("segments", [])
    if not isinstance(raw_segments, list):
        return {}
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        speaker = _canonical_speaker_label(
            str(raw_segment.get("speaker", "UNKNOWN")),
            alias_map,
        )
        text = str(raw_segment.get("text", "")).strip()
        if text:
            grouped[speaker].append(text)
    return dict(grouped)


def _summary_turns(summary_xml: str) -> list[SummaryTurn]:
    try:
        return parse_summary_xml(summary_xml)
    except ValueError:
        stripped = re.sub(r"<[^>]+>", " ", summary_xml).strip()
        if not stripped:
            return []
        return [
            SummaryTurn(speaker="UNKNOWN", text=stripped, emo_preset=DEFAULT_EMO_PRESET)
        ]


def _speaker_message_lists_from_summary(
    summary_xml: str,
    alias_map: dict[str, str],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for turn in _summary_turns(summary_xml):
        speaker = _canonical_speaker_label(turn.speaker, alias_map)
        text = turn.text.strip()
        if text:
            grouped[speaker].append(text)
    return dict(grouped)


def _compute_composite_score(
    *,
    factualness: float,
    naturalness: float,
    speaker_grammar_similarity: float,
) -> float:
    values = [factualness, naturalness, speaker_grammar_similarity]
    return round(sum(values) / len(values), 4)


def _prepare_benchmark_audio_cfg(
    audio_cfg: dict[str, Any],
    *,
    run_root: Path,
) -> dict[str, Any]:
    resolved = deepcopy(audio_cfg)
    resolved["work_dir"] = str((run_root / "audio_pipeline").resolve())
    speaker_samples_cfg = _coerce_mapping(resolved.get("speaker_samples", {}))
    speaker_samples_cfg["enabled"] = True
    speaker_samples_cfg["source_audio"] = "source"
    resolved["speaker_samples"] = speaker_samples_cfg
    voice_clone_cfg = _coerce_mapping(resolved.get("voice_clone", {}))
    voice_clone_cfg["enabled"] = True
    voice_clone_cfg["use_fp16"] = True
    live_drafting_cfg = _coerce_mapping(voice_clone_cfg.get("live_drafting", {}))
    live_drafting_cfg["enabled"] = True
    voice_clone_cfg["live_drafting"] = live_drafting_cfg
    resolved["voice_clone"] = voice_clone_cfg
    return resolved


def _working_transcript_path(
    *, run_root: Path, sample: PreparedBenchmarkSample
) -> Path:
    return (
        run_root / "working_transcripts" / f"{sample.sample_id}.transcript.json"
    ).resolve()


def _voice_clone_archive_dir(
    *, run_root: Path, sample: PreparedBenchmarkSample
) -> Path:
    return (run_root / "voice_clone_outputs" / sample.sample_id).resolve()


def _source_audio_destination(
    *,
    run_root: Path,
    sample: PreparedBenchmarkSample,
    source_audio_url: str,
) -> Path:
    parsed = urlparse(source_audio_url)
    suffix = Path(parsed.path).suffix or ".mp3"
    return (run_root / "source_audio" / f"{sample.sample_id}{suffix}").resolve()


def _normalize_source_audio_url(
    sample: PreparedBenchmarkSample, transcript: Transcript
) -> str:
    metadata = transcript.metadata
    candidates = (
        str(metadata.get("source_audio_url", "")).strip(),
        str(sample.metadata.get("source_audio_url", "")).strip(),
        str(sample.episode_key).strip(),
    )
    for value in candidates:
        if value:
            return value
    raise BenchmarkExecutionError(
        f"SPoRC sample '{sample.sample_id}' is missing a source audio URL or path."
    )


def _write_transcript_payload(path: Path, transcript: Transcript) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(transcript.to_payload(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _copy_sample_transcript(
    *,
    sample: PreparedBenchmarkSample,
    run_root: Path,
    transcript: Transcript | None = None,
    normalize_for_transcript_only: bool = False,
) -> tuple[Transcript, Path]:
    source_transcript = transcript
    if source_transcript is None:
        source_payload = json.loads(
            Path(sample.transcript_path).read_text(encoding="utf-8")
        )
        source_transcript = Transcript.from_mapping(source_payload)
    if normalize_for_transcript_only:
        source_transcript = _normalize_transcript_for_transcript_only_benchmark(
            transcript=source_transcript,
            sample=sample,
        )
    working_path = _working_transcript_path(run_root=run_root, sample=sample)
    _write_transcript_payload(working_path, source_transcript)
    return source_transcript, working_path


def _download_source_audio(
    *,
    source_audio_url: str,
    destination: Path,
    force_download: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force_download:
        return destination
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink(missing_ok=True)
    try:
        with requests.get(
            source_audio_url,
            stream=True,
            timeout=_SOURCE_AUDIO_TIMEOUT,
            headers=_SOURCE_AUDIO_HEADERS,
        ) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temp_path.replace(destination)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise BenchmarkExecutionError(
            "Failed to download SPoRC source audio for sample "
            f"'{source_audio_url}' -> '{destination}': {exc}"
        ) from exc
    return destination


def _ensure_local_source_audio(
    *,
    sample: PreparedBenchmarkSample,
    transcript: Transcript,
    working_transcript_path: Path,
    run_root: Path,
    force_download: bool,
) -> tuple[Transcript, Path]:
    existing_source_audio = str(
        transcript.metadata.get("source_audio_path", "")
    ).strip()
    if existing_source_audio:
        candidate = Path(existing_source_audio).expanduser()
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        if candidate.exists():
            return transcript, candidate.resolve()

    source_audio_url = _normalize_source_audio_url(sample, transcript)
    parsed = urlparse(source_audio_url)
    if parsed.scheme in {"http", "https"}:
        local_audio_path = _download_source_audio(
            source_audio_url=source_audio_url,
            destination=_source_audio_destination(
                run_root=run_root,
                sample=sample,
                source_audio_url=source_audio_url,
            ),
            force_download=force_download,
        )
    else:
        local_audio_path = Path(source_audio_url).expanduser()
        if not local_audio_path.is_absolute():
            local_audio_path = (REPO_ROOT / local_audio_path).resolve()
        if not local_audio_path.exists():
            raise BenchmarkExecutionError(
                "SPoRC source audio path does not exist for sample "
                f"'{sample.sample_id}': {local_audio_path}"
            )

    updated_metadata = dict(transcript.metadata)
    updated_metadata["source_audio_url"] = source_audio_url
    updated_metadata["source_audio_path"] = str(local_audio_path.resolve())
    updated_transcript = transcript.with_metadata(updated_metadata)
    _write_transcript_payload(working_transcript_path, updated_transcript)
    return updated_transcript, local_audio_path.resolve()


def _resolve_existing_manifest_path(transcript: Transcript) -> Path | None:
    manifest_value = str(
        transcript.metadata.get("speaker_samples_manifest_path", "")
    ).strip()
    if not manifest_value:
        return None
    candidate = Path(manifest_value).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if candidate.exists():
        return candidate.resolve()
    return None


def _prepare_speaker_samples_for_sample(
    *,
    sample: PreparedBenchmarkSample,
    transcript: Transcript,
    working_transcript_path: Path,
    run_root: Path,
    audio_cfg: dict[str, Any],
    force_download_audio: bool,
) -> tuple[Transcript, Path, Path]:
    existing_manifest = _resolve_existing_manifest_path(transcript)
    if existing_manifest is not None:
        updated_transcript, source_audio_path = _ensure_local_source_audio(
            sample=sample,
            transcript=transcript,
            working_transcript_path=working_transcript_path,
            run_root=run_root,
            force_download=force_download_audio,
        )
        return updated_transcript, existing_manifest, source_audio_path.resolve()

    updated_transcript, source_audio_path = _ensure_local_source_audio(
        sample=sample,
        transcript=transcript,
        working_transcript_path=working_transcript_path,
        run_root=run_root,
        force_download=force_download_audio,
    )
    speaker_samples_cfg = _coerce_mapping(audio_cfg.get("speaker_samples", {}))
    source_mode = str(speaker_samples_cfg.get("source_audio", "source"))
    resolved_source_audio_path = resolve_sample_source_audio_path(
        source_mode=source_mode,
        transcript_metadata=updated_transcript.metadata,
        configured_audio_path=str(audio_cfg.get("audio_path", "")),
        base_dir=REPO_ROOT,
    )
    output_dir = (run_root / "speaker_samples" / sample.sample_id).resolve()
    generator = build_speaker_sample_generator(audio_cfg)
    generation_result = generator.generate(
        transcript_payload=updated_transcript.to_payload(),
        source_audio_path=resolved_source_audio_path,
        output_dir=output_dir,
        progress_callback=None,
    )
    updated_metadata = dict(updated_transcript.metadata)
    updated_metadata.update(
        {
            "speaker_samples_manifest_path": str(generation_result.manifest_path),
            "speaker_samples_dir": str(generation_result.output_dir),
            "speaker_sample_count": len(generation_result.artifacts),
            "speaker_samples_generated_at_utc": generation_result.generated_at_utc,
        }
    )
    prepared_transcript = updated_transcript.with_metadata(updated_metadata)
    _write_transcript_payload(working_transcript_path, prepared_transcript)
    return (
        prepared_transcript,
        generation_result.manifest_path.resolve(),
        source_audio_path,
    )


def _build_draft_audio_state_path(
    *,
    voice_clone_output_dir: Path,
    sample: PreparedBenchmarkSample,
) -> Path:
    return (
        voice_clone_output_dir
        / f"live_draft_{sample.sample_id}_{sample.target_seconds}s.state.json"
    ).resolve()


def _archive_voice_clone_outputs(
    *,
    sample: PreparedBenchmarkSample,
    voice_clone_output_dir: Path,
    run_root: Path,
) -> tuple[Path | None, Path | None, Path | None]:
    if not voice_clone_output_dir.exists():
        return None, None, None
    destination = _voice_clone_archive_dir(run_root=run_root, sample=sample)
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(voice_clone_output_dir, destination)
    manifest_path = destination / "manifest.json"
    merged_audio_path = destination / "voice_cloned.wav"
    return (
        destination,
        manifest_path if manifest_path.exists() else None,
        merged_audio_path if merged_audio_path.exists() else None,
    )


async def _execute_sample(
    *,
    sample: PreparedBenchmarkSample,
    target_seconds: int,
    preloaded_transcript: Transcript | None,
    orchestrator: Orchestrator,
    event_capture: EventCapture,
    retrieval_enabled: bool,
    duration_tolerance_ratio: float,
    max_iterations: int | None,
    judge: SPoRCJudge,
    calibration: Any,
    run_root: Path,
    artifact_recorder: _SampleArtifactRecorder,
    audio_cfg: dict[str, Any],
    transcript_only: bool,
    voice_clone_orchestrator: Any | None,
    force_download_audio: bool,
    infra_retry_attempts: int = 0,
) -> SampleBenchmarkResult:
    sample_started = time.perf_counter()
    retrieval_before = event_capture.count("retrieval_stats")
    tool_before = event_capture.count("tool_invocation")
    source_audio_path: Path | None = None
    speaker_samples_manifest_path: Path | None = None
    draft_audio_state_path: Path | None = None
    archived_voice_clone_dir: Path | None = None
    archived_voice_clone_manifest: Path | None = None
    archived_voice_clone_merged_audio: Path | None = None
    diagnostics: dict[str, Any] | None = None
    candidate_summary = ""
    artifact_recorder.activate_sample(sample=sample, target_seconds=target_seconds)
    try:
        transcript, working_transcript_path = _copy_sample_transcript(
            sample=sample,
            run_root=run_root,
            transcript=preloaded_transcript,
            normalize_for_transcript_only=transcript_only,
        )
        if transcript_only:
            prepared_transcript = transcript
        else:
            if voice_clone_orchestrator is None:
                raise BenchmarkExecutionError(
                    "voice_clone_orchestrator is required when transcript_only is disabled."
                )
            prepared_transcript, speaker_samples_manifest_path, source_audio_path = (
                _prepare_speaker_samples_for_sample(
                    sample=sample,
                    transcript=transcript,
                    working_transcript_path=working_transcript_path,
                    run_root=run_root,
                    audio_cfg=audio_cfg,
                    force_download_audio=force_download_audio,
                )
            )
        transcript_payload = prepared_transcript.to_payload()
        source_text = _format_transcript(transcript_payload)
        full_transcript_text = source_text if not retrieval_enabled else ""
        if not transcript_only and voice_clone_orchestrator is not None:
            draft_audio_state_path = _build_draft_audio_state_path(
                voice_clone_output_dir=voice_clone_orchestrator.output_dir,
                sample=sample,
            )
        if retrieval_enabled:
            await orchestrator.index_transcript(transcript_payload)
        diagnostics = await orchestrator.run_loop_with_diagnostics(
            target_seconds=target_seconds,
            duration_tolerance_ratio=duration_tolerance_ratio,
            max_iterations=max_iterations,
            full_transcript_text=full_transcript_text,
            speaker_samples_manifest_path=speaker_samples_manifest_path,
            draft_audio_state_path=draft_audio_state_path,
        )
        candidate_summary = str(diagnostics.get("draft", "")).strip()
        candidate_seconds = float(
            diagnostics.get("final_estimated_seconds", 0.0) or 0.0
        )
        if (
            candidate_summary
            and not transcript_only
            and voice_clone_orchestrator is not None
        ):
            if draft_audio_state_path is None or speaker_samples_manifest_path is None:
                raise BenchmarkExecutionError(
                    "Live draft finalization requires draft audio state and speaker sample manifest paths."
                )
            finalized_draft_audio_state_path = draft_audio_state_path
            finalized_speaker_samples_manifest_path = speaker_samples_manifest_path
            live_draft_session = LiveDraftVoiceCloneSession.from_orchestrator(
                orchestrator=voice_clone_orchestrator,
                state_path=finalized_draft_audio_state_path,
                speaker_samples_manifest_path=finalized_speaker_samples_manifest_path,
            )
            live_draft_session.finalize()
            (
                archived_voice_clone_dir,
                archived_voice_clone_manifest,
                archived_voice_clone_merged_audio,
            ) = _archive_voice_clone_outputs(
                sample=sample,
                voice_clone_output_dir=voice_clone_orchestrator.output_dir,
                run_root=run_root,
            )
        if candidate_seconds <= 0 and candidate_summary:
            try:
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

        alias_map = _coerce_alias_map(transcript_payload, sample)
        transcript_per_speaker_messages = _speaker_message_lists_from_transcript(
            transcript_payload,
            alias_map,
        )
        summary_per_speaker_messages = _speaker_message_lists_from_summary(
            candidate_summary,
            alias_map,
        )
        speaker_notes = _coerce_speaker_notes(transcript_payload, sample)
        judge_result = None
        if candidate_summary:
            judge_result = judge.evaluate(
                source_text=source_text,
                summary_xml=candidate_summary,
                transcript_per_speaker_messages=transcript_per_speaker_messages,
                summary_per_speaker_messages=summary_per_speaker_messages,
                speaker_alias_map=alias_map,
                speaker_notes=speaker_notes,
            )
        judge_scores = judge_result.scores if judge_result is not None else None
        factualness = (
            float(judge_scores.get("Factualness", 0.0)) if judge_scores else 0.0
        )
        naturalness = (
            float(judge_scores.get("Naturalness", 0.0)) if judge_scores else 0.0
        )
        speaker_grammar_similarity = (
            float(judge_scores.get("Speaker Grammar Similarity", 0.0))
            if judge_scores
            else 0.0
        )
        metrics = SampleMetricBundle(
            factualness=factualness,
            naturalness=naturalness,
            speaker_grammar_similarity=speaker_grammar_similarity,
            composite=_compute_composite_score(
                factualness=factualness,
                naturalness=naturalness,
                speaker_grammar_similarity=speaker_grammar_similarity,
            ),
        )
        duration_seconds = round(time.perf_counter() - sample_started, 3)
        retrieval_after = event_capture.count("retrieval_stats")
        tool_after = event_capture.count("tool_invocation")
        result = SampleBenchmarkResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            episode_key=sample.episode_key,
            status="completed" if candidate_summary else "failed",
            target_seconds=target_seconds,
            candidate_seconds=round(candidate_seconds, 3),
            converged=bool(diagnostics.get("converged", False)),
            iterations_run=int(diagnostics.get("iterations_run", 0)),
            final_status=str(diagnostics.get("final_status", "unknown")),
            duration_seconds=duration_seconds,
            retrieval_events=max(0, retrieval_after - retrieval_before),
            tool_invocations=max(0, tool_after - tool_before),
            metrics=metrics,
            source_audio_path=(str(source_audio_path) if source_audio_path else None),
            speaker_samples_manifest_path=(
                str(speaker_samples_manifest_path)
                if speaker_samples_manifest_path is not None
                else None
            ),
            draft_audio_state_path=(
                str(draft_audio_state_path)
                if draft_audio_state_path is not None
                else None
            ),
            voice_clone_output_dir=(
                str(archived_voice_clone_dir)
                if archived_voice_clone_dir is not None
                else None
            ),
            voice_clone_manifest_path=(
                str(archived_voice_clone_manifest)
                if archived_voice_clone_manifest is not None
                else None
            ),
            voice_clone_merged_audio_path=(
                str(archived_voice_clone_merged_audio)
                if archived_voice_clone_merged_audio is not None
                else None
            ),
            transcript_per_speaker_messages=transcript_per_speaker_messages,
            summary_per_speaker_messages=summary_per_speaker_messages,
            speaker_alias_map=alias_map,
            speaker_notes=speaker_notes,
            judge_scores=judge_scores,
            judge_status=judge_result.status
            if judge_result is not None
            else "not_scored",
            judge_error=judge_result.error_message
            if judge_result is not None
            else None,
        )
    except Exception as exc:
        final_status, error_message = _classify_sample_failure(exc)
        duration_seconds = round(time.perf_counter() - sample_started, 3)
        retrieval_after = event_capture.count("retrieval_stats")
        tool_after = event_capture.count("tool_invocation")
        result = replace(
            _build_failed_result(
                sample=sample,
                target_seconds=target_seconds,
                final_status=final_status,
                error_message=error_message,
                duration_seconds=duration_seconds,
                retrieval_events=max(0, retrieval_after - retrieval_before),
                tool_invocations=max(0, tool_after - tool_before),
            ),
            source_audio_path=(str(source_audio_path) if source_audio_path else None),
            speaker_samples_manifest_path=(
                str(speaker_samples_manifest_path)
                if speaker_samples_manifest_path is not None
                else None
            ),
            draft_audio_state_path=(
                str(draft_audio_state_path)
                if draft_audio_state_path is not None
                else None
            ),
            voice_clone_output_dir=(
                str(archived_voice_clone_dir)
                if archived_voice_clone_dir is not None
                else None
            ),
            voice_clone_manifest_path=(
                str(archived_voice_clone_manifest)
                if archived_voice_clone_manifest is not None
                else None
            ),
            voice_clone_merged_audio_path=(
                str(archived_voice_clone_merged_audio)
                if archived_voice_clone_merged_audio is not None
                else None
            ),
        )
    result = _finalize_result_target_membership(
        result,
        infra_retry_attempts=infra_retry_attempts,
    )
    try:
        artifact_paths = artifact_recorder.finalize_sample(
            result=result,
            diagnostics=diagnostics,
            candidate_summary=candidate_summary,
        )
        if artifact_paths is not None:
            result = replace(
                result,
                sample_artifact_dir=artifact_paths.sample_artifact_dir,
                artifact_manifest_path=artifact_paths.artifact_manifest_path,
                summarizer_trace_dir=artifact_paths.summarizer_trace_dir,
                critic_trace_dir=artifact_paths.critic_trace_dir,
                judge_trace_dir=artifact_paths.judge_trace_dir,
                final_summary_path=artifact_paths.final_summary_path,
                final_critic_feedback_path=artifact_paths.final_critic_feedback_path,
                loop_diagnostics_path=artifact_paths.loop_diagnostics_path,
                sample_result_artifact_path=artifact_paths.sample_result_artifact_path,
            )
        return result
    finally:
        artifact_recorder.clear_active_sample()


async def _run_sample_with_retries(
    *,
    sample: PreparedBenchmarkSample,
    target_seconds: int,
    preloaded_transcript: Transcript | None,
    runtime: _ManagedBenchmarkRuntime | None,
    args: argparse.Namespace,
    run_root: Path,
    base_cfg: Any,
    summarizer_provider_config: dict[str, Any],
    critic_provider_config: dict[str, Any],
    judge_provider_config: dict[str, Any],
    embedding_profile: EmbeddingProfile,
    transcript_only: bool,
    calibration: Any,
    voice_clone_orchestrator: Any | None,
    logging_enabled: bool,
    force_dynamic_ports: bool,
    duration_tolerance_ratio: float,
    max_iterations: int | None,
    benchmark_audio_cfg: dict[str, Any],
) -> tuple[SampleBenchmarkResult, _ManagedBenchmarkRuntime | None, bool]:
    """Run one sample with bounded infra retries before replacement."""
    infra_retry_attempts = 0
    while True:
        if runtime is None:
            try:
                runtime = await _start_managed_runtime(
                    args=args,
                    run_root=run_root,
                    base_cfg=base_cfg,
                    summarizer_provider_config=summarizer_provider_config,
                    critic_provider_config=critic_provider_config,
                    judge_provider_config=judge_provider_config,
                    embedding_profile=embedding_profile,
                    transcript_only=transcript_only,
                    calibration=calibration,
                    voice_clone_orchestrator=voice_clone_orchestrator,
                    logging_enabled=logging_enabled,
                    force_dynamic_ports=force_dynamic_ports,
                )
                force_dynamic_ports = False
            except Exception as exc:
                final_status, error_message = _classify_sample_failure(exc)
                if final_status not in _RETRYABLE_INFRA_FINAL_STATUSES:
                    raise
                infra_retry_attempts += 1
                if infra_retry_attempts <= _SAMPLE_INFRA_RETRY_LIMIT:
                    event_bus.publish(
                        "status_message",
                        (
                            f"Infra retry {infra_retry_attempts}/{_SAMPLE_INFRA_RETRY_LIMIT} "
                            f"for {sample.sample_id} after startup failure: {error_message}"
                        ),
                    )
                    force_dynamic_ports = True
                    continue
                result = replace(
                    _build_failed_result(
                        sample=sample,
                        target_seconds=target_seconds,
                        final_status=final_status,
                        error_message=error_message,
                        duration_seconds=0.0,
                    ),
                    infra_retry_attempts=infra_retry_attempts,
                )
                return result, runtime, force_dynamic_ports

        result = await _execute_sample(
            sample=sample,
            target_seconds=target_seconds,
            preloaded_transcript=preloaded_transcript,
            orchestrator=runtime.orchestrator,
            event_capture=runtime.event_capture,
            retrieval_enabled=runtime.retrieval_enabled,
            duration_tolerance_ratio=duration_tolerance_ratio,
            max_iterations=max_iterations,
            judge=runtime.judge,
            calibration=calibration,
            run_root=run_root,
            artifact_recorder=runtime.artifact_recorder,
            audio_cfg=benchmark_audio_cfg,
            transcript_only=transcript_only,
            voice_clone_orchestrator=voice_clone_orchestrator,
            force_download_audio=bool(args.force_download_audio),
        )
        if (
            result.final_status in _RETRYABLE_INFRA_FINAL_STATUSES
            and infra_retry_attempts < _SAMPLE_INFRA_RETRY_LIMIT
        ):
            infra_retry_attempts += 1
            event_bus.publish(
                "status_message",
                (
                    f"Infra retry {infra_retry_attempts}/{_SAMPLE_INFRA_RETRY_LIMIT} "
                    f"for {sample.sample_id} after {result.final_status}: "
                    f"{result.error_message or result.final_status}"
                ),
            )
            await _stop_managed_runtime(runtime)
            runtime = None
            force_dynamic_ports = True
            continue
        if infra_retry_attempts:
            result = replace(result, infra_retry_attempts=infra_retry_attempts)
        return result, runtime, force_dynamic_ports


async def _run_benchmark_async(args: argparse.Namespace) -> BenchmarkExecutionReport:
    requested_dataset_manifest_path = Path(args.dataset_manifest).resolve()
    dataset_manifest_path = requested_dataset_manifest_path
    if args.prepare_if_missing and not dataset_manifest_path.exists():
        prepared = prepare_sporc_benchmark(
            prepared_root=dataset_manifest_path.parent,
            dataset_variant=str(args.dataset_variant),
            episode_data_path=str(args.episode_data_path or ""),
            turn_data_path=str(args.turn_data_path or ""),
            reference_wpm=float(args.reference_wpm),
            force_download=bool(args.force_download_dataset),
            status_reporter=(
                lambda message: event_bus.publish("status_message", message)
            ),
        )
        dataset_manifest_path = prepared.manifest_path
    dataset_manifest_path = ensure_compact_prepared_dataset_manifest(
        dataset_manifest_path,
    )

    dataset_manifest = load_prepared_dataset_manifest(
        dataset_manifest_path,
        include_samples=False,
    )

    raw_checkpoint_path = str(getattr(args, "checkpoint_path", "") or "").strip()
    checkpoint_path = (
        Path(raw_checkpoint_path).resolve() if raw_checkpoint_path else None
    )
    checkpoint_payload: dict[str, Any] = {}
    resumed_from_checkpoint = bool(getattr(args, "resume_from_checkpoint", False))
    resume_count = 0
    recovered_stall_count = 0
    if resumed_from_checkpoint:
        if checkpoint_path is None or not checkpoint_path.exists():
            raise BenchmarkExecutionError(
                "--resume-from-checkpoint requires an existing --checkpoint-path."
            )
        checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
        run_id = str(checkpoint_payload.get("run_id", "") or "").strip()
        if not run_id:
            raise BenchmarkExecutionError(
                f"SPoRC checkpoint is missing run_id: {checkpoint_path}"
            )
        resume_count = int(checkpoint_payload.get("resume_count", 0)) + 1
        recovered_stall_count = int(checkpoint_payload.get("recovered_stall_count", 0))
    else:
        run_id = _make_run_id()
    run_root = Path(args.output_root).resolve() / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_base_config(Path(args.config).resolve())
    _configure_benchmark_logging(
        base_cfg,
        log_file=str((run_root / "agent_interactions.log").resolve()),
        print_to_terminal=not bool(getattr(args, "emit_progress_events", False)),
    )
    transcript_only = bool(getattr(args, "transcript_only", False))
    duration_preset = str(getattr(args, "duration_preset", "") or "").strip()
    override_target_seconds = _resolve_override_target_seconds(args)
    duration_tolerance_ratio = float(base_cfg.orchestrator.duration_tolerance_ratio)
    sample_seed = (
        int(args.sample_seed)
        if getattr(args, "sample_seed", None) is not None
        else None
    )
    sample_shard_index, sample_shard_count = _resolve_sample_shard_args(args)
    shared_sample_queue_root = _resolve_shared_sample_queue_root(args)
    use_shared_sample_queue = shared_sample_queue_root is not None
    if use_shared_sample_queue and (
        sample_shard_index != 1 or sample_shard_count != 1
    ):
        raise BenchmarkExecutionError(
            "--shared-sample-queue-root cannot be combined with sample-shard selection."
        )
    sample_limit = int(args.max_samples)
    shared_sample_queue_connection: sqlite3.Connection | None = None
    candidate_samples: list[PreparedBenchmarkSample] = []
    sample_lookup: dict[str, PreparedBenchmarkSample] = {}
    if shared_sample_queue_root is not None:
        shared_sample_queue_connection = _open_shared_sample_queue_connection(
            shared_sample_queue_root
        )
        sample_order, target_sample_count = _initialize_shared_sample_queue_from_manifest(
            shared_sample_queue_connection,
            manifest_path=dataset_manifest.manifest_path,
            requested_target_sample_count=sample_limit,
            sample_seed=sample_seed,
        )
    else:
        prepared_samples = tuple(
            iter_prepared_dataset_samples(
                dataset_manifest.manifest_path,
                max_samples=None,
            )
        )
        ordered_candidates = _select_samples(
            prepared_samples,
            max_samples=0,
            sample_seed=sample_seed,
        )
        target_candidates = (
            list(ordered_candidates)
            if sample_limit <= 0
            else list(ordered_candidates[:sample_limit])
        )
        candidate_samples = _select_sample_shard(
            ordered_candidates,
            shard_index=sample_shard_index,
            shard_count=sample_shard_count,
        )
        target_sample_count = len(
            _select_sample_shard(
                target_candidates,
                shard_index=sample_shard_index,
                shard_count=sample_shard_count,
            )
        )
        sample_lookup = {sample.sample_id: sample for sample in candidate_samples}
        sample_order = [sample.sample_id for sample in candidate_samples]
    selected_sample_ids = frozenset(sample_order)
    if resumed_from_checkpoint:
        checkpoint_shard_index = int(
            checkpoint_payload.get("sample_shard_index", 1) or 1
        )
        checkpoint_shard_count = int(
            checkpoint_payload.get("sample_shard_count", 1) or 1
        )
        if (
            checkpoint_shard_index != sample_shard_index
            or checkpoint_shard_count != sample_shard_count
        ):
            raise BenchmarkExecutionError(
                "Resume checkpoint sample shard does not match the requested execute shard."
            )
        checkpoint_order = checkpoint_payload.get("sample_order", [])
        if checkpoint_order and list(checkpoint_order) != sample_order:
            raise BenchmarkExecutionError(
                "Resume checkpoint sample order no longer matches the selected benchmark samples."
            )
        checkpoint_manifest = str(
            checkpoint_payload.get("dataset_manifest_path", "") or ""
        ).strip()
        allowed_manifest_paths = {
            requested_dataset_manifest_path.resolve(),
            dataset_manifest.manifest_path.resolve(),
        }
        if checkpoint_manifest and Path(checkpoint_manifest).resolve() not in allowed_manifest_paths:
            raise BenchmarkExecutionError(
                "Resume checkpoint dataset manifest does not match the requested dataset manifest."
            )
    sample_selection_label = (
        f"seeded_random({sample_seed})" if sample_seed is not None else "manifest_order"
    )
    shard_label = (
        f"{sample_shard_index}/{sample_shard_count}"
        if sample_shard_count > 1
        else "full"
    )
    results_by_sample_id: dict[str, SampleBenchmarkResult] = {}
    for raw_result in checkpoint_payload.get("completed_results", []):
        if not isinstance(raw_result, dict):
            continue
        restored_result = _deserialize_sample_result(raw_result)
        results_by_sample_id[restored_result.sample_id] = restored_result
    unexpected_checkpoint_samples = [
        sample_id
        for sample_id in results_by_sample_id
        if sample_id not in selected_sample_ids
    ]
    if unexpected_checkpoint_samples:
        raise BenchmarkExecutionError(
            "Resume checkpoint contains samples outside the active benchmark selection: "
            f"{unexpected_checkpoint_samples}"
        )
    preloaded_transcripts: dict[str, Transcript] = {}
    preflight_results: dict[str, SampleBenchmarkResult] = {}
    curated_long_form_manifest_path = (
        DEFAULT_DATASET_MANIFEST_PATH.parent / "benchmark_medium_10_manifest.json"
    ).resolve()
    if (
        dataset_manifest.manifest_path.resolve()
        == DEFAULT_DATASET_MANIFEST_PATH.resolve()
        and override_target_seconds in {300, 900}
    ):
        event_bus.publish(
            "status_message",
            (
                "Long-form SPoRC target override selected against the default sample "
                "manifest. Incompatible teaser-style transcripts will be "
                f"preflight-skipped. Prefer curated long-form manifest: "
                f"{curated_long_form_manifest_path}"
            ),
        )

    def _load_candidate_sample(sample_id: str) -> PreparedBenchmarkSample | None:
        sample = sample_lookup.get(sample_id)
        if sample is not None:
            return sample
        if shared_sample_queue_connection is None:
            return None
        return _load_shared_sample(
            shared_sample_queue_connection,
            sample_id=sample_id,
        )

    if _count_target_results(results_by_sample_id) >= target_sample_count:
        final_report = _write_execution_report_snapshot(
            run_root=run_root,
            run_id=run_id,
            dataset_manifest=dataset_manifest,
            summarizer_provider=str(args.summarizer_provider),
            critic_provider=str(args.critic_provider or args.summarizer_provider),
            judge_provider=str(
                args.judge_provider or args.critic_provider or args.summarizer_provider
            ),
            embedding_mode=str(args.embedding_mode),
            sample_order=sample_order,
            results_by_sample_id=results_by_sample_id,
            duration_tolerance_ratio=duration_tolerance_ratio,
            transcript_only=transcript_only,
            duration_preset=duration_preset,
            override_target_seconds=override_target_seconds,
            sample_seed=sample_seed,
            target_sample_count=target_sample_count,
            checkpoint_path=checkpoint_path,
            resumed_from_checkpoint=resumed_from_checkpoint,
            resume_count=resume_count,
            sample_shard_index=sample_shard_index,
            sample_shard_count=sample_shard_count,
        )
        if checkpoint_path is not None:
            _write_checkpoint_payload(
                checkpoint_path,
                {
                    "checkpoint_version": _CHECKPOINT_VERSION,
                    "run_id": run_id,
                    "dataset_manifest_path": str(requested_dataset_manifest_path),
                    "dataset_variant": dataset_manifest.dataset_variant,
                    "duration_preset": duration_preset,
                    "override_target_seconds": override_target_seconds,
                    "transcript_only": transcript_only,
                    "sample_seed": sample_seed,
                    "sample_shard_index": sample_shard_index,
                    "sample_shard_count": sample_shard_count,
                    "target_sample_count": target_sample_count,
                    "sample_order": sample_order,
                    "completed_results": [
                        asdict(result)
                        for result in _ordered_results(
                            sample_order,
                            results_by_sample_id,
                        )
                    ],
                    "active_sample": None,
                    "resume_count": resume_count,
                    "recovered_stall_count": recovered_stall_count,
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
        if shared_sample_queue_connection is not None:
            shared_sample_queue_connection.close()
        return final_report

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

    embedding_profile: EmbeddingProfile = _resolve_embedding_profile(
        base_cfg, args.embedding_mode
    )
    logging_enabled = bool(base_cfg.get("logging", {}).get("enabled", False))
    base_audio_cfg = _coerce_mapping(
        OmegaConf.to_container(base_cfg.audio, resolve=True)
    )
    benchmark_audio_cfg = (
        base_audio_cfg
        if transcript_only
        else _prepare_benchmark_audio_cfg(base_audio_cfg, run_root=run_root)
    )
    calibration = _build_stub_calibration(
        run_root=run_root,
        audio_cfg=benchmark_audio_cfg,
    )
    voice_clone_orchestrator = None
    if not transcript_only:
        voice_clone_orchestrator = build_voice_clone_orchestrator(
            benchmark_audio_cfg,
            project_root=REPO_ROOT,
        )
        if voice_clone_orchestrator is None:
            raise BenchmarkExecutionError(
                "SPoRC benchmark requires audio.voice_clone.enabled=true unless --transcript-only is used."
            )
    summarizer_provider_config = _prepare_provider_config(
        summarizer_profile,
        role="summarizer",
        transcript_only=transcript_only,
    )
    critic_provider_config = _prepare_provider_config(
        critic_profile,
        role="critic",
        transcript_only=transcript_only,
    )
    judge_provider_config = _prepare_provider_config(
        judge_profile,
        role="judge",
        transcript_only=transcript_only,
    )
    max_iterations = _normalize_max_iterations(args.max_iterations)

    progress_tracker: _BenchmarkProgressTracker | None = None
    runtime: _ManagedBenchmarkRuntime | None = None
    force_dynamic_ports = False

    def _persist_checkpoint() -> None:
        if checkpoint_path is None:
            return
        _write_checkpoint_payload(
            checkpoint_path,
            {
                "checkpoint_version": _CHECKPOINT_VERSION,
                "run_id": run_id,
                "dataset_manifest_path": str(requested_dataset_manifest_path),
                "dataset_variant": dataset_manifest.dataset_variant,
                "duration_preset": duration_preset,
                "override_target_seconds": override_target_seconds,
                "transcript_only": transcript_only,
                "sample_seed": sample_seed,
                "sample_shard_index": sample_shard_index,
                "sample_shard_count": sample_shard_count,
                "target_sample_count": target_sample_count,
                "sample_order": sample_order,
                "completed_results": [
                    asdict(result)
                    for result in _ordered_results(sample_order, results_by_sample_id)
                ],
                "active_sample": (
                    progress_tracker.active_sample_snapshot()
                    if progress_tracker is not None
                    else None
                ),
                "resume_count": resume_count,
                "recovered_stall_count": recovered_stall_count,
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

    final_report = _write_execution_report_snapshot(
        run_root=run_root,
        run_id=run_id,
        dataset_manifest=dataset_manifest,
        summarizer_provider=summarizer_profile.provider_id,
        critic_provider=critic_profile.provider_id,
        judge_provider=judge_profile.provider_id,
        embedding_mode=embedding_profile.embedding_id,
        sample_order=sample_order,
        results_by_sample_id=results_by_sample_id,
        duration_tolerance_ratio=duration_tolerance_ratio,
        transcript_only=transcript_only,
        duration_preset=duration_preset,
        override_target_seconds=override_target_seconds,
        sample_seed=sample_seed,
        target_sample_count=target_sample_count,
        checkpoint_path=checkpoint_path,
        resumed_from_checkpoint=resumed_from_checkpoint,
        resume_count=resume_count,
        sample_shard_index=sample_shard_index,
        sample_shard_count=sample_shard_count,
    )
    progress_tracker = _BenchmarkProgressTracker(
        args=args,
        duration_preset=str(duration_preset or "dataset_default"),
        persist_callback=_persist_checkpoint,
    )
    progress_tracker.subscribe()
    progress_tracker.start()
    _persist_checkpoint()

    if resumed_from_checkpoint:
        active_sample_payload = checkpoint_payload.get("active_sample")
        if isinstance(active_sample_payload, dict):
            active_sample_id = str(
                active_sample_payload.get("sample_id", "") or ""
            ).strip()
            if active_sample_id and active_sample_id not in results_by_sample_id:
                resumed_sample = _load_candidate_sample(active_sample_id)
                if resumed_sample is not None:
                    active_target_seconds = _resolve_sample_target_seconds(
                        resumed_sample,
                        args,
                    )
                    resume_failure_reason = str(
                        getattr(args, "resume_failure_reason", "")
                        or _RESUME_FAILURE_DEFAULT_STATUS
                    ).strip()
                    if not resume_failure_reason:
                        resume_failure_reason = _RESUME_FAILURE_DEFAULT_STATUS
                    resumed_failure = _finalize_result_target_membership(
                        _build_resume_failure_result(
                            sample=resumed_sample,
                            target_seconds=active_target_seconds,
                            reason=resume_failure_reason,
                            active_sample=active_sample_payload,
                        )
                    )
                    results_by_sample_id[active_sample_id] = resumed_failure
                    if resume_failure_reason == _SWEEP_STALL_FINAL_STATUS:
                        recovered_stall_count += 1
                    if shared_sample_queue_connection is not None:
                        _mark_shared_sample_completed(
                            shared_sample_queue_connection,
                            sample_id=active_sample_id,
                            final_status=resumed_failure.final_status,
                            counts_toward_target=resumed_failure.counts_toward_target,
                            replacement_reason=resumed_failure.replacement_reason,
                        )
                    _persist_checkpoint()
                    final_report = _write_execution_report_snapshot(
                        run_root=run_root,
                        run_id=run_id,
                        dataset_manifest=dataset_manifest,
                        summarizer_provider=summarizer_profile.provider_id,
                        critic_provider=critic_profile.provider_id,
                        judge_provider=judge_profile.provider_id,
                        embedding_mode=embedding_profile.embedding_id,
                        sample_order=sample_order,
                        results_by_sample_id=results_by_sample_id,
                        duration_tolerance_ratio=duration_tolerance_ratio,
                        transcript_only=transcript_only,
                        duration_preset=duration_preset,
                        override_target_seconds=override_target_seconds,
                        sample_seed=sample_seed,
                        target_sample_count=target_sample_count,
                        checkpoint_path=checkpoint_path,
                        resumed_from_checkpoint=resumed_from_checkpoint,
                        resume_count=resume_count,
                        sample_shard_index=sample_shard_index,
                        sample_shard_count=sample_shard_count,
                    )
                    _emit_progress_event(
                        args,
                        event="sample_done",
                        duration_preset=str(duration_preset or "dataset_default"),
                        payload={
                            "sample_index": min(
                                _count_target_results(results_by_sample_id) + 1,
                                max(target_sample_count, 1),
                            ),
                            "sample_count": target_sample_count,
                            "sample_id": active_sample_id,
                            "final_status": resumed_failure.final_status,
                            "counts_toward_target": resumed_failure.counts_toward_target,
                            "composite": round(resumed_failure.metrics.composite, 4),
                            "duration_seconds": round(
                                resumed_failure.duration_seconds,
                                3,
                            ),
                        },
                    )

    event_bus.publish(
        "status_message",
        (
            f"Starting SPoRC summarizer-critic benchmark run {run_id} with "
            f"{target_sample_count} retained sample(s); summarizer={summarizer_profile.provider_id}; "
            f"critic={critic_profile.provider_id}; judge={judge_profile.provider_id}; "
            f"judge_dimensions={', '.join(DIMENSION_NAMES)}; "
            f"transcript_only={transcript_only}; "
            f"target_override={override_target_seconds if override_target_seconds is not None else 'dataset_default'}; "
            f"sample_selection={sample_selection_label}; "
            f"sample_shard={shard_label}; resume_count={resume_count}."
        ),
    )
    _emit_progress_event(
        args,
        event="run_start",
        duration_preset=str(duration_preset or "dataset_default"),
        payload={
            "run_id": run_id,
            "sample_count": target_sample_count,
            "transcript_only": transcript_only,
            "sample_seed": sample_seed,
            "resume_count": resume_count,
            "sample_shard_index": sample_shard_index,
            "sample_shard_count": sample_shard_count,
            "checkpoint_path": str(checkpoint_path)
            if checkpoint_path is not None
            else "",
        },
    )

    if _count_target_results(results_by_sample_id) >= target_sample_count:
        _emit_progress_event(
            args,
            event="run_complete",
            duration_preset=str(duration_preset or "dataset_default"),
            payload={
                "run_id": run_id,
                "status": final_report.status,
                "sample_count": target_sample_count,
                "report_path": final_report.report_path,
                "results_path": final_report.results_path,
            },
        )
        progress_tracker.stop()
        progress_tracker.unsubscribe()
        return final_report

    benchmark_started = time.perf_counter()
    try:
        shared_queue_worker_label = (
            checkpoint_path.name
            if checkpoint_path is not None
            else f"pid-{os.getpid()}"
        )
        sample_iter = iter(candidate_samples)
        while True:
            if _count_target_results(results_by_sample_id) >= target_sample_count:
                break
            if shared_sample_queue_connection is None:
                try:
                    sample = next(sample_iter)
                except StopIteration:
                    break
            else:
                claimed_sample_id = _claim_next_shared_sample_id(
                    shared_sample_queue_connection,
                    worker_label=shared_queue_worker_label,
                    worker_pid=os.getpid(),
                )
                if claimed_sample_id is None:
                    break
                sample = _load_shared_sample(
                    shared_sample_queue_connection,
                    sample_id=claimed_sample_id,
                )
                restored_result = results_by_sample_id.get(sample.sample_id)
                if restored_result is not None:
                    _mark_shared_sample_completed(
                        shared_sample_queue_connection,
                        sample_id=sample.sample_id,
                        final_status=restored_result.final_status,
                        counts_toward_target=restored_result.counts_toward_target,
                        replacement_reason=restored_result.replacement_reason,
                    )
                    continue
            if sample.sample_id in results_by_sample_id:
                continue
            active_target_seconds = _resolve_sample_target_seconds(sample, args)
            retained_before_sample = _count_target_results(results_by_sample_id)
            sample_index = min(retained_before_sample + 1, max(target_sample_count, 1))
            remaining_samples, average_text, eta_text = _progress_snapshot(
                completed_samples=retained_before_sample,
                total_samples=target_sample_count,
                elapsed_seconds=time.perf_counter() - benchmark_started,
            )
            progress_tracker.activate_sample(
                sample_index=sample_index,
                sample_count=target_sample_count,
                sample_id=sample.sample_id,
                target_seconds=active_target_seconds,
            )
            _emit_progress_event(
                args,
                event="sample_start",
                duration_preset=str(duration_preset or "dataset_default"),
                payload={
                    "sample_index": sample_index,
                    "sample_count": target_sample_count,
                    "sample_id": sample.sample_id,
                    "target_seconds": active_target_seconds,
                    "max_iterations": (
                        "unbounded" if max_iterations is None else str(max_iterations)
                    ),
                },
            )
            event_bus.publish(
                "status_message",
                (
                    f"Sample slot {sample_index}/{target_sample_count} start: "
                    f"{sample.sample_id} target={active_target_seconds}s "
                    f"max_iterations={'unbounded' if max_iterations is None else str(max_iterations)} "
                    f"remaining={remaining_samples} rolling_avg={average_text} eta={eta_text}"
                ),
            )
            preflight_result = preflight_results.get(sample.sample_id)
            preloaded_transcript = preloaded_transcripts.get(sample.sample_id)
            if preflight_result is None and preloaded_transcript is None:
                (
                    preloaded_transcript,
                    preflight_stats,
                    preflight_reason,
                ) = _evaluate_sample_preflight(
                    sample=sample,
                    target_seconds=active_target_seconds,
                    reference_wpm=float(args.reference_wpm),
                )
                if preflight_reason is not None:
                    preflight_result = _build_preflight_failed_result(
                        sample=sample,
                        target_seconds=active_target_seconds,
                        reason=preflight_reason,
                        stats=preflight_stats,
                    )
                    preflight_results[sample.sample_id] = preflight_result
                    event_bus.publish(
                        "status_message",
                        (
                            f"Preflight skip: {sample.sample_id} "
                            f"target={active_target_seconds}s "
                            f"reason={preflight_result.preflight_reason}"
                        ),
                    )
                else:
                    preloaded_transcripts[sample.sample_id] = preloaded_transcript
            if preflight_result is not None:
                result = preflight_result
            else:
                result, runtime, force_dynamic_ports = await _run_sample_with_retries(
                    sample=sample,
                    target_seconds=active_target_seconds,
                    preloaded_transcript=preloaded_transcript,
                    runtime=runtime,
                    args=args,
                    run_root=run_root,
                    base_cfg=base_cfg,
                    summarizer_provider_config=summarizer_provider_config,
                    critic_provider_config=critic_provider_config,
                    judge_provider_config=judge_provider_config,
                    embedding_profile=embedding_profile,
                    transcript_only=transcript_only,
                    calibration=calibration,
                    voice_clone_orchestrator=voice_clone_orchestrator,
                    logging_enabled=logging_enabled,
                    force_dynamic_ports=force_dynamic_ports,
                    duration_tolerance_ratio=duration_tolerance_ratio,
                    max_iterations=max_iterations,
                    benchmark_audio_cfg=benchmark_audio_cfg,
                )
            results_by_sample_id[sample.sample_id] = result
            if shared_sample_queue_connection is not None:
                _mark_shared_sample_completed(
                    shared_sample_queue_connection,
                    sample_id=sample.sample_id,
                    final_status=result.final_status,
                    counts_toward_target=result.counts_toward_target,
                    replacement_reason=result.replacement_reason,
                )
            progress_tracker.clear_active_sample()
            _persist_checkpoint()
            final_report = _write_execution_report_snapshot(
                run_root=run_root,
                run_id=run_id,
                dataset_manifest=dataset_manifest,
                summarizer_provider=summarizer_profile.provider_id,
                critic_provider=critic_profile.provider_id,
                judge_provider=judge_profile.provider_id,
                embedding_mode=embedding_profile.embedding_id,
                sample_order=sample_order,
                results_by_sample_id=results_by_sample_id,
                duration_tolerance_ratio=duration_tolerance_ratio,
                transcript_only=transcript_only,
                duration_preset=duration_preset,
                override_target_seconds=override_target_seconds,
                sample_seed=sample_seed,
                target_sample_count=target_sample_count,
                checkpoint_path=checkpoint_path,
                resumed_from_checkpoint=resumed_from_checkpoint,
                resume_count=resume_count,
                sample_shard_index=sample_shard_index,
                sample_shard_count=sample_shard_count,
            )
            retained_after_sample = _count_target_results(results_by_sample_id)
            remaining_samples, average_text, eta_text = _progress_snapshot(
                completed_samples=retained_after_sample,
                total_samples=target_sample_count,
                elapsed_seconds=time.perf_counter() - benchmark_started,
            )
            _emit_progress_event(
                args,
                event="sample_done",
                duration_preset=str(duration_preset or "dataset_default"),
                payload={
                    "sample_index": sample_index,
                    "sample_count": target_sample_count,
                    "sample_id": sample.sample_id,
                    "final_status": result.final_status,
                    "counts_toward_target": result.counts_toward_target,
                    "composite": round(result.metrics.composite, 4),
                    "duration_seconds": round(result.duration_seconds, 3),
                },
            )
            event_bus.publish(
                "status_message",
                (
                    f"Sample slot {sample_index}/{target_sample_count} done: "
                    f"{sample.sample_id} status={result.final_status} "
                    f"counted={result.counts_toward_target} "
                    f"remaining={remaining_samples} rolling_avg={average_text} eta={eta_text} "
                    f"composite={result.metrics.composite:.3f} elapsed={result.duration_seconds:.1f}s"
                ),
            )
            if result.status == "failed" and preflight_result is None:
                await _stop_managed_runtime(runtime)
                runtime = None
                force_dynamic_ports = True

        _emit_progress_event(
            args,
            event="run_complete",
            duration_preset=str(duration_preset or "dataset_default"),
            payload={
                "run_id": run_id,
                "status": final_report.status,
                "sample_count": target_sample_count,
                "report_path": final_report.report_path,
                "results_path": final_report.results_path,
            },
        )
        return final_report
    finally:
        if progress_tracker is not None:
            progress_tracker.stop()
            progress_tracker.unsubscribe()
        if shared_sample_queue_connection is not None:
            shared_sample_queue_connection.close()
        await _stop_managed_runtime(runtime)


def execute_benchmark(args: argparse.Namespace) -> BenchmarkExecutionReport:
    """Run the SPoRC summarizer-critic benchmark and return the report."""
    try:
        return asyncio.run(_run_benchmark_async(args))
    except (BenchmarkExecutionError, DatasetPreparationError, MatrixConfigError) as exc:
        raise BenchmarkExecutionError(str(exc)) from exc


def build_execute_parser(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags for the SPoRC execute subcommand."""
    parser.add_argument(
        "--dataset-manifest", default=str(DEFAULT_DATASET_MANIFEST_PATH)
    )
    parser.add_argument("--prepare-if-missing", action="store_true")
    parser.add_argument("--force-download-dataset", action="store_true")
    parser.add_argument(
        "--force-download-audio",
        action="store_true",
        help="Redownload cached episode audio before speaker-sample generation.",
    )
    parser.add_argument(
        "--dataset-variant",
        choices=("sample", "full"),
        default=DEFAULT_DATASET_VARIANT,
    )
    parser.add_argument("--episode-data-path", default="")
    parser.add_argument("--turn-data-path", default="")
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
    parser.add_argument("--judge-source-char-limit", type=int, default=30000)
    parser.add_argument(
        "--duration-preset",
        choices=tuple(DURATION_PRESET_SECONDS.keys()),
        default="",
        help="Override every sample target with a named duration preset.",
    )
    parser.add_argument(
        "--target-seconds-override",
        type=int,
        default=None,
        help="Override every sample target with an explicit positive duration in seconds.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help=(
            "Retained sample target for execution. Replacement-eligible infra/preflight "
            "outcomes do not consume this target. Use 0 to run all prepared SPoRC examples."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help=(
            "Deterministically randomize sample ordering and prefix selection by "
            "seed. Omit to preserve manifest order."
        ),
    )
    parser.add_argument(
        "--transcript-only",
        action="store_true",
        help=(
            "Benchmark the summarizer-critic loop from transcript only. "
            "Skip source-audio downloads, speaker-sample generation, and live-draft voice cloning."
        ),
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Optional iteration cap. Omit for unbounded iterations.",
    )
    parser.add_argument(
        "--summarizer-timeout-seconds",
        type=float,
        default=None,
        help="Optional summarizer request timeout. Omit for no timeout.",
    )
    parser.add_argument(
        "--critic-timeout-seconds",
        type=float,
        default=None,
        help="Optional critic request timeout. Omit for no timeout.",
    )
    parser.add_argument(
        "--retrieval-timeout-seconds",
        type=float,
        default=None,
        help="Optional retrieval request timeout. Omit for no timeout.",
    )
    parser.add_argument(
        "--index-timeout-seconds",
        type=float,
        default=None,
        help="Optional indexing request timeout. Omit for no timeout.",
    )
    parser.add_argument("--retrieval-port", type=int, default=None)
    parser.add_argument("--summarizer-port", type=int, default=None)
    parser.add_argument("--critic-port", type=int, default=None)
    parser.add_argument(
        "--disable-stream-output",
        action="store_true",
        help="Suppress streamed token rendering for this execute run.",
    )
    parser.add_argument(
        "--emit-progress-events",
        action="store_true",
        help="Emit machine-readable sample progress events on stdout.",
    )
    parser.add_argument(
        "--emit-progress-heartbeats",
        action="store_true",
        help="Emit periodic machine-readable heartbeat events for the active sample.",
    )
    parser.add_argument(
        "--progress-heartbeat-seconds",
        type=float,
        default=30.0,
        help="Heartbeat interval for --emit-progress-heartbeats.",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="",
        help="Optional execute checkpoint path used for sweep recovery.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume a prior execute run from --checkpoint-path.",
    )
    parser.add_argument(
        "--resume-failure-reason",
        default=_RESUME_FAILURE_DEFAULT_STATUS,
        help="Final-status label used when resuming an orphaned active sample.",
    )
    parser.add_argument(
        "--shared-sample-queue-root",
        default="",
        help=(
            "Optional internal preset-level queue root used by duration-sweep "
            "worker pooling."
        ),
    )
    parser.add_argument(
        "--sample-shard-index",
        type=int,
        default=1,
        help=(
            "Optional 1-based shard index used by duration-sweep fan-out. "
            "Run the whole selected sample set by leaving this at 1."
        ),
    )
    parser.add_argument(
        "--sample-shard-count",
        type=int,
        default=1,
        help=(
            "Optional shard count used by duration-sweep fan-out. "
            "Each shard executes a disjoint modulo partition of the selected samples."
        ),
    )
