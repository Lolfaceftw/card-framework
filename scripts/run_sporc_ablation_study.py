"""Plan, run, and report the SPoRC summarizer-critic ablation study.

The ablation hierarchy is intentionally aligned with the human-correlation
findings: DeepSeek V4 Pro non-thinking factualness is the primary automatic
endpoint, duration/control metrics are secondary safeguards, and naturalness,
speaker-grammar similarity, and composite are reported as exploratory.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable

from omegaconf import OmegaConf

from card_framework.benchmark.summarizer_critic.reporting_utils import (
    format_confidence_interval,
    mean_confidence_interval_95,
    render_markdown_table,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_PREPARED_FULL_ROOT,
    DURATION_PRESET_SECONDS,
)
from card_framework.shared.paths import DEFAULT_CONFIG_PATH, REPO_ROOT


DEFAULT_DATASET_MANIFEST = (
    DEFAULT_PREPARED_FULL_ROOT / "benchmark_longform_15k_max3_manifest.json"
).resolve()
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts" / "summarizer_critic_benchmark" / "sporc_ablation_studies"
).resolve()
DEFAULT_PAPER_SECTION_PATH = (
    REPO_ROOT / "research" / "EEE_199_IEEE" / "sections" / "ablation_study.tex"
).resolve()
DEFAULT_PAPER_MAIN_PATH = (
    REPO_ROOT / "research" / "EEE_199_IEEE" / "main.tex"
).resolve()
VLLM_THINKING_PROVIDER_ID = "vllm_qwen36_27b_thinking"
VLLM_NONTHINKING_PROVIDER_ID = "vllm_qwen36_27b_nonthinking"
DEEPSEEK_JUDGE_PROVIDER_ID = "deepseek_v4_pro_nonthinking_judge"
PRIMARY_METRIC = "factualness"
EXPLORATORY_METRICS = (
    "naturalness",
    "speaker_grammar_similarity",
    "composite",
)
PROGRESS_EVENT_KEY = "sporc_progress_event"


@dataclass(frozen=True)
class AblationCell:
    """One concrete SPoRC execute run in the ablation matrix."""

    cell_id: str
    ablation: str
    summarizer_thinking: bool
    critic_thinking: bool
    summarizer_provider: str
    critic_provider: str
    judge_provider: str
    memory_enabled: bool
    max_iterations: int
    config_path: str
    output_root: str
    checkpoint_path: str
    log_path: str
    command: tuple[str, ...]


@dataclass
class AblationProgressSnapshot:
    """Compact progress state for tests and terminal rendering."""

    completed_samples: int
    total_samples: int
    completed_cells: int
    total_cells: int
    ema_sample_seconds: float | None
    eta_seconds: float | None
    current_cell_id: str
    current_sample_label: str
    current_stage: str


class AblationProgressTracker:
    """Render a lightweight run-level progress bar and ETA for ablation cells."""

    def __init__(
        self,
        *,
        total_cells: int,
        total_samples: int,
        eta_learning_rate: float,
        progress_interval_seconds: float,
        enabled: bool = True,
    ) -> None:
        self.total_cells = max(0, int(total_cells))
        self.total_samples = max(0, int(total_samples))
        self.eta_learning_rate = min(1.0, max(0.0, float(eta_learning_rate)))
        self.progress_interval_seconds = max(0.1, float(progress_interval_seconds))
        self.enabled = bool(enabled)
        self.completed_cells = 0
        self.completed_samples = 0
        self.ema_sample_seconds: float | None = None
        self.started_monotonic = time.monotonic()
        self.last_render_monotonic = 0.0
        self.last_line_length = 0
        self.current_cell_id = ""
        self.current_cell_index = 0
        self.current_cell_sample_count = 0
        self.current_cell_completed_at_start = 0
        self.current_sample_index: int | None = None
        self.current_sample_count: int | None = None
        self.current_sample_id = ""
        self.current_stage = ""
        self.current_iteration = ""
        self.active_sample_started_monotonic: float | None = None

    def start_cell(
        self,
        *,
        cell: AblationCell,
        cell_index: int,
        precompleted_sample_count: int = 0,
    ) -> None:
        self.current_cell_id = cell.cell_id
        self.current_cell_index = int(cell_index)
        self.current_cell_sample_count = 0
        self.current_cell_completed_at_start = self.completed_samples
        self.completed_samples = min(
            self.total_samples,
            self.completed_samples + max(0, int(precompleted_sample_count)),
        )
        self.current_sample_index = None
        self.current_sample_count = None
        self.current_sample_id = ""
        self.current_stage = "starting"
        self.current_iteration = ""
        self.active_sample_started_monotonic = None
        self.render(force=True)

    def skip_cell(self, *, cell: AblationCell, sample_count: int) -> None:
        self.current_cell_id = cell.cell_id
        self.completed_cells = min(self.total_cells, self.completed_cells + 1)
        self.completed_samples = min(
            self.total_samples,
            self.completed_samples + max(0, int(sample_count)),
        )
        self.current_stage = "skipped"
        self.render(force=True)

    def handle_event(self, *, cell: AblationCell, payload: dict[str, Any]) -> None:
        event = str(payload.get(PROGRESS_EVENT_KEY, "") or "").strip()
        if event == "run_start":
            self.current_cell_id = cell.cell_id
            self.current_cell_sample_count = max(
                0,
                int(payload.get("sample_count", self.current_cell_sample_count) or 0),
            )
            self.current_sample_count = self.current_cell_sample_count or None
            self.current_stage = "run_start"
            self.render(force=True)
            return
        if event == "sample_start":
            self.current_sample_index = _coerce_optional_int(payload.get("sample_index"))
            self.current_sample_count = _coerce_optional_int(payload.get("sample_count"))
            self.current_sample_id = str(payload.get("sample_id", "") or "")
            self.current_stage = "sample_start"
            self.current_iteration = str(payload.get("max_iterations", "") or "")
            self.active_sample_started_monotonic = time.monotonic()
            self.render(force=True)
            return
        if event == "sample_heartbeat":
            self.current_sample_index = _coerce_optional_int(
                payload.get("sample_index")
            )
            self.current_sample_count = _coerce_optional_int(
                payload.get("sample_count")
            )
            self.current_sample_id = str(payload.get("sample_id", "") or "")
            self.current_stage = str(payload.get("stage", "") or "heartbeat")
            raw_iteration = payload.get("iteration")
            self.current_iteration = "" if raw_iteration is None else str(raw_iteration)
            self.render(force=False)
            return
        if event == "sample_done":
            self.current_sample_index = _coerce_optional_int(payload.get("sample_index"))
            self.current_sample_count = _coerce_optional_int(payload.get("sample_count"))
            self.current_sample_id = str(payload.get("sample_id", "") or "")
            self.current_stage = str(payload.get("final_status", "") or "sample_done")
            self._update_sample_average(payload)
            if bool(payload.get("counts_toward_target", False)):
                self.completed_samples = min(
                    self.total_samples,
                    self.completed_samples + 1,
                )
            self.render(force=True)
            return
        if event == "run_complete":
            self.current_stage = str(payload.get("status", "") or "run_complete")
            target_after_cell = self.current_cell_completed_at_start + (
                self.current_cell_sample_count or 0
            )
            if target_after_cell > self.completed_samples:
                self.completed_samples = min(self.total_samples, target_after_cell)
            self.completed_cells = min(self.total_cells, self.completed_cells + 1)
            self.render(force=True)

    def _update_sample_average(self, payload: dict[str, Any]) -> None:
        duration_seconds = _coerce_optional_float(payload.get("duration_seconds"))
        if duration_seconds is None and self.active_sample_started_monotonic is not None:
            duration_seconds = time.monotonic() - self.active_sample_started_monotonic
        if duration_seconds is None or duration_seconds <= 0:
            return
        if self.ema_sample_seconds is None:
            self.ema_sample_seconds = duration_seconds
            return
        alpha = self.eta_learning_rate
        self.ema_sample_seconds = (
            alpha * duration_seconds + (1.0 - alpha) * self.ema_sample_seconds
        )

    def eta_seconds(self) -> float | None:
        remaining = max(0, self.total_samples - self.completed_samples)
        if remaining <= 0:
            return 0.0
        if self.ema_sample_seconds is not None:
            return remaining * self.ema_sample_seconds
        elapsed = max(0.0, time.monotonic() - self.started_monotonic)
        if self.completed_samples > 0:
            return remaining * (elapsed / self.completed_samples)
        return None

    def snapshot(self) -> AblationProgressSnapshot:
        return AblationProgressSnapshot(
            completed_samples=self.completed_samples,
            total_samples=self.total_samples,
            completed_cells=self.completed_cells,
            total_cells=self.total_cells,
            ema_sample_seconds=self.ema_sample_seconds,
            eta_seconds=self.eta_seconds(),
            current_cell_id=self.current_cell_id,
            current_sample_label=self._sample_label(),
            current_stage=self.current_stage,
        )

    def render(self, *, force: bool) -> None:
        if not self.enabled:
            return
        now = time.monotonic()
        if not force and now - self.last_render_monotonic < self.progress_interval_seconds:
            return
        self.last_render_monotonic = now
        line = self._render_line()
        if sys.stdout.isatty():
            padding = " " * max(0, self.last_line_length - len(line))
            sys.stdout.write("\r" + line + padding)
            sys.stdout.flush()
            self.last_line_length = len(line)
            return
        print(line, flush=True)

    def clear_line(self) -> None:
        if not (self.enabled and sys.stdout.isatty() and self.last_line_length):
            return
        sys.stdout.write("\r" + (" " * self.last_line_length) + "\r")
        sys.stdout.flush()
        self.last_line_length = 0

    def finish(self) -> None:
        if self.enabled and sys.stdout.isatty() and self.last_line_length:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.last_line_length = 0

    def _render_line(self) -> str:
        bar = _progress_bar(
            completed=self.completed_samples,
            total=self.total_samples,
        )
        eta = _format_seconds(self.eta_seconds())
        avg = _format_seconds(self.ema_sample_seconds)
        stage = self.current_stage or "pending"
        iteration = f" iter={self.current_iteration}" if self.current_iteration else ""
        return (
            f"Ablation {bar} {self.completed_samples}/{self.total_samples} samples "
            f"| cells {self.completed_cells}/{self.total_cells} "
            f"| ETA {eta} | avg {avg} "
            f"| cell {self.current_cell_index}/{self.total_cells} {self.current_cell_id} "
            f"| sample {self._sample_label()} {stage}{iteration}"
        )

    def _sample_label(self) -> str:
        if self.current_sample_index is None:
            return "-/-"
        sample_count = self.current_sample_count or self.current_cell_sample_count or "?"
        return f"{self.current_sample_index}/{sample_count}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _first_env(names: Iterable[str], default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _bool_label(value: bool) -> str:
    return "thinking" if value else "nonthinking"


def _provider_id_for_thinking(enabled: bool) -> str:
    return VLLM_THINKING_PROVIDER_ID if enabled else VLLM_NONTHINKING_PROVIDER_ID


def _optional_positive_int(value: int | None) -> int | None:
    """Return a positive integer or None for unset provider controls."""
    if value is None:
        return None
    value_int = int(value)
    return value_int if value_int > 0 else None


def _coerce_optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _coerce_optional_float(value: object) -> float | None:
    try:
        return float(value) if value is not None and str(value).strip() else None
    except (TypeError, ValueError):
        return None


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "learning"
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _progress_bar(*, completed: int, total: int) -> str:
    width = max(10, min(40, shutil.get_terminal_size((100, 20)).columns // 4))
    if total <= 0:
        return "[" + ("?" * width) + "]"
    fraction = min(1.0, max(0.0, completed / total))
    filled = int(round(width * fraction))
    return "[" + ("#" * filled) + ("." * (width - filled)) + "]"


def _parse_progress_event(line: str) -> dict[str, Any] | None:
    text = line.strip()
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or PROGRESS_EVENT_KEY not in payload:
        return None
    return payload


def _mode_pairs(thinking_grid: str) -> tuple[tuple[bool, bool], ...]:
    if thinking_grid == "matched":
        return ((True, True), (False, False))
    return ((True, True), (True, False), (False, True), (False, False))


def _sanitize_cell_id(raw_value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in raw_value).strip("_")


def _write_provider_profiles(args: argparse.Namespace, output_root: Path) -> Path:
    """Write a run-local provider profile file with explicit thinking settings."""
    provider_profiles_path = output_root / "provider_profiles.ablation.yaml"
    provider_profiles_path.parent.mkdir(parents=True, exist_ok=True)
    vllm_thinking_llm = {
        "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
        "base_url": args.vllm_base_url,
        "api_key": args.vllm_api_key,
        "model": args.vllm_model,
        "request_timeout_seconds": args.vllm_request_timeout_seconds,
        "sporc_enable_thinking": True,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": True},
        },
    }
    vllm_nonthinking_llm = {
        "_target_": "card_framework.providers.vllm_provider.VLLMProvider",
        "base_url": args.vllm_base_url,
        "api_key": args.vllm_api_key,
        "model": args.vllm_model,
        "request_timeout_seconds": args.vllm_request_timeout_seconds,
        "sporc_enable_thinking": False,
        "extra_body": {
            "chat_template_kwargs": {"enable_thinking": False},
        },
    }
    vllm_max_tokens = _optional_positive_int(args.vllm_max_tokens)
    if vllm_max_tokens is not None:
        vllm_thinking_llm["default_max_tokens"] = vllm_max_tokens
        vllm_nonthinking_llm["default_max_tokens"] = vllm_max_tokens
    profiles = {
        "providers": [
            {
                "id": VLLM_THINKING_PROVIDER_ID,
                "description": (
                    "Qwen/Qwen3.6-27B served through vLLM with thinking enabled "
                    "for SPoRC ablation runs."
                ),
                "required_env": [],
                "llm": vllm_thinking_llm,
            },
            {
                "id": VLLM_NONTHINKING_PROVIDER_ID,
                "description": (
                    "Qwen/Qwen3.6-27B served through vLLM with thinking disabled "
                    "for SPoRC ablation runs."
                ),
                "required_env": [],
                "llm": vllm_nonthinking_llm,
            },
            {
                "id": DEEPSEEK_JUDGE_PROVIDER_ID,
                "description": (
                    "DeepSeek V4 Pro non-thinking judge used for the "
                    "human-aligned factualness endpoint."
                ),
                "required_env": [args.deepseek_api_key_env],
                "llm": {
                    "_target_": (
                        "card_framework.providers.deepseek_provider.DeepSeekProvider"
                    ),
                    "api_key": f"$ENV{{{args.deepseek_api_key_env}}}",
                    "model": args.deepseek_model,
                    "base_url": args.deepseek_base_url,
                    "request_timeout_seconds": args.deepseek_request_timeout_seconds,
                    "default_max_tokens": args.deepseek_max_tokens,
                    "thinking_mode": "disabled",
                    "stream": False,
                    "temperature": args.deepseek_temperature,
                    "response_format": {"type": "json_object"},
                },
            },
        ]
    }
    OmegaConf.save(config=OmegaConf.create(profiles), f=provider_profiles_path)
    return provider_profiles_path.resolve()


def _write_cell_config(
    *,
    base_config_path: Path,
    cell_root: Path,
    memory_enabled: bool,
) -> Path:
    """Write a cell-local config so loop-memory artifacts do not leak across cells."""
    cfg = OmegaConf.load(base_config_path)
    if "orchestrator" not in cfg:
        cfg.orchestrator = {}
    if "loop_memory" not in cfg.orchestrator:
        cfg.orchestrator.loop_memory = {}
    if memory_enabled:
        memory_path = cell_root / "loop_memory" / "summarizer_loop_memory.json"
        cfg.orchestrator.loop_memory.artifact_path = str(memory_path.resolve())
    else:
        cfg.orchestrator.loop_memory.artifact_path = ""
    config_path = cell_root / "config.ablation.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=config_path)
    return config_path.resolve()


def _build_execute_command(
    *,
    args: argparse.Namespace,
    provider_profiles_path: Path,
    cell_root: Path,
    cell_config_path: Path,
    summarizer_provider: str,
    critic_provider: str,
    max_iterations: int,
) -> tuple[str, ...]:
    command: list[str] = [
        sys.executable,
        "-m",
        "card_framework.benchmark.summarizer_critic.sporc",
        "execute",
        "--dataset-manifest",
        str(Path(args.dataset_manifest).resolve()),
        "--output-root",
        str((cell_root / "runs").resolve()),
        "--provider-profiles",
        str(provider_profiles_path),
        "--config",
        str(cell_config_path),
        "--summarizer-provider",
        summarizer_provider,
        "--critic-provider",
        critic_provider,
        "--judge-provider",
        DEEPSEEK_JUDGE_PROVIDER_ID,
        "--embedding-mode",
        "disabled",
        "--duration-preset",
        args.duration_preset,
        "--max-samples",
        str(args.max_samples),
        "--sample-seed",
        str(args.sample_seed),
        "--transcript-only",
        "--max-iterations",
        str(max_iterations),
        "--checkpoint-path",
        str((cell_root / "checkpoint.json").resolve()),
        "--summarizer-timeout-seconds",
        str(args.summarizer_timeout_seconds),
        "--critic-timeout-seconds",
        str(args.critic_timeout_seconds),
        "--retrieval-timeout-seconds",
        str(args.retrieval_timeout_seconds),
        "--index-timeout-seconds",
        str(args.index_timeout_seconds),
        "--disable-stream-output",
        "--emit-progress-events",
        "--emit-progress-heartbeats",
        "--progress-heartbeat-seconds",
        str(args.progress_interval_seconds),
    ]
    if args.prepare_if_missing:
        command.append("--prepare-if-missing")
    return tuple(command)


def _build_ablation_cells(
    args: argparse.Namespace,
    *,
    output_root: Path,
    provider_profiles_path: Path,
) -> tuple[AblationCell, ...]:
    base_config_path = Path(args.config).resolve()
    cells: list[AblationCell] = []
    ablations = tuple(args.ablations or ("baseline", "no_memory", "no_critic"))
    for summarizer_thinking, critic_thinking in _mode_pairs(args.thinking_grid):
        mode_prefix = (
            f"s_{_bool_label(summarizer_thinking)}__"
            f"c_{_bool_label(critic_thinking)}"
        )
        for ablation in ablations:
            memory_enabled = ablation != "no_memory"
            max_iterations = (
                0 if ablation == "no_critic" else int(args.critic_feedback_budget)
            )
            cell_id = _sanitize_cell_id(f"{mode_prefix}__{ablation}")
            cell_root = output_root / "cells" / cell_id
            cell_config_path = _write_cell_config(
                base_config_path=base_config_path,
                cell_root=cell_root,
                memory_enabled=memory_enabled,
            )
            summarizer_provider = _provider_id_for_thinking(summarizer_thinking)
            critic_provider = _provider_id_for_thinking(critic_thinking)
            command = _build_execute_command(
                args=args,
                provider_profiles_path=provider_profiles_path,
                cell_root=cell_root,
                cell_config_path=cell_config_path,
                summarizer_provider=summarizer_provider,
                critic_provider=critic_provider,
                max_iterations=max_iterations,
            )
            cells.append(
                AblationCell(
                    cell_id=cell_id,
                    ablation=ablation,
                    summarizer_thinking=summarizer_thinking,
                    critic_thinking=critic_thinking,
                    summarizer_provider=summarizer_provider,
                    critic_provider=critic_provider,
                    judge_provider=DEEPSEEK_JUDGE_PROVIDER_ID,
                    memory_enabled=memory_enabled,
                    max_iterations=max_iterations,
                    config_path=str(cell_config_path),
                    output_root=str((cell_root / "runs").resolve()),
                    checkpoint_path=str((cell_root / "checkpoint.json").resolve()),
                    log_path=str((cell_root / "execute.log").resolve()),
                    command=command,
                )
            )
    return tuple(cells)


def _quote_command(command: tuple[str, ...]) -> str:
    return subprocess.list2cmdline(list(command))


def _write_plan(
    *,
    args: argparse.Namespace,
    output_root: Path,
    provider_profiles_path: Path,
    cells: tuple[AblationCell, ...],
) -> Path:
    plan_path = output_root / "ablation_plan.json"
    payload = {
        "generated_at_utc": _utc_now(),
        "score_hierarchy": {
            "primary": PRIMARY_METRIC,
            "secondary_controls": [
                "duration_within_tolerance",
                "mean_absolute_duration_delta_seconds",
                "success_rate",
                "replacement_count",
            ],
            "exploratory": list(EXPLORATORY_METRICS),
        },
        "protocol": {
            "dataset_manifest": str(Path(args.dataset_manifest).resolve()),
            "sample_seed": args.sample_seed,
            "max_samples_per_cell": args.max_samples,
            "duration_preset": args.duration_preset,
            "target_seconds": DURATION_PRESET_SECONDS[args.duration_preset],
            "thinking_grid": args.thinking_grid,
            "critic_feedback_budget": args.critic_feedback_budget,
            "embedding_mode": "disabled",
            "transcript_only": True,
            "vllm_model": args.vllm_model,
            "vllm_base_url": args.vllm_base_url,
            "vllm_default_max_tokens": _optional_positive_int(args.vllm_max_tokens),
            "deepseek_model": args.deepseek_model,
            "deepseek_default_max_tokens": args.deepseek_max_tokens,
            "deepseek_thinking_mode": "disabled",
            "eta_learning_rate": args.eta_learning_rate,
            "progress_interval_seconds": args.progress_interval_seconds,
        },
        "provider_profiles_path": str(provider_profiles_path),
        "cells": [
            {
                **asdict(cell),
                "command_display": _quote_command(cell.command),
            }
            for cell in cells
        ],
    }
    _write_json(plan_path, payload)
    return plan_path.resolve()


def _latest_cell_report(cell: AblationCell) -> Path | None:
    run_root = Path(cell.output_root)
    candidates = sorted(
        (
            path
            for path in run_root.rglob("report.json")
            if path.is_file() and path.parent != run_root
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


def _cell_report_completed(report_path: Path | None) -> bool:
    """Return whether a prior cell report completed successfully."""
    if report_path is None:
        return False
    try:
        payload = _load_report(report_path)
    except (OSError, json.JSONDecodeError):
        return False
    return str(payload.get("status", "")).strip().lower() == "completed"


def _cell_needs_extension(
    report_path: Path | None,
    *,
    requested_sample_count: int,
) -> bool:
    """Return whether a completed cell is short of the newly requested target."""
    if requested_sample_count <= 0 or not _cell_report_completed(report_path):
        return False
    completed_count = _completed_cell_sample_count(report_path, fallback=0)
    return completed_count < requested_sample_count


def _command_with_resume_from_checkpoint(command: tuple[str, ...]) -> tuple[str, ...]:
    if "--resume-from-checkpoint" in command:
        return command
    return (*command, "--resume-from-checkpoint")


def _run_cell(
    cell: AblationCell,
    *,
    progress: AblationProgressTracker | None = None,
    command: tuple[str, ...] | None = None,
) -> int:
    effective_command = command or cell.command
    log_path = Path(cell.log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write(f"$ {_quote_command(effective_command)}\n\n")
        log_handle.flush()
        process = subprocess.Popen(
            effective_command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log_handle.write(line)
            log_handle.flush()
            progress_event = _parse_progress_event(line)
            if progress_event is not None:
                if progress is not None:
                    progress.handle_event(cell=cell, payload=progress_event)
                continue
            if progress is not None:
                progress.clear_line()
            print(line, end="")
        return process.wait()


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sample_metric(sample: dict[str, Any], metric_name: str) -> float:
    metrics = sample.get("metrics")
    if not isinstance(metrics, dict):
        return 0.0
    return float(metrics.get(metric_name, 0.0) or 0.0)


def _mean(values: Iterable[float]) -> float:
    sample = [float(value) for value in values]
    if not sample:
        return 0.0
    return sum(sample) / len(sample)


def _summarize_cell_report(
    *,
    cell: AblationCell,
    report_path: Path,
) -> dict[str, Any]:
    report = _load_report(report_path)
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    samples = report.get("samples") if isinstance(report.get("samples"), list) else []
    retained_executed = [
        sample
        for sample in samples
        if bool(sample.get("counts_toward_target", True))
        and str(sample.get("status", "")) != "skipped"
    ]
    tolerance_ratio = float(report.get("duration_tolerance_ratio", 0.0) or 0.0)
    within_tolerance = [
        sample
        for sample in retained_executed
        if abs(
            float(sample.get("candidate_seconds", 0.0) or 0.0)
            - float(sample.get("target_seconds", 0.0) or 0.0)
        )
        <= float(sample.get("target_seconds", 0.0) or 0.0) * tolerance_ratio
    ]
    factualness_values = [
        _sample_metric(sample, "factualness") for sample in retained_executed
    ]
    naturalness_values = [
        _sample_metric(sample, "naturalness") for sample in retained_executed
    ]
    speaker_values = [
        _sample_metric(sample, "speaker_grammar_similarity")
        for sample in retained_executed
    ]
    composite_values = [
        _sample_metric(sample, "composite") for sample in retained_executed
    ]
    duration_delta_values = [
        abs(
            float(sample.get("candidate_seconds", 0.0) or 0.0)
            - float(sample.get("target_seconds", 0.0) or 0.0)
        )
        for sample in retained_executed
    ]
    return {
        "cell_id": cell.cell_id,
        "ablation": cell.ablation,
        "summarizer_thinking": cell.summarizer_thinking,
        "critic_thinking": cell.critic_thinking,
        "mode": (
            f"S:{_bool_label(cell.summarizer_thinking)} / "
            f"C:{_bool_label(cell.critic_thinking)}"
        ),
        "memory_enabled": cell.memory_enabled,
        "max_iterations": cell.max_iterations,
        "status": report.get("status", ""),
        "report_path": str(report_path),
        "results_path": report.get("results_path", ""),
        "sample_artifacts_root": report.get("sample_artifacts_root", ""),
        "n": int(summary.get("executed_count", len(retained_executed)) or 0),
        "retained_n": int(summary.get("sample_count", len(retained_executed)) or 0),
        "target_n": int(summary.get("target_sample_count", 0) or 0),
        "success_count": int(summary.get("success_count", 0) or 0),
        "failure_count": int(summary.get("failure_count", 0) or 0),
        "replacement_count": int(summary.get("replacement_count", 0) or 0),
        "within_tolerance_count": len(within_tolerance),
        "within_tolerance_rate": (
            len(within_tolerance) / len(retained_executed)
            if retained_executed
            else 0.0
        ),
        "mean_absolute_duration_delta_seconds": _mean(duration_delta_values),
        "mean_iterations_run": _mean(
            float(sample.get("iterations_run", 0.0) or 0.0)
            for sample in retained_executed
        ),
        "mean_tool_invocations": _mean(
            float(sample.get("tool_invocations", 0.0) or 0.0)
            for sample in retained_executed
        ),
        "mean_factualness": _mean(factualness_values),
        "mean_factualness_ci95": mean_confidence_interval_95(
            factualness_values,
            lower_bound=0.0,
            upper_bound=1.0,
        ),
        "mean_naturalness": _mean(naturalness_values),
        "mean_speaker_grammar_similarity": _mean(speaker_values),
        "mean_composite": _mean(composite_values),
    }


def _load_cells_from_plan(plan_path: Path) -> tuple[AblationCell, ...]:
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    raw_cells = payload.get("cells")
    if not isinstance(raw_cells, list):
        raise RuntimeError(f"Plan file does not contain cells: {plan_path}")
    cells: list[AblationCell] = []
    for raw_cell in raw_cells:
        raw = dict(raw_cell)
        raw.pop("command_display", None)
        raw["command"] = tuple(raw["command"])
        cells.append(AblationCell(**raw))
    return tuple(cells)


def _aggregate_reports(*, output_root: Path, cells: tuple[AblationCell, ...]) -> dict[str, Any]:
    cell_summaries: list[dict[str, Any]] = []
    missing_cells: list[str] = []
    for cell in cells:
        report_path = _latest_cell_report(cell)
        if report_path is None:
            missing_cells.append(cell.cell_id)
            continue
        cell_summaries.append(_summarize_cell_report(cell=cell, report_path=report_path))
    return {
        "generated_at_utc": _utc_now(),
        "primary_metric": PRIMARY_METRIC,
        "exploratory_metrics": list(EXPLORATORY_METRICS),
        "output_root": str(output_root),
        "missing_cells": missing_cells,
        "cells": cell_summaries,
    }


def _format_ci(bounds: tuple[float, float] | None) -> str:
    return format_confidence_interval(bounds)


def _render_markdown_report(aggregate: dict[str, Any]) -> str:
    rows = []
    for cell in aggregate["cells"]:
        rows.append(
            [
                cell["mode"],
                cell["ablation"],
                str(cell["n"]),
                f"{cell['mean_factualness']:.3f}",
                _format_ci(cell["mean_factualness_ci95"]),
                f"{cell['within_tolerance_rate']:.3f}",
                f"{cell['mean_absolute_duration_delta_seconds']:.1f}",
                f"{cell['success_count']}/{cell['n']}",
                str(cell["replacement_count"]),
                f"{cell['mean_iterations_run']:.1f}",
                f"{cell['mean_tool_invocations']:.1f}",
                f"{cell['mean_naturalness']:.3f}",
                f"{cell['mean_speaker_grammar_similarity']:.3f}",
                f"{cell['mean_composite']:.3f}",
                cell["report_path"],
            ]
        )
    lines = [
        "# SPoRC Ablation Study",
        "",
        (
            "Primary automatic endpoint: DeepSeek V4 Pro non-thinking "
            "factualness. Naturalness, speaker-grammar similarity, and "
            "composite are exploratory because the human-alignment evidence is "
            "strongest for factualness."
        ),
        "",
        "## Cell Summary",
        "",
        render_markdown_table(
            [
                "mode",
                "ablation",
                "n",
                "factual",
                "factual_95_ci",
                "within_tol",
                "abs_delta_s",
                "success",
                "replaced",
                "iters",
                "tools",
                "natural",
                "speaker_grammar",
                "composite",
                "report",
            ],
            rows,
        ),
    ]
    missing_cells = aggregate.get("missing_cells") or []
    if missing_cells:
        lines.extend(
            [
                "",
                "## Missing Cells",
                "",
                "\n".join(f"- `{cell_id}`" for cell_id in missing_cells),
            ]
        )
    return "\n".join(lines) + "\n"


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for raw, escaped in replacements.items():
        text = text.replace(raw, escaped)
    return text


def _compact_mode_label(cell: dict[str, Any]) -> str:
    summarizer = "T" if bool(cell["summarizer_thinking"]) else "NT"
    critic = "T" if bool(cell["critic_thinking"]) else "NT"
    return f"S:{summarizer}/C:{critic}"


def _variant_label(raw_value: object) -> str:
    return {
        "baseline": "full",
        "no_memory": "no memory",
        "no_critic": "no critic",
    }.get(str(raw_value), str(raw_value).replace("_", " "))


def _figure_axis_bounds(values: Iterable[float]) -> tuple[float, float, list[float]]:
    sample = [float(value) for value in values]
    if not sample:
        return 0.0, 1.0, [0.0, 0.5, 1.0]
    step = 0.05
    lower = max(0.0, min(sample) - 0.02)
    upper = min(1.0, max(sample) + 0.02)
    axis_min = max(0.0, math.floor(lower / step) * step)
    axis_max = min(1.0, math.ceil(upper / step) * step)
    if axis_max - axis_min < 0.15:
        center = (axis_min + axis_max) / 2.0
        axis_min = max(0.0, math.floor((center - 0.075) / step) * step)
        axis_max = min(1.0, math.ceil((center + 0.075) / step) * step)
    ticks: list[float] = []
    current = axis_min
    while current <= axis_max + 1e-9:
        ticks.append(round(current, 2))
        current += step
    return axis_min, axis_max, ticks


def _render_ablation_factualness_figure(cells: list[dict[str, Any]]) -> str:
    retained_cells = [cell for cell in cells if int(cell.get("n", 0) or 0)]
    if not retained_cells:
        return ""

    variant_order = (
        ("no_critic", "no\\\\critic", 0.0),
        ("no_memory", "no\\\\memory", 1.8),
        ("baseline", "full", 3.6),
    )
    preferred_modes = ("S:T/C:T", "S:NT/C:NT", "S:T/C:NT", "S:NT/C:T")
    modes = {_compact_mode_label(cell) for cell in retained_cells}
    mode_order = [mode for mode in preferred_modes if mode in modes]
    mode_order.extend(sorted(modes.difference(mode_order)))

    by_mode_variant = {
        (_compact_mode_label(cell), str(cell.get("ablation", ""))): cell
        for cell in retained_cells
    }
    bounds_values: list[float] = []
    for cell in retained_cells:
        bounds_values.append(float(cell.get("mean_factualness", 0.0) or 0.0))
        ci_bounds = cell.get("mean_factualness_ci95")
        if isinstance(ci_bounds, (list, tuple)) and len(ci_bounds) == 2:
            bounds_values.extend(float(bound) for bound in ci_bounds)
    axis_min, axis_max, ticks = _figure_axis_bounds(bounds_values)
    plot_height = 3.0

    def _plot_y(value: object) -> float:
        raw_value = float(value)
        return ((raw_value - axis_min) / (axis_max - axis_min)) * plot_height

    styles = (
        ("blue!65!black", "very thick", "circle"),
        ("red!70!black", "very thick,dashed", "square"),
        ("green!45!black", "thick,densely dotted", "circle"),
        ("black!70", "thick,dash dot", "square"),
    )
    lines = [
        "\\begin{figure}[t]",
        "\\centering",
        "\\begin{tikzpicture}[font=\\scriptsize,x=1cm,y=1cm]",
        "\\draw[black!70] (-0.35,0) -- (4.15,0);",
        "\\draw[black!70] (-0.35,0) -- (-0.35,3.05);",
    ]
    for tick in ticks:
        y_coord = _plot_y(tick)
        lines.extend(
            [
                f"\\draw[black!18] (-0.35,{y_coord:.3f}) -- (4.05,{y_coord:.3f});",
                f"\\draw[black!70] (-0.39,{y_coord:.3f}) -- (-0.31,{y_coord:.3f});",
                f"\\node[anchor=east] at (-0.43,{y_coord:.3f}) {{{tick:.2f}}};",
            ]
        )
    for _, label, x_coord in variant_order:
        lines.extend(
            [
                f"\\draw[black!70] ({x_coord:.3f},0) -- ({x_coord:.3f},-0.04);",
                (
                    f"\\node[anchor=north,align=center] at "
                    f"({x_coord:.3f},-0.10) {{{label}}};"
                ),
            ]
        )
    lines.extend(
        [
            "\\node[rotate=90,anchor=south] at (-0.95,1.50) {Mean factualness};",
        ]
    )
    for mode_index, mode in enumerate(mode_order):
        color, line_style, marker = styles[mode_index % len(styles)]
        points: list[tuple[float, float, dict[str, Any]]] = []
        for variant_key, _, x_coord in variant_order:
            cell = by_mode_variant.get((mode, variant_key))
            if cell is None:
                continue
            y_coord = _plot_y(cell.get("mean_factualness", 0.0))
            points.append((x_coord, y_coord, cell))
        if not points:
            continue
        path = " -- ".join(f"({x:.3f},{y:.3f})" for x, y, _ in points)
        lines.append(f"\\draw[{color},{line_style}] {path};")
        for x_coord, y_coord, cell in points:
            ci_bounds = cell.get("mean_factualness_ci95")
            if isinstance(ci_bounds, (list, tuple)) and len(ci_bounds) == 2:
                lower_y = _plot_y(ci_bounds[0])
                upper_y = _plot_y(ci_bounds[1])
                lines.extend(
                    [
                        (
                            f"\\draw[{color},opacity=0.55,line width=0.35pt] "
                            f"({x_coord:.3f},{lower_y:.3f}) -- "
                            f"({x_coord:.3f},{upper_y:.3f});"
                        ),
                        (
                            f"\\draw[{color},opacity=0.55,line width=0.35pt] "
                            f"({x_coord - 0.055:.3f},{lower_y:.3f}) -- "
                            f"({x_coord + 0.055:.3f},{lower_y:.3f});"
                        ),
                        (
                            f"\\draw[{color},opacity=0.55,line width=0.35pt] "
                            f"({x_coord - 0.055:.3f},{upper_y:.3f}) -- "
                            f"({x_coord + 0.055:.3f},{upper_y:.3f});"
                        ),
                    ]
                )
            if marker == "square":
                lines.append(
                    (
                        f"\\draw[{color},fill=white,line width=0.45pt] "
                        f"({x_coord - 0.045:.3f},{y_coord - 0.045:.3f}) "
                        f"rectangle "
                        f"({x_coord + 0.045:.3f},{y_coord + 0.045:.3f});"
                    )
                )
            else:
                lines.append(
                    f"\\fill[{color}] ({x_coord:.3f},{y_coord:.3f}) "
                    "circle[radius=1.35pt];"
                )
        legend_x = 0.05 + (mode_index % 2) * 1.85
        legend_y = 3.37 - (mode_index // 2) * 0.22
        lines.append(
            (
                f"\\draw[{color},{line_style}] ({legend_x:.3f},{legend_y:.3f}) "
                f"-- ({legend_x + 0.30:.3f},{legend_y:.3f});"
            )
        )
        lines.append(
            f"\\node[anchor=west] at ({legend_x + 0.37:.3f},{legend_y:.3f}) "
            f"{{{_latex_escape(mode)}}};"
        )
    lines.extend(
        [
            "\\end{tikzpicture}",
            (
                "\\caption{Mean factualness by ablation variant. Error bars "
                "show the same 95\\% confidence intervals reported in "
                "Table~\\ref{tab:sporc-ablation}.}"
            ),
            "\\label{fig:sporc-ablation-factualness}",
            "\\end{figure}",
        ]
    )
    return "\n".join(lines) + "\n\n"


def _render_paper_section(aggregate: dict[str, Any]) -> str:
    cells = [cell for cell in aggregate.get("cells", []) if int(cell.get("n", 0) or 0)]
    rows: list[str] = []
    for cell in cells:
        rows.append(
            " & ".join(
                [
                    _latex_escape(_compact_mode_label(cell)),
                    _latex_escape(_variant_label(cell["ablation"])),
                    str(cell["n"]),
                    f"{cell['mean_factualness']:.3f}",
                    _latex_escape(_format_ci(cell["mean_factualness_ci95"])),
                    f"{cell['within_tolerance_rate']:.2f}",
                    f"{cell['mean_absolute_duration_delta_seconds']:.1f}",
                    f"{cell['mean_iterations_run']:.1f}",
                ]
            )
            + r" \\"
        )
    table_body = (
        "\n".join(rows)
        if rows
        else r"\multicolumn{8}{c}{No completed ablation cells yet.} \\"
    )
    total_samples = sum(int(cell["n"]) for cell in cells)
    total_replacements = sum(int(cell["replacement_count"]) for cell in cells)
    cell_count = len(cells)
    retained_sentence = (
        f"The ablation run contains {cell_count} cells and "
        f"{total_samples} retained samples in total. "
        f"All cells met the duration tolerance for every retained sample, "
        f"and the runs used {total_replacements} replacement samples."
        if cells
        else "No completed ablation cells were available when this section was generated."
    )
    factualness_figure = _render_ablation_factualness_figure(cells)
    figure_sentence = (
        " Figure~\\ref{fig:sporc-ablation-factualness} plots the same "
        "factualness means by ablation variant and thinking condition, making "
        "the lack of a consistent monotonic pattern visible."
        if cells
        else ""
    )
    if cells:
        ranked_cells = sorted(
            cells,
            key=lambda cell: float(cell.get("mean_factualness", 0.0) or 0.0),
            reverse=True,
        )
        top_cell = ranked_cells[0]
        runner_up = ranked_cells[1] if len(ranked_cells) > 1 else ranked_cells[0]
        result_sentence = (
            "The two highest factualness means were "
            f"{_latex_escape(_compact_mode_label(top_cell))} "
            f"{_latex_escape(_variant_label(top_cell['ablation']))} "
            f"({top_cell['mean_factualness']:.3f}, 95\\% CI "
            f"{_latex_escape(_format_ci(top_cell['mean_factualness_ci95']))}) "
            "and "
            f"{_latex_escape(_compact_mode_label(runner_up))} "
            f"{_latex_escape(_variant_label(runner_up['ablation']))} "
            f"({runner_up['mean_factualness']:.3f}, 95\\% CI "
            f"{_latex_escape(_format_ci(runner_up['mean_factualness_ci95']))})."
        )
    else:
        result_sentence = ""
    return (
        "% Generated by scripts/run_sporc_ablation_study.py\n"
        "\\label{sec:ablation-study}\n"
        "We test three system variants on a seeded random SPoRC sample set: the "
        "full summarizer--critic loop, the same loop without persisted loop "
        "memory, and a summarizer-only version with the critic removed. The "
        "no-memory variant disables the JSON-backed unresolved-issue and "
        "remedy-history mechanism described in Appendix~\\ref{app:loop-memory}. "
        "The Qwen/Qwen3.6-27B vLLM endpoint is run with thinking enabled and "
        "disabled for the summarizer and critic. Based on the human-alignment "
        "analysis, factualness is the main automatic score; naturalness, "
        "speaker-grammar similarity, and composite are kept as exploratory "
        "artifact-level diagnostics.\n\n"
        f"{retained_sentence} {result_sentence} Because the confidence intervals "
        "overlap, we read the ablation as showing that these variants perform "
        "similarly under this protocol, not as a clear ranking of memory or critic "
        f"contributions.{figure_sentence}\n\n"
        f"{factualness_figure}"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{SPoRC ablation factualness results.}\n"
        "\\label{tab:sporc-ablation}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{llrccrrr}\n"
        "\\toprule\n"
        "Mode & Variant & $n$ & Factual & 95\\% CI & Tol. & $|\\Delta|$s & Iter. \\\\\n"
        "\\midrule\n"
        f"{table_body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "}%\n"
        "\\end{table}\n"
    )


def _ensure_paper_input(main_path: Path) -> None:
    if not main_path.exists():
        return
    text = main_path.read_text(encoding="utf-8")
    if "sections/ablation_study" in text:
        return
    anchor = "\\input{sections/results}"
    replacement = (
        "\\input{sections/results}\n\n"
        "\\section{Ablation Study}\n"
        "\\input{sections/ablation_study}"
    )
    if anchor not in text:
        raise RuntimeError(f"Could not find {anchor!r} in {main_path}")
    main_path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")


def _write_reports(
    *,
    output_root: Path,
    aggregate: dict[str, Any],
    update_paper: bool,
    paper_section_path: Path,
    paper_main_path: Path,
) -> tuple[Path, Path, Path | None]:
    json_path = output_root / "ablation_report.json"
    markdown_path = output_root / "ablation_report.md"
    _write_json(json_path, aggregate)
    markdown_path.write_text(_render_markdown_report(aggregate), encoding="utf-8")
    paper_path: Path | None = None
    paper_section = _render_paper_section(aggregate)
    generated_paper_path = output_root / "ablation_study_section.tex"
    generated_paper_path.write_text(paper_section, encoding="utf-8")
    if update_paper:
        if not aggregate.get("cells"):
            raise RuntimeError(
                "No completed ablation cells were found; refusing to update the paper."
            )
        paper_section_path.parent.mkdir(parents=True, exist_ok=True)
        paper_section_path.write_text(paper_section, encoding="utf-8")
        _ensure_paper_input(paper_main_path)
        paper_path = paper_section_path
    return json_path.resolve(), markdown_path.resolve(), paper_path


def _resolve_output_root(raw_output_root: str) -> Path:
    if raw_output_root:
        return Path(raw_output_root).resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (DEFAULT_OUTPUT_ROOT / stamp).resolve()


def _planned_total_samples(
    *,
    args: argparse.Namespace,
    cells: tuple[AblationCell, ...],
) -> int:
    per_cell_target = max(0, int(args.max_samples))
    if per_cell_target <= 0:
        return 0
    return per_cell_target * len(cells)


def _completed_cell_sample_count(report_path: Path | None, *, fallback: int) -> int:
    if report_path is None:
        return fallback
    try:
        payload = _load_report(report_path)
    except (OSError, json.JSONDecodeError):
        return fallback
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return fallback
    for key in ("target_sample_count", "sample_count", "executed_count"):
        value = _coerce_optional_int(summary.get(key))
        if value is not None:
            return max(0, value)
    return fallback


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the SPoRC summarizer-critic ablation study."
    )
    parser.add_argument("--mode", choices=("plan", "run", "report"), default="plan")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--dataset-manifest", default=str(DEFAULT_DATASET_MANIFEST))
    parser.add_argument("--prepare-if-missing", action="store_true")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--duration-preset", choices=tuple(DURATION_PRESET_SECONDS), default="5m")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--critic-feedback-budget", type=int, default=1)
    parser.add_argument(
        "--ablation",
        dest="ablations",
        action="append",
        choices=("baseline", "no_memory", "no_critic"),
        help="Repeat to run a subset of ablation variants. Defaults to all three.",
    )
    parser.add_argument(
        "--thinking-grid",
        choices=("matched", "factorial"),
        default="matched",
        help=(
            "matched runs thinking/thinking and nonthinking/nonthinking. "
            "factorial also runs the cross-role interaction cells."
        ),
    )
    parser.add_argument(
        "--vllm-base-url",
        default=_first_env(
            (
                "CARD_FRAMEWORK_VLLM_BASE_URL",
                "CARD_FRAMEWORK_VLLM_URL",
                "VLLM_BASE_URL",
                "VLLM_URL",
            ),
            "http://127.0.0.1:8000/v1",
        ),
    )
    parser.add_argument(
        "--vllm-api-key",
        default=_first_env(("CARD_FRAMEWORK_VLLM_API_KEY", "VLLM_API_KEY"), "EMPTY"),
    )
    parser.add_argument(
        "--vllm-model",
        default=_first_env(("CARD_FRAMEWORK_VLLM_MODEL", "VLLM_MODEL"), "Qwen/Qwen3.6-27B"),
    )
    parser.add_argument(
        "--vllm-max-tokens",
        type=int,
        default=0,
        help="Set a vLLM max_tokens request budget. Use 0 to omit max_tokens.",
    )
    parser.add_argument("--vllm-request-timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--deepseek-base-url",
        default=_first_env(("DEEPSEEK_BASE_URL",), "https://api.deepseek.com"),
    )
    parser.add_argument("--deepseek-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--deepseek-model", default="deepseek-v4-pro")
    parser.add_argument("--deepseek-max-tokens", type=int, default=384_000)
    parser.add_argument("--deepseek-temperature", type=float, default=0.0)
    parser.add_argument("--deepseek-request-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--summarizer-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--critic-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--retrieval-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--index-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--eta-learning-rate",
        type=float,
        default=0.25,
        help=(
            "Learning rate for the exponential moving average used in ETA "
            "estimates. Higher values react faster to the latest sample."
        ),
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=5.0,
        help="Minimum seconds between heartbeat progress-bar redraws.",
    )
    parser.add_argument(
        "--disable-progress-bar",
        action="store_true",
        help="Disable the wrapper progress bar while keeping child logs.",
    )
    parser.add_argument("--update-paper", action="store_true")
    parser.add_argument("--paper-section-path", default=str(DEFAULT_PAPER_SECTION_PATH))
    parser.add_argument("--paper-main-path", default=str(DEFAULT_PAPER_MAIN_PATH))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = _resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    existing_plan_path = output_root / "ablation_plan.json"
    if args.mode == "report" and existing_plan_path.exists():
        plan_path = existing_plan_path.resolve()
        cells = _load_cells_from_plan(plan_path)
        print(f"Loaded plan: {plan_path}")
    else:
        provider_profiles_path = _write_provider_profiles(args, output_root)
        cells = _build_ablation_cells(
            args,
            output_root=output_root,
            provider_profiles_path=provider_profiles_path,
        )
        plan_path = _write_plan(
            args=args,
            output_root=output_root,
            provider_profiles_path=provider_profiles_path,
            cells=cells,
        )
        print(f"Plan written: {plan_path}")

    if args.mode == "run":
        progress = AblationProgressTracker(
            total_cells=len(cells),
            total_samples=_planned_total_samples(args=args, cells=cells),
            eta_learning_rate=args.eta_learning_rate,
            progress_interval_seconds=args.progress_interval_seconds,
            enabled=not bool(args.disable_progress_bar),
        )
        try:
            fallback_sample_count = max(0, int(args.max_samples))
            for cell_index, cell in enumerate(cells, start=1):
                latest_report = _latest_cell_report(cell)
                completed_sample_count = _completed_cell_sample_count(
                    latest_report,
                    fallback=0,
                )
                should_extend_cell = _cell_needs_extension(
                    latest_report,
                    requested_sample_count=fallback_sample_count,
                )
                if _cell_report_completed(latest_report) and not should_extend_cell:
                    progress.clear_line()
                    print(f"Skipping completed cell: {cell.cell_id}")
                    progress.skip_cell(
                        cell=cell,
                        sample_count=_completed_cell_sample_count(
                            latest_report,
                            fallback=fallback_sample_count,
                        ),
                    )
                    continue
                progress.clear_line()
                command = cell.command
                if should_extend_cell:
                    checkpoint_path = Path(cell.checkpoint_path)
                    if not checkpoint_path.exists():
                        progress.clear_line()
                        print(
                            "Cannot extend completed cell without checkpoint: "
                            f"{cell.cell_id} ({checkpoint_path})"
                        )
                        return 1
                    command = _command_with_resume_from_checkpoint(cell.command)
                    print(
                        f"Extending cell: {cell.cell_id} "
                        f"({completed_sample_count}->{fallback_sample_count} samples)"
                    )
                else:
                    print(f"Running cell: {cell.cell_id}")
                progress.start_cell(
                    cell=cell,
                    cell_index=cell_index,
                    precompleted_sample_count=(
                        completed_sample_count if should_extend_cell else 0
                    ),
                )
                exit_code = _run_cell(cell, progress=progress, command=command)
                if exit_code != 0:
                    progress.clear_line()
                    print(f"Cell failed with exit code {exit_code}: {cell.cell_id}")
                    return exit_code
        finally:
            progress.finish()

    if args.mode in {"run", "report"}:
        aggregate = _aggregate_reports(output_root=output_root, cells=cells)
        json_path, markdown_path, paper_path = _write_reports(
            output_root=output_root,
            aggregate=aggregate,
            update_paper=bool(args.update_paper),
            paper_section_path=Path(args.paper_section_path).resolve(),
            paper_main_path=Path(args.paper_main_path).resolve(),
        )
        print(f"Ablation JSON report: {json_path}")
        print(f"Ablation Markdown report: {markdown_path}")
        if paper_path is not None:
            print(f"Paper section updated: {paper_path}")
        if aggregate.get("missing_cells"):
            print(f"Missing cells: {', '.join(aggregate['missing_cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
