"""CLI entrypoints for the SPoRC summarizer-critic benchmark package."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
from typing import TextIO

from tqdm.auto import tqdm

from card_framework.benchmark.process_utils import (
    detached_popen_kwargs,
    terminate_process_tree,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_FULL_DATASET_MANIFEST_PATH,
    DEFAULT_GPU_THRESHOLD_GIB,
    DEFAULT_HUMAN_SAMPLING_ROOT,
    DEFAULT_LLM_SAMPLING_ROOT,
    DEFAULT_LONGFORM_AUDIT_PATH,
    DEFAULT_LONGFORM_MANIFEST_PATH,
    DEFAULT_PREPARED_FULL_ROOT,
    DEFAULT_PREPARED_ROOT,
    DEFAULT_RAM_THRESHOLD_PERCENT,
    DEFAULT_REFERENCE_WPM,
    DEFAULT_SUPERVISOR_ROOT,
    DEFAULT_SWEEP_ROOT,
    BenchmarkExecutionReport,
    BenchmarkSummary,
    DURATION_PRESET_SECONDS,
    DurationSweepPresetSummary,
    DurationSweepReport,
    SampleBenchmarkResult,
    SampleMetricBundle,
    SupervisedRunRecord,
)
from card_framework.benchmark.summarizer_critic.reporting_utils import (
    mean_confidence_interval_95,
)
from card_framework.shared.perf import (
    PerfTimer,
    StatusReporter,
    measure_elapsed_time,
    report_loading_duration,
)
from card_framework.shared.paths import REPO_ROOT


_DATASET_HELPERS: tuple[object, object] | None = None
_RUNNER_HELPERS: tuple[object, object, object] | None = None
_TELEMETRY_HELPERS: tuple[object, object, object, object] | None = None
_REPORTING_HELPERS: tuple[object, object] | None = None
_PROGRESS_EVENT_KEY = "sporc_progress_event"
_SWEEP_PROGRESS_HEARTBEAT_SECONDS = 15.0
_SWEEP_CHILD_NO_HEARTBEAT_SECONDS = 90.0
_SWEEP_CHILD_DEFAULT_STALL_SECONDS = 600.0
_SWEEP_STALL_FINAL_STATUS = "sweep_watchdog_stalled"
_ANSI_RESET = "\033[0m"
_ANSI_DIM = "\033[2m"
_ANSI_BOLD = "\033[1m"
_ANSI_RED = "\033[31m"
_ANSI_GREEN = "\033[32m"
_ANSI_YELLOW = "\033[33m"
_ANSI_BLUE = "\033[34m"
_ANSI_CYAN = "\033[36m"


@dataclass(slots=True)
class _DurationSweepChildState:
    """Track one preset child process inside a duration sweep."""

    duration_preset: str
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: TextIO
    reader_thread: threading.Thread | None
    retrieval_port: int
    summarizer_port: int
    critic_port: int
    max_samples: int
    checkpoint_path: Path
    output_root: Path
    shared_sample_queue_root: Path | None = None
    shard_index: int = 1
    shard_count: int = 1
    sample_count: int | None = None
    completed_samples: int = 0
    active_sample_index: int | None = None
    active_sample_id: str | None = None
    active_stage: str | None = None
    active_iteration: int | None = None
    last_heartbeat_at: float | None = None
    last_activity_at: float | None = None
    last_activity_age_seconds: float | None = None
    restart_count: int = 0
    recovered_stall_count: int = 0


@dataclass(slots=True)
class _DurationSweepPresetState:
    """Track all shard children for one active duration preset."""

    duration_preset: str
    max_samples: int
    worker_budget: int
    queue_root: Path
    sample_count: int | None = None
    child_states: dict[int, _DurationSweepChildState] = field(default_factory=dict)
    completed_child_summaries: dict[int, DurationSweepPresetSummary] = field(
        default_factory=dict
    )


@dataclass(slots=True, frozen=True)
class _DurationSweepProgressState:
    """Snapshot aggregate sweep progress for operator-facing status lines."""

    rendered: str
    total_completed: int
    total_samples: int | None
    active_text: str
    completed_presets: int
    preset_count: int


class _SweepProgressBar:
    """Manage an optional footer tqdm bar for duration sweeps."""

    def __init__(self, *, stream: TextIO) -> None:
        self._stream = stream
        self._bar: tqdm | None = None

    def is_enabled(self) -> bool:
        """Return whether a footer bar is active."""
        return self._bar is not None

    def start(self) -> None:
        """Create the footer tqdm bar when stderr is interactive."""
        if not _supports_terminal_ui(self._stream):
            return
        self._bar = tqdm(
            total=None,
            desc="Sweep 0/0 presets",
            unit="sample",
            dynamic_ncols=True,
            leave=True,
            file=self._stream,
        )

    def update(self, progress_state: _DurationSweepProgressState) -> bool:
        """Update the active tqdm bar from aggregate sweep progress."""
        if self._bar is None:
            return False
        self._bar.set_description(
            (
                f"Sweep {progress_state.completed_presets}/"
                f"{progress_state.preset_count} presets"
            ),
            refresh=False,
        )
        self._bar.total = progress_state.total_samples
        self._bar.n = progress_state.total_completed
        self._bar.set_postfix_str(
            f"active={_truncate_sweep_active_text(progress_state.active_text)}",
            refresh=False,
        )
        self._bar.refresh()
        return True

    def write(self, message: str) -> bool:
        """Write one status line without breaking the footer bar."""
        if self._bar is None:
            return False
        self._bar.write(message, file=self._stream)
        return True

    def close(self) -> None:
        """Close the footer bar if it is active."""
        if self._bar is None:
            return
        self._bar.close()
        self._bar = None


_ACTIVE_SWEEP_PROGRESS_BAR: _SweepProgressBar | None = None


def _supports_terminal_ui(stream: TextIO) -> bool:
    """Return whether terminal-oriented UI should target this stream."""
    is_tty = bool(getattr(stream, "isatty", lambda: False)())
    if not is_tty:
        return False
    return str(os.environ.get("TERM", "")).lower() != "dumb"


def _supports_terminal_color(stream: TextIO) -> bool:
    """Return whether ANSI color should target this stream."""
    if not _supports_terminal_ui(stream):
        return False
    return not bool(os.environ.get("NO_COLOR"))


def _colorize_cli_status(message: str, *, stream: TextIO) -> str:
    """Apply lightweight ANSI colors to operator-facing CLI status lines."""
    if not _supports_terminal_color(stream):
        return message
    lowered = message.lower()
    style = ""
    if "failed" in lowered or "error" in lowered or "threshold breach" in lowered:
        style = f"{_ANSI_BOLD}{_ANSI_RED}"
    elif "cancel" in lowered or "restart" in lowered or "continuing preset" in lowered:
        style = f"{_ANSI_BOLD}{_ANSI_YELLOW}"
    elif "finished preset" in lowered or "loaded " in lowered or "status=completed" in lowered:
        style = f"{_ANSI_BOLD}{_ANSI_GREEN}"
    elif "starting duration sweep" in lowered or "started preset" in lowered:
        style = f"{_ANSI_BOLD}{_ANSI_CYAN}"
    elif lowered.startswith("loading "):
        style = f"{_ANSI_BOLD}{_ANSI_BLUE}"
    elif lowered.startswith("sweep progress:"):
        style = f"{_ANSI_DIM}{_ANSI_CYAN}"
    if not style:
        return message
    return f"{style}{message}{_ANSI_RESET}"


def _truncate_sweep_active_text(active_text: str, *, max_chars: int = 80) -> str:
    """Keep the tqdm postfix readable when many shards are active."""
    normalized = " ".join(active_text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def _emit_cli_status(message: str) -> None:
    """Emit one operator-facing status line for CLI progress."""
    rendered_message = _colorize_cli_status(message, stream=sys.stderr)
    if _ACTIVE_SWEEP_PROGRESS_BAR is not None and _ACTIVE_SWEEP_PROGRESS_BAR.write(
        rendered_message
    ):
        return
    print(rendered_message, file=sys.stderr, flush=True)


def _load_dataset_helpers(
    *,
    status_reporter: StatusReporter | None = None,
) -> tuple[object, object]:
    """Load SPoRC dataset helpers lazily for CLI commands."""
    global _DATASET_HELPERS
    if _DATASET_HELPERS is None:
        with report_loading_duration(
            "SPoRC dataset helpers",
            reporter=status_reporter,
        ):
            from card_framework.benchmark.summarizer_critic.sporc.dataset import (
                curate_longform_sporc_manifest,
                prepare_sporc_benchmark,
            )

        _DATASET_HELPERS = (curate_longform_sporc_manifest, prepare_sporc_benchmark)
    return _DATASET_HELPERS


def _load_runner_helpers(
    *,
    status_reporter: StatusReporter | None = None,
) -> tuple[object, object, object]:
    """Load SPoRC execute-path helpers lazily for CLI commands."""
    global _RUNNER_HELPERS
    if _RUNNER_HELPERS is None:
        with report_loading_duration(
            "SPoRC runner helpers",
            reporter=status_reporter,
        ):
            from card_framework.benchmark.summarizer_critic.sporc.runner import (
                BenchmarkExecutionError,
                build_execute_parser,
                execute_benchmark,
            )

        _RUNNER_HELPERS = (
            BenchmarkExecutionError,
            build_execute_parser,
            execute_benchmark,
        )
    return _RUNNER_HELPERS


def _load_telemetry_helpers(
    *,
    status_reporter: StatusReporter | None = None,
) -> tuple[object, object, object, object]:
    """Load shared telemetry helpers lazily for supervise flows."""
    global _TELEMETRY_HELPERS
    if _TELEMETRY_HELPERS is None:
        with report_loading_duration(
            "SPoRC telemetry helpers",
            reporter=status_reporter,
        ):
            from card_framework.benchmark.summarizer_critic.qmsum.telemetry import (
                collect_telemetry_sample,
                detect_threshold_breach,
                sleep_with_poll,
                write_telemetry_line,
            )

        _TELEMETRY_HELPERS = (
            collect_telemetry_sample,
            detect_threshold_breach,
            sleep_with_poll,
            write_telemetry_line,
        )
    return _TELEMETRY_HELPERS


def _load_reporting_helpers(
    *,
    status_reporter: StatusReporter | None = None,
) -> tuple[object, object]:
    """Load SPoRC reporting helpers lazily for CLI commands."""
    global _REPORTING_HELPERS
    if _REPORTING_HELPERS is None:
        with report_loading_duration(
            "SPoRC reporting helpers",
            reporter=status_reporter,
        ):
            from card_framework.benchmark.summarizer_critic.sporc.reporting import (
                write_duration_sweep_artifacts,
                write_supervisor_session_artifacts,
            )

        _REPORTING_HELPERS = (
            write_duration_sweep_artifacts,
            write_supervisor_session_artifacts,
        )
    return _REPORTING_HELPERS


def _benchmark_execution_error_type() -> type[Exception]:
    """Return the lazily imported benchmark execution error type."""
    benchmark_execution_error, _, _ = _load_runner_helpers()
    return benchmark_execution_error


def _build_parser(
    *,
    target_command: str | None = None,
    status_reporter: StatusReporter | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SPoRC real-dataset benchmark for the summarizer-critic framework."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-dataset", help="Download or prepare SPoRC data"
    )
    prepare.add_argument("--prepared-root", default="")
    prepare.add_argument(
        "--dataset-variant", choices=("sample", "full"), default="sample"
    )
    prepare.add_argument("--episode-data-path", default="")
    prepare.add_argument("--turn-data-path", default="")
    prepare.add_argument("--reference-wpm", type=float, default=DEFAULT_REFERENCE_WPM)
    prepare.add_argument("--force-download", action="store_true")

    curate_longform = subparsers.add_parser(
        "curate-longform",
        help="Filter a prepared SPoRC manifest down to strict long-form dialogue episodes",
    )
    curate_longform.add_argument(
        "--source-manifest",
        default=str(DEFAULT_FULL_DATASET_MANIFEST_PATH),
    )
    curate_longform.add_argument(
        "--output-manifest",
        default=str(DEFAULT_LONGFORM_MANIFEST_PATH),
    )
    curate_longform.add_argument(
        "--audit-path",
        default=str(DEFAULT_LONGFORM_AUDIT_PATH),
    )

    dedupe_manifest = subparsers.add_parser(
        "dedupe-manifest",
        help="Dedupe an inline SPoRC manifest in place using metadata-first checks",
    )
    dedupe_manifest.add_argument(
        "--manifest-path",
        default=str(DEFAULT_LONGFORM_MANIFEST_PATH),
    )
    dedupe_manifest.add_argument(
        "--audit-path",
        default="",
        help="Optional audit JSON path. Defaults to a sibling *.dedupe_audit.json file.",
    )

    export_transcripts = subparsers.add_parser(
        "export-transcripts",
        help="Export one combined raw transcript text file for selected manifest episodes",
    )
    export_transcripts.add_argument(
        "--manifest-path",
        default=str(DEFAULT_LONGFORM_MANIFEST_PATH),
    )
    export_transcripts.add_argument(
        "--episodes",
        required=True,
        help=(
            "Episode positions to export, for example `7,10,15` or `7-10`. "
            'Bracket form like "[7,10]" is treated as the inclusive range 7-10.'
        ),
    )
    export_transcripts.add_argument(
        "--output-path",
        default="",
        help="Optional combined transcript TXT output path.",
    )
    export_transcripts.add_argument(
        "--backwards-compat",
        action="store_true",
        help=(
            "Interpret --episodes as curated longform manifest positions even "
            "when --manifest-path points at the original prepared manifest. "
            "Selections are mapped back onto the target manifest by episode_key."
        ),
    )
    export_transcripts.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Write exact matching JSONL rows from the dataset turn_data_path "
            "with no transcript formatting, headers, or normalization from us."
        ),
    )

    human_sample = subparsers.add_parser(
        "human-sample",
        help="Run an interactive Rich workflow for human transcript segment sampling",
    )
    human_sample.add_argument(
        "--dataset-manifest",
        default=str(DEFAULT_LONGFORM_MANIFEST_PATH),
    )
    human_sample.add_argument(
        "--sampling-root",
        default=str(DEFAULT_HUMAN_SAMPLING_ROOT),
    )
    human_sample.add_argument("--resume-session", default="")
    human_sample.add_argument(
        "--duration-preset",
        choices=("30s", "1m", "5m"),
        default="",
    )
    human_sample.add_argument("--episodes", type=int, default=0)
    human_sample.add_argument("--words-per-minute", type=float, default=0.0)
    human_sample.add_argument("--sample-factor", type=float, default=0.0)
    human_sample.add_argument("--max-windows-per-episode", type=int, default=0)

    llm_sample = subparsers.add_parser(
        "llm-sample",
        help="Run an automatic Anthropic-backed transcript sampling workflow",
    )
    llm_sample.add_argument(
        "--dataset-manifest",
        default=str(DEFAULT_LONGFORM_MANIFEST_PATH),
    )
    llm_sample.add_argument(
        "--sampling-root",
        default=str(DEFAULT_LLM_SAMPLING_ROOT),
    )
    llm_sample.add_argument(
        "--duration-preset",
        choices=("30s", "1m", "5m"),
        default="30s",
    )
    llm_sample.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Optional fixed number of samples to request per transcript. Overrides the WPM-based budget mode when greater than zero.",
    )
    llm_sample.add_argument("--episodes", type=int, default=20)
    llm_sample.add_argument(
        "--select-episode",
        default="",
        help=(
            "Optional 1-based inclusive manifest episode range for llm-sample, "
            'for example "[7,10]" to select episodes 7 through 10. '
            "Overrides --episodes when provided."
        ),
    )
    llm_sample.add_argument("--words-per-minute", type=float, default=170.0)
    llm_sample.add_argument("--sample-factor", type=float, default=5.0)
    llm_sample.add_argument("--model", default="claude-opus-4-6")
    llm_sample.add_argument(
        "--thinking-effort",
        choices=("low", "medium", "high", "max"),
        default="medium",
    )
    llm_sample.add_argument("--disable-thinking", action="store_true")
    llm_sample.add_argument("--max-output-tokens", type=int, default=16000)

    if target_command in {None, "execute", "supervise", "duration-sweep"}:
        _, build_execute_parser, _ = _load_runner_helpers(
            status_reporter=status_reporter,
        )

        execute = subparsers.add_parser(
            "execute", help="Run the SPoRC summarizer-critic benchmark"
        )
        build_execute_parser(execute)

        supervise = subparsers.add_parser(
            "supervise", help="Loop execute runs under RAM/GPU guardrails"
        )
        build_execute_parser(supervise)
        supervise.add_argument("--session-root", default=str(DEFAULT_SUPERVISOR_ROOT))
        supervise.add_argument("--max-runs", type=int, default=1)
        supervise.add_argument("--between-runs-seconds", type=float, default=3.0)
        supervise.add_argument("--sample-interval-seconds", type=float, default=5.0)
        supervise.add_argument(
            "--ram-threshold-percent", type=float, default=DEFAULT_RAM_THRESHOLD_PERCENT
        )
        supervise.add_argument(
            "--gpu-threshold-gibibytes", type=float, default=DEFAULT_GPU_THRESHOLD_GIB
        )

        duration_sweep = subparsers.add_parser(
            "duration-sweep",
            help="Run SPoRC duration presets in transcript-only or voice-clone mode",
        )
        build_execute_parser(duration_sweep)
        duration_sweep.add_argument(
            "--mode",
            choices=("menu", "single", "batch-all"),
            default="menu",
        )
        duration_sweep.add_argument(
            "--examples",
            type=int,
            default=None,
            help=(
                "Optional per-preset sample cap for duration sweeps. "
                "Overrides --max-samples when provided; use 0 to run all prepared examples."
            ),
        )
        duration_sweep.add_argument(
            "--duration-presets",
            default=",".join(DURATION_PRESET_SECONDS.keys()),
            help="Comma-separated duration presets for batch-all mode.",
        )
        duration_sweep.add_argument(
            "--sweep-root",
            default=str(DEFAULT_SWEEP_ROOT),
            help="Root directory for aggregated duration-sweep artifacts.",
        )
        duration_sweep.add_argument(
            "--enable-voice-clone",
            action="store_true",
            help="Use the faithful live-draft voice-clone path instead of transcript-only mode.",
        )
        duration_sweep.add_argument(
            "--parallelism",
            type=int,
            default=0,
            help=(
                "How many preset runs to execute concurrently. "
                "Use 0 to run all selected presets in parallel."
            ),
        )
        duration_sweep.add_argument(
            "--concurrency",
            type=int,
            default=1,
            help=(
                "How many execute workers to seed per active preset. "
                "When an earlier preset finishes, its workers move forward to the "
                "next unfinished preset in order."
            ),
        )
        duration_sweep.add_argument(
            "--child-stall-seconds",
            type=float,
            default=_SWEEP_CHILD_DEFAULT_STALL_SECONDS,
            help="Restart a sweep child when its active sample shows no new activity for this many seconds.",
        )
        duration_sweep.add_argument(
            "--max-child-restarts",
            type=int,
            default=2,
            help=(
                "Warning threshold for watchdog checkpoint resumes per preset child. "
                "Stalled samples are still marked failed and later samples continue."
            ),
        )
    return parser


def _render_duration_sweep_menu() -> str:
    """Render the non-interactive duration sweep menu."""
    lines = [
        "SPoRC Duration Sweep Menu",
        "",
        "Available duration presets:",
    ]
    for label, seconds in DURATION_PRESET_SECONDS.items():
        lines.append(f"- {label}: {seconds} seconds")
    lines.extend(
        [
            "",
            "Recommended commands:",
            "- Single preset: uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep --mode single --duration-preset 5m --max-samples 10 --transcript-only",
            "- All presets: uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep --mode batch-all --max-samples 10 --parallelism 0 --transcript-only",
            "- All presets with 3 workers per seeded preset: uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep --mode batch-all --max-samples 30 --parallelism 0 --concurrency 3 --transcript-only",
            "",
            "Notes:",
            "- The sweep defaults to a non-interactive menu because this CLI is automation-friendly.",
            "- Use --mode batch-all to bypass the menu and benchmark every preset.",
            "- Use --parallelism 0 to run all selected presets concurrently.",
            "- Use --concurrency to seed multiple workers per active preset and let finished presets donate those workers forward.",
            "- Use --enable-voice-clone only when audio synthesis is part of the experiment.",
        ]
    )
    return "\n".join(lines)


def _parse_duration_preset_list(raw_value: str) -> list[str]:
    """Parse and validate a comma-separated duration preset list."""
    presets = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not presets:
        raise _benchmark_execution_error_type()(
            "At least one duration preset is required."
        )
    invalid = [item for item in presets if item not in DURATION_PRESET_SECONDS]
    if invalid:
        raise _benchmark_execution_error_type()(
            f"Unsupported duration preset(s): {', '.join(invalid)}"
        )
    return presets


def _format_eta(seconds: float | None) -> str:
    """Format a compact ETA string."""
    if seconds is None or seconds < 0:
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
    """Return remaining-count, throughput, and ETA strings."""
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
        eta_text = _format_eta(
            (remaining_samples / average_examples_per_minute) * 60.0
        )
    return remaining_samples, average_text, eta_text


def _resolve_prepare_root(
    *,
    dataset_variant: str,
    prepared_root: str,
) -> Path:
    raw_prepared_root = str(prepared_root or "").strip()
    if raw_prepared_root:
        return Path(raw_prepared_root).resolve()
    if dataset_variant == "full":
        return DEFAULT_PREPARED_FULL_ROOT.resolve()
    return DEFAULT_PREPARED_ROOT.resolve()


def _resolve_duration_sweep_examples(args: argparse.Namespace) -> int:
    """Resolve the per-preset sample cap for a duration sweep."""
    raw_value = args.examples if args.examples is not None else args.max_samples
    try:
        examples = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise _benchmark_execution_error_type()(
            "Duration sweep examples must be an integer."
        ) from exc
    if examples < 0:
        raise _benchmark_execution_error_type()(
            "Duration sweep examples must be zero or greater."
        )
    return examples


def _resolve_duration_sweep_parallelism(
    *,
    requested_parallelism: int,
    preset_count: int,
) -> int:
    """Resolve an effective sweep concurrency from the selected presets."""
    if preset_count <= 0:
        raise _benchmark_execution_error_type()(
            "At least one duration preset is required."
        )
    if requested_parallelism < 0:
        raise _benchmark_execution_error_type()(
            "Duration sweep parallelism must be zero or greater."
        )
    if requested_parallelism == 0:
        return preset_count
    return min(requested_parallelism, preset_count)


def _resolve_duration_sweep_concurrency(*, requested_concurrency: int) -> int:
    """Resolve the per-preset shard fan-out for duration sweeps."""
    if requested_concurrency <= 0:
        raise _benchmark_execution_error_type()(
            "Duration sweep concurrency must be greater than zero."
        )
    return requested_concurrency


def _format_duration_sweep_child_label(
    duration_preset: str,
    *,
    shard_index: int,
    shard_count: int,
) -> str:
    """Render one operator-facing preset child label."""
    if shard_count <= 1:
        return duration_preset
    return f"{duration_preset}[s{shard_index}/{shard_count}]"


def _build_duration_sweep_child_stem(
    duration_preset: str,
    *,
    shard_index: int,
    shard_count: int,
) -> str:
    """Build the filesystem stem for one duration-sweep child."""
    if shard_count <= 1:
        return duration_preset
    return f"{duration_preset}.shard-{shard_index}-of-{shard_count}"


def _estimate_duration_sweep_shard_sample_count(
    *,
    max_samples: int,
    shard_index: int,
    shard_count: int,
) -> int:
    """Estimate one shard's selected sample count from a bounded preset cap."""
    if max_samples <= 0:
        return 0
    base_count, remainder = divmod(max_samples, shard_count)
    return base_count + (1 if shard_index <= remainder else 0)


def _build_initial_duration_sweep_worker_assignments(
    *,
    presets: list[str],
    seeded_preset_count: int,
    preset_concurrency: int,
) -> tuple[dict[str, list[int]], list[str], int]:
    """Distribute the initial worker slots round-robin across seeded presets."""
    seeded_presets = presets[:seeded_preset_count]
    assignments = {duration_preset: [] for duration_preset in seeded_presets}
    total_worker_budget = seeded_preset_count * preset_concurrency
    for worker_id in range(1, total_worker_budget + 1):
        duration_preset = seeded_presets[(worker_id - 1) % seeded_preset_count]
        assignments[duration_preset].append(worker_id)
    return assignments, presets[seeded_preset_count:], total_worker_budget


def _count_duration_sweep_preset_completed_samples(
    preset_state: _DurationSweepPresetState,
) -> int:
    """Count finished samples across active and completed children for one preset."""
    return sum(
        summary.sample_count for summary in preset_state.completed_child_summaries.values()
    ) + sum(child_state.completed_samples for child_state in preset_state.child_states.values())


def _resolve_duration_sweep_preset_sample_count(
    *,
    preset_state: _DurationSweepPresetState | None,
    completed_summary: DurationSweepPresetSummary | None,
    examples_per_preset: int,
) -> int | None:
    """Resolve the planned sample count for one preset when it is known."""
    if completed_summary is not None:
        return completed_summary.sample_count
    if preset_state is not None and preset_state.sample_count is not None:
        return preset_state.sample_count
    if examples_per_preset > 0:
        return examples_per_preset
    return None


def _find_next_duration_sweep_donation_target(
    *,
    presets: list[str],
    completed_summaries: dict[str, DurationSweepPresetSummary],
    current_preset: str,
) -> str | None:
    """Return the next unfinished preset after the completed donor preset."""
    current_index = presets.index(current_preset)
    for candidate in presets[current_index + 1 :]:
        if candidate not in completed_summaries:
            return candidate
    return None


def _estimate_duration_sweep_eta_seconds(
    *,
    presets: list[str],
    active_presets: dict[str, _DurationSweepPresetState],
    completed_summaries: dict[str, DurationSweepPresetSummary],
    examples_per_preset: int,
    accumulated_worker_seconds: float,
) -> float | None:
    """Estimate sweep ETA from observed per-worker throughput and future worker donations."""
    observed_completed_work = 0.0
    remaining_work_by_preset: dict[str, float] = {}
    active_workers_by_preset: dict[str, int] = {}

    for duration_preset in presets:
        target_seconds = float(DURATION_PRESET_SECONDS[duration_preset])
        completed_summary = completed_summaries.get(duration_preset)
        preset_state = active_presets.get(duration_preset)
        planned_samples = _resolve_duration_sweep_preset_sample_count(
            preset_state=preset_state,
            completed_summary=completed_summary,
            examples_per_preset=examples_per_preset,
        )
        if planned_samples is None:
            return None
        if completed_summary is not None:
            observed_completed_work += planned_samples * target_seconds
            continue
        completed_samples = (
            _count_duration_sweep_preset_completed_samples(preset_state)
            if preset_state is not None
            else 0
        )
        observed_completed_work += completed_samples * target_seconds
        remaining_work_by_preset[duration_preset] = (
            max(planned_samples - completed_samples, 0) * target_seconds
        )
        active_workers_by_preset[duration_preset] = (
            len(preset_state.child_states) if preset_state is not None else 0
        )

    if not remaining_work_by_preset:
        return 0.0
    if observed_completed_work <= 0.0 or accumulated_worker_seconds <= 0.0:
        return None
    per_worker_work_rate = observed_completed_work / accumulated_worker_seconds
    if per_worker_work_rate <= 0.0:
        return None

    simulated_remaining_seconds = 0.0
    simulated_remaining_work = dict(remaining_work_by_preset)
    simulated_workers = dict(active_workers_by_preset)
    while True:
        active_work_items = [
            (duration_preset, remaining_work, simulated_workers.get(duration_preset, 0))
            for duration_preset, remaining_work in simulated_remaining_work.items()
            if remaining_work > 0.0 and simulated_workers.get(duration_preset, 0) > 0
        ]
        if not active_work_items:
            return (
                0.0
                if all(remaining_work <= 0.0 for remaining_work in simulated_remaining_work.values())
                else None
            )
        next_completion_seconds = min(
            remaining_work / (worker_count * per_worker_work_rate)
            for _, remaining_work, worker_count in active_work_items
        )
        if next_completion_seconds <= 0.0:
            return None
        simulated_remaining_seconds += next_completion_seconds
        finished_presets: list[str] = []
        for duration_preset, remaining_work, worker_count in active_work_items:
            updated_work = remaining_work - (
                worker_count * per_worker_work_rate * next_completion_seconds
            )
            if updated_work <= 1e-9:
                simulated_remaining_work[duration_preset] = 0.0
                finished_presets.append(duration_preset)
            else:
                simulated_remaining_work[duration_preset] = updated_work
        if not finished_presets:
            return None
        for duration_preset in finished_presets:
            donated_workers = simulated_workers.get(duration_preset, 0)
            simulated_workers[duration_preset] = 0
            donor_index = presets.index(duration_preset)
            donation_target = next(
                (
                    candidate
                    for candidate in presets[donor_index + 1 :]
                    if simulated_remaining_work.get(candidate, 0.0) > 0.0
                ),
                None,
            )
            if donation_target is not None:
                simulated_workers[donation_target] = (
                    simulated_workers.get(donation_target, 0) + donated_workers
                )
        if all(remaining_work <= 0.0 for remaining_work in simulated_remaining_work.values()):
            return simulated_remaining_seconds


def _reserve_port() -> int:
    """Reserve one localhost port number for a child benchmark process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _reserve_distinct_port_triplet(used_ports: set[int]) -> tuple[int, int, int]:
    """Reserve three distinct localhost ports not yet used in this sweep."""
    ports: list[int] = []
    while len(ports) < 3:
        port = _reserve_port()
        if port in used_ports:
            continue
        used_ports.add(port)
        ports.append(port)
    return ports[0], ports[1], ports[2]


def _build_failed_preset_summary(
    *,
    duration_preset: str,
    sample_count: int,
    error_message: str,
    restart_count: int = 0,
    recovered_stall_count: int = 0,
) -> DurationSweepPresetSummary:
    """Build a failed preset summary when execute_benchmark raises."""
    return DurationSweepPresetSummary(
        duration_preset=duration_preset,
        target_seconds=int(DURATION_PRESET_SECONDS[duration_preset]),
        status="failed",
        sample_count=sample_count,
        executed_count=0,
        skipped_count=0,
        success_count=0,
        failure_count=sample_count,
        within_tolerance_count=0,
        within_tolerance_rate=0.0,
        mean_candidate_seconds=0.0,
        mean_absolute_duration_delta_seconds=0.0,
        mean_factualness=0.0,
        mean_naturalness=0.0,
        mean_speaker_grammar_similarity=0.0,
        mean_composite=0.0,
        report_path=None,
        notes_path=None,
        results_path=None,
        mean_composite_ci95_lower=None,
        mean_composite_ci95_upper=None,
        error_message=error_message,
        restart_count=restart_count,
        recovered_stall_count=recovered_stall_count,
    )


def _load_execution_report(report_path: Path) -> BenchmarkExecutionReport:
    """Load one SPoRC execute report from JSON into benchmark dataclasses."""
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid SPoRC report payload in {report_path}.")

    raw_summary = payload.pop("summary", None)
    if not isinstance(raw_summary, dict):
        raise ValueError(f"Missing summary payload in {report_path}.")

    raw_samples = payload.pop("samples", None)
    if not isinstance(raw_samples, list):
        raise ValueError(f"Missing sample payload in {report_path}.")

    samples: list[SampleBenchmarkResult] = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            raise ValueError(f"Invalid sample payload in {report_path}.")
        sample_payload = dict(raw_sample)
        raw_metrics = sample_payload.pop("metrics", None)
        if not isinstance(raw_metrics, dict):
            raise ValueError(f"Missing sample metrics in {report_path}.")
        samples.append(
            SampleBenchmarkResult(
                metrics=SampleMetricBundle(**raw_metrics),
                **sample_payload,
            )
        )

    return BenchmarkExecutionReport(
        summary=BenchmarkSummary(**raw_summary),
        samples=samples,
        **payload,
    )


def _summarize_duration_report(
    report: BenchmarkExecutionReport,
    *,
    duration_preset: str,
    restart_count: int = 0,
    recovered_stall_count: int | None = None,
) -> DurationSweepPresetSummary:
    """Collapse one execute report into a sweep-level preset summary."""
    sample_count = len(report.samples)
    executed_samples = [
        sample for sample in report.samples if sample.status != "skipped"
    ]
    executed_count = len(executed_samples)
    tolerance_ratio = float(report.duration_tolerance_ratio)
    within_tolerance_count = sum(
        1
        for sample in executed_samples
        if abs(sample.candidate_seconds - sample.target_seconds)
        <= sample.target_seconds * tolerance_ratio
    )
    mean_candidate_seconds = (
        sum(sample.candidate_seconds for sample in executed_samples) / executed_count
        if executed_count
        else 0.0
    )
    mean_absolute_duration_delta_seconds = (
        sum(
            abs(sample.candidate_seconds - sample.target_seconds)
            for sample in executed_samples
        )
        / executed_count
        if executed_count
        else 0.0
    )
    composite_ci95 = mean_confidence_interval_95(
        [sample.metrics.composite for sample in executed_samples]
    )
    derived_recovered_stall_count = sum(
        1
        for sample in report.samples
        if sample.final_status == _SWEEP_STALL_FINAL_STATUS
    )
    return DurationSweepPresetSummary(
        duration_preset=duration_preset,
        target_seconds=int(DURATION_PRESET_SECONDS[duration_preset]),
        status=report.status,
        sample_count=sample_count,
        executed_count=report.summary.executed_count,
        skipped_count=report.summary.skipped_count,
        success_count=report.summary.success_count,
        failure_count=report.summary.failure_count,
        within_tolerance_count=within_tolerance_count,
        within_tolerance_rate=(
            round(within_tolerance_count / executed_count, 4)
            if executed_count
            else 0.0
        ),
        mean_candidate_seconds=round(mean_candidate_seconds, 4),
        mean_absolute_duration_delta_seconds=round(
            mean_absolute_duration_delta_seconds,
            4,
        ),
        mean_factualness=report.summary.mean_factualness,
        mean_naturalness=report.summary.mean_naturalness,
        mean_speaker_grammar_similarity=(
            report.summary.mean_speaker_grammar_similarity
        ),
        mean_composite=report.summary.mean_composite,
        report_path=report.report_path,
        notes_path=report.notes_path,
        results_path=report.results_path,
        mean_composite_ci95_lower=(
            round(composite_ci95[0], 4) if composite_ci95 is not None else None
        ),
        mean_composite_ci95_upper=(
            round(composite_ci95[1], 4) if composite_ci95 is not None else None
        ),
        error_message=None,
        restart_count=max(int(report.resume_count), int(restart_count)),
        recovered_stall_count=(
            max(derived_recovered_stall_count, int(recovered_stall_count))
            if recovered_stall_count is not None
            else derived_recovered_stall_count
        ),
    )


def _parse_duration_sweep_progress_line(line: str) -> dict[str, object] | None:
    """Parse one machine-readable child progress event line."""
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    event_name = payload.get(_PROGRESS_EVENT_KEY)
    return payload if isinstance(event_name, str) else None


def _update_duration_sweep_child_state(
    state: _DurationSweepChildState,
    payload: dict[str, object],
) -> None:
    """Apply one parsed progress event to the tracked child state."""
    event_name = str(payload.get(_PROGRESS_EVENT_KEY, ""))
    now_monotonic = time.monotonic()
    raw_sample_count = payload.get("sample_count")
    if isinstance(raw_sample_count, int) and raw_sample_count >= 0:
        state.sample_count = raw_sample_count

    if event_name == "sample_start":
        raw_sample_index = payload.get("sample_index")
        if isinstance(raw_sample_index, int) and raw_sample_index > 0:
            state.active_sample_index = raw_sample_index
        raw_sample_id = payload.get("sample_id")
        state.active_sample_id = (
            str(raw_sample_id) if isinstance(raw_sample_id, str) and raw_sample_id else None
        )
        state.active_stage = "sample_start"
        state.active_iteration = None
        state.last_heartbeat_at = now_monotonic
        state.last_activity_at = now_monotonic
        state.last_activity_age_seconds = 0.0
        return

    if event_name == "sample_heartbeat":
        raw_sample_index = payload.get("sample_index")
        if isinstance(raw_sample_index, int) and raw_sample_index > 0:
            state.active_sample_index = raw_sample_index
        raw_sample_id = payload.get("sample_id")
        if isinstance(raw_sample_id, str) and raw_sample_id:
            state.active_sample_id = raw_sample_id
        raw_stage = payload.get("stage")
        state.active_stage = (
            str(raw_stage) if isinstance(raw_stage, str) and raw_stage else None
        )
        raw_iteration = payload.get("iteration")
        state.active_iteration = (
            int(raw_iteration)
            if isinstance(raw_iteration, int)
            else None
        )
        raw_activity_age = payload.get("last_activity_age_seconds")
        if isinstance(raw_activity_age, (int, float)):
            state.last_activity_age_seconds = float(raw_activity_age)
            state.last_activity_at = max(
                0.0,
                now_monotonic - float(raw_activity_age),
            )
        state.last_heartbeat_at = now_monotonic
        return

    if event_name == "sample_done":
        state.completed_samples += 1
        raw_final_status = payload.get("final_status")
        if raw_final_status == _SWEEP_STALL_FINAL_STATUS:
            state.recovered_stall_count += 1
        state.active_sample_index = None
        state.active_sample_id = None
        state.active_stage = None
        state.active_iteration = None
        state.last_heartbeat_at = None
        state.last_activity_at = None
        state.last_activity_age_seconds = None


def _update_duration_sweep_preset_state(
    preset_state: _DurationSweepPresetState,
    payload: dict[str, object],
) -> None:
    """Apply preset-level progress metadata derived from one child payload."""
    raw_sample_count = payload.get("sample_count")
    if isinstance(raw_sample_count, int) and raw_sample_count >= 0:
        preset_state.sample_count = raw_sample_count


def _format_duration_sweep_progress(
    *,
    presets: list[str],
    active_presets: dict[str, _DurationSweepPresetState],
    completed_summaries: dict[str, DurationSweepPresetSummary],
    timer: PerfTimer,
    examples_per_preset: int,
    accumulated_worker_seconds: float,
) -> _DurationSweepProgressState:
    """Render one aggregate duration-sweep progress line."""
    active_labels: list[str] = []
    total_completed = 0
    total_samples = 0
    total_known = True

    for duration_preset in presets:
        completed_summary = completed_summaries.get(duration_preset)
        if completed_summary is not None:
            total_completed += completed_summary.sample_count
            total_samples += completed_summary.sample_count
            continue

        preset_state = active_presets.get(duration_preset)
        if preset_state is None:
            if examples_per_preset > 0:
                total_samples += examples_per_preset
            else:
                total_known = False
            continue

        total_completed += _count_duration_sweep_preset_completed_samples(preset_state)
        planned_samples = _resolve_duration_sweep_preset_sample_count(
            preset_state=preset_state,
            completed_summary=None,
            examples_per_preset=examples_per_preset,
        )
        if planned_samples is None:
            total_known = False
        else:
            total_samples += planned_samples

        for child_state in sorted(
            preset_state.child_states.values(),
            key=lambda state: state.shard_index,
        ):
            if child_state.process.poll() is not None:
                continue
            child_label = _format_duration_sweep_child_label(
                duration_preset,
                shard_index=child_state.shard_index,
                shard_count=child_state.shard_count,
            )
            planned_samples = child_state.sample_count
            if (
                child_state.active_sample_index is not None
                and planned_samples is not None
            ):
                label = (
                    f"{child_label}:{child_state.active_sample_index}/{planned_samples}"
                )
            elif child_state.active_sample_index is not None:
                label = f"{child_label}:{child_state.active_sample_index}/?"
            elif planned_samples is not None:
                label = (
                    f"{child_label}:{child_state.completed_samples}/{planned_samples}"
                )
            else:
                label = f"{child_label}:starting"
            if child_state.active_stage:
                label = (
                    f"{label}@{child_state.active_stage}"
                    if child_state.active_iteration is None
                    else f"{label}@{child_state.active_stage}#{child_state.active_iteration}"
                )
            if child_state.restart_count > 0:
                label = f"{label}(r{child_state.restart_count})"
            active_labels.append(label)

    if total_known and total_samples > 0:
        _, average_text, _ = _progress_snapshot(
            completed_samples=total_completed,
            total_samples=total_samples,
            elapsed_seconds=timer.elapsed_seconds(),
        )
        eta_seconds = _estimate_duration_sweep_eta_seconds(
            presets=presets,
            active_presets=active_presets,
            completed_summaries=completed_summaries,
            examples_per_preset=examples_per_preset,
            accumulated_worker_seconds=accumulated_worker_seconds,
        )
        eta_text = _format_eta(eta_seconds)
        overall_text = f"{total_completed}/{total_samples}"
    else:
        average_text = "n/a"
        eta_text = "n/a"
        overall_text = f"{total_completed}/?"

    active_text = ", ".join(active_labels) if active_labels else "idle"
    return _DurationSweepProgressState(
        rendered=(
            f"Sweep progress: overall={overall_text} "
            f"active=[{active_text}] "
            f"completed_presets={len(completed_summaries)}/{len(presets)} "
            f"rolling_avg={average_text} eta={eta_text} "
            f"elapsed={timer.elapsed_seconds():.1f}s"
        ),
        total_completed=total_completed,
        total_samples=total_samples if total_known else None,
        active_text=active_text,
        completed_presets=len(completed_summaries),
        preset_count=len(presets),
    )


def _start_duration_sweep_reader(
    *,
    state: _DurationSweepChildState,
    event_queue: queue.Queue[tuple[str, int, int, dict[str, object]]],
) -> threading.Thread | None:
    """Start a background line pump for one duration-sweep child."""
    stream = state.process.stdout
    if stream is None:
        return None

    def _pump() -> None:
        for line in stream:
            state.log_handle.write(line)
            state.log_handle.flush()
            payload = _parse_duration_sweep_progress_line(line)
            if payload is not None:
                event_queue.put(
                    (
                        state.duration_preset,
                        state.shard_index,
                        state.process.pid,
                        payload,
                    )
                )

    thread = threading.Thread(
        target=_pump,
        name=(
            "sporc-duration-sweep-"
            f"{_build_duration_sweep_child_stem(state.duration_preset, shard_index=state.shard_index, shard_count=state.shard_count)}-"
            f"{state.process.pid}"
        ),
        daemon=True,
    )
    thread.start()
    return thread


def _launch_duration_sweep_child(
    *,
    args: argparse.Namespace,
    sweep_root: Path,
    duration_preset: str,
    shard_index: int,
    shard_count: int,
    max_samples: int,
    used_ports: set[int],
    event_queue: queue.Queue[tuple[str, int, int, dict[str, object]]],
    shared_sample_queue_root: Path | None = None,
    checkpoint_path: Path | None = None,
    resume_from_checkpoint: bool = False,
    resume_failure_reason: str | None = None,
    restart_count: int = 0,
    recovered_stall_count: int = 0,
) -> _DurationSweepChildState:
    """Launch one preset child execute process for the duration sweep."""
    progress_heartbeat_seconds = float(
        getattr(args, "progress_heartbeat_seconds", 30.0)
    )
    logs_root = sweep_root / "logs"
    logs_root.mkdir(parents=True, exist_ok=True)
    checkpoints_root = sweep_root / "checkpoints"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    child_stem = _build_duration_sweep_child_stem(
        duration_preset,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    child_label = _format_duration_sweep_child_label(
        duration_preset,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    log_path = logs_root / f"{child_stem}.log"
    resolved_checkpoint_path = (
        checkpoint_path
        if checkpoint_path is not None
        else checkpoints_root / f"{child_stem}.checkpoint.json"
    )
    log_handle = log_path.open(
        "a" if resume_from_checkpoint and log_path.exists() else "w",
        encoding="utf-8",
    )
    retrieval_port, summarizer_port, critic_port = _reserve_distinct_port_triplet(
        used_ports
    )
    child_command = _build_execute_child_command(
        args,
        duration_preset=duration_preset,
        target_seconds_override=None,
        max_samples=max_samples,
        transcript_only=not bool(args.enable_voice_clone),
        retrieval_port=retrieval_port,
        summarizer_port=summarizer_port,
        critic_port=critic_port,
        disable_stream_output=True,
        emit_progress_events=True,
        checkpoint_path=resolved_checkpoint_path,
        resume_from_checkpoint=resume_from_checkpoint,
        resume_failure_reason=resume_failure_reason,
        emit_progress_heartbeats=True,
        progress_heartbeat_seconds=progress_heartbeat_seconds,
        apply_sweep_timeout_defaults=True,
        shared_sample_queue_root=shared_sample_queue_root,
        sample_shard_index=(1 if shared_sample_queue_root is not None else shard_index),
        sample_shard_count=(1 if shared_sample_queue_root is not None else shard_count),
    )
    log_handle.write(
        (
            f"[Sweep] {'Restarting' if resume_from_checkpoint else 'Starting'} preset {child_label}\n"
            f"[Sweep] Command: {' '.join(child_command)}\n"
            f"[Sweep] Ports: retrieval={retrieval_port} "
            f"summarizer={summarizer_port} critic={critic_port}\n"
            f"[Sweep] Checkpoint: {resolved_checkpoint_path}\n"
        )
    )
    log_handle.flush()
    try:
        process = subprocess.Popen(
            child_command,
            cwd=str(REPO_ROOT),
            env=_child_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            **detached_popen_kwargs(),
        )
    except Exception:
        log_handle.close()
        raise
    state = _DurationSweepChildState(
        duration_preset=duration_preset,
        shard_index=shard_index,
        shard_count=shard_count,
        process=process,
        log_path=log_path,
        log_handle=log_handle,
        reader_thread=None,
        retrieval_port=retrieval_port,
        summarizer_port=summarizer_port,
        critic_port=critic_port,
        max_samples=max_samples,
        checkpoint_path=resolved_checkpoint_path,
        output_root=Path(
            getattr(args, "output_root", REPO_ROOT / "artifacts" / "sporc_runs")
        ).resolve(),
        shared_sample_queue_root=shared_sample_queue_root,
        restart_count=restart_count,
        recovered_stall_count=recovered_stall_count,
    )
    state.reader_thread = _start_duration_sweep_reader(
        state=state,
        event_queue=event_queue,
    )
    return state


def _discover_report_path_from_checkpoint(
    state: _DurationSweepChildState,
) -> Path | None:
    """Resolve the latest report snapshot path from the execute checkpoint."""
    if not state.checkpoint_path.exists():
        return None
    try:
        checkpoint_payload = json.loads(
            state.checkpoint_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(checkpoint_payload, dict):
        return None
    run_id = str(checkpoint_payload.get("run_id", "") or "").strip()
    if not run_id:
        return None
    report_path = (state.output_root / run_id / "report.json").resolve()
    return report_path if report_path.exists() else None


def _stop_duration_sweep_child(
    state: _DurationSweepChildState,
    *,
    reason: str,
) -> int | None:
    """Terminate one running sweep child and close its background reader."""
    if state.process.poll() is not None and state.log_handle.closed:
        return state.process.poll()
    if not state.log_handle.closed:
        state.log_handle.write(f"[Sweep] {reason}\n")
        state.log_handle.flush()
    if state.process.poll() is None:
        _terminate_process_tree(state.process)
    try:
        exit_code = state.process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        exit_code = state.process.poll()
    if state.reader_thread is not None:
        state.reader_thread.join(timeout=5.0)
    if exit_code is not None and not state.log_handle.closed:
        state.log_handle.write(f"[Sweep] Child exited with code {exit_code}\n")
        state.log_handle.flush()
    if not state.log_handle.closed:
        state.log_handle.close()
    return exit_code


def _stop_active_duration_sweep_children(
    active_presets: dict[str, _DurationSweepPresetState],
    *,
    reason: str,
) -> None:
    """Stop every still-running sweep child and close its active resources."""
    for preset_state in active_presets.values():
        for child_state in preset_state.child_states.values():
            _stop_duration_sweep_child(child_state, reason=reason)


def _detect_duration_sweep_child_stall(
    state: _DurationSweepChildState,
    *,
    now_monotonic: float,
    child_stall_seconds: float,
) -> str | None:
    """Return a watchdog restart reason when a child appears stalled."""
    if state.process.poll() is not None or state.active_sample_id is None:
        return None
    if (
        state.last_heartbeat_at is not None
        and now_monotonic - state.last_heartbeat_at > _SWEEP_CHILD_NO_HEARTBEAT_SECONDS
    ):
        return (
            f"sample {state.active_sample_id} stopped heartbeating for "
            f"{now_monotonic - state.last_heartbeat_at:.1f}s "
            f"(stage={state.active_stage or 'unknown'})."
        )
    if (
        state.last_activity_age_seconds is not None
        and state.last_activity_age_seconds > child_stall_seconds
    ):
        return (
            f"sample {state.active_sample_id} reported no new activity for "
            f"{state.last_activity_age_seconds:.1f}s "
            f"(stage={state.active_stage or 'unknown'})."
        )
    return None


def _restart_duration_sweep_child(
    *,
    state: _DurationSweepChildState,
    args: argparse.Namespace,
    sweep_root: Path,
    used_ports: set[int],
    event_queue: queue.Queue[tuple[str, int, int, dict[str, object]]],
    reason: str,
) -> _DurationSweepChildState:
    """Restart one stalled preset child from its execute checkpoint."""
    _stop_duration_sweep_child(
        state,
        reason=(
            "Watchdog restarting preset "
            f"{_format_duration_sweep_child_label(state.duration_preset, shard_index=state.shard_index, shard_count=state.shard_count)} "
            f"after stall: {reason}"
        ),
    )
    restarted_state = _launch_duration_sweep_child(
        args=args,
        sweep_root=sweep_root,
        duration_preset=state.duration_preset,
        shard_index=state.shard_index,
        shard_count=state.shard_count,
        max_samples=state.max_samples,
        used_ports=used_ports,
        event_queue=event_queue,
        shared_sample_queue_root=state.shared_sample_queue_root,
        checkpoint_path=state.checkpoint_path,
        resume_from_checkpoint=True,
        resume_failure_reason=_SWEEP_STALL_FINAL_STATUS,
        restart_count=state.restart_count + 1,
        recovered_stall_count=state.recovered_stall_count,
    )
    restarted_state.sample_count = state.sample_count
    restarted_state.completed_samples = state.completed_samples
    return restarted_state


def _launch_duration_sweep_preset(
    *,
    args: argparse.Namespace,
    sweep_root: Path,
    duration_preset: str,
    max_samples: int,
    worker_ids: list[int],
    worker_budget: int,
    used_ports: set[int],
    event_queue: queue.Queue[tuple[str, int, int, dict[str, object]]],
    preset_state: _DurationSweepPresetState | None = None,
) -> _DurationSweepPresetState:
    """Launch one or more worker-slot children for a duration preset."""
    if not worker_ids:
        raise ValueError("Duration sweep launch requires at least one worker id.")
    resolved_preset_state = preset_state or _DurationSweepPresetState(
        duration_preset=duration_preset,
        max_samples=max_samples,
        worker_budget=worker_budget,
        queue_root=(sweep_root / "queues" / duration_preset),
    )
    try:
        for worker_id in worker_ids:
            resolved_preset_state.child_states[worker_id] = _launch_duration_sweep_child(
                args=args,
                sweep_root=sweep_root,
                duration_preset=duration_preset,
                shard_index=worker_id,
                shard_count=worker_budget,
                max_samples=max_samples,
                used_ports=used_ports,
                event_queue=event_queue,
                shared_sample_queue_root=resolved_preset_state.queue_root,
            )
    except Exception:
        for child_state in resolved_preset_state.child_states.values():
            _stop_duration_sweep_child(
                child_state,
                reason=(
                    "Aborting preset launch after a sibling shard failed to start."
                ),
            )
        raise
    return resolved_preset_state


def _aggregate_duration_sweep_preset(
    *,
    duration_preset: str,
    child_summaries: dict[int, DurationSweepPresetSummary],
) -> DurationSweepPresetSummary:
    """Combine shard child summaries into one preset-level sweep summary."""
    ordered_children = [
        child_summaries[shard_index] for shard_index in sorted(child_summaries)
    ]
    if len(ordered_children) == 1:
        return ordered_children[0]

    sample_count = 0
    executed_count = 0
    skipped_count = 0
    success_count = 0
    failure_count = 0
    within_tolerance_count = 0
    candidate_seconds_total = 0.0
    absolute_delta_total = 0.0
    factualness_total = 0.0
    naturalness_total = 0.0
    speaker_grammar_total = 0.0
    composite_total = 0.0
    composite_values: list[float] = []
    ci_available = True
    error_messages: list[str] = []
    restart_count = 0
    recovered_stall_count = 0

    artifact_children = [
        summary for summary in ordered_children if summary.sample_count > 0
    ]
    selected_artifact_child = artifact_children[0] if len(artifact_children) == 1 else None

    for child_summary in ordered_children:
        sample_count += int(child_summary.sample_count)
        executed_count += int(child_summary.executed_count)
        skipped_count += int(child_summary.skipped_count)
        success_count += int(child_summary.success_count)
        failure_count += int(child_summary.failure_count)
        within_tolerance_count += int(child_summary.within_tolerance_count)
        candidate_seconds_total += (
            float(child_summary.mean_candidate_seconds) * int(child_summary.executed_count)
        )
        absolute_delta_total += (
            float(child_summary.mean_absolute_duration_delta_seconds)
            * int(child_summary.executed_count)
        )
        factualness_total += (
            float(child_summary.mean_factualness) * int(child_summary.executed_count)
        )
        naturalness_total += (
            float(child_summary.mean_naturalness) * int(child_summary.executed_count)
        )
        speaker_grammar_total += (
            float(child_summary.mean_speaker_grammar_similarity)
            * int(child_summary.executed_count)
        )
        composite_total += (
            float(child_summary.mean_composite) * int(child_summary.executed_count)
        )
        restart_count += int(child_summary.restart_count)
        recovered_stall_count += int(child_summary.recovered_stall_count)
        if child_summary.error_message:
            error_messages.append(child_summary.error_message)

        child_report_path = str(child_summary.report_path or "").strip()
        if child_summary.executed_count <= 0:
            continue
        if not child_report_path:
            ci_available = False
            continue
        child_report = _load_execution_report(Path(child_report_path))
        composite_values.extend(
            sample.metrics.composite
            for sample in child_report.samples
            if sample.status != "skipped"
        )

    if executed_count == 0:
        status = "failed"
    elif failure_count == 0 and skipped_count == 0:
        status = "completed"
    else:
        status = "partial"

    composite_ci95 = (
        mean_confidence_interval_95(composite_values)
        if ci_available and len(composite_values) == executed_count
        else None
    )

    return DurationSweepPresetSummary(
        duration_preset=duration_preset,
        target_seconds=int(DURATION_PRESET_SECONDS[duration_preset]),
        status=status,
        sample_count=sample_count,
        executed_count=executed_count,
        skipped_count=skipped_count,
        success_count=success_count,
        failure_count=failure_count,
        within_tolerance_count=within_tolerance_count,
        within_tolerance_rate=(
            round(within_tolerance_count / executed_count, 4)
            if executed_count
            else 0.0
        ),
        mean_candidate_seconds=round(
            candidate_seconds_total / executed_count,
            4,
        )
        if executed_count
        else 0.0,
        mean_absolute_duration_delta_seconds=round(
            absolute_delta_total / executed_count,
            4,
        )
        if executed_count
        else 0.0,
        mean_factualness=round(factualness_total / executed_count, 4)
        if executed_count
        else 0.0,
        mean_naturalness=round(naturalness_total / executed_count, 4)
        if executed_count
        else 0.0,
        mean_speaker_grammar_similarity=round(
            speaker_grammar_total / executed_count,
            4,
        )
        if executed_count
        else 0.0,
        mean_composite=round(composite_total / executed_count, 4)
        if executed_count
        else 0.0,
        report_path=selected_artifact_child.report_path if selected_artifact_child else None,
        notes_path=selected_artifact_child.notes_path if selected_artifact_child else None,
        results_path=selected_artifact_child.results_path if selected_artifact_child else None,
        mean_composite_ci95_lower=(
            round(composite_ci95[0], 4) if composite_ci95 is not None else None
        ),
        mean_composite_ci95_upper=(
            round(composite_ci95[1], 4) if composite_ci95 is not None else None
        ),
        error_message="; ".join(error_messages) if error_messages else None,
        restart_count=restart_count,
        recovered_stall_count=recovered_stall_count,
    )


def _finalize_duration_sweep_child(
    *,
    state: _DurationSweepChildState,
    fallback_sample_count: int,
    forced_status: str | None = None,
    forced_error_message: str | None = None,
) -> DurationSweepPresetSummary:
    """Finalize one duration-sweep child process into a preset summary."""
    try:
        exit_code = state.process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        _terminate_process_tree(state.process)
        exit_code = state.process.wait(timeout=5.0)
    if state.reader_thread is not None:
        state.reader_thread.join(timeout=5.0)
    state.log_handle.write(f"[Sweep] Child exited with code {exit_code}\n")
    state.log_handle.flush()
    state.log_handle.close()

    terminal_payload = _discover_terminal_payload(state.log_path)
    report_path_value = terminal_payload.get("report_path")
    report_path: Path | None = None
    if isinstance(report_path_value, str) and report_path_value.strip():
        candidate_report_path = Path(report_path_value).resolve()
        if candidate_report_path.exists():
            report_path = candidate_report_path
    if report_path is None:
        report_path = _discover_report_path_from_checkpoint(state)
    if report_path is not None:
        report_summary = _summarize_duration_report(
            _load_execution_report(report_path),
            duration_preset=state.duration_preset,
            restart_count=state.restart_count,
            recovered_stall_count=state.recovered_stall_count,
        )
        if forced_status is None and forced_error_message is None:
            return report_summary
        return replace(
            report_summary,
            status=forced_status or report_summary.status,
            error_message=forced_error_message,
        )

    error_message = forced_error_message or terminal_payload.get("error")
    if not isinstance(error_message, str) or not error_message.strip():
        error_message = (
            f"Preset {state.duration_preset} exited with code {exit_code} "
            "without writing a report."
        )
    resolved_sample_count = (
        state.sample_count if state.sample_count is not None else fallback_sample_count
    )
    return _build_failed_preset_summary(
        duration_preset=state.duration_preset,
        sample_count=max(resolved_sample_count, 0),
        error_message=error_message,
        restart_count=state.restart_count,
        recovered_stall_count=state.recovered_stall_count,
    )


def _run_duration_sweep(args: argparse.Namespace) -> int:
    """Execute one or more duration presets and write aggregate artifacts."""
    global _ACTIVE_SWEEP_PROGRESS_BAR
    if args.mode == "menu":
        print(_render_duration_sweep_menu())
        return 0

    if args.mode == "single":
        single_preset = str(args.duration_preset or "").strip()
        if single_preset not in DURATION_PRESET_SECONDS:
            raise _benchmark_execution_error_type()(
                "--mode single requires one valid --duration-preset value."
            )
        presets = [single_preset]
    else:
        presets = _parse_duration_preset_list(str(args.duration_presets))
    examples = _resolve_duration_sweep_examples(args)
    effective_parallelism = _resolve_duration_sweep_parallelism(
        requested_parallelism=int(args.parallelism),
        preset_count=len(presets),
    )
    preset_concurrency = _resolve_duration_sweep_concurrency(
        requested_concurrency=int(getattr(args, "concurrency", 1))
    )
    write_duration_sweep_artifacts, _ = _load_reporting_helpers(
        status_reporter=_emit_cli_status,
    )

    sweep_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_root = Path(args.sweep_root).resolve() / sweep_id
    sweep_root.mkdir(parents=True, exist_ok=True)
    initial_worker_assignments, pending_presets, total_worker_budget = (
        _build_initial_duration_sweep_worker_assignments(
            presets=presets,
            seeded_preset_count=effective_parallelism,
            preset_concurrency=preset_concurrency,
        )
    )
    active_presets: dict[str, _DurationSweepPresetState] = {}
    preset_summaries_by_preset: dict[str, DurationSweepPresetSummary] = {}
    used_ports: set[int] = set()
    event_queue: queue.Queue[tuple[str, int, int, dict[str, object]]] = queue.Queue()
    last_progress: str | None = None
    next_heartbeat_at = 0.0
    accumulated_worker_seconds = 0.0
    child_stall_seconds = float(
        getattr(args, "child_stall_seconds", _SWEEP_CHILD_DEFAULT_STALL_SECONDS)
    )
    max_child_restarts = int(getattr(args, "max_child_restarts", 2))
    previous_progress_bar = _ACTIVE_SWEEP_PROGRESS_BAR
    sweep_progress_bar = _SweepProgressBar(stream=sys.stderr)
    sweep_progress_bar.start()
    _ACTIVE_SWEEP_PROGRESS_BAR = (
        sweep_progress_bar if sweep_progress_bar.is_enabled() else None
    )

    _emit_cli_status(
        (
            f"Starting duration sweep {sweep_id} with presets={', '.join(presets)} "
            f"parallelism={effective_parallelism} "
            f"preset_concurrency={preset_concurrency} "
            f"worker_budget={total_worker_budget} "
            f"examples_per_preset={examples if examples > 0 else 'all'} "
            f"transcript_only={not bool(args.enable_voice_clone)}."
        )
    )

    try:
        with measure_elapsed_time() as sweep_timer:
            previous_worker_sample_at = time.monotonic()
            for duration_preset in presets[:effective_parallelism]:
                preset_state = _launch_duration_sweep_preset(
                    args=args,
                    sweep_root=sweep_root,
                    duration_preset=duration_preset,
                    max_samples=examples,
                    worker_ids=initial_worker_assignments[duration_preset],
                    worker_budget=total_worker_budget,
                    used_ports=used_ports,
                    event_queue=event_queue,
                )
                active_presets[duration_preset] = preset_state
                for child_state in sorted(
                    preset_state.child_states.values(),
                    key=lambda state: state.shard_index,
                ):
                    child_label = _format_duration_sweep_child_label(
                        duration_preset,
                        shard_index=child_state.shard_index,
                        shard_count=child_state.shard_count,
                    )
                    _emit_cli_status(
                        (
                            f"Started preset {child_label} "
                            f"(retrieval={child_state.retrieval_port}, "
                            f"summarizer={child_state.summarizer_port}, "
                            f"critic={child_state.critic_port})."
                        )
                    )

            while active_presets:
                progress_changed = False
                try:
                    duration_preset, shard_index, child_pid, payload = event_queue.get(
                        timeout=0.5
                    )
                except queue.Empty:
                    pass
                else:
                    preset_state = active_presets.get(duration_preset)
                    child_state = (
                        preset_state.child_states.get(shard_index)
                        if preset_state is not None
                        else None
                    )
                    if child_state is not None and child_state.process.pid == child_pid:
                        _update_duration_sweep_child_state(child_state, payload)
                        _update_duration_sweep_preset_state(preset_state, payload)
                        progress_changed = True
                    while True:
                        try:
                            (
                                duration_preset,
                                shard_index,
                                child_pid,
                                payload,
                            ) = event_queue.get_nowait()
                        except queue.Empty:
                            break
                        preset_state = active_presets.get(duration_preset)
                        child_state = (
                            preset_state.child_states.get(shard_index)
                            if preset_state is not None
                            else None
                        )
                        if child_state is not None and child_state.process.pid == child_pid:
                            _update_duration_sweep_child_state(child_state, payload)
                            _update_duration_sweep_preset_state(preset_state, payload)
                            progress_changed = True

                now_monotonic = time.monotonic()
                active_worker_count = sum(
                    len(preset_state.child_states)
                    for preset_state in active_presets.values()
                )
                accumulated_worker_seconds += (
                    max(now_monotonic - previous_worker_sample_at, 0.0)
                    * active_worker_count
                )
                previous_worker_sample_at = now_monotonic
                for duration_preset, preset_state in list(active_presets.items()):
                    for shard_index, child_state in list(preset_state.child_states.items()):
                        child_label = _format_duration_sweep_child_label(
                            duration_preset,
                            shard_index=child_state.shard_index,
                            shard_count=child_state.shard_count,
                        )
                        stall_reason = _detect_duration_sweep_child_stall(
                            child_state,
                            now_monotonic=now_monotonic,
                            child_stall_seconds=child_stall_seconds,
                        )
                        if stall_reason is not None:
                            if child_state.restart_count >= max_child_restarts:
                                child_state.log_handle.write(
                                    "[Sweep] Watchdog exceeded the configured "
                                    "restart threshold for preset "
                                    f"{child_label}; marking the stalled sample "
                                    "failed and continuing the preset from "
                                    f"checkpoint: {stall_reason}\n"
                                )
                                child_state.log_handle.flush()
                            restarted_state = _restart_duration_sweep_child(
                                state=child_state,
                                args=args,
                                sweep_root=sweep_root,
                                used_ports=used_ports,
                                event_queue=event_queue,
                                reason=stall_reason,
                            )
                            preset_state.child_states[shard_index] = restarted_state
                            progress_changed = True
                            if child_state.restart_count >= max_child_restarts:
                                _emit_cli_status(
                                    (
                                        f"Continuing preset {child_label} after "
                                        "stalled sample exceeded the configured "
                                        f"restart threshold ({max_child_restarts}): "
                                        f"{stall_reason}"
                                    )
                                )
                            else:
                                _emit_cli_status(
                                    (
                                        f"Restarted preset {child_label} "
                                        f"restart={restarted_state.restart_count}/"
                                        f"{max_child_restarts} "
                                        f"reason={stall_reason}"
                                    )
                                )
                            continue
                        if child_state.process.poll() is None:
                            continue
                        preset_state.completed_child_summaries[shard_index] = (
                            _finalize_duration_sweep_child(
                                state=child_state,
                                fallback_sample_count=child_state.completed_samples,
                            )
                        )
                        del preset_state.child_states[shard_index]
                        progress_changed = True
                        _emit_cli_status(
                            (
                                f"Finished preset {child_label} "
                                f"status={preset_state.completed_child_summaries[shard_index].status}."
                            )
                        )

                    if preset_state.child_states:
                        continue
                    preset_summaries_by_preset[duration_preset] = (
                        _aggregate_duration_sweep_preset(
                            duration_preset=duration_preset,
                            child_summaries=preset_state.completed_child_summaries,
                        )
                    )
                    donated_worker_ids = sorted(
                        preset_state.completed_child_summaries.keys()
                    )
                    del active_presets[duration_preset]
                    progress_changed = True
                    _emit_cli_status(
                        (
                            f"Finished preset {duration_preset} "
                            f"status={preset_summaries_by_preset[duration_preset].status}."
                        )
                    )
                    donation_target = _find_next_duration_sweep_donation_target(
                        presets=presets,
                        completed_summaries=preset_summaries_by_preset,
                        current_preset=duration_preset,
                    )
                    if donation_target is not None:
                        if donation_target in pending_presets:
                            pending_presets.remove(donation_target)
                        target_state = active_presets.get(donation_target)
                        target_state = _launch_duration_sweep_preset(
                            args=args,
                            sweep_root=sweep_root,
                            duration_preset=donation_target,
                            max_samples=examples,
                            worker_ids=donated_worker_ids,
                            worker_budget=total_worker_budget,
                            used_ports=used_ports,
                            event_queue=event_queue,
                            preset_state=target_state,
                        )
                        active_presets[donation_target] = target_state
                        for worker_id in donated_worker_ids:
                            child_state = target_state.child_states[worker_id]
                            child_label = _format_duration_sweep_child_label(
                                donation_target,
                                shard_index=child_state.shard_index,
                                shard_count=child_state.shard_count,
                            )
                            _emit_cli_status(
                                (
                                    f"Moved worker {worker_id} from {duration_preset} "
                                    f"to {child_label} "
                                    f"(retrieval={child_state.retrieval_port}, "
                                    f"summarizer={child_state.summarizer_port}, "
                                    f"critic={child_state.critic_port})."
                                )
                            )

                elapsed_seconds = sweep_timer.elapsed_seconds()
                if (
                    progress_changed
                    or elapsed_seconds >= next_heartbeat_at
                    or last_progress is None
                ):
                    progress_state = _format_duration_sweep_progress(
                        presets=presets,
                        active_presets=active_presets,
                        completed_summaries=preset_summaries_by_preset,
                        timer=sweep_timer,
                        examples_per_preset=examples,
                        accumulated_worker_seconds=accumulated_worker_seconds,
                    )
                    if sweep_progress_bar.update(progress_state):
                        last_progress = progress_state.rendered
                    elif progress_state.rendered != last_progress:
                        _emit_cli_status(progress_state.rendered)
                        last_progress = progress_state.rendered
                    next_heartbeat_at = (
                        elapsed_seconds + _SWEEP_PROGRESS_HEARTBEAT_SECONDS
                    )
    except KeyboardInterrupt:
        _emit_cli_status(
            "Cancelling duration sweep; closing active child connections..."
        )
        _stop_active_duration_sweep_children(
            active_presets,
            reason="Cancellation requested by operator; shutting down child.",
        )
        raise
    finally:
        _stop_active_duration_sweep_children(
            active_presets,
            reason="Sweep cleanup closing active child resources.",
        )
        sweep_progress_bar.close()
        _ACTIVE_SWEEP_PROGRESS_BAR = previous_progress_bar

    preset_summaries = [
        preset_summaries_by_preset[duration_preset] for duration_preset in presets
    ]

    aggregate_report = DurationSweepReport(
        sweep_id=sweep_id,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        mode=str(args.mode),
        examples_per_preset=examples,
        transcript_only=not bool(args.enable_voice_clone),
        summarizer_provider=str(args.summarizer_provider),
        critic_provider=str(args.critic_provider or args.summarizer_provider),
        judge_provider=str(
            args.judge_provider or args.critic_provider or args.summarizer_provider
        ),
        embedding_mode=str(args.embedding_mode),
        commands_executed=[" ".join(["python", *sys.argv])],
        preset_summaries=preset_summaries,
        report_path="",
        results_path="",
        sample_seed=(
            int(args.sample_seed)
            if getattr(args, "sample_seed", None) is not None
            else None
        ),
        preset_concurrency=preset_concurrency,
    )
    report_path, results_path = write_duration_sweep_artifacts(
        sweep_root=sweep_root,
        report=aggregate_report,
    )
    finalized_report = replace(
        aggregate_report,
        report_path=str(report_path),
        results_path=str(results_path),
    )
    write_duration_sweep_artifacts(sweep_root=sweep_root, report=finalized_report)
    print(
        json.dumps(
            {
                "status": (
                    "completed"
                    if all(
                        summary.status == "completed" for summary in preset_summaries
                    )
                    else "partial"
                ),
                "report_path": str(report_path),
                "results_path": str(results_path),
                "sweep_id": sweep_id,
            }
        )
    )
    return (
        0 if all(summary.status == "completed" for summary in preset_summaries) else 1
    )


def _child_environment() -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    source_root = REPO_ROOT / "src"
    source_text = str(source_root)
    current = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]
    if source_text not in current:
        env["PYTHONPATH"] = os.pathsep.join([source_text, *current])
    return env


def _write_supervisor_output(
    *,
    text: str,
    log_handle: TextIO,
    write_lock: threading.Lock,
) -> None:
    with write_lock:
        sys.stdout.write(text)
        sys.stdout.flush()
        log_handle.write(text)
        log_handle.flush()


def _truncate_supervisor_preview(text: str, *, max_chars: int = 120) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3]}..."


def _stream_process_output(
    *,
    process: subprocess.Popen[str],
    log_handle: TextIO,
    write_lock: threading.Lock,
    stream_state: dict[str, str],
) -> threading.Thread | None:
    stream = process.stdout
    if stream is None:
        return None

    def _pump() -> None:
        while True:
            chunk = stream.read(1)
            if not chunk:
                break
            partial_line = stream_state.get("partial_line", "") + chunk
            stream_state["partial_line"] = partial_line
            while "\n" in stream_state["partial_line"]:
                line, remainder = stream_state["partial_line"].split("\n", 1)
                stream_state["last_line"] = line.strip()
                stream_state["partial_line"] = remainder
            _write_supervisor_output(
                text=chunk,
                log_handle=log_handle,
                write_lock=write_lock,
            )

    thread = threading.Thread(
        target=_pump, name=f"sporc-supervisor-{process.pid}", daemon=True
    )
    thread.start()
    return thread


def _build_execute_child_command(
    args: argparse.Namespace,
    *,
    duration_preset: str | None = None,
    target_seconds_override: int | None | str = "",
    max_samples: int | None = None,
    transcript_only: bool | None = None,
    retrieval_port: int | None = None,
    summarizer_port: int | None = None,
    critic_port: int | None = None,
    disable_stream_output: bool = False,
    emit_progress_events: bool = False,
    checkpoint_path: Path | None = None,
    resume_from_checkpoint: bool = False,
    resume_failure_reason: str | None = None,
    emit_progress_heartbeats: bool = False,
    progress_heartbeat_seconds: float | None = None,
    apply_sweep_timeout_defaults: bool = False,
    shared_sample_queue_root: Path | None = None,
    sample_shard_index: int | None = None,
    sample_shard_count: int | None = None,
) -> list[str]:
    """Build a child execute command with optional sweep-specific overrides."""
    command = [
        sys.executable,
        "-m",
        "card_framework.benchmark.summarizer_critic.sporc",
        "execute",
    ]
    summarizer_timeout = args.summarizer_timeout_seconds
    critic_timeout = args.critic_timeout_seconds
    retrieval_timeout = args.retrieval_timeout_seconds
    index_timeout = args.index_timeout_seconds
    if apply_sweep_timeout_defaults:
        summarizer_timeout = (
            summarizer_timeout if summarizer_timeout is not None else 900.0
        )
        critic_timeout = critic_timeout if critic_timeout is not None else 900.0
        retrieval_timeout = (
            retrieval_timeout if retrieval_timeout is not None else 300.0
        )
        index_timeout = index_timeout if index_timeout is not None else 300.0

    forwarded = {
        "--dataset-manifest": args.dataset_manifest,
        "--dataset-variant": args.dataset_variant,
        "--episode-data-path": args.episode_data_path,
        "--turn-data-path": args.turn_data_path,
        "--reference-wpm": args.reference_wpm,
        "--output-root": args.output_root,
        "--provider-profiles": args.provider_profiles,
        "--config": args.config,
        "--summarizer-provider": args.summarizer_provider,
        "--critic-provider": args.critic_provider,
        "--judge-provider": args.judge_provider,
        "--embedding-mode": args.embedding_mode,
        "--judge-source-char-limit": args.judge_source_char_limit,
        "--duration-preset": (
            duration_preset if duration_preset is not None else args.duration_preset
        ),
        "--target-seconds-override": (
            target_seconds_override
            if target_seconds_override != ""
            else args.target_seconds_override
        ),
        "--max-samples": max_samples if max_samples is not None else args.max_samples,
        "--sample-seed": args.sample_seed,
        "--max-iterations": args.max_iterations,
        "--summarizer-timeout-seconds": summarizer_timeout,
        "--critic-timeout-seconds": critic_timeout,
        "--retrieval-timeout-seconds": retrieval_timeout,
        "--index-timeout-seconds": index_timeout,
        "--retrieval-port": retrieval_port,
        "--summarizer-port": summarizer_port,
        "--critic-port": critic_port,
        "--checkpoint-path": str(checkpoint_path) if checkpoint_path is not None else "",
        "--progress-heartbeat-seconds": progress_heartbeat_seconds,
        "--resume-failure-reason": resume_failure_reason,
        "--shared-sample-queue-root": (
            str(shared_sample_queue_root) if shared_sample_queue_root is not None else ""
        ),
        "--sample-shard-index": (
            sample_shard_index
            if sample_shard_index is not None
            else getattr(args, "sample_shard_index", 1)
        ),
        "--sample-shard-count": (
            sample_shard_count
            if sample_shard_count is not None
            else getattr(args, "sample_shard_count", 1)
        ),
    }
    if args.prepare_if_missing:
        command.append("--prepare-if-missing")
    if args.force_download_dataset:
        command.append("--force-download-dataset")
    if args.force_download_audio:
        command.append("--force-download-audio")
    resolved_transcript_only = (
        transcript_only if transcript_only is not None else args.transcript_only
    )
    if resolved_transcript_only:
        command.append("--transcript-only")
    if disable_stream_output:
        command.append("--disable-stream-output")
    if emit_progress_events:
        command.append("--emit-progress-events")
    if emit_progress_heartbeats:
        command.append("--emit-progress-heartbeats")
    if resume_from_checkpoint:
        command.append("--resume-from-checkpoint")
    for flag, value in forwarded.items():
        if value in (None, ""):
            continue
        command.extend([flag, str(value)])
    return command


def _build_supervised_child_command(args: argparse.Namespace) -> list[str]:
    """Build the execute child command for a supervisor session."""
    return _build_execute_child_command(args)


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    terminate_process_tree(process, timeout_seconds=5.0)


def _discover_terminal_payload(log_path: Path) -> dict[str, object]:
    """Return the last JSON object emitted in a child log, if any."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in reversed([item.strip() for item in text.splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def _discover_report_paths(log_path: Path) -> tuple[str | None, str | None, str | None]:
    terminal_payload = _discover_terminal_payload(log_path)
    report_path = terminal_payload.get("report_path")
    notes_path = terminal_payload.get("notes_path")
    results_path = terminal_payload.get("results_path")
    if isinstance(report_path, str):
        return report_path, str(notes_path or ""), str(results_path or "")
    return None, None, None


def _run_supervised_session(args: argparse.Namespace) -> int:
    (
        collect_telemetry_sample,
        detect_threshold_breach,
        sleep_with_poll,
        write_telemetry_line,
    ) = _load_telemetry_helpers(status_reporter=_emit_cli_status)
    _, write_supervisor_session_artifacts = _load_reporting_helpers(
        status_reporter=_emit_cli_status,
    )
    started_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_root = Path(args.session_root).resolve() / started_at
    session_root.mkdir(parents=True, exist_ok=True)
    notes = [
        (
            f"Thresholds: RAM>={float(args.ram_threshold_percent):.1f}% or "
            f"GPU>={float(args.gpu_threshold_gibibytes):.2f} GiB."
        )
    ]
    records: list[SupervisedRunRecord] = []
    target_runs = int(args.max_runs) if int(args.max_runs) > 0 else sys.maxsize
    for run_index in range(1, target_runs + 1):
        run_root = session_root / f"run_{run_index:04d}"
        run_root.mkdir(parents=True, exist_ok=True)
        log_path = run_root / "process.log"
        telemetry_path = run_root / "telemetry.jsonl"
        started_monotonic = time.monotonic()
        started_utc = datetime.now(timezone.utc).isoformat()
        peak_ram = 0.0
        peak_gpu = 0.0
        kill_reason: str | None = None
        with log_path.open("w", encoding="utf-8") as log_handle:
            write_lock = threading.Lock()
            stream_state = {"last_line": "", "partial_line": ""}
            child_command = _build_supervised_child_command(args)
            _write_supervisor_output(
                text=(
                    f"[Supervisor] Starting run {run_index}/{target_runs} in {run_root}\n"
                    f"[Supervisor] Command: {' '.join(child_command)}\n"
                    f"[Supervisor] Logging child output to {log_path}\n"
                ),
                log_handle=log_handle,
                write_lock=write_lock,
            )
            process = subprocess.Popen(
                child_command,
                cwd=str(REPO_ROOT),
                env=_child_environment(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **detached_popen_kwargs(),
            )
            pump_thread = _stream_process_output(
                process=process,
                log_handle=log_handle,
                write_lock=write_lock,
                stream_state=stream_state,
            )
            while process.poll() is None:
                sample = collect_telemetry_sample()
                current_gpu = max(
                    (snapshot.used_gibibytes for snapshot in sample.gpu_memory),
                    default=0.0,
                )
                peak_ram = max(peak_ram, sample.system_memory.percent_used)
                peak_gpu = max(peak_gpu, current_gpu)
                write_telemetry_line(telemetry_path, sample)
                elapsed_seconds = time.monotonic() - started_monotonic
                last_line = _truncate_supervisor_preview(
                    stream_state.get("last_line", "")
                )
                last_line_suffix = f" last={last_line}" if last_line else ""
                _write_supervisor_output(
                    text=(
                        f"\n[Supervisor] heartbeat run={run_index} elapsed={elapsed_seconds:.1f}s "
                        f"ram={sample.system_memory.percent_used:.1f}% peak_ram={peak_ram:.1f}% "
                        f"gpu={current_gpu:.2f}GiB peak_gpu={peak_gpu:.2f}GiB"
                        f"{last_line_suffix}\n"
                    ),
                    log_handle=log_handle,
                    write_lock=write_lock,
                )
                kill_reason = detect_threshold_breach(
                    sample=sample,
                    ram_threshold_percent=float(args.ram_threshold_percent),
                    gpu_threshold_gibibytes=float(args.gpu_threshold_gibibytes),
                )
                if kill_reason is not None:
                    _write_supervisor_output(
                        text=f"[Supervisor] Threshold breach detected: {kill_reason}\n",
                        log_handle=log_handle,
                        write_lock=write_lock,
                    )
                    _terminate_process_tree(process)
                    break
                sleep_with_poll(float(args.sample_interval_seconds))
            exit_code = process.wait(timeout=5.0)
            if pump_thread is not None:
                pump_thread.join(timeout=5.0)
            _write_supervisor_output(
                text=(
                    f"\n[Supervisor] Run {run_index} finished with exit={exit_code} "
                    f"peak_ram={peak_ram:.1f}% peak_gpu={peak_gpu:.2f}GiB\n"
                ),
                log_handle=log_handle,
                write_lock=write_lock,
            )
        finished_utc = datetime.now(timezone.utc).isoformat()
        report_path, notes_path, results_path = _discover_report_paths(log_path)
        status = (
            "threshold_killed"
            if kill_reason
            else ("completed" if exit_code == 0 else "failed")
        )
        records.append(
            SupervisedRunRecord(
                run_index=run_index,
                status=status,
                exit_code=exit_code,
                started_at_utc=started_utc,
                finished_at_utc=finished_utc,
                duration_seconds=round(time.monotonic() - started_monotonic, 3),
                peak_system_memory_percent=round(peak_ram, 3),
                peak_gpu_memory_gibibytes=round(peak_gpu, 3),
                report_path=report_path,
                notes_path=notes_path,
                results_path=results_path,
                log_path=str(log_path),
                telemetry_path=str(telemetry_path),
                kill_reason=kill_reason,
            )
        )
        notes.append(
            (
                f"Run {run_index}: status={status} exit={exit_code} "
                f"peak_ram={peak_ram:.1f}% peak_gpu={peak_gpu:.2f} GiB"
            )
        )
        write_supervisor_session_artifacts(
            session_root=session_root,
            notes=notes,
            records=records,
        )
        if run_index < target_runs and float(args.between_runs_seconds) > 0:
            sleep_with_poll(float(args.between_runs_seconds))
    return 0 if all(record.status == "completed" for record in records) else 1


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    target_command = next(
        (value for value in raw_argv if not str(value).startswith("-")),
        None,
    )
    parser = _build_parser(
        target_command=target_command,
        status_reporter=_emit_cli_status,
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-dataset":
            _, prepare_sporc_benchmark = _load_dataset_helpers(
                status_reporter=_emit_cli_status,
            )
            prepared_root = _resolve_prepare_root(
                dataset_variant=str(args.dataset_variant),
                prepared_root=str(args.prepared_root or ""),
            )
            manifest = prepare_sporc_benchmark(
                prepared_root=prepared_root,
                dataset_variant=str(args.dataset_variant),
                episode_data_path=str(args.episode_data_path or ""),
                turn_data_path=str(args.turn_data_path or ""),
                reference_wpm=float(args.reference_wpm),
                force_download=bool(args.force_download),
                status_reporter=_emit_cli_status,
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "dataset_manifest": str(manifest.manifest_path),
                        "sample_count": manifest.sample_count,
                        "dataset_variant": manifest.dataset_variant,
                        "prepared_root": str(prepared_root),
                        "episode_data_path": str(manifest.episode_data_path),
                        "turn_data_path": str(manifest.turn_data_path),
                    }
                )
            )
            return 0
        if args.command == "curate-longform":
            curate_longform_sporc_manifest, _ = _load_dataset_helpers(
                status_reporter=_emit_cli_status,
            )
            manifest, audit_path = curate_longform_sporc_manifest(
                source_manifest_path=Path(args.source_manifest).resolve(),
                output_manifest_path=Path(args.output_manifest).resolve(),
                audit_path=Path(args.audit_path).resolve(),
                status_reporter=_emit_cli_status,
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "source_manifest": str(Path(args.source_manifest).resolve()),
                        "dataset_manifest": str(manifest.manifest_path),
                        "audit_path": str(audit_path),
                        "sample_count": manifest.sample_count,
                        "dataset_variant": manifest.dataset_variant,
                    }
                )
            )
            return 0
        if args.command == "dedupe-manifest":
            from card_framework.benchmark.summarizer_critic.sporc.dedupe_manifest import (
                run_dedupe_manifest_cli,
            )

            result = run_dedupe_manifest_cli(
                str(args.manifest_path),
                audit_path=str(args.audit_path),
                status_reporter=_emit_cli_status,
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "manifest_path": str(result.manifest_path),
                        "audit_path": str(result.audit_path),
                        "original_sample_count": result.original_sample_count,
                        "deduped_sample_count": result.deduped_sample_count,
                        "removed_sample_count": result.removed_sample_count,
                        "candidate_group_count": result.candidate_group_count,
                        "transcript_signature_reads": result.transcript_signature_reads,
                    }
                )
            )
            return 0
        if args.command == "export-transcripts":
            from card_framework.benchmark.summarizer_critic.sporc.export_transcripts import (
                run_export_transcripts_cli,
            )

            result = run_export_transcripts_cli(
                args,
                status_reporter=_emit_cli_status,
            )
            print(
                json.dumps(
                    {
                        "status": "completed",
                        "manifest_path": str(result.manifest_path),
                        "output_path": str(result.output_path),
                        "episode_positions": list(result.episode_positions),
                        "episode_count": result.episode_count,
                        "line_count": result.line_count,
                    }
                )
            )
            return 0
        if args.command == "human-sample":
            from card_framework.benchmark.summarizer_critic.sporc.human_sampler import (
                run_human_sampling_session,
            )

            return run_human_sampling_session(args)
        if args.command == "llm-sample":
            from card_framework.benchmark.summarizer_critic.sporc.llm_sampler import (
                run_llm_sampling_session,
            )

            return run_llm_sampling_session(args)
        if args.command == "execute":
            benchmark_execution_error, _, execute_benchmark = _load_runner_helpers(
                status_reporter=_emit_cli_status,
            )
            try:
                report = execute_benchmark(args)
            except benchmark_execution_error as exc:
                print(json.dumps({"status": "failed", "error": str(exc)}))
                return 1
            print(
                json.dumps(
                    {
                        "status": report.status,
                        "report_path": report.report_path,
                        "notes_path": report.notes_path,
                        "results_path": report.results_path,
                        "run_id": report.run_id,
                        "mean_composite": round(report.summary.mean_composite, 4),
                    }
                )
            )
            return 0 if report.status == "completed" else 1
        if args.command == "duration-sweep":
            try:
                return _run_duration_sweep(args)
            except _benchmark_execution_error_type() as exc:
                print(json.dumps({"status": "failed", "error": str(exc)}))
                return 1
        return _run_supervised_session(args)
    except KeyboardInterrupt:
        _emit_cli_status("Cancelled.")
        return 130
