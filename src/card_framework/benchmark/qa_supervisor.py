"""Supervise repeated QA benchmark runs with resource guardrails."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from card_framework.audio_pipeline.gpu_heartbeat import GpuDeviceSnapshot
from card_framework.shared.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PROVIDER_PROFILES_PATH,
    DEFAULT_QA_CONFIG_PATH,
    REPO_ROOT,
)

_NVIDIA_GPU_MEMORY_COMMAND: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used",
    "--format=csv,noheader,nounits",
)


class QASupervisorRuntimeError(RuntimeError):
    """Raise when the QA supervisor cannot proceed."""


@dataclass(slots=True, frozen=True)
class SystemMemorySnapshot:
    """Describe one host-memory telemetry sample."""

    total_bytes: int
    available_bytes: int
    used_bytes: int
    used_percent: float


@dataclass(slots=True, frozen=True)
class TelemetrySample:
    """Capture one combined host and GPU telemetry sample."""

    sampled_at_utc: str
    system_memory: SystemMemorySnapshot
    gpu_devices: tuple[GpuDeviceSnapshot, ...]


@dataclass(slots=True, frozen=True)
class BenchmarkRunOutcome:
    """Summarize one supervised child process run."""

    run_label: str
    command: list[str]
    status: str
    exit_code: int | None
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    peak_memory_percent: float
    peak_memory_used_gib: float
    peak_gpu_used_mib: int
    threshold_reason: str | None
    benchmark_run_id: str | None
    benchmark_score: int | None
    report_path: str | None
    log_path: str
    telemetry_path: str
    error_excerpt: str | None


class _MemoryStatusEx(ctypes.Structure):
    """Mirror the Windows MEMORYSTATUSEX structure."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _utc_now() -> str:
    """Return the current UTC timestamp as ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def _child_process_environment() -> dict[str, str]:
    """Build an environment that preserves repo-local source imports."""
    environment = dict(os.environ)
    source_root = REPO_ROOT / "src"
    if not source_root.exists():
        return environment

    source_root_text = str(source_root)
    existing_pythonpath = environment.get("PYTHONPATH", "")
    existing_entries = [entry for entry in existing_pythonpath.split(os.pathsep) if entry]
    if source_root_text not in existing_entries:
        environment["PYTHONPATH"] = os.pathsep.join([source_root_text, *existing_entries])
    return environment


def collect_system_memory_snapshot() -> SystemMemorySnapshot:
    """Collect total host memory usage without third-party dependencies."""
    if sys.platform == "win32":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise QASupervisorRuntimeError(
                "GlobalMemoryStatusEx failed while collecting host memory telemetry."
            )
        total_bytes = int(status.ullTotalPhys)
        available_bytes = int(status.ullAvailPhys)
        used_bytes = total_bytes - available_bytes
        return SystemMemorySnapshot(
            total_bytes=total_bytes,
            available_bytes=available_bytes,
            used_bytes=used_bytes,
            used_percent=float(status.dwMemoryLoad),
        )

    if hasattr(os, "sysconf"):
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        total_bytes = page_size * total_pages
        available_bytes = page_size * available_pages
        used_bytes = total_bytes - available_bytes
        used_percent = (
            (used_bytes / float(total_bytes)) * 100.0 if total_bytes else 0.0
        )
        return SystemMemorySnapshot(
            total_bytes=total_bytes,
            available_bytes=available_bytes,
            used_bytes=used_bytes,
            used_percent=used_percent,
        )

    raise QASupervisorRuntimeError(
        f"Unsupported platform for host memory telemetry: {sys.platform}"
    )


def collect_nvidia_gpu_memory_snapshots(
    *,
    command_timeout_seconds: float,
) -> tuple[GpuDeviceSnapshot, ...]:
    """Collect dedicated-memory snapshots for visible NVIDIA GPUs."""
    try:
        result = subprocess.run(
            list(_NVIDIA_GPU_MEMORY_COMMAND),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=command_timeout_seconds,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ()

    devices: list[GpuDeviceSnapshot] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",", maxsplit=3)]
        if len(parts) != 4:
            continue
        try:
            devices.append(
                GpuDeviceSnapshot(
                    gpu_index=int(parts[0]),
                    name=parts[1],
                    dedicated_total_mb=int(parts[2]),
                    dedicated_used_mb=int(parts[3]),
                )
            )
        except ValueError:
            continue
    return tuple(devices)


def collect_telemetry_sample(*, gpu_timeout_seconds: float) -> TelemetrySample:
    """Collect one telemetry sample for RAM and GPU VRAM."""
    return TelemetrySample(
        sampled_at_utc=_utc_now(),
        system_memory=collect_system_memory_snapshot(),
        gpu_devices=collect_nvidia_gpu_memory_snapshots(
            command_timeout_seconds=gpu_timeout_seconds
        ),
    )


def _sample_exceeds_threshold(
    *,
    sample: TelemetrySample,
    ram_threshold_percent: float,
    gpu_threshold_mib: int,
) -> str | None:
    """Return a human-readable threshold breach reason when any limit is exceeded."""
    if sample.system_memory.used_percent >= ram_threshold_percent:
        return (
            f"Host RAM reached {sample.system_memory.used_percent:.1f}% "
            f"(threshold {ram_threshold_percent:.1f}%)."
        )

    for device in sample.gpu_devices:
        if device.dedicated_used_mb >= gpu_threshold_mib:
            return (
                f"GPU {device.gpu_index} VRAM reached {device.dedicated_used_mb} MiB "
                f"({device.name}); threshold {gpu_threshold_mib} MiB."
            )
    return None


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a child process tree aggressively."""
    if process.poll() is not None:
        return

    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        return

    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()


def _read_log_excerpt(log_path: Path, *, max_lines: int = 20) -> str | None:
    """Return trailing lines from one run log file."""
    if not log_path.exists():
        return None
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return None
    excerpt = "\n".join(lines[-max_lines:]).strip()
    return excerpt or None


def _discover_latest_report(
    output_root: Path,
) -> tuple[str | None, int | None, Path | None]:
    """Return the latest QA report run id, score, and path when present."""
    report_paths = sorted(output_root.glob("*/qa_report.json"))
    if not report_paths:
        return None, None, None
    report_path = max(report_paths, key=lambda path: path.stat().st_mtime)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id", "")).strip() or None
    raw_score = payload.get("score", {}).get("score_out_of_100")
    score = int(raw_score) if isinstance(raw_score, (int, float)) else None
    return run_id, score, report_path


def _render_ascii_table(outcomes: list[BenchmarkRunOutcome]) -> str:
    """Render run outcomes as a fixed-width ASCII table."""
    headers = [
        "run",
        "status",
        "exit",
        "secs",
        "peak_ram%",
        "peak_gpu_mib",
        "score",
        "reason",
    ]
    rows = [
        [
            outcome.run_label,
            outcome.status,
            "-" if outcome.exit_code is None else str(outcome.exit_code),
            f"{outcome.duration_seconds:.1f}",
            f"{outcome.peak_memory_percent:.1f}",
            str(outcome.peak_gpu_used_mib),
            "-" if outcome.benchmark_score is None else str(outcome.benchmark_score),
            (outcome.threshold_reason or "")[:48],
        ]
        for outcome in outcomes
    ]
    widths = []
    for index, header in enumerate(headers):
        candidate_widths = [len(header)]
        candidate_widths.extend(len(row[index]) for row in rows)
        widths.append(max(candidate_widths))

    def _format_row(values: list[str]) -> str:
        return "| " + " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(values)
        ) + " |"

    divider = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    parts = [divider, _format_row(headers), divider]
    parts.extend(_format_row(row) for row in rows)
    parts.append(divider)
    return "\n".join(parts)


def _write_session_artifacts(
    *,
    session_root: Path,
    outcomes: list[BenchmarkRunOutcome],
    session_started_at_utc: str,
    notes: list[str],
) -> None:
    """Write structured session artifacts after each supervised run."""
    payload = {
        "session_started_at_utc": session_started_at_utc,
        "updated_at_utc": _utc_now(),
        "runs": [asdict(outcome) for outcome in outcomes],
        "notes": list(notes),
    }
    (session_root / "session.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    table = _render_ascii_table(outcomes)
    results_lines = [
        "# QA Supervisor Results",
        "",
        f"Session start: `{session_started_at_utc}`",
        f"Updated: `{payload['updated_at_utc']}`",
        "",
        "```text",
        table,
        "```",
        "",
        "## Run Artifacts",
    ]
    for outcome in outcomes:
        results_lines.append(
            f"- `{outcome.run_label}`: status=`{outcome.status}` log=`{outcome.log_path}` telemetry=`{outcome.telemetry_path}`"
        )
        if outcome.report_path:
            results_lines.append(
                f"- `{outcome.run_label}` report=`{outcome.report_path}` score=`{outcome.benchmark_score}`"
            )
    (session_root / "results.md").write_text(
        "\n".join(results_lines) + "\n",
        encoding="utf-8",
    )

    note_lines = ["# QA Supervisor Notes", ""]
    for note in notes:
        note_lines.append(f"- {note}")
    (session_root / "notes.md").write_text(
        "\n".join(note_lines) + "\n",
        encoding="utf-8",
    )


def _supervise_command(
    *,
    run_label: str,
    command: list[str],
    run_root: Path,
    telemetry_interval_seconds: float,
    ram_threshold_percent: float,
    gpu_threshold_mib: int,
    gpu_timeout_seconds: float,
) -> tuple[BenchmarkRunOutcome, list[str]]:
    """Execute one child command under RAM and GPU telemetry supervision."""
    log_path = run_root / "process.log"
    telemetry_path = run_root / "telemetry.jsonl"
    run_root.mkdir(parents=True, exist_ok=True)

    started_at = _utc_now()
    started_monotonic = time.monotonic()
    peak_memory_percent = 0.0
    peak_memory_used_bytes = 0
    peak_gpu_used_mib = 0
    threshold_reason: str | None = None

    with log_path.open("w", encoding="utf-8", newline="\n") as log_handle, telemetry_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as telemetry_handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=_child_process_environment(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        while True:
            if process.poll() is not None:
                break

            sample = collect_telemetry_sample(gpu_timeout_seconds=gpu_timeout_seconds)
            peak_memory_percent = max(
                peak_memory_percent,
                sample.system_memory.used_percent,
            )
            peak_memory_used_bytes = max(
                peak_memory_used_bytes,
                sample.system_memory.used_bytes,
            )
            if sample.gpu_devices:
                peak_gpu_used_mib = max(
                    peak_gpu_used_mib,
                    max(device.dedicated_used_mb for device in sample.gpu_devices),
                )
            telemetry_handle.write(
                json.dumps(
                    {
                        "sampled_at_utc": sample.sampled_at_utc,
                        "system_memory": asdict(sample.system_memory),
                        "gpu_devices": [asdict(device) for device in sample.gpu_devices],
                    }
                )
                + "\n"
            )
            telemetry_handle.flush()

            threshold_reason = _sample_exceeds_threshold(
                sample=sample,
                ram_threshold_percent=ram_threshold_percent,
                gpu_threshold_mib=gpu_threshold_mib,
            )
            if threshold_reason is not None:
                _terminate_process_tree(process)
                break
            time.sleep(telemetry_interval_seconds)

        try:
            exit_code = process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            _terminate_process_tree(process)
            exit_code = process.wait(timeout=5.0)

    finished_at = _utc_now()
    duration_seconds = time.monotonic() - started_monotonic
    benchmark_run_id, benchmark_score, report_path = _discover_latest_report(
        run_root / "qa_runs"
    )
    error_excerpt = _read_log_excerpt(log_path)
    status = "completed"
    if threshold_reason is not None:
        status = "threshold_killed"
    elif exit_code != 0:
        status = "failed"

    outcome = BenchmarkRunOutcome(
        run_label=run_label,
        command=command,
        status=status,
        exit_code=exit_code,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_seconds=duration_seconds,
        peak_memory_percent=peak_memory_percent,
        peak_memory_used_gib=peak_memory_used_bytes / float(1024**3),
        peak_gpu_used_mib=peak_gpu_used_mib,
        threshold_reason=threshold_reason,
        benchmark_run_id=benchmark_run_id,
        benchmark_score=benchmark_score,
        report_path=str(report_path) if report_path else None,
        log_path=str(log_path),
        telemetry_path=str(telemetry_path),
        error_excerpt=error_excerpt,
    )
    notes = [
        (
            f"{run_label}: status={outcome.status} exit={outcome.exit_code} "
            f"peak_ram={outcome.peak_memory_percent:.1f}% "
            f"peak_gpu={outcome.peak_gpu_used_mib} MiB "
            f"duration={outcome.duration_seconds:.1f}s"
        )
    ]
    if threshold_reason:
        notes.append(f"{run_label}: threshold breach -> {threshold_reason}")
    if benchmark_run_id:
        notes.append(
            f"{run_label}: benchmark report {benchmark_run_id} score={benchmark_score}"
        )
    if error_excerpt:
        notes.append(f"{run_label}: log tail\n\n```text\n{error_excerpt}\n```")
    return outcome, notes


def _preflight_qa_benchmark_import() -> str | None:
    """Return an import failure detail for the QA benchmark child process."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", "import card_framework.benchmark.qa"],
            cwd=str(REPO_ROOT),
            env=_child_process_environment(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"QA benchmark import preflight could not run: {exc}"

    if completed.returncode == 0:
        return None

    detail = (completed.stderr or completed.stdout).strip()
    if not detail:
        detail = f"import exited with code {completed.returncode}"
    return detail


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for supervised QA benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Supervise repeated QA benchmark runs with RAM/GPU guardrails."
    )
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
        "--qa-config",
        default=str(DEFAULT_QA_CONFIG_PATH),
        help="QA benchmark-specific config path",
    )
    parser.add_argument(
        "--provider",
        default="vllm_default",
        help="Default provider profile id for both QA agents",
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
        "--session-output-dir",
        default="artifacts/qa_supervisor",
        help="Output root for supervisor notes, logs, telemetry, and result summaries",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=1,
        help="Number of benchmark runs to execute after the smoke test; 0 means unbounded.",
    )
    parser.add_argument(
        "--between-runs-seconds",
        type=float,
        default=3.0,
        help="Pause between benchmark runs.",
    )
    parser.add_argument(
        "--telemetry-interval-seconds",
        type=float,
        default=5.0,
        help="Telemetry sampling interval for the child process.",
    )
    parser.add_argument(
        "--gpu-timeout-seconds",
        type=float,
        default=3.0,
        help="Per-sample nvidia-smi timeout.",
    )
    parser.add_argument(
        "--ram-threshold-percent",
        type=float,
        default=80.0,
        help="Terminate the child when host RAM usage reaches this percent.",
    )
    parser.add_argument(
        "--gpu-threshold-mib",
        type=int,
        default=12288,
        help="Terminate the child when any GPU dedicated VRAM usage reaches this MiB threshold.",
    )
    parser.add_argument(
        "--smoke-test-seconds",
        type=float,
        default=8.0,
        help="Duration for the preflight telemetry smoke test.",
    )
    return parser


def execute_command(args: argparse.Namespace) -> int:
    """Execute the smoke test and the requested supervised QA benchmark runs."""
    if args.telemetry_interval_seconds <= 0:
        raise QASupervisorRuntimeError("--telemetry-interval-seconds must be > 0")
    if args.gpu_timeout_seconds <= 0:
        raise QASupervisorRuntimeError("--gpu-timeout-seconds must be > 0")
    if args.between_runs_seconds < 0:
        raise QASupervisorRuntimeError("--between-runs-seconds must be >= 0")
    if args.smoke_test_seconds <= 0:
        raise QASupervisorRuntimeError("--smoke-test-seconds must be > 0")
    if not 0 < args.ram_threshold_percent <= 100:
        raise QASupervisorRuntimeError("--ram-threshold-percent must be within (0, 100]")
    if args.gpu_threshold_mib <= 0:
        raise QASupervisorRuntimeError("--gpu-threshold-mib must be > 0")
    if args.max_runs < 0:
        raise QASupervisorRuntimeError("--max-runs must be >= 0")

    session_started_at = _utc_now()
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_root = (REPO_ROOT / args.session_output_dir / session_id).resolve()
    session_root.mkdir(parents=True, exist_ok=True)

    summary_xml_path = Path(args.summary_xml).resolve()
    source_transcript_path = Path(args.source_transcript).resolve()
    if not summary_xml_path.exists():
        raise QASupervisorRuntimeError(
            f"summary xml path does not exist: {summary_xml_path}"
        )
    if not source_transcript_path.exists():
        raise QASupervisorRuntimeError(
            f"source transcript path does not exist: {source_transcript_path}"
        )

    outcomes: list[BenchmarkRunOutcome] = []
    notes: list[str] = [
        (
            "Session started with thresholds "
            f"RAM>={args.ram_threshold_percent:.1f}% or "
            f"GPU>={args.gpu_threshold_mib} MiB."
        ),
        (
            "QA inputs: "
            f"summary_xml={summary_xml_path} "
            f"source_transcript={source_transcript_path}"
        ),
    ]

    smoke_command = [
        sys.executable,
        "-c",
        f"import time; time.sleep({args.smoke_test_seconds:.3f})",
    ]
    smoke_outcome, smoke_notes = _supervise_command(
        run_label="smoke_test",
        command=smoke_command,
        run_root=session_root / "smoke_test",
        telemetry_interval_seconds=args.telemetry_interval_seconds,
        ram_threshold_percent=args.ram_threshold_percent,
        gpu_threshold_mib=args.gpu_threshold_mib,
        gpu_timeout_seconds=args.gpu_timeout_seconds,
    )
    outcomes.append(smoke_outcome)
    notes.extend(smoke_notes)
    _write_session_artifacts(
        session_root=session_root,
        outcomes=outcomes,
        session_started_at_utc=session_started_at,
        notes=notes,
    )
    if smoke_outcome.status != "completed":
        return 1

    preflight_error = _preflight_qa_benchmark_import()
    if preflight_error is not None:
        notes.append(f"QA benchmark import preflight failed: {preflight_error}")
        _write_session_artifacts(
            session_root=session_root,
            outcomes=outcomes,
            session_started_at_utc=session_started_at,
            notes=notes,
        )
        return 1

    target_runs = args.max_runs if args.max_runs != 0 else sys.maxsize
    for run_index in range(1, target_runs + 1):
        run_label = f"benchmark_run_{run_index:03d}"
        benchmark_command = [
            sys.executable,
            "-m",
            "card_framework.benchmark.qa",
            "--summary-xml",
            str(summary_xml_path),
            "--source-transcript",
            str(source_transcript_path),
            "--provider-profiles",
            str(Path(args.provider_profiles).resolve()),
            "--config",
            str(Path(args.config).resolve()),
            "--qa-config",
            str(Path(args.qa_config).resolve()),
            "--provider",
            str(args.provider),
            "--output-dir",
            str((session_root / run_label / "qa_runs").resolve()),
            "--output-format",
            "json",
            "--color",
            "never",
        ]
        if str(args.creator_provider).strip():
            benchmark_command.extend(["--creator-provider", str(args.creator_provider)])
        if str(args.evaluator_provider).strip():
            benchmark_command.extend(["--evaluator-provider", str(args.evaluator_provider)])

        outcome, run_notes = _supervise_command(
            run_label=run_label,
            command=benchmark_command,
            run_root=session_root / run_label,
            telemetry_interval_seconds=args.telemetry_interval_seconds,
            ram_threshold_percent=args.ram_threshold_percent,
            gpu_threshold_mib=args.gpu_threshold_mib,
            gpu_timeout_seconds=args.gpu_timeout_seconds,
        )
        outcomes.append(outcome)
        notes.extend(run_notes)
        _write_session_artifacts(
            session_root=session_root,
            outcomes=outcomes,
            session_started_at_utc=session_started_at,
            notes=notes,
        )
        if run_index < target_runs and args.between_runs_seconds > 0:
            time.sleep(args.between_runs_seconds)

    return 0 if all(outcome.status == "completed" for outcome in outcomes) else 1


def main(argv: list[str] | None = None) -> int:
    """Program entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return execute_command(args)
    except QASupervisorRuntimeError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
