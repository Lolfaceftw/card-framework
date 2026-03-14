"""Monitor benchmark runs with RAM and GPU telemetry."""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Literal

from card_framework.benchmark.artifacts import utc_now_iso
from card_framework.shared.paths import REPO_ROOT

_DEFAULT_COMMAND: tuple[str, ...] = (
    "uv",
    "run",
    "python",
    "-m",
    "card_framework.benchmark.run",
    "execute",
    "--preset",
    "smoke",
)
_DEFAULT_RESULTS_ROOT = Path("artifacts/benchmark_watchdog")
_DEFAULT_UV_CACHE_DIR = Path(".uv-cache")


@dataclass(slots=True, frozen=True)
class SystemMemorySnapshot:
    """Capture one point-in-time system RAM reading."""

    total_bytes: int
    available_bytes: int
    percent_used: float


@dataclass(slots=True, frozen=True)
class GpuMemorySnapshot:
    """Capture one point-in-time GPU VRAM reading."""

    gpu_index: int
    name: str
    total_mebibytes: int
    used_mebibytes: int

    @property
    def used_gibibytes(self) -> float:
        """Return used VRAM in GiB."""
        return self.used_mebibytes / 1024.0


@dataclass(slots=True, frozen=True)
class TelemetrySample:
    """Bundle one watchdog telemetry sample."""

    timestamp_utc: str
    system_memory: SystemMemorySnapshot
    gpu_memory: tuple[GpuMemorySnapshot, ...] = ()


@dataclass(slots=True, frozen=True)
class ThresholdBreach:
    """Describe one resource threshold breach."""

    kind: Literal["system_ram_percent", "gpu_vram_gibibytes"]
    message: str


@dataclass(slots=True, frozen=True)
class RunRecord:
    """Persist the outcome of one watchdog run."""

    run_index: int
    started_at_utc: str
    finished_at_utc: str
    duration_seconds: float
    command: list[str]
    status: str
    exit_code: int | None
    peak_system_memory_percent: float
    peak_gpu_memory_gibibytes: float
    report_paths: list[str]
    kill_reason: str = ""
    stdout_path: str = ""
    stderr_path: str = ""
    telemetry_path: str = ""
    report_summaries: list[str] = field(default_factory=list)


class _MemoryStatusEx(ctypes.Structure):
    """Windows ``GlobalMemoryStatusEx`` structure."""

    _fields_ = [
        ("dwLength", ctypes.c_uint32),
        ("dwMemoryLoad", ctypes.c_uint32),
        ("ullTotalPhys", ctypes.c_uint64),
        ("ullAvailPhys", ctypes.c_uint64),
        ("ullTotalPageFile", ctypes.c_uint64),
        ("ullAvailPageFile", ctypes.c_uint64),
        ("ullTotalVirtual", ctypes.c_uint64),
        ("ullAvailVirtual", ctypes.c_uint64),
        ("ullAvailExtendedVirtual", ctypes.c_uint64),
    ]


def parse_nvidia_smi_memory_csv(output: str) -> tuple[GpuMemorySnapshot, ...]:
    """Parse ``nvidia-smi`` CSV output into typed snapshots."""

    snapshots: list[GpuMemorySnapshot] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise ValueError(
                "Expected nvidia-smi output lines in the form "
                "'index, name, total_mb, used_mb'."
            )
        gpu_index_text, name, total_text, used_text = parts
        snapshots.append(
            GpuMemorySnapshot(
                gpu_index=int(gpu_index_text),
                name=name,
                total_mebibytes=int(total_text),
                used_mebibytes=int(used_text),
            )
        )
    return tuple(snapshots)


def detect_threshold_breaches(
    sample: TelemetrySample,
    *,
    ram_threshold_percent: float,
    gpu_threshold_gibibytes: float,
) -> tuple[ThresholdBreach, ...]:
    """Return threshold breaches for one telemetry sample."""

    breaches: list[ThresholdBreach] = []
    if sample.system_memory.percent_used >= ram_threshold_percent:
        breaches.append(
            ThresholdBreach(
                kind="system_ram_percent",
                message=(
                    f"System RAM reached {sample.system_memory.percent_used:.1f}% "
                    f"(threshold {ram_threshold_percent:.1f}%)."
                ),
            )
        )
    for snapshot in sample.gpu_memory:
        if snapshot.used_gibibytes >= gpu_threshold_gibibytes:
            breaches.append(
                ThresholdBreach(
                    kind="gpu_vram_gibibytes",
                    message=(
                        f"GPU {snapshot.gpu_index} '{snapshot.name}' reached "
                        f"{snapshot.used_gibibytes:.2f} GiB VRAM "
                        f"(threshold {gpu_threshold_gibibytes:.2f} GiB)."
                    ),
                )
            )
    return tuple(breaches)


def render_ascii_results_table(records: list[RunRecord]) -> str:
    """Render results as a markdown-compatible ASCII table."""

    headers = (
        "Run",
        "Status",
        "Exit",
        "Seconds",
        "Peak RAM %",
        "Peak GPU GiB",
        "Reports",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for record in records:
        report_value = ", ".join(record.report_paths) if record.report_paths else "-"
        lines.append(
            "| "
            + " | ".join(
                (
                    str(record.run_index),
                    record.status,
                    "-" if record.exit_code is None else str(record.exit_code),
                    f"{record.duration_seconds:.1f}",
                    f"{record.peak_system_memory_percent:.1f}",
                    f"{record.peak_gpu_memory_gibibytes:.2f}",
                    report_value,
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _resolve_command(raw_command: list[str]) -> tuple[str, ...]:
    """Resolve a command override or return the default smoke benchmark command."""

    if not raw_command:
        return _DEFAULT_COMMAND
    resolved = list(raw_command)
    if resolved and resolved[0] == "--":
        resolved = resolved[1:]
    return tuple(resolved) if resolved else _DEFAULT_COMMAND


def _summarize_report_payload(report_path: Path) -> str:
    """Return a short human-readable summary for one report JSON file."""

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    status = str(payload.get("status", "unknown"))
    aggregates = payload.get("aggregates")
    if isinstance(aggregates, dict):
        passed = aggregates.get("passed_samples")
        total = aggregates.get("total_samples")
        critic_pass_rate = aggregates.get("critic_pass_rate")
        return (
            f"{report_path.name}: status={status}, passed_samples={passed}/{total}, "
            f"critic_pass_rate={float(critic_pass_rate):.3f}"
        )
    score = payload.get("score")
    if isinstance(score, dict):
        score_out_of_100 = score.get("score_out_of_100")
        grounding = score.get("summary_grounding_pass_rate")
        return (
            f"{report_path.name}: status={status}, score={score_out_of_100}/100, "
            f"grounding={float(grounding):.3f}"
        )
    return f"{report_path.name}: status={status}"


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for the benchmark watchdog."""

    parser = argparse.ArgumentParser(
        description="Loop the real benchmark command under RAM and GPU telemetry.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum benchmark runs before exit. 0 means loop forever.",
    )
    parser.add_argument(
        "--sample-interval-seconds",
        type=float,
        default=5.0,
        help="Telemetry polling interval in seconds.",
    )
    parser.add_argument(
        "--ram-threshold-percent",
        type=float,
        default=80.0,
        help="Kill the benchmark when system RAM usage reaches this percentage.",
    )
    parser.add_argument(
        "--gpu-threshold-gibibytes",
        type=float,
        default=12.0,
        help="Kill the benchmark when any GPU reaches this VRAM usage in GiB.",
    )
    parser.add_argument(
        "--results-root",
        default=str(_DEFAULT_RESULTS_ROOT),
        help="Directory for watchdog artifacts.",
    )
    parser.add_argument(
        "--uv-cache-dir",
        default=str(_DEFAULT_UV_CACHE_DIR),
        help="Repo-local UV cache directory used for child runs.",
    )
    parser.add_argument(
        "--preset",
        default="smoke",
        choices=("smoke", "hourly", "full"),
        help="Preset used when no explicit command override is provided.",
    )
    parser.add_argument(
        "--manifest",
        default="",
        help="Optional manifest override for the benchmark command.",
    )
    parser.add_argument(
        "--providers",
        default="",
        help="Optional comma-separated provider override for the benchmark command.",
    )
    parser.add_argument(
        "--judge-provider",
        default="",
        help="Optional judge provider override for the benchmark command.",
    )
    parser.add_argument(
        "--judge-repeats",
        type=int,
        default=0,
        help="Optional judge repeat override for the benchmark command.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional sample cap override for the benchmark command.",
    )
    parser.add_argument(
        "--disable-order-swap",
        action="store_true",
        help="Disable order-swapped judge diagnostics in the benchmark command.",
    )
    parser.add_argument(
        "--notes-path",
        default="",
        help="Optional markdown notes file path. Defaults under results-root.",
    )
    parser.add_argument(
        "--results-path",
        default="",
        help="Optional markdown results file path. Defaults under results-root.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Optional explicit command override after --.",
    )
    return parser


def _default_command_for_args(args: argparse.Namespace) -> tuple[str, ...]:
    """Build the default benchmark command from parsed CLI arguments."""

    command = list(_DEFAULT_COMMAND[:-1]) + [str(args.preset)]
    if args.manifest:
        command.extend(["--manifest", str(args.manifest)])
    if args.providers:
        command.extend(["--providers", str(args.providers)])
    if args.judge_provider:
        command.extend(["--judge-provider", str(args.judge_provider)])
    if args.judge_repeats > 0:
        command.extend(["--judge-repeats", str(int(args.judge_repeats))])
    if args.max_samples > 0:
        command.extend(["--max-samples", str(int(args.max_samples))])
    if args.disable_order_swap:
        command.append("--disable-order-swap")
    return tuple(command)


def _resolve_results_paths(
    *,
    results_root: Path,
    notes_path: str,
    results_path: str,
) -> tuple[Path, Path]:
    """Resolve markdown output paths."""

    resolved_notes = (
        (REPO_ROOT / notes_path).resolve()
        if notes_path
        else (results_root / "notes.md").resolve()
    )
    resolved_results = (
        (REPO_ROOT / results_path).resolve()
        if results_path
        else (results_root / "results.md").resolve()
    )
    return resolved_notes, resolved_results


def _system_memory_snapshot() -> SystemMemorySnapshot:
    """Collect current system RAM metrics."""

    if os.name == "nt":
        memory_status = _MemoryStatusEx()
        memory_status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            raise RuntimeError("GlobalMemoryStatusEx failed.")
        return SystemMemorySnapshot(
            total_bytes=int(memory_status.ullTotalPhys),
            available_bytes=int(memory_status.ullAvailPhys),
            percent_used=float(memory_status.dwMemoryLoad),
        )

    total_pages = os.sysconf("SC_PHYS_PAGES")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    total_bytes = int(total_pages * page_size)
    available_bytes = int(available_pages * page_size)
    percent_used = 0.0
    if total_bytes > 0:
        percent_used = ((total_bytes - available_bytes) / total_bytes) * 100.0
    return SystemMemorySnapshot(
        total_bytes=total_bytes,
        available_bytes=available_bytes,
        percent_used=percent_used,
    )


def _gpu_memory_snapshots() -> tuple[GpuMemorySnapshot, ...]:
    """Collect current GPU VRAM metrics when ``nvidia-smi`` is available."""

    if shutil.which("nvidia-smi") is None:
        return ()
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=5.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ()
    if completed.returncode != 0:
        return ()
    try:
        return parse_nvidia_smi_memory_csv(completed.stdout)
    except ValueError:
        return ()


def _telemetry_sample() -> TelemetrySample:
    """Capture one telemetry sample."""

    return TelemetrySample(
        timestamp_utc=utc_now_iso(),
        system_memory=_system_memory_snapshot(),
        gpu_memory=_gpu_memory_snapshots(),
    )


def _kill_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a benchmark subprocess and its children."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        return
    os.killpg(os.getpgid(process.pid), signal.SIGKILL)


def _extract_summary_payload(
    stdout_path: Path,
    stderr_path: Path,
) -> dict[str, object] | None:
    """Return the last JSON object emitted by the child command when possible."""

    for candidate_path in (stdout_path, stderr_path):
        if not candidate_path.exists():
            continue
        text = candidate_path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _report_paths_from_payload(payload: dict[str, object] | None) -> list[str]:
    """Extract existing report paths from a child summary payload."""

    if payload is None:
        return []
    paths: list[str] = []
    for key in ("report_path", "quality_report_path", "verification_path"):
        raw_value = payload.get(key)
        if not isinstance(raw_value, str) or not raw_value.strip():
            continue
        candidate = Path(raw_value)
        if candidate.exists():
            paths.append(str(candidate))
    return paths


def _report_summaries(report_paths: list[str]) -> list[str]:
    """Build short summaries for emitted reports."""

    summaries: list[str] = []
    for raw_path in report_paths:
        candidate = Path(raw_path)
        if not candidate.exists():
            continue
        try:
            summaries.append(_summarize_report_payload(candidate))
        except (json.JSONDecodeError, OSError, ValueError, TypeError):
            summaries.append(f"{candidate.name}: unable to summarize")
    return summaries


def _run_one(
    *,
    run_index: int,
    command: tuple[str, ...],
    results_root: Path,
    uv_cache_dir: Path,
    sample_interval_seconds: float,
    ram_threshold_percent: float,
    gpu_threshold_gibibytes: float,
) -> RunRecord:
    """Run one monitored benchmark iteration."""

    started_at = utc_now_iso()
    started_monotonic = time.monotonic()
    run_dir = results_root / f"run_{run_index:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    telemetry_path = run_dir / "telemetry.jsonl"

    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(uv_cache_dir)
    env.setdefault("UV_OFFLINE", "1")
    env.setdefault("HF_HOME", str((results_root / "hf_home").resolve()))
    env.setdefault(
        "HF_HUB_CACHE",
        str((results_root / "hf_home" / "hub").resolve()),
    )
    env.setdefault(
        "TRANSFORMERS_CACHE",
        str((results_root / "hf_home" / "transformers").resolve()),
    )
    env.setdefault("PYTHONUNBUFFERED", "1")

    if os.name == "nt":
        popen_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        popen_kwargs = {"start_new_session": True}

    peak_ram = 0.0
    peak_gpu_gib = 0.0
    exit_code: int | None = None
    status = "completed"
    kill_reason = ""

    with (
        stdout_path.open("w", encoding="utf-8") as stdout_handle,
        stderr_path.open("w", encoding="utf-8") as stderr_handle,
        telemetry_path.open("w", encoding="utf-8") as telemetry_handle,
    ):
        process = subprocess.Popen(
            list(command),
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
        while process.poll() is None:
            sample = _telemetry_sample()
            peak_ram = max(peak_ram, sample.system_memory.percent_used)
            peak_gpu_gib = max(
                peak_gpu_gib,
                max((snapshot.used_gibibytes for snapshot in sample.gpu_memory), default=0.0),
            )
            telemetry_handle.write(json.dumps(asdict(sample)) + "\n")
            telemetry_handle.flush()

            breaches = detect_threshold_breaches(
                sample,
                ram_threshold_percent=ram_threshold_percent,
                gpu_threshold_gibibytes=gpu_threshold_gibibytes,
            )
            if breaches:
                status = "threshold_killed"
                kill_reason = breaches[0].kind
                _kill_process_tree(process)
                break
            time.sleep(sample_interval_seconds)

        if status != "threshold_killed":
            exit_code = process.wait()
            status = "completed" if exit_code == 0 else "failed"

    finished_at = utc_now_iso()
    duration_seconds = round(time.monotonic() - started_monotonic, 3)
    summary_payload = _extract_summary_payload(stdout_path, stderr_path)
    report_paths = _report_paths_from_payload(summary_payload)
    return RunRecord(
        run_index=run_index,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        duration_seconds=duration_seconds,
        command=list(command),
        status=status,
        exit_code=exit_code,
        peak_system_memory_percent=round(peak_ram, 3),
        peak_gpu_memory_gibibytes=round(peak_gpu_gib, 3),
        report_paths=report_paths,
        kill_reason=kill_reason,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        telemetry_path=str(telemetry_path),
        report_summaries=_report_summaries(report_paths),
    )


def _append_markdown_note(notes_path: Path, record: RunRecord) -> None:
    """Append one markdown note entry for a finished run."""

    notes_path.parent.mkdir(parents=True, exist_ok=True)
    report_summary_lines = (
        "\n".join(f"- {summary}" for summary in record.report_summaries)
        if record.report_summaries
        else "- No report payloads were available."
    )
    entry = (
        f"\n## Run {record.run_index}\n"
        f"- Started: `{record.started_at_utc}`\n"
        f"- Finished: `{record.finished_at_utc}`\n"
        f"- Status: `{record.status}`\n"
        f"- Exit code: `{record.exit_code}`\n"
        f"- Peak RAM: `{record.peak_system_memory_percent:.1f}%`\n"
        f"- Peak GPU: `{record.peak_gpu_memory_gibibytes:.2f} GiB`\n"
        f"- Kill reason: `{record.kill_reason or 'none'}`\n"
        f"- Command: `{' '.join(record.command)}`\n"
        f"- Stdout: `{record.stdout_path}`\n"
        f"- Stderr: `{record.stderr_path}`\n"
        f"- Telemetry: `{record.telemetry_path}`\n"
        f"- Report summaries:\n{report_summary_lines}\n"
    )
    if not notes_path.exists():
        notes_path.write_text("# Benchmark Watchdog Notes\n", encoding="utf-8")
    with notes_path.open("a", encoding="utf-8") as handle:
        handle.write(entry)


def _write_results_markdown(results_path: Path, records: list[RunRecord]) -> None:
    """Rewrite the markdown results summary."""

    results_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Watchdog Results",
        "",
        "```text",
        render_ascii_results_table(records),
        "```",
        "",
    ]
    for record in records:
        lines.append(
            f"- Run {record.run_index}: status={record.status}, exit={record.exit_code}, "
            f"peak_ram={record.peak_system_memory_percent:.1f}%, "
            f"peak_gpu={record.peak_gpu_memory_gibibytes:.2f} GiB, "
            f"kill_reason={record.kill_reason or 'none'}"
        )
    results_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute_watchdog(args: argparse.Namespace) -> int:
    """Execute the monitored benchmark loop."""

    results_root = (REPO_ROOT / args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    uv_cache_dir = (REPO_ROOT / args.uv_cache_dir).resolve()
    uv_cache_dir.mkdir(parents=True, exist_ok=True)
    notes_path, results_path = _resolve_results_paths(
        results_root=results_root,
        notes_path=str(args.notes_path),
        results_path=str(args.results_path),
    )

    command = _resolve_command(list(args.command))
    if not args.command:
        command = _default_command_for_args(args)

    records: list[RunRecord] = []
    run_index = 1
    while args.max_runs <= 0 or run_index <= args.max_runs:
        record = _run_one(
            run_index=run_index,
            command=command,
            results_root=results_root,
            uv_cache_dir=uv_cache_dir,
            sample_interval_seconds=float(args.sample_interval_seconds),
            ram_threshold_percent=float(args.ram_threshold_percent),
            gpu_threshold_gibibytes=float(args.gpu_threshold_gibibytes),
        )
        records.append(record)
        _append_markdown_note(notes_path, record)
        _write_results_markdown(results_path, records)
        run_index += 1

    print(
        json.dumps(
            {
                "status": "completed",
                "runs": len(records),
                "notes_path": str(notes_path),
                "results_path": str(results_path),
                "results_root": str(results_root),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Program entrypoint."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    return execute_watchdog(args)


if __name__ == "__main__":
    raise SystemExit(main())
