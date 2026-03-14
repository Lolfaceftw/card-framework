"""CLI entrypoints for the summarizer-critic benchmark package."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import TextIO

from card_framework.benchmark.summarizer_critic.config import (
    DEFAULT_GPU_THRESHOLD_GIB,
    DEFAULT_PREPARED_ROOT,
    DEFAULT_QMSUM_REPO_DIR,
    DEFAULT_RAM_THRESHOLD_PERCENT,
    DEFAULT_REFERENCE_WPM,
    DEFAULT_SUPERVISOR_ROOT,
    SupervisedRunRecord,
)
from card_framework.benchmark.summarizer_critic.dataset import (
    prepare_official_qmsum_benchmark,
)
from card_framework.benchmark.summarizer_critic.reporting import (
    write_supervisor_session_artifacts,
)
from card_framework.benchmark.summarizer_critic.runner import (
    BenchmarkExecutionError,
    build_execute_parser,
    execute_benchmark,
)
from card_framework.benchmark.summarizer_critic.telemetry import (
    collect_telemetry_sample,
    detect_threshold_breach,
    sleep_with_poll,
    write_telemetry_line,
)
from card_framework.shared.paths import REPO_ROOT


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Duration-first real-dataset benchmark for the summarizer-critic framework."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-dataset", help="Fetch and prepare official QMSum data"
    )
    prepare.add_argument("--qmsum-repo-dir", default=str(DEFAULT_QMSUM_REPO_DIR))
    prepare.add_argument("--prepared-root", default=str(DEFAULT_PREPARED_ROOT))
    prepare.add_argument("--reference-wpm", type=float, default=DEFAULT_REFERENCE_WPM)
    prepare.add_argument("--force-download", action="store_true")

    execute = subparsers.add_parser(
        "execute", help="Run the real summarizer-critic benchmark"
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
    return parser


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
    """Compress a child-log line into a readable heartbeat preview."""
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
        target=_pump, name=f"sc-supervisor-{process.pid}", daemon=True
    )
    thread.start()
    return thread


def _build_supervised_child_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "card_framework.benchmark.summarizer_critic",
        "execute",
    ]
    forwarded = {
        "--dataset-manifest": args.dataset_manifest,
        "--qmsum-repo-dir": args.qmsum_repo_dir,
        "--reference-wpm": args.reference_wpm,
        "--output-root": args.output_root,
        "--provider-profiles": args.provider_profiles,
        "--config": args.config,
        "--summarizer-provider": args.summarizer_provider,
        "--critic-provider": args.critic_provider,
        "--judge-provider": args.judge_provider,
        "--embedding-mode": args.embedding_mode,
        "--judge-rubric": args.judge_rubric,
        "--judge-repeats": args.judge_repeats,
        "--alignscore-model": args.alignscore_model,
        "--alignscore-source-chunk-words": args.alignscore_source_chunk_words,
        "--judge-source-char-limit": args.judge_source_char_limit,
        "--max-samples": args.max_samples,
        "--max-iterations": args.max_iterations,
        "--summarizer-timeout-seconds": args.summarizer_timeout_seconds,
        "--critic-timeout-seconds": args.critic_timeout_seconds,
        "--retrieval-timeout-seconds": args.retrieval_timeout_seconds,
        "--index-timeout-seconds": args.index_timeout_seconds,
    }
    if args.prepare_if_missing:
        command.append("--prepare-if-missing")
    if args.force_download_dataset:
        command.append("--force-download-dataset")
    if args.disable_order_swap:
        command.append("--disable-order-swap")
    for flag, value in forwarded.items():
        if value in (None, ""):
            continue
        command.extend([flag, str(value)])
    return command


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)
        return
    process.kill()


def _discover_report_paths(log_path: Path) -> tuple[str | None, str | None, str | None]:
    if not log_path.exists():
        return None, None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in reversed([item.strip() for item in text.splitlines() if item.strip()]):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        report_path = payload.get("report_path")
        notes_path = payload.get("notes_path")
        results_path = payload.get("results_path")
        if isinstance(report_path, str):
            return report_path, str(notes_path or ""), str(results_path or "")
    return None, None, None


def _run_supervised_session(args: argparse.Namespace) -> int:
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
                peak_gpu = max(
                    peak_gpu,
                    current_gpu,
                )
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
            session_root=session_root, notes=notes, records=records
        )
        if run_index < target_runs and float(args.between_runs_seconds) > 0:
            sleep_with_poll(float(args.between_runs_seconds))
    return 0 if all(record.status == "completed" for record in records) else 1


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-dataset":
        manifest = prepare_official_qmsum_benchmark(
            qmsum_repo_dir=Path(args.qmsum_repo_dir).resolve(),
            prepared_root=Path(args.prepared_root).resolve(),
            reference_wpm=float(args.reference_wpm),
            force_download=bool(args.force_download),
        )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "dataset_manifest": str(manifest.manifest_path),
                    "sample_count": manifest.sample_count,
                    "qmsum_repo_dir": str(manifest.qmsum_repo_dir),
                }
            )
        )
        return 0
    if args.command == "execute":
        try:
            report = execute_benchmark(args)
        except BenchmarkExecutionError as exc:
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
    return _run_supervised_session(args)
