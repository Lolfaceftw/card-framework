"""CLI entrypoints for the SPoRC summarizer-critic benchmark package."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import TextIO

from card_framework.benchmark.summarizer_critic.qmsum.telemetry import (
    collect_telemetry_sample,
    detect_threshold_breach,
    sleep_with_poll,
    write_telemetry_line,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_FULL_DATASET_MANIFEST_PATH,
    DEFAULT_GPU_THRESHOLD_GIB,
    DEFAULT_LONGFORM_AUDIT_PATH,
    DEFAULT_LONGFORM_MANIFEST_PATH,
    DEFAULT_PREPARED_FULL_ROOT,
    DEFAULT_PREPARED_ROOT,
    DEFAULT_RAM_THRESHOLD_PERCENT,
    DEFAULT_REFERENCE_WPM,
    DEFAULT_SUPERVISOR_ROOT,
    DEFAULT_SWEEP_ROOT,
    BenchmarkExecutionReport,
    DURATION_PRESET_SECONDS,
    DurationSweepPresetSummary,
    DurationSweepReport,
    SupervisedRunRecord,
)
from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    curate_longform_sporc_manifest,
    prepare_sporc_benchmark,
)
from card_framework.benchmark.summarizer_critic.sporc.reporting import (
    write_duration_sweep_artifacts,
    write_supervisor_session_artifacts,
)
from card_framework.benchmark.summarizer_critic.sporc.runner import (
    BenchmarkExecutionError,
    build_execute_parser,
    execute_benchmark,
)
from card_framework.shared.paths import REPO_ROOT


def _build_parser() -> argparse.ArgumentParser:
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
        default=10,
        help="How many prepared SPoRC examples to run per duration preset.",
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
            "- Single preset: uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep --mode single --duration-preset 5m --examples 10 --transcript-only",
            "- All presets: uv run python -m card_framework.benchmark.summarizer_critic.sporc duration-sweep --mode batch-all --examples 10 --transcript-only",
            "",
            "Notes:",
            "- The sweep defaults to a non-interactive menu because this CLI is automation-friendly.",
            "- Use --mode batch-all to bypass the menu and benchmark every preset.",
            "- Use --enable-voice-clone only when audio synthesis is part of the experiment.",
        ]
    )
    return "\n".join(lines)


def _parse_duration_preset_list(raw_value: str) -> list[str]:
    """Parse and validate a comma-separated duration preset list."""
    presets = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not presets:
        raise BenchmarkExecutionError("At least one duration preset is required.")
    invalid = [item for item in presets if item not in DURATION_PRESET_SECONDS]
    if invalid:
        raise BenchmarkExecutionError(
            f"Unsupported duration preset(s): {', '.join(invalid)}"
        )
    return presets


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


def _build_duration_execute_args(
    args: argparse.Namespace,
    *,
    duration_preset: str,
) -> argparse.Namespace:
    """Translate duration-sweep CLI args into execute-compatible args."""
    execute_args = argparse.Namespace(**vars(args))
    execute_args.command = "execute"
    execute_args.duration_preset = duration_preset
    execute_args.target_seconds_override = None
    execute_args.max_samples = int(args.examples)
    execute_args.transcript_only = not bool(args.enable_voice_clone)
    return execute_args


def _build_failed_preset_summary(
    *,
    duration_preset: str,
    sample_count: int,
    error_message: str,
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
        error_message=error_message,
    )


def _summarize_duration_report(
    report: BenchmarkExecutionReport,
    *,
    duration_preset: str,
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
        error_message=None,
    )


def _run_duration_sweep(args: argparse.Namespace) -> int:
    """Execute one or more duration presets and write aggregate artifacts."""
    if args.mode == "menu":
        print(_render_duration_sweep_menu())
        return 0

    presets = (
        [str(args.duration_preset)]
        if args.mode == "single"
        else _parse_duration_preset_list(str(args.duration_presets))
    )
    examples = int(args.examples)
    if examples <= 0:
        raise BenchmarkExecutionError("--examples must be positive.")

    sweep_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sweep_root = Path(args.sweep_root).resolve() / sweep_id
    sweep_root.mkdir(parents=True, exist_ok=True)

    preset_summaries: list[DurationSweepPresetSummary] = []
    for duration_preset in presets:
        execute_args = _build_duration_execute_args(
            args,
            duration_preset=duration_preset,
        )
        try:
            report = execute_benchmark(execute_args)
        except BenchmarkExecutionError as exc:
            preset_summaries.append(
                _build_failed_preset_summary(
                    duration_preset=duration_preset,
                    sample_count=examples,
                    error_message=str(exc),
                )
            )
            continue
        preset_summaries.append(
            _summarize_duration_report(report, duration_preset=duration_preset)
        )

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


def _build_supervised_child_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "card_framework.benchmark.summarizer_critic.sporc",
        "execute",
    ]
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
        "--duration-preset": args.duration_preset,
        "--target-seconds-override": args.target_seconds_override,
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
    if args.force_download_audio:
        command.append("--force-download-audio")
    if args.transcript_only:
        command.append("--transcript-only")
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
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-dataset":
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
        manifest, audit_path = curate_longform_sporc_manifest(
            source_manifest_path=Path(args.source_manifest).resolve(),
            output_manifest_path=Path(args.output_manifest).resolve(),
            audit_path=Path(args.audit_path).resolve(),
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
    if args.command == "duration-sweep":
        try:
            return _run_duration_sweep(args)
        except BenchmarkExecutionError as exc:
            print(json.dumps({"status": "failed", "error": str(exc)}))
            return 1
    return _run_supervised_session(args)
