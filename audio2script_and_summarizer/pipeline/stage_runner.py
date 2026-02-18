"""Subprocess stage runner utilities for pipeline orchestration."""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable


def extract_module_name(cmd: list[str]) -> str:
    """Extract ``python -m`` module name from a subprocess command."""
    if "-m" in cmd:
        module_index = cmd.index("-m")
        if module_index + 1 < len(cmd):
            return cmd[module_index + 1]
    if cmd:
        return Path(cmd[0]).name
    return "-"


def run_stage_command(
    *,
    cmd: list[str],
    current_env: dict[str, str],
    use_dashboard: bool,
    stage_name: str,
    heartbeat_seconds: float,
    model_info: str,
    active_dashboard: Any | None,
    parse_stream_event_line: Callable[[str], dict[str, Any] | None],
    route_stream_event: Callable[[Any, dict[str, Any]], bool],
    logger: logging.Logger,
    subprocess_module: Any = subprocess,
) -> None:
    """Run a child command and optionally stream output into dashboard logs.

    Args:
        cmd: Subprocess command line.
        current_env: Environment variables for the child process.
        use_dashboard: Stream process output into the split UI when True.
        stage_name: Human-readable stage label for status updates.
        heartbeat_seconds: How often to emit heartbeat while output is silent.
        model_info: Model/runtime details shown in the controls panel.
        active_dashboard: Dashboard instance used for live updates.
        parse_stream_event_line: Event parser for stream marker lines.
        route_stream_event: Stream event router callback.
        logger: Logger instance.

    Raises:
        subprocess.CalledProcessError: When child process exits non-zero.
    """
    command_display = subprocess_module.list2cmdline(cmd)
    module_name = extract_module_name(cmd)
    logger.info(
        "Starting stage subprocess",
        extra={
            "component": "run_pipeline",
            "job_id": stage_name,
        },
    )
    logger.debug(
        "Stage launch details: stage=%s module=%s command=%s",
        stage_name,
        module_name,
        command_display,
    )
    if active_dashboard is not None and active_dashboard.enabled:
        active_dashboard.set_status(
            stage_name=stage_name,
            substep="Launching subprocess",
            module_name=module_name,
            command_display=command_display,
            model_info=model_info,
            pid=None,
            reset_elapsed=True,
        )
        active_dashboard.event(f"{stage_name}: launching {module_name}")

    if not use_dashboard:
        started_at = time.monotonic()
        subprocess_module.run(cmd, check=True, env=current_env)
        elapsed_total = int(time.monotonic() - started_at)
        logger.info("%s completed successfully in %ss", stage_name, elapsed_total)
        return

    child_env = current_env.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    process = subprocess_module.Popen(
        cmd,
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if active_dashboard is not None and active_dashboard.enabled:
        active_dashboard.set_status(
            stage_name=stage_name,
            substep="Subprocess running",
            module_name=module_name,
            command_display=command_display,
            model_info=model_info,
            pid=process.pid,
        )
        active_dashboard.event(f"{stage_name}: subprocess started (pid={process.pid})")

    output_stream = process.stdout
    if output_stream is None:
        raise RuntimeError("Stage subprocess did not expose a readable stdout stream")

    queue_sentinel = object()
    output_queue: queue.Queue[object] = queue.Queue()

    def _read_child_output() -> None:
        for raw_line in output_stream:
            output_queue.put(raw_line)
        output_queue.put(queue_sentinel)

    reader_thread = threading.Thread(
        target=_read_child_output,
        name=f"pipeline-reader-{stage_name}",
        daemon=True,
    )
    reader_thread.start()

    started_at = time.monotonic()
    next_heartbeat = started_at + max(0.0, heartbeat_seconds)
    output_finished = False

    while True:
        try:
            queued_item = output_queue.get(timeout=0.2)
        except queue.Empty:
            queued_item = None

        if queued_item is queue_sentinel:
            output_finished = True
        elif isinstance(queued_item, str):
            line = queued_item.rstrip("\n")
            if line and active_dashboard is not None:
                logger.debug("%s | %s", stage_name, line)
                stream_event_payload = parse_stream_event_line(line)
                if stream_event_payload is not None:
                    route_stream_event(active_dashboard, stream_event_payload)
                    if stage_name == "Stage 2 (Summarizer)":
                        event_name = (
                            str(stream_event_payload.get("event", "")).strip().lower()
                        )
                        stream_status_substep = ""
                        if event_name == "summary_json_ready":
                            stream_status_substep = (
                                "Summary JSON ready; finalizing subprocess"
                            )
                        elif event_name == "done":
                            stream_status_substep = (
                                "DeepSeek stream done; waiting for subprocess completion"
                            )
                        if stream_status_substep:
                            active_dashboard.set_status(
                                stage_name=stage_name,
                                substep=stream_status_substep,
                                module_name=module_name,
                                command_display=command_display,
                                model_info=model_info,
                                pid=process.pid,
                            )
                            active_dashboard.set_progress_detail(
                                f"{stage_name} - {stream_status_substep}"
                            )
                    next_heartbeat = time.monotonic() + max(0.0, heartbeat_seconds)
                    continue
                handled_status_marker = False
                if line.startswith("[PROGRESS] "):
                    progress_payload = line[len("[PROGRESS] ") :].strip()
                    tokens = progress_payload.split()
                    if len(tokens) >= 2 and tokens[0].lower() == "htdemucs":
                        progress_value = tokens[1].lower()
                        if progress_value == "start":
                            active_dashboard.start_detail_progress(
                                "htdemucs source separation"
                            )
                            active_dashboard.set_status(
                                stage_name=stage_name,
                                substep="Running htdemucs source separation",
                                module_name=module_name,
                                command_display=command_display,
                                model_info=model_info,
                                pid=process.pid,
                            )
                            handled_status_marker = True
                        elif progress_value.isdigit():
                            active_dashboard.update_detail_progress(int(progress_value))
                            active_dashboard.set_status(
                                stage_name=stage_name,
                                substep=f"htdemucs source separation ({progress_value}%)",
                                module_name=module_name,
                                command_display=command_display,
                                model_info=model_info,
                                pid=process.pid,
                            )
                            handled_status_marker = True
                        elif progress_value in {"done", "complete"}:
                            active_dashboard.update_detail_progress(100)
                            active_dashboard.finish_detail_progress("htdemucs complete")
                            handled_status_marker = True
                        elif progress_value == "failed":
                            active_dashboard.finish_detail_progress("htdemucs failed")
                            handled_status_marker = True
                elif line.startswith("[MODEL] "):
                    model_info = line[len("[MODEL] ") :].strip() or model_info
                    active_dashboard.set_status(
                        stage_name=stage_name,
                        substep="Loading model artifacts",
                        module_name=module_name,
                        command_display=command_display,
                        model_info=model_info,
                        pid=process.pid,
                    )
                    active_dashboard.set_progress_detail(
                        f"{stage_name} - Loading model artifacts"
                    )
                    handled_status_marker = True
                elif line.startswith("[STATUS] "):
                    status_text = line[len("[STATUS] ") :].strip()
                    active_dashboard.set_status(
                        stage_name=stage_name,
                        substep=status_text or "Running subprocess",
                        module_name=module_name,
                        command_display=command_display,
                        model_info=model_info,
                        pid=process.pid,
                    )
                    active_dashboard.set_progress_detail(
                        f"{stage_name} - {status_text or 'Running subprocess'}"
                    )
                    handled_status_marker = True
                if not handled_status_marker:
                    active_dashboard.log(line)
                    active_dashboard.set_status(
                        stage_name=stage_name,
                        substep="Streaming subprocess output",
                        module_name=module_name,
                        command_display=command_display,
                        model_info=model_info,
                        pid=process.pid,
                    )
            next_heartbeat = time.monotonic() + max(0.0, heartbeat_seconds)

        now = time.monotonic()
        if heartbeat_seconds > 0 and now >= next_heartbeat:
            elapsed_seconds = int(now - started_at)
            if active_dashboard is not None and active_dashboard.enabled:
                active_dashboard.set_status(
                    stage_name=stage_name,
                    substep=f"Waiting for subprocess output ({elapsed_seconds}s)",
                    module_name=module_name,
                    command_display=command_display,
                    model_info=model_info,
                    pid=process.pid,
                )
            next_heartbeat = now + heartbeat_seconds

        if output_finished and process.poll() is not None and output_queue.empty():
            break

    reader_thread.join(timeout=0.4)
    return_code = process.wait()
    elapsed_total = int(time.monotonic() - started_at)
    if active_dashboard is not None and active_dashboard.enabled:
        if stage_name == "Stage 2 (Summarizer)":
            active_dashboard.close_deepseek_stream_panel()
        active_dashboard.set_status(
            stage_name=stage_name,
            substep=f"Completed with exit code {return_code}",
            module_name=module_name,
            command_display=command_display,
            model_info=model_info,
            pid=None,
        )
        active_dashboard.event(
            f"{stage_name}: completed with exit code {return_code} ({elapsed_total}s)"
        )
    logger.info("%s finished with exit code %s in %ss", stage_name, return_code, elapsed_total)

    if return_code != 0:
        logger.error("%s failed with non-zero exit code %s", stage_name, return_code)
        raise subprocess_module.CalledProcessError(return_code, cmd)
