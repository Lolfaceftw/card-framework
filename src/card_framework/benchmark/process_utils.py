"""Cross-platform subprocess group helpers for benchmark-style supervisors."""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any


def detached_popen_kwargs() -> dict[str, Any]:
    """Return Popen kwargs that isolate a child into its own process group."""
    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def terminate_process_tree(
    process: subprocess.Popen[str],
    *,
    timeout_seconds: float = 5.0,
) -> None:
    """Terminate a child process and its descendants across platforms."""
    if process.poll() is not None:
        return

    if os.name == "nt":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            process.wait(timeout=timeout_seconds)
            return
        except (OSError, PermissionError, ProcessLookupError, ValueError):
            pass
        except subprocess.TimeoutExpired:
            pass
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout_seconds)
        return

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return

    try:
        process.wait(timeout=timeout_seconds)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError):
        return

    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout_seconds)
