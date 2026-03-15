"""Tests for cross-platform benchmark subprocess helpers."""

from __future__ import annotations

import signal
import subprocess

from card_framework.benchmark import process_utils


class _ProcessStub:
    """Minimal process stub for process-tree termination tests."""

    def __init__(
        self,
        *,
        pid: int = 1234,
        poll_result: int | None = None,
        wait_side_effects: tuple[object, ...] = (),
    ) -> None:
        self.pid = pid
        self._poll_result = poll_result
        self._wait_side_effects = list(wait_side_effects)
        self.wait_calls: list[float | None] = []
        self.kill_calls = 0

    def poll(self) -> int | None:
        """Return the configured poll result."""
        return self._poll_result

    def wait(self, timeout: float | None = None) -> int:
        """Record wait calls and replay configured side effects."""
        self.wait_calls.append(timeout)
        if self._wait_side_effects:
            effect = self._wait_side_effects.pop(0)
            if isinstance(effect, BaseException):
                raise effect
            return int(effect)
        return 0

    def kill(self) -> None:
        """Record process-level kill requests."""
        self.kill_calls += 1


def test_detached_popen_kwargs_uses_windows_process_group(monkeypatch) -> None:
    """Use CREATE_NEW_PROCESS_GROUP on Windows hosts."""
    monkeypatch.setattr(process_utils.os, "name", "nt", raising=False)
    monkeypatch.setattr(
        process_utils.subprocess,
        "CREATE_NEW_PROCESS_GROUP",
        512,
        raising=False,
    )

    kwargs = process_utils.detached_popen_kwargs()

    assert kwargs == {"creationflags": 512}


def test_detached_popen_kwargs_uses_start_new_session_on_posix(monkeypatch) -> None:
    """Use start_new_session on POSIX hosts."""
    monkeypatch.setattr(process_utils.os, "name", "posix", raising=False)

    kwargs = process_utils.detached_popen_kwargs()

    assert kwargs == {"start_new_session": True}


def test_terminate_process_tree_uses_taskkill_on_windows(monkeypatch) -> None:
    """Terminate a Windows child process tree with taskkill."""
    monkeypatch.setattr(process_utils.os, "name", "nt", raising=False)
    recorded_commands: list[list[str]] = []
    monkeypatch.setattr(
        process_utils.subprocess,
        "run",
        lambda command, **kwargs: recorded_commands.append(command),
    )
    process = _ProcessStub()

    process_utils.terminate_process_tree(process, timeout_seconds=5.0)

    assert recorded_commands == [["taskkill", "/T", "/F", "/PID", "1234"]]
    assert process.wait_calls == [5.0]
    assert process.kill_calls == 0


def test_terminate_process_tree_escalates_to_sigkill_on_posix(monkeypatch) -> None:
    """Escalate from SIGTERM to SIGKILL when a POSIX child ignores termination."""
    monkeypatch.setattr(process_utils.os, "name", "posix", raising=False)
    monkeypatch.setattr(process_utils.os, "getpgid", lambda pid: 4321, raising=False)
    monkeypatch.setattr(process_utils.signal, "SIGKILL", signal.SIGTERM, raising=False)
    recorded_signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(
        process_utils.os,
        "killpg",
        lambda pgid, sig: recorded_signals.append((pgid, sig)),
        raising=False,
    )
    process = _ProcessStub(
        wait_side_effects=(
            subprocess.TimeoutExpired(cmd="child", timeout=5.0),
            0,
        )
    )

    process_utils.terminate_process_tree(process, timeout_seconds=5.0)

    assert recorded_signals == [
        (4321, signal.SIGTERM),
        (4321, signal.SIGTERM),
    ]
    assert process.wait_calls == [5.0, 5.0]
    assert process.kill_calls == 0
