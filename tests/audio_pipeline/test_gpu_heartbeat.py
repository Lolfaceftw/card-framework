"""Tests for Windows dedicated-GPU heartbeat telemetry and lifecycle."""

from __future__ import annotations

import json
import time

import pytest

from audio_pipeline.gpu_heartbeat import (
    GpuDeviceSnapshot,
    GpuPressureAlert,
    VoiceCloneGpuHeartbeatService,
    WindowsNvidiaDedicatedGpuProbe,
    parse_voice_clone_gpu_heartbeat_config,
)


class _FakeCommandExecutor:
    """Deterministic command executor for probe contract tests."""

    def __init__(
        self,
        *,
        nvidia_output: str,
        process_memory_rows: list[dict[str, object]],
        process_tree_rows: list[dict[str, object]],
    ) -> None:
        self._nvidia_output = nvidia_output
        self._process_memory_json = json.dumps(process_memory_rows)
        self._process_tree_json = json.dumps(process_tree_rows)
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: list[str] | tuple[str, ...], _timeout: float) -> str:
        cmd_tuple = tuple(str(part) for part in command)
        self.commands.append(cmd_tuple)
        if cmd_tuple and cmd_tuple[0] == "nvidia-smi":
            return self._nvidia_output
        if "GPU Process Memory(*)\\Dedicated Usage" in " ".join(cmd_tuple):
            return self._process_memory_json
        if "Get-CimInstance Win32_Process" in " ".join(cmd_tuple):
            return self._process_tree_json
        raise RuntimeError(f"Unexpected command: {' '.join(cmd_tuple)}")


class _FailingProbe:
    """Probe stub that always raises for service resilience testing."""

    def __init__(self) -> None:
        self.calls = 0

    def detect_pressure(
        self,
        *,
        threshold_ratio: float,
        top_other_processes: int,
        pipeline_root_pid: int,
    ) -> tuple[GpuPressureAlert, ...]:
        del threshold_ratio, top_other_processes, pipeline_root_pid
        self.calls += 1
        raise RuntimeError("telemetry unavailable")


class _SingleAlertProbe:
    """Probe stub that emits one alert then idles."""

    def __init__(self) -> None:
        self.calls = 0

    def detect_pressure(
        self,
        *,
        threshold_ratio: float,
        top_other_processes: int,
        pipeline_root_pid: int,
    ) -> tuple[GpuPressureAlert, ...]:
        del threshold_ratio, top_other_processes, pipeline_root_pid
        self.calls += 1
        if self.calls > 1:
            return ()
        return (
            GpuPressureAlert(
                device=GpuDeviceSnapshot(
                    gpu_index=0,
                    name="NVIDIA RTX",
                    dedicated_total_mb=12288,
                    dedicated_used_mb=12000,
                ),
                threshold_ratio=0.95,
                sampled_at_utc="2026-03-03T00:00:00+00:00",
                other_processes=(),
            ),
        )


def test_parse_voice_clone_gpu_heartbeat_config_defaults() -> None:
    config = parse_voice_clone_gpu_heartbeat_config({})

    assert config.enabled is True
    assert config.interval_seconds == 10.0
    assert config.dedicated_usage_threshold_ratio == 0.95
    assert config.top_other_processes == 5
    assert config.command_timeout_seconds == 3.0


def test_parse_voice_clone_gpu_heartbeat_config_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="dedicated_usage_threshold_ratio"):
        parse_voice_clone_gpu_heartbeat_config(
            {"dedicated_usage_threshold_ratio": 1.5}
        )


def test_windows_probe_detects_pressure_and_top_external_apps() -> None:
    executor = _FakeCommandExecutor(
        nvidia_output=(
            "0, NVIDIA GeForce RTX 3060, 12288, 11893\n"
            "1, NVIDIA GeForce GTX 1050, 4096, 100\n"
        ),
        process_memory_rows=[
            {
                "instance_name": "pid_100_luid_0x0_phys_0",
                "pid": 100,
                "dedicated_bytes": 400_000_000,
            },
            {
                "instance_name": "pid_200_luid_0x0_phys_0",
                "pid": 200,
                "dedicated_bytes": 250_000_000,
            },
            {
                "instance_name": "pid_300_luid_0x0_phys_0",
                "pid": 300,
                "dedicated_bytes": 2_000_000_000,
            },
            {
                "instance_name": "pid_301_luid_0x0_phys_0",
                "pid": 301,
                "dedicated_bytes": 900_000_000,
            },
            {
                "instance_name": "pid_302_luid_0x0_phys_1",
                "pid": 302,
                "dedicated_bytes": 3_000_000_000,
            },
        ],
        process_tree_rows=[
            {"pid": 100, "parent_pid": 50, "name": "python.exe"},
            {"pid": 200, "parent_pid": 100, "name": "uv.exe"},
            {"pid": 300, "parent_pid": 1, "name": "zen.exe"},
            {"pid": 301, "parent_pid": 1, "name": "Code.exe"},
            {"pid": 302, "parent_pid": 1, "name": "obs64.exe"},
        ],
    )
    probe = WindowsNvidiaDedicatedGpuProbe(
        command_timeout_seconds=1.0,
        command_executor=executor,
    )

    alerts = probe.detect_pressure(
        threshold_ratio=0.95,
        top_other_processes=2,
        pipeline_root_pid=100,
    )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.device.gpu_index == 0
    assert alert.device.dedicated_used_mb == 11893
    # Own process tree pids 100/200 are excluded; non-breached GPU pid 302 excluded.
    assert [process.pid for process in alert.other_processes] == [300, 301]
    assert [process.process_name for process in alert.other_processes] == [
        "zen.exe",
        "Code.exe",
    ]


def test_windows_probe_skips_process_queries_when_usage_is_below_threshold() -> None:
    executor = _FakeCommandExecutor(
        nvidia_output="0, NVIDIA GeForce RTX 3060, 12288, 2000\n",
        process_memory_rows=[],
        process_tree_rows=[],
    )
    probe = WindowsNvidiaDedicatedGpuProbe(
        command_timeout_seconds=1.0,
        command_executor=executor,
    )

    alerts = probe.detect_pressure(
        threshold_ratio=0.95,
        top_other_processes=3,
        pipeline_root_pid=100,
    )

    assert alerts == ()
    assert len(executor.commands) == 1
    assert executor.commands[0][0] == "nvidia-smi"


def test_gpu_heartbeat_service_emits_alerts() -> None:
    status_messages: list[str] = []
    system_messages: list[str] = []
    probe = _SingleAlertProbe()
    service = VoiceCloneGpuHeartbeatService(
        probe=probe,  # type: ignore[arg-type]
        emit_status=status_messages.append,
        emit_system=system_messages.append,
        interval_seconds=0.01,
        threshold_ratio=0.95,
        top_other_processes=5,
    )

    service.start(pipeline_root_pid=100)
    deadline = time.time() + 0.2
    while not status_messages and time.time() < deadline:
        time.sleep(0.01)
    service.stop()

    assert status_messages
    assert "dedicated GPU memory pressure" in status_messages[0]
    assert system_messages == []


def test_gpu_heartbeat_service_warns_once_when_probe_fails() -> None:
    status_messages: list[str] = []
    system_messages: list[str] = []
    probe = _FailingProbe()
    service = VoiceCloneGpuHeartbeatService(
        probe=probe,  # type: ignore[arg-type]
        emit_status=status_messages.append,
        emit_system=system_messages.append,
        interval_seconds=0.01,
        threshold_ratio=0.95,
        top_other_processes=5,
    )

    service.start(pipeline_root_pid=100)
    time.sleep(0.06)
    service.stop()

    assert status_messages == []
    assert len(system_messages) == 1
    assert "GPU heartbeat probe unavailable" in system_messages[0]
    assert probe.calls >= 2

