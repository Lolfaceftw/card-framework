"""Windows dedicated-GPU heartbeat monitor for voice-clone stage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import random
import re
import subprocess
import threading
import time
from typing import Any, Callable, Protocol

from card_framework.audio_pipeline.runtime import utc_now_iso

_NVIDIA_GPU_MEMORY_COMMAND: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=index,name,memory.total,memory.used",
    "--format=csv,noheader,nounits",
)
_POWERSHELL_PROCESS_MEMORY_SCRIPT = (
    "$samples = Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' | "
    "Select-Object -ExpandProperty CounterSamples; "
    "$rows = foreach ($sample in $samples) { "
    "if ($sample.InstanceName -match 'pid_(\\d+)_') { "
    "[pscustomobject]@{ "
    "instance_name = $sample.InstanceName; "
    "pid = [int]$matches[1]; "
    "dedicated_bytes = [int64]$sample.CookedValue "
    "} "
    "} "
    "}; "
    "@($rows) | ConvertTo-Json -Compress"
)
_POWERSHELL_PROCESS_TREE_SCRIPT = (
    "Get-CimInstance Win32_Process | "
    "Select-Object "
    "@{Name='pid';Expression={$_.ProcessId}}, "
    "@{Name='parent_pid';Expression={$_.ParentProcessId}}, "
    "@{Name='name';Expression={$_.Name}} | "
    "ConvertTo-Json -Compress"
)
_POWERSHELL_PREFIX: tuple[str, ...] = (
    "powershell",
    "-NoProfile",
    "-Command",
)
_PHYS_INDEX_PATTERN = re.compile(r"_phys_(?P<gpu>\d+)")


CommandExecutor = Callable[[Sequence[str], float], str]
PublishMessage = Callable[[str], None]


@dataclass(slots=True, frozen=True)
class VoiceCloneGpuHeartbeatConfig:
    """Validated runtime configuration for voice-clone GPU heartbeat."""

    enabled: bool = True
    interval_seconds: float = 10.0
    dedicated_usage_threshold_ratio: float = 0.95
    top_other_processes: int = 5
    command_timeout_seconds: float = 3.0

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError(
                "audio.voice_clone.gpu_heartbeat.interval_seconds must be > 0."
            )
        if not 0.0 < self.dedicated_usage_threshold_ratio <= 1.0:
            raise ValueError(
                "audio.voice_clone.gpu_heartbeat.dedicated_usage_threshold_ratio must be within (0.0, 1.0]."
            )
        if self.top_other_processes <= 0:
            raise ValueError(
                "audio.voice_clone.gpu_heartbeat.top_other_processes must be > 0."
            )
        if self.command_timeout_seconds <= 0:
            raise ValueError(
                "audio.voice_clone.gpu_heartbeat.command_timeout_seconds must be > 0."
            )


def parse_voice_clone_gpu_heartbeat_config(
    raw_cfg: Mapping[str, Any],
) -> VoiceCloneGpuHeartbeatConfig:
    """
    Parse and validate GPU heartbeat config from voice-clone config mapping.

    Args:
        raw_cfg: ``audio.voice_clone.gpu_heartbeat`` section.

    Returns:
        Parsed and validated heartbeat config.
    """
    return VoiceCloneGpuHeartbeatConfig(
        enabled=bool(raw_cfg.get("enabled", True)),
        interval_seconds=float(raw_cfg.get("interval_seconds", 10.0)),
        dedicated_usage_threshold_ratio=float(
            raw_cfg.get("dedicated_usage_threshold_ratio", 0.95)
        ),
        top_other_processes=int(raw_cfg.get("top_other_processes", 5)),
        command_timeout_seconds=float(raw_cfg.get("command_timeout_seconds", 3.0)),
    )


@dataclass(slots=True, frozen=True)
class GpuDeviceSnapshot:
    """Dedicated-memory snapshot for one physical GPU."""

    gpu_index: int
    name: str
    dedicated_total_mb: int
    dedicated_used_mb: int

    def __post_init__(self) -> None:
        if self.gpu_index < 0:
            raise ValueError("gpu_index must be >= 0.")
        if self.dedicated_total_mb <= 0:
            raise ValueError("dedicated_total_mb must be > 0.")
        if self.dedicated_used_mb < 0:
            raise ValueError("dedicated_used_mb must be >= 0.")

    @property
    def dedicated_usage_ratio(self) -> float:
        """Return dedicated-memory usage ratio in range [0, +inf)."""
        return self.dedicated_used_mb / float(self.dedicated_total_mb)


@dataclass(slots=True, frozen=True)
class GpuProcessSnapshot:
    """Dedicated-memory usage aggregated for one process across breached GPUs."""

    pid: int
    process_name: str
    dedicated_bytes: int
    gpu_indexes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.pid <= 0:
            raise ValueError("pid must be > 0.")
        if self.dedicated_bytes < 0:
            raise ValueError("dedicated_bytes must be >= 0.")
        if not self.process_name.strip():
            raise ValueError("process_name must be non-empty.")

    @property
    def dedicated_mebibytes(self) -> float:
        """Return dedicated usage converted to MiB."""
        return self.dedicated_bytes / float(1024 * 1024)


@dataclass(slots=True, frozen=True)
class GpuPressureAlert:
    """Pressure alert emitted when dedicated-memory usage crosses threshold."""

    device: GpuDeviceSnapshot
    threshold_ratio: float
    sampled_at_utc: str
    other_processes: tuple[GpuProcessSnapshot, ...]


class DedicatedGpuTelemetryProbe(Protocol):
    """Port for platform-specific dedicated-GPU telemetry probes."""

    def detect_pressure(
        self,
        *,
        threshold_ratio: float,
        top_other_processes: int,
        pipeline_root_pid: int,
    ) -> tuple[GpuPressureAlert, ...]:
        """Return pressure alerts for GPUs whose dedicated-memory ratio crosses threshold."""


@dataclass(slots=True, frozen=True)
class _ProcessNode:
    """Internal process-tree node."""

    pid: int
    parent_pid: int
    name: str


@dataclass(slots=True, frozen=True)
class _ProcessCounterSample:
    """Internal GPU process-counter sample."""

    instance_name: str
    pid: int
    dedicated_bytes: int


@dataclass(slots=True)
class WindowsNvidiaDedicatedGpuProbe(DedicatedGpuTelemetryProbe):
    """
    Windows implementation using ``nvidia-smi`` + PowerShell perf counters.

    This adapter intentionally keeps command parsing isolated from the domain
    heartbeat service so failures can be translated into explicit warnings.
    """

    command_timeout_seconds: float = 3.0
    max_retries: int = 2
    retry_base_seconds: float = 0.2
    command_executor: CommandExecutor | None = None
    sleep: Callable[[float], None] = field(default=time.sleep)

    def __post_init__(self) -> None:
        if self.command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be > 0.")
        if self.max_retries <= 0:
            raise ValueError("max_retries must be > 0.")
        if self.retry_base_seconds < 0:
            raise ValueError("retry_base_seconds must be >= 0.")
        if self.command_executor is None:
            self.command_executor = _run_subprocess_command

    def detect_pressure(
        self,
        *,
        threshold_ratio: float,
        top_other_processes: int,
        pipeline_root_pid: int,
    ) -> tuple[GpuPressureAlert, ...]:
        """
        Detect dedicated-memory pressure and resolve top external process consumers.

        Args:
            threshold_ratio: Dedicated-memory pressure threshold in (0.0, 1.0].
            top_other_processes: Number of top external processes to include.
            pipeline_root_pid: Root PID of the active pipeline process tree.

        Returns:
            Alerts for each breached GPU.

        Raises:
            RuntimeError: If telemetry commands fail or outputs are malformed.
        """
        if not 0.0 < threshold_ratio <= 1.0:
            raise ValueError("threshold_ratio must be within (0.0, 1.0].")
        if top_other_processes <= 0:
            raise ValueError("top_other_processes must be > 0.")
        if pipeline_root_pid <= 0:
            raise ValueError("pipeline_root_pid must be > 0.")

        devices = self._collect_gpu_devices()
        breached_devices = tuple(
            device for device in devices if device.dedicated_usage_ratio >= threshold_ratio
        )
        if not breached_devices:
            return ()

        breached_gpu_indexes = {device.gpu_index for device in breached_devices}
        process_tree = self._collect_process_tree()
        own_process_pids = _collect_descendant_pids(
            process_tree=process_tree,
            root_pid=pipeline_root_pid,
        )
        other_processes = self._collect_top_other_processes(
            breached_gpu_indexes=breached_gpu_indexes,
            process_tree=process_tree,
            excluded_pids=own_process_pids,
            top_n=top_other_processes,
        )
        sampled_at_utc = utc_now_iso()
        return tuple(
            GpuPressureAlert(
                device=device,
                threshold_ratio=threshold_ratio,
                sampled_at_utc=sampled_at_utc,
                other_processes=other_processes,
            )
            for device in breached_devices
        )

    def _collect_gpu_devices(self) -> tuple[GpuDeviceSnapshot, ...]:
        """Collect per-GPU dedicated-memory totals and usage from ``nvidia-smi``."""
        output = self._execute_with_retry(_NVIDIA_GPU_MEMORY_COMMAND)
        devices: list[GpuDeviceSnapshot] = []
        for raw_line in output.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",", maxsplit=3)]
            if len(parts) != 4:
                continue
            try:
                gpu_index = int(parts[0])
                total_mb = int(parts[2])
                used_mb = int(parts[3])
            except ValueError:
                continue
            devices.append(
                GpuDeviceSnapshot(
                    gpu_index=gpu_index,
                    name=parts[1],
                    dedicated_total_mb=total_mb,
                    dedicated_used_mb=used_mb,
                )
            )
        if not devices:
            raise RuntimeError(
                "Failed to parse GPU dedicated-memory output from nvidia-smi."
            )
        return tuple(devices)

    def _collect_process_tree(self) -> dict[int, _ProcessNode]:
        """Collect process parent-child mapping for exclusion of pipeline-owned PIDs."""
        command = (*_POWERSHELL_PREFIX, _POWERSHELL_PROCESS_TREE_SCRIPT)
        output = self._execute_with_retry(command)
        rows = _parse_json_rows(output)
        process_tree: dict[int, _ProcessNode] = {}
        for row in rows:
            pid = _coerce_int(row.get("pid"))
            parent_pid = _coerce_int(row.get("parent_pid"))
            if pid is None or parent_pid is None:
                continue
            name = str(row.get("name", "")).strip()
            if pid <= 0:
                continue
            process_tree[pid] = _ProcessNode(
                pid=pid,
                parent_pid=max(0, parent_pid),
                name=name or f"pid_{pid}",
            )
        return process_tree

    def _collect_top_other_processes(
        self,
        *,
        breached_gpu_indexes: set[int],
        process_tree: Mapping[int, _ProcessNode],
        excluded_pids: set[int],
        top_n: int,
    ) -> tuple[GpuProcessSnapshot, ...]:
        """Collect top external dedicated-memory consumers on breached GPUs."""
        command = (*_POWERSHELL_PREFIX, _POWERSHELL_PROCESS_MEMORY_SCRIPT)
        output = self._execute_with_retry(command)
        rows = _parse_json_rows(output)
        samples: list[_ProcessCounterSample] = []
        for row in rows:
            instance_name = str(row.get("instance_name", ""))
            pid = _coerce_int(row.get("pid"))
            dedicated_bytes = _coerce_int(row.get("dedicated_bytes"))
            if pid is None or dedicated_bytes is None:
                continue
            if pid <= 0 or dedicated_bytes <= 0:
                continue
            samples.append(
                _ProcessCounterSample(
                    instance_name=instance_name,
                    pid=pid,
                    dedicated_bytes=dedicated_bytes,
                )
            )

        usage_by_pid: dict[int, int] = {}
        gpu_indexes_by_pid: dict[int, set[int]] = {}
        for sample in samples:
            gpu_index = _parse_phys_gpu_index(sample.instance_name)
            if gpu_index is None or gpu_index not in breached_gpu_indexes:
                continue
            if sample.pid in excluded_pids:
                continue
            usage_by_pid[sample.pid] = usage_by_pid.get(sample.pid, 0) + sample.dedicated_bytes
            gpu_indexes = gpu_indexes_by_pid.setdefault(sample.pid, set())
            gpu_indexes.add(gpu_index)

        process_snapshots: list[GpuProcessSnapshot] = []
        for pid, dedicated_bytes in usage_by_pid.items():
            if dedicated_bytes <= 0:
                continue
            process_name = process_tree.get(pid, _ProcessNode(pid, 0, f"pid_{pid}")).name
            process_snapshots.append(
                GpuProcessSnapshot(
                    pid=pid,
                    process_name=process_name,
                    dedicated_bytes=dedicated_bytes,
                    gpu_indexes=tuple(sorted(gpu_indexes_by_pid.get(pid, set()))),
                )
            )

        process_snapshots.sort(key=lambda item: item.dedicated_bytes, reverse=True)
        return tuple(process_snapshots[:top_n])

    def _execute_with_retry(self, command: Sequence[str]) -> str:
        """Execute one telemetry command with bounded retry and jitter."""
        command_executor = self.command_executor or _run_subprocess_command
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return command_executor(command, self.command_timeout_seconds)
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay_seconds = (
                    self.retry_base_seconds * float(attempt)
                    + random.uniform(0.0, 0.1)
                )
                self.sleep(delay_seconds)
        assert last_error is not None
        raise RuntimeError(
            "GPU telemetry command failed after retries: "
            f"{' '.join(str(part) for part in command)}"
        ) from last_error


@dataclass(slots=True)
class VoiceCloneGpuHeartbeatService:
    """
    Heartbeat service that emits pressure alerts during voice cloning.

    The service is intentionally side-effectful (background thread + callbacks)
    and isolated from orchestration business logic.
    """

    probe: DedicatedGpuTelemetryProbe
    emit_status: PublishMessage
    emit_system: PublishMessage
    interval_seconds: float = 10.0
    threshold_ratio: float = 0.95
    top_other_processes: int = 5
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stop_event: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _pipeline_root_pid: int = field(default=0, init=False, repr=False)
    _warned_probe_failure: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be > 0.")
        if not 0.0 < self.threshold_ratio <= 1.0:
            raise ValueError("threshold_ratio must be within (0.0, 1.0].")
        if self.top_other_processes <= 0:
            raise ValueError("top_other_processes must be > 0.")

    def start(self, *, pipeline_root_pid: int) -> None:
        """
        Start the heartbeat loop if not already running.

        Args:
            pipeline_root_pid: Root PID of process tree that should be excluded
                from "other apps" attribution.
        """
        if pipeline_root_pid <= 0:
            raise ValueError("pipeline_root_pid must be > 0.")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._pipeline_root_pid = pipeline_root_pid
            self._warned_probe_failure = False
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop,
                name="voice-clone-gpu-heartbeat",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stop the heartbeat loop and wait briefly for thread termination."""
        with self._lock:
            thread = self._thread
            self._thread = None
            self._stop_event.set()
        if thread is not None:
            thread.join(timeout=min(2.0, self.interval_seconds + 0.2))

    def _run_loop(self) -> None:
        """Run heartbeat checks until stop event is set."""
        while not self._stop_event.is_set():
            self._poll_once()
            if self._stop_event.wait(self.interval_seconds):
                break

    def _poll_once(self) -> None:
        """Run one pressure check and emit alerts if threshold is breached."""
        try:
            alerts = self.probe.detect_pressure(
                threshold_ratio=self.threshold_ratio,
                top_other_processes=self.top_other_processes,
                pipeline_root_pid=self._pipeline_root_pid,
            )
        except Exception as exc:
            if not self._warned_probe_failure:
                self.emit_system(
                    "Voice clone stage: GPU heartbeat probe unavailable; "
                    f"continuing without GPU pressure attribution ({exc})."
                )
                self._warned_probe_failure = True
            return

        for alert in alerts:
            self.emit_status(format_gpu_pressure_alert(alert))


def format_gpu_pressure_alert(alert: GpuPressureAlert) -> str:
    """Format one pressure alert for user-facing status output."""
    usage_pct = alert.device.dedicated_usage_ratio * 100.0
    base_message = (
        "Voice clone stage: dedicated GPU memory pressure "
        f"{usage_pct:.1f}% on GPU {alert.device.gpu_index} "
        f"({alert.device.dedicated_used_mb}/{alert.device.dedicated_total_mb} MiB used)."
    )
    if not alert.other_processes:
        return f"{base_message} Top other apps: unavailable."
    top_apps = ", ".join(
        (
            f"{process.process_name}(pid={process.pid}, "
            f"{process.dedicated_mebibytes:.1f} MiB)"
        )
        for process in alert.other_processes
    )
    return f"{base_message} Top other apps: {top_apps}"


def _run_subprocess_command(command: Sequence[str], timeout_seconds: float) -> str:
    """Run one command and return stdout, raising ``RuntimeError`` on failure."""
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            "Command timed out while collecting GPU telemetry: "
            f"{' '.join(str(part) for part in command)}"
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(
            "Command failed while collecting GPU telemetry: "
            f"{' '.join(str(part) for part in command)}; {detail}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "Command unavailable while collecting GPU telemetry: "
            f"{' '.join(str(part) for part in command)}"
        ) from exc
    return completed.stdout


def _parse_json_rows(raw_output: str) -> list[dict[str, object]]:
    """Parse command JSON output into a normalized list of object rows."""
    text = raw_output.strip()
    if not text:
        return []
    payload = json.loads(text)
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _parse_phys_gpu_index(instance_name: str) -> int | None:
    """Extract physical GPU index from Windows GPU counter instance name."""
    match = _PHYS_INDEX_PATTERN.search(instance_name)
    if match is None:
        return None
    try:
        return int(match.group("gpu"))
    except ValueError:
        return None


def _coerce_int(value: object) -> int | None:
    """Coerce numeric command fields into ``int`` when possible."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def _collect_descendant_pids(
    *,
    process_tree: Mapping[int, _ProcessNode],
    root_pid: int,
) -> set[int]:
    """Collect root process and all descendants from parent pointers."""
    descendants: set[int] = {root_pid}
    while True:
        added = {
            pid
            for pid, node in process_tree.items()
            if node.parent_pid in descendants and pid not in descendants
        }
        if not added:
            break
        descendants.update(added)
    return descendants

