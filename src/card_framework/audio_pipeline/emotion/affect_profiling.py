"""Resource profiling helpers for affect-analysis runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any


_POWERSHELL_PREFIX: tuple[str, ...] = (
    "powershell",
    "-NoProfile",
    "-Command",
)
_POWERSHELL_GPU_PROCESS_MEMORY_SCRIPT = (
    "$dedicated = Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' | "
    "Select-Object -ExpandProperty CounterSamples; "
    "$shared = Get-Counter '\\GPU Process Memory(*)\\Shared Usage' | "
    "Select-Object -ExpandProperty CounterSamples; "
    "$rows = @{}; "
    "foreach ($sample in $dedicated) { "
    "if ($sample.InstanceName -match 'pid_(\\d+)_') { "
    "$rows[$sample.InstanceName] = [pscustomobject]@{ "
    "instance_name = $sample.InstanceName; "
    "pid = [int]$matches[1]; "
    "dedicated_bytes = [int64]$sample.CookedValue; "
    "shared_bytes = [int64]0 "
    "} "
    "} "
    "}; "
    "foreach ($sample in $shared) { "
    "if ($sample.InstanceName -match 'pid_(\\d+)_') { "
    "if (-not $rows.ContainsKey($sample.InstanceName)) { "
    "$rows[$sample.InstanceName] = [pscustomobject]@{ "
    "instance_name = $sample.InstanceName; "
    "pid = [int]$matches[1]; "
    "dedicated_bytes = [int64]0; "
    "shared_bytes = [int64]$sample.CookedValue "
    "} "
    "} else { "
    "$rows[$sample.InstanceName].shared_bytes = [int64]$sample.CookedValue "
    "} "
    "} "
    "}; "
    "@($rows.Values) | ConvertTo-Json -Compress"
)
_POWERSHELL_GPU_ADAPTER_MEMORY_SCRIPT = (
    "$dedicated = Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' | "
    "Select-Object -ExpandProperty CounterSamples; "
    "$shared = Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage' | "
    "Select-Object -ExpandProperty CounterSamples; "
    "$rows = @{}; "
    "foreach ($sample in $dedicated) { "
    "$rows[$sample.InstanceName] = [pscustomobject]@{ "
    "instance_name = $sample.InstanceName; "
    "dedicated_bytes = [int64]$sample.CookedValue; "
    "shared_bytes = [int64]0 "
    "} "
    "}; "
    "foreach ($sample in $shared) { "
    "if (-not $rows.ContainsKey($sample.InstanceName)) { "
    "$rows[$sample.InstanceName] = [pscustomobject]@{ "
    "instance_name = $sample.InstanceName; "
    "dedicated_bytes = [int64]0; "
    "shared_bytes = [int64]$sample.CookedValue "
    "} "
    "} else { "
    "$rows[$sample.InstanceName].shared_bytes = [int64]$sample.CookedValue "
    "} "
    "}; "
    "@($rows.Values) | ConvertTo-Json -Compress"
)


@dataclass(slots=True, frozen=True)
class ProcessGpuMemorySample:
    """One GPU process-memory counter row."""

    instance_name: str
    pid: int
    dedicated_bytes: int
    shared_bytes: int


@dataclass(slots=True, frozen=True)
class AdapterGpuMemorySample:
    """One GPU adapter-memory counter row."""

    instance_name: str
    dedicated_bytes: int
    shared_bytes: int


@dataclass(slots=True, frozen=True)
class AggregatedProcessGpuMemory:
    """Aggregated GPU process memory for one PID."""

    pid: int
    dedicated_bytes: int
    shared_bytes: int

    @property
    def dedicated_mebibytes(self) -> float:
        """Return dedicated GPU memory in MiB."""
        return self.dedicated_bytes / float(1024 * 1024)

    @property
    def shared_mebibytes(self) -> float:
        """Return shared GPU memory in MiB."""
        return self.shared_bytes / float(1024 * 1024)


@dataclass(slots=True, frozen=True)
class AggregatedAdapterGpuMemory:
    """Aggregated GPU adapter memory across all visible adapters."""

    dedicated_bytes: int
    shared_bytes: int

    @property
    def dedicated_mebibytes(self) -> float:
        """Return dedicated GPU memory in MiB."""
        return self.dedicated_bytes / float(1024 * 1024)

    @property
    def shared_mebibytes(self) -> float:
        """Return shared GPU memory in MiB."""
        return self.shared_bytes / float(1024 * 1024)


@dataclass(slots=True, frozen=True)
class ResourceProfile:
    """Captured resource profile for one affect-analysis step."""

    label: str
    pid: int
    elapsed_seconds: float
    sample_count: int
    baseline_rss_mb: float | None = None
    peak_rss_mb: float | None = None
    baseline_gpu_dedicated_mb: float | None = None
    peak_gpu_dedicated_mb: float | None = None
    baseline_gpu_shared_mb: float | None = None
    peak_gpu_shared_mb: float | None = None
    baseline_total_gpu_dedicated_mb: float | None = None
    peak_total_gpu_dedicated_mb: float | None = None
    baseline_total_gpu_shared_mb: float | None = None
    peak_total_gpu_shared_mb: float | None = None
    torch_peak_allocated_mb: float | None = None
    torch_peak_reserved_mb: float | None = None

    def to_payload(self) -> dict[str, Any]:
        """Serialize this profile as a JSON-friendly mapping."""
        return asdict(self)


def parse_process_memory_rows(output: str) -> list[ProcessGpuMemorySample]:
    """Parse PowerShell GPU process-memory JSON rows."""
    if not output.strip():
        return []
    payload = json.loads(output)
    rows = payload if isinstance(payload, list) else [payload]
    samples: list[ProcessGpuMemorySample] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        samples.append(
            ProcessGpuMemorySample(
                instance_name=str(row.get("instance_name", "")),
                pid=int(row.get("pid", 0) or 0),
                dedicated_bytes=int(row.get("dedicated_bytes", 0) or 0),
                shared_bytes=int(row.get("shared_bytes", 0) or 0),
            )
        )
    return samples


def parse_adapter_memory_rows(output: str) -> list[AdapterGpuMemorySample]:
    """Parse PowerShell GPU adapter-memory JSON rows."""
    if not output.strip():
        return []
    payload = json.loads(output)
    rows = payload if isinstance(payload, list) else [payload]
    samples: list[AdapterGpuMemorySample] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        samples.append(
            AdapterGpuMemorySample(
                instance_name=str(row.get("instance_name", "")),
                dedicated_bytes=int(row.get("dedicated_bytes", 0) or 0),
                shared_bytes=int(row.get("shared_bytes", 0) or 0),
            )
        )
    return samples


def aggregate_process_memory_for_pid(
    samples: list[ProcessGpuMemorySample],
    pid: int,
) -> AggregatedProcessGpuMemory:
    """Aggregate GPU process-memory rows for one PID."""
    dedicated_bytes = 0
    shared_bytes = 0
    for sample in samples:
        if sample.pid != pid:
            continue
        dedicated_bytes += sample.dedicated_bytes
        shared_bytes += sample.shared_bytes
    return AggregatedProcessGpuMemory(
        pid=pid,
        dedicated_bytes=dedicated_bytes,
        shared_bytes=shared_bytes,
    )


def aggregate_adapter_memory(
    samples: list[AdapterGpuMemorySample],
) -> AggregatedAdapterGpuMemory:
    """Aggregate GPU adapter-memory rows."""
    dedicated_bytes = sum(sample.dedicated_bytes for sample in samples)
    shared_bytes = sum(sample.shared_bytes for sample in samples)
    return AggregatedAdapterGpuMemory(
        dedicated_bytes=dedicated_bytes,
        shared_bytes=shared_bytes,
    )


class ResourceUsageMonitor:
    """Poll process RSS and GPU memory while a step is running."""

    def __init__(
        self,
        *,
        pid: int,
        device: str,
        sample_interval_seconds: float = 0.2,
        command_timeout_seconds: float = 3.0,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be > 0.")
        if command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be > 0.")
        self.pid = int(pid)
        self.device = str(device).strip().lower()
        self.sample_interval_seconds = float(sample_interval_seconds)
        self.command_timeout_seconds = float(command_timeout_seconds)
        self._samples: list[tuple[float | None, float | None, float | None, float | None, float | None]] = []
        self._started_at = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start polling in the background."""
        self._samples.clear()
        self._stop_event.clear()
        self._started_at = time.perf_counter()
        self._capture_sample()
        self._maybe_reset_torch_peaks()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, *, label: str) -> ResourceProfile:
        """Stop polling and return the aggregated profile."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.sample_interval_seconds * 2.0, 0.5))
        self._capture_sample()
        elapsed_seconds = max(time.perf_counter() - self._started_at, 0.0)
        rss_values = [sample[0] for sample in self._samples if sample[0] is not None]
        gpu_dedicated_values = [sample[1] for sample in self._samples if sample[1] is not None]
        gpu_shared_values = [sample[2] for sample in self._samples if sample[2] is not None]
        total_gpu_dedicated_values = [sample[3] for sample in self._samples if sample[3] is not None]
        total_gpu_shared_values = [sample[4] for sample in self._samples if sample[4] is not None]
        torch_peak_allocated_mb, torch_peak_reserved_mb = self._read_torch_peaks()
        return ResourceProfile(
            label=label,
            pid=self.pid,
            elapsed_seconds=round(elapsed_seconds, 3),
            sample_count=len(self._samples),
            baseline_rss_mb=rss_values[0] if rss_values else None,
            peak_rss_mb=max(rss_values) if rss_values else None,
            baseline_gpu_dedicated_mb=gpu_dedicated_values[0] if gpu_dedicated_values else None,
            peak_gpu_dedicated_mb=max(gpu_dedicated_values) if gpu_dedicated_values else None,
            baseline_gpu_shared_mb=gpu_shared_values[0] if gpu_shared_values else None,
            peak_gpu_shared_mb=max(gpu_shared_values) if gpu_shared_values else None,
            baseline_total_gpu_dedicated_mb=(
                total_gpu_dedicated_values[0] if total_gpu_dedicated_values else None
            ),
            peak_total_gpu_dedicated_mb=(
                max(total_gpu_dedicated_values) if total_gpu_dedicated_values else None
            ),
            baseline_total_gpu_shared_mb=(
                total_gpu_shared_values[0] if total_gpu_shared_values else None
            ),
            peak_total_gpu_shared_mb=(
                max(total_gpu_shared_values) if total_gpu_shared_values else None
            ),
            torch_peak_allocated_mb=torch_peak_allocated_mb,
            torch_peak_reserved_mb=torch_peak_reserved_mb,
        )

    def _run(self) -> None:
        """Poll until the stop event is set."""
        while not self._stop_event.wait(self.sample_interval_seconds):
            self._capture_sample()

    def _capture_sample(self) -> None:
        """Capture one instantaneous sample."""
        rss_mb = _process_rss_mb(self.pid)
        gpu_dedicated_mb: float | None = None
        gpu_shared_mb: float | None = None
        total_gpu_dedicated_mb: float | None = None
        total_gpu_shared_mb: float | None = None
        if self.device == "cuda":
            process_gpu = _sample_process_gpu_memory(
                pid=self.pid,
                timeout_seconds=self.command_timeout_seconds,
            )
            if process_gpu is not None:
                gpu_dedicated_mb = process_gpu.dedicated_mebibytes
                gpu_shared_mb = process_gpu.shared_mebibytes
            adapter_gpu = _sample_adapter_gpu_memory(
                timeout_seconds=self.command_timeout_seconds,
            )
            if adapter_gpu is not None:
                total_gpu_dedicated_mb = adapter_gpu.dedicated_mebibytes
                total_gpu_shared_mb = adapter_gpu.shared_mebibytes
        self._samples.append(
            (
                rss_mb,
                gpu_dedicated_mb,
                gpu_shared_mb,
                total_gpu_dedicated_mb,
                total_gpu_shared_mb,
            )
        )

    def _maybe_reset_torch_peaks(self) -> None:
        """Reset torch CUDA peak counters when available."""
        if self.device != "cuda":
            return
        try:
            import torch
        except Exception:
            return
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not callable(getattr(cuda, "is_available", None)):
            return
        if not cuda.is_available():
            return
        try:
            cuda.reset_peak_memory_stats()
        except Exception:
            pass

    def _read_torch_peaks(self) -> tuple[float | None, float | None]:
        """Read torch peak allocation metrics when available."""
        if self.device != "cuda":
            return None, None
        try:
            import torch
        except Exception:
            return None, None
        cuda = getattr(torch, "cuda", None)
        if cuda is None or not callable(getattr(cuda, "is_available", None)):
            return None, None
        if not cuda.is_available():
            return None, None
        try:
            allocated_mb = torch.cuda.max_memory_allocated() / float(1024 * 1024)
        except Exception:
            allocated_mb = None
        try:
            reserved_mb = torch.cuda.max_memory_reserved() / float(1024 * 1024)
        except Exception:
            reserved_mb = None
        return allocated_mb, reserved_mb


def _process_rss_mb(pid: int) -> float | None:
    """Return process RSS in MiB when psutil is available."""
    try:
        import psutil
    except Exception:
        return None
    try:
        process = psutil.Process(pid)
        return process.memory_info().rss / float(1024 * 1024)
    except Exception:
        return None


def _sample_process_gpu_memory(
    *,
    pid: int,
    timeout_seconds: float,
) -> AggregatedProcessGpuMemory | None:
    """Collect GPU process memory for one PID when available."""
    if os.name != "nt":
        return None
    try:
        output = _run_powershell_counter_script(
            _POWERSHELL_GPU_PROCESS_MEMORY_SCRIPT,
            timeout_seconds=timeout_seconds,
        )
        return aggregate_process_memory_for_pid(parse_process_memory_rows(output), pid)
    except Exception:
        return None


def _sample_adapter_gpu_memory(
    *,
    timeout_seconds: float,
) -> AggregatedAdapterGpuMemory | None:
    """Collect total GPU adapter memory when available."""
    if os.name != "nt":
        return None
    try:
        output = _run_powershell_counter_script(
            _POWERSHELL_GPU_ADAPTER_MEMORY_SCRIPT,
            timeout_seconds=timeout_seconds,
        )
        return aggregate_adapter_memory(parse_adapter_memory_rows(output))
    except Exception:
        return None


def _run_powershell_counter_script(script: str, *, timeout_seconds: float) -> str:
    """Run one PowerShell counter script and return its stdout."""
    completed = subprocess.run(
        [*_POWERSHELL_PREFIX, script],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "PowerShell GPU counter sampling failed: "
            f"{(completed.stderr or '').strip()[:300]}"
        )
    return completed.stdout.strip()


def write_resource_profile(profile: ResourceProfile, output_path: Path) -> Path:
    """Persist one resource profile as formatted JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile.to_payload(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
