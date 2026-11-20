"""RAM and GPU telemetry helpers for supervised benchmark sessions."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from card_framework.benchmark.artifacts import utc_now_iso


@dataclass(slots=True, frozen=True)
class SystemMemorySnapshot:
    """One point-in-time host RAM measurement."""

    total_bytes: int
    available_bytes: int
    percent_used: float


@dataclass(slots=True, frozen=True)
class GpuMemorySnapshot:
    """One point-in-time GPU VRAM measurement."""

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
    """Combined RAM and GPU telemetry sample."""

    timestamp_utc: str
    system_memory: SystemMemorySnapshot
    gpu_memory: tuple[GpuMemorySnapshot, ...]


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
    """Parse ``nvidia-smi`` CSV output into GPU snapshots."""
    snapshots: list[GpuMemorySnapshot] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        snapshots.append(
            GpuMemorySnapshot(
                gpu_index=int(parts[0]),
                name=parts[1],
                total_mebibytes=int(parts[2]),
                used_mebibytes=int(parts[3]),
            )
        )
    return tuple(snapshots)


def system_memory_snapshot() -> SystemMemorySnapshot:
    """Collect current host RAM usage."""
    if os.name == "nt":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError("GlobalMemoryStatusEx failed.")
        return SystemMemorySnapshot(
            total_bytes=int(status.ullTotalPhys),
            available_bytes=int(status.ullAvailPhys),
            percent_used=float(status.dwMemoryLoad),
        )
    total_pages = os.sysconf("SC_PHYS_PAGES")
    available_pages = os.sysconf("SC_AVPHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    total_bytes = int(total_pages * page_size)
    available_bytes = int(available_pages * page_size)
    percent_used = (
        0.0
        if total_bytes <= 0
        else ((total_bytes - available_bytes) / total_bytes) * 100.0
    )
    return SystemMemorySnapshot(
        total_bytes=total_bytes,
        available_bytes=available_bytes,
        percent_used=percent_used,
    )


def gpu_memory_snapshots() -> tuple[GpuMemorySnapshot, ...]:
    """Collect NVIDIA VRAM usage when ``nvidia-smi`` is available."""
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
    return parse_nvidia_smi_memory_csv(completed.stdout)


def collect_telemetry_sample() -> TelemetrySample:
    """Collect one combined telemetry sample."""
    return TelemetrySample(
        timestamp_utc=utc_now_iso(),
        system_memory=system_memory_snapshot(),
        gpu_memory=gpu_memory_snapshots(),
    )


def detect_threshold_breach(
    *,
    sample: TelemetrySample,
    ram_threshold_percent: float,
    gpu_threshold_gibibytes: float,
) -> str | None:
    """Return a threshold breach reason when RAM or VRAM is over limit."""
    if sample.system_memory.percent_used >= ram_threshold_percent:
        return (
            f"System RAM reached {sample.system_memory.percent_used:.1f}% "
            f"(threshold {ram_threshold_percent:.1f}%)."
        )
    for snapshot in sample.gpu_memory:
        if snapshot.used_gibibytes >= gpu_threshold_gibibytes:
            return (
                f"GPU {snapshot.gpu_index} '{snapshot.name}' reached "
                f"{snapshot.used_gibibytes:.2f} GiB (threshold {gpu_threshold_gibibytes:.2f} GiB)."
            )
    return None


def write_telemetry_line(path: Path, sample: TelemetrySample) -> None:
    """Append one telemetry sample to a JSONL file."""
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(sample)) + "\n")


def sleep_with_poll(seconds: float) -> None:
    """Sleep in small increments so supervised loops stay responsive."""
    remaining = max(0.0, seconds)
    while remaining > 0:
        step = min(0.25, remaining)
        time.sleep(step)
        remaining -= step
