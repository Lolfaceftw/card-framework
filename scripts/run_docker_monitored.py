from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def get_host_memory_sample() -> dict[str, float] | None:
    try:
        vm = psutil.virtual_memory()
    except Exception:
        return None
    return {
        "total_mb": round(vm.total / (1024 * 1024), 2),
        "available_mb": round(vm.available / (1024 * 1024), 2),
        "used_mb": round(vm.used / (1024 * 1024), 2),
        "used_ratio": round(vm.percent / 100.0, 4),
    }


def get_gpu_samples() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    rows: list[dict[str, object]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_gpu_pct": int(parts[2]),
                    "utilization_memory_pct": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                    "memory_total_mb": int(parts[5]),
                }
            )
        except ValueError:
            continue
    return rows


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--args-json", required=True)
    parser.add_argument("--stdout-path", required=True)
    parser.add_argument("--stderr-path", required=True)
    parser.add_argument("--monitor-path", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--sample-interval-seconds", type=float, default=5.0)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    args_json = Path(args.args_json).resolve()
    stdout_path = Path(args.stdout_path).resolve()
    stderr_path = Path(args.stderr_path).resolve()
    monitor_path = Path(args.monitor_path).resolve()
    status_path = Path(args.status_path).resolve()

    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    monitor_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    docker_args = json.loads(args_json.read_text(encoding="utf-8-sig"))
    if not isinstance(docker_args, list) or not all(
        isinstance(item, str) for item in docker_args
    ):
        raise SystemExit("args-json must contain a JSON string list")

    started_at = now_iso()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        proc = subprocess.Popen(
            ["docker", *docker_args],
            cwd=str(repo_root),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
        while proc.poll() is None:
            append_jsonl(
                monitor_path,
                {
                    "timestamp": now_iso(),
                    "pid": proc.pid,
                    "run_state": "running",
                    "host_memory": get_host_memory_sample(),
                    "gpus": get_gpu_samples(),
                },
            )
            time.sleep(max(args.sample_interval_seconds, 1.0))
        exit_code = int(proc.returncode or 0)

    append_jsonl(
        monitor_path,
        {
            "timestamp": now_iso(),
            "pid": proc.pid,
            "run_state": "finished",
            "exit_code": exit_code,
            "host_memory": get_host_memory_sample(),
            "gpus": get_gpu_samples(),
        },
    )
    status_path.write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "exit_code": exit_code,
                "started_at": started_at,
                "finished_at": now_iso(),
                "stdout_path": str(stdout_path),
                "stderr_path": str(stderr_path),
                "monitor_path": str(monitor_path),
                "args_json": str(args_json),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
