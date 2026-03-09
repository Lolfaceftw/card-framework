"""Artifact helpers for benchmark evidence and provenance."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def utc_now_iso() -> str:
    """Return current UTC timestamp as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(value: Any) -> Any:
    """Convert nested dataclasses and containers into JSON-serializable objects."""
    if is_dataclass(value):
        return to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


def sha256_bytes(payload: bytes) -> str:
    """Compute SHA-256 for raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Compute SHA-256 for a file on disk."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_with_hash(path: Path, payload: Any) -> str:
    """Write JSON payload and return SHA-256 of file bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(to_jsonable(payload), indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(data)
    return sha256_bytes(data)


def git_info(repo_root: Path) -> tuple[str, str]:
    """Return ``(git_commit, git_branch)`` for the repository root."""

    def _run_git(args: list[str]) -> str:
        process = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if process.returncode != 0:
            return "unknown"
        return process.stdout.strip() or "unknown"

    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    return commit, branch


def build_verification_payload(
    *,
    run_id: str,
    report_path: Path,
    report_sha256: str,
    junit_path: Path,
    junit_sha256: str,
    junit_totals: dict[str, int],
    commands_executed: list[str],
    git_commit: str,
    git_branch: str,
) -> dict[str, Any]:
    """Create a machine-readable verification payload for benchmark artifacts."""
    return {
        "status": "not_verified",
        "evidence": {
            "report_json": {
                "path": str(report_path),
                "sha256": report_sha256,
                "producer_command": "uv run python -m benchmark.run execute ...",
            },
            "junit_xml": {
                "path": str(junit_path),
                "sha256": junit_sha256,
                "producer_command": "uv run pytest --junitxml ...",
                "totals": junit_totals,
            },
            "provenance": {
                "run_id": run_id,
                "git_commit": git_commit,
                "git_branch": git_branch,
                "generated_at_utc": utc_now_iso(),
            },
            "commands_executed": commands_executed,
        },
    }
