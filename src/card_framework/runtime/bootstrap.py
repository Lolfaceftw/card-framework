"""Prepare writable runtime assets for installed-package inference."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from typing import Any

from card_framework.audio_pipeline.errors import DependencyMissingError
from card_framework.audio_pipeline.runtime import resolve_command_path
from card_framework.shared.runtime_layout import RuntimeLayout

_MIN_WEIGHT_BYTES = 1_000_000
_WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".ckpt"}
_CTC_FORCED_ALIGNER_COMMIT = "e23e1525bae810f0582b6e539ce7aec63fd01196"
_CTC_FORCED_ALIGNER_ARCHIVE_URL = (
    "https://github.com/MahmoudAshraf97/ctc-forced-aligner/archive/"
    f"{_CTC_FORCED_ALIGNER_COMMIT}.tar.gz"
)


@dataclass(slots=True)
class RuntimeBootstrapError(RuntimeError):
    """Describe a fail-fast runtime bootstrap error."""

    step: str
    message: str
    command: tuple[str, ...] | None = None
    stderr_tail: str = ""

    def __str__(self) -> str:
        lines = [f"[{self.step}] {self.message}"]
        if self.command:
            lines.append(f"Command: {' '.join(self.command)}")
        if self.stderr_tail:
            lines.append(f"Details: {self.stderr_tail}")
        return "\n".join(lines)


def ensure_runtime_requirements(
    *,
    require_uv: bool,
    require_ffmpeg: bool,
    uv_executable: str = "uv",
) -> None:
    """Validate external tools required by runtime bootstrap and execution."""
    missing_tools: list[str] = []
    if require_uv:
        try:
            resolve_uv_executable(uv_executable=uv_executable)
        except RuntimeBootstrapError:
            missing_tools.append("uv")
    if require_ffmpeg:
        try:
            resolve_command_path("ffmpeg")
        except DependencyMissingError:
            missing_tools.append("ffmpeg")
    if missing_tools:
        raise RuntimeBootstrapError(
            step="preflight",
            message=(
                "Missing required command(s): "
                f"{', '.join(missing_tools)}. {build_tool_guidance(missing_tools)}"
            ),
        )


def ensure_index_tts_runtime(
    *,
    layout: RuntimeLayout,
    uv_executable: str = "uv",
    python_executable: str | None = None,
    force_sync: bool = False,
    force_model_download: bool = False,
) -> None:
    """Ensure the writable IndexTTS runtime project and checkpoints are ready."""
    resolved_uv_executable = resolve_uv_executable(uv_executable=uv_executable)
    resolved_python_executable = _resolve_python_executable(
        python_executable=python_executable
    )
    _ensure_runtime_directories(layout=layout)
    source_fingerprints = _project_fingerprints(layout.vendor_source_dir)
    _copy_vendor_project_if_needed(
        layout=layout,
        source_fingerprints=source_fingerprints,
    )
    _sync_vendor_project_if_needed(
        layout=layout,
        uv_executable=resolved_uv_executable,
        python_executable=resolved_python_executable,
        source_fingerprints=source_fingerprints,
        force_sync=force_sync,
    )
    _ensure_model_checkpoints(
        layout=layout,
        uv_executable=resolved_uv_executable,
        force_download=force_model_download,
    )


def ensure_ctc_forced_aligner_runtime(*, python_executable: str) -> None:
    """Install the pinned CTC forced-aligner package when it is missing."""
    if importlib.util.find_spec("ctc_forced_aligner") is not None:
        return

    install_command, install_cwd = _build_python_package_install_command(
        python_executable=python_executable,
        package_spec=_CTC_FORCED_ALIGNER_ARCHIVE_URL,
    )
    _run_cmd(
        step="ctc_forced_aligner_install",
        command=install_command,
        cwd=install_cwd,
    )


def resolve_uv_executable(*, uv_executable: str = "uv") -> str:
    """Resolve the `uv` executable path for packaged runtime bootstrap."""
    normalized = uv_executable.strip() or "uv"

    configured_path = Path(normalized).expanduser()
    if configured_path.is_file():
        return str(configured_path.resolve())

    scripts_dir_value = sysconfig.get_path("scripts")
    if scripts_dir_value:
        scripts_dir = Path(scripts_dir_value)
        candidate_names = [normalized]
        if Path(normalized).suffix.lower() != ".exe":
            candidate_names.append(f"{normalized}.exe")
        for candidate_name in candidate_names:
            candidate_path = scripts_dir / candidate_name
            if candidate_path.is_file():
                return str(candidate_path.resolve())

    resolved_path = shutil.which(normalized)
    if resolved_path is not None:
        return resolved_path

    raise RuntimeBootstrapError(
        step="preflight",
        message=(
            f"Missing required command '{normalized}'. "
            f"{build_tool_guidance(['uv'])}"
        ),
    )


def _resolve_python_executable(*, python_executable: str | None) -> str:
    """Resolve the interpreter path that packaged nested uv commands must reuse."""
    normalized = (python_executable or sys.executable).strip() or sys.executable
    candidate = Path(normalized).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return normalized


def _build_python_package_install_command(
    *,
    python_executable: str,
    package_spec: str,
    uv_executable: str = "uv",
) -> tuple[list[str], Path | None]:
    """Choose a package install command that works in both uv and pip environments."""
    try:
        resolved_uv_executable = resolve_uv_executable(uv_executable=uv_executable)
    except RuntimeBootstrapError:
        return (
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                package_spec,
            ],
            None,
        )

    return (
        [
            resolved_uv_executable,
            "pip",
            "install",
            "--python",
            python_executable,
            package_spec,
        ],
        Path.cwd(),
    )


def build_tool_guidance(missing_tools: Sequence[str]) -> str:
    """Return concise install guidance for missing external tools."""
    guidance_map = {
        "uv": "Install uv into the active environment: `python -m pip install -U uv`.",
        "ffmpeg": (
            "Install FFmpeg and add it to PATH, or let packaged infer use the "
            "`imageio-ffmpeg` fallback."
        ),
    }
    guidance_parts = [guidance_map[tool] for tool in missing_tools if tool in guidance_map]
    return " ".join(guidance_parts)


def checkpoints_ready(checkpoints_dir: Path) -> tuple[bool, str]:
    """Return whether IndexTTS checkpoints appear ready for inference."""
    config_path = checkpoints_dir / "config.yaml"
    if not config_path.exists():
        return False, f"Missing config file: {config_path}"

    for weight_file in checkpoints_dir.rglob("*"):
        if not weight_file.is_file():
            continue
        if weight_file.suffix.lower() not in _WEIGHT_SUFFIXES:
            continue
        try:
            if weight_file.stat().st_size >= _MIN_WEIGHT_BYTES:
                return True, f"Ready with weights: {weight_file.name}"
        except OSError:
            continue
    return False, "No model weight files were found in checkpoints."


def _ensure_runtime_directories(*, layout: RuntimeLayout) -> None:
    """Create the writable runtime roots when missing."""
    layout.runtime_home.mkdir(parents=True, exist_ok=True)
    layout.bootstrap_state_path.parent.mkdir(parents=True, exist_ok=True)
    layout.checkpoints_dir.mkdir(parents=True, exist_ok=True)


def _copy_vendor_project_if_needed(
    *,
    layout: RuntimeLayout,
    source_fingerprints: Mapping[str, str],
) -> None:
    """Copy packaged IndexTTS source into the writable runtime project dir."""
    if not layout.vendor_source_dir.exists():
        raise RuntimeBootstrapError(
            step="vendor_copy",
            message=(
                "Packaged IndexTTS source is missing: "
                f"{layout.vendor_source_dir}"
            ),
        )

    destination_ready = _runtime_project_matches_source(
        runtime_dir=layout.vendor_runtime_dir,
        source_fingerprints=source_fingerprints,
    )
    if destination_ready:
        return

    layout.vendor_runtime_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(
            layout.vendor_source_dir,
            layout.vendor_runtime_dir,
            dirs_exist_ok=True,
        )
    except OSError as exc:
        raise RuntimeBootstrapError(
            step="vendor_copy",
            message=(
                "Failed to copy packaged IndexTTS source into the writable runtime "
                f"home: {layout.vendor_runtime_dir}"
            ),
        ) from exc


def _runtime_project_matches_source(
    *,
    runtime_dir: Path,
    source_fingerprints: Mapping[str, str],
) -> bool:
    """Return whether the extracted runtime project matches packaged source hashes."""
    try:
        runtime_fingerprints = _project_fingerprints(runtime_dir)
    except RuntimeBootstrapError:
        return False
    return (
        runtime_fingerprints["pyproject_hash"] == source_fingerprints["pyproject_hash"]
        and runtime_fingerprints["lock_hash"] == source_fingerprints["lock_hash"]
    )


def _sync_vendor_project_if_needed(
    *,
    layout: RuntimeLayout,
    uv_executable: str,
    python_executable: str,
    source_fingerprints: Mapping[str, str],
    force_sync: bool,
) -> None:
    """Run `uv sync --locked` for the extracted writable IndexTTS project when needed."""
    state = _read_bootstrap_state(layout.bootstrap_state_path)
    vendor_state = state.get("vendor_project", {})
    if not isinstance(vendor_state, dict):
        vendor_state = {}

    runtime_venv_path = layout.vendor_runtime_dir / ".venv"
    needs_sync = (
        force_sync
        or not runtime_venv_path.exists()
        or vendor_state.get("source_pyproject_hash") != source_fingerprints["pyproject_hash"]
        or vendor_state.get("source_lock_hash") != source_fingerprints["lock_hash"]
        or vendor_state.get("python_executable") != python_executable
    )
    if needs_sync:
        _run_cmd(
            step="vendor_sync",
            command=[
                uv_executable,
                "sync",
                "--locked",
                "--python",
                python_executable,
            ],
            cwd=layout.vendor_runtime_dir,
        )

    vendor_state = {
        "source_pyproject_hash": source_fingerprints["pyproject_hash"],
        "source_lock_hash": source_fingerprints["lock_hash"],
        "python_executable": python_executable,
    }
    state["vendor_project"] = vendor_state
    _write_bootstrap_state(layout.bootstrap_state_path, state)


def _ensure_model_checkpoints(
    *,
    layout: RuntimeLayout,
    uv_executable: str,
    force_download: bool,
) -> None:
    """Download IndexTTS model checkpoints into the writable runtime home when missing."""
    ready, _reason = checkpoints_ready(layout.checkpoints_dir)
    if ready and not force_download:
        return

    hf_error: RuntimeBootstrapError | None = None
    try:
        _run_cmd(
            step="model_download",
            command=[
                uv_executable,
                "tool",
                "run",
                "--from",
                "huggingface-hub[cli,hf_xet]",
                "hf",
                "download",
                "IndexTeam/IndexTTS-2",
                "--local-dir",
                str(layout.checkpoints_dir),
                "--repo-type",
                "model",
            ],
            cwd=layout.runtime_home,
        )
        hf_ready, _hf_reason = checkpoints_ready(layout.checkpoints_dir)
        if hf_ready:
            return
    except RuntimeBootstrapError as exc:
        hf_error = exc

    modelscope_error: RuntimeBootstrapError | None = None
    try:
        _run_cmd(
            step="model_download",
            command=[
                uv_executable,
                "tool",
                "run",
                "--from",
                "modelscope",
                "modelscope",
                "download",
                "--model",
                "IndexTeam/IndexTTS-2",
                "--local_dir",
                str(layout.checkpoints_dir),
            ],
            cwd=layout.runtime_home,
        )
    except RuntimeBootstrapError as exc:
        modelscope_error = exc

    final_ready, final_reason = checkpoints_ready(layout.checkpoints_dir)
    if final_ready:
        return

    message = "Failed to provision IndexTTS2 checkpoints. "
    if hf_error is not None:
        message += f"HF failed: {hf_error.message}. "
    if modelscope_error is not None:
        message += f"ModelScope failed: {modelscope_error.message}. "
    message += f"Readiness check: {final_reason}"
    raise RuntimeBootstrapError(step="model_download", message=message)


def _project_fingerprints(project_dir: Path) -> dict[str, str]:
    """Build `pyproject.toml` and `uv.lock` hashes for one project directory."""
    pyproject_path = project_dir / "pyproject.toml"
    lock_path = project_dir / "uv.lock"
    if not pyproject_path.exists():
        raise RuntimeBootstrapError(
            step="fingerprint",
            message=f"Missing pyproject.toml in {project_dir}.",
        )
    if not lock_path.exists():
        raise RuntimeBootstrapError(
            step="fingerprint",
            message=f"Missing uv.lock in {project_dir}.",
        )
    return {
        "pyproject_hash": _sha256_file(pyproject_path),
        "lock_hash": _sha256_file(lock_path),
    }


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_bootstrap_state(path: Path) -> dict[str, Any]:
    """Read JSON bootstrap state, returning an empty mapping on absence or decode failure."""
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_bootstrap_state(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist JSON bootstrap state atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _run_cmd(
    *,
    step: str,
    command: Sequence[str],
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one subprocess command and raise a structured bootstrap error on failure."""
    completed = subprocess.run(
        list(command),
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeBootstrapError(
            step=step,
            message=f"Command failed with exit code {completed.returncode}.",
            command=tuple(command),
            stderr_tail=detail[-1200:],
        )
    return completed
