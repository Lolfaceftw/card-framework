"""All-in-one setup and execution bootstrap for the audio + voice-clone pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Literal, Mapping, Sequence, TypeAlias, cast

REPO_ROOT = Path(__file__).resolve().parent
INDEX_TTS_REPO_URL = "https://github.com/index-tts/index-tts.git"
INDEX_TTS_DIR = REPO_ROOT / "third_party" / "index_tts"
SETUP_STATE_PATH = REPO_ROOT / "artifacts" / "bootstrap" / "setup_state.json"
MIN_WEIGHT_BYTES = 1_000_000
WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".ckpt"}
REQUIRED_REPO_PATHS = (
    "main.py",
    "conf/config.yaml",
    "audio_pipeline/runners/indextts_infer_runner.py",
)
PipelineStartStage: TypeAlias = Literal["stage-1", "stage-2", "stage-3"]


@dataclass(slots=True, frozen=True)
class BootstrapError(RuntimeError):
    """Represent a fail-fast bootstrap error for one setup stage."""

    step: str
    message: str
    command: tuple[str, ...] | None = None
    stderr_tail: str = ""

    def render(self) -> str:
        """Render a user-facing error string with command details."""
        lines = [f"[{self.step}] {self.message}"]
        if self.command:
            lines.append(f"Command: {' '.join(self.command)}")
        if self.stderr_tail:
            lines.append(f"Details: {self.stderr_tail}")
        return "\n".join(lines)

    def __str__(self) -> str:
        """Return message string suitable for exception display and test matching."""
        return self.render()


@dataclass(slots=True, frozen=True)
class StepRecord:
    """Track one step outcome for final summary output."""

    name: str
    status: str
    detail: str


@dataclass(slots=True, frozen=True)
class RepoSyncResult:
    """Result metadata for IndexTTS repository synchronization."""

    cloned: bool
    updated: bool
    pull_skipped_dirty: bool
    lfs_pulled: bool


@dataclass(slots=True, frozen=True)
class ModelProvisionResult:
    """Result metadata for IndexTTS model checkpoint provisioning."""

    downloaded: bool
    source: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse bootstrap CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Setup dependencies/models and run the full stage-1->stage-3 pipeline with "
            "speaker samples + IndexTTS2 voice cloning."
        )
    )
    parser.add_argument(
        "--audio-path",
        help="Source audio path for pipeline.start_stage=stage-1.",
    )
    parser.add_argument(
        "--voiceclone-from-summary",
        dest="voiceclone_from_summary",
        help=(
            "Shortcut mode. Skip stage-1/stage-2 and run stage-3 voice clone "
            "directly from an existing summary XML file."
        ),
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Pass through one extra Hydra override. Repeat this flag for multiple "
            "values, for example: --override pipeline.start_stage=stage-2"
        ),
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Run setup checks/sync/download only and skip pipeline execution.",
    )
    parser.add_argument(
        "--skip-repo-update",
        action="store_true",
        help="Skip update pull for existing third_party/index_tts checkout.",
    )
    parser.add_argument(
        "--force-sync",
        action="store_true",
        help="Force `uv sync --locked` for both projects even when state is unchanged.",
    )
    parser.add_argument(
        "--force-model-download",
        action="store_true",
        help="Force IndexTTS2 model download even if checkpoints appear ready.",
    )
    parser.add_argument(
        "--uv-executable",
        default="uv",
        help="Executable for uv commands (default: uv).",
    )
    parser.add_argument(
        "--git-executable",
        default="git",
        help="Executable for git commands (default: git).",
    )
    return parser.parse_args(argv)


def utc_now_compact() -> str:
    """Return UTC timestamp suitable for run IDs."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    """Return UTC ISO-8601 timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_cmd(
    *,
    step: str,
    command: Sequence[str],
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    stream_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """
    Execute a shell command and raise ``BootstrapError`` on non-zero exit code.

    Args:
        step: Logical step name for error reporting.
        command: Argument list for subprocess execution.
        cwd: Optional working directory.
        env: Optional environment mapping.
        stream_output: Stream child process stdout/stderr to terminal in real time.

    Returns:
        Completed process object.

    Raises:
        BootstrapError: If command execution fails.
    """
    if stream_output:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            text=True,
            env=dict(env) if env is not None else None,
        )
    else:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd) if cwd is not None else None,
            check=False,
            capture_output=True,
            text=True,
            env=dict(env) if env is not None else None,
        )
    if completed.returncode != 0:
        if stream_output:
            detail = "See streamed command output above for details."
        else:
            detail = (completed.stderr or completed.stdout or "").strip()
        raise BootstrapError(
            step=step,
            message=f"Command failed with exit code {completed.returncode}.",
            command=tuple(command),
            stderr_tail=detail[-1200:],
        )
    return completed


def check_prerequisites(*, git_executable: str) -> None:
    """
    Validate system and repository prerequisites before setup begins.

    Args:
        git_executable: Git binary name/path.

    Raises:
        BootstrapError: If tools or required repo paths are missing.
    """
    required_tools = ("uv", "git", "ffmpeg")
    missing_tools = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing_tools:
        raise BootstrapError(
            step="preflight",
            message=(
                "Missing required command(s): "
                f"{', '.join(missing_tools)}. {build_tool_guidance(missing_tools)}"
            ),
        )

    try:
        run_cmd(
            step="preflight",
            command=[git_executable, "lfs", "version"],
            cwd=REPO_ROOT,
        )
    except BootstrapError as exc:
        raise BootstrapError(
            step="preflight",
            message="Git LFS is required but unavailable. Install Git LFS and re-run.",
            command=exc.command,
            stderr_tail=exc.stderr_tail,
        ) from exc

    missing_paths = [
        rel_path for rel_path in REQUIRED_REPO_PATHS if not (REPO_ROOT / rel_path).exists()
    ]
    if missing_paths:
        raise BootstrapError(
            step="preflight",
            message=(
                "Repository is missing required path(s): "
                f"{', '.join(missing_paths)}. Run from repository root."
            ),
        )


def build_tool_guidance(missing_tools: Sequence[str]) -> str:
    """Return concise Windows-first install guidance for missing tools."""
    guidance_map = {
        "uv": "Install uv: `pip install -U uv`.",
        "git": "Install Git for Windows: https://git-scm.com/download/win",
        "ffmpeg": "Install FFmpeg and add it to PATH: https://www.gyan.dev/ffmpeg/builds/",
    }
    guidance_parts = [guidance_map[tool] for tool in missing_tools if tool in guidance_map]
    return " ".join(guidance_parts)


def ensure_indextts_repo(*, git_executable: str, skip_update: bool) -> RepoSyncResult:
    """
    Ensure ``third_party/index_tts`` exists, is up to date, and has LFS artifacts.

    Args:
        git_executable: Git binary name/path.
        skip_update: Skip update pull for existing checkout.

    Returns:
        Repository sync result metadata.

    Raises:
        BootstrapError: If clone/fetch/pull/lfs steps fail.
    """
    cloned = False
    updated = False
    pull_skipped_dirty = False

    if not INDEX_TTS_DIR.exists():
        INDEX_TTS_DIR.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(
            step="repo_sync",
            command=[git_executable, "clone", INDEX_TTS_REPO_URL, str(INDEX_TTS_DIR)],
            cwd=REPO_ROOT,
        )
        cloned = True
    elif not (INDEX_TTS_DIR / ".git").exists():
        raise BootstrapError(
            step="repo_sync",
            message=(
                "IndexTTS directory exists but is not a git repository: "
                f"{INDEX_TTS_DIR}"
            ),
        )

    if not skip_update and not cloned:
        if _is_git_dirty(git_executable=git_executable, repo_dir=INDEX_TTS_DIR):
            pull_skipped_dirty = True
        else:
            run_cmd(
                step="repo_sync",
                command=[git_executable, "fetch", "--all", "--prune"],
                cwd=INDEX_TTS_DIR,
            )
            run_cmd(
                step="repo_sync",
                command=[git_executable, "pull", "--ff-only"],
                cwd=INDEX_TTS_DIR,
            )
            updated = True

    run_cmd(
        step="repo_sync",
        command=[git_executable, "lfs", "pull"],
        cwd=INDEX_TTS_DIR,
    )
    return RepoSyncResult(
        cloned=cloned,
        updated=updated,
        pull_skipped_dirty=pull_skipped_dirty,
        lfs_pulled=True,
    )


def _is_git_dirty(*, git_executable: str, repo_dir: Path) -> bool:
    """Return whether a git working tree contains tracked/untracked changes."""
    result = run_cmd(
        step="repo_sync",
        command=[git_executable, "status", "--porcelain"],
        cwd=repo_dir,
    )
    return bool(result.stdout.strip())


def smart_sync_projects(*, uv_executable: str, force_sync: bool) -> tuple[str, ...]:
    """
    Run ``uv sync --locked`` for project roots only when required.

    Args:
        uv_executable: uv binary name/path.
        force_sync: Force sync regardless of state.

    Returns:
        Tuple of project keys that were synced.

    Raises:
        BootstrapError: If required project files are missing or sync command fails.
    """
    project_specs = (
        ("root", REPO_ROOT),
        ("index_tts", INDEX_TTS_DIR),
    )
    state = _read_setup_state()
    projects_state = state.get("projects", {})
    if not isinstance(projects_state, dict):
        projects_state = {}

    synced: list[str] = []
    for key, project_dir in project_specs:
        fingerprints = _project_fingerprints(project_dir=project_dir)
        previous = projects_state.get(key, {})
        if not isinstance(previous, dict):
            previous = {}
        venv_exists = (project_dir / ".venv").exists()
        needs_sync = (
            force_sync
            or not venv_exists
            or previous.get("pyproject_hash") != fingerprints["pyproject_hash"]
            or previous.get("lock_hash") != fingerprints["lock_hash"]
        )
        if needs_sync:
            run_cmd(
                step="dependency_sync",
                command=[uv_executable, "sync", "--locked"],
                cwd=project_dir,
            )
            synced.append(key)
        projects_state[key] = fingerprints

    state["projects"] = projects_state
    state["generated_at_utc"] = utc_now_iso()
    _write_setup_state(state)
    return tuple(synced)


def _project_fingerprints(*, project_dir: Path) -> dict[str, str]:
    """Build hash fingerprints for ``pyproject.toml`` and ``uv.lock``."""
    pyproject_path = project_dir / "pyproject.toml"
    lock_path = project_dir / "uv.lock"
    if not pyproject_path.exists():
        raise BootstrapError(
            step="dependency_sync",
            message=f"Missing pyproject.toml in {project_dir}.",
        )
    if not lock_path.exists():
        raise BootstrapError(
            step="dependency_sync",
            message=f"Missing uv.lock in {project_dir}.",
        )
    return {
        "pyproject_hash": _sha256_file(pyproject_path),
        "lock_hash": _sha256_file(lock_path),
    }


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_setup_state() -> dict[str, Any]:
    """Read setup-state JSON; return empty mapping when absent/invalid."""
    if not SETUP_STATE_PATH.exists():
        return {}
    try:
        payload = json.loads(SETUP_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_setup_state(payload: Mapping[str, Any]) -> None:
    """Persist setup-state JSON atomically."""
    SETUP_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = SETUP_STATE_PATH.with_suffix(".json.tmp")
    temp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(SETUP_STATE_PATH)


def checkpoints_ready(checkpoints_dir: Path) -> tuple[bool, str]:
    """
    Validate whether IndexTTS checkpoints appear complete enough for inference.

    Args:
        checkpoints_dir: Checkpoints directory path.

    Returns:
        Tuple of (is_ready, reason).
    """
    config_path = checkpoints_dir / "config.yaml"
    if not config_path.exists():
        return False, f"Missing config file: {config_path}"

    for weight_file in checkpoints_dir.rglob("*"):
        if not weight_file.is_file():
            continue
        if weight_file.suffix.lower() not in WEIGHT_SUFFIXES:
            continue
        try:
            if weight_file.stat().st_size >= MIN_WEIGHT_BYTES:
                return True, f"Ready with weights: {weight_file.name}"
        except OSError:
            continue
    return False, "No model weight files were found in checkpoints."


def ensure_indextts_model(
    *,
    uv_executable: str,
    force_download: bool,
) -> ModelProvisionResult:
    """
    Ensure IndexTTS2 model artifacts are available (HF first, ModelScope fallback).

    Args:
        uv_executable: uv binary name/path.
        force_download: Force download attempts even when checkpoints look ready.

    Returns:
        Model provisioning metadata.

    Raises:
        BootstrapError: If both download paths fail or checkpoints remain invalid.
    """
    checkpoints_dir = INDEX_TTS_DIR / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    ready, _reason = checkpoints_ready(checkpoints_dir)
    if ready and not force_download:
        return ModelProvisionResult(downloaded=False, source="existing")

    hf_error: BootstrapError | None = None
    try:
        run_cmd(
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
                str(checkpoints_dir),
                "--repo-type",
                "model",
            ],
            cwd=REPO_ROOT,
        )
        hf_ready, _hf_reason = checkpoints_ready(checkpoints_dir)
        if hf_ready:
            return ModelProvisionResult(downloaded=True, source="huggingface")
    except BootstrapError as exc:
        hf_error = exc

    modelscope_error: BootstrapError | None = None
    try:
        run_cmd(
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
                str(checkpoints_dir),
            ],
            cwd=REPO_ROOT,
        )
    except BootstrapError as exc:
        modelscope_error = exc

    final_ready, final_reason = checkpoints_ready(checkpoints_dir)
    if final_ready:
        return ModelProvisionResult(downloaded=True, source="modelscope")

    message = "Failed to provision IndexTTS2 checkpoints. "
    if hf_error is not None:
        message += f"HF failed: {hf_error.message}. "
    if modelscope_error is not None:
        message += f"ModelScope failed: {modelscope_error.message}. "
    message += f"Readiness check: {final_reason}"
    raise BootstrapError(step="model_download", message=message)


def resolve_audio_input(path_value: str) -> Path:
    """
    Resolve and validate one user-provided audio path.

    Args:
        path_value: Raw user input path string.

    Returns:
        Absolute resolved path to an existing file.

    Raises:
        BootstrapError: If value is empty or invalid.
    """
    normalized = path_value.strip().strip('"').strip("'")
    if not normalized:
        raise BootstrapError(step="input", message="Audio path must not be empty.")
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if not candidate.exists():
        raise BootstrapError(step="input", message=f"Audio path does not exist: {candidate}")
    if not candidate.is_file():
        raise BootstrapError(step="input", message=f"Audio path is not a file: {candidate}")
    return candidate


def prompt_or_validate_audio_path(cli_audio_path: str | None) -> Path:
    """
    Resolve audio path from CLI flag or interactive prompt.

    Args:
        cli_audio_path: Optional audio path provided via CLI.

    Returns:
        Validated absolute audio file path.
    """
    if cli_audio_path is not None:
        return resolve_audio_input(cli_audio_path)

    while True:
        raw = input("Enter source audio path: ").strip()
        try:
            return resolve_audio_input(raw)
        except BootstrapError as exc:
            print(exc.render())


def resolve_path_input(path_value: str, *, field_name: str) -> Path:
    """
    Resolve and validate one generic user-provided file path.

    Args:
        path_value: Raw user input path string.
        field_name: Human-readable field label for validation errors.

    Returns:
        Absolute resolved path to an existing file.

    Raises:
        BootstrapError: If value is empty or invalid.
    """
    normalized = path_value.strip().strip('"').strip("'")
    if not normalized:
        raise BootstrapError(step="input", message=f"{field_name} must not be empty.")
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    if not candidate.exists():
        raise BootstrapError(step="input", message=f"{field_name} does not exist: {candidate}")
    if not candidate.is_file():
        raise BootstrapError(step="input", message=f"{field_name} is not a file: {candidate}")
    return candidate


def normalize_cli_overrides(raw_overrides: Sequence[str]) -> tuple[str, ...]:
    """
    Validate and normalize pass-through Hydra CLI overrides.

    Args:
        raw_overrides: Raw override strings from repeated ``--override`` flags.

    Returns:
        Tuple of validated ``KEY=VALUE`` override strings.

    Raises:
        BootstrapError: If any override is empty or malformed.
    """
    normalized: list[str] = []
    for raw_override in raw_overrides:
        candidate = str(raw_override).strip()
        if not candidate:
            raise BootstrapError(
                step="input",
                message="Empty --override value is not allowed. Use KEY=VALUE format.",
            )
        if "=" not in candidate:
            raise BootstrapError(
                step="input",
                message=(
                    f"Invalid --override '{candidate}'. Expected KEY=VALUE "
                    "(example: pipeline.start_stage=stage-2)."
                ),
            )
        key, _value = candidate.split("=", 1)
        if not key.strip():
            raise BootstrapError(
                step="input",
                message=(
                    f"Invalid --override '{candidate}'. Override key must be non-empty."
                ),
            )
        normalized.append(candidate)
    return tuple(normalized)


def build_shortcut_overrides(
    *,
    voiceclone_from_summary: str | None,
    run_id: str,
) -> tuple[str, ...]:
    """
    Build opinionated override bundle shortcuts for common tasks.

    Args:
        voiceclone_from_summary: Optional summary XML path for voice-clone shortcut.
        run_id: UTC run identifier used for generated shortcut artifacts.

    Returns:
        Tuple of Hydra overrides contributed by enabled shortcuts.

    Raises:
        BootstrapError: If shortcut input path is invalid.
    """
    if voiceclone_from_summary is None:
        return ()

    summary_path = resolve_path_input(
        voiceclone_from_summary,
        field_name="Summary XML path",
    )
    speaker_manifest_path = _find_latest_speaker_samples_manifest()
    synthetic_transcript_path = _write_voiceclone_shortcut_transcript(
        run_id=run_id,
        speaker_samples_manifest_path=speaker_manifest_path,
    )
    return (
        "pipeline.start_stage=stage-3",
        f"pipeline.final_summary_path={_path_for_override(summary_path)}",
        f"transcript_path={_path_for_override(synthetic_transcript_path)}",
        "audio.speaker_samples.enabled=false",
        "audio.voice_clone.enabled=true",
    )


def _find_latest_speaker_samples_manifest() -> Path:
    """
    Return the most recent speaker-sample manifest generated by prior runs.

    Raises:
        BootstrapError: If no manifest can be discovered.
    """
    manifests = sorted(
        (REPO_ROOT / "artifacts" / "audio_stage" / "runs").glob(
            "*/speaker_samples/manifest.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not manifests:
        raise BootstrapError(
            step="input",
            message=(
                "No speaker-sample manifest was found under "
                "artifacts/audio_stage/runs/*/speaker_samples/manifest.json. "
                "Run an audio stage first or pass --override transcript_path=<transcript.json> "
                "that contains metadata.speaker_samples_manifest_path."
            ),
        )
    return manifests[0].resolve()


def _write_voiceclone_shortcut_transcript(
    *,
    run_id: str,
    speaker_samples_manifest_path: Path,
) -> Path:
    """
    Write a minimal transcript JSON carrying speaker-manifest metadata.

    Args:
        run_id: UTC run identifier for deterministic output path.
        speaker_samples_manifest_path: Resolved speaker sample manifest path.

    Returns:
        Path to generated transcript JSON.

    Raises:
        BootstrapError: If output cannot be written.
    """
    output_path = (
        REPO_ROOT
        / "artifacts"
        / "bootstrap"
        / "voiceclone_shortcuts"
        / f"{run_id}.transcript.json"
    ).resolve()
    payload = {
        "segments": [],
        "metadata": {
            "speaker_samples_manifest_path": _path_for_override(
                speaker_samples_manifest_path
            )
        },
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise BootstrapError(
            step="input",
            message=f"Failed to write shortcut transcript metadata file: {output_path}",
        ) from exc
    return output_path


def _resolve_last_override_value(
    *,
    overrides: Sequence[str],
    key: str,
    default: str,
) -> str:
    """Resolve effective override value for one key using last-wins ordering."""
    prefix = f"{key}="
    resolved = default
    for override in overrides:
        if override.startswith(prefix):
            resolved = override[len(prefix) :]
    return resolved


def _has_override_key(overrides: Sequence[str], key: str) -> bool:
    """Return whether overrides include an assignment for one key."""
    prefix = f"{key}="
    return any(override.startswith(prefix) for override in overrides)


def resolve_start_stage(overrides: Sequence[str]) -> PipelineStartStage:
    """
    Resolve and validate effective ``pipeline.start_stage`` from overrides.

    Args:
        overrides: Ordered Hydra overrides with last-wins semantics.

    Returns:
        Effective normalized start stage.

    Raises:
        BootstrapError: If the effective stage value is invalid.
    """
    start_stage = _resolve_last_override_value(
        overrides=overrides,
        key="pipeline.start_stage",
        default="stage-1",
    ).strip().lower()
    if start_stage not in {"stage-1", "stage-2", "stage-3"}:
        raise BootstrapError(
            step="input",
            message=(
                "pipeline.start_stage must be one of: "
                "stage-1, stage-2, stage-3."
            ),
        )
    return cast(PipelineStartStage, start_stage)


def requires_audio_path_input(overrides: Sequence[str]) -> bool:
    """
    Return whether runtime audio path input is required for the effective stage plan.

    Args:
        overrides: Ordered Hydra overrides with last-wins semantics.

    Returns:
        ``True`` when effective ``pipeline.start_stage`` is ``stage-1``.
    """
    return resolve_start_stage(overrides) == "stage-1"


def build_run_overrides(
    *,
    run_id: str,
    start_stage: PipelineStartStage = "stage-1",
    audio_path: Path | None = None,
) -> list[str]:
    """
    Build deterministic Hydra overrides for full pipeline + voice cloning run.

    Args:
        run_id: UTC run identifier.
        start_stage: Effective stage that should drive default overrides.
        audio_path: Optional resolved source audio path.

    Returns:
        Ordered list of Hydra override key-value strings.
    """
    run_work_dir = (REPO_ROOT / "artifacts" / "audio_stage" / "runs" / run_id).resolve()
    transcript_path = (REPO_ROOT / "artifacts" / "transcripts" / f"{run_id}.transcript.json")
    transcript_path = transcript_path.resolve()
    overrides = [
        f"pipeline.start_stage={start_stage}",
        f"audio.work_dir={_path_for_override(run_work_dir)}",
        "audio.speaker_samples.enabled=true",
        "audio.speaker_samples.source_audio=vocals",
        "audio.speaker_samples.target_duration_seconds=30",
        "audio.voice_clone.enabled=true",
        "audio.voice_clone.provider=indextts",
        "audio.voice_clone.execution_backend=subprocess",
        "logging.print_to_terminal=true",
        "logging.summarizer_critic_print_to_terminal=false",
        f"audio.voice_clone.runner_project_dir={_path_for_override(INDEX_TTS_DIR)}",
        (
            "audio.voice_clone.cfg_path="
            f"{_path_for_override(INDEX_TTS_DIR / 'checkpoints' / 'config.yaml')}"
        ),
        f"audio.voice_clone.model_dir={_path_for_override(INDEX_TTS_DIR / 'checkpoints')}",
    ]
    if start_stage == "stage-1":
        overrides.extend(
            [
                f"audio.output_transcript_path={_path_for_override(transcript_path)}",
                f"transcript_path={_path_for_override(transcript_path)}",
            ]
        )
    if audio_path is not None:
        overrides.append(f"audio.audio_path={_path_for_override(audio_path)}")
    return overrides


def _find_latest_transcript_path() -> Path:
    """
    Return the most recent transcript artifact JSON from prior runs.

    Raises:
        BootstrapError: If no transcript JSON file can be discovered.
    """
    candidates = sorted(
        (REPO_ROOT / "artifacts" / "transcripts").glob("*.transcript.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise BootstrapError(
            step="input",
            message=(
                "No transcript JSON was found under artifacts/transcripts/*.transcript.json. "
                "Run stage-1 first or pass --override transcript_path=<transcript.json>."
            ),
        )
    return candidates[0].resolve()


def ensure_transcript_override_for_stage(
    *,
    overrides: list[str],
    start_stage: PipelineStartStage,
    run_id: str,
) -> None:
    """
    Ensure non-stage-1 runs have a valid transcript override.

    Args:
        overrides: Mutable ordered override list that will be sent to ``main.py``.
        start_stage: Effective normalized start stage.
        run_id: UTC run identifier used for generated shortcut artifacts.

    Raises:
        BootstrapError: If required fallback artifacts cannot be resolved.
    """
    if start_stage == "stage-1" or _has_override_key(overrides, "transcript_path"):
        return

    if start_stage == "stage-2":
        transcript_path = _find_latest_transcript_path()
        overrides.append(f"transcript_path={_path_for_override(transcript_path)}")
        return

    speaker_manifest_path = _find_latest_speaker_samples_manifest()
    synthetic_transcript_path = _write_voiceclone_shortcut_transcript(
        run_id=run_id,
        speaker_samples_manifest_path=speaker_manifest_path,
    )
    overrides.append(f"transcript_path={_path_for_override(synthetic_transcript_path)}")


def _path_for_override(path: Path) -> str:
    """Return absolute path string safe for Hydra CLI override parsing."""
    return path.resolve().as_posix()


def run_pipeline(*, uv_executable: str, overrides: Sequence[str]) -> None:
    """
    Execute the configured pipeline run.

    Args:
        uv_executable: uv binary name/path.
        overrides: Hydra override values.

    Raises:
        BootstrapError: If pipeline command exits non-zero.
    """
    command = [uv_executable, "run", "main.py", *overrides]
    run_cmd(
        step="pipeline_run",
        command=command,
        cwd=REPO_ROOT,
        stream_output=True,
    )


def print_summary(
    *,
    records: Sequence[StepRecord],
    run_id: str | None,
    interrupted: bool = False,
) -> None:
    """Print compact step summary and expected output artifact locations."""
    print("\n===== setup_and_run summary =====")
    for record in records:
        print(f"[{record.status}] {record.name}: {record.detail}")
    if interrupted:
        print("[info] Execution interrupted by user.")
    if run_id is None:
        return

    run_work_dir = (REPO_ROOT / "artifacts" / "audio_stage" / "runs" / run_id).resolve()
    transcript_path = (REPO_ROOT / "artifacts" / "transcripts" / f"{run_id}.transcript.json")
    transcript_path = transcript_path.resolve()
    speaker_manifest = run_work_dir / "speaker_samples" / "manifest.json"
    voice_clone_manifest = run_work_dir / "voice_clone" / "manifest.json"

    print(f"run_id: {run_id}")
    print(f"work_dir: {run_work_dir}")
    print(f"transcript: {transcript_path}")
    print(f"speaker_samples_manifest: {speaker_manifest}")
    print(f"voice_clone_manifest: {voice_clone_manifest}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run setup phases and, by default, execute the full pipeline."""
    args = parse_args(argv)
    records: list[StepRecord] = []
    run_id: str | None = None

    try:
        print("[1/6] Preflight checks")
        check_prerequisites(git_executable=args.git_executable)
        records.append(StepRecord(name="preflight", status="ok", detail="Tools and paths validated."))

        print("[2/6] IndexTTS repository sync")
        repo_result = ensure_indextts_repo(
            git_executable=args.git_executable,
            skip_update=bool(args.skip_repo_update),
        )
        repo_detail = (
            f"cloned={repo_result.cloned}, updated={repo_result.updated}, "
            f"pull_skipped_dirty={repo_result.pull_skipped_dirty}, "
            f"lfs_pulled={repo_result.lfs_pulled}"
        )
        records.append(StepRecord(name="repo_sync", status="ok", detail=repo_detail))

        print("[3/6] Dependency sync")
        synced_projects = smart_sync_projects(
            uv_executable=args.uv_executable,
            force_sync=bool(args.force_sync),
        )
        if synced_projects:
            detail = f"synced projects: {', '.join(synced_projects)}"
        else:
            detail = "sync skipped (fingerprints unchanged)"
        records.append(StepRecord(name="dependency_sync", status="ok", detail=detail))

        print("[4/6] Model provisioning")
        model_result = ensure_indextts_model(
            uv_executable=args.uv_executable,
            force_download=bool(args.force_model_download),
        )
        model_detail = (
            f"downloaded={model_result.downloaded}, source={model_result.source}"
        )
        records.append(StepRecord(name="model_download", status="ok", detail=model_detail))

        if args.setup_only:
            print("[5/6] Setup-only mode enabled; skipping pipeline execution.")
            records.append(
                StepRecord(
                    name="pipeline_run",
                    status="skipped",
                    detail="Skipped due to --setup-only.",
                )
            )
            print_summary(records=records, run_id=None)
            return 0

        print("[5/6] Resolve runtime input")
        run_id = utc_now_compact()
        shortcut_overrides = build_shortcut_overrides(
            voiceclone_from_summary=args.voiceclone_from_summary,
            run_id=run_id,
        )
        pass_through_overrides = normalize_cli_overrides(args.override)
        effective_start_stage = resolve_start_stage(
            [*shortcut_overrides, *pass_through_overrides]
        )
        base_overrides = build_run_overrides(
            run_id=run_id,
            start_stage=effective_start_stage,
        )
        overrides = [*base_overrides, *shortcut_overrides, *pass_through_overrides]
        ensure_transcript_override_for_stage(
            overrides=overrides,
            start_stage=effective_start_stage,
            run_id=run_id,
        )
        if requires_audio_path_input(overrides):
            audio_path = prompt_or_validate_audio_path(args.audio_path)
            overrides.append(f"audio.audio_path={_path_for_override(audio_path)}")
            records.append(
                StepRecord(name="input", status="ok", detail=f"audio_path={audio_path}")
            )
        elif args.audio_path is not None:
            audio_path = resolve_audio_input(args.audio_path)
            overrides.append(f"audio.audio_path={_path_for_override(audio_path)}")
            records.append(
                StepRecord(
                    name="input",
                    status="ok",
                    detail=f"audio_path={audio_path} (optional)",
                )
            )
        else:
            records.append(
                StepRecord(
                    name="input",
                    status="ok",
                    detail="audio_path=not_required_for_selected_stage",
                )
            )

        print("[6/6] Running pipeline")
        run_pipeline(uv_executable=args.uv_executable, overrides=overrides)
        records.append(
            StepRecord(
                name="pipeline_run",
                status="ok",
                detail="main.py completed successfully.",
            )
        )
        print_summary(records=records, run_id=run_id)
        return 0
    except BootstrapError as exc:
        records.append(StepRecord(name=exc.step, status="failed", detail=exc.message))
        print(exc.render())
        print_summary(records=records, run_id=run_id)
        return 1
    except KeyboardInterrupt:
        print_summary(records=records, run_id=run_id, interrupted=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
