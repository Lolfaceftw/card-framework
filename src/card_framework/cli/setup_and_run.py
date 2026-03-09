"""All-in-one setup and execution bootstrap for the staged audio pipeline."""

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

from card_framework.shared.paths import REPO_ROOT

MAIN_MODULE = "card_framework.cli.main"
CALIBRATE_MODULE = "card_framework.cli.calibrate"
SETUP_STATE_PATH = REPO_ROOT / "artifacts" / "bootstrap" / "setup_state.json"
DEFAULT_CONFIG_PATH: Path | None = None
INDEX_TTS_DIR: Path | None = None
INDEX_TTS_CHECKPOINTS_DIR: Path | None = None
MIN_WEIGHT_BYTES = 1_000_000
WEIGHT_SUFFIXES = {".safetensors", ".pt", ".pth", ".bin", ".ckpt"}
PREFERRED_ROOT_AUDIO_FILENAMES = (
    "audio.wav",
    "audio.flac",
    "audio.mp3",
    "audio.m4a",
    "audio.aac",
    "audio.ogg",
    "audio.opus",
)
BASE_REQUIRED_REPO_PATHS = (
    "src/card_framework/cli/main.py",
    "src/card_framework/config/config.yaml",
)
VOICE_CLONE_REQUIRED_REPO_PATHS = (
    "src/card_framework/audio_pipeline/runners/indextts_infer_runner.py",
    "src/card_framework/audio_pipeline/runners/indextts_persistent_runner.py",
    "src/card_framework/_vendor/index_tts/pyproject.toml",
    "src/card_framework/_vendor/index_tts/uv.lock",
)
PipelineStartStage: TypeAlias = Literal["stage-1", "stage-2", "stage-3", "stage-4"]


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
            "Setup dependencies/models and run the staged pipeline with optional "
            "speaker samples, voice cloning, and Stage-4 interjections."
        )
    )
    parser.add_argument(
        "--audio-path",
        help=(
            "Source audio path for pipeline.start_stage=stage-1. When omitted, "
            "the bootstrap auto-uses stage-2 if a reusable transcript JSON is "
            "already present."
        ),
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
        help=(
            "Compatibility flag. Vendored IndexTTS source now lives under "
            "src/card_framework/_vendor and is not updated via nested git pull."
        ),
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


def resolve_repo_config_path() -> Path:
    """Return the active repo config path, preferring the packaged src layout."""
    if DEFAULT_CONFIG_PATH is not None:
        return Path(DEFAULT_CONFIG_PATH).resolve()
    return (REPO_ROOT / "src" / "card_framework" / "config" / "config.yaml").resolve()


def resolve_index_tts_dir() -> Path:
    """Return the active vendored IndexTTS source directory for this repo."""
    if INDEX_TTS_DIR is not None:
        return Path(INDEX_TTS_DIR).resolve()
    return (
        REPO_ROOT / "src" / "card_framework" / "_vendor" / "index_tts"
    ).resolve()


def resolve_index_tts_checkpoints_dir() -> Path:
    """Return the active IndexTTS checkpoints directory for this repo."""
    if INDEX_TTS_CHECKPOINTS_DIR is not None:
        return Path(INDEX_TTS_CHECKPOINTS_DIR).resolve()
    return (REPO_ROOT / "checkpoints" / "index_tts").resolve()


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


def check_prerequisites(
    *,
    git_executable: str,
    require_git: bool = True,
    require_ffmpeg: bool = True,
) -> None:
    """
    Validate system and repository prerequisites before setup begins.

    Args:
        git_executable: Retained for compatibility with older CLI invocations.
        require_git: Whether voice-clone runtime files must be present.
        require_ffmpeg: Whether runtime stages require ffmpeg.

    Raises:
        BootstrapError: If tools or required repo paths are missing.
    """
    del git_executable
    required_tools = ["uv"]
    if require_ffmpeg:
        required_tools.append("ffmpeg")
    missing_tools = [tool for tool in required_tools if shutil.which(tool) is None]
    if missing_tools:
        raise BootstrapError(
            step="preflight",
            message=(
                "Missing required command(s): "
                f"{', '.join(missing_tools)}. {build_tool_guidance(missing_tools)}"
            ),
        )

    required_paths = list(BASE_REQUIRED_REPO_PATHS)
    if require_git:
        required_paths.extend(VOICE_CLONE_REQUIRED_REPO_PATHS)
    missing_paths = [
        rel_path for rel_path in required_paths if not (REPO_ROOT / rel_path).exists()
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
    Ensure the vendored IndexTTS project exists under ``src/card_framework/_vendor``.

    Args:
        git_executable: Retained for compatibility with older call sites.
        skip_update: Retained for compatibility with older call sites.

    Returns:
        Repository sync result metadata.

    Raises:
        BootstrapError: If the vendored project is incomplete.
    """
    del git_executable, skip_update
    index_tts_dir = resolve_index_tts_dir()
    required_paths = (
        index_tts_dir,
        index_tts_dir / "pyproject.toml",
        index_tts_dir / "uv.lock",
        index_tts_dir / "indextts",
    )
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        raise BootstrapError(
            step="repo_sync",
            message=(
                "Vendored IndexTTS source is missing required path(s): "
                + ", ".join(str(path) for path in missing_paths)
            ),
        )
    return RepoSyncResult(
        cloned=False,
        updated=False,
        pull_skipped_dirty=False,
        lfs_pulled=False,
    )


def smart_sync_projects(
    *,
    uv_executable: str,
    force_sync: bool,
    include_index_tts: bool = True,
) -> tuple[str, ...]:
    """
    Run ``uv sync --locked`` for project roots only when required.

    Args:
        uv_executable: uv binary name/path.
        force_sync: Force sync regardless of state.
        include_index_tts: Whether to sync the nested IndexTTS project.

    Returns:
        Tuple of project keys that were synced.

    Raises:
        BootstrapError: If required project files are missing or sync command fails.
    """
    project_specs = [("root", REPO_ROOT)]
    if include_index_tts:
        project_specs.append(("index_tts", resolve_index_tts_dir()))
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
    checkpoints_dir = resolve_index_tts_checkpoints_dir()
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


def _discover_existing_transcript_path() -> Path | None:
    """
    Return the preferred reusable transcript JSON path when one already exists.

    Preference order is:
    1. ``transcript.json`` at the repo root.
    2. The most recently modified ``*.transcript.json`` at the repo root.
    3. The most recently modified ``artifacts/transcripts/*.transcript.json``.
    """
    root_transcript = (REPO_ROOT / "transcript.json").resolve()
    if root_transcript.is_file():
        return root_transcript

    root_candidates = sorted(
        (
            path.resolve()
            for path in REPO_ROOT.glob("*.transcript.json")
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if root_candidates:
        return root_candidates[0]

    artifact_candidates = sorted(
        (
            path.resolve()
            for path in (REPO_ROOT / "artifacts" / "transcripts").glob(
                "*.transcript.json"
            )
            if path.is_file()
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if artifact_candidates:
        return artifact_candidates[0]

    return None


def _discover_existing_audio_path() -> Path | None:
    """Return the preferred reusable root-level source audio path when present."""
    for filename in PREFERRED_ROOT_AUDIO_FILENAMES:
        candidate = (REPO_ROOT / filename).resolve()
        if candidate.is_file():
            return candidate
    return None


def _discover_existing_summary_path() -> Path | None:
    """Return the preferred reusable root-level summary XML path when present."""
    candidate = (REPO_ROOT / "summary.xml").resolve()
    if candidate.is_file():
        return candidate
    return None


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


def resolve_repo_config_boolean(
    *,
    key_path: Sequence[str],
    fallback: bool,
) -> bool:
    """
    Read a boolean default from the repo Hydra config using a stdlib-only parser.

    The bootstrap intentionally stays import-light so it can run before dependency
    synchronization. This helper therefore reads the small YAML subset that the
    repo uses for nested boolean flags instead of importing Hydra or OmegaConf.

    Args:
        key_path: Nested config keys to resolve, for example
            ``("audio", "interjector", "enabled")``.
        fallback: Boolean value returned when the config file is missing, the
            path is absent, or the resolved value is not a boolean scalar.

    Returns:
        Boolean config value when found, otherwise ``fallback``.
    """
    config_path = resolve_repo_config_path()
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError:
        return fallback

    section_stack: list[tuple[int, str]] = []
    for raw_line in config_text.splitlines():
        line_without_comment = raw_line.split("#", 1)[0].rstrip()
        if not line_without_comment.strip():
            continue

        stripped = line_without_comment.lstrip(" ")
        indent = len(line_without_comment) - len(stripped)
        key, separator, raw_value = stripped.partition(":")
        if not separator:
            continue

        while section_stack and indent <= section_stack[-1][0]:
            section_stack.pop()

        key = key.strip()
        value = raw_value.strip()
        current_path = [*(name for _, name in section_stack), key]
        if current_path != list(key_path):
            if not value:
                section_stack.append((indent, key))
            continue

        normalized = value.lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return fallback

    return fallback


def resolve_repo_backed_boolean_override(
    overrides: Sequence[str],
    *,
    key: str,
    repo_key_path: Sequence[str],
    fallback: bool,
) -> bool:
    """
    Resolve a boolean override, defaulting to the repo config when unset.

    Args:
        overrides: Ordered Hydra overrides with last-wins semantics.
        key: Hydra override key to resolve.
        repo_key_path: Nested path inside the packaged application config.
        fallback: Safety fallback used when the repo config cannot be read.

    Returns:
        Effective boolean value after applying CLI overrides on top of the repo
        config default.
    """
    repo_default = resolve_repo_config_boolean(
        key_path=repo_key_path,
        fallback=fallback,
    )
    return resolve_boolean_override(
        overrides,
        key=key,
        default=repo_default,
    )


def resolve_boolean_override(
    overrides: Sequence[str],
    *,
    key: str,
    default: bool,
) -> bool:
    """Resolve a boolean Hydra override using last-wins semantics."""
    raw_value = _resolve_last_override_value(
        overrides=overrides,
        key=key,
        default="true" if default else "false",
    )
    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise BootstrapError(
        step="input",
        message=f"{key} must be a boolean value (for example true or false).",
    )


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
    if start_stage not in {"stage-1", "stage-2", "stage-3", "stage-4"}:
        raise BootstrapError(
            step="input",
            message=(
                "pipeline.start_stage must be one of: "
                "stage-1, stage-2, stage-3, stage-4."
            ),
    )
    return cast(PipelineStartStage, start_stage)


def resolve_bootstrap_start_stage(
    *,
    shortcut_overrides: Sequence[str],
    pass_through_overrides: Sequence[str],
    cli_audio_path: str | None,
) -> PipelineStartStage:
    """
    Resolve the bootstrap start stage, auto-promoting to stage-2 when safe.

    Explicit ``pipeline.start_stage`` overrides always win. When the caller did
    not request a stage and did not provide ``--audio-path``, an already
    available transcript JSON promotes the bootstrap from stage-1 to stage-2.

    Args:
        shortcut_overrides: Wrapper-contributed overrides such as shortcut modes.
        pass_through_overrides: Raw user-provided Hydra overrides.
        cli_audio_path: Optional ``--audio-path`` CLI value.

    Returns:
        Effective pipeline start stage for this bootstrap invocation.
    """
    combined_overrides = [*shortcut_overrides, *pass_through_overrides]
    if _has_override_key(combined_overrides, "pipeline.start_stage"):
        return resolve_start_stage(combined_overrides)
    if _has_override_key(combined_overrides, "transcript_path"):
        return "stage-2"
    if cli_audio_path is not None:
        return "stage-1"
    if _discover_existing_transcript_path() is not None:
        return "stage-2"
    return "stage-1"


def build_start_stage_selection_detail(
    *,
    start_stage: PipelineStartStage,
    voiceclone_from_summary: str | None,
    pass_through_overrides: Sequence[str],
    cli_audio_path: str | None,
) -> str:
    """
    Describe the effective bootstrap start stage for operator-facing logging.

    Args:
        start_stage: Resolved effective stage.
        voiceclone_from_summary: Optional shortcut summary path from CLI.
        pass_through_overrides: Raw user-provided Hydra overrides.
        cli_audio_path: Optional ``--audio-path`` CLI value.

    Returns:
        Short human-readable explanation of stage selection.
    """
    if voiceclone_from_summary is not None:
        summary_path = resolve_path_input(
            voiceclone_from_summary,
            field_name="Voice clone summary path",
        )
        return (
            f"pipeline.start_stage={start_stage} via --voiceclone-from-summary "
            f"using {summary_path}."
        )

    if _has_override_key(pass_through_overrides, "pipeline.start_stage"):
        return f"pipeline.start_stage={start_stage} from explicit override."

    if _has_override_key(pass_through_overrides, "transcript_path"):
        detail = (
            f"pipeline.start_stage={start_stage} because transcript_path override "
            "was provided."
        )
    elif cli_audio_path is not None:
        detail = (
            f"pipeline.start_stage={start_stage} because --audio-path was provided."
        )
    else:
        discovered_transcript_path = _discover_existing_transcript_path()
        if start_stage == "stage-2" and discovered_transcript_path is not None:
            detail = (
                f"pipeline.start_stage={start_stage} because reusable transcript "
                f"was found at {discovered_transcript_path}."
            )
        else:
            detail = (
                f"pipeline.start_stage={start_stage} because no reusable transcript "
                "was found."
            )

    existing_summary_path = _discover_existing_summary_path()
    if (
        start_stage == "stage-2"
        and existing_summary_path is not None
        and voiceclone_from_summary is None
    ):
        detail += (
            " This run will rerun summarization before voice cloning. To clone the "
            "existing summary directly, use "
            f"--voiceclone-from-summary {_path_for_override(existing_summary_path)}."
        )
    return detail


def build_calibration_warning_message(
    *,
    start_stage: PipelineStartStage,
) -> str | None:
    """
    Return an operator hint when calibration may resemble voice cloning.

    Args:
        start_stage: Effective normalized pipeline start stage.

    Returns:
        One-line warning message or ``None`` when not needed.
    """
    if start_stage == "stage-2":
        return (
            "Calibration may emit IndexTTS inference logs here; those are temporary "
            "calibration phrase syntheses before summarization, not the final "
            "voice-clone stage."
        )
    if start_stage == "stage-3":
        return (
            "Calibration may emit IndexTTS inference logs here; those are temporary "
            "calibration phrase syntheses before the stage-3 voice-clone pass."
        )
    return None


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
    enable_voice_clone: bool = True,
    enable_interjector: bool = False,
) -> list[str]:
    """
    Build deterministic Hydra overrides for full pipeline + voice cloning run.

    Args:
        run_id: UTC run identifier.
        start_stage: Effective stage that should drive default overrides.
        audio_path: Optional resolved source audio path.
        enable_voice_clone: Whether wrapper defaults should enable stage-3 voice cloning.
        enable_interjector: Whether wrapper defaults should enable stage-4 interjection.

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
        f"audio.interjector.enabled={'true' if enable_interjector else 'false'}",
        "logging.print_to_terminal=true",
    ]
    if enable_voice_clone:
        index_tts_dir = resolve_index_tts_dir()
        checkpoints_dir = resolve_index_tts_checkpoints_dir()
        overrides.extend(
            [
                "audio.voice_clone.enabled=true",
                "audio.voice_clone.provider=indextts",
                "audio.voice_clone.execution_backend=subprocess",
                f"audio.voice_clone.runner_project_dir={_path_for_override(index_tts_dir)}",
                (
                    "audio.voice_clone.cfg_path="
                    f"{_path_for_override(checkpoints_dir / 'config.yaml')}"
                ),
                (
                    "audio.voice_clone.model_dir="
                    f"{_path_for_override(checkpoints_dir)}"
                ),
            ]
        )
    else:
        overrides.append("audio.voice_clone.enabled=false")
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
    candidate = _discover_existing_transcript_path()
    if candidate is None:
        raise BootstrapError(
            step="input",
            message=(
                "No transcript JSON was found at the repo root "
                "(transcript.json or *.transcript.json) or under "
                "artifacts/transcripts/*.transcript.json. "
                "Run stage-1 first or pass --override transcript_path=<transcript.json>."
            ),
        )
    return candidate


def ensure_transcript_override_for_stage(
    *,
    overrides: list[str],
    start_stage: PipelineStartStage,
    run_id: str,
) -> None:
    """
    Ensure non-stage-1 runs have a valid transcript override.

    Args:
        overrides: Mutable ordered override list that will be sent to
            ``card_framework.cli.main``.
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

    if start_stage == "stage-4":
        return

    speaker_manifest_path = _find_latest_speaker_samples_manifest()
    synthetic_transcript_path = _write_voiceclone_shortcut_transcript(
        run_id=run_id,
        speaker_samples_manifest_path=speaker_manifest_path,
    )
    overrides.append(f"transcript_path={_path_for_override(synthetic_transcript_path)}")


def resolve_calibration_transcript_path(
    *,
    overrides: Sequence[str],
    start_stage: PipelineStartStage,
) -> Path | None:
    """
    Resolve the transcript input that calibration may reuse for non-stage-1 runs.

    Args:
        overrides: Ordered Hydra override list with last-wins semantics.
        start_stage: Effective normalized pipeline start stage.

    Returns:
        Resolved transcript path when calibration should consume one, otherwise
        ``None``.
    """
    if start_stage in {"stage-1", "stage-4"}:
        return None
    transcript_override = _resolve_last_override_value(
        overrides=overrides,
        key="transcript_path",
        default="",
    ).strip()
    if not transcript_override:
        return None
    return resolve_path_input(transcript_override, field_name="Transcript path")


def resolve_transcript_override_path(*, overrides: Sequence[str]) -> Path | None:
    """Resolve the effective ``transcript_path`` override when one is present."""
    transcript_override = _resolve_last_override_value(
        overrides=overrides,
        key="transcript_path",
        default="",
    ).strip()
    if not transcript_override:
        return None
    return resolve_path_input(transcript_override, field_name="Transcript path")


def resolve_audio_override_path(*, overrides: Sequence[str]) -> Path | None:
    """Resolve the effective ``audio.audio_path`` override when one is present."""
    audio_override = _resolve_last_override_value(
        overrides=overrides,
        key="audio.audio_path",
        default="",
    ).strip()
    if not audio_override:
        return None
    return resolve_audio_input(audio_override)


def transcript_has_usable_vocals_audio_path(transcript_path: Path) -> bool:
    """
    Return whether transcript metadata already points at an existing vocals audio.

    Args:
        transcript_path: Transcript JSON file to inspect.

    Returns:
        ``True`` when metadata contains an existing ``vocals_audio_path`` entry.

    Raises:
        BootstrapError: If the transcript JSON cannot be read or decoded.
    """
    candidate = resolve_transcript_metadata_path(
        transcript_path=transcript_path,
        metadata_key="vocals_audio_path",
    )
    return candidate is not None and candidate.is_file()


def transcript_has_usable_speaker_samples_manifest(transcript_path: Path) -> bool:
    """
    Return whether transcript metadata already points at an existing sample manifest.

    Args:
        transcript_path: Transcript JSON file to inspect.

    Returns:
        ``True`` when metadata contains an existing
        ``speaker_samples_manifest_path`` entry.

    Raises:
        BootstrapError: If the transcript JSON cannot be read or decoded.
    """
    candidate = resolve_transcript_metadata_path(
        transcript_path=transcript_path,
        metadata_key="speaker_samples_manifest_path",
    )
    return candidate is not None and candidate.is_file()


def resolve_transcript_source_audio_path(transcript_path: Path) -> Path | None:
    """
    Resolve an existing source audio path from transcript metadata when present.

    Args:
        transcript_path: Transcript JSON file to inspect.

    Returns:
        Existing resolved source audio path, otherwise ``None``.

    Raises:
        BootstrapError: If the transcript JSON cannot be read or decoded.
    """
    candidate = resolve_transcript_metadata_path(
        transcript_path=transcript_path,
        metadata_key="source_audio_path",
    )
    if candidate is None or not candidate.is_file():
        return None
    return candidate


def resolve_transcript_metadata_path(
    *,
    transcript_path: Path,
    metadata_key: str,
) -> Path | None:
    """
    Resolve one optional transcript metadata path relative to the repo root.

    Args:
        transcript_path: Transcript JSON file to inspect.
        metadata_key: Metadata key containing the path-like value.

    Returns:
        Resolved path when the metadata entry is present, otherwise ``None``.

    Raises:
        BootstrapError: If the transcript JSON cannot be read or decoded.
    """
    metadata = _load_transcript_metadata(transcript_path)
    raw_value = str(metadata.get(metadata_key, "")).strip()
    if not raw_value:
        return None
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    return candidate


def _load_transcript_metadata(transcript_path: Path) -> dict[str, object]:
    """Read transcript metadata payload using UTF-8 BOM-tolerant decoding."""
    try:
        payload = json.loads(transcript_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError(
            step="input",
            message=f"Failed to read transcript metadata: {transcript_path}",
        ) from exc
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        return {}
    return metadata


def stage_two_requires_audio_fallback(
    *,
    transcript_path: Path | None,
    speaker_samples_enabled: bool,
) -> bool:
    """
    Return whether a stage-2 transcript still needs configured source audio.

    Args:
        transcript_path: Effective transcript JSON path for the run.
        speaker_samples_enabled: Whether the pipeline will generate speaker samples.

    Returns:
        ``True`` when stage-2 must bootstrap fresh speaker samples from source
        audio because no reusable speaker-sample manifest exists yet.
    """
    if not speaker_samples_enabled or transcript_path is None:
        return False
    return not transcript_has_usable_speaker_samples_manifest(transcript_path)


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
    command = [uv_executable, "run", "python", "-m", MAIN_MODULE, *overrides]
    run_cmd(
        step="pipeline_run",
        command=command,
        cwd=REPO_ROOT,
        stream_output=True,
    )


def run_calibration(
    *,
    uv_executable: str,
    transcript_path: Path | None = None,
    audio_path: Path | None = None,
    speaker_samples_manifest_path: Path | None = None,
    force: bool = False,
) -> None:
    """Run the dedicated calibration helper with resolved inputs."""
    command = [uv_executable, "run", "python", "-m", CALIBRATE_MODULE]
    if speaker_samples_manifest_path is not None:
        command.extend(
            [
                "--speaker-samples-manifest",
                _path_for_override(speaker_samples_manifest_path),
            ]
        )
    if transcript_path is not None:
        command.extend(["--transcript-path", _path_for_override(transcript_path)])
    if audio_path is not None:
        command.extend(["--audio-path", _path_for_override(audio_path)])
    if force:
        command.append("--force")
    run_cmd(
        step="calibration",
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
    interjector_manifest = run_work_dir / "interjector" / "interjector_manifest.json"

    print(f"run_id: {run_id}")
    print(f"work_dir: {run_work_dir}")
    print(f"transcript: {transcript_path}")
    print(f"speaker_samples_manifest: {speaker_manifest}")
    print(f"voice_clone_manifest: {voice_clone_manifest}")
    print(f"interjector_manifest: {interjector_manifest}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run setup phases and, by default, execute the full pipeline."""
    args = parse_args(argv)
    records: list[StepRecord] = []
    run_id: str | None = None

    try:
        pass_through_overrides = normalize_cli_overrides(args.override)
        preflight_shortcut_overrides: list[str] = []
        if args.voiceclone_from_summary is not None:
            preflight_shortcut_overrides.extend(
                [
                    "pipeline.start_stage=stage-3",
                    "audio.speaker_samples.enabled=false",
                    "audio.voice_clone.enabled=true",
                ]
            )
        effective_start_stage = resolve_bootstrap_start_stage(
            shortcut_overrides=preflight_shortcut_overrides,
            pass_through_overrides=pass_through_overrides,
            cli_audio_path=args.audio_path,
        )
        combined_bootstrap_overrides = [
            *preflight_shortcut_overrides,
            *pass_through_overrides,
        ]
        voice_clone_enabled = resolve_repo_backed_boolean_override(
            combined_bootstrap_overrides,
            key="audio.voice_clone.enabled",
            repo_key_path=("audio", "voice_clone", "enabled"),
            fallback=True,
        )
        live_draft_audio_enabled = resolve_repo_backed_boolean_override(
            combined_bootstrap_overrides,
            key="audio.voice_clone.live_drafting.enabled",
            repo_key_path=("audio", "voice_clone", "live_drafting", "enabled"),
            fallback=True,
        )
        live_draft_audio_enabled = voice_clone_enabled and live_draft_audio_enabled
        interjector_enabled = resolve_repo_backed_boolean_override(
            combined_bootstrap_overrides,
            key="audio.interjector.enabled",
            repo_key_path=("audio", "interjector", "enabled"),
            fallback=False,
        )
        calibration_runtime_enabled = (
            effective_start_stage in {"stage-1", "stage-2"}
            and not live_draft_audio_enabled
        )
        synthesis_enabled = (
            voice_clone_enabled or interjector_enabled or calibration_runtime_enabled
        )
        speaker_samples_enabled = resolve_repo_backed_boolean_override(
            combined_bootstrap_overrides,
            key="audio.speaker_samples.enabled",
            repo_key_path=("audio", "speaker_samples", "enabled"),
            fallback=True,
        )
        ffmpeg_required = (
            effective_start_stage == "stage-1"
            or speaker_samples_enabled
            or synthesis_enabled
        )

        print("[1/6] Preflight checks")
        check_prerequisites(
            git_executable=args.git_executable,
            require_git=synthesis_enabled,
            require_ffmpeg=ffmpeg_required,
        )
        records.append(StepRecord(name="preflight", status="ok", detail="Tools and paths validated."))

        print("[2/6] IndexTTS repository sync")
        if synthesis_enabled:
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
        else:
            records.append(
                StepRecord(
                    name="repo_sync",
                    status="skipped",
                    detail=(
                        "Skipped because voice-clone runtime is not required for the "
                        "selected stage and overrides."
                    ),
                )
            )

        print("[3/6] Dependency sync")
        synced_projects = smart_sync_projects(
            uv_executable=args.uv_executable,
            force_sync=bool(args.force_sync),
            include_index_tts=synthesis_enabled,
        )
        if synced_projects:
            detail = f"synced projects: {', '.join(synced_projects)}"
        else:
            detail = "sync skipped (fingerprints unchanged)"
        records.append(StepRecord(name="dependency_sync", status="ok", detail=detail))

        print("[4/6] Model provisioning")
        if synthesis_enabled:
            model_result = ensure_indextts_model(
                uv_executable=args.uv_executable,
                force_download=bool(args.force_model_download),
            )
            model_detail = (
                f"downloaded={model_result.downloaded}, source={model_result.source}"
            )
            records.append(StepRecord(name="model_download", status="ok", detail=model_detail))
        else:
            records.append(
                StepRecord(
                    name="model_download",
                    status="skipped",
                    detail=(
                        "Skipped because voice-clone runtime is not required for the "
                        "selected stage and overrides."
                    ),
                )
            )

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

        print("[5/7] Resolve runtime input")
        run_id = utc_now_compact()
        shortcut_overrides = build_shortcut_overrides(
            voiceclone_from_summary=args.voiceclone_from_summary,
            run_id=run_id,
        )
        base_overrides = build_run_overrides(
            run_id=run_id,
            start_stage=effective_start_stage,
            enable_voice_clone=voice_clone_enabled,
            enable_interjector=interjector_enabled,
        )
        overrides = [*base_overrides, *shortcut_overrides, *pass_through_overrides]
        stage_selection_detail = build_start_stage_selection_detail(
            start_stage=effective_start_stage,
            voiceclone_from_summary=args.voiceclone_from_summary,
            pass_through_overrides=pass_through_overrides,
            cli_audio_path=args.audio_path,
        )
        print(f"Selected {stage_selection_detail}")
        records.append(
            StepRecord(
                name="start_stage",
                status="ok",
                detail=stage_selection_detail,
            )
        )
        ensure_transcript_override_for_stage(
            overrides=overrides,
            start_stage=effective_start_stage,
            run_id=run_id,
        )
        runtime_transcript_path = (
            None
            if effective_start_stage == "stage-1"
            else resolve_transcript_override_path(overrides=overrides)
        )
        resolved_audio_path = resolve_audio_override_path(overrides=overrides)
        if requires_audio_path_input(overrides):
            if resolved_audio_path is None:
                resolved_audio_path = prompt_or_validate_audio_path(args.audio_path)
                overrides.append(
                    f"audio.audio_path={_path_for_override(resolved_audio_path)}"
                )
                input_detail = f"audio_path={resolved_audio_path}"
            else:
                input_detail = f"audio_path={resolved_audio_path} (override)"
            records.append(StepRecord(name="input", status="ok", detail=input_detail))
        elif args.audio_path is not None:
            resolved_audio_path = resolve_audio_input(args.audio_path)
            overrides.append(f"audio.audio_path={_path_for_override(resolved_audio_path)}")
            records.append(
                StepRecord(
                    name="input",
                    status="ok",
                    detail=f"audio_path={resolved_audio_path} (optional)",
                )
            )
        elif stage_two_requires_audio_fallback(
            transcript_path=runtime_transcript_path,
            speaker_samples_enabled=speaker_samples_enabled,
        ):
            detail_suffix = ""
            resolved_audio_path = (
                resolve_transcript_source_audio_path(runtime_transcript_path)
                if runtime_transcript_path is not None
                else None
            )
            if resolved_audio_path is not None:
                detail_suffix = "from transcript metadata for stage-2 speaker-sample inference"
            else:
                resolved_audio_path = _discover_existing_audio_path()
                if resolved_audio_path is not None:
                    detail_suffix = "auto-detected for stage-2 speaker-sample inference"
            if resolved_audio_path is None:
                resolved_audio_path = prompt_or_validate_audio_path(None)
                detail_suffix = "required for stage-2 speaker-sample inference"
            overrides.append(f"audio.audio_path={_path_for_override(resolved_audio_path)}")
            records.append(
                StepRecord(
                    name="input",
                    status="ok",
                    detail=f"audio_path={resolved_audio_path} ({detail_suffix})",
                )
            )
        elif resolved_audio_path is not None:
            records.append(
                StepRecord(
                    name="input",
                    status="ok",
                    detail=f"audio_path={resolved_audio_path} (override)",
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

        print("[6/7] Duration setup")
        if effective_start_stage == "stage-1" and calibration_runtime_enabled:
            calibration_warning_message = build_calibration_warning_message(
                start_stage=effective_start_stage,
            )
            if calibration_warning_message is not None:
                print(f"Note: {calibration_warning_message}")
            records.append(
                StepRecord(
                    name="calibration",
                    status="deferred",
                    detail=(
                        "Deferred to card_framework.cli.main after transcript "
                        "generation so the audio "
                        "stage does not run twice."
                    ),
                )
            )
        elif live_draft_audio_enabled:
            records.append(
                StepRecord(
                    name="calibration",
                    status="skipped",
                    detail=(
                        "Skipped because live stage-2/stage-3 drafting uses actual "
                        "rendered audio durations."
                    ),
                )
            )
        elif effective_start_stage == "stage-4":
            records.append(
                StepRecord(
                    name="calibration",
                    status="skipped",
                    detail="Skipped because stage-4 interjection does not require calibration.",
                )
            )
        elif effective_start_stage == "stage-3":
            records.append(
                StepRecord(
                    name="calibration",
                    status="skipped",
                    detail="Skipped because stage-3 voice cloning does not require calibration.",
                )
            )
        else:
            resolved_transcript_override = resolve_calibration_transcript_path(
                overrides=overrides,
                start_stage=effective_start_stage,
            )
            calibration_warning_message = build_calibration_warning_message(
                start_stage=effective_start_stage,
            )
            if calibration_warning_message is not None:
                print(f"Note: {calibration_warning_message}")
            run_calibration(
                uv_executable=args.uv_executable,
                transcript_path=resolved_transcript_override,
                audio_path=resolved_audio_path,
            )
            records.append(
                StepRecord(
                    name="calibration",
                    status="ok",
                    detail="Calibration artifact is ready.",
                )
            )

        print("[7/7] Running pipeline")
        run_pipeline(uv_executable=args.uv_executable, overrides=overrides)
        records.append(
            StepRecord(
                name="pipeline_run",
                status="ok",
                detail="card_framework.cli.main completed successfully.",
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
