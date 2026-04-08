"""All-in-one setup and execution bootstrap for the staged audio pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Literal, Mapping, Sequence, TypeAlias, cast
from uuid import uuid4

from card_framework.cli.frontend_models import (
    BenchmarkKind,
    BenchmarkLaunchRequest,
    FrontendResult,
    InteractiveMenuSelection,
)
from card_framework.cli.frontend_services import (
    describe_benchmark_request,
    describe_pipeline_selection,
    load_frontend_defaults,
)
from card_framework.runtime.bootstrap import ensure_index_tts_runtime
from card_framework.runtime.pipeline_sessions import (
    PipelineResumePlan,
    PipelineSessionStore,
)
from card_framework.shared.paths import (
    DEFAULT_CONFIG_PATH as SHARED_DEFAULT_CONFIG_PATH,
    REPO_ROOT,
    VENDOR_INDEX_TTS_DIR,
    discover_reusable_transcript_path,
)
from card_framework.shared.perf import report_loading_duration
from card_framework.shared.runtime_layout import resolve_runtime_layout

MAIN_MODULE = "card_framework.cli.main"
CALIBRATE_MODULE = "card_framework.cli.calibrate"
SETUP_AND_RUN_MODULE = "card_framework.cli.setup_and_run"
DEFAULT_CONFIG_PATH: Path | None = None
INDEX_TTS_DIR: Path | None = None
INDEX_TTS_CHECKPOINTS_DIR: Path | None = None
CONFIG_FILE_PATH: Path | None = None
WORKSPACE_ROOT: Path | None = None
OUTPUT_ROOT: Path | None = None
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
DEFAULT_DURATION_SWEEP_PRESET_LABELS = ("30s", "1m", "5m")
DURATION_SWEEP_PRESET_SECONDS = {
    "30s": 30,
    "1m": 60,
    "5m": 300,
}
PROFILE_DISPLAY_NAMES: dict[str, str] = {
    "anthropic_claude_opus": "Anthropic Claude Opus (cloud, requires ANTHROPIC_API_KEY)",
    "deepseek_chat": "DeepSeek Chat",
    "deepseek_reasoner": "DeepSeek Reasoner",
    "disabled": "Disabled",
    "glm_46": "ZAI GLM 4.6",
    "google_gemini": "Google Gemini 2.5 Pro",
    "huggingface_toolace": "Hugging Face ToolACE",
    "minilm_cpu": "MiniLM CPU embeddings",
    "nanbeige_thinking": "Nanbeige Thinking",
    "qwen3_4b": "Qwen3 4B embeddings",
    "transformers_local": "Local Transformers tool model",
    "vllm_local": "vLLM / OpenAI-compatible endpoint",
}
_MENU_CONSOLE: Any | None = None
_MENU_CONSOLE_UNAVAILABLE = object()


@dataclass(slots=True)
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


def _emit_setup_status(message: str) -> None:
    """Emit one operator-facing setup status line to stderr."""
    print(message, file=sys.stderr, flush=True)


def _emit_session_store_warning(*, action: str, exc: Exception) -> None:
    """Report session-store failures without hiding the primary pipeline error."""
    _emit_setup_status(f"Session store warning during {action}: {exc}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse bootstrap CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Setup dependencies/models and run the staged pipeline with optional "
            "speaker samples, voice cloning, and Stage-4 interjections."
        )
    )
    menu_group = parser.add_mutually_exclusive_group()
    menu_group.add_argument(
        "--menu",
        action="store_true",
        help=(
            "Force the interactive setup frontend even when explicit run inputs "
            "were already supplied."
        ),
    )
    menu_group.add_argument(
        "--no-menu",
        action="store_true",
        help=(
            "Skip the interactive setup frontend and use only CLI flags plus "
            "the existing auto-stage behavior."
        ),
    )
    parser.add_argument(
        "--audio-path",
        action="append",
        default=[],
        help=(
            "Source audio path for pipeline.start_stage=stage-1. When omitted, "
            "the bootstrap auto-uses stage-2 if a reusable transcript JSON is "
            "already present. Repeat this flag to run a sequential batch sweep "
            "across multiple audio files."
        ),
    )
    parser.add_argument(
        "--duration-sweep",
        action="store_true",
        help=(
            "When one or more --audio-path values are supplied, reuse each "
            "episode transcript and rerun stage-2 across duration presets "
            "instead of redoing stage-1 for every target."
        ),
    )
    parser.add_argument(
        "--duration-presets",
        help=(
            "Comma-separated duration presets for --duration-sweep. Defaults "
            "to 30s,1m,5m."
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
        "--resume-session",
        help="Resume a previously saved pipeline session by UUID.",
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
    parser.add_argument("--workspace-root", help=argparse.SUPPRESS)
    parser.add_argument("--output-root", help=argparse.SUPPRESS)
    parser.add_argument("--config-file", help=argparse.SUPPRESS)
    parser.add_argument("--frontend-benchmark-request-json", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def utc_now_compact() -> str:
    """Return UTC timestamp suitable for run IDs."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    """Return UTC ISO-8601 timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_repo_config_path() -> Path:
    """Return the active repo config path, preferring the packaged src layout."""
    if CONFIG_FILE_PATH is not None:
        return Path(CONFIG_FILE_PATH).resolve()
    if DEFAULT_CONFIG_PATH is not None:
        return Path(DEFAULT_CONFIG_PATH).resolve()
    return SHARED_DEFAULT_CONFIG_PATH.resolve()


def resolve_workspace_root() -> Path:
    """Return the caller workspace root used for relative operator-facing paths."""
    if WORKSPACE_ROOT is not None:
        return Path(WORKSPACE_ROOT).resolve()
    return Path.cwd().resolve()


def resolve_output_root() -> Path:
    """Return the output root used for packaged-style flat artifact layouts."""
    if OUTPUT_ROOT is not None:
        return Path(OUTPUT_ROOT).resolve()
    return resolve_workspace_root()


def has_explicit_output_root() -> bool:
    """Return whether the wrapper was given an explicit output root."""
    return OUTPUT_ROOT is not None


def resolve_setup_state_path() -> Path:
    """Return the per-workspace setup-state JSON path."""
    return (
        resolve_workspace_root() / "artifacts" / "bootstrap" / "setup_state.json"
    ).resolve()


def resolve_pipeline_session_store_path() -> Path:
    """Return the SQLite database path used for pipeline session persistence."""
    return (
        resolve_output_root() / "artifacts" / "pipeline_sessions" / "sessions.sqlite3"
    ).resolve()


def resolve_pipeline_session_root(session_id: str) -> Path:
    """Return the artifact root used for one pipeline session UUID."""
    return (
        resolve_output_root() / "artifacts" / "pipeline_sessions" / session_id
    ).resolve()


def resolve_pipeline_session_summary_path(session_id: str) -> Path:
    """Return the session-scoped summary checkpoint path for one UUID."""
    return (resolve_pipeline_session_root(session_id) / "summary.xml").resolve()


def build_pipeline_session_artifacts(
    *,
    run_id: str,
    transcript_path_override: Path | None = None,
) -> dict[str, str]:
    """Build the deterministic artifact path catalog for one pipeline session."""
    if has_explicit_output_root():
        run_work_dir = (resolve_output_root() / "audio_stage").resolve()
        transcript_path = (
            transcript_path_override.resolve()
            if transcript_path_override is not None
            else (resolve_output_root() / "transcript.json").resolve()
        )
        workspace_summary_path = (resolve_output_root() / "summary.xml").resolve()
    else:
        workspace_root = resolve_workspace_root()
        run_work_dir = (
            workspace_root / "artifacts" / "audio_stage" / "runs" / run_id
        ).resolve()
        transcript_path = (
            transcript_path_override.resolve()
            if transcript_path_override is not None
            else (
                workspace_root
                / "artifacts"
                / "transcripts"
                / f"{run_id}.transcript.json"
            ).resolve()
        )
        workspace_summary_path = (workspace_root / "summary.xml").resolve()
    return {
        "work_dir": str(run_work_dir),
        "transcript_path": str(transcript_path),
        "workspace_summary_path": str(workspace_summary_path),
        "session_summary_path": str(resolve_pipeline_session_summary_path(run_id)),
        "voice_clone_manifest_path": str(
            (run_work_dir / "voice_clone" / "manifest.json").resolve()
        ),
        "interjector_manifest_path": str(
            (run_work_dir / "interjector" / "interjector_manifest.json").resolve()
        ),
    }


def resolve_session_transcript_artifact_override(
    *,
    start_stage: PipelineStartStage,
    transcript_path: Path | None,
) -> Path | None:
    """Persist external transcript inputs for stage-2/stage-3 session resume."""
    if start_stage in {"stage-2", "stage-3"} and transcript_path is not None:
        return transcript_path.resolve()
    return None


def resolve_root_project_dir() -> Path | None:
    """Return the repository checkout root when running from source."""
    candidate = REPO_ROOT.resolve()
    if (candidate / "pyproject.toml").exists() and (candidate / ".git").exists():
        return candidate
    return None


def is_repo_checkout_runtime() -> bool:
    """Return whether the wrapper is executing from the repository checkout."""
    return resolve_root_project_dir() is not None


def resolve_index_tts_dir() -> Path:
    """Return the active vendored IndexTTS source directory for this repo."""
    if INDEX_TTS_DIR is not None:
        return Path(INDEX_TTS_DIR).resolve()
    root_project_dir = resolve_root_project_dir()
    if root_project_dir is not None:
        return (
            root_project_dir / "src" / "card_framework" / "_vendor" / "index_tts"
        ).resolve()
    return VENDOR_INDEX_TTS_DIR.resolve()


def resolve_index_tts_runner_project_dir() -> Path:
    """Return the IndexTTS project dir that runtime subprocesses should execute from."""
    if is_repo_checkout_runtime():
        return resolve_index_tts_dir()
    layout = resolve_runtime_layout()
    return (
        layout.vendor_runtime_dir.resolve()
        if layout.vendor_runtime_dir.exists()
        else layout.vendor_source_dir.resolve()
    )


def resolve_index_tts_checkpoints_dir() -> Path:
    """Return the active IndexTTS checkpoints directory for this repo."""
    if INDEX_TTS_CHECKPOINTS_DIR is not None:
        return Path(INDEX_TTS_CHECKPOINTS_DIR).resolve()
    root_project_dir = resolve_root_project_dir()
    if root_project_dir is not None:
        return (root_project_dir / "checkpoints" / "index_tts").resolve()
    return resolve_runtime_layout().checkpoints_dir.resolve()


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
    uv_executable: str,
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
    required_tools = [uv_executable]
    if require_ffmpeg:
        required_tools.append("ffmpeg")
    missing_tools = [
        tool
        for tool in required_tools
        if not Path(tool).expanduser().is_file() and shutil.which(tool) is None
    ]
    if missing_tools:
        raise BootstrapError(
            step="preflight",
            message=(
                "Missing required command(s): "
                f"{', '.join(missing_tools)}. {build_tool_guidance(missing_tools)}"
            ),
        )

    required_paths: list[Path] = [
        Path(__file__).with_name("main.py"),
        resolve_repo_config_path(),
    ]
    if require_git:
        index_tts_dir = resolve_index_tts_dir()
        required_paths.extend(
            [
                index_tts_dir,
                index_tts_dir / "pyproject.toml",
                index_tts_dir / "uv.lock",
                index_tts_dir / "indextts",
            ]
        )
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        raise BootstrapError(
            step="preflight",
            message=(
                f"Runtime is missing required path(s): {', '.join(missing_paths)}."
            ),
        )


def build_tool_guidance(missing_tools: Sequence[str]) -> str:
    """Return concise cross-platform install guidance for missing tools."""
    guidance_map = {
        "uv": "Install uv into the active environment: `python -m pip install -U uv`.",
        "git": "Install Git and ensure `git` is on PATH: https://git-scm.com/downloads",
        "ffmpeg": "Install FFmpeg and add it to PATH: https://ffmpeg.org/download.html",
    }
    guidance_parts = [
        guidance_map[tool] for tool in missing_tools if tool in guidance_map
    ]
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
    root_project_dir = resolve_root_project_dir()
    project_specs: list[tuple[str, Path]] = []
    if root_project_dir is not None:
        project_specs.append(("root", root_project_dir))
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
    setup_state_path = resolve_setup_state_path()
    if not setup_state_path.exists():
        return {}
    try:
        payload = json.loads(setup_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_setup_state(payload: Mapping[str, Any]) -> None:
    """Persist setup-state JSON atomically."""
    setup_state_path = resolve_setup_state_path()
    _write_json_atomic(setup_state_path, payload)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist one JSON payload atomically with a UTF-8 trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _slugify_batch_audio_name(audio_path: Path) -> str:
    """Return a short filesystem-safe slug for one batch audio input."""
    raw_name = audio_path.stem.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", raw_name).strip("-")
    if not slug:
        return "audio"
    return slug[:48]


def _ensure_unique_directory_path(path: Path) -> Path:
    """Return a non-existent directory path by suffixing when needed."""
    if not path.exists():
        return path
    for counter in range(2, 10_000):
        candidate = path.parent / f"{path.name}-{counter}"
        if not candidate.exists():
            return candidate
    raise BootstrapError(
        step="input",
        message=f"Failed to allocate a unique output directory under {path.parent}.",
    )


def resolve_batch_output_root(
    *,
    workspace_root: Path,
    output_root: Path,
    explicit_output_root: bool,
) -> tuple[str, Path]:
    """Resolve the parent output directory used for one multi-audio batch run."""
    batch_id = utc_now_compact()
    if explicit_output_root:
        batch_root = output_root.resolve()
    else:
        batch_root = _ensure_unique_directory_path(
            (workspace_root / "artifacts" / "pipeline_batches" / batch_id).resolve()
        )
    batch_root.mkdir(parents=True, exist_ok=True)
    return batch_id, batch_root


def resolve_batch_run_output_root(
    *,
    batch_root: Path,
    audio_path: Path,
    index: int,
) -> Path:
    """Resolve one isolated per-audio output directory inside a batch root."""
    candidate = batch_root / f"{index:03d}_{_slugify_batch_audio_name(audio_path)}"
    return _ensure_unique_directory_path(candidate.resolve())


def parse_duration_sweep_presets(raw_value: str | None) -> tuple[str, ...]:
    """Parse supported setup-and-run duration sweep preset labels."""
    if raw_value is None:
        return DEFAULT_DURATION_SWEEP_PRESET_LABELS
    labels = [
        item.strip().lower() for item in str(raw_value).split(",") if item.strip()
    ]
    if not labels:
        raise BootstrapError(
            step="input",
            message="At least one duration preset is required for --duration-sweep.",
        )
    invalid = [label for label in labels if label not in DURATION_SWEEP_PRESET_SECONDS]
    if invalid:
        raise BootstrapError(
            step="input",
            message=(
                "Unsupported duration preset(s) for --duration-sweep: "
                + ", ".join(invalid)
                + ". Supported presets: "
                + ", ".join(DEFAULT_DURATION_SWEEP_PRESET_LABELS)
            ),
        )
    ordered_labels: list[str] = []
    for label in labels:
        if label not in ordered_labels:
            ordered_labels.append(label)
    return tuple(ordered_labels)


def resolve_duration_sweep_output_root(
    *,
    workspace_root: Path,
    output_root: Path,
    explicit_output_root: bool,
) -> tuple[str, Path]:
    """Resolve the parent output directory used for one duration sweep run."""
    sweep_id = utc_now_compact()
    if explicit_output_root:
        sweep_root = output_root.resolve()
    else:
        sweep_root = _ensure_unique_directory_path(
            (
                workspace_root / "artifacts" / "pipeline_duration_sweeps" / sweep_id
            ).resolve()
        )
    sweep_root.mkdir(parents=True, exist_ok=True)
    return sweep_id, sweep_root


def resolve_duration_sweep_episode_output_root(
    *,
    sweep_root: Path,
    audio_path: Path,
    index: int,
) -> Path:
    """Return the deterministic per-episode directory inside a duration sweep."""
    return (
        sweep_root / f"{index:03d}_{_slugify_batch_audio_name(audio_path)}"
    ).resolve()


def resolve_duration_sweep_preset_output_root(
    *,
    episode_root: Path,
    preset_label: str,
) -> Path:
    """Return the deterministic per-preset directory for one episode sweep."""
    return (episode_root / preset_label).resolve()


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
        candidate = (resolve_workspace_root() / candidate).resolve()
    if not candidate.exists():
        raise BootstrapError(
            step="input", message=f"Audio path does not exist: {candidate}"
        )
    if not candidate.is_file():
        raise BootstrapError(
            step="input", message=f"Audio path is not a file: {candidate}"
        )
    return candidate


def resolve_audio_inputs(path_values: Sequence[str]) -> tuple[Path, ...]:
    """Resolve and validate multiple user-provided audio paths."""
    return tuple(resolve_audio_input(path_value) for path_value in path_values)


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
        candidate = (resolve_workspace_root() / candidate).resolve()
    if not candidate.exists():
        raise BootstrapError(
            step="input", message=f"{field_name} does not exist: {candidate}"
        )
    if not candidate.is_file():
        raise BootstrapError(
            step="input", message=f"{field_name} is not a file: {candidate}"
        )
    return candidate


def resolve_directory_input(path_value: str, *, field_name: str) -> Path:
    """Resolve one generic directory path, creating no files automatically."""
    normalized = path_value.strip().strip('"').strip("'")
    if not normalized:
        raise BootstrapError(step="input", message=f"{field_name} must not be empty.")
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve()
    if not candidate.exists():
        raise BootstrapError(
            step="input", message=f"{field_name} does not exist: {candidate}"
        )
    if not candidate.is_dir():
        raise BootstrapError(
            step="input", message=f"{field_name} is not a directory: {candidate}"
        )
    return candidate


def resolve_file_input(
    path_value: str,
    *,
    field_name: str,
    base_dir: Path,
) -> Path:
    """Resolve one file path relative to a supplied base directory."""
    normalized = path_value.strip().strip('"').strip("'")
    if not normalized:
        raise BootstrapError(step="input", message=f"{field_name} must not be empty.")
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    if not candidate.exists():
        raise BootstrapError(
            step="input", message=f"{field_name} does not exist: {candidate}"
        )
    if not candidate.is_file():
        raise BootstrapError(
            step="input", message=f"{field_name} is not a file: {candidate}"
        )
    return candidate


def is_interactive_console() -> bool:
    """Return whether the current invocation can safely prompt a human operator."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def has_explicit_cli_run_inputs(args: argparse.Namespace) -> bool:
    """Return whether explicit CLI flags already define a run path."""
    return any(
        (
            bool(getattr(args, "audio_path", ())),
            bool(getattr(args, "voiceclone_from_summary", None)),
            bool(getattr(args, "resume_session", None)),
            bool(getattr(args, "override", ())),
            bool(getattr(args, "setup_only", False)),
        )
    )


def should_launch_interactive_menu(args: argparse.Namespace) -> bool:
    """Return whether setup_and_run should open the interactive menu."""
    if bool(getattr(args, "no_menu", False)):
        return False
    if bool(getattr(args, "menu", False)):
        return True
    if not is_interactive_console():
        return False
    return not has_explicit_cli_run_inputs(args)


def _profile_display_name(profile_name: str) -> str:
    """Return one operator-facing label for a runtime profile key."""
    if profile_name in PROFILE_DISPLAY_NAMES:
        return PROFILE_DISPLAY_NAMES[profile_name]
    return profile_name.replace("_", " ").strip() or profile_name


def _get_menu_console() -> Any | None:
    """Return a cached Rich console for the interactive menu when available."""
    global _MENU_CONSOLE
    if _MENU_CONSOLE is _MENU_CONSOLE_UNAVAILABLE:
        return None
    if _MENU_CONSOLE is None:
        try:
            from rich.console import Console
        except ImportError:  # pragma: no cover - optional fallback
            _MENU_CONSOLE = _MENU_CONSOLE_UNAVAILABLE
            return None
        _MENU_CONSOLE = Console(highlight=False, soft_wrap=True)
    return _MENU_CONSOLE


def _menu_print(message: str, *, style: str = "") -> None:
    """Print one menu status line with Rich styling when available."""
    console = _get_menu_console()
    if console is None:
        print(message)
        return
    if style:
        console.print(message, style=style)
        return
    console.print(message)


def _menu_input(prompt: str, *, style: str = "bold bright_cyan") -> str:
    """Prompt for input with Rich styling when available."""
    console = _get_menu_console()
    if console is None:
        return input(prompt)
    return console.input(f"[{style}]{prompt}[/{style}]")


def _menu_warning(message: str) -> None:
    """Render one operator-facing warning inside the menu."""
    _menu_print(message, style="bold yellow")


def _menu_error(message: str) -> None:
    """Render one operator-facing error inside the menu."""
    _menu_print(message, style="bold red")


def _prompt_menu_choice(
    *,
    title: str,
    options: Sequence[tuple[str, str]],
    default_index: int = 0,
) -> str:
    """Prompt for one numbered menu choice."""
    console = _get_menu_console()
    if console is None:
        print(f"\n{title}")
        for index, (_value, label) in enumerate(options, start=1):
            default_marker = " [default]" if index - 1 == default_index else ""
            print(f"  {index}. {label}{default_marker}")
    else:
        from rich.panel import Panel
        from rich.table import Table

        table = Table(
            show_header=False,
            box=None,
            pad_edge=False,
            collapse_padding=True,
        )
        table.add_column(style="bold cyan", no_wrap=True)
        table.add_column(style="white")
        for index, (_value, label) in enumerate(options, start=1):
            option_style = (
                "bold bright_white" if index - 1 == default_index else "white"
            )
            default_note = " [dim](default)[/dim]" if index - 1 == default_index else ""
            table.add_row(
                f"{index}.", f"[{option_style}]{label}[/{option_style}]{default_note}"
            )
        console.print(
            Panel.fit(
                table,
                title=f"[bold magenta]{title}[/bold magenta]",
                border_style="bright_blue",
                padding=(0, 1),
            )
        )
    while True:
        raw = _menu_input("Select an option: ").strip()
        if not raw:
            return options[default_index][0]
        if raw.isdigit():
            choice_index = int(raw) - 1
            if 0 <= choice_index < len(options):
                return options[choice_index][0]
        _menu_warning("Please enter one of the numbered choices above.")


def _prompt_menu_text(
    prompt: str,
    *,
    default: str = "",
    allow_blank: bool = False,
) -> str:
    """Prompt for one short text value."""
    rendered_prompt = prompt
    if default:
        rendered_prompt += f" [{default}]"
    rendered_prompt += ": "
    while True:
        raw = _menu_input(rendered_prompt).strip()
        if raw:
            return raw
        if default:
            return default
        if allow_blank:
            return ""
        _menu_warning("A value is required here.")


def _prompt_optional_file_input(
    *,
    prompt: str,
    field_name: str,
) -> Path | None:
    """Prompt for an optional file path, returning None when left blank."""
    while True:
        raw = _menu_input(f"{prompt} (leave blank to skip): ").strip()
        if not raw:
            return None
        try:
            return resolve_path_input(raw, field_name=field_name)
        except BootstrapError as exc:
            _menu_error(exc.render())


def _prompt_audio_paths_for_menu(*, allow_multiple: bool) -> tuple[Path, ...]:
    """Prompt for one or more audio file paths."""
    if not allow_multiple:
        while True:
            raw = _menu_input("Source audio path: ").strip()
            try:
                return (resolve_audio_input(raw),)
            except BootstrapError as exc:
                _menu_error(exc.render())

    _menu_print(
        "\nEnter one audio path per line. Press Enter on a blank line when you are done.",
        style="bold blue",
    )
    audio_paths: list[Path] = []
    while True:
        raw = _menu_input(f"Audio path {len(audio_paths) + 1}: ").strip()
        if not raw:
            if len(audio_paths) >= 2:
                return tuple(audio_paths)
            _menu_warning("Batch mode needs at least two audio files.")
            continue
        try:
            audio_paths.append(resolve_audio_input(raw))
        except BootstrapError as exc:
            _menu_error(exc.render())


def _build_profile_choice_options(
    *,
    profile_group: str,
    keep_current_label: str,
    include_default_stage_option: bool = False,
    default_stage_label: str = "Use shared LLM",
) -> list[tuple[str, str]]:
    """Build ordered interactive menu options for one profile group."""
    options: list[tuple[str, str]] = [("__keep_current__", keep_current_label)]
    if include_default_stage_option:
        options.append(("default", default_stage_label))
    for profile_name in resolve_repo_config_child_keys(
        key_path=("profiles", profile_group)
    ):
        options.append((profile_name, _profile_display_name(profile_name)))
    return options


def run_interactive_menu() -> InteractiveMenuSelection:
    """Collect one interactive setup-and-run launch plan from the operator."""
    console = _get_menu_console()
    current_llm = resolve_repo_config_string(
        key_path=("selection", "llm"),
        fallback="vllm_local",
    )
    current_embedding = resolve_repo_config_string(
        key_path=("selection", "embedding"),
        fallback="disabled",
    )
    current_stage_defaults = {
        stage_name: resolve_repo_config_string(
            key_path=("selection", "stage_llm", stage_name),
            fallback="default",
        )
        for stage_name in ("summarizer", "critic", "interjector")
    }
    current_voice_clone_enabled = resolve_repo_config_boolean(
        key_path=("audio", "voice_clone", "enabled"),
        fallback=True,
    )
    current_interjector_enabled = resolve_repo_config_boolean(
        key_path=("audio", "interjector", "enabled"),
        fallback=False,
    )

    if console is None:
        print("\nInteractive setup_and_run menu")
        print(f"Config: {resolve_repo_config_path()}")
        print("Selections apply to this run only unless you later save them in config.")
    else:
        from rich.panel import Panel
        from rich.table import Table

        current_settings = Table(
            show_header=False,
            box=None,
            pad_edge=False,
            collapse_padding=True,
        )
        current_settings.add_column(style="bold cyan", no_wrap=True)
        current_settings.add_column(style="white")
        current_settings.add_row("Config", str(resolve_repo_config_path()))
        current_settings.add_row("LLM", _profile_display_name(current_llm))
        current_settings.add_row("Embedding", _profile_display_name(current_embedding))
        current_settings.add_row(
            "Voice clone",
            "Enabled" if current_voice_clone_enabled else "Disabled",
        )
        current_settings.add_row(
            "Interjector",
            "Enabled" if current_interjector_enabled else "Disabled",
        )
        console.print(
            Panel.fit(
                current_settings,
                title="[bold bright_white]Setup And Run[/bold bright_white]",
                subtitle="[dim]Selections apply to this run only unless you later save them in config.[/dim]",
                border_style="bright_magenta",
                padding=(0, 1),
            )
        )

    mode = _prompt_menu_choice(
        title="Run mode",
        options=[
            ("stage1_single", "Full pipeline from one source audio file"),
            ("stage1_batch", "Sequential batch sweep from multiple audio files"),
            (
                "stage1_batch_duration_sweep",
                "Batch audio + duration sweep (30s, 1m, 5m per episode)",
            ),
            ("stage2_transcript", "Resume from an existing transcript JSON (stage-2)"),
            ("stage3_summary", "Voice clone from an existing summary XML (stage-3)"),
            ("resume_session", "Resume a saved session by UUID"),
            ("setup_only", "Setup checks/models only without running the pipeline"),
        ],
        default_index=0,
    )

    output_root_raw = _prompt_menu_text(
        "Output root",
        default="",
        allow_blank=True,
    )
    output_root = output_root_raw or None

    overrides: list[str] = []
    if mode in {
        "stage1_single",
        "stage1_batch",
        "stage1_batch_duration_sweep",
        "stage2_transcript",
    }:
        llm_choice = _prompt_menu_choice(
            title="Shared LLM profile",
            options=_build_profile_choice_options(
                profile_group="llm",
                keep_current_label=(
                    "Keep current config default "
                    f"({_profile_display_name(current_llm)})"
                ),
            ),
            default_index=0,
        )
        if llm_choice != "__keep_current__":
            overrides.append(f"selection.llm={llm_choice}")

        stage_strategy_default_index = 0 if llm_choice == "__keep_current__" else 1
        stage_strategy = _prompt_menu_choice(
            title="Stage LLM routing",
            options=[
                ("config_default", "Keep per-stage selections from config"),
                ("shared_for_all", "Use the shared LLM for all stages"),
                (
                    "custom",
                    "Choose summarizer, critic, and interjector profiles separately",
                ),
            ],
            default_index=stage_strategy_default_index,
        )
        if stage_strategy == "shared_for_all":
            for stage_name in ("summarizer", "critic", "interjector"):
                overrides.append(f"selection.stage_llm.{stage_name}=default")
        elif stage_strategy == "custom":
            for stage_name in ("summarizer", "critic", "interjector"):
                stage_choice = _prompt_menu_choice(
                    title=f"{stage_name.title()} LLM profile",
                    options=_build_profile_choice_options(
                        profile_group="llm",
                        keep_current_label=(
                            "Keep config default "
                            f"({current_stage_defaults[stage_name]})"
                        ),
                        include_default_stage_option=True,
                        default_stage_label="Use the shared LLM",
                    ),
                    default_index=0,
                )
                if stage_choice != "__keep_current__":
                    overrides.append(f"selection.stage_llm.{stage_name}={stage_choice}")

        embedding_choice = _prompt_menu_choice(
            title="Embedding profile",
            options=_build_profile_choice_options(
                profile_group="embedding",
                keep_current_label=(
                    "Keep current config default "
                    f"({_profile_display_name(current_embedding)})"
                ),
            ),
            default_index=0,
        )
        if embedding_choice != "__keep_current__":
            overrides.append(f"selection.embedding={embedding_choice}")

        voice_clone_choice = _prompt_menu_choice(
            title="Voice clone stage",
            options=[
                (
                    "__keep_current__",
                    "Keep current config default "
                    f"({'enabled' if current_voice_clone_enabled else 'disabled'})",
                ),
                ("true", "Enable voice clone"),
                ("false", "Disable voice clone"),
            ],
            default_index=0,
        )
        if voice_clone_choice != "__keep_current__":
            overrides.append(f"audio.voice_clone.enabled={voice_clone_choice}")

        if voice_clone_choice == "false":
            overrides.append("audio.interjector.enabled=false")
        else:
            interjector_choice = _prompt_menu_choice(
                title="Stage-4 interjector",
                options=[
                    (
                        "__keep_current__",
                        "Keep current config default "
                        f"({'enabled' if current_interjector_enabled else 'disabled'})",
                    ),
                    ("true", "Enable interjector"),
                    ("false", "Disable interjector"),
                ],
                default_index=0,
            )
            if interjector_choice != "__keep_current__":
                overrides.append(f"audio.interjector.enabled={interjector_choice}")

    if mode == "stage1_single":
        audio_paths = _prompt_audio_paths_for_menu(allow_multiple=False)
        return InteractiveMenuSelection(
            mode="stage1_single",
            audio_paths=tuple(str(path) for path in audio_paths),
            output_root=output_root,
            overrides=tuple(overrides),
        )
    if mode == "stage1_batch":
        audio_paths = _prompt_audio_paths_for_menu(allow_multiple=True)
        return InteractiveMenuSelection(
            mode="stage1_batch",
            audio_paths=tuple(str(path) for path in audio_paths),
            output_root=output_root,
            overrides=tuple(overrides),
        )
    if mode == "stage1_batch_duration_sweep":
        audio_paths = _prompt_audio_paths_for_menu(allow_multiple=True)
        _menu_print(
            "Duration sweep presets: "
            + ", ".join(DEFAULT_DURATION_SWEEP_PRESET_LABELS),
            style="bold bright_green",
        )
        return InteractiveMenuSelection(
            mode="stage1_batch_duration_sweep",
            audio_paths=tuple(str(path) for path in audio_paths),
            duration_presets=DEFAULT_DURATION_SWEEP_PRESET_LABELS,
            output_root=output_root,
            overrides=tuple(overrides),
        )
    if mode == "stage2_transcript":
        transcript_path = resolve_path_input(
            _prompt_menu_text("Transcript JSON path"),
            field_name="Transcript JSON path",
        )
        optional_audio_path = _prompt_optional_file_input(
            prompt="Optional source audio path for stage-2 speaker-sample fallback",
            field_name="Source audio path",
        )
        audio_paths = (
            (str(optional_audio_path),) if optional_audio_path is not None else ()
        )
        return InteractiveMenuSelection(
            mode="stage2_transcript",
            audio_paths=audio_paths,
            transcript_path=str(transcript_path),
            output_root=output_root,
            overrides=tuple(overrides),
        )
    if mode == "stage3_summary":
        summary_path = resolve_path_input(
            _prompt_menu_text("Summary XML path"),
            field_name="Summary XML path",
        )
        return InteractiveMenuSelection(
            mode="stage3_summary",
            voiceclone_from_summary=str(summary_path),
            output_root=output_root,
        )
    if mode == "resume_session":
        return InteractiveMenuSelection(
            mode="resume_session",
            resume_session=_prompt_menu_text("Session UUID"),
            output_root=output_root,
        )
    return InteractiveMenuSelection(
        mode="setup_only",
        setup_only=True,
        output_root=output_root,
    )


def apply_interactive_menu_selection(
    args: argparse.Namespace,
    selection: InteractiveMenuSelection,
) -> argparse.Namespace:
    """Project one interactive menu selection back onto parsed CLI args."""
    args.audio_path = list(selection.audio_paths)
    args.duration_sweep = bool(selection.duration_presets)
    args.duration_presets = (
        ",".join(selection.duration_presets) if selection.duration_presets else None
    )
    args.voiceclone_from_summary = selection.voiceclone_from_summary
    args.resume_session = selection.resume_session
    args.setup_only = bool(selection.setup_only)
    if selection.output_root is not None:
        args.output_root = selection.output_root
    if selection.transcript_path is not None:
        args.override = list(args.override) + [
            f"transcript_path={_path_for_override(Path(selection.transcript_path))}"
        ]
    if selection.mode == "stage4_interjector":
        if selection.voiceclone_from_summary is None:
            raise BootstrapError(
                step="input",
                message="Stage-4 frontend selection requires a summary XML path.",
            )
        if selection.voice_clone_manifest_path is None:
            raise BootstrapError(
                step="input",
                message="Stage-4 frontend selection requires a voice-clone manifest path.",
            )
        args.voiceclone_from_summary = None
        args.override = list(args.override) + [
            "pipeline.start_stage=stage-4",
            f"pipeline.final_summary_path={_path_for_override(Path(selection.voiceclone_from_summary))}",
            f"pipeline.voice_clone_manifest_path={_path_for_override(Path(selection.voice_clone_manifest_path))}",
        ]
    if selection.overrides:
        args.override = list(args.override) + list(selection.overrides)
    return args


def launch_interactive_frontend(*, uv_executable: str) -> FrontendResult:
    """Launch the default interactive frontend with a legacy fallback path."""
    try:
        from card_framework.cli.textual_frontend import run_textual_frontend

        defaults = load_frontend_defaults(
            config_path=resolve_repo_config_path(),
            workspace_root=resolve_workspace_root(),
        )
        return run_textual_frontend(defaults=defaults, uv_executable=uv_executable)
    except Exception as exc:
        _emit_setup_status(
            "Textual frontend unavailable; falling back to the legacy prompt menu. "
            f"Reason: {exc}"
        )
        return FrontendResult(
            action="pipeline",
            pipeline_selection=run_interactive_menu(),
        )


def serialize_frontend_benchmark_request(request: BenchmarkLaunchRequest) -> str:
    """Serialize one benchmark request for a monitored wrapper child process."""
    return json.dumps(
        {
            "kind": request.kind,
            "command": list(request.command),
            "description": request.description,
        },
        separators=(",", ":"),
    )


def deserialize_frontend_benchmark_request(payload: str) -> BenchmarkLaunchRequest:
    """Deserialize one benchmark request passed through the wrapper CLI."""
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise BootstrapError(
            step="input",
            message="Frontend benchmark request payload must be a JSON object.",
        )
    kind = str(data.get("kind", "")).strip()
    description = str(data.get("description", "")).strip()
    raw_command = data.get("command", ())
    if not kind or not description or not isinstance(raw_command, list):
        raise BootstrapError(
            step="input",
            message="Frontend benchmark request payload is missing required fields.",
        )
    command = tuple(str(part) for part in raw_command if str(part).strip())
    if not command:
        raise BootstrapError(
            step="input",
            message="Frontend benchmark request payload did not contain a command.",
        )
    return BenchmarkLaunchRequest(
        kind=cast(BenchmarkKind, kind),
        command=command,
        description=description,
    )


def build_textual_setup_and_run_command(args: argparse.Namespace) -> tuple[str, ...]:
    """Build a non-interactive child command for Textual-monitored wrapper runs."""
    uv_executable = str(getattr(args, "uv_executable", "uv")).strip() or "uv"
    git_executable = str(getattr(args, "git_executable", "git")).strip() or "git"
    if is_repo_checkout_runtime():
        command: list[str] = [
            uv_executable,
            "run",
            "python",
            "-u",
            "-m",
            SETUP_AND_RUN_MODULE,
        ]
    else:
        command = [sys.executable, "-u", "-m", SETUP_AND_RUN_MODULE]
    command.append("--no-menu")
    for audio_path in getattr(args, "audio_path", ()) or ():
        command.extend(["--audio-path", str(audio_path)])
    if bool(getattr(args, "duration_sweep", False)):
        command.append("--duration-sweep")
    duration_presets = getattr(args, "duration_presets", None)
    if duration_presets:
        command.extend(["--duration-presets", str(duration_presets)])
    voiceclone_from_summary = getattr(args, "voiceclone_from_summary", None)
    if voiceclone_from_summary:
        command.extend(["--voiceclone-from-summary", str(voiceclone_from_summary)])
    resume_session = getattr(args, "resume_session", None)
    if resume_session:
        command.extend(["--resume-session", str(resume_session)])
    for override in getattr(args, "override", ()) or ():
        command.extend(["--override", str(override)])
    if bool(getattr(args, "setup_only", False)):
        command.append("--setup-only")
    if bool(getattr(args, "skip_repo_update", False)):
        command.append("--skip-repo-update")
    if bool(getattr(args, "force_sync", False)):
        command.append("--force-sync")
    if bool(getattr(args, "force_model_download", False)):
        command.append("--force-model-download")
    command.extend(["--uv-executable", uv_executable])
    command.extend(["--git-executable", git_executable])
    workspace_root = getattr(args, "workspace_root", None)
    if workspace_root:
        command.extend(["--workspace-root", str(workspace_root)])
    output_root = getattr(args, "output_root", None)
    if output_root:
        command.extend(["--output-root", str(output_root)])
    config_file = getattr(args, "config_file", None)
    if config_file:
        command.extend(["--config-file", str(config_file)])
    if bool(getattr(args, "skip_repo_update", False)):
        command.append("--skip-repo-update")
    if bool(getattr(args, "force_sync", False)):
        command.append("--force-sync")
    if bool(getattr(args, "force_model_download", False)):
        command.append("--force-model-download")
    return tuple(command)


def launch_textual_execution_monitor(
    *,
    title: str,
    summary_lines: Sequence[str],
    expected_stages: Sequence[str] = (),
    command: Sequence[str],
    cwd: Path,
) -> int:
    """Run one child command inside the Textual execution monitor."""
    try:
        from card_framework.cli.textual_monitor import (
            TextualMonitorRequest,
            run_textual_monitor,
        )

        return run_textual_monitor(
            request=TextualMonitorRequest(
                title=title,
                summary_lines=tuple(summary_lines),
                expected_stages=tuple(expected_stages),
                command=tuple(command),
                cwd=cwd,
                env_overrides={"PYTHONUNBUFFERED": "1"},
            )
        )
    except Exception as exc:
        _emit_setup_status(
            "Textual execution monitor unavailable; falling back to direct child run. "
            f"Reason: {exc}"
        )
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            check=False,
            text=True,
        )
        if completed.returncode == 130:
            raise KeyboardInterrupt
        return int(completed.returncode)


def _selection_override_bool(
    selection: InteractiveMenuSelection,
    *,
    key: str,
    default: bool,
) -> bool:
    """Resolve one boolean override from the frontend selection payload."""
    prefix = f"{key}="
    for override in reversed(selection.overrides):
        if not override.startswith(prefix):
            continue
        value = override.split("=", 1)[1].strip().lower()
        return value in {"1", "true", "yes", "on"}
    return default


def build_textual_pipeline_expected_stages(
    selection: InteractiveMenuSelection,
    *,
    resume_stage: PipelineStartStage | None = None,
) -> tuple[str, ...]:
    """Return the operator-facing runtime stage checklist for one selection."""
    if selection.setup_only:
        return ("Setup",)
    voice_clone_enabled = _selection_override_bool(
        selection,
        key="audio.voice_clone.enabled",
        default=resolve_repo_config_boolean(
            key_path=("audio", "voice_clone", "enabled"),
            fallback=True,
        ),
    )
    interjector_enabled = _selection_override_bool(
        selection,
        key="audio.interjector.enabled",
        default=resolve_repo_config_boolean(
            key_path=("audio", "interjector", "enabled"),
            fallback=False,
        ),
    )
    defer_speaker_samples = _selection_override_bool(
        selection,
        key="audio.speaker_samples.defer_until_voice_clone",
        default=resolve_repo_config_boolean(
            key_path=("audio", "speaker_samples", "defer_until_voice_clone"),
            fallback=False,
        ),
    )
    effective_mode = selection.mode
    if selection.mode == "resume_session":
        if resume_stage == "stage-4":
            effective_mode = "stage4_interjector"
        elif resume_stage == "stage-3":
            effective_mode = "stage3_summary"
        elif resume_stage == "stage-2":
            effective_mode = "stage2_transcript"
        else:
            effective_mode = "stage1_single"
    stages: list[str] = []
    if effective_mode in {
        "stage1_single",
        "stage1_batch",
        "stage1_batch_duration_sweep",
    }:
        stages.extend(["Source separation", "Transcription", "Diarization"])
    if effective_mode in {
        "stage1_single",
        "stage1_batch",
        "stage1_batch_duration_sweep",
        "stage2_transcript",
    }:
        if defer_speaker_samples:
            stages.append("Summarizer & Critic")
        else:
            stages.extend(["Speaker samples", "Summarizer & Critic"])
        if voice_clone_enabled:
            if defer_speaker_samples:
                stages.append("Speaker samples")
            stages.append("Voice cloning")
            if interjector_enabled:
                stages.append("Interjection")
    elif effective_mode == "stage3_summary":
        stages.append("Voice cloning")
    elif effective_mode == "stage4_interjector":
        stages.append("Interjection")
    deduped: list[str] = []
    for stage in stages:
        if stage not in deduped:
            deduped.append(stage)
    return tuple(deduped)


def run_textual_pipeline_selection(
    *,
    args: argparse.Namespace,
    selection: InteractiveMenuSelection,
) -> int:
    """Run one selected pipeline workflow inside the Textual execution monitor."""
    resume_stage: PipelineStartStage | None = None
    if selection.mode == "resume_session" and selection.resume_session:
        try:
            session_store = PipelineSessionStore(resolve_pipeline_session_store_path())
            resume_stage = session_store.compute_resume_plan(
                session_id=str(selection.resume_session).strip()
            ).resume_stage
        except Exception as exc:
            _emit_setup_status(
                "Textual resume-stage detection unavailable; using the full checklist. "
                f"Reason: {exc}"
            )
    return launch_textual_execution_monitor(
        title="setup_and_run",
        summary_lines=describe_pipeline_selection(selection),
        expected_stages=build_textual_pipeline_expected_stages(
            selection,
            resume_stage=resume_stage,
        ),
        command=build_textual_setup_and_run_command(args),
        cwd=resolve_workspace_root(),
    )


def build_textual_benchmark_command(
    *,
    request: BenchmarkLaunchRequest,
    args: argparse.Namespace,
) -> tuple[str, ...]:
    """Build a monitored wrapper child command for one benchmark request."""
    uv_executable = str(getattr(args, "uv_executable", "uv")).strip() or "uv"
    git_executable = str(getattr(args, "git_executable", "git")).strip() or "git"
    if is_repo_checkout_runtime():
        command: list[str] = [
            uv_executable,
            "run",
            "python",
            "-u",
            "-m",
            SETUP_AND_RUN_MODULE,
        ]
    else:
        command = [sys.executable, "-u", "-m", SETUP_AND_RUN_MODULE]
    command.extend(
        [
            "--no-menu",
            "--uv-executable",
            uv_executable,
            "--git-executable",
            git_executable,
            "--frontend-benchmark-request-json",
            serialize_frontend_benchmark_request(request),
        ]
    )
    workspace_root = getattr(args, "workspace_root", None)
    if workspace_root:
        command.extend(["--workspace-root", str(workspace_root)])
    output_root = getattr(args, "output_root", None)
    if output_root:
        command.extend(["--output-root", str(output_root)])
    config_file = getattr(args, "config_file", None)
    if config_file:
        command.extend(["--config-file", str(config_file)])
    if bool(getattr(args, "skip_repo_update", False)):
        command.append("--skip-repo-update")
    if bool(getattr(args, "force_sync", False)):
        command.append("--force-sync")
    if bool(getattr(args, "force_model_download", False)):
        command.append("--force-model-download")
    return tuple(command)


def run_textual_benchmark_request(
    *,
    request: BenchmarkLaunchRequest,
    args: argparse.Namespace,
) -> int:
    """Run one selected benchmark workflow inside the Textual execution monitor."""
    return launch_textual_execution_monitor(
        title=request.description,
        summary_lines=describe_benchmark_request(request),
        expected_stages=("Benchmark preflight", "Benchmark run"),
        command=build_textual_benchmark_command(request=request, args=args),
        cwd=resolve_workspace_root(),
    )


def _frontend_benchmark_needs_ffmpeg(request: BenchmarkLaunchRequest) -> bool:
    """Return whether the selected benchmark path needs ffmpeg preflight checks."""
    if request.kind == "summary_matrix":
        return True
    if request.kind == "sporc_benchmark":
        return "--transcript-only" not in request.command
    if request.kind == "sporc_duration_sweep":
        return "--enable-voice-clone" in request.command
    return False


def _frontend_benchmark_needs_git(request: BenchmarkLaunchRequest) -> bool:
    """Return whether the selected benchmark path needs Git preflight checks."""
    if _frontend_benchmark_needs_voice_clone_runtime(request):
        return True
    if request.kind == "qmsum_benchmark":
        return (
            "--prepare-if-missing" in request.command
            or "--force-download-dataset" in request.command
        )
    return False


def _frontend_benchmark_needs_voice_clone_runtime(
    request: BenchmarkLaunchRequest,
) -> bool:
    """Return whether the benchmark launch path needs the IndexTTS runtime."""
    if request.kind == "summary_matrix":
        return True
    if request.kind == "sporc_benchmark":
        return "--transcript-only" not in request.command
    if request.kind == "sporc_duration_sweep":
        return "--enable-voice-clone" in request.command
    return False


def run_frontend_benchmark_request(
    *,
    request: BenchmarkLaunchRequest,
    uv_executable: str,
    git_executable: str,
    skip_repo_update: bool = False,
    force_sync: bool = False,
    force_model_download: bool = False,
) -> int:
    """Run one benchmark command emitted by the interactive frontend."""
    needs_voice_clone_runtime = _frontend_benchmark_needs_voice_clone_runtime(request)
    print("[1/3] Benchmark preflight")
    check_prerequisites(
        uv_executable=uv_executable,
        git_executable=git_executable,
        require_git=_frontend_benchmark_needs_git(request),
        require_ffmpeg=_frontend_benchmark_needs_ffmpeg(request),
    )
    print("[2/3] Benchmark runtime preparation")
    if is_repo_checkout_runtime():
        if needs_voice_clone_runtime:
            with report_loading_duration(
                "vendored IndexTTS runtime",
                reporter=_emit_setup_status,
            ):
                ensure_indextts_repo(
                    git_executable=git_executable,
                    skip_update=skip_repo_update,
                )
        with report_loading_duration(
            "runtime dependency sync",
            reporter=_emit_setup_status,
        ):
            smart_sync_projects(
                uv_executable=uv_executable,
                force_sync=force_sync,
                include_index_tts=needs_voice_clone_runtime,
            )
        if needs_voice_clone_runtime:
            with report_loading_duration(
                "IndexTTS model provisioning",
                reporter=_emit_setup_status,
            ):
                ensure_indextts_model(
                    uv_executable=uv_executable,
                    force_download=force_model_download,
                )
    else:
        print("[skip] Repo runtime preparation not needed.")
    print("[3/3] Running benchmark")
    result = subprocess.run(
        list(request.command),
        cwd=str(resolve_workspace_root()),
        check=False,
        text=True,
    )
    if result.returncode == 130:
        raise KeyboardInterrupt
    return int(result.returncode)


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


def validate_resume_session_inputs(
    *,
    resume_session: str | None,
    voiceclone_from_summary: str | None,
    audio_paths: Sequence[str],
    pass_through_overrides: Sequence[str],
) -> None:
    """Reject resume inputs that would conflict with the stored session state."""
    if not str(resume_session or "").strip():
        return
    if voiceclone_from_summary is not None:
        raise BootstrapError(
            step="input",
            message="--resume-session cannot be combined with --voiceclone-from-summary.",
        )
    if audio_paths:
        raise BootstrapError(
            step="input",
            message="--resume-session cannot be combined with --audio-path.",
        )
    conflicting_keys = {
        "pipeline.start_stage",
        "transcript_path",
        "pipeline.final_summary_path",
        "pipeline.voice_clone_manifest_path",
    }
    conflicting = [
        override
        for override in pass_through_overrides
        if any(override.startswith(f"{key}=") for key in conflicting_keys)
    ]
    if conflicting:
        raise BootstrapError(
            step="input",
            message=(
                "--resume-session cannot be combined with stage/input overrides: "
                + ", ".join(conflicting)
            ),
        )


def validate_duration_sweep_inputs(
    *,
    duration_sweep_enabled: bool,
    audio_paths: Sequence[str],
    voiceclone_from_summary: str | None,
    resume_session: str | None,
    pass_through_overrides: Sequence[str],
) -> None:
    """Reject input combinations that would break duration-sweep runs."""
    if not duration_sweep_enabled:
        return
    if not audio_paths:
        raise BootstrapError(
            step="input",
            message="--duration-sweep requires at least one --audio-path value.",
        )
    if voiceclone_from_summary is not None:
        raise BootstrapError(
            step="input",
            message=(
                "--duration-sweep cannot be combined with --voiceclone-from-summary."
            ),
        )
    if str(resume_session or "").strip():
        raise BootstrapError(
            step="input",
            message="--duration-sweep cannot be combined with --resume-session.",
        )
    if _has_override_key(pass_through_overrides, "pipeline.start_stage"):
        requested_stage = resolve_start_stage(pass_through_overrides)
        if requested_stage != "stage-1":
            raise BootstrapError(
                step="input",
                message=(
                    "--duration-sweep manages the stage transitions automatically "
                    "and only supports pipeline.start_stage=stage-1."
                ),
            )
    managed_override_keys = (
        "audio.audio_path",
        "audio.work_dir",
        "audio.output_transcript_path",
        "transcript_path",
        "session.enabled",
        "session.session_id",
        "session.store_path",
        "session.summary_output_path",
        "pipeline.final_summary_path",
        "pipeline.voice_clone_manifest_path",
        "orchestrator.target_seconds",
        "orchestrator.target_minutes",
    )
    conflicting = [
        override
        for override in pass_through_overrides
        if any(override.startswith(f"{key}=") for key in managed_override_keys)
    ]
    if conflicting:
        raise BootstrapError(
            step="input",
            message=(
                "--duration-sweep manages per-run inputs, stage targets, sessions, "
                "and artifact paths automatically. Remove these overrides: "
                + ", ".join(conflicting)
            ),
        )


def validate_batch_audio_inputs(
    *,
    audio_paths: Sequence[str],
    voiceclone_from_summary: str | None,
    resume_session: str | None,
    pass_through_overrides: Sequence[str],
) -> None:
    """Reject input combinations that would break isolated multi-audio runs."""
    if len(audio_paths) < 2:
        return
    if voiceclone_from_summary is not None:
        raise BootstrapError(
            step="input",
            message=(
                "Batch --audio-path runs cannot be combined with "
                "--voiceclone-from-summary."
            ),
        )
    if str(resume_session or "").strip():
        raise BootstrapError(
            step="input",
            message=(
                "Batch --audio-path runs cannot be combined with --resume-session."
            ),
        )
    if _has_override_key(pass_through_overrides, "pipeline.start_stage"):
        requested_stage = resolve_start_stage(pass_through_overrides)
        if requested_stage != "stage-1":
            raise BootstrapError(
                step="input",
                message=(
                    "Batch --audio-path runs only support pipeline.start_stage=stage-1."
                ),
            )
    managed_override_keys = (
        "audio.audio_path",
        "audio.work_dir",
        "audio.output_transcript_path",
        "transcript_path",
        "session.enabled",
        "session.session_id",
        "session.store_path",
        "session.summary_output_path",
        "pipeline.final_summary_path",
        "pipeline.voice_clone_manifest_path",
    )
    conflicting = [
        override
        for override in pass_through_overrides
        if any(override.startswith(f"{key}=") for key in managed_override_keys)
    ]
    if conflicting:
        raise BootstrapError(
            step="input",
            message=(
                "Batch --audio-path runs manage per-run inputs, sessions, and "
                "artifact paths automatically. Remove these overrides: "
                + ", ".join(conflicting)
            ),
        )


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
    return discover_reusable_transcript_path(resolve_workspace_root())


def _discover_existing_audio_path() -> Path | None:
    """Return the preferred reusable root-level source audio path when present."""
    for filename in PREFERRED_ROOT_AUDIO_FILENAMES:
        candidate = (resolve_workspace_root() / filename).resolve()
        if candidate.is_file():
            return candidate
    return None


def _discover_existing_summary_path() -> Path | None:
    """Return the preferred reusable root-level summary XML path when present."""
    candidate = (resolve_workspace_root() / "summary.xml").resolve()
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
        (resolve_workspace_root() / "artifacts" / "audio_stage" / "runs").glob(
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
        resolve_output_root()
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


def _replace_or_append_override(
    *,
    overrides: list[str],
    key: str,
    value: str,
) -> None:
    """Replace one override in place using last-wins semantics."""
    prefix = f"{key}="
    replacement = f"{key}={value}"
    for index in range(len(overrides) - 1, -1, -1):
        if overrides[index].startswith(prefix):
            overrides[index] = replacement
            return
    overrides.append(replacement)


def _remove_override_key(*, overrides: list[str], key: str) -> None:
    """Remove every override assignment for one key."""
    prefix = f"{key}="
    overrides[:] = [
        override for override in overrides if not override.startswith(prefix)
    ]


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


def resolve_repo_config_string(
    *,
    key_path: Sequence[str],
    fallback: str,
) -> str:
    """Read a scalar string value from the active repo config with a light parser."""
    resolved = _resolve_repo_config_scalar(key_path=key_path)
    if resolved is None:
        return fallback
    normalized = str(resolved).strip().strip('"').strip("'")
    return normalized or fallback


def resolve_repo_config_child_keys(
    *,
    key_path: Sequence[str],
) -> tuple[str, ...]:
    """Return the immediate child mapping keys for one repo-config section path."""
    config_path = resolve_repo_config_path()
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError:
        return ()

    parent_path = list(key_path)
    section_stack: list[tuple[int, str]] = []
    child_keys: list[str] = []

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

        normalized_key = key.strip()
        value = raw_value.strip()
        current_path = [*(name for _, name in section_stack), normalized_key]

        if current_path[:-1] == parent_path and normalized_key not in child_keys:
            child_keys.append(normalized_key)

        if not value:
            section_stack.append((indent, normalized_key))

    return tuple(child_keys)


def _resolve_repo_config_scalar(*, key_path: Sequence[str]) -> str | None:
    """Resolve one scalar YAML value from the active repo config when present."""
    config_path = resolve_repo_config_path()
    try:
        config_text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None

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

        normalized_key = key.strip()
        value = raw_value.strip()
        current_path = [*(name for _, name in section_stack), normalized_key]
        if current_path == list(key_path):
            return value

        if not value:
            section_stack.append((indent, normalized_key))

    return None


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
    start_stage = (
        _resolve_last_override_value(
            overrides=overrides,
            key="pipeline.start_stage",
            default="stage-1",
        )
        .strip()
        .lower()
    )
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
    session_artifacts = build_pipeline_session_artifacts(run_id=run_id)
    run_work_dir = Path(session_artifacts["work_dir"]).resolve()
    transcript_path = Path(session_artifacts["transcript_path"]).resolve()
    if has_explicit_output_root():
        eta_store_path = (
            resolve_output_root() / "audio_stage" / "eta_history.sqlite3"
        ).resolve()
    else:
        eta_store_path = (
            resolve_workspace_root()
            / "artifacts"
            / "audio_stage"
            / "eta_history.sqlite3"
        ).resolve()
    overrides = [
        f"pipeline.start_stage={start_stage}",
        "session.enabled=true",
        f"session.session_id={run_id}",
        f"session.store_path={_path_for_override(resolve_pipeline_session_store_path())}",
        (
            "session.summary_output_path="
            f"{_path_for_override(Path(session_artifacts['session_summary_path']))}"
        ),
        f"audio.work_dir={_path_for_override(run_work_dir)}",
        f"audio.eta.store_path={_path_for_override(eta_store_path)}",
        "audio.speaker_samples.enabled=true",
        "audio.speaker_samples.source_audio=vocals",
        "audio.speaker_samples.target_duration_seconds=30",
        f"audio.interjector.enabled={'true' if enable_interjector else 'false'}",
        "logging.print_to_terminal=true",
    ]
    if enable_voice_clone:
        index_tts_dir = resolve_index_tts_runner_project_dir()
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
                (f"audio.voice_clone.model_dir={_path_for_override(checkpoints_dir)}"),
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


def build_pipeline_session_options(
    *,
    requested_start_stage: PipelineStartStage,
    voice_clone_enabled: bool,
    interjector_enabled: bool,
    workspace_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build the stored session options payload used for resume planning."""
    return {
        "requested_start_stage": requested_start_stage,
        "voice_clone_enabled": voice_clone_enabled,
        "interjector_enabled": interjector_enabled,
        "workspace_root": str(workspace_root.resolve()),
        "output_root": str(output_root.resolve()),
    }


def build_resume_overrides(plan: PipelineResumePlan) -> list[str]:
    """Build the effective override list needed to resume one stored session."""
    if plan.resume_stage is None:
        raise BootstrapError(
            step="input",
            message=f"Pipeline session is already complete: {plan.session_id}",
        )
    overrides = list(plan.prepared_overrides)
    _replace_or_append_override(
        overrides=overrides,
        key="pipeline.start_stage",
        value=plan.resume_stage,
    )
    _remove_override_key(overrides=overrides, key="pipeline.final_summary_path")
    _remove_override_key(overrides=overrides, key="pipeline.voice_clone_manifest_path")
    transcript_path = plan.artifacts.get("transcript_path", "")
    session_summary_path = plan.artifacts.get("session_summary_path", "")
    voice_clone_manifest_path = plan.artifacts.get("voice_clone_manifest_path", "")
    if plan.resume_stage in {"stage-2", "stage-3"} and transcript_path:
        _replace_or_append_override(
            overrides=overrides,
            key="transcript_path",
            value=_path_for_override(Path(transcript_path)),
        )
    if plan.resume_stage in {"stage-3", "stage-4"}:
        if not session_summary_path:
            raise BootstrapError(
                step="input",
                message=(
                    "Cannot resume this session because its summary checkpoint path "
                    "is missing."
                ),
            )
        _replace_or_append_override(
            overrides=overrides,
            key="pipeline.final_summary_path",
            value=_path_for_override(Path(session_summary_path)),
        )
    if plan.resume_stage == "stage-4":
        if not voice_clone_manifest_path:
            raise BootstrapError(
                step="input",
                message=(
                    "Cannot resume this session because its voice-clone manifest "
                    "checkpoint path is missing."
                ),
            )
        _replace_or_append_override(
            overrides=overrides,
            key="pipeline.voice_clone_manifest_path",
            value=_path_for_override(Path(voice_clone_manifest_path)),
        )
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
        candidate = (resolve_workspace_root() / candidate).resolve()
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
    config_path = resolve_repo_config_path()
    command: list[str]
    if is_repo_checkout_runtime():
        command = [uv_executable, "run", "python", "-m", MAIN_MODULE]
    else:
        command = [sys.executable, "-m", MAIN_MODULE]
    command.extend(
        [
            "--config-path",
            str(config_path.parent),
            "--config-name",
            config_path.stem,
            "hydra.run.dir=.",
            "hydra.output_subdir=null",
            *overrides,
        ]
    )
    completed = subprocess.run(
        command,
        cwd=str(resolve_output_root()),
        check=False,
        text=True,
    )
    if completed.returncode == 130:
        raise KeyboardInterrupt
    if completed.returncode != 0:
        raise BootstrapError(
            step="pipeline_run",
            message=f"Command failed with exit code {completed.returncode}.",
            command=tuple(command),
            stderr_tail="See streamed command output above for details.",
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
    command: list[str]
    workspace_root = resolve_workspace_root()
    if is_repo_checkout_runtime():
        command = [uv_executable, "run", "python", "-m", CALIBRATE_MODULE]
    else:
        command = [sys.executable, "-m", CALIBRATE_MODULE]
    command.extend(
        [
            "--project-root",
            str(workspace_root),
            "--config-file",
            str(resolve_repo_config_path()),
        ]
    )
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
        cwd=workspace_root,
        stream_output=True,
    )


def _build_batch_run_result(
    *,
    index: int,
    audio_path: Path,
    output_root: Path,
    run_id: str | None,
    status: str,
    error: str = "",
) -> dict[str, Any]:
    """Build one serializable batch-result payload for the manifest."""
    payload: dict[str, Any] = {
        "index": index,
        "audio_path": str(audio_path.resolve()),
        "output_root": str(output_root.resolve()),
        "session_id": run_id,
        "status": status,
    }
    if error:
        payload["error"] = error
    if run_id is not None:
        payload["artifacts"] = build_pipeline_session_artifacts(run_id=run_id)
    return payload


def _build_expected_output_artifacts(
    *,
    output_root: Path,
    transcript_path: Path | None = None,
) -> dict[str, str]:
    """Build the deterministic output-artifact paths for one explicit output root."""
    resolved_root = output_root.resolve()
    transcript_candidate = (
        transcript_path.resolve()
        if transcript_path is not None
        else (resolved_root / "transcript.json").resolve()
    )
    return {
        "work_dir": str((resolved_root / "audio_stage").resolve()),
        "transcript_path": str(transcript_candidate),
        "workspace_summary_path": str((resolved_root / "summary.xml").resolve()),
        "session_summary_path": str((resolved_root / "summary.xml").resolve()),
        "voice_clone_manifest_path": str(
            (resolved_root / "audio_stage" / "voice_clone" / "manifest.json").resolve()
        ),
        "interjector_manifest_path": str(
            (
                resolved_root
                / "audio_stage"
                / "interjector"
                / "interjector_manifest.json"
            ).resolve()
        ),
    }


def _is_duration_sweep_output_complete(
    output_root: Path,
    *,
    voice_clone_enabled: bool,
    interjector_enabled: bool,
) -> bool:
    """Return whether one duration-sweep preset output already looks complete."""
    artifacts = _build_expected_output_artifacts(output_root=output_root)
    if not Path(artifacts["workspace_summary_path"]).is_file():
        return False
    if interjector_enabled:
        return Path(artifacts["interjector_manifest_path"]).is_file()
    if voice_clone_enabled:
        return Path(artifacts["voice_clone_manifest_path"]).is_file()
    return True


def _find_reusable_duration_sweep_transcript(
    *,
    episode_root: Path,
    preset_labels: Sequence[str],
    source_audio_path: Path,
) -> Path | None:
    """Return the best reusable transcript for one episode sweep when available."""
    fallback_candidate: Path | None = None
    for preset_label in preset_labels:
        candidate = (
            resolve_duration_sweep_preset_output_root(
                episode_root=episode_root,
                preset_label=preset_label,
            )
            / "transcript.json"
        ).resolve()
        if not candidate.is_file():
            continue
        try:
            transcript_source_audio_path = resolve_transcript_source_audio_path(
                candidate
            )
        except BootstrapError:
            continue
        if (
            transcript_source_audio_path is not None
            and transcript_source_audio_path.resolve() != source_audio_path.resolve()
        ):
            continue
        try:
            has_speaker_samples_manifest = (
                transcript_has_usable_speaker_samples_manifest(candidate)
            )
        except BootstrapError:
            has_speaker_samples_manifest = False
        if has_speaker_samples_manifest:
            return candidate
        if fallback_candidate is None:
            fallback_candidate = candidate
    return fallback_candidate


def _build_duration_sweep_run_result(
    *,
    episode_index: int,
    audio_path: Path,
    output_root: Path,
    duration_preset: str,
    target_seconds: int,
    status: str,
    start_stage: PipelineStartStage | None = None,
    session_id: str | None = None,
    transcript_path: Path | None = None,
    reused_existing: bool = False,
    error: str = "",
) -> dict[str, Any]:
    """Build one serializable duration-sweep result payload."""
    payload: dict[str, Any] = {
        "episode_index": episode_index,
        "audio_path": str(audio_path.resolve()),
        "duration_preset": duration_preset,
        "target_seconds": target_seconds,
        "output_root": str(output_root.resolve()),
        "status": status,
        "reused_existing": reused_existing,
    }
    if start_stage is not None:
        payload["start_stage"] = start_stage
    if session_id is not None:
        payload["session_id"] = session_id
        payload["artifacts"] = build_pipeline_session_artifacts(
            run_id=session_id,
            transcript_path_override=transcript_path,
        )
    elif transcript_path is not None:
        payload["transcript_path"] = str(transcript_path.resolve())
        payload["artifacts"] = _build_expected_output_artifacts(
            output_root=output_root,
            transcript_path=transcript_path,
        )
    else:
        payload["artifacts"] = _build_expected_output_artifacts(output_root=output_root)
    if error:
        payload["error"] = error
    return payload


def write_batch_manifest(
    *,
    batch_id: str,
    batch_root: Path,
    run_results: Sequence[Mapping[str, Any]],
) -> Path:
    """Persist the current batch run ledger for operator inspection."""
    manifest_path = (batch_root / "batch_manifest.json").resolve()
    succeeded_runs = sum(
        1 for result in run_results if str(result.get("status", "")).strip() == "ok"
    )
    failed_runs = sum(
        1
        for result in run_results
        if str(result.get("status", "")).strip() in {"failed", "paused"}
    )
    payload = {
        "batch_id": batch_id,
        "generated_at_utc": utc_now_iso(),
        "workspace_root": str(resolve_workspace_root()),
        "batch_output_root": str(batch_root.resolve()),
        "total_runs": len(run_results),
        "succeeded_runs": succeeded_runs,
        "failed_runs": failed_runs,
        "runs": list(run_results),
    }
    _write_json_atomic(manifest_path, payload)
    return manifest_path


def print_batch_summary(
    *,
    batch_root: Path,
    manifest_path: Path,
    run_results: Sequence[Mapping[str, Any]],
) -> None:
    """Print a compact operator-facing summary for a multi-audio batch."""
    succeeded_runs = sum(
        1 for result in run_results if str(result.get("status", "")).strip() == "ok"
    )
    failed_runs = sum(
        1
        for result in run_results
        if str(result.get("status", "")).strip() in {"failed", "paused"}
    )
    print("\n===== setup_and_run batch summary =====")
    print(f"batch_output_root: {batch_root.resolve()}")
    print(f"batch_manifest: {manifest_path.resolve()}")
    print(f"total_runs: {len(run_results)}")
    print(f"succeeded_runs: {succeeded_runs}")
    print(f"failed_runs: {failed_runs}")
    for result in run_results:
        status = str(result.get("status", "")).strip() or "unknown"
        audio_path = str(result.get("audio_path", "")).strip() or "<unknown-audio>"
        output_root = str(result.get("output_root", "")).strip() or "<unknown-output>"
        line = f"[{status}] {audio_path} -> {output_root}"
        error = str(result.get("error", "")).strip()
        if error:
            line += f" ({error})"
        print(line)


def write_duration_sweep_manifest(
    *,
    sweep_id: str,
    sweep_root: Path,
    duration_presets: Sequence[str],
    run_results: Sequence[Mapping[str, Any]],
) -> Path:
    """Persist the current duration-sweep ledger for operator inspection."""
    manifest_path = (sweep_root / "duration_sweep_manifest.json").resolve()
    completed_runs = sum(
        1
        for result in run_results
        if str(result.get("status", "")).strip() in {"ok", "skipped"}
    )
    failed_runs = sum(
        1
        for result in run_results
        if str(result.get("status", "")).strip() in {"failed", "paused"}
    )
    payload = {
        "sweep_id": sweep_id,
        "generated_at_utc": utc_now_iso(),
        "workspace_root": str(resolve_workspace_root()),
        "sweep_output_root": str(sweep_root.resolve()),
        "duration_presets": [
            {
                "label": preset_label,
                "target_seconds": DURATION_SWEEP_PRESET_SECONDS[preset_label],
            }
            for preset_label in duration_presets
        ],
        "total_runs": len(run_results),
        "completed_runs": completed_runs,
        "failed_runs": failed_runs,
        "runs": list(run_results),
    }
    _write_json_atomic(manifest_path, payload)
    return manifest_path


def print_duration_sweep_summary(
    *,
    sweep_root: Path,
    manifest_path: Path,
    run_results: Sequence[Mapping[str, Any]],
) -> None:
    """Print a compact operator-facing summary for a duration sweep."""
    completed_runs = sum(
        1
        for result in run_results
        if str(result.get("status", "")).strip() in {"ok", "skipped"}
    )
    failed_runs = sum(
        1
        for result in run_results
        if str(result.get("status", "")).strip() in {"failed", "paused"}
    )
    print("\n===== setup_and_run duration sweep summary =====")
    print(f"sweep_output_root: {sweep_root.resolve()}")
    print(f"sweep_manifest: {manifest_path.resolve()}")
    print(f"total_runs: {len(run_results)}")
    print(f"completed_runs: {completed_runs}")
    print(f"failed_runs: {failed_runs}")
    for result in run_results:
        status = str(result.get("status", "")).strip() or "unknown"
        audio_path = str(result.get("audio_path", "")).strip() or "<unknown-audio>"
        preset_label = (
            str(result.get("duration_preset", "")).strip() or "<unknown-preset>"
        )
        output_root = str(result.get("output_root", "")).strip() or "<unknown-output>"
        start_stage = str(result.get("start_stage", "")).strip()
        line = f"[{status}] [{preset_label}] {audio_path} -> {output_root}"
        if start_stage:
            line += f" ({start_stage})"
        error = str(result.get("error", "")).strip()
        if error:
            line += f" ({error})"
        print(line)


def run_audio_duration_sweep(
    *,
    audio_paths: Sequence[Path],
    duration_presets: Sequence[str],
    workspace_root: Path,
    base_output_root: Path,
    explicit_output_root: bool,
    pass_through_overrides: Sequence[str],
    voice_clone_enabled: bool,
    interjector_enabled: bool,
    live_draft_audio_enabled: bool,
    calibration_runtime_enabled: bool,
    uv_executable: str,
) -> int:
    """Run one duration sweep that reuses each episode transcript across presets."""
    global OUTPUT_ROOT

    original_output_root = OUTPUT_ROOT
    sweep_id, sweep_root = resolve_duration_sweep_output_root(
        workspace_root=workspace_root,
        output_root=base_output_root,
        explicit_output_root=explicit_output_root,
    )
    run_results: list[dict[str, Any]] = []
    manifest_path = write_duration_sweep_manifest(
        sweep_id=sweep_id,
        sweep_root=sweep_root,
        duration_presets=duration_presets,
        run_results=run_results,
    )
    print(
        "Duration sweep enabled for "
        f"{len(audio_paths)} audio file(s) across presets "
        f"{', '.join(duration_presets)}. Sweep outputs will be written under "
        f"{sweep_root}."
    )

    try:
        for episode_index, audio_path in enumerate(audio_paths, start=1):
            episode_root = resolve_duration_sweep_episode_output_root(
                sweep_root=sweep_root,
                audio_path=audio_path,
                index=episode_index,
            )
            episode_root.mkdir(parents=True, exist_ok=True)
            reusable_transcript_path = _find_reusable_duration_sweep_transcript(
                episode_root=episode_root,
                preset_labels=duration_presets,
                source_audio_path=audio_path,
            )
            bootstrap_failed = False
            print(
                f"\n===== duration sweep episode {episode_index}/{len(audio_paths)} =====\n"
                f"audio_path: {audio_path}\n"
                f"episode_root: {episode_root}"
            )

            for preset_label in duration_presets:
                target_seconds = DURATION_SWEEP_PRESET_SECONDS[preset_label]
                preset_output_root = resolve_duration_sweep_preset_output_root(
                    episode_root=episode_root,
                    preset_label=preset_label,
                )
                preset_output_root.mkdir(parents=True, exist_ok=True)
                if _is_duration_sweep_output_complete(
                    preset_output_root,
                    voice_clone_enabled=voice_clone_enabled,
                    interjector_enabled=interjector_enabled,
                ):
                    print(
                        f"[skip] preset={preset_label} already complete at "
                        f"{preset_output_root}"
                    )
                    reusable_transcript_path = (
                        reusable_transcript_path
                        or _find_reusable_duration_sweep_transcript(
                            episode_root=episode_root,
                            preset_labels=duration_presets,
                            source_audio_path=audio_path,
                        )
                    )
                    run_results.append(
                        _build_duration_sweep_run_result(
                            episode_index=episode_index,
                            audio_path=audio_path,
                            output_root=preset_output_root,
                            duration_preset=preset_label,
                            target_seconds=target_seconds,
                            status="skipped",
                            start_stage="stage-2"
                            if reusable_transcript_path is not None
                            else "stage-1",
                            transcript_path=reusable_transcript_path,
                            reused_existing=True,
                        )
                    )
                    manifest_path = write_duration_sweep_manifest(
                        sweep_id=sweep_id,
                        sweep_root=sweep_root,
                        duration_presets=duration_presets,
                        run_results=run_results,
                    )
                    continue
                if reusable_transcript_path is None and bootstrap_failed:
                    error_message = (
                        "Skipped because this episode has no reusable transcript "
                        "after an earlier stage-1 bootstrap failure."
                    )
                    print(f"[skip] preset={preset_label} {error_message}")
                    run_results.append(
                        _build_duration_sweep_run_result(
                            episode_index=episode_index,
                            audio_path=audio_path,
                            output_root=preset_output_root,
                            duration_preset=preset_label,
                            target_seconds=target_seconds,
                            status="failed",
                            start_stage="stage-1",
                            error=error_message,
                        )
                    )
                    manifest_path = write_duration_sweep_manifest(
                        sweep_id=sweep_id,
                        sweep_root=sweep_root,
                        duration_presets=duration_presets,
                        run_results=run_results,
                    )
                    continue

                OUTPUT_ROOT = preset_output_root
                records: list[StepRecord] = []
                run_id: str | None = None
                session_store: PipelineSessionStore | None = None
                start_stage: PipelineStartStage = (
                    "stage-2" if reusable_transcript_path is not None else "stage-1"
                )
                transcript_path_override = (
                    reusable_transcript_path if start_stage == "stage-2" else None
                )
                run_artifacts: dict[str, str] | None = None
                print(
                    f"\n----- preset {preset_label} ({target_seconds}s) -----\n"
                    f"start_stage: {start_stage}\n"
                    f"output_root: {preset_output_root}"
                )
                try:
                    run_id = str(uuid4())
                    run_artifacts = build_pipeline_session_artifacts(
                        run_id=run_id,
                        transcript_path_override=transcript_path_override,
                    )
                    overrides = build_run_overrides(
                        run_id=run_id,
                        start_stage=start_stage,
                        audio_path=audio_path,
                        enable_voice_clone=voice_clone_enabled,
                        enable_interjector=interjector_enabled,
                    )
                    if transcript_path_override is not None:
                        overrides.append(
                            "transcript_path="
                            f"{_path_for_override(transcript_path_override)}"
                        )
                    overrides.append(f"orchestrator.target_seconds={target_seconds}")
                    overrides.extend(pass_through_overrides)
                    if start_stage == "stage-2":
                        stage_selection_detail = (
                            "pipeline.start_stage=stage-2 because duration sweep is "
                            "reusing a prior transcript artifact."
                        )
                    else:
                        stage_selection_detail = (
                            "pipeline.start_stage=stage-1 because this episode does "
                            "not yet have a reusable transcript artifact."
                        )
                    print(f"Selected {stage_selection_detail}")
                    records.append(
                        StepRecord(
                            name="start_stage",
                            status="ok",
                            detail=stage_selection_detail,
                        )
                    )
                    records.append(
                        StepRecord(
                            name="input",
                            status="ok",
                            detail=f"audio_path={audio_path}",
                        )
                    )
                    records.append(
                        StepRecord(
                            name="duration_target",
                            status="ok",
                            detail=(
                                f"preset={preset_label}, target_seconds="
                                f"{target_seconds}"
                            ),
                        )
                    )

                    session_store = PipelineSessionStore(
                        resolve_pipeline_session_store_path()
                    )
                    session_store.create_or_replace_session(
                        session_id=run_id,
                        requested_start_stage=start_stage,
                        prepared_overrides=list(overrides),
                        artifacts=run_artifacts,
                        options=build_pipeline_session_options(
                            requested_start_stage=start_stage,
                            voice_clone_enabled=voice_clone_enabled,
                            interjector_enabled=interjector_enabled,
                            workspace_root=workspace_root,
                            output_root=preset_output_root,
                        ),
                    )

                    if start_stage == "stage-1" and calibration_runtime_enabled:
                        records.append(
                            StepRecord(
                                name="calibration",
                                status="deferred",
                                detail=(
                                    "Deferred to card_framework.cli.main after "
                                    "transcript generation so the audio stage does "
                                    "not run twice."
                                ),
                            )
                        )
                    elif live_draft_audio_enabled:
                        records.append(
                            StepRecord(
                                name="calibration",
                                status="skipped",
                                detail=(
                                    "Skipped because live stage-2/stage-3 drafting "
                                    "uses actual rendered audio durations."
                                ),
                            )
                        )
                    elif start_stage == "stage-2":
                        calibration_warning_message = build_calibration_warning_message(
                            start_stage=start_stage,
                        )
                        if calibration_warning_message is not None:
                            print(f"Note: {calibration_warning_message}")
                        with report_loading_duration(
                            "voice-clone calibration helper",
                            reporter=_emit_setup_status,
                        ):
                            run_calibration(
                                uv_executable=uv_executable,
                                transcript_path=transcript_path_override,
                                audio_path=audio_path,
                            )
                        records.append(
                            StepRecord(
                                name="calibration",
                                status="ok",
                                detail="Calibration artifact is ready.",
                            )
                        )
                    else:
                        records.append(
                            StepRecord(
                                name="calibration",
                                status="skipped",
                                detail=(
                                    "Skipped because this duration preset does not "
                                    "need a separate pre-run calibration step."
                                ),
                            )
                        )

                    run_pipeline(uv_executable=uv_executable, overrides=overrides)
                    records.append(
                        StepRecord(
                            name="pipeline_run",
                            status="ok",
                            detail="card_framework.cli.main completed successfully.",
                        )
                    )
                    print_summary(
                        records=records,
                        run_id=run_id,
                        artifacts=run_artifacts,
                    )
                    run_results.append(
                        _build_duration_sweep_run_result(
                            episode_index=episode_index,
                            audio_path=audio_path,
                            output_root=preset_output_root,
                            duration_preset=preset_label,
                            target_seconds=target_seconds,
                            status="ok",
                            start_stage=start_stage,
                            session_id=run_id,
                            transcript_path=transcript_path_override,
                        )
                    )
                    if start_stage == "stage-1":
                        reusable_transcript_path = (
                            preset_output_root / "transcript.json"
                        ).resolve()
                except BootstrapError as exc:
                    child_failure_record = None
                    if session_store is not None and run_id is not None:
                        try:
                            child_failure_record = session_store.get_session(run_id)
                        except Exception as session_exc:
                            _emit_session_store_warning(
                                action=f"failure lookup for session {run_id}",
                                exc=session_exc,
                            )
                    child_failure_already_persisted = (
                        exc.step == "pipeline_run"
                        and child_failure_record is not None
                        and child_failure_record.status
                        in {"failed", "paused", "completed"}
                    )
                    if (
                        session_store is not None
                        and run_id is not None
                        and not child_failure_already_persisted
                    ):
                        try:
                            session_store.mark_session_failed(
                                session_id=run_id,
                                current_stage=None,
                                reason=exc.message,
                            )
                        except Exception as session_exc:
                            _emit_session_store_warning(
                                action=f"failure persistence for session {run_id}",
                                exc=session_exc,
                            )
                    records.append(
                        StepRecord(name=exc.step, status="failed", detail=exc.message)
                    )
                    print(exc.render())
                    print_summary(
                        records=records,
                        run_id=run_id,
                        artifacts=run_artifacts,
                    )
                    run_results.append(
                        _build_duration_sweep_run_result(
                            episode_index=episode_index,
                            audio_path=audio_path,
                            output_root=preset_output_root,
                            duration_preset=preset_label,
                            target_seconds=target_seconds,
                            status="failed",
                            start_stage=start_stage,
                            session_id=run_id,
                            transcript_path=transcript_path_override,
                            error=exc.message,
                        )
                    )
                    if start_stage == "stage-1":
                        bootstrap_failed = True
                        reusable_transcript_path = None
                except KeyboardInterrupt:
                    if session_store is not None and run_id is not None:
                        session_store.mark_session_paused(
                            session_id=run_id,
                            current_stage=None,
                            reason="Pipeline interrupted by user.",
                        )
                    print_summary(
                        records=records,
                        run_id=run_id,
                        interrupted=True,
                        artifacts=run_artifacts,
                    )
                    run_results.append(
                        _build_duration_sweep_run_result(
                            episode_index=episode_index,
                            audio_path=audio_path,
                            output_root=preset_output_root,
                            duration_preset=preset_label,
                            target_seconds=target_seconds,
                            status="paused",
                            start_stage=start_stage,
                            session_id=run_id,
                            transcript_path=transcript_path_override,
                            error="Pipeline interrupted by user.",
                        )
                    )
                    manifest_path = write_duration_sweep_manifest(
                        sweep_id=sweep_id,
                        sweep_root=sweep_root,
                        duration_presets=duration_presets,
                        run_results=run_results,
                    )
                    print_duration_sweep_summary(
                        sweep_root=sweep_root,
                        manifest_path=manifest_path,
                        run_results=run_results,
                    )
                    raise
                manifest_path = write_duration_sweep_manifest(
                    sweep_id=sweep_id,
                    sweep_root=sweep_root,
                    duration_presets=duration_presets,
                    run_results=run_results,
                )

        print_duration_sweep_summary(
            sweep_root=sweep_root,
            manifest_path=manifest_path,
            run_results=run_results,
        )
        return (
            0
            if all(result.get("status") in {"ok", "skipped"} for result in run_results)
            else 1
        )
    finally:
        OUTPUT_ROOT = original_output_root


def run_audio_batch(
    *,
    audio_paths: Sequence[Path],
    workspace_root: Path,
    base_output_root: Path,
    explicit_output_root: bool,
    pass_through_overrides: Sequence[str],
    voice_clone_enabled: bool,
    interjector_enabled: bool,
    live_draft_audio_enabled: bool,
    calibration_runtime_enabled: bool,
    uv_executable: str,
) -> int:
    """Run the stage-1 pipeline sequentially for multiple audio inputs."""
    global OUTPUT_ROOT

    original_output_root = OUTPUT_ROOT
    batch_id, batch_root = resolve_batch_output_root(
        workspace_root=workspace_root,
        output_root=base_output_root,
        explicit_output_root=explicit_output_root,
    )
    run_results: list[dict[str, Any]] = []
    manifest_path = write_batch_manifest(
        batch_id=batch_id,
        batch_root=batch_root,
        run_results=run_results,
    )
    print(
        "Batch mode enabled for "
        f"{len(audio_paths)} audio file(s). Batch outputs will be written under "
        f"{batch_root}."
    )

    try:
        for index, audio_path in enumerate(audio_paths, start=1):
            run_output_root = resolve_batch_run_output_root(
                batch_root=batch_root,
                audio_path=audio_path,
                index=index,
            )
            run_output_root.mkdir(parents=True, exist_ok=True)
            OUTPUT_ROOT = run_output_root
            records: list[StepRecord] = []
            run_id: str | None = None
            session_store: PipelineSessionStore | None = None

            print(
                f"\n===== batch run {index}/{len(audio_paths)} =====\n"
                f"audio_path: {audio_path}\n"
                f"output_root: {run_output_root}"
            )

            try:
                run_id = str(uuid4())
                overrides = build_run_overrides(
                    run_id=run_id,
                    start_stage="stage-1",
                    audio_path=audio_path,
                    enable_voice_clone=voice_clone_enabled,
                    enable_interjector=interjector_enabled,
                )
                overrides.extend(pass_through_overrides)
                stage_selection_detail = (
                    "pipeline.start_stage=stage-1 because multiple --audio-path "
                    "inputs were provided."
                )
                print(f"Selected {stage_selection_detail}")
                records.append(
                    StepRecord(
                        name="start_stage",
                        status="ok",
                        detail=stage_selection_detail,
                    )
                )
                records.append(
                    StepRecord(
                        name="input",
                        status="ok",
                        detail=f"audio_path={audio_path}",
                    )
                )

                session_store = PipelineSessionStore(
                    resolve_pipeline_session_store_path()
                )
                session_store.create_or_replace_session(
                    session_id=run_id,
                    requested_start_stage="stage-1",
                    prepared_overrides=list(overrides),
                    artifacts=build_pipeline_session_artifacts(run_id=run_id),
                    options=build_pipeline_session_options(
                        requested_start_stage="stage-1",
                        voice_clone_enabled=voice_clone_enabled,
                        interjector_enabled=interjector_enabled,
                        workspace_root=workspace_root,
                        output_root=run_output_root,
                    ),
                )

                if calibration_runtime_enabled:
                    records.append(
                        StepRecord(
                            name="calibration",
                            status="deferred",
                            detail=(
                                "Deferred to card_framework.cli.main after transcript "
                                "generation so the audio stage does not run twice."
                            ),
                        )
                    )
                elif live_draft_audio_enabled:
                    records.append(
                        StepRecord(
                            name="calibration",
                            status="skipped",
                            detail=(
                                "Skipped because live stage-2/stage-3 drafting uses "
                                "actual rendered audio durations."
                            ),
                        )
                    )
                else:
                    records.append(
                        StepRecord(
                            name="calibration",
                            status="skipped",
                            detail="Skipped because this batch run does not need calibration.",
                        )
                    )

                run_pipeline(uv_executable=uv_executable, overrides=overrides)
                records.append(
                    StepRecord(
                        name="pipeline_run",
                        status="ok",
                        detail="card_framework.cli.main completed successfully.",
                    )
                )
                print_summary(records=records, run_id=run_id)
                run_results.append(
                    _build_batch_run_result(
                        index=index,
                        audio_path=audio_path,
                        output_root=run_output_root,
                        run_id=run_id,
                        status="ok",
                    )
                )
            except BootstrapError as exc:
                child_failure_record = None
                if session_store is not None and run_id is not None:
                    try:
                        child_failure_record = session_store.get_session(run_id)
                    except Exception as session_exc:
                        _emit_session_store_warning(
                            action=f"failure lookup for session {run_id}",
                            exc=session_exc,
                        )
                child_failure_already_persisted = (
                    exc.step == "pipeline_run"
                    and child_failure_record is not None
                    and child_failure_record.status in {"failed", "paused", "completed"}
                )
                if (
                    session_store is not None
                    and run_id is not None
                    and not child_failure_already_persisted
                ):
                    try:
                        session_store.mark_session_failed(
                            session_id=run_id,
                            current_stage=None,
                            reason=exc.message,
                        )
                    except Exception as session_exc:
                        _emit_session_store_warning(
                            action=f"failure persistence for session {run_id}",
                            exc=session_exc,
                        )
                records.append(
                    StepRecord(name=exc.step, status="failed", detail=exc.message)
                )
                print(exc.render())
                print_summary(records=records, run_id=run_id)
                run_results.append(
                    _build_batch_run_result(
                        index=index,
                        audio_path=audio_path,
                        output_root=run_output_root,
                        run_id=run_id,
                        status="failed",
                        error=exc.message,
                    )
                )
            except KeyboardInterrupt:
                if session_store is not None and run_id is not None:
                    session_store.mark_session_paused(
                        session_id=run_id,
                        current_stage=None,
                        reason="Pipeline interrupted by user.",
                    )
                print_summary(records=records, run_id=run_id, interrupted=True)
                run_results.append(
                    _build_batch_run_result(
                        index=index,
                        audio_path=audio_path,
                        output_root=run_output_root,
                        run_id=run_id,
                        status="paused",
                        error="Pipeline interrupted by user.",
                    )
                )
                manifest_path = write_batch_manifest(
                    batch_id=batch_id,
                    batch_root=batch_root,
                    run_results=run_results,
                )
                print_batch_summary(
                    batch_root=batch_root,
                    manifest_path=manifest_path,
                    run_results=run_results,
                )
                raise
            manifest_path = write_batch_manifest(
                batch_id=batch_id,
                batch_root=batch_root,
                run_results=run_results,
            )

        print_batch_summary(
            batch_root=batch_root,
            manifest_path=manifest_path,
            run_results=run_results,
        )
        return 0 if all(result.get("status") == "ok" for result in run_results) else 1
    finally:
        OUTPUT_ROOT = original_output_root


def print_summary(
    *,
    records: Sequence[StepRecord],
    run_id: str | None,
    interrupted: bool = False,
    artifacts: Mapping[str, str] | None = None,
) -> None:
    """Print compact step summary and expected output artifact locations."""
    print("\n===== setup_and_run summary =====")
    for record in records:
        print(f"[{record.status}] {record.name}: {record.detail}")
    if interrupted:
        print("[info] Execution interrupted by user.")
    if run_id is None:
        return

    resolved_artifacts = (
        dict(artifacts)
        if artifacts is not None
        else build_pipeline_session_artifacts(run_id=run_id)
    )
    run_work_dir = Path(resolved_artifacts["work_dir"]).resolve()
    transcript_path = Path(resolved_artifacts["transcript_path"]).resolve()
    speaker_manifest = run_work_dir / "speaker_samples" / "manifest.json"
    voice_clone_manifest = Path(
        resolved_artifacts["voice_clone_manifest_path"]
    ).resolve()
    interjector_manifest = Path(
        resolved_artifacts["interjector_manifest_path"]
    ).resolve()
    session_summary_path = Path(
        resolved_artifacts.get(
            "session_summary_path", transcript_path.parent / "summary.xml"
        )
    ).resolve()

    print(f"session_id: {run_id}")
    print(f"work_dir: {run_work_dir}")
    print(f"transcript: {transcript_path}")
    print(f"session_summary: {session_summary_path}")
    print(f"speaker_samples_manifest: {speaker_manifest}")
    print(f"voice_clone_manifest: {voice_clone_manifest}")
    print(f"interjector_manifest: {interjector_manifest}")


def main(argv: Sequence[str] | None = None) -> int:
    """Run setup phases and, by default, execute the full pipeline."""
    global CONFIG_FILE_PATH, WORKSPACE_ROOT, OUTPUT_ROOT
    args = parse_args(argv)
    workspace_root = Path.cwd().resolve()
    if args.workspace_root:
        workspace_root = resolve_directory_input(
            args.workspace_root,
            field_name="Workspace root",
        )
    output_root = workspace_root
    if args.output_root:
        output_root_candidate = Path(args.output_root).expanduser()
        if not output_root_candidate.is_absolute():
            output_root_candidate = (workspace_root / output_root_candidate).resolve()
        output_root = output_root_candidate.resolve()
    config_file_path: Path | None = None
    if args.config_file:
        config_file_path = resolve_file_input(
            args.config_file,
            field_name="Config file",
            base_dir=workspace_root,
        )
    WORKSPACE_ROOT = workspace_root
    CONFIG_FILE_PATH = config_file_path
    if args.menu and not is_interactive_console():
        print("[input] --menu requires an interactive terminal.")
        return 1
    frontend_benchmark_request: BenchmarkLaunchRequest | None = None
    if args.frontend_benchmark_request_json:
        frontend_benchmark_request = deserialize_frontend_benchmark_request(
            str(args.frontend_benchmark_request_json)
        )
    if should_launch_interactive_menu(args):
        frontend_result = launch_interactive_frontend(
            uv_executable=str(args.uv_executable).strip() or "uv"
        )
        if frontend_result.action == "cancel":
            return 0
        if frontend_result.action == "benchmark":
            if frontend_result.benchmark_request is None:
                print(
                    "[input] Interactive frontend did not provide a benchmark command."
                )
                return 1
            frontend_benchmark_request = frontend_result.benchmark_request
        if frontend_result.pipeline_selection is None:
            if frontend_benchmark_request is None:
                print(
                    "[input] Interactive frontend did not provide a pipeline selection."
                )
                return 1
        if frontend_benchmark_request is None:
            if frontend_result.pipeline_selection is None:
                print(
                    "[input] Interactive frontend did not provide a pipeline selection."
                )
                return 1
            args = apply_interactive_menu_selection(
                args, frontend_result.pipeline_selection
            )
            if frontend_result.ui_mode == "textual":
                return run_textual_pipeline_selection(
                    args=args,
                    selection=frontend_result.pipeline_selection,
                )
        elif frontend_result.ui_mode == "textual":
            return run_textual_benchmark_request(
                request=frontend_benchmark_request,
                args=args,
            )
        output_root = workspace_root
        if args.output_root:
            output_root_candidate = Path(args.output_root).expanduser()
            if not output_root_candidate.is_absolute():
                output_root_candidate = (
                    workspace_root / output_root_candidate
                ).resolve()
            output_root = output_root_candidate.resolve()
    OUTPUT_ROOT = output_root if args.output_root else None
    raw_audio_paths = tuple(args.audio_path or [])
    cli_audio_path = raw_audio_paths[0] if len(raw_audio_paths) == 1 else None
    batch_mode = len(raw_audio_paths) > 1
    records: list[StepRecord] = []
    run_id: str | None = None
    session_store: PipelineSessionStore | None = None
    resume_plan: PipelineResumePlan | None = None

    try:
        if frontend_benchmark_request is not None:
            return run_frontend_benchmark_request(
                request=frontend_benchmark_request,
                uv_executable=str(args.uv_executable).strip() or "uv",
                git_executable=str(args.git_executable).strip() or "git",
                skip_repo_update=bool(args.skip_repo_update),
                force_sync=bool(args.force_sync),
                force_model_download=bool(args.force_model_download),
            )
        pass_through_overrides = normalize_cli_overrides(args.override)
        if args.duration_presets and not args.duration_sweep:
            raise BootstrapError(
                step="input",
                message=(
                    "--duration-presets requires --duration-sweep so the wrapper "
                    "knows to fan out duration targets."
                ),
            )
        duration_sweep_presets = (
            parse_duration_sweep_presets(args.duration_presets)
            if bool(args.duration_sweep)
            else ()
        )
        validate_resume_session_inputs(
            resume_session=args.resume_session,
            voiceclone_from_summary=args.voiceclone_from_summary,
            audio_paths=raw_audio_paths,
            pass_through_overrides=pass_through_overrides,
        )
        validate_duration_sweep_inputs(
            duration_sweep_enabled=bool(args.duration_sweep),
            audio_paths=raw_audio_paths,
            voiceclone_from_summary=args.voiceclone_from_summary,
            resume_session=args.resume_session,
            pass_through_overrides=pass_through_overrides,
        )
        validate_batch_audio_inputs(
            audio_paths=raw_audio_paths,
            voiceclone_from_summary=args.voiceclone_from_summary,
            resume_session=args.resume_session,
            pass_through_overrides=pass_through_overrides,
        )
        multi_audio_paths = (
            resolve_audio_inputs(raw_audio_paths)
            if batch_mode or bool(args.duration_sweep)
            else ()
        )
        preflight_shortcut_overrides: list[str] = []
        if args.voiceclone_from_summary is not None:
            preflight_shortcut_overrides.extend(
                [
                    "pipeline.start_stage=stage-3",
                    "audio.speaker_samples.enabled=false",
                    "audio.voice_clone.enabled=true",
                ]
            )
        if args.resume_session:
            session_store = PipelineSessionStore(resolve_pipeline_session_store_path())
            resume_plan = session_store.compute_resume_plan(
                session_id=str(args.resume_session).strip()
            )
            run_id = resume_plan.session_id
            if resume_plan.is_completed:
                records.append(
                    StepRecord(
                        name="pipeline_run",
                        status="skipped",
                        detail=(
                            f"Session {run_id} is already complete; no resume needed."
                        ),
                    )
                )
                print_summary(records=records, run_id=run_id)
                return 0
            effective_start_stage = resume_plan.resume_stage
            if effective_start_stage is None:
                raise BootstrapError(
                    step="input",
                    message=(
                        "Resume planning did not resolve a next stage for session "
                        f"{run_id}."
                    ),
                )
        elif batch_mode or bool(args.duration_sweep):
            effective_start_stage = "stage-1"
        else:
            effective_start_stage = resolve_bootstrap_start_stage(
                shortcut_overrides=preflight_shortcut_overrides,
                pass_through_overrides=pass_through_overrides,
                cli_audio_path=cli_audio_path,
            )
        combined_bootstrap_overrides = [
            *preflight_shortcut_overrides,
            *pass_through_overrides,
        ]
        effective_boolean_overrides = (
            list(resume_plan.prepared_overrides)
            if resume_plan is not None
            else combined_bootstrap_overrides
        )
        voice_clone_enabled = resolve_repo_backed_boolean_override(
            effective_boolean_overrides,
            key="audio.voice_clone.enabled",
            repo_key_path=("audio", "voice_clone", "enabled"),
            fallback=True,
        )
        live_draft_audio_enabled = resolve_repo_backed_boolean_override(
            effective_boolean_overrides,
            key="audio.voice_clone.live_drafting.enabled",
            repo_key_path=("audio", "voice_clone", "live_drafting", "enabled"),
            fallback=True,
        )
        live_draft_audio_enabled = voice_clone_enabled and live_draft_audio_enabled
        interjector_enabled = resolve_repo_backed_boolean_override(
            effective_boolean_overrides,
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
            effective_boolean_overrides,
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
            uv_executable=args.uv_executable,
            git_executable=args.git_executable,
            require_git=synthesis_enabled,
            require_ffmpeg=ffmpeg_required,
        )
        records.append(
            StepRecord(
                name="preflight", status="ok", detail="Tools and paths validated."
            )
        )

        print("[2/6] IndexTTS repository sync")
        if synthesis_enabled:
            with report_loading_duration(
                "vendored IndexTTS runtime",
                reporter=_emit_setup_status,
            ):
                repo_result = ensure_indextts_repo(
                    git_executable=args.git_executable,
                    skip_update=bool(args.skip_repo_update),
                )
            repo_detail = (
                f"cloned={repo_result.cloned}, updated={repo_result.updated}, "
                f"pull_skipped_dirty={repo_result.pull_skipped_dirty}, "
                f"lfs_pulled={repo_result.lfs_pulled}"
            )
            records.append(
                StepRecord(name="repo_sync", status="ok", detail=repo_detail)
            )
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
        if synthesis_enabled and not is_repo_checkout_runtime():
            layout = resolve_runtime_layout()
            ensure_index_tts_runtime(
                layout=layout,
                uv_executable=args.uv_executable,
                python_executable=sys.executable,
                force_sync=bool(args.force_sync),
                force_model_download=bool(args.force_model_download),
                reporter=_emit_setup_status,
            )
            detail = f"packaged runtime ready at {layout.vendor_runtime_dir}"
        else:
            with report_loading_duration(
                "runtime dependency sync",
                reporter=_emit_setup_status,
            ):
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
        if synthesis_enabled and not is_repo_checkout_runtime():
            records.append(
                StepRecord(
                    name="model_download",
                    status="ok",
                    detail=f"packaged checkpoints ready at {resolve_index_tts_checkpoints_dir()}",
                )
            )
        elif synthesis_enabled:
            with report_loading_duration(
                "IndexTTS model provisioning",
                reporter=_emit_setup_status,
            ):
                model_result = ensure_indextts_model(
                    uv_executable=args.uv_executable,
                    force_download=bool(args.force_model_download),
                )
            model_detail = (
                f"downloaded={model_result.downloaded}, source={model_result.source}"
            )
            records.append(
                StepRecord(name="model_download", status="ok", detail=model_detail)
            )
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

        if args.duration_sweep:
            return run_audio_duration_sweep(
                audio_paths=multi_audio_paths,
                duration_presets=duration_sweep_presets,
                workspace_root=workspace_root,
                base_output_root=output_root,
                explicit_output_root=bool(args.output_root),
                pass_through_overrides=pass_through_overrides,
                voice_clone_enabled=voice_clone_enabled,
                interjector_enabled=interjector_enabled,
                live_draft_audio_enabled=live_draft_audio_enabled,
                calibration_runtime_enabled=calibration_runtime_enabled,
                uv_executable=args.uv_executable,
            )

        if batch_mode:
            return run_audio_batch(
                audio_paths=multi_audio_paths,
                workspace_root=workspace_root,
                base_output_root=output_root,
                explicit_output_root=bool(args.output_root),
                pass_through_overrides=pass_through_overrides,
                voice_clone_enabled=voice_clone_enabled,
                interjector_enabled=interjector_enabled,
                live_draft_audio_enabled=live_draft_audio_enabled,
                calibration_runtime_enabled=calibration_runtime_enabled,
                uv_executable=args.uv_executable,
            )

        if session_store is None:
            session_store = PipelineSessionStore(resolve_pipeline_session_store_path())

        print("[5/7] Resolve runtime input")
        if resume_plan is not None:
            run_id = resume_plan.session_id
            overrides = build_resume_overrides(resume_plan)
            stage_selection_detail = (
                f"Resuming session {run_id} from {effective_start_stage}."
            )
        else:
            run_id = str(uuid4())
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
                cli_audio_path=cli_audio_path,
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
        session_transcript_override = resolve_session_transcript_artifact_override(
            start_stage=effective_start_stage,
            transcript_path=runtime_transcript_path,
        )
        resolved_audio_path = resolve_audio_override_path(overrides=overrides)
        if requires_audio_path_input(overrides):
            if resolved_audio_path is None:
                resolved_audio_path = prompt_or_validate_audio_path(cli_audio_path)
                overrides.append(
                    f"audio.audio_path={_path_for_override(resolved_audio_path)}"
                )
                input_detail = f"audio_path={resolved_audio_path}"
            else:
                input_detail = f"audio_path={resolved_audio_path} (override)"
            records.append(StepRecord(name="input", status="ok", detail=input_detail))
        elif cli_audio_path is not None:
            resolved_audio_path = resolve_audio_input(cli_audio_path)
            overrides.append(
                f"audio.audio_path={_path_for_override(resolved_audio_path)}"
            )
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
                detail_suffix = (
                    "from transcript metadata for stage-2 speaker-sample inference"
                )
            else:
                resolved_audio_path = _discover_existing_audio_path()
                if resolved_audio_path is not None:
                    detail_suffix = "auto-detected for stage-2 speaker-sample inference"
            if resolved_audio_path is None:
                resolved_audio_path = prompt_or_validate_audio_path(None)
                detail_suffix = "required for stage-2 speaker-sample inference"
            overrides.append(
                f"audio.audio_path={_path_for_override(resolved_audio_path)}"
            )
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

        if (
            session_store is not None
            and run_id is not None
            and resume_plan is not None
            and effective_start_stage is not None
        ):
            session_store.cleanup_for_stage_restart(
                session_id=run_id,
                stage_name=effective_start_stage,
            )

        if session_store is not None and run_id is not None and resume_plan is None:
            session_store.create_or_replace_session(
                session_id=run_id,
                requested_start_stage=effective_start_stage,
                prepared_overrides=list(overrides),
                artifacts=build_pipeline_session_artifacts(
                    run_id=run_id,
                    transcript_path_override=session_transcript_override,
                ),
                options=build_pipeline_session_options(
                    requested_start_stage=effective_start_stage,
                    voice_clone_enabled=voice_clone_enabled,
                    interjector_enabled=interjector_enabled,
                    workspace_root=workspace_root,
                    output_root=output_root,
                ),
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
            with report_loading_duration(
                "voice-clone calibration helper",
                reporter=_emit_setup_status,
            ):
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
        child_failure_record = None
        if session_store is not None and run_id is not None:
            try:
                child_failure_record = session_store.get_session(run_id)
            except Exception as session_exc:
                _emit_session_store_warning(
                    action=f"failure lookup for session {run_id}",
                    exc=session_exc,
                )
        child_failure_already_persisted = (
            exc.step == "pipeline_run"
            and child_failure_record is not None
            and child_failure_record.status in {"failed", "paused", "completed"}
        )
        if (
            session_store is not None
            and run_id is not None
            and not child_failure_already_persisted
        ):
            try:
                session_store.mark_session_failed(
                    session_id=run_id,
                    current_stage=None,
                    reason=exc.message,
                )
            except Exception as session_exc:
                _emit_session_store_warning(
                    action=f"failure persistence for session {run_id}",
                    exc=session_exc,
                )
        records.append(StepRecord(name=exc.step, status="failed", detail=exc.message))
        print(exc.render())
        print_summary(records=records, run_id=run_id)
        return 1
    except KeyboardInterrupt:
        if batch_mode:
            return 130
        if session_store is not None and run_id is not None:
            try:
                session_store.mark_session_paused(
                    session_id=run_id,
                    current_stage=None,
                    reason="Pipeline interrupted by user.",
                )
            except Exception as session_exc:
                _emit_session_store_warning(
                    action=f"pause persistence for session {run_id}",
                    exc=session_exc,
                )
        print_summary(records=records, run_id=run_id, interrupted=True)
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
