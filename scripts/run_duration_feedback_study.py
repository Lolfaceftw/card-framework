"""Run the paired estimated-versus-measured duration-feedback study.

References:
    [1] S. W. Roberts, "Control Chart Tests Based on Geometric Moving
        Averages," Technometrics, vol. 1, no. 3, pp. 239-250, 1959,
        doi: 10.1080/00401706.1959.10489860.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from omegaconf import OmegaConf

from card_framework.audio_pipeline.runtime import probe_audio_duration_ms
from card_framework.shared.paths import REPO_ROOT, SPORC_HUMAN_EVAL_ASSETS_DIR

CONTAINER_REPO_ROOT = PurePosixPath("/workspace/card-framework")
CONTAINER_EMOTION2VEC_MODEL_PATH = PurePosixPath(
    "/root/.cache/huggingface/modelscope/models/iic/emotion2vec_plus_large"
)
CONTAINER_AUDEERING_MODEL_PATH = PurePosixPath(
    "/root/.cache/huggingface/transformers/"
    "models--audeering--wav2vec2-large-robust-12-ft-emotion-msp-dim/"
    "snapshots/6eba34a2485ea31cb03600241787c3a5edab8626"
)
DEFAULT_OUTPUT_PARENT = (
    REPO_ROOT / "artifacts" / "duration_feedback_study"
).resolve()
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_CONTEXT_WINDOW_TOKENS = 262_144
DEFAULT_CONDITIONS = ("estimated", "measured")
DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0
DEFAULT_PROGRESS_EMA_ALPHA = 0.3
DEFAULT_CONNECTION_MAX_ATTEMPTS = 12
DEFAULT_CONNECTION_RETRY_DELAY_SECONDS = 30.0
DEFAULT_CONNECTION_STABILITY_SUCCESSES = 3
TEMPORARY_CONNECTION_EXIT_CODE = 75
FINAL_AUDIO_RELATIVE_PATH = Path(
    "audio_stage/interjector/voice_cloned_interjected.wav"
)


@dataclass(slots=True, frozen=True)
class PreparedInput:
    """Describe one immutable source episode and duration target.

    Attributes:
        episode_index: One-based episode index from the tracked asset package.
        episode_title: Human-readable episode title.
        episode_label: Filesystem-safe episode directory name.
        duration_preset: Requested preset label, such as ``30s``.
        target_seconds: Exact requested duration in seconds.
        transcript_path: Shared prepared transcript used by both conditions.
        calibration_path: Episode-specific WPM calibration artifact.
    """

    episode_index: int
    episode_title: str
    episode_label: str
    duration_preset: str
    target_seconds: int
    transcript_path: Path
    calibration_path: Path


@dataclass(slots=True, frozen=True)
class StudyProgressSnapshot:
    """Describe one immutable view of study execution progress.

    Attributes:
        completed_attempts: Attempts that have exited and entered the study
            record, including failed attempts.
        total_attempts: Attempts scheduled by the selected episodes, targets,
            conditions, and repetitions.
        current_attempt_id: Active attempt identifier, or an empty string when
            execution is between attempts.
        current_attempt_elapsed_seconds: Wall time consumed by the active
            attempt, or ``None`` when no attempt is active.
        elapsed_seconds: Wall time since the tracker was created.
        ema_attempt_seconds: Exponentially weighted mean runtime of completed
            attempts, or ``None`` before the first attempt completes.
        eta_seconds: Approximate time remaining under the current runtime EMA,
            or ``None`` before an observed runtime is available.
    """

    completed_attempts: int
    total_attempts: int
    current_attempt_id: str
    current_attempt_elapsed_seconds: float | None
    elapsed_seconds: float
    ema_attempt_seconds: float | None
    eta_seconds: float | None


class StudyProgressTracker:
    """Print periodic study progress and an EWMA-based runtime estimate.

    The estimate applies one exponential moving average to completed attempt
    runtimes. It is an operator aid, not a research measurement: 30-, 60-, and
    300-second targets can have materially different runtimes. Failed attempts
    remain part of the estimate because the runner still spends wall time on
    them and counts them in the scheduled study.
    """

    def __init__(
        self,
        *,
        total_attempts: int,
        interval_seconds: float,
        ema_alpha: float,
    ) -> None:
        """Initialize progress state for one fixed study schedule.

        Args:
            total_attempts: Positive number of attempts in the frozen schedule.
            interval_seconds: Positive minimum time between heartbeat lines.
            ema_alpha: Weight assigned to the newest completed runtime. Values
                must be greater than zero and at most one. The project default
                of ``0.3`` is an operator-facing smoothing choice.

        Raises:
            ValueError: Any numeric setting falls outside its accepted range.
        """
        if total_attempts <= 0:
            raise ValueError("total_attempts must be greater than zero.")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero.")
        if not 0 < ema_alpha <= 1:
            raise ValueError("ema_alpha must be greater than zero and at most one.")
        self.total_attempts = int(total_attempts)
        self.interval_seconds = float(interval_seconds)
        self.ema_alpha = float(ema_alpha)
        self.completed_attempts = 0
        self.ema_attempt_seconds: float | None = None
        self.current_attempt_id = ""
        self.current_attempt_started_monotonic: float | None = None
        self.started_monotonic = time.monotonic()
        self.last_render_monotonic = 0.0

    def start_attempt(self, attempt_id: str) -> None:
        """Mark one scheduled attempt active and print its initial status.

        Args:
            attempt_id: Unique identifier already written to the study summary.

        Raises:
            RuntimeError: Another attempt remains active.
            ValueError: The identifier is empty or all attempts already ended.
        """
        if self.current_attempt_started_monotonic is not None:
            raise RuntimeError("Cannot start a second progress attempt concurrently.")
        if not attempt_id.strip():
            raise ValueError("attempt_id must be non-empty.")
        if self.completed_attempts >= self.total_attempts:
            raise ValueError("All scheduled attempts are already complete.")
        self.current_attempt_id = attempt_id
        self.current_attempt_started_monotonic = time.monotonic()
        self.render(force=True)

    def heartbeat(self) -> None:
        """Print a periodic line when the configured interval has elapsed."""
        self.render(force=False)

    def finish_attempt(self, *, elapsed_seconds: float, outcome: str) -> None:
        """Record one exited attempt, update the EWMA, and print completion.

        Args:
            elapsed_seconds: Positive measured wall time for the attempt.
            outcome: Persisted attempt status included in the terminal line.

        Raises:
            RuntimeError: No attempt is currently active.
            ValueError: The measured runtime or outcome is invalid.
        """
        if self.current_attempt_started_monotonic is None:
            raise RuntimeError("Cannot finish progress without an active attempt.")
        if elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be greater than zero.")
        if not outcome.strip():
            raise ValueError("outcome must be non-empty.")
        # IEEE citation: The geometrically decaying runtime weights follow [1].
        if self.ema_attempt_seconds is None:
            self.ema_attempt_seconds = float(elapsed_seconds)
        else:
            self.ema_attempt_seconds = (
                self.ema_alpha * float(elapsed_seconds)
                + (1.0 - self.ema_alpha) * self.ema_attempt_seconds
            )
        finished_attempt_id = self.current_attempt_id
        self.completed_attempts += 1
        self.current_attempt_id = ""
        self.current_attempt_started_monotonic = None
        self.render(force=True, outcome=f"{finished_attempt_id}:{outcome}")

    def snapshot(self, *, now_monotonic: float | None = None) -> StudyProgressSnapshot:
        """Build a progress snapshot using one monotonic-clock observation.

        Args:
            now_monotonic: Explicit clock value for deterministic tests. When
                omitted, the process monotonic clock is read once.

        Returns:
            Current counts, elapsed values, EWMA, and approximate remaining
            runtime. ETA is unavailable before the first attempt completes.
        """
        now = time.monotonic() if now_monotonic is None else now_monotonic
        current_elapsed = None
        if self.current_attempt_started_monotonic is not None:
            current_elapsed = max(
                0.0,
                now - self.current_attempt_started_monotonic,
            )
        eta_seconds = self._eta_seconds(current_elapsed)
        return StudyProgressSnapshot(
            completed_attempts=self.completed_attempts,
            total_attempts=self.total_attempts,
            current_attempt_id=self.current_attempt_id,
            current_attempt_elapsed_seconds=current_elapsed,
            elapsed_seconds=max(0.0, now - self.started_monotonic),
            ema_attempt_seconds=self.ema_attempt_seconds,
            eta_seconds=eta_seconds,
        )

    def render(self, *, force: bool, outcome: str = "") -> None:
        """Print one progress line when forced or when the interval expires.

        Args:
            force: Emit regardless of the time since the previous line.
            outcome: Optional just-finished attempt identifier and status.
        """
        now = time.monotonic()
        if not force and now - self.last_render_monotonic < self.interval_seconds:
            return
        self.last_render_monotonic = now
        snapshot = self.snapshot(now_monotonic=now)
        percent = 100.0 * snapshot.completed_attempts / snapshot.total_attempts
        active = snapshot.current_attempt_id or "-"
        suffix = f" | finished={outcome}" if outcome else ""
        print(
            "[study-progress] "
            f"{snapshot.completed_attempts}/{snapshot.total_attempts} "
            f"({percent:.1f}%) | active={active} "
            f"| active_elapsed={_format_duration(snapshot.current_attempt_elapsed_seconds)} "
            f"| total_elapsed={_format_duration(snapshot.elapsed_seconds)} "
            f"| ema_attempt={_format_duration(snapshot.ema_attempt_seconds)} "
            f"| eta_approx={_format_duration(snapshot.eta_seconds)}{suffix}",
            flush=True,
        )

    def _eta_seconds(self, current_elapsed: float | None) -> float | None:
        """Estimate remaining wall time from the observed attempt EWMA.

        Args:
            current_elapsed: Runtime already consumed by the active attempt, or
                ``None`` between attempts.

        Returns:
            Approximate remaining seconds, zero after the final attempt, or
            ``None`` until at least one attempt runtime has been observed.
        """
        remaining_attempts = self.total_attempts - self.completed_attempts
        if remaining_attempts <= 0:
            return 0.0
        if self.ema_attempt_seconds is None:
            return None
        if current_elapsed is None:
            return remaining_attempts * self.ema_attempt_seconds
        active_remaining = max(0.0, self.ema_attempt_seconds - current_elapsed)
        return active_remaining + (
            max(0, remaining_attempts - 1) * self.ema_attempt_seconds
        )


def _format_duration(seconds: float | None) -> str:
    """Format a nonnegative duration for compact progress output.

    Args:
        seconds: Duration in seconds, or ``None`` when no estimate exists.

    Returns:
        ``HH:MM:SS`` for known values and ``pending`` for unavailable values.
    """
    if seconds is None:
        return "pending"
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}"


def build_parser() -> argparse.ArgumentParser:
    """Build the fixed-study command-line parser.

    Returns:
        Parser exposing study scope, common resource limits, and output location.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run paired stage-2 CARD outputs with estimated-duration feedback and "
            "rendered-audio duration feedback under strict context handling."
        )
    )
    parser.add_argument(
        "--package-root",
        default=str(SPORC_HUMAN_EVAL_ASSETS_DIR.resolve()),
        help="Tracked SPoRC human-evaluation asset package.",
    )
    parser.add_argument(
        "--output-root",
        default="",
        help="New study directory. An existing path is rejected.",
    )
    parser.add_argument(
        "--episode-indices",
        default="",
        help="Comma-separated one-based episode indices; empty selects all five.",
    )
    parser.add_argument(
        "--duration-presets",
        default="30s,1m,5m",
        help="Comma-separated targets selected from 30s, 1m, and 5m.",
    )
    parser.add_argument(
        "--conditions",
        default=",".join(DEFAULT_CONDITIONS),
        help="Comma-separated conditions selected from estimated and measured.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="Number of scheduled repetitions for every episode-target pair.",
    )
    parser.add_argument("--llm-model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--context-window-tokens",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW_TOKENS,
    )
    parser.add_argument("--orchestrator-max-iterations", type=int, default=60)
    parser.add_argument(
        "--skip-setup",
        action="store_true",
        help="Skip Docker dependency/model preparation when it already completed.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare immutable source inputs and stop before calibration and runs.",
    )
    parser.add_argument(
        "--reuse-study-root",
        default="",
        help=(
            "Existing completed-preparation study root whose transcripts, affect "
            "manifests, and calibrations are reused read-only."
        ),
    )
    parser.add_argument(
        "--connection-max-attempts",
        type=int,
        default=DEFAULT_CONNECTION_MAX_ATTEMPTS,
        help="Maximum endpoint probes in each connection-readiness gate.",
    )
    parser.add_argument(
        "--connection-retry-delay-seconds",
        type=float,
        default=DEFAULT_CONNECTION_RETRY_DELAY_SECONDS,
        help="Seconds to wait after a failed endpoint probe.",
    )
    parser.add_argument(
        "--connection-stability-successes",
        type=int,
        default=DEFAULT_CONNECTION_STABILITY_SUCCESSES,
        help="Consecutive successful probes required before expensive local work.",
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        help="Seconds between study progress and approximate ETA lines.",
    )
    parser.add_argument(
        "--progress-ema-alpha",
        type=float,
        default=DEFAULT_PROGRESS_EMA_ALPHA,
        help="Newest-attempt weight for the runtime EWMA in (0, 1].",
    )
    return parser


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp for study records.

    Returns:
        Current UTC time with an explicit timezone suffix.
    """
    return datetime.now(timezone.utc).isoformat()


def _resolve_output_root(raw_value: str) -> Path:
    """Resolve a new output directory without overwriting an earlier study.

    Args:
        raw_value: Optional absolute path or repository-relative path.

    Returns:
        Absolute output path that does not currently exist.

    Raises:
        FileExistsError: The resolved directory already exists.
    """
    if raw_value.strip():
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = REPO_ROOT / candidate
        resolved = candidate.resolve()
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        resolved = (DEFAULT_OUTPUT_PARENT / f"{stamp}_{uuid4().hex[:8]}").resolve()
    if resolved.exists():
        raise FileExistsError(f"Study output path already exists: {resolved}")
    return resolved


def _parse_conditions(raw_value: str) -> tuple[str, ...]:
    """Validate the explicitly requested duration-feedback conditions.

    Args:
        raw_value: Comma-separated condition names.

    Returns:
        Ordered unique condition names.

    Raises:
        ValueError: No condition is supplied or an unsupported name appears.
    """
    values = tuple(dict.fromkeys(item.strip().lower() for item in raw_value.split(",") if item.strip()))
    if not values:
        raise ValueError("At least one study condition is required.")
    unsupported = sorted(set(values) - set(DEFAULT_CONDITIONS))
    if unsupported:
        raise ValueError("Unsupported study condition(s): " + ", ".join(unsupported))
    return values


def _parse_episode_indices(raw_value: str) -> set[int] | None:
    """Parse optional one-based episode indices for local asset staging.

    Args:
        raw_value: Comma-separated episode numbers or an empty string.

    Returns:
        Selected indices, or ``None`` when every packaged episode is requested.

    Raises:
        ValueError: Any supplied index is less than one.
    """
    values = {int(item.strip()) for item in raw_value.split(",") if item.strip()}
    if any(value < 1 for value in values):
        raise ValueError("episode indices must be one-based positive integers.")
    return values or None


def _to_container_path(host_path: Path) -> str:
    """Map a repository path to the Docker workspace mount.

    Args:
        host_path: Existing or planned path located below the repository root.

    Returns:
        POSIX path visible inside the Compose service.

    Raises:
        ValueError: The path is outside the repository checkout.
    """
    relative = host_path.resolve().relative_to(REPO_ROOT.resolve())
    return str(CONTAINER_REPO_ROOT.joinpath(*relative.parts))


def _write_json(path: Path, payload: Any) -> None:
    """Write one complete UTF-8 JSON record atomically.

    Args:
        path: Destination JSON path.
        payload: JSON-serializable data to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _run_logged(command: list[str], *, log_path: Path) -> int:
    """Execute a host command and retain its combined output.

    Args:
        command: Exact process arguments without shell interpolation.
        log_path: File that receives stdout and stderr.

    Returns:
        Child process exit code.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return int(completed.returncode)


def _stage_package_assets(
    *,
    source_root: Path,
    destination_root: Path,
    selected_indices: set[int] | None,
) -> Path:
    """Copy selected local media into the Docker-mounted study directory.

    Args:
        source_root: Existing asset package that contains the local media files.
        destination_root: New package directory inside the study output root.
        selected_indices: Episode numbers to copy, or ``None`` for every episode.

    Returns:
        Destination package root containing the selected source assets.

    Raises:
        FileNotFoundError: A required transcript, audio slice, manifest, or WAV
            is absent from the supplied package.
        ValueError: The package summary has no episode list.
    """
    summary_path = (source_root / "package_summary.json").resolve()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"Package summary has no episode list: {summary_path}")
    destination_root.mkdir(parents=True)
    _write_json(destination_root / "package_summary.json", payload)
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("Each package episode must be a JSON object.")
        episode_index = int(episode["order"])
        if selected_indices is not None and episode_index not in selected_indices:
            continue
        relative_files = [
            str(episode["transcript_slice"]),
            str(episode["audio_slice"]),
        ]
        voice_folder = Path(str(episode["voice_samples_folder"]).replace("\\", "/"))
        relative_files.append(str(voice_folder / "manifest.json"))
        raw_voice_files = episode.get("voice_sample_files")
        if not isinstance(raw_voice_files, list) or not raw_voice_files:
            raise ValueError(f"Episode {episode_index} has no voice_sample_files list.")
        relative_files.extend(str(voice_folder / str(name)) for name in raw_voice_files)
        for raw_relative in relative_files:
            relative = Path(raw_relative.replace("\\", "/"))
            source = (source_root / relative).resolve()
            if not source.is_file():
                raise FileNotFoundError(f"Required study asset is missing: {source}")
            destination = (destination_root / relative).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    return destination_root.resolve()


def _prepare_inputs(args: argparse.Namespace, output_root: Path) -> Path:
    """Materialize tracked transcripts and speaker manifests once.

    Args:
        args: Parsed study settings.
        output_root: New root for all preparation and experiment results.

    Returns:
        Path to the preparation summary produced by the existing asset driver.

    Raises:
        RuntimeError: Preparation exits unsuccessfully.
    """
    prepared_root = (output_root / "prepared_inputs").resolve()
    staged_package_root = _stage_package_assets(
        source_root=Path(args.package_root).expanduser().resolve(),
        destination_root=(output_root / "source_package").resolve(),
        selected_indices=_parse_episode_indices(str(args.episode_indices)),
    )
    command = [
        "uv", "run", "--no-sync", "python",
        str((REPO_ROOT / "scripts/run_sporc_human_eval_stage2_sweep.py").resolve()),
        "--prepare-only",
        "--package-root", str(staged_package_root),
        "--output-root", str(prepared_root),
        "--notes-path", str((output_root / "logs/preparation_notes.md").resolve()),
        "--duration-presets", str(args.duration_presets),
        "--episode-indices", str(args.episode_indices),
        "--llm-profile", "vllm_local",
        "--llm-model", str(args.llm_model),
    ]
    return_code = _run_logged(
        command,
        log_path=(output_root / "logs/preparation.log").resolve(),
    )
    if return_code != 0:
        raise RuntimeError(f"Input preparation failed with exit code {return_code}.")
    return (prepared_root / "run_summary.json").resolve()


def _load_prepared_inputs(
    summary_path: Path, artifact_root: Path
) -> tuple[PreparedInput, ...]:
    """Load prepared targets while sharing one source transcript per episode.

    Args:
        summary_path: Preparation summary generated by the tracked asset driver.
        artifact_root: Study root that owns the per-episode calibrations.

    Returns:
        Prepared target records with identical episode input paths across targets.

    Raises:
        ValueError: The preparation record is empty or malformed.
    """
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError(f"Preparation summary has no runs: {summary_path}")
    shared_transcripts: dict[int, Path] = {}
    labels: dict[int, str] = {}
    for raw in raw_runs:
        episode_index = int(raw["episode_index"])
        prepared_root = Path(str(raw["output_root"])).resolve()
        shared_transcripts.setdefault(episode_index, Path(str(raw["transcript_path"])).resolve())
        labels.setdefault(episode_index, prepared_root.parent.name)
    records = []
    for raw in raw_runs:
        episode_index = int(raw["episode_index"])
        episode_label = labels[episode_index]
        records.append(
            PreparedInput(
                episode_index=episode_index,
                episode_title=str(raw["episode_title"]),
                episode_label=episode_label,
                duration_preset=str(raw["duration_preset"]),
                target_seconds=int(raw["target_seconds"]),
                transcript_path=shared_transcripts[episode_index],
                calibration_path=(
                    artifact_root
                    / "calibrations"
                    / episode_label
                    / "voice_clone_calibration.json"
                ).resolve(),
            )
        )
    return tuple(records)


def _host_path_from_container(container_path: str) -> Path:
    """Map one retained Docker workspace path to its host repository path.

    Args:
        container_path: Absolute POSIX path recorded by a study container.

    Returns:
        Absolute host path beneath the current repository checkout.

    Raises:
        ValueError: The retained path is outside the repository bind mount.
    """
    path = PurePosixPath(container_path)
    relative = path.relative_to(CONTAINER_REPO_ROOT)
    return REPO_ROOT.joinpath(*relative.parts).resolve()


def _resolve_reuse_root(raw_value: str, output_root: Path) -> Path:
    """Resolve a source study inside the current Docker-mounted checkout.

    Args:
        raw_value: Absolute or repository-relative source study directory.
        output_root: New study directory that must remain separate.

    Returns:
        Existing source study root beneath the current repository.

    Raises:
        FileNotFoundError: The source study directory does not exist.
        ValueError: The source equals the new output root or sits outside the
            repository mounted into the study container.
    """
    source_root = Path(raw_value).expanduser()
    if not source_root.is_absolute():
        source_root = REPO_ROOT / source_root
    resolved = source_root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Reused study root does not exist: {resolved}")
    if resolved == output_root.resolve():
        raise ValueError("The reused study root must differ from the new output root.")
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            "--reuse-study-root must be inside the current card-framework checkout "
            f"mounted at {REPO_ROOT}."
        ) from error
    return resolved


def _require_reused_artifact(path: Path, source_root: Path, label: str) -> Path:
    """Validate one reused artifact without changing the source study.

    Args:
        path: Candidate host artifact path from a retained record.
        source_root: Existing study root selected for read-only reuse.
        label: Human-readable artifact name included in validation errors.

    Returns:
        Resolved artifact path beneath ``source_root``.

    Raises:
        FileNotFoundError: The artifact does not exist as a regular file.
        ValueError: The artifact resolves outside the selected source study.
    """
    resolved = path.resolve()
    resolved.relative_to(source_root.resolve())
    if not resolved.is_file():
        raise FileNotFoundError(f"Reused {label} is missing: {resolved}")
    return resolved


def _load_reused_study(
    args: argparse.Namespace,
    source_root: Path,
) -> tuple[Path, tuple[PreparedInput, ...], list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and validate immutable preparation from an earlier study root.

    The source study remains read-only. Selected transcripts must retain an
    affect manifest under that root, and every selected episode must retain a
    voice-clone calibration artifact. The requested model and exact selected
    episode-duration cells must agree with the earlier preparation record.

    Args:
        args: Current study arguments, including model and selected cells.
        source_root: Existing study directory used as the artifact source.

    Returns:
        Preparation-summary path, selected prepared inputs, and prior affect
        and calibration records restricted to the selected episodes.

    Raises:
        FileNotFoundError: A required summary or artifact is missing.
        ValueError: Source metadata, model identity, selection, or paths are
            inconsistent with the current study request.
    """
    root = source_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Reused study root does not exist: {root}")
    study_summary_path = _require_reused_artifact(
        root / "study_summary.json",
        root,
        "study summary",
    )
    preparation_summary_path = _require_reused_artifact(
        root / "prepared_inputs" / "run_summary.json",
        root,
        "preparation summary",
    )
    study_summary = json.loads(study_summary_path.read_text(encoding="utf-8"))
    if str(study_summary.get("model", "")) != str(args.llm_model):
        raise ValueError("The reused study model does not match --llm-model.")
    all_prepared = _load_prepared_inputs(preparation_summary_path, root)
    selected_indices = _parse_episode_indices(str(args.episode_indices))
    selected_presets = tuple(
        dict.fromkeys(
            value.strip()
            for value in str(args.duration_presets).split(",")
            if value.strip()
        )
    )
    if not selected_presets:
        raise ValueError("At least one duration preset is required.")
    selected = tuple(
        item
        for item in all_prepared
        if (selected_indices is None or item.episode_index in selected_indices)
        and item.duration_preset in selected_presets
    )
    available_indices = {item.episode_index for item in selected}
    expected_indices = (
        {item.episode_index for item in all_prepared}
        if selected_indices is None
        else selected_indices
    )
    expected_cells = {
        (episode_index, preset)
        for episode_index in expected_indices
        for preset in selected_presets
    }
    actual_cells = {(item.episode_index, item.duration_preset) for item in selected}
    if available_indices != expected_indices or actual_cells != expected_cells:
        raise ValueError(
            "The reused study does not contain every requested episode-duration cell."
        )
    for item in selected:
        transcript_path = _require_reused_artifact(
            item.transcript_path,
            root,
            "prepared transcript",
        )
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        metadata = transcript.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                f"Reused transcript has no metadata object: {transcript_path}"
            )
        raw_manifest = metadata.get("affect_analysis_manifest_path")
        if not isinstance(raw_manifest, str) or not raw_manifest.strip():
            raise ValueError(
                f"Reused transcript has no affect manifest: {transcript_path}"
            )
        _require_reused_artifact(
            _host_path_from_container(raw_manifest),
            root,
            "affect manifest",
        )
        _require_reused_artifact(item.calibration_path, root, "calibration")
    affect_records = study_summary.get("affect_preprocessing")
    calibration_records = study_summary.get("calibrations")
    if not isinstance(affect_records, list) or not isinstance(
        calibration_records, list
    ):
        raise ValueError("The reused study summary lacks preparation records.")
    selected_episode_indices = {item.episode_index for item in selected}
    selected_affect = [
        dict(record)
        for record in affect_records
        if int(record["episode_index"]) in selected_episode_indices
    ]
    selected_calibrations = [
        dict(record)
        for record in calibration_records
        if int(record["episode_index"]) in selected_episode_indices
    ]
    if {
        int(record["episode_index"]) for record in selected_affect
    } != selected_episode_indices:
        raise ValueError(
            "The reused study lacks an affect record for a selected episode."
        )
    if {
        int(record["episode_index"]) for record in selected_calibrations
    } != selected_episode_indices:
        raise ValueError(
            "The reused study lacks a calibration record for a selected episode."
        )
    for record in [*selected_affect, *selected_calibrations]:
        if int(record.get("return_code", -1)) != 0:
            raise ValueError(
                "The reused study contains an unsuccessful preparation record."
            )
        record["reused_from_study_root"] = str(root)
    return preparation_summary_path, selected, selected_affect, selected_calibrations


def _docker_prefix(*, offline_models: bool) -> list[str]:
    """Build the Compose prefix for one setup or study-stage container.

    Args:
        offline_models: Whether Hugging Face loaders must use only the
            persistent model cache. Study stages enable this after setup so
            registry availability cannot alter their elapsed time.

    Returns:
        Docker Compose arguments containing no credential value.
    """
    prefix = [
        "docker", "compose", "--profile", "gpu", "run", "--rm",
        "-e", "CARD_FRAMEWORK_VLLM_API_KEY",
    ]
    if offline_models:
        prefix.extend(
            [
                "-e",
                "HF_HUB_OFFLINE=1",
                "-e",
                "TRANSFORMERS_OFFLINE=1",
            ]
        )
    return prefix


def _run_setup(output_root: Path) -> None:
    """Prepare the existing GPU image and model files without updating Git.

    Args:
        output_root: Study root that owns the setup log.

    Raises:
        RuntimeError: Docker setup exits unsuccessfully.
    """
    command = [
        *_docker_prefix(offline_models=False), "setup-and-run-gpu", "--setup-only",
        "--skip-repo-update", "--override", "audio.device=cuda",
        "--override", "audio.voice_clone.device=cuda",
    ]
    return_code = _run_logged(command, log_path=output_root / "logs/setup.log")
    if return_code != 0:
        raise RuntimeError(f"Docker GPU setup failed with exit code {return_code}.")


def _build_vllm_preflight_command(model: str) -> list[str]:
    """Build one authenticated model-discovery command for the Docker network.

    The command checks the OpenAI-compatible ``/models`` endpoint from the
    same Compose service used by each timed attempt. The API key is forwarded
    by environment-variable name and never appears in the retained arguments.

    Args:
        model: Exact model identifier required by every study condition.

    Returns:
        Docker Compose argument vector for one request with a 15-second socket
        timeout and no retry behavior.

    Raises:
        ValueError: The requested model identifier is empty.
    """
    requested_model = str(model).strip()
    if not requested_model:
        raise ValueError("The vLLM preflight model must be non-empty.")
    script = f"""
import http.client
import json
import os
import sys
import urllib.error
import urllib.request


def main():
    base = os.environ.get("CARD_FRAMEWORK_VLLM_BASE_URL", "").rstrip("/")
    key = os.environ.get("CARD_FRAMEWORK_VLLM_API_KEY", "")
    requested = sys.argv[1]
    if not base or not key:
        print(json.dumps({{"status": "configuration_error"}}))
        return 20
    request = urllib.request.Request(
        base + "/models",
        headers={{"Authorization": "Bearer " + key}},
    )
    try:
        response = urllib.request.urlopen(request, timeout=15)
    except urllib.error.HTTPError as error:
        temporary = error.code == 429 or error.code >= 500
        print(json.dumps({{"status": "http_error", "http_status": error.code}}))
        return {TEMPORARY_CONNECTION_EXIT_CODE} if temporary else 20
    except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
        print(json.dumps({{"status": "connection_error", "error_type": type(error).__name__}}))
        return {TEMPORARY_CONNECTION_EXIT_CODE}
    try:
        payload = json.load(response)
        models = {{str(item["id"]) for item in payload["data"]}}
    except (http.client.HTTPException, OSError) as error:
        print(json.dumps({{"status": "connection_error", "error_type": type(error).__name__}}))
        return {TEMPORARY_CONNECTION_EXIT_CODE}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({{"status": "invalid_response", "error_type": type(error).__name__}}))
        return 21
    if requested not in models:
        print(json.dumps({{"status": "model_absent", "requested_model": requested}}))
        return 22
    print(json.dumps({{
        "status": "ready",
        "requested_model": requested,
        "advertised_model_count": len(models),
    }}))
    return 0


raise SystemExit(main())
""".strip()
    return [
        *_docker_prefix(offline_models=True),
        "--entrypoint",
        "python",
        "setup-and-run-gpu",
        "-c",
        script,
        requested_model,
    ]


def _preflight_vllm(model: str, log_path: Path) -> dict[str, Any]:
    """Check endpoint reachability and exact model availability once.

    Args:
        model: Exact model identifier required by every timed attempt.
        log_path: Unique file that retains output from this exact probe.

    Returns:
        Timestamped preflight record with the exit code and retained log path.
        The caller decides whether a nonzero exit code aborts the study.
    """
    resolved_log_path = log_path.resolve()
    started = _utc_now()
    return_code = _run_logged(
        _build_vllm_preflight_command(model),
        log_path=resolved_log_path,
    )
    return {
        "started_at": started,
        "finished_at": _utc_now(),
        "return_code": return_code,
        "log_path": str(resolved_log_path),
    }


def _wait_for_vllm_readiness(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    model: str,
    output_root: Path,
    phase: str,
    max_attempts: int,
    retry_delay_seconds: float,
    required_consecutive_successes: int,
) -> dict[str, Any]:
    """Retry endpoint probes until the requested readiness streak is reached.

    Each probe receives a separate log. A failed probe resets the success
    streak, and the operator sees both the result and the next wait. This gate
    runs outside measured attempts so network recovery time cannot enter an
    attempt's elapsed time or consume a scheduled experimental cell.

    Args:
        summary: Mutable study summary that receives live probe history.
        summary_path: JSON summary rewritten after every state transition.
        model: Exact model identifier that the endpoint must advertise.
        output_root: New study root that owns readiness logs.
        phase: Filesystem-safe name for the point in the run being guarded.
        max_attempts: Positive maximum number of endpoint probes.
        retry_delay_seconds: Nonnegative delay after an unsuccessful probe.
        required_consecutive_successes: Positive success streak needed before
            the gate reports readiness.

    Returns:
        Durable readiness record containing phase status and every probe.

    Raises:
        ValueError: A numeric setting is invalid or ``phase`` is unsafe.
    """
    if max_attempts <= 0:
        raise ValueError("connection_max_attempts must be greater than zero.")
    if retry_delay_seconds < 0:
        raise ValueError("connection_retry_delay_seconds cannot be negative.")
    if required_consecutive_successes <= 0:
        raise ValueError("required_consecutive_successes must be greater than zero.")
    if required_consecutive_successes > max_attempts:
        raise ValueError("required_consecutive_successes cannot exceed max_attempts.")
    if not phase or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in phase
    ):
        raise ValueError(f"Unsafe connection-readiness phase: {phase!r}")
    record: dict[str, Any] = {
        "phase": phase,
        "started_at": _utc_now(),
        "required_consecutive_successes": required_consecutive_successes,
        "max_attempts": max_attempts,
        "retry_delay_seconds": retry_delay_seconds,
        "checks": [],
        "status": "checking",
    }
    summary["connection_readiness"].append(record)
    _write_json(summary_path, summary)
    consecutive_successes = 0
    for probe_number in range(1, max_attempts + 1):
        print(
            f"[connection] {phase} probe {probe_number}/{max_attempts} "
            f"(stable {consecutive_successes}/{required_consecutive_successes})",
            flush=True,
        )
        log_path = (
            output_root
            / "logs"
            / "connection"
            / phase
            / f"probe_{probe_number:02d}.log"
        )
        check = _preflight_vllm(model, log_path)
        check["probe_number"] = probe_number
        record["checks"].append(check)
        _write_json(summary_path, summary)
        if check["return_code"] == 0:
            consecutive_successes += 1
            print(
                f"[connection] {phase} ready check passed "
                f"({consecutive_successes}/{required_consecutive_successes})",
                flush=True,
            )
            if consecutive_successes >= required_consecutive_successes:
                record["status"] = "ready"
                record["finished_at"] = _utc_now()
                _write_json(summary_path, summary)
                return record
            continue
        consecutive_successes = 0
        if check["return_code"] != TEMPORARY_CONNECTION_EXIT_CODE:
            record["status"] = "unavailable_permanent"
            record["finished_at"] = _utc_now()
            print(
                f"[connection] {phase} stopped after a permanent preflight "
                f"error (log: {log_path})",
                flush=True,
            )
            _write_json(summary_path, summary)
            return record
        if probe_number < max_attempts:
            print(
                f"[connection] {phase} failed; retrying in "
                f"{retry_delay_seconds:g}s (log: {log_path})",
                flush=True,
            )
            time.sleep(retry_delay_seconds)
    record["status"] = "unavailable"
    record["finished_at"] = _utc_now()
    _write_json(summary_path, summary)
    print(
        f"[connection] {phase} unavailable after {max_attempts} probes; "
        f"last log: {record['checks'][-1]['log_path']}",
        flush=True,
    )
    return record


def _write_calibration_config(item: PreparedInput, output_root: Path) -> Path:
    """Write an audio-only config for an unmeasured episode calibration.

    Args:
        item: Prepared episode whose shared speaker manifest is calibrated.
        output_root: Study root that stores generated calibration configs.

    Returns:
        Host path to the generated OmegaConf YAML file.
    """
    config = OmegaConf.load(REPO_ROOT / "conf/config.yaml")
    audio = OmegaConf.to_container(config.audio, resolve=True)
    if not isinstance(audio, dict):
        raise ValueError("The repository audio configuration did not resolve to a mapping.")
    audio["device"] = "cuda"
    voice_clone = audio["voice_clone"]
    if not isinstance(voice_clone, dict):
        raise ValueError("audio.voice_clone must resolve to a mapping.")
    voice_clone["device"] = "cuda"
    calibration = voice_clone["calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("audio.voice_clone.calibration must resolve to a mapping.")
    calibration["artifact_path"] = _to_container_path(item.calibration_path)
    path = (output_root / "calibrations" / item.episode_label / "config.yaml").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create({"audio": audio}), f=path)
    return path


def _calibrate_episode(item: PreparedInput, output_root: Path) -> dict[str, Any]:
    """Generate one baseline WPM artifact before timed condition runs.

    Args:
        item: Prepared episode input shared across all targets and conditions.
        output_root: Study root for calibration logs and status.

    Returns:
        Calibration record containing paths and the process exit code.

    Raises:
        RuntimeError: Calibration fails or does not create the expected artifact.
    """
    transcript = json.loads(item.transcript_path.read_text(encoding="utf-8"))
    manifest = str(transcript["metadata"]["speaker_samples_manifest_path"])
    config_path = _write_calibration_config(item, output_root)
    command = [
        *_docker_prefix(offline_models=True), "--entrypoint", "uv", "setup-and-run-gpu",
        "run", "python", "-m", "card_framework.cli.calibrate",
        "--project-root", str(CONTAINER_REPO_ROOT),
        "--config-file", _to_container_path(config_path),
        "--speaker-samples-manifest", manifest,
    ]
    log_path = output_root / "logs/calibration" / f"{item.episode_label}.log"
    started = _utc_now()
    return_code = _run_logged(command, log_path=log_path)
    record = {
        "episode_index": item.episode_index,
        "started_at": started,
        "finished_at": _utc_now(),
        "return_code": return_code,
        "artifact_path": str(item.calibration_path),
        "log_path": str(log_path.resolve()),
    }
    if return_code != 0 or not item.calibration_path.is_file():
        raise RuntimeError(
            f"Episode {item.episode_index} calibration failed with exit code {return_code}."
        )
    return record


def _write_affect_config(item: PreparedInput, output_root: Path) -> Path:
    """Write an audio config that loads both affect models from Docker caches.

    The study setup stores pinned ModelScope and Hugging Face snapshots in the
    persistent Compose model volume. Passing those verified directories to the
    model loaders avoids registry lookups during study preparation while
    retaining the exact models selected by the repository configuration.

    Args:
        item: Prepared episode whose transcript receives affect annotations.
        output_root: Study root that stores generated preprocessing configs.

    Returns:
        Host path to the generated OmegaConf YAML file. The corresponding
        container path remains available through the repository bind mount.

    Raises:
        ValueError: The repository audio or categorical affect configuration
            does not resolve to a mutable mapping.
    """
    config = OmegaConf.load(REPO_ROOT / "conf/config.yaml")
    audio = OmegaConf.to_container(config.audio, resolve=True)
    if not isinstance(audio, dict):
        raise ValueError("The repository audio configuration did not resolve to a mapping.")
    audio["device"] = "cuda"
    affect_analysis = audio["affect_analysis"]
    if not isinstance(affect_analysis, dict):
        raise ValueError("audio.affect_analysis must resolve to a mapping.")
    categorical = affect_analysis["categorical"]
    if not isinstance(categorical, dict):
        raise ValueError("audio.affect_analysis.categorical must resolve to a mapping.")
    categorical["device"] = "cuda"
    categorical["model_name"] = str(CONTAINER_EMOTION2VEC_MODEL_PATH)
    dimensional = affect_analysis["dimensional"]
    if not isinstance(dimensional, dict):
        raise ValueError("audio.affect_analysis.dimensional must resolve to a mapping.")
    dimensional["device"] = "cuda"
    dimensional["model_name"] = str(CONTAINER_AUDEERING_MODEL_PATH)
    path = (output_root / "preprocessing" / item.episode_label / "config.yaml").resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=OmegaConf.create({"audio": audio}), f=path)
    return path


def _prepare_affect_episode(
    item: PreparedInput,
    output_root: Path,
) -> dict[str, Any]:
    """Freeze audio-grounded affect features before paired timed attempts.

    Args:
        item: Shared episode transcript selected for every target and condition.
        output_root: Study root for preprocessing outputs and logs.

    Returns:
        Preprocessing record containing the transcript, log, and exit status.

    Raises:
        RuntimeError: Affect preparation fails or leaves no manifest metadata.
    """
    work_dir = (
        output_root / "preprocessing" / item.episode_label / "audio_stage"
    ).resolve()
    config_path = _write_affect_config(item, output_root)
    command = [
        *_docker_prefix(offline_models=True),
        "--entrypoint",
        "uv",
        "setup-and-run-gpu",
        "run",
        "python",
        "-m",
        "card_framework.cli.prepare_affect",
        "--transcript-path",
        _to_container_path(item.transcript_path),
        "--work-dir",
        _to_container_path(work_dir),
        "--project-root",
        str(CONTAINER_REPO_ROOT),
        "--config-file",
        _to_container_path(config_path),
    ]
    log_path = output_root / "logs/affect" / f"{item.episode_label}.log"
    started = _utc_now()
    return_code = _run_logged(command, log_path=log_path)
    transcript = json.loads(item.transcript_path.read_text(encoding="utf-8"))
    manifest = str(
        transcript.get("metadata", {}).get("affect_analysis_manifest_path", "")
    ).strip()
    record = {
        "episode_index": item.episode_index,
        "started_at": started,
        "finished_at": _utc_now(),
        "return_code": return_code,
        "transcript_path": str(item.transcript_path),
        "manifest_path": manifest,
        "log_path": str(log_path.resolve()),
    }
    if return_code != 0 or not manifest:
        raise RuntimeError(
            f"Episode {item.episode_index} affect preparation failed with "
            f"exit code {return_code}."
        )
    return record


def _condition_order(
    conditions: tuple[str, ...], repetition: int
) -> tuple[str, ...]:
    """Counterbalance paired conditions across successive repetitions.

    Args:
        conditions: Validated condition names selected for the study.
        repetition: One-based repetition index.

    Returns:
        Execution order for the requested repetition. Two-condition studies
        alternate their first condition; single-condition studies are unchanged.
    """
    if len(conditions) == 2 and repetition % 2 == 0:
        return tuple(reversed(conditions))
    return conditions


def _build_run_command(
    *,
    item: PreparedInput,
    condition: str,
    attempt_root: Path,
    model: str,
    context_window_tokens: int,
    max_iterations: int,
) -> list[str]:
    """Build one strict Docker command for a scheduled study attempt.

    Args:
        item: Shared input and target definition for the attempt.
        condition: ``estimated`` or ``measured`` duration feedback.
        attempt_root: Isolated output directory for this single attempt.
        model: Exact authenticated vLLM model identifier.
        context_window_tokens: Server-advertised input context capacity.
        max_iterations: Common summarizer-critic loop budget.

    Returns:
        Docker Compose argument vector. The API key value is absent from it.

    Raises:
        ValueError: The condition or numeric study limits are invalid.
    """
    if condition not in DEFAULT_CONDITIONS:
        raise ValueError(f"Unsupported study condition: {condition}")
    if context_window_tokens <= 0:
        raise ValueError("context_window_tokens must be greater than zero.")
    if max_iterations <= 0:
        raise ValueError("orchestrator_max_iterations must be greater than zero.")
    live_drafting = "true" if condition == "measured" else "false"
    loop_memory_path = (attempt_root / "summarizer_loop_memory.json").resolve()
    loop_diagnostics_path = (attempt_root / "loop_diagnostics.json").resolve()
    overrides = [
        "pipeline.start_stage=stage-2",
        f"transcript_path={_to_container_path(item.transcript_path)}",
        "selection.embedding=disabled",
        "selection.llm=vllm_local",
        "selection.stage_llm.summarizer=vllm_local",
        "selection.stage_llm.critic=vllm_local",
        "selection.stage_llm.interjector=vllm_local",
        f"+profiles.llm.vllm_local.model={model}",
        f"profiles.llm.vllm_local.context_window_tokens={context_window_tokens}",
        "+profiles.llm.vllm_local.default_max_tokens=null",
        f"orchestrator.target_seconds={item.target_seconds}",
        f"orchestrator.max_iterations={max_iterations}",
        "orchestrator.post_render_duration_verification.enabled=false",
        "orchestrator.prompt_compaction.enabled=false",
        "orchestrator.prompt_compaction.retry_on_overflow=false",
        "orchestrator.prompt_compaction.max_overflow_retries=0",
        f"orchestrator.loop_memory.artifact_path={_to_container_path(loop_memory_path)}",
        f"orchestrator.loop_diagnostics.artifact_path={_to_container_path(loop_diagnostics_path)}",
        "orchestrator.timeouts.summarizer=900",
        "orchestrator.timeouts.critic=900",
        "orchestrator.timeouts.retrieval=300",
        "orchestrator.timeouts.index=300",
        "agents.summarizer.max_tool_turns=50",
        "agents.critic.max_tool_turns=50",
        "agents.critic.require_structured_verdict=true",
        "agents.critic.reasoning_loop_guard.enabled=false",
        "agents.critic.reasoning_loop_guard.stream_abort_enabled=false",
        "agents.critic.reasoning_loop_guard.max_stream_abort_retries=0",
        "audio.device=cuda",
        "audio.retry.attempts=1",
        f"audio.work_dir={_to_container_path((attempt_root / 'audio_stage').resolve())}",
        f"audio.eta.store_path={_to_container_path((attempt_root / 'eta_history.sqlite3').resolve())}",
        "audio.affect_analysis.enabled=true",
        "audio.affect_analysis.reuse_existing_manifest=true",
        "audio.voice_clone.device=cuda",
        "audio.voice_clone.enabled=true",
        f"audio.voice_clone.live_drafting.enabled={live_drafting}",
        f"audio.voice_clone.calibration.artifact_path={_to_container_path(item.calibration_path)}",
        "audio.interjector.enabled=true",
        "audio.interjector.analysis_max_tokens=null",
        "audio.interjector.repair.prompt_max_tokens=null",
        "logging.level=INFO",
        f"logging.log_file={_to_container_path((attempt_root / 'agent_interactions.log').resolve())}",
        "logging.print_to_terminal=false",
        "logging.summarizer_critic_print_to_terminal=false",
        f"pipeline.final_summary_path={_to_container_path((attempt_root / 'summary.xml').resolve())}",
        "session.enabled=false",
    ]
    command = [
        *_docker_prefix(offline_models=True),
        "--entrypoint",
        "uv",
        "setup-and-run-gpu",
        "run",
        "python",
        "-m",
        "card_framework.cli.main",
        "--config-path",
        str(CONTAINER_REPO_ROOT / "conf"),
        "--config-name",
        "config",
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
    ]
    command.extend(overrides)
    return command


def _run_monitored(
    *,
    command: list[str],
    log_root: Path,
    attempt_id: str,
    progress: StudyProgressTracker,
) -> tuple[int, dict[str, str]]:
    """Run one Docker attempt with separate logs and resource sampling.

    Args:
        command: Exact Docker command built by :func:`_build_run_command`.
        log_root: Study log directory.
        attempt_id: Unique filesystem-safe attempt identifier.
        progress: Study-level tracker that prints periodic heartbeats while the
            monitored wrapper remains active.

    Returns:
        Exit code and paths to the retained execution records.
    """
    attempt_log_root = (log_root / "runs" / attempt_id).resolve()
    args_path = attempt_log_root / "docker.args.json"
    stdout_path = attempt_log_root / "stdout.log"
    stderr_path = attempt_log_root / "stderr.log"
    monitor_path = attempt_log_root / "resources.jsonl"
    status_path = attempt_log_root / "status.json"
    _write_json(args_path, command[1:])
    wrapper = [
        "uv", "run", "--no-sync", "python",
        str((REPO_ROOT / "scripts/run_docker_monitored.py").resolve()),
        "--repo-root", str(REPO_ROOT),
        "--args-json", str(args_path),
        "--stdout-path", str(stdout_path),
        "--stderr-path", str(stderr_path),
        "--monitor-path", str(monitor_path),
        "--status-path", str(status_path),
        "--sample-interval-seconds", "5",
    ]
    process = subprocess.Popen(wrapper, cwd=str(REPO_ROOT))
    while True:
        try:
            return_code = process.wait(timeout=progress.interval_seconds)
            break
        except subprocess.TimeoutExpired:
            progress.heartbeat()
    paths = {
        "docker_args_path": str(args_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "resource_monitor_path": str(monitor_path),
        "status_path": str(status_path),
    }
    return int(return_code), paths


def _measure_attempt(attempt_root: Path, target_seconds: int) -> dict[str, Any]:
    """Measure the final waveform without feeding its duration back to CARD.

    Args:
        attempt_root: Isolated completed attempt directory.
        target_seconds: Requested output duration.

    Returns:
        Objective duration metrics for the final interjected waveform.

    Raises:
        FileNotFoundError: The scheduled output did not produce the final WAV.
        RuntimeError: Audio duration probing fails.
    """
    audio_path = (attempt_root / FINAL_AUDIO_RELATIVE_PATH).resolve()
    if not audio_path.is_file():
        raise FileNotFoundError(f"Final audio was not produced: {audio_path}")
    duration_ms = probe_audio_duration_ms(audio_path)
    if duration_ms is None:
        raise RuntimeError(f"Unable to measure final audio duration: {audio_path}")
    duration_seconds = duration_ms / 1000.0
    signed_error_seconds = duration_seconds - target_seconds
    relative_error = signed_error_seconds / target_seconds
    return {
        "final_audio_path": str(audio_path),
        "duration_seconds": duration_seconds,
        "signed_error_seconds": signed_error_seconds,
        "absolute_relative_error": abs(relative_error),
        "within_five_percent": abs(relative_error) <= 0.05,
    }


def _read_json_object(path: Path, *, record_name: str) -> dict[str, Any]:
    """Load one required JSON object used for study measurements.

    Args:
        path: Existing JSON file to read.
        record_name: Human-readable record label included in validation errors.

    Returns:
        Parsed JSON mapping.

    Raises:
        FileNotFoundError: The required record does not exist.
        ValueError: The JSON root is not an object.
        json.JSONDecodeError: The file contains malformed JSON.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Required {record_name} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{record_name} must be a JSON object: {path}")
    return payload


def _collect_loop_metrics(attempt_root: Path) -> dict[str, Any]:
    """Read exact summarizer and critic iteration results for one attempt.

    Args:
        attempt_root: Isolated directory written by the timed application run.

    Returns:
        Loop convergence, pass count, revision count, and critic status sequence.
        A revision is every summarizer pass after the initial draft.

    Raises:
        FileNotFoundError: The application did not write loop diagnostics.
        ValueError: Required iteration fields are absent or inconsistent.
    """
    payload = _read_json_object(
        attempt_root / "loop_diagnostics.json",
        record_name="loop diagnostics",
    )
    iterations_run = payload.get("iterations_run")
    details = payload.get("iteration_details")
    if not isinstance(iterations_run, int) or iterations_run < 1:
        raise ValueError("Loop diagnostics iterations_run must be a positive integer.")
    if not isinstance(details, list) or len(details) != iterations_run:
        raise ValueError(
            "Loop diagnostics iteration_details must match iterations_run."
        )
    critic_statuses: list[str] = []
    for detail in details:
        if not isinstance(detail, dict):
            raise ValueError("Every loop diagnostic detail must be a JSON object.")
        critic_statuses.append(str(detail["critic_status"]))
    return {
        "converged": bool(payload.get("converged", False)),
        "final_status": str(payload["final_status"]),
        "summarizer_pass_count": iterations_run,
        "revision_count": iterations_run - 1,
        "critic_pass_count": sum(status != "not_run" for status in critic_statuses),
        "critic_statuses": critic_statuses,
    }


def _collect_llm_metrics(attempt_root: Path) -> dict[str, Any]:
    """Aggregate complete call traces without guessing omitted token usage.

    Args:
        attempt_root: Isolated directory containing the ``agent_calls`` archive.

    Returns:
        Exact call counts by inferred agent, failure count, and token totals for
        records where the server supplied usage metadata. ``token_usage`` is
        ``None`` when no call reported usage.

    Raises:
        FileNotFoundError: No complete call records were written.
        ValueError: A call record has an invalid agent, status, or usage shape.
    """
    trace_paths = sorted((attempt_root / "agent_calls").glob("*.json"))
    if not trace_paths:
        raise FileNotFoundError(
            f"No complete LLM call records found under {attempt_root / 'agent_calls'}"
        )
    calls_by_agent: dict[str, int] = {}
    failed_call_count = 0
    records_with_usage = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    for path in trace_paths:
        payload = _read_json_object(path, record_name="LLM call trace")
        agent = str(payload.get("agent", "")).strip()
        status = str(payload.get("status", "")).strip()
        if not agent or status not in {"completed", "failed"}:
            raise ValueError(f"Invalid agent or status in LLM call trace: {path}")
        calls_by_agent[agent] = calls_by_agent.get(agent, 0) + 1
        failed_call_count += int(status == "failed")
        provider_response = payload.get("provider_response")
        if not isinstance(provider_response, dict):
            continue
        usage = provider_response.get("usage")
        if usage is None:
            continue
        if not isinstance(usage, dict):
            raise ValueError(f"Invalid usage metadata in LLM call trace: {path}")
        token_values = (
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            usage.get("total_tokens"),
        )
        if any(not isinstance(value, int) or value < 0 for value in token_values):
            raise ValueError(f"Invalid token counts in LLM call trace: {path}")
        records_with_usage += 1
        prompt_tokens += token_values[0]
        completion_tokens += token_values[1]
        total_tokens += token_values[2]
    token_usage = None
    if records_with_usage:
        token_usage = {
            "records_with_usage": records_with_usage,
            "records_without_usage": len(trace_paths) - records_with_usage,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    return {
        "call_count": len(trace_paths),
        "failed_call_count": failed_call_count,
        "calls_by_agent": calls_by_agent,
        "token_usage": token_usage,
    }


def _collect_synthesis_metrics(
    attempt_root: Path,
    *,
    condition: str,
) -> dict[str, int]:
    """Count voice-clone and interjection synthesis calls from retained state.

    Args:
        attempt_root: Isolated directory containing audio-stage records.
        condition: Scheduled duration-feedback condition for the attempt.

    Returns:
        Exact voice-clone, interjection, and combined synthesis call counts.

    Raises:
        FileNotFoundError: A required audio manifest or measured-state file is
            absent.
        ValueError: Manifest counts, artifacts, or measured state are malformed.
    """
    voice_manifest = _read_json_object(
        attempt_root / "audio_stage/voice_clone/manifest.json",
        record_name="voice-clone manifest",
    )
    voice_artifacts = voice_manifest.get("artifacts")
    if not isinstance(voice_artifacts, list):
        raise ValueError("Voice-clone manifest artifacts must be a list.")
    voice_artifact_count = voice_manifest.get("artifact_count")
    if voice_artifact_count != len(voice_artifacts):
        raise ValueError("Voice-clone artifact_count does not match artifacts.")
    voice_synthesis_count = len(voice_artifacts)
    if condition == "measured":
        state_paths = sorted(
            (attempt_root / "audio_stage/voice_clone").glob("live_draft_*.state.json")
        )
        if len(state_paths) != 1:
            raise ValueError(
                "Measured condition must produce exactly one live-draft state file."
            )
        state = _read_json_object(
            state_paths[0],
            record_name="live-draft voice-clone state",
        )
        state_synthesis_count = state.get("synthesis_count")
        state_turn_audio = state.get("turn_audio")
        if not isinstance(state_synthesis_count, int) or state_synthesis_count < 0:
            raise ValueError("Live-draft synthesis_count must be non-negative.")
        if not isinstance(state_turn_audio, dict):
            raise ValueError("Live-draft turn_audio must be a JSON object.")
        state_output_paths = {
            str(record["output_audio_path"])
            for record in state_turn_audio.values()
            if isinstance(record, dict)
        }
        if len(state_output_paths) != len(state_turn_audio):
            raise ValueError("Live-draft turn records must include output_audio_path.")
        final_output_paths = {
            str(record["output_audio_path"])
            for record in voice_artifacts
            if isinstance(record, dict)
        }
        if len(final_output_paths) != len(voice_artifacts):
            raise ValueError("Voice-clone artifacts must include output_audio_path.")
        final_batch_count = 0
        if final_output_paths != state_output_paths:
            final_batch_count = len(voice_artifacts)
        voice_synthesis_count = state_synthesis_count + final_batch_count
    elif condition != "estimated":
        raise ValueError(f"Unsupported study condition: {condition}")

    interjector_manifest = _read_json_object(
        attempt_root / "audio_stage/interjector/interjector_manifest.json",
        record_name="interjector manifest",
    )
    accepted = interjector_manifest.get("artifacts")
    rejected = interjector_manifest.get("rejected_artifacts")
    if not isinstance(accepted, list) or not isinstance(rejected, list):
        raise ValueError("Interjector artifact lists are malformed.")
    if interjector_manifest.get("artifact_count") != len(accepted):
        raise ValueError("Interjector artifact_count does not match artifacts.")
    if interjector_manifest.get("rejected_artifact_count") != len(rejected):
        raise ValueError(
            "Interjector rejected_artifact_count does not match rejected_artifacts."
        )
    interjection_synthesis_count = 0
    for artifact in [*accepted, *rejected]:
        if not isinstance(artifact, dict) or not isinstance(
            artifact.get("repair_attempts"), list
        ):
            raise ValueError("Interjector artifacts must include repair_attempts.")
        interjection_synthesis_count += len(artifact["repair_attempts"])
    return {
        "voice_clone_synthesis_count": voice_synthesis_count,
        "interjection_synthesis_count": interjection_synthesis_count,
        "total_synthesis_count": voice_synthesis_count + interjection_synthesis_count,
    }


def _collect_attempt_metrics(
    attempt_root: Path,
    *,
    condition: str,
) -> dict[str, Any]:
    """Collect exact loop, model-call, and audio-synthesis metrics.

    Args:
        attempt_root: Isolated directory for a completed timed attempt.
        condition: Scheduled estimated or measured feedback condition.

    Returns:
        Nested metrics copied into the durable study summary.

    Raises:
        FileNotFoundError: Any required retained record is absent.
        ValueError: A retained record is internally inconsistent.
    """
    return {
        "loop": _collect_loop_metrics(attempt_root),
        "llm": _collect_llm_metrics(attempt_root),
        "synthesis": _collect_synthesis_metrics(
            attempt_root,
            condition=condition,
        ),
    }


def _git_revision() -> str:
    """Read the exact application commit used to launch the study.

    Returns:
        Current Git commit SHA from the isolated application worktree.

    Raises:
        subprocess.CalledProcessError: Git cannot resolve ``HEAD``.
    """
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _record_readiness_gate(
    *,
    summary: dict[str, Any],
    summary_path: Path,
    model: str,
    output_root: Path,
    phase: str,
    max_attempts: int,
    retry_delay_seconds: float,
    required_consecutive_successes: int,
) -> bool:
    """Run and persist one connection-readiness gate.

    Args:
        summary: Mutable study summary that owns readiness history.
        summary_path: JSON path rewritten after the gate finishes.
        model: Exact endpoint model required by the study.
        output_root: New study root that owns probe logs.
        phase: Filesystem-safe readiness phase identifier.
        max_attempts: Positive maximum number of probes.
        retry_delay_seconds: Nonnegative wait after failed probes.
        required_consecutive_successes: Success streak required by this gate.

    Returns:
        ``True`` when the endpoint is ready and ``False`` after the probe
        budget is exhausted.
    """
    gate = _wait_for_vllm_readiness(
        summary=summary,
        summary_path=summary_path,
        model=model,
        output_root=output_root,
        phase=phase,
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
        required_consecutive_successes=required_consecutive_successes,
    )
    return gate["status"] == "ready"


def _finish_connection_abort(
    summary: dict[str, Any],
    summary_path: Path,
    phase: str,
) -> int:
    """Finalize a study that exhausted one readiness gate before an attempt.

    Args:
        summary: Mutable study summary for the current new output root.
        summary_path: Durable JSON summary path.
        phase: Readiness phase that could not establish connectivity.

    Returns:
        Process exit code ``2`` for an unavailable required service.
    """
    summary["status"] = "aborted_connection_unavailable"
    summary["connection_abort_phase"] = phase
    summary["finished_at_utc"] = _utc_now()
    _write_json(summary_path, summary)
    return 2


def main(argv: list[str] | None = None) -> int:
    """Prepare inputs, calibrate episodes, and execute every scheduled attempt.

    Args:
        argv: Optional CLI arguments. ``None`` reads the process command line.

    Returns:
        Zero when every scheduled attempt succeeds, one when an attempt fails,
        and two when a connection-readiness gate exhausts its retry budget.

    Raises:
        EnvironmentError: The authenticated vLLM key is absent.
        ValueError: Study scope or common resource limits are invalid.
        RuntimeError: Input preparation, setup, or calibration fails.
    """
    load_dotenv(dotenv_path=REPO_ROOT / ".env", override=False)
    args = build_parser().parse_args(argv)
    conditions = _parse_conditions(str(args.conditions))
    if args.repetitions <= 0:
        raise ValueError("repetitions must be greater than zero.")
    if args.progress_interval_seconds <= 0:
        raise ValueError("progress_interval_seconds must be greater than zero.")
    if not 0 < args.progress_ema_alpha <= 1:
        raise ValueError(
            "progress_ema_alpha must be greater than zero and at most one."
        )
    if args.connection_max_attempts <= 0:
        raise ValueError("connection_max_attempts must be greater than zero.")
    if args.connection_retry_delay_seconds < 0:
        raise ValueError("connection_retry_delay_seconds cannot be negative.")
    if args.connection_stability_successes <= 0:
        raise ValueError("connection_stability_successes must be greater than zero.")
    if args.connection_stability_successes > args.connection_max_attempts:
        raise ValueError(
            "connection_stability_successes cannot exceed connection_max_attempts."
        )
    if args.prepare_only and str(args.reuse_study_root).strip():
        raise ValueError("--prepare-only cannot be combined with --reuse-study-root.")
    output_root = _resolve_output_root(str(args.output_root))
    output_root.mkdir(parents=True)
    summary_path = output_root / "study_summary.json"
    summary: dict[str, Any] = {
        "study_id": output_root.name,
        "created_at_utc": _utc_now(),
        "status": "initializing",
        "application_revision": _git_revision(),
        "model": str(args.llm_model),
        "context_window_tokens": int(args.context_window_tokens),
        "orchestrator_max_iterations": int(args.orchestrator_max_iterations),
        "conditions": list(conditions),
        "repetitions": int(args.repetitions),
        "progress_interval_seconds": float(args.progress_interval_seconds),
        "progress_ema_alpha": float(args.progress_ema_alpha),
        "prompt_compaction_enabled": False,
        "overflow_retry_enabled": False,
        "post_render_duration_verification_enabled": False,
        "llm_seed": None,
        "deterministic_pairing_claimed": False,
        "preparation_summary_path": None,
        "reused_study_root": None,
        "vllm_preflight": None,
        "connection_readiness": [],
        "connection_max_attempts": int(args.connection_max_attempts),
        "connection_retry_delay_seconds": float(args.connection_retry_delay_seconds),
        "connection_stability_successes": int(args.connection_stability_successes),
        "affect_preprocessing": [],
        "calibrations": [],
        "runs": [],
    }
    _write_json(summary_path, summary)
    if args.prepare_only:
        preparation_summary_path = _prepare_inputs(args, output_root)
        summary["preparation_summary_path"] = str(preparation_summary_path)
        summary["status"] = "prepared"
        summary["finished_at_utc"] = _utc_now()
        _write_json(summary_path, summary)
        return 0
    api_key = os.environ.get("CARD_FRAMEWORK_VLLM_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("CARD_FRAMEWORK_VLLM_API_KEY is required.")
    initial_phase = "before_preparation"
    if not _record_readiness_gate(
        summary=summary,
        summary_path=summary_path,
        model=str(args.llm_model),
        output_root=output_root,
        phase=initial_phase,
        max_attempts=int(args.connection_max_attempts),
        retry_delay_seconds=float(args.connection_retry_delay_seconds),
        required_consecutive_successes=int(args.connection_stability_successes),
    ):
        return _finish_connection_abort(summary, summary_path, initial_phase)
    summary["vllm_preflight"] = summary["connection_readiness"][0]
    if not args.skip_setup:
        _run_setup(output_root)
    reuse_value = str(args.reuse_study_root).strip()
    if reuse_value:
        reuse_root = _resolve_reuse_root(reuse_value, output_root)
        (
            preparation_summary_path,
            prepared,
            reused_affect,
            reused_calibrations,
        ) = _load_reused_study(args, reuse_root)
        summary["reused_study_root"] = str(reuse_root)
        summary["affect_preprocessing"] = reused_affect
        summary["calibrations"] = reused_calibrations
    else:
        preparation_summary_path = _prepare_inputs(args, output_root)
        prepared = _load_prepared_inputs(preparation_summary_path, output_root)
    summary["preparation_summary_path"] = str(preparation_summary_path)
    total_attempts = len(prepared) * len(conditions) * int(args.repetitions)
    progress = StudyProgressTracker(
        total_attempts=total_attempts,
        interval_seconds=float(args.progress_interval_seconds),
        ema_alpha=float(args.progress_ema_alpha),
    )
    _write_json(summary_path, summary)
    first_by_episode: dict[int, PreparedInput] = {}
    for item in prepared:
        first_by_episode.setdefault(item.episode_index, item)
    if not reuse_value:
        for item in first_by_episode.values():
            summary["affect_preprocessing"].append(
                _prepare_affect_episode(item, output_root)
            )
            _write_json(summary_path, summary)
            summary["calibrations"].append(_calibrate_episode(item, output_root))
            _write_json(summary_path, summary)
    stable_phase = "before_scheduled_attempts"
    if not _record_readiness_gate(
        summary=summary,
        summary_path=summary_path,
        model=str(args.llm_model),
        output_root=output_root,
        phase=stable_phase,
        max_attempts=int(args.connection_max_attempts),
        retry_delay_seconds=float(args.connection_retry_delay_seconds),
        required_consecutive_successes=int(args.connection_stability_successes),
    ):
        return _finish_connection_abort(summary, summary_path, stable_phase)
    failed = False
    log_root = (output_root / "logs").resolve()
    scheduled_attempt_number = 0
    for repetition in range(1, int(args.repetitions) + 1):
        for item in prepared:
            for condition in _condition_order(conditions, repetition):
                scheduled_attempt_number += 1
                attempt_id = (
                    f"rep_{repetition:02d}_{item.episode_label}_"
                    f"{item.duration_preset}_{condition}"
                )
                attempt_root = (
                    output_root
                    / "runs"
                    / f"repetition_{repetition:02d}"
                    / item.episode_label
                    / item.duration_preset
                    / condition
                ).resolve()
                connection_phase = f"before_attempt_{scheduled_attempt_number:03d}"
                if not _record_readiness_gate(
                    summary=summary,
                    summary_path=summary_path,
                    model=str(args.llm_model),
                    output_root=output_root,
                    phase=connection_phase,
                    max_attempts=int(args.connection_max_attempts),
                    retry_delay_seconds=float(args.connection_retry_delay_seconds),
                    required_consecutive_successes=1,
                ):
                    return _finish_connection_abort(
                        summary,
                        summary_path,
                        connection_phase,
                    )
                attempt_root.mkdir(parents=True)
                record: dict[str, Any] = {
                    "attempt_id": attempt_id,
                    "repetition": repetition,
                    "episode_index": item.episode_index,
                    "episode_title": item.episode_title,
                    "duration_preset": item.duration_preset,
                    "target_seconds": item.target_seconds,
                    "condition": condition,
                    "transcript_path": str(item.transcript_path),
                    "calibration_path": str(item.calibration_path),
                    "output_root": str(attempt_root),
                    "status": "running",
                    "started_at": _utc_now(),
                }
                summary["runs"].append(record)
                _write_json(summary_path, summary)
                progress.start_attempt(attempt_id)
                command = _build_run_command(
                    item=item,
                    condition=condition,
                    attempt_root=attempt_root,
                    model=str(args.llm_model),
                    context_window_tokens=int(args.context_window_tokens),
                    max_iterations=int(args.orchestrator_max_iterations),
                )
                attempt_started = time.perf_counter()
                return_code, log_paths = _run_monitored(
                    command=command,
                    log_root=log_root,
                    attempt_id=attempt_id,
                    progress=progress,
                )
                record.update(log_paths)
                record["return_code"] = return_code
                attempt_elapsed_seconds = time.perf_counter() - attempt_started
                record["elapsed_seconds"] = round(attempt_elapsed_seconds, 3)
                record["finished_at"] = _utc_now()
                if return_code == 0:
                    try:
                        record["measurements"] = _measure_attempt(
                            attempt_root, item.target_seconds
                        )
                        record["attempt_metrics"] = _collect_attempt_metrics(
                            attempt_root,
                            condition=condition,
                        )
                        record["status"] = "completed"
                    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
                        record["status"] = "failed_measurement"
                        record["error"] = str(error)
                        failed = True
                else:
                    record["status"] = "failed"
                    failed = True
                _write_json(summary_path, summary)
                progress.finish_attempt(
                    elapsed_seconds=attempt_elapsed_seconds,
                    outcome=str(record["status"]),
                )
    summary["finished_at_utc"] = _utc_now()
    summary["status"] = "completed_with_failures" if failed else "completed"
    _write_json(summary_path, summary)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
