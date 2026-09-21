"""Run the paired estimated-versus-measured duration-feedback study."""

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


def _load_prepared_inputs(summary_path: Path, output_root: Path) -> tuple[PreparedInput, ...]:
    """Load prepared targets while sharing one source transcript per episode.

    Args:
        summary_path: Preparation summary generated by the tracked asset driver.
        output_root: Study root used for per-episode calibration paths.

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
                    output_root / "calibrations" / episode_label / "voice_clone_calibration.json"
                ).resolve(),
            )
        )
    return tuple(records)


def _docker_prefix() -> list[str]:
    """Build the Compose prefix that forwards the API key by name.

    Returns:
        Docker Compose arguments containing no credential value.
    """
    return [
        "docker", "compose", "--profile", "gpu", "run", "--rm",
        "-e", "CARD_FRAMEWORK_VLLM_API_KEY",
    ]


def _run_setup(output_root: Path) -> None:
    """Prepare the existing GPU image and model files without updating Git.

    Args:
        output_root: Study root that owns the setup log.

    Raises:
        RuntimeError: Docker setup exits unsuccessfully.
    """
    command = [
        *_docker_prefix(), "setup-and-run-gpu", "--setup-only",
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
    script = (
        "import json, os, sys, urllib.request; "
        "base=os.environ['CARD_FRAMEWORK_VLLM_BASE_URL'].rstrip('/'); "
        "key=os.environ['CARD_FRAMEWORK_VLLM_API_KEY']; "
        "request=urllib.request.Request(base + '/models', "
        "headers={'Authorization': 'Bearer ' + key}); "
        "response=urllib.request.urlopen(request, timeout=15); "
        "payload=json.load(response); "
        "models={str(item['id']) for item in payload['data']}; "
        "requested=sys.argv[1]; "
        "assert requested in models, 'Requested model is absent from /models'; "
        "print(json.dumps({'status': 'ready', 'requested_model': requested, "
        "'advertised_model_count': len(models)}))"
    )
    return [
        *_docker_prefix(),
        "--entrypoint",
        "python",
        "setup-and-run-gpu",
        "-c",
        script,
        requested_model,
    ]


def _preflight_vllm(model: str, output_root: Path) -> dict[str, Any]:
    """Check endpoint reachability and exact model availability once.

    Args:
        model: Exact model identifier required by every timed attempt.
        output_root: Study root that owns the preflight log.

    Returns:
        Timestamped preflight record with the exit code and retained log path.
        The caller decides whether a nonzero exit code aborts the study.
    """
    log_path = (output_root / "logs/vllm_preflight.log").resolve()
    started = _utc_now()
    return_code = _run_logged(
        _build_vllm_preflight_command(model),
        log_path=log_path,
    )
    return {
        "started_at": started,
        "finished_at": _utc_now(),
        "return_code": return_code,
        "log_path": str(log_path),
    }


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
        *_docker_prefix(), "--entrypoint", "uv", "setup-and-run-gpu",
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
        *_docker_prefix(),
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
        *_docker_prefix(),
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
) -> tuple[int, dict[str, str]]:
    """Run one Docker attempt with separate logs and resource sampling.

    Args:
        command: Exact Docker command built by :func:`_build_run_command`.
        log_root: Study log directory.
        attempt_id: Unique filesystem-safe attempt identifier.

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
    completed = subprocess.run(wrapper, cwd=str(REPO_ROOT), check=False)
    paths = {
        "docker_args_path": str(args_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "resource_monitor_path": str(monitor_path),
        "status_path": str(status_path),
    }
    return int(completed.returncode), paths


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


def main(argv: list[str] | None = None) -> int:
    """Prepare inputs, calibrate episodes, and execute every scheduled attempt.

    Args:
        argv: Optional CLI arguments. ``None`` reads the process command line.

    Returns:
        Zero when every scheduled attempt succeeds; one when any attempt fails.

    Raises:
        EnvironmentError: The authenticated vLLM key is absent.
        ValueError: Study scope or common resource limits are invalid.
        RuntimeError: Input preparation, setup, or calibration fails.
    """
    args = build_parser().parse_args(argv)
    conditions = _parse_conditions(str(args.conditions))
    if args.repetitions <= 0:
        raise ValueError("repetitions must be greater than zero.")
    output_root = _resolve_output_root(str(args.output_root))
    output_root.mkdir(parents=True)
    summary_path = output_root / "study_summary.json"
    preparation_summary_path = _prepare_inputs(args, output_root)
    prepared = _load_prepared_inputs(preparation_summary_path, output_root)
    summary: dict[str, Any] = {
        "study_id": output_root.name,
        "created_at_utc": _utc_now(),
        "application_revision": _git_revision(),
        "model": str(args.llm_model),
        "context_window_tokens": int(args.context_window_tokens),
        "orchestrator_max_iterations": int(args.orchestrator_max_iterations),
        "conditions": list(conditions),
        "repetitions": int(args.repetitions),
        "prompt_compaction_enabled": False,
        "overflow_retry_enabled": False,
        "post_render_duration_verification_enabled": False,
        "llm_seed": None,
        "deterministic_pairing_claimed": False,
        "preparation_summary_path": str(preparation_summary_path),
        "vllm_preflight": None,
        "affect_preprocessing": [],
        "calibrations": [],
        "runs": [],
    }
    _write_json(summary_path, summary)
    if args.prepare_only:
        return 0
    api_key = os.environ.get("CARD_FRAMEWORK_VLLM_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("CARD_FRAMEWORK_VLLM_API_KEY is required.")
    if not args.skip_setup:
        _run_setup(output_root)
    summary["vllm_preflight"] = _preflight_vllm(str(args.llm_model), output_root)
    _write_json(summary_path, summary)
    if summary["vllm_preflight"]["return_code"] != 0:
        raise RuntimeError(
            "The vLLM endpoint/model preflight failed; see logs/vllm_preflight.log."
        )
    first_by_episode: dict[int, PreparedInput] = {}
    for item in prepared:
        first_by_episode.setdefault(item.episode_index, item)
    for item in first_by_episode.values():
        summary["affect_preprocessing"].append(
            _prepare_affect_episode(item, output_root)
        )
        _write_json(summary_path, summary)
        summary["calibrations"].append(_calibrate_episode(item, output_root))
        _write_json(summary_path, summary)
    failed = False
    log_root = (output_root / "logs").resolve()
    for repetition in range(1, int(args.repetitions) + 1):
        for item in prepared:
            for condition in _condition_order(conditions, repetition):
                attempt_id = (
                    f"rep_{repetition:02d}_{item.episode_label}_"
                    f"{item.duration_preset}_{condition}"
                )
                attempt_root = (
                    output_root / "runs" / f"repetition_{repetition:02d}"
                    / item.episode_label / item.duration_preset / condition
                ).resolve()
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
                )
                record.update(log_paths)
                record["return_code"] = return_code
                record["elapsed_seconds"] = round(
                    time.perf_counter() - attempt_started,
                    3,
                )
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
    summary["finished_at_utc"] = _utc_now()
    summary["status"] = "completed_with_failures" if failed else "completed"
    _write_json(summary_path, summary)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
