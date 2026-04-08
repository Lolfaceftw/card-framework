"""Pure helpers for the setup-and-run Textual frontend."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping

from omegaconf import OmegaConf

from card_framework.benchmark.summarizer_critic.qmsum.config import (
    DEFAULT_DATASET_MANIFEST_PATH as DEFAULT_QMSUM_DATASET_MANIFEST_PATH,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_DATASET_MANIFEST_PATH as DEFAULT_SPORC_DATASET_MANIFEST_PATH,
    DEFAULT_FULL_DATASET_MANIFEST_PATH,
    DEFAULT_SWEEP_ROOT,
)
from card_framework.cli.frontend_models import (
    BenchmarkKind,
    BenchmarkLaunchRequest,
    InteractiveMenuSelection,
)
from card_framework.shared.paths import (
    DEFAULT_BENCHMARK_MANIFEST_PATH,
    DEFAULT_DIARIZATION_MANIFEST_PATH,
    DEFAULT_QA_CONFIG_PATH,
    DEFAULT_PROVIDER_PROFILES_PATH,
    DEFAULT_CONFIG_PATH,
    discover_reusable_transcript_path,
)

PROFILE_DISPLAY_NAMES: dict[str, str] = {
    "anthropic_claude_opus": "Anthropic Claude Opus (cloud)",
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
DEFAULT_DURATION_SWEEP_PRESET_LABELS = ("30s", "1m", "5m")
DEFAULT_BENCHMARK_KIND = "sporc_benchmark"
PREFERRED_ROOT_AUDIO_FILENAMES = (
    "audio.wav",
    "audio.flac",
    "audio.mp3",
    "audio.m4a",
    "audio.aac",
    "audio.ogg",
    "audio.opus",
)
SUPPORTED_AUDIO_PATH_SUFFIXES = tuple(
    Path(filename).suffix.lower() for filename in PREFERRED_ROOT_AUDIO_FILENAMES
)
PIPELINE_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("stage1_single", "Full pipeline from one source audio file"),
    ("stage1_batch", "Sequential batch sweep from multiple audio files"),
    (
        "stage1_batch_duration_sweep",
        "Duration sweep from one or more audio files",
    ),
    ("stage2_transcript", "Jump to stage-2 from an existing transcript JSON"),
    ("stage3_summary", "Jump to stage-3 from an existing summary XML"),
    (
        "stage4_interjector",
        "Jump to stage-4 from an existing summary XML and voice-clone manifest",
    ),
    ("resume_session", "Resume a saved pipeline session by UUID"),
    ("setup_only", "Run setup checks/models only"),
)
STAGE_ROUTING_OPTIONS: tuple[tuple[str, str], ...] = (
    ("config_default", "Keep per-stage selections from config"),
    ("shared_for_all", "Use the shared LLM for all stages"),
    ("custom", "Choose summarizer, critic, and interjector profiles separately"),
)
BENCHMARK_KIND_OPTIONS: tuple[tuple[BenchmarkKind, str], ...] = (
    ("sporc_benchmark", "SPoRC summarizer-critic benchmark"),
    ("sporc_duration_sweep", "SPoRC duration sweep benchmark"),
    ("qmsum_benchmark", "QMSum summarizer-critic benchmark"),
    ("summarizer_benchmark", "Matrix summarizer benchmark"),
    ("qa_benchmark", "QA benchmark"),
    ("diarization_benchmark", "Diarization benchmark"),
    ("summary_matrix", "Summary matrix experiment"),
)
PIPELINE_MODE_LABELS: dict[str, str] = {
    value: label for value, label in PIPELINE_MODE_OPTIONS
}
BENCHMARK_KIND_LABELS: dict[str, str] = {
    value: label for value, label in BENCHMARK_KIND_OPTIONS
}


@dataclass(slots=True, frozen=True)
class PipelineProfileDefaults:
    """Current config-backed pipeline defaults displayed by the frontend."""

    llm: str
    embedding: str
    stage_llm: dict[str, str]
    voice_clone_enabled: bool
    interjector_enabled: bool


@dataclass(slots=True, frozen=True)
class FrontendDefaults:
    """Initial values and options for the Textual frontend."""

    workspace_root: Path
    config_path: Path
    config_text: str
    pipeline_defaults: PipelineProfileDefaults
    llm_profiles: tuple[tuple[str, str], ...]
    embedding_profiles: tuple[tuple[str, str], ...]
    discovered_audio_path: str
    discovered_transcript_path: str
    discovered_summary_path: str
    summary_benchmark_manifest: str
    qa_config_path: str
    qmsum_dataset_manifest: str
    sporc_dataset_manifest: str
    sporc_full_dataset_manifest: str
    diarization_manifest: str
    sporc_sweep_root: str


class FrontendValidationError(ValueError):
    """Raised when a frontend form cannot be projected into a launch request."""


def _describe_path_list(paths: tuple[str, ...]) -> str:
    """Render a short human-facing summary for one tuple of paths."""
    if not paths:
        return "(none)"
    if len(paths) == 1:
        return paths[0]
    return f"{len(paths)} items (first: {paths[0]})"


def describe_pipeline_selection(selection: InteractiveMenuSelection) -> tuple[str, ...]:
    """Return user-facing summary lines for one pipeline launch plan."""
    lines = [
        f"Workflow: {PIPELINE_MODE_LABELS.get(selection.mode, selection.mode)}",
        f"Output root: {selection.output_root or '(workspace default)'}",
    ]
    if selection.audio_paths:
        lines.append(f"Audio input: {_describe_path_list(selection.audio_paths)}")
    if selection.duration_presets:
        lines.append(
            "Duration presets: "
            + ", ".join(str(value) for value in selection.duration_presets)
        )
    if selection.transcript_path:
        lines.append(f"Transcript: {selection.transcript_path}")
    if selection.voiceclone_from_summary:
        lines.append(f"Summary XML: {selection.voiceclone_from_summary}")
    if selection.voice_clone_manifest_path:
        lines.append(f"Voice-clone manifest: {selection.voice_clone_manifest_path}")
    if selection.resume_session:
        lines.append(f"Resume session: {selection.resume_session}")
    if selection.setup_only:
        lines.append("Execution: setup checks only")
    else:
        lines.append("Execution: launch immediately after validation")
    lines.append(f"Runtime overrides: {len(selection.overrides)}")
    if selection.overrides:
        lines.extend(["", "Hydra overrides:", *selection.overrides])
    return tuple(lines)


def describe_benchmark_request(request: BenchmarkLaunchRequest) -> tuple[str, ...]:
    """Return user-facing summary lines for one benchmark launch plan."""
    return (
        f"Workflow: {BENCHMARK_KIND_LABELS.get(request.kind, request.kind)}",
        f"Action: {request.description}",
        f"Command parts: {len(request.command)}",
    )


def _display_name(profile_name: str) -> str:
    return PROFILE_DISPLAY_NAMES.get(
        profile_name, profile_name.replace("_", " ").strip()
    )


def _read_config_text(config_path: Path) -> str:
    return config_path.read_text(encoding="utf-8")


def _safe_bool(config: Any, path: tuple[str, ...], fallback: bool) -> bool:
    current = config
    for part in path:
        if not isinstance(current, (dict,)) and not hasattr(current, "get"):
            return fallback
        current = current.get(part)
        if current is None:
            return fallback
    return bool(current)


def _safe_string(config: Any, path: tuple[str, ...], fallback: str) -> str:
    current = config
    for part in path:
        if not isinstance(current, (dict,)) and not hasattr(current, "get"):
            return fallback
        current = current.get(part)
        if current is None:
            return fallback
    text = str(current).strip()
    return text or fallback


def _child_options(config: Any, group: str) -> tuple[tuple[str, str], ...]:
    profiles = config.get("profiles") if hasattr(config, "get") else None
    group_payload = None
    if profiles is not None and hasattr(profiles, "get"):
        group_payload = profiles.get(group)
    if group_payload is None:
        return ()
    options: list[tuple[str, str]] = []
    for key in group_payload.keys():
        profile_key = str(key).strip()
        if profile_key:
            options.append((profile_key, _display_name(profile_key)))
    return tuple(options)


def _discover_existing_audio_path(workspace_root: Path) -> str:
    for filename in PREFERRED_ROOT_AUDIO_FILENAMES:
        candidate = (workspace_root / filename).resolve()
        if candidate.is_file():
            return candidate.as_posix()
    return ""


def _discover_existing_summary_path(workspace_root: Path) -> str:
    candidate = (workspace_root / "summary.xml").resolve()
    if candidate.is_file():
        return candidate.as_posix()
    return ""


def _fuzzy_match(candidate: str, pattern: str) -> bool:
    """Return whether pattern characters appear in order inside candidate."""
    if not pattern:
        return True
    candidate_lower = candidate.lower()
    pattern_lower = pattern.lower()
    index = 0
    for char in pattern_lower:
        index = candidate_lower.find(char, index)
        if index < 0:
            return False
        index += 1
    return True


def _existing_parent_for_suggestions(candidate: Path, workspace_root: Path) -> Path:
    """Return the closest existing directory that can seed path suggestions."""
    resolved_workspace_root = workspace_root.resolve()
    current = candidate.expanduser().resolve()
    try:
        current.relative_to(resolved_workspace_root)
    except ValueError:
        return resolved_workspace_root
    if current.exists() and current.is_dir():
        return current.resolve()
    for parent in current.parents:
        try:
            parent.relative_to(resolved_workspace_root)
        except ValueError:
            break
        if parent.exists() and parent.is_dir():
            return parent.resolve()
    return resolved_workspace_root


@lru_cache(maxsize=256)
def _cached_directory_snapshot(
    base_dir_text: str,
    directory_mtime_ns: int,
) -> tuple[tuple[str, bool, str, str], ...]:
    """Return cached directory entries for path suggestions."""
    del directory_mtime_ns
    try:
        with os.scandir(base_dir_text) as entries:
            snapshot: list[tuple[str, bool, str, str]] = []
            for entry in entries:
                is_dir = entry.is_dir()
                name = entry.name
                rendered = Path(entry.path).absolute().as_posix() + (
                    "/" if is_dir else ""
                )
                snapshot.append(
                    (
                        name,
                        is_dir,
                        "" if is_dir else Path(name).suffix.lower(),
                        rendered,
                    )
                )
            return tuple(snapshot)
    except OSError:
        return ()


def _directory_snapshot(base_dir: Path) -> tuple[tuple[str, bool, str, str], ...]:
    """Return a cached snapshot of a directory's immediate children."""
    resolved_base_dir = base_dir.resolve()
    try:
        directory_mtime_ns = resolved_base_dir.stat().st_mtime_ns
    except OSError:
        return ()
    return _cached_directory_snapshot(str(resolved_base_dir), directory_mtime_ns)


def compute_path_suggestions(
    *,
    raw_value: str,
    workspace_root: Path,
    allowed_suffixes: tuple[str, ...],
    limit: int = 8,
) -> tuple[str, ...]:
    """Return fuzzy path suggestions rooted at the best matching directory."""
    value = str(raw_value).strip().strip('"').strip("'")
    if not value:
        return ()
    resolved_workspace_root = workspace_root.resolve()
    candidate = Path(value).expanduser() if value else resolved_workspace_root
    if not candidate.is_absolute():
        candidate = (resolved_workspace_root / candidate).resolve()
    clamped_to_workspace_root = False
    try:
        candidate.relative_to(resolved_workspace_root)
    except ValueError:
        candidate = resolved_workspace_root
        clamped_to_workspace_root = True
    directory_hint = clamped_to_workspace_root or value.endswith(("/", "\\"))
    if not directory_hint and candidate.exists() and candidate.is_file():
        return ()
    if directory_hint:
        base_dir = _existing_parent_for_suggestions(candidate, resolved_workspace_root)
        pattern = ""
    else:
        base_dir = _existing_parent_for_suggestions(
            candidate.parent,
            resolved_workspace_root,
        )
        pattern = candidate.name

    suggestions: list[tuple[tuple[int, int, str], str]] = []
    for name, is_dir, suffix, rendered in _directory_snapshot(base_dir):
        if not is_dir and allowed_suffixes and suffix not in allowed_suffixes:
            continue
        display_name = name + ("/" if is_dir else "")
        if pattern and not _fuzzy_match(display_name, pattern):
            continue
        suggestions.append(
            (
                (
                    0 if is_dir else 1,
                    0 if display_name.lower().startswith(pattern.lower()) else 1,
                    display_name.lower(),
                ),
                rendered,
            )
        )
    suggestions.sort(key=lambda item: item[0])
    return tuple(rendered for _sort_key, rendered in suggestions[:limit])


def load_frontend_defaults(
    *,
    config_path: Path,
    workspace_root: Path,
) -> FrontendDefaults:
    """Load config-backed defaults and discover likely input artifacts."""
    config_text = _read_config_text(config_path)
    try:
        parsed_config = OmegaConf.create(config_text)
    except Exception:  # pragma: no cover - invalid config should still be editable
        parsed_config = OmegaConf.create({})
    llm_profiles = _child_options(parsed_config, "llm") or (
        ("vllm_local", _display_name("vllm_local")),
    )
    embedding_profiles = _child_options(parsed_config, "embedding") or (
        ("disabled", _display_name("disabled")),
    )
    pipeline_defaults = PipelineProfileDefaults(
        llm=_safe_string(parsed_config, ("selection", "llm"), "vllm_local"),
        embedding=_safe_string(parsed_config, ("selection", "embedding"), "disabled"),
        stage_llm={
            stage: _safe_string(
                parsed_config,
                ("selection", "stage_llm", stage),
                "default",
            )
            for stage in ("summarizer", "critic", "interjector")
        },
        voice_clone_enabled=_safe_bool(
            parsed_config,
            ("audio", "voice_clone", "enabled"),
            True,
        ),
        interjector_enabled=_safe_bool(
            parsed_config,
            ("audio", "interjector", "enabled"),
            False,
        ),
    )
    discovered_transcript_path = discover_reusable_transcript_path(workspace_root)
    return FrontendDefaults(
        workspace_root=workspace_root.resolve(),
        config_path=config_path.resolve(),
        config_text=config_text,
        pipeline_defaults=pipeline_defaults,
        llm_profiles=llm_profiles,
        embedding_profiles=embedding_profiles,
        discovered_audio_path=_discover_existing_audio_path(workspace_root),
        discovered_transcript_path=(
            discovered_transcript_path.resolve().as_posix()
            if discovered_transcript_path is not None
            else ""
        ),
        discovered_summary_path=_discover_existing_summary_path(workspace_root),
        summary_benchmark_manifest=DEFAULT_BENCHMARK_MANIFEST_PATH.resolve().as_posix(),
        qa_config_path=DEFAULT_QA_CONFIG_PATH.resolve().as_posix(),
        qmsum_dataset_manifest=DEFAULT_QMSUM_DATASET_MANIFEST_PATH.resolve().as_posix(),
        sporc_dataset_manifest=DEFAULT_SPORC_DATASET_MANIFEST_PATH.resolve().as_posix(),
        sporc_full_dataset_manifest=DEFAULT_FULL_DATASET_MANIFEST_PATH.resolve().as_posix(),
        diarization_manifest=DEFAULT_DIARIZATION_MANIFEST_PATH.resolve().as_posix(),
        sporc_sweep_root=DEFAULT_SWEEP_ROOT.resolve().as_posix(),
    )


def validate_config_text(config_text: str) -> None:
    """Fail fast when config editor content is not valid YAML/OmegaConf."""
    try:
        parsed = OmegaConf.create(config_text)
    except Exception as exc:  # pragma: no cover - OmegaConf raises mixed types
        raise FrontendValidationError(f"Config is not valid YAML: {exc}") from exc
    if parsed is None:
        raise FrontendValidationError("Config must not be empty.")


def save_config_text(*, config_path: Path, config_text: str) -> None:
    """Validate and atomically persist edited config text."""
    validate_config_text(config_text)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        "w",
        encoding="utf-8",
        delete=False,
        dir=str(config_path.parent),
        suffix=config_path.suffix or ".tmp",
    ) as handle:
        handle.write(config_text)
        temp_path = Path(handle.name)
    os.replace(temp_path, config_path)


def parse_nonempty_lines(raw_text: str) -> tuple[str, ...]:
    """Parse one value per line from a multiline widget."""
    return tuple(line.strip() for line in raw_text.splitlines() if line.strip())


def parse_override_lines(raw_text: str) -> tuple[str, ...]:
    """Parse line-delimited Hydra overrides with basic validation."""
    overrides = parse_nonempty_lines(raw_text)
    invalid = [
        item
        for item in overrides
        if "=" not in item or not item.split("=", 1)[0].strip()
    ]
    if invalid:
        raise FrontendValidationError(
            "Each extra override must use KEY=VALUE format: " + ", ".join(invalid)
        )
    return overrides


def _reject_conflicting_pipeline_overrides(
    *,
    mode: str,
    extra_overrides: tuple[str, ...],
) -> None:
    """Reject extra overrides that would invalidate the selected frontend mode."""
    reserved_keys_by_mode: dict[str, tuple[str, ...]] = {
        "stage2_transcript": (
            "pipeline.start_stage",
            "transcript_path",
            "audio.audio_path",
            "pipeline.final_summary_path",
            "pipeline.voice_clone_manifest_path",
        ),
        "stage3_summary": (
            "pipeline.start_stage",
            "transcript_path",
            "pipeline.final_summary_path",
            "pipeline.voice_clone_manifest_path",
        ),
        "stage4_interjector": (
            "pipeline.start_stage",
            "transcript_path",
            "pipeline.final_summary_path",
            "pipeline.voice_clone_manifest_path",
        ),
        "resume_session": (
            "pipeline.start_stage",
            "transcript_path",
            "pipeline.final_summary_path",
            "pipeline.voice_clone_manifest_path",
        ),
    }
    reserved_keys = reserved_keys_by_mode.get(mode, ())
    conflicting = [
        override
        for override in extra_overrides
        if any(override.startswith(f"{key}=") for key in reserved_keys)
    ]
    if conflicting:
        raise FrontendValidationError(
            "These extra overrides conflict with the selected run mode: "
            + ", ".join(conflicting)
        )


def _append_flag(command: list[str], flag: str, value: str) -> None:
    if str(value).strip():
        command.extend([flag, str(value).strip()])


def _append_bool_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def _build_pipeline_runtime_overrides(
    *,
    defaults: PipelineProfileDefaults,
    shared_llm: str,
    embedding: str,
    stage_strategy: str,
    stage_choices: Mapping[str, str],
    voice_clone_enabled: bool,
    interjector_enabled: bool,
    extra_overrides: tuple[str, ...],
) -> tuple[str, ...]:
    overrides: list[str] = []
    if shared_llm != defaults.llm:
        overrides.append(f"selection.llm={shared_llm}")
    if embedding != defaults.embedding:
        overrides.append(f"selection.embedding={embedding}")
    if stage_strategy == "shared_for_all":
        for stage in ("summarizer", "critic", "interjector"):
            if defaults.stage_llm.get(stage, "default") != "default":
                overrides.append(f"selection.stage_llm.{stage}=default")
    elif stage_strategy == "custom":
        for stage in ("summarizer", "critic", "interjector"):
            choice = stage_choices.get(stage, defaults.stage_llm.get(stage, "default"))
            if choice != defaults.stage_llm.get(stage, "default"):
                overrides.append(f"selection.stage_llm.{stage}={choice}")
    if voice_clone_enabled != defaults.voice_clone_enabled:
        overrides.append(
            f"audio.voice_clone.enabled={'true' if voice_clone_enabled else 'false'}"
        )
    if not voice_clone_enabled:
        if defaults.interjector_enabled:
            overrides.append("audio.interjector.enabled=false")
    elif interjector_enabled != defaults.interjector_enabled:
        overrides.append(
            f"audio.interjector.enabled={'true' if interjector_enabled else 'false'}"
        )
    overrides.extend(extra_overrides)
    return tuple(overrides)


def build_pipeline_selection(
    *,
    defaults: PipelineProfileDefaults,
    values: Mapping[str, Any],
) -> InteractiveMenuSelection:
    """Project frontend pipeline form state into the legacy selection contract."""
    mode = str(values.get("mode", "stage1_single")).strip() or "stage1_single"
    output_root = str(values.get("output_root", "")).strip() or None
    primary_audio_path = str(values.get("primary_audio_path", "")).strip()
    additional_audio_paths = parse_nonempty_lines(
        str(values.get("additional_audio_paths", ""))
    )
    audio_paths = tuple(
        path
        for path in (primary_audio_path, *additional_audio_paths)
        if str(path).strip()
    )
    transcript_path = str(values.get("transcript_path", "")).strip() or None
    summary_path = str(values.get("summary_path", "")).strip() or None
    stage4_manifest_path = (
        str(values.get("voice_clone_manifest_path", "")).strip() or None
    )
    resume_session = str(values.get("resume_session", "")).strip() or None
    stage2_audio_path = str(values.get("stage2_audio_path", "")).strip() or None
    duration_presets = parse_nonempty_lines(str(values.get("duration_presets", "")))
    extra_overrides = parse_override_lines(str(values.get("extra_overrides", "")))
    _reject_conflicting_pipeline_overrides(mode=mode, extra_overrides=extra_overrides)
    runtime_overrides = _build_pipeline_runtime_overrides(
        defaults=defaults,
        shared_llm=str(values.get("shared_llm", defaults.llm)).strip() or defaults.llm,
        embedding=str(values.get("embedding", defaults.embedding)).strip()
        or defaults.embedding,
        stage_strategy=str(values.get("stage_strategy", "config_default")).strip()
        or "config_default",
        stage_choices={
            stage: str(
                values.get(
                    f"stage_llm_{stage}", defaults.stage_llm.get(stage, "default")
                )
            ).strip()
            or defaults.stage_llm.get(stage, "default")
            for stage in ("summarizer", "critic", "interjector")
        },
        voice_clone_enabled=bool(
            values.get("voice_clone_enabled", defaults.voice_clone_enabled)
        ),
        interjector_enabled=bool(
            values.get("interjector_enabled", defaults.interjector_enabled)
        ),
        extra_overrides=extra_overrides,
    )
    if mode == "stage1_single":
        if not audio_paths:
            raise FrontendValidationError(
                "A source audio path is required for a stage-1 run."
            )
        return InteractiveMenuSelection(
            mode="stage1_single",
            audio_paths=(audio_paths[0],),
            output_root=output_root,
            overrides=runtime_overrides,
        )
    if mode == "stage1_batch":
        if len(audio_paths) < 2:
            raise FrontendValidationError(
                "Batch mode requires at least two audio paths."
            )
        return InteractiveMenuSelection(
            mode="stage1_batch",
            audio_paths=audio_paths,
            output_root=output_root,
            overrides=runtime_overrides,
        )
    if mode == "stage1_batch_duration_sweep":
        if not audio_paths:
            raise FrontendValidationError(
                "Duration sweep mode requires at least one audio path."
            )
        return InteractiveMenuSelection(
            mode="stage1_batch_duration_sweep",
            audio_paths=audio_paths,
            duration_presets=duration_presets or DEFAULT_DURATION_SWEEP_PRESET_LABELS,
            output_root=output_root,
            overrides=runtime_overrides,
        )
    if mode == "stage2_transcript":
        if transcript_path is None:
            raise FrontendValidationError(
                "A transcript JSON path is required for stage-2."
            )
        stage2_audio_paths = (stage2_audio_path,) if stage2_audio_path else ()
        return InteractiveMenuSelection(
            mode="stage2_transcript",
            audio_paths=stage2_audio_paths,
            transcript_path=transcript_path,
            output_root=output_root,
            overrides=runtime_overrides,
        )
    if mode == "stage3_summary":
        if summary_path is None:
            raise FrontendValidationError("A summary XML path is required for stage-3.")
        return InteractiveMenuSelection(
            mode="stage3_summary",
            voiceclone_from_summary=summary_path,
            output_root=output_root,
            overrides=runtime_overrides,
        )
    if mode == "stage4_interjector":
        if summary_path is None:
            raise FrontendValidationError("A summary XML path is required for stage-4.")
        if stage4_manifest_path is None:
            raise FrontendValidationError(
                "A voice-clone manifest path is required for stage-4."
            )
        stage4_overrides = [
            override
            for override in runtime_overrides
            if not override.startswith("audio.interjector.enabled=")
        ]
        stage4_overrides.append("audio.interjector.enabled=true")
        return InteractiveMenuSelection(
            mode="stage4_interjector",
            voiceclone_from_summary=summary_path,
            voice_clone_manifest_path=stage4_manifest_path,
            output_root=output_root,
            overrides=tuple(stage4_overrides),
        )
    if mode == "resume_session":
        if resume_session is None:
            raise FrontendValidationError("A session UUID is required to resume a run.")
        return InteractiveMenuSelection(
            mode="resume_session",
            resume_session=resume_session,
            output_root=output_root,
        )
    return InteractiveMenuSelection(
        mode="setup_only",
        setup_only=True,
        output_root=output_root,
    )


def _build_summary_matrix_request(
    values: Mapping[str, Any], uv_executable: str
) -> BenchmarkLaunchRequest:
    vllm_host = str(values.get("matrix_vllm_host", "")).strip()
    if not vllm_host:
        raise FrontendValidationError("A vLLM host is required for the summary matrix.")
    input_mode = str(values.get("matrix_input_mode", "audio")).strip() or "audio"
    input_path = str(values.get("matrix_input_path", "")).strip()
    if not input_path:
        raise FrontendValidationError(
            "Provide an audio or transcript path for the summary matrix."
        )
    command = [
        uv_executable,
        "run",
        "python",
        "-m",
        "card_framework.cli.run_summary_matrix",
    ]
    _append_flag(command, "--vllm-host", vllm_host)
    _append_flag(
        command,
        "--audio-path" if input_mode == "audio" else "--transcript-path",
        input_path,
    )
    _append_flag(
        command, "--output-dir", str(values.get("benchmark_output_dir", "")).strip()
    )
    _append_flag(
        command,
        "--deepseek-api-key",
        str(values.get("matrix_deepseek_api_key", "")).strip(),
    )
    _append_flag(
        command,
        "--deepseek-model",
        str(values.get("matrix_deepseek_model", "")).strip(),
    )
    _append_flag(
        command,
        "--deepseek-base-url",
        str(values.get("matrix_deepseek_base_url", "")).strip(),
    )
    _append_bool_flag(
        command, "--fail-fast", bool(values.get("matrix_fail_fast", False))
    )
    command.extend(parse_nonempty_lines(str(values.get("benchmark_extra_args", ""))))
    return BenchmarkLaunchRequest(
        kind="summary_matrix",
        command=tuple(command),
        description="Run summary matrix experiment",
    )


def _build_summarizer_benchmark_request(
    values: Mapping[str, Any], uv_executable: str
) -> BenchmarkLaunchRequest:
    command = [
        uv_executable,
        "run",
        "python",
        "-m",
        "card_framework.benchmark.run",
        "execute",
    ]
    _append_flag(
        command, "--manifest", str(values.get("summarizer_manifest", "")).strip()
    )
    _append_flag(command, "--preset", str(values.get("summarizer_preset", "")).strip())
    _append_flag(
        command, "--output-dir", str(values.get("benchmark_output_dir", "")).strip()
    )
    _append_flag(
        command, "--providers", str(values.get("summarizer_providers", "")).strip()
    )
    _append_flag(
        command,
        "--judge-provider",
        str(values.get("summarizer_judge_provider", "")).strip(),
    )
    _append_flag(
        command, "--max-samples", str(values.get("summarizer_max_samples", "")).strip()
    )
    command.extend(parse_nonempty_lines(str(values.get("benchmark_extra_args", ""))))
    return BenchmarkLaunchRequest(
        kind="summarizer_benchmark",
        command=tuple(command),
        description="Run summarizer benchmark matrix",
    )


def _build_qa_benchmark_request(
    values: Mapping[str, Any], uv_executable: str
) -> BenchmarkLaunchRequest:
    summary_xml = str(values.get("qa_summary_xml", "")).strip()
    source_transcript = str(values.get("qa_source_transcript", "")).strip()
    if not summary_xml or not source_transcript:
        raise FrontendValidationError(
            "QA benchmark requires both a summary XML path and a source transcript path."
        )
    command = [uv_executable, "run", "python", "-m", "card_framework.benchmark.qa"]
    _append_flag(command, "--summary-xml", summary_xml)
    _append_flag(command, "--source-transcript", source_transcript)
    _append_flag(
        command,
        "--provider-profiles",
        DEFAULT_PROVIDER_PROFILES_PATH.resolve().as_posix(),
    )
    _append_flag(command, "--config", DEFAULT_CONFIG_PATH.resolve().as_posix())
    _append_flag(command, "--qa-config", str(values.get("qa_config_path", "")).strip())
    _append_flag(command, "--provider", str(values.get("qa_provider", "")).strip())
    _append_flag(
        command,
        "--creator-provider",
        str(values.get("qa_creator_provider", "")).strip(),
    )
    _append_flag(
        command,
        "--evaluator-provider",
        str(values.get("qa_evaluator_provider", "")).strip(),
    )
    _append_flag(
        command, "--output-dir", str(values.get("benchmark_output_dir", "")).strip()
    )
    _append_flag(
        command, "--output-format", str(values.get("qa_output_format", "")).strip()
    )
    _append_flag(command, "--color", str(values.get("qa_color", "")).strip())
    command.extend(parse_nonempty_lines(str(values.get("benchmark_extra_args", ""))))
    return BenchmarkLaunchRequest(
        kind="qa_benchmark",
        command=tuple(command),
        description="Run QA benchmark",
    )


def _build_qmsum_benchmark_request(
    values: Mapping[str, Any], uv_executable: str
) -> BenchmarkLaunchRequest:
    command = [
        uv_executable,
        "run",
        "python",
        "-m",
        "card_framework.benchmark.summarizer_critic",
        "execute",
    ]
    _append_flag(
        command,
        "--dataset-manifest",
        str(values.get("qmsum_dataset_manifest", "")).strip(),
    )
    _append_flag(
        command, "--output-root", str(values.get("benchmark_output_dir", "")).strip()
    )
    _append_flag(
        command,
        "--summarizer-provider",
        str(values.get("qmsum_summarizer_provider", "")).strip(),
    )
    _append_flag(
        command,
        "--critic-provider",
        str(values.get("qmsum_critic_provider", "")).strip(),
    )
    _append_flag(
        command, "--judge-provider", str(values.get("qmsum_judge_provider", "")).strip()
    )
    _append_flag(
        command, "--embedding-mode", str(values.get("qmsum_embedding_mode", "")).strip()
    )
    _append_flag(
        command, "--max-samples", str(values.get("qmsum_max_samples", "")).strip()
    )
    _append_flag(
        command, "--judge-repeats", str(values.get("qmsum_judge_repeats", "")).strip()
    )
    _append_flag(
        command, "--max-iterations", str(values.get("qmsum_max_iterations", "")).strip()
    )
    _append_bool_flag(
        command,
        "--prepare-if-missing",
        bool(values.get("qmsum_prepare_if_missing", False)),
    )
    _append_bool_flag(
        command,
        "--force-download-dataset",
        bool(values.get("qmsum_force_download_dataset", False)),
    )
    command.extend(parse_nonempty_lines(str(values.get("benchmark_extra_args", ""))))
    return BenchmarkLaunchRequest(
        kind="qmsum_benchmark",
        command=tuple(command),
        description="Run QMSum summarizer-critic benchmark",
    )


def _build_sporc_benchmark_request(
    values: Mapping[str, Any], uv_executable: str
) -> BenchmarkLaunchRequest:
    command = [
        uv_executable,
        "run",
        "python",
        "-m",
        "card_framework.benchmark.summarizer_critic.sporc",
        "execute",
    ]
    _append_flag(
        command,
        "--dataset-manifest",
        str(values.get("sporc_dataset_manifest", "")).strip(),
    )
    _append_flag(
        command,
        "--dataset-variant",
        str(values.get("sporc_dataset_variant", "")).strip(),
    )
    _append_flag(
        command, "--output-root", str(values.get("benchmark_output_dir", "")).strip()
    )
    _append_flag(
        command,
        "--summarizer-provider",
        str(values.get("sporc_summarizer_provider", "")).strip(),
    )
    _append_flag(
        command,
        "--critic-provider",
        str(values.get("sporc_critic_provider", "")).strip(),
    )
    _append_flag(
        command, "--judge-provider", str(values.get("sporc_judge_provider", "")).strip()
    )
    _append_flag(
        command, "--embedding-mode", str(values.get("sporc_embedding_mode", "")).strip()
    )
    _append_flag(
        command,
        "--duration-preset",
        str(values.get("sporc_duration_preset", "")).strip(),
    )
    _append_flag(
        command, "--max-samples", str(values.get("sporc_max_samples", "")).strip()
    )
    _append_flag(
        command, "--max-iterations", str(values.get("sporc_max_iterations", "")).strip()
    )
    _append_bool_flag(
        command, "--transcript-only", bool(values.get("sporc_transcript_only", True))
    )
    command.extend(parse_nonempty_lines(str(values.get("benchmark_extra_args", ""))))
    return BenchmarkLaunchRequest(
        kind="sporc_benchmark",
        command=tuple(command),
        description="Run SPoRC summarizer-critic benchmark",
    )


def _build_sporc_duration_sweep_request(
    values: Mapping[str, Any], uv_executable: str
) -> BenchmarkLaunchRequest:
    command = [
        uv_executable,
        "run",
        "python",
        "-m",
        "card_framework.benchmark.summarizer_critic.sporc",
        "duration-sweep",
    ]
    _append_flag(
        command,
        "--dataset-manifest",
        str(values.get("sporc_dataset_manifest", "")).strip(),
    )
    _append_flag(
        command,
        "--dataset-variant",
        str(values.get("sporc_dataset_variant", "")).strip(),
    )
    _append_flag(
        command, "--sweep-root", str(values.get("sporc_sweep_root", "")).strip()
    )
    _append_flag(command, "--mode", str(values.get("sporc_sweep_mode", "")).strip())
    _append_flag(
        command,
        "--duration-presets",
        str(values.get("sporc_sweep_duration_presets", "")).strip(),
    )
    _append_flag(
        command, "--examples", str(values.get("sporc_sweep_examples", "")).strip()
    )
    _append_flag(
        command, "--parallelism", str(values.get("sporc_sweep_parallelism", "")).strip()
    )
    _append_flag(
        command, "--concurrency", str(values.get("sporc_sweep_concurrency", "")).strip()
    )
    _append_bool_flag(
        command,
        "--enable-voice-clone",
        bool(values.get("sporc_sweep_enable_voice_clone", False)),
    )
    command.extend(parse_nonempty_lines(str(values.get("benchmark_extra_args", ""))))
    return BenchmarkLaunchRequest(
        kind="sporc_duration_sweep",
        command=tuple(command),
        description="Run SPoRC duration sweep benchmark",
    )


def _build_diarization_benchmark_request(
    values: Mapping[str, Any], uv_executable: str
) -> BenchmarkLaunchRequest:
    command = [
        uv_executable,
        "run",
        "python",
        "-m",
        "card_framework.benchmark.diarization",
        "execute",
    ]
    _append_flag(
        command, "--manifest", str(values.get("diarization_manifest", "")).strip()
    )
    _append_flag(
        command, "--output-dir", str(values.get("benchmark_output_dir", "")).strip()
    )
    _append_flag(
        command, "--providers", str(values.get("diarization_providers", "")).strip()
    )
    _append_flag(command, "--device", str(values.get("diarization_device", "")).strip())
    _append_flag(
        command, "--max-samples", str(values.get("diarization_max_samples", "")).strip()
    )
    _append_flag(command, "--collar", str(values.get("diarization_collar", "")).strip())
    _append_bool_flag(
        command, "--skip-overlap", bool(values.get("diarization_skip_overlap", False))
    )
    command.extend(parse_nonempty_lines(str(values.get("benchmark_extra_args", ""))))
    return BenchmarkLaunchRequest(
        kind="diarization_benchmark",
        command=tuple(command),
        description="Run diarization benchmark",
    )


_BENCHMARK_BUILDERS: dict[
    BenchmarkKind, Callable[[Mapping[str, Any], str], BenchmarkLaunchRequest]
] = {
    "summary_matrix": _build_summary_matrix_request,
    "summarizer_benchmark": _build_summarizer_benchmark_request,
    "qa_benchmark": _build_qa_benchmark_request,
    "qmsum_benchmark": _build_qmsum_benchmark_request,
    "sporc_benchmark": _build_sporc_benchmark_request,
    "sporc_duration_sweep": _build_sporc_duration_sweep_request,
    "diarization_benchmark": _build_diarization_benchmark_request,
}


def build_benchmark_request(
    *,
    kind: BenchmarkKind,
    values: Mapping[str, Any],
    uv_executable: str,
) -> BenchmarkLaunchRequest:
    """Dispatch benchmark request construction by benchmark kind."""
    return _BENCHMARK_BUILDERS[kind](values, uv_executable)
