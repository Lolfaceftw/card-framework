"""Public library API for whole-pipeline CARD inference."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import getpass
import importlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any, Literal
import warnings

from card_framework.runtime.bootstrap import (
    RuntimeBootstrapError,
    ensure_ctc_forced_aligner_runtime,
    ensure_index_tts_runtime,
    ensure_runtime_requirements,
    resolve_uv_executable,
)
from card_framework.shared.paths import DEFAULT_CONFIG_PATH
from card_framework.shared.runtime_layout import RuntimeLayout, resolve_runtime_layout

_CONFIG_ENV_VAR = "CARD_FRAMEWORK_CONFIG"
_UV_ENV_VAR = "CARD_FRAMEWORK_UV_EXECUTABLE"
_VLLM_URL_ENV_VAR = "CARD_FRAMEWORK_VLLM_URL"
_VLLM_API_KEY_ENV_VAR = "CARD_FRAMEWORK_VLLM_API_KEY"
_SUPPORTED_PLATFORM = "Windows"
_SUPPORTED_CUDA_VERSION = "12.6"
_DEFAULT_VLLM_API_KEY = "EMPTY"
_PYTORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu126"
_PYTORCH_FALLBACK_INDEX_URL = "https://pypi.org/simple"
_VLLM_PROVIDER_TARGET = "card_framework.providers.vllm_provider.VLLMProvider"
_CUDA_VERSION_PATTERN = re.compile(r"(\d+\.\d+)")
_PROVIDER_API_KEY_REQUIREMENTS: dict[str, dict[str, object]] = {
    "card_framework.providers.deepseek_provider.DeepSeekProvider": {
        "credential_id": "deepseek_api_key",
        "prompt_label": "DeepSeek API key",
        "env_var_names": ("DEEPSEEK_API_KEY",),
    },
    "card_framework.providers.google_genai_provider.GoogleGenAIProvider": {
        "credential_id": "gemini_api_key",
        "prompt_label": "Gemini API key",
        "env_var_names": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    },
    "card_framework.providers.glm_provider.GLMProvider": {
        "credential_id": "zai_api_key",
        "prompt_label": "ZAI API key",
        "env_var_names": ("ZAI_API_KEY", "GLM_API_KEY"),
    },
    "card_framework.providers.huggingface_provider.HuggingfaceProvider": {
        "credential_id": "huggingface_token",
        "prompt_label": "Hugging Face access token",
        "env_var_names": (
            "HUGGINGFACE_TOKEN",
            "HF_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
        ),
    },
    "card_framework.providers.nanbeige_provider.NanbeigeProvider": {
        "credential_id": "huggingface_token",
        "prompt_label": "Hugging Face access token",
        "env_var_names": (
            "HUGGINGFACE_TOKEN",
            "HF_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
        ),
    },
}


@dataclass(slots=True, frozen=True)
class InferenceResult:
    """Describe the main artifacts emitted by one `infer` call."""

    output_dir: Path
    transcript_path: Path
    summary_xml_path: Path
    voice_clone_manifest_path: Path | None
    voice_clone_audio_path: Path | None
    interjector_manifest_path: Path | None
    final_audio_path: Path | None
    start_stage: Literal["stage-1"] = "stage-1"


@dataclass(slots=True, frozen=True)
class _CredentialRequirement:
    """Describe one missing runtime credential field."""

    credential_id: str
    field_path: tuple[str, ...]
    provider_label: str
    prompt_label: str
    env_var_names: tuple[str, ...]
    secret: bool = True


@dataclass(slots=True, frozen=True)
class _TorchCudaRuntimeState:
    """Capture one installed PyTorch CUDA runtime inspection."""

    cuda_version: str | None
    cuda_available: bool
    import_error: str | None = None

    @property
    def is_supported(self) -> bool:
        """Return whether the inspected runtime satisfies packaged CUDA needs."""
        return bool(
            self.cuda_version
            and self.cuda_version.startswith(_SUPPORTED_CUDA_VERSION)
            and self.cuda_available
        )

    @property
    def display_label(self) -> str:
        """Return the user-facing runtime label for errors and warnings."""
        return self.cuda_version or "cpu-only or unknown build"


@dataclass(slots=True, frozen=True)
class _HostCudaDetection:
    """Describe one host-level CUDA detection result."""

    version: str | None
    source: str | None


@dataclass(slots=True, frozen=True)
class _TorchRuntimeInstaller:
    """Describe the package manager command used for PyTorch self-repair."""

    tool: Literal["pip", "uv"]
    executable: str
    working_directory: Path | None = None


def infer(
    audio_wav: str | Path,
    output_dir: str | Path,
    target_duration_seconds: int,
    *,
    device: Literal["cpu", "cuda"],
    vllm_url: str | None = None,
    vllm_api_key: str | None = None,
) -> InferenceResult:
    """Run the full CARD pipeline for one source-audio input.

    Args:
        audio_wav: Source audio file for the stage-1 pipeline.
        output_dir: Directory that will receive the emitted run artifacts.
        target_duration_seconds: Required summary-duration target for this run.
        device: Runtime device for packaged inference. Supported values are
            ``"cpu"`` and ``"cuda"``.
        vllm_url: Optional OpenAI-compatible endpoint URL to force for all LLM stages.
        vllm_api_key: Optional API key paired with ``vllm_url`` or other vLLM configs.

    Returns:
        The main artifact bundle produced by the pipeline.

    Raises:
        FileNotFoundError: The audio input or configured YAML file does not exist.
        RuntimeError: Runtime prerequisites, missing credentials, unsupported
            platforms, or the pipeline subprocess fail.
        ValueError: The audio path, config path, or duration target is invalid.
    """
    _ensure_supported_platform()
    resolved_device = _resolve_requested_device(device)
    audio_path = _resolve_audio_input(audio_wav)
    resolved_target_duration_seconds = _resolve_target_duration_seconds(
        target_duration_seconds
    )
    resolved_output_dir = Path(output_dir).expanduser().resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    base_config_path = _resolve_config_path()
    base_config = _load_config_mapping(base_config_path)

    with _prepared_inference_config(
        base_config_path=base_config_path,
        base_config=base_config,
        device=resolved_device,
        vllm_url=vllm_url,
        vllm_api_key=vllm_api_key,
    ) as (config_path, config):
        audio_cfg = _as_mapping(config.get("audio", {}))
        voice_clone_cfg = _as_mapping(audio_cfg.get("voice_clone", {}))
        interjector_cfg = _as_mapping(audio_cfg.get("interjector", {}))
        live_drafting_cfg = _as_mapping(voice_clone_cfg.get("live_drafting", {}))

        voice_clone_enabled = bool(voice_clone_cfg.get("enabled", False))
        interjector_enabled = bool(interjector_cfg.get("enabled", False))
        live_drafting_enabled = voice_clone_enabled and bool(
            live_drafting_cfg.get("enabled", True)
        )
        calibration_enabled = not live_drafting_enabled

        raw_uv_executable = str(os.environ.get(_UV_ENV_VAR, "uv")).strip() or "uv"
        require_uv = voice_clone_enabled or interjector_enabled or calibration_enabled
        ensure_runtime_requirements(
            require_uv=require_uv,
            require_ffmpeg=True,
            uv_executable=raw_uv_executable,
        )
        uv_executable = (
            resolve_uv_executable(uv_executable=raw_uv_executable)
            if require_uv
            else raw_uv_executable
        )

        layout = resolve_runtime_layout()
        if require_uv:
            ensure_index_tts_runtime(layout=layout, uv_executable=uv_executable)

        command = _build_pipeline_command(
            audio_path=audio_path,
            output_dir=resolved_output_dir,
            config_path=config_path,
            layout=layout,
            target_duration_seconds=resolved_target_duration_seconds,
            voice_clone_enabled=voice_clone_enabled,
            uv_executable=uv_executable,
        )
        _run_pipeline_command(command=command, output_dir=resolved_output_dir)
        return _build_inference_result(
            output_dir=resolved_output_dir,
            config=config,
        )


def _ensure_supported_platform() -> None:
    """Reject unsupported packaged-runtime platforms before any other work starts."""
    current_platform = platform.system().strip() or "Unknown"
    if current_platform == _SUPPORTED_PLATFORM:
        return
    raise RuntimeError(
        "card_framework.infer(...) currently supports packaged whole-pipeline runs "
        f"on {_SUPPORTED_PLATFORM} only. Detected platform: {current_platform}. "
        "macOS and Linux are not yet validated for the public pip-installed runtime."
    )


@contextmanager
def _prepared_inference_config(
    *,
    base_config_path: Path,
    base_config: Mapping[str, Any],
    device: Literal["cpu", "cuda"],
    vllm_url: str | None,
    vllm_api_key: str | None,
) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Prepare an effective config for one inference call and clean it up."""
    effective_config = deepcopy(dict(base_config))
    config_modified = _apply_device_override(effective_config, device=device)

    resolved_vllm_url = _resolve_optional_text(vllm_url) or _resolve_optional_env(
        (_VLLM_URL_ENV_VAR, "VLLM_BASE_URL")
    )
    resolved_vllm_api_key = _resolve_optional_text(
        vllm_api_key
    ) or _resolve_optional_env((_VLLM_API_KEY_ENV_VAR, "VLLM_API_KEY"))

    if resolved_vllm_url:
        _apply_vllm_first_override(
            effective_config,
            base_url=resolved_vllm_url,
            api_key=resolved_vllm_api_key or _DEFAULT_VLLM_API_KEY,
        )
        config_modified = True
    elif resolved_vllm_api_key:
        config_modified = _apply_optional_vllm_api_key(
            effective_config,
            api_key=resolved_vllm_api_key,
        ) or config_modified

    config_modified = (
        _resolve_runtime_credentials(effective_config) or config_modified
    )
    config_modified = (
        _prepare_forced_alignment_runtime(effective_config) or config_modified
    )

    if not config_modified:
        yield base_config_path, effective_config
        return

    with TemporaryDirectory(prefix="card-framework-infer-") as temp_dir:
        temp_config_path = Path(temp_dir) / base_config_path.name
        _write_config_mapping(temp_config_path, effective_config)
        yield temp_config_path, effective_config


def _apply_vllm_first_override(
    config: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
) -> None:
    """Force all runtime LLM stages onto one OpenAI-compatible endpoint."""
    config["llm"] = {
        "_target_": _VLLM_PROVIDER_TARGET,
        "base_url": base_url,
        "api_key": api_key or _DEFAULT_VLLM_API_KEY,
    }
    stage_llm_cfg = _as_mapping(config.get("stage_llm", {}))
    for stage_name in ("summarizer", "critic", "interjector"):
        stage_llm_cfg[stage_name] = {}
    config["stage_llm"] = stage_llm_cfg


def _apply_device_override(
    config: dict[str, Any],
    *,
    device: Literal["cpu", "cuda"],
) -> bool:
    """Apply one explicit packaged-runtime device choice to relevant config nodes."""
    modified = False
    audio_cfg = _as_mapping(config.get("audio", {}))
    if _resolve_optional_text(str(audio_cfg.get("device", ""))) != device:
        audio_cfg["device"] = device
        modified = True

    voice_clone_cfg = _as_mapping(audio_cfg.get("voice_clone", {}))
    if _resolve_optional_text(str(voice_clone_cfg.get("device", ""))) != device:
        voice_clone_cfg["device"] = device
        audio_cfg["voice_clone"] = voice_clone_cfg
        modified = True

    config["audio"] = audio_cfg

    embedding_cfg = _as_mapping(config.get("embedding", {}))
    embedding_target = _resolve_optional_text(str(embedding_cfg.get("_target_", "")))
    if embedding_target and "NoOpEmbeddingProvider" not in embedding_target:
        if _resolve_optional_text(str(embedding_cfg.get("device", ""))) != device:
            embedding_cfg["device"] = device
            config["embedding"] = embedding_cfg
            modified = True

    return modified


def _apply_optional_vllm_api_key(config: dict[str, Any], *, api_key: str) -> bool:
    """Inject an explicit vLLM API key into any configured vLLM provider nodes."""
    if not api_key:
        return False
    modified = False
    for field_path, section_cfg in _iter_llm_sections(config):
        if _resolve_optional_text(str(section_cfg.get("_target_", ""))) != _VLLM_PROVIDER_TARGET:
            continue
        if _resolve_optional_text(str(section_cfg.get("api_key", ""))) == api_key:
            continue
        _set_nested_value(config, field_path + ("api_key",), api_key)
        modified = True
    return modified


def _resolve_runtime_credentials(config: dict[str, Any]) -> bool:
    """Resolve required credentials from config, environment, or secure prompts."""
    requirements = _collect_credential_requirements(config)
    if not requirements:
        return False

    prompted_values: dict[str, str] = {}
    missing_messages: list[str] = []
    modified = False

    for requirement in requirements:
        existing_value = _resolve_nested_text(config, requirement.field_path)
        if existing_value:
            continue

        resolved_value = _resolve_optional_env(requirement.env_var_names)
        if not resolved_value and _can_prompt_interactively():
            resolved_value = prompted_values.get(requirement.credential_id)
            if not resolved_value:
                resolved_value = _prompt_for_requirement(requirement)
                if resolved_value:
                    prompted_values[requirement.credential_id] = resolved_value

        if not resolved_value:
            missing_messages.append(_format_requirement_guidance(requirement))
            continue

        _set_nested_value(config, requirement.field_path, resolved_value)
        modified = True

    if missing_messages:
        raise RuntimeError(
            "CARD infer is missing required runtime credentials.\n"
            "Provide them in CARD_FRAMEWORK_CONFIG, set the documented environment "
            "variable, or rerun from an interactive terminal so infer() can prompt "
            "securely.\n"
            + "\n".join(f"- {message}" for message in missing_messages)
        )

    return modified


def _prepare_forced_alignment_runtime(config: dict[str, Any]) -> bool:
    """Bootstrap forced alignment when possible and fall back cleanly when not."""
    audio_cfg = _as_mapping(config.get("audio", {}))
    asr_cfg = _as_mapping(audio_cfg.get("asr", {}))
    forced_alignment_cfg = _as_mapping(asr_cfg.get("forced_alignment", {}))
    interjector_cfg = _as_mapping(audio_cfg.get("interjector", {}))

    forced_alignment_enabled = bool(forced_alignment_cfg.get("enabled", True))
    interjector_enabled = bool(interjector_cfg.get("enabled", False))
    if not forced_alignment_enabled and not interjector_enabled:
        return False

    try:
        ensure_ctc_forced_aligner_runtime(python_executable=sys.executable)
        return False
    except RuntimeBootstrapError as exc:
        modified = False
        if forced_alignment_enabled:
            forced_alignment_cfg["enabled"] = False
            forced_alignment_cfg["required"] = False
            asr_cfg["forced_alignment"] = forced_alignment_cfg
            audio_cfg["asr"] = asr_cfg
            config["audio"] = audio_cfg
            modified = True

        warnings.warn(
            "CARD could not bootstrap the pinned ctc-forced-aligner runtime; "
            "packaged inference will continue with approximate timing fallbacks. "
            f"Details: {exc}",
            RuntimeWarning,
            stacklevel=3,
        )
        return modified


def _collect_credential_requirements(
    config: Mapping[str, Any],
) -> list[_CredentialRequirement]:
    """Collect missing credential requirements from the effective config."""
    requirements: list[_CredentialRequirement] = []

    for field_path, section_cfg in _iter_llm_sections(config):
        target = _resolve_optional_text(str(section_cfg.get("_target_", "")))
        if not target:
            continue
        if target == _VLLM_PROVIDER_TARGET:
            if not _resolve_optional_text(str(section_cfg.get("base_url", ""))):
                requirements.append(
                    _CredentialRequirement(
                        credential_id="vllm_base_url",
                        field_path=field_path + ("base_url",),
                        provider_label=".".join(field_path),
                        prompt_label="vLLM base URL",
                        env_var_names=(_VLLM_URL_ENV_VAR, "VLLM_BASE_URL"),
                        secret=False,
                    )
                )
            continue

        provider_requirement = _PROVIDER_API_KEY_REQUIREMENTS.get(target)
        if not provider_requirement:
            continue
        requirements.append(
            _CredentialRequirement(
                credential_id=str(provider_requirement["credential_id"]),
                field_path=field_path + ("api_key",),
                provider_label=".".join(field_path),
                prompt_label=str(provider_requirement["prompt_label"]),
                env_var_names=tuple(provider_requirement["env_var_names"]),
            )
        )

    audio_cfg = _as_mapping(config.get("audio", {}))
    diarization_cfg = _as_mapping(audio_cfg.get("diarization", {}))
    if _resolve_optional_text(str(diarization_cfg.get("provider", ""))) == "pyannote_community1":
        pyannote_cfg = _as_mapping(diarization_cfg.get("pyannote", {}))
        auth_token_env = _resolve_optional_text(
            str(pyannote_cfg.get("auth_token_env", ""))
        )
        env_var_names = (
            (auth_token_env,)
            if auth_token_env
            else ("HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN")
        )
        requirements.append(
            _CredentialRequirement(
                credential_id="huggingface_token",
                field_path=("audio", "diarization", "pyannote", "auth_token"),
                provider_label="audio.diarization.pyannote",
                prompt_label="Hugging Face access token",
                env_var_names=env_var_names,
            )
        )

    deduped: list[_CredentialRequirement] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for requirement in requirements:
        dedupe_key = (requirement.credential_id, requirement.field_path)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(requirement)
    return deduped


def _iter_llm_sections(
    config: Mapping[str, Any],
) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    """Yield the shared and per-stage LLM config sections."""
    llm_cfg = _as_mapping(config.get("llm", {}))
    if llm_cfg:
        yield ("llm",), llm_cfg

    stage_llm_cfg = _as_mapping(config.get("stage_llm", {}))
    for stage_name in ("summarizer", "critic", "interjector"):
        section_cfg = _as_mapping(stage_llm_cfg.get(stage_name, {}))
        if section_cfg:
            yield ("stage_llm", stage_name), section_cfg


def _prompt_for_requirement(requirement: _CredentialRequirement) -> str:
    """Prompt for one missing credential in an interactive terminal."""
    prompt = (
        f"Enter {requirement.prompt_label} for {requirement.provider_label}: "
    )
    if requirement.secret:
        return getpass.getpass(prompt).strip()
    return input(prompt).strip()


def _format_requirement_guidance(requirement: _CredentialRequirement) -> str:
    """Format one actionable missing-credential error line."""
    field_name = ".".join(requirement.field_path)
    env_names = ", ".join(requirement.env_var_names) or "<none>"
    if requirement.credential_id == "vllm_base_url":
        return (
            f"{requirement.provider_label} needs a reachable vLLM URL. Set "
            f"`{field_name}` in CARD_FRAMEWORK_CONFIG, pass `vllm_url=...`, or set "
            f"one of: {env_names}."
        )
    return (
        f"{requirement.provider_label} needs {requirement.prompt_label}. Set "
        f"`{field_name}` in CARD_FRAMEWORK_CONFIG or set one of: {env_names}."
    )


def _can_prompt_interactively() -> bool:
    """Return whether secure interactive prompting is available."""
    stdin = getattr(sys, "stdin", None)
    stdout = getattr(sys, "stdout", None)
    if stdin is None or stdout is None:
        return False
    return bool(stdin.isatty() and stdout.isatty())


def _resolve_requested_device(device: str) -> Literal["cpu", "cuda"]:
    """Validate one packaged-runtime device request."""
    normalized = _resolve_optional_text(device).lower()
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device must be either 'cpu' or 'cuda'.")
    if normalized == "cuda":
        _ensure_supported_cuda_runtime()
    return normalized


def _ensure_supported_cuda_runtime() -> None:
    """Validate the packaged CUDA runtime contract for public infer calls."""
    runtime_state = _inspect_torch_cuda_runtime()
    if runtime_state.is_supported:
        return

    host_cuda = _detect_host_cuda_runtime()
    auto_repair_attempted = False
    if host_cuda.version and host_cuda.version.startswith(_SUPPORTED_CUDA_VERSION):
        auto_repair_attempted = True
        _install_supported_torch_cuda_runtime()
        runtime_state = _inspect_torch_cuda_runtime()
        if runtime_state.is_supported:
            warnings.warn(
                "infer(device='cuda') detected host CUDA 12.6 and automatically "
                "replaced the installed PyTorch build with the CUDA 12.6 wheels.",
                RuntimeWarning,
                stacklevel=3,
            )
            return

    if auto_repair_attempted:
        raise RuntimeError(
            "infer(device='cuda') detected host CUDA "
            f"{_SUPPORTED_CUDA_VERSION} via {host_cuda.source or 'an automatic probe'} "
            "and reinstalled the PyTorch CUDA 12.6 wheels, but the runtime is still "
            f"unsupported. Detected PyTorch CUDA runtime: {runtime_state.display_label}."
        )

    message = (
        "infer(device='cuda') currently supports only CUDA "
        f"{_SUPPORTED_CUDA_VERSION}. Detected PyTorch CUDA runtime: "
        f"{runtime_state.display_label}."
    )
    if runtime_state.import_error:
        message += f" Torch import detail: {runtime_state.import_error}."
    if host_cuda.version:
        message += (
            f" Automatic repair did not run because the host reported CUDA "
            f"{host_cuda.version} via {host_cuda.source or 'an automatic probe'}, "
            f"not CUDA {_SUPPORTED_CUDA_VERSION}."
        )
    else:
        message += (
            f" Automatic repair did not run because the host CUDA "
            f"{_SUPPORTED_CUDA_VERSION} runtime could not be confirmed."
        )
    raise RuntimeError(message)


def _inspect_torch_cuda_runtime() -> _TorchCudaRuntimeState:
    """Inspect the installed PyTorch CUDA runtime without importing it in-process."""
    inspection_script = """
import json

payload = {
    "cuda_version": None,
    "cuda_available": False,
    "import_error": None,
}

try:
    import torch
except Exception as exc:  # pragma: no cover - executed in the child process
    payload["import_error"] = f"{type(exc).__name__}: {exc}"
else:
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    payload["cuda_version"] = str(cuda_version).strip() if cuda_version else None
    cuda_module = getattr(torch, "cuda", None)
    is_available = getattr(cuda_module, "is_available", None)
    if callable(is_available):
        try:
            payload["cuda_available"] = bool(is_available())
        except Exception as exc:  # pragma: no cover - executed in the child process
            payload["import_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(payload))
"""
    completed = subprocess.run(
        [sys.executable, "-c", inspection_script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        error_message = detail[-500:] if detail else "inspection subprocess failed"
        return _TorchCudaRuntimeState(
            cuda_version=None,
            cuda_available=False,
            import_error=error_message,
        )

    try:
        payload = json.loads(completed.stdout.strip() or "{}")
    except json.JSONDecodeError:
        return _TorchCudaRuntimeState(
            cuda_version=None,
            cuda_available=False,
            import_error="torch inspection returned invalid JSON",
        )
    if not isinstance(payload, dict):
        return _TorchCudaRuntimeState(
            cuda_version=None,
            cuda_available=False,
            import_error="torch inspection returned an invalid payload",
        )
    return _TorchCudaRuntimeState(
        cuda_version=_resolve_optional_text(str(payload.get("cuda_version", "") or "")),
        cuda_available=bool(payload.get("cuda_available", False)),
        import_error=_resolve_optional_text(str(payload.get("import_error", "") or "")),
    )


def _detect_host_cuda_runtime() -> _HostCudaDetection:
    """Detect the host CUDA version that packaged infer can safely auto-repair to."""
    cuda_path_126 = _resolve_optional_text(os.environ.get("CUDA_PATH_V12_6", ""))
    if cuda_path_126:
        cuda_dir = Path(cuda_path_126).expanduser()
        if cuda_dir.exists():
            return _HostCudaDetection(
                version=_SUPPORTED_CUDA_VERSION,
                source="CUDA_PATH_V12_6",
            )

    cuda_path = _resolve_optional_text(os.environ.get("CUDA_PATH", ""))
    if cuda_path:
        cuda_dir = Path(cuda_path).expanduser()
        detected_version = _detect_cuda_version_from_directory(cuda_dir)
        if detected_version:
            return _HostCudaDetection(version=detected_version, source="CUDA_PATH")

    for command, source in ((["nvidia-smi"], "nvidia-smi"), (["nvcc", "--version"], "nvcc")):
        detected_version = _detect_cuda_version_from_command(command)
        if detected_version:
            return _HostCudaDetection(version=detected_version, source=source)

    return _HostCudaDetection(version=None, source=None)


def _detect_cuda_version_from_directory(cuda_dir: Path) -> str | None:
    """Read a CUDA version from one candidate toolkit directory."""
    if not cuda_dir.exists():
        return None

    version_json_path = cuda_dir / "version.json"
    if version_json_path.exists():
        try:
            payload = json.loads(version_json_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = None
        version_text = _extract_cuda_version_text(payload)
        if version_text:
            return version_text

    candidate_names = [cuda_dir.name, str(cuda_dir)]
    for candidate in candidate_names:
        version_text = _extract_cuda_version_text(candidate)
        if version_text:
            return version_text
    return None


def _extract_cuda_version_text(value: object) -> str | None:
    """Extract the first CUDA-like version string from one nested payload."""
    if isinstance(value, str):
        match = _CUDA_VERSION_PATTERN.search(value)
        if match is None:
            return None
        return match.group(1)
    if isinstance(value, Mapping):
        for nested_value in value.values():
            version_text = _extract_cuda_version_text(nested_value)
            if version_text:
                return version_text
    if isinstance(value, list):
        for nested_value in value:
            version_text = _extract_cuda_version_text(nested_value)
            if version_text:
                return version_text
    return None


def _detect_cuda_version_from_command(command: list[str]) -> str | None:
    """Run one host CUDA discovery command and parse its reported version."""
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    output = "\n".join(
        part.strip() for part in (completed.stdout, completed.stderr) if part
    )
    if not output:
        return None
    for pattern in (r"CUDA Version:\s*(\d+\.\d+)", r"release\s+(\d+\.\d+)"):
        match = re.search(pattern, output)
        if match is not None:
            return match.group(1)
    return None


def _install_supported_torch_cuda_runtime() -> None:
    """Replace the installed PyTorch wheels with the supported CUDA 12.6 build."""
    installer = _resolve_torch_runtime_installer()
    _clear_loaded_torch_modules()
    if installer.tool == "uv":
        _run_package_manager_command(
            step="pytorch_uninstall",
            command=[
                installer.executable,
                "pip",
                "uninstall",
                "--python",
                sys.executable,
                "torch",
                "torchaudio",
            ],
            cwd=installer.working_directory,
        )
        _run_package_manager_command(
            step="pytorch_install",
            command=[
                installer.executable,
                "pip",
                "install",
                "--python",
                sys.executable,
                "--reinstall",
                "--torch-backend",
                "cu126",
                "torch",
                "torchaudio",
            ],
            cwd=installer.working_directory,
        )
    else:
        _run_package_manager_command(
            step="pytorch_uninstall",
            command=[
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "--yes",
                "torch",
                "torchaudio",
            ],
        )
        _run_package_manager_command(
            step="pytorch_install",
            command=[
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-input",
                "--upgrade",
                "--force-reinstall",
                "--index-url",
                _PYTORCH_CUDA_INDEX_URL,
                "--extra-index-url",
                _PYTORCH_FALLBACK_INDEX_URL,
                "torch",
                "torchaudio",
            ],
        )
    _clear_loaded_torch_modules()
    importlib.invalidate_caches()


def _clear_loaded_torch_modules() -> None:
    """Drop loaded torch modules so later imports see any repaired wheel set."""
    for module_name in list(sys.modules):
        if module_name == "torch" or module_name.startswith("torch."):
            sys.modules.pop(module_name, None)


def _resolve_torch_runtime_installer() -> _TorchRuntimeInstaller:
    """Choose the package manager used for packaged PyTorch self-repair."""
    uv_project_root = _detect_uv_project_root(Path.cwd())
    if uv_project_root is not None:
        raw_uv_executable = str(os.environ.get(_UV_ENV_VAR, "uv")).strip() or "uv"
        try:
            uv_executable = resolve_uv_executable(uv_executable=raw_uv_executable)
        except RuntimeBootstrapError:
            uv_executable = ""
        if uv_executable:
            return _TorchRuntimeInstaller(
                tool="uv",
                executable=uv_executable,
                working_directory=uv_project_root,
            )
    return _TorchRuntimeInstaller(tool="pip", executable=sys.executable)


def _detect_uv_project_root(start_dir: Path) -> Path | None:
    """Return the nearest parent directory that looks like a uv-managed project."""
    for candidate in (start_dir, *start_dir.parents):
        if (candidate / "uv.lock").exists():
            return candidate

        pyproject_path = candidate / "pyproject.toml"
        if not pyproject_path.exists():
            continue
        try:
            pyproject_text = pyproject_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if "[tool.uv" in pyproject_text or "[dependency-groups]" in pyproject_text:
            return candidate
    return None


def _run_package_manager_command(
    *,
    step: str,
    command: list[str],
    cwd: Path | None = None,
) -> None:
    """Run one installer command for packaged runtime self-healing."""
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        return

    detail = (completed.stderr or completed.stdout or "").strip()
    detail_tail = (
        detail[-1200:] if detail else "No installer error output was captured."
    )
    raise RuntimeError(
        "infer(device='cuda') could not automatically repair the PyTorch CUDA "
        f"{_SUPPORTED_CUDA_VERSION} runtime during `{step}`. "
        f"Command: {' '.join(command)}. Details: {detail_tail}"
    )


def _resolve_audio_input(audio_wav: str | Path) -> Path:
    """Resolve and validate one source-audio input path."""
    candidate = Path(audio_wav).expanduser()
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"Audio input does not exist: {candidate}")
    if not candidate.is_file():
        raise ValueError(f"Audio input is not a file: {candidate}")
    return candidate


def _resolve_config_path() -> Path:
    """Return the full config path used for one inference call."""
    configured_path = str(os.environ.get(_CONFIG_ENV_VAR, "")).strip()
    if not configured_path:
        return DEFAULT_CONFIG_PATH.resolve()
    candidate = Path(configured_path).expanduser()
    if not candidate.is_absolute():
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Configured CARD framework config does not exist: {candidate}"
        )
    if candidate.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            "CARD_FRAMEWORK_CONFIG must point to a .yaml or .yml file."
        )
    return candidate


def _resolve_target_duration_seconds(target_duration_seconds: int) -> int:
    """Validate the required duration target for one inference call."""
    if isinstance(target_duration_seconds, bool) or not isinstance(
        target_duration_seconds, int
    ):
        raise ValueError("target_duration_seconds must be an integer number of seconds.")
    if target_duration_seconds <= 0:
        raise ValueError("target_duration_seconds must be greater than zero.")
    return target_duration_seconds


def _load_omegaconf() -> Any:
    """Import and return OmegaConf with a consistent error message."""
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise RuntimeError(
            "omegaconf is required to load CARD runtime configuration."
        ) from exc
    return OmegaConf


def _load_config_mapping(config_path: Path) -> dict[str, Any]:
    """Load one Hydra-style YAML config as a plain dictionary."""
    omega_conf = _load_omegaconf()
    config = omega_conf.load(config_path)
    resolved = omega_conf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"Config did not resolve to a mapping: {config_path}")
    return resolved


def _write_config_mapping(config_path: Path, config: Mapping[str, Any]) -> None:
    """Persist one config mapping for the subprocess runtime."""
    omega_conf = _load_omegaconf()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    omega_conf.save(config=omega_conf.create(dict(config)), f=str(config_path))


def _build_pipeline_command(
    *,
    audio_path: Path,
    output_dir: Path,
    config_path: Path,
    layout: RuntimeLayout,
    target_duration_seconds: int,
    voice_clone_enabled: bool,
    uv_executable: str,
) -> list[str]:
    """Build the subprocess command used to execute the existing Hydra pipeline."""
    command = [
        sys.executable,
        "-m",
        "card_framework.cli.main",
        "--config-path",
        str(config_path.parent),
        "--config-name",
        config_path.stem,
        "hydra.run.dir=.",
        "hydra.output_subdir=null",
        "pipeline.start_stage=stage-1",
        f"orchestrator.target_seconds={target_duration_seconds}",
        f"audio.audio_path={_hydra_path(audio_path)}",
        "audio.output_transcript_path=transcript.json",
        "transcript_path=transcript.json",
        "audio.work_dir=audio_stage",
        "logging.log_file=agent_interactions.log",
        "logging.print_to_terminal=false",
        "logging.summarizer_critic_print_to_terminal=false",
    ]
    if voice_clone_enabled:
        command.extend(
            [
                f"audio.voice_clone.runner_project_dir={_hydra_path(layout.vendor_runtime_dir)}",
                f"audio.voice_clone.cfg_path={_hydra_path(layout.checkpoints_dir / 'config.yaml')}",
                f"audio.voice_clone.model_dir={_hydra_path(layout.checkpoints_dir)}",
                f"audio.voice_clone.uv_executable={uv_executable}",
            ]
        )
    del output_dir
    return command


def _run_pipeline_command(*, command: list[str], output_dir: Path) -> None:
    """Run the pipeline subprocess and raise a concise error on failure."""
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    completed = subprocess.run(
        command,
        cwd=output_dir,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if completed.returncode != 0:
        details = _tail_text(completed.stderr or completed.stdout or "", 4000)
        raise RuntimeError(
            "CARD pipeline execution failed.\n"
            f"Command: {' '.join(command)}\n"
            f"Details:\n{details}"
        )


def _build_inference_result(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
) -> InferenceResult:
    """Construct the typed result object from configured artifact paths."""
    audio_cfg = _as_mapping(config.get("audio", {}))
    voice_clone_cfg = _as_mapping(audio_cfg.get("voice_clone", {}))
    interjector_cfg = _as_mapping(audio_cfg.get("interjector", {}))

    transcript_path = output_dir / "transcript.json"
    summary_xml_path = output_dir / "summary.xml"
    work_dir = output_dir / "audio_stage"

    voice_clone_dir = work_dir / str(voice_clone_cfg.get("output_dir_name", "voice_clone"))
    voice_clone_manifest_path = _optional_existing_path(
        voice_clone_dir / str(voice_clone_cfg.get("manifest_filename", "manifest.json"))
    )
    voice_clone_audio_path = _optional_existing_path(
        voice_clone_dir / str(voice_clone_cfg.get("merged_output_filename", "voice_cloned.wav"))
    )

    interjector_dir = work_dir / str(interjector_cfg.get("output_dir_name", "interjector"))
    interjector_manifest_path = _optional_existing_path(
        interjector_dir / str(interjector_cfg.get("manifest_filename", "interjector_manifest.json"))
    )
    interjector_audio_path = _optional_existing_path(
        interjector_dir / str(
            interjector_cfg.get("merged_output_filename", "voice_cloned_interjected.wav")
        )
    )

    return InferenceResult(
        output_dir=output_dir,
        transcript_path=transcript_path.resolve(),
        summary_xml_path=summary_xml_path.resolve(),
        voice_clone_manifest_path=voice_clone_manifest_path,
        voice_clone_audio_path=voice_clone_audio_path,
        interjector_manifest_path=interjector_manifest_path,
        final_audio_path=interjector_audio_path or voice_clone_audio_path,
    )


def _resolve_optional_env(env_var_names: tuple[str, ...]) -> str:
    """Return the first non-empty environment value from the provided names."""
    for env_var_name in env_var_names:
        value = str(os.environ.get(env_var_name, "")).strip()
        if value:
            return value
    return ""


def _resolve_optional_text(value: str | None) -> str:
    """Normalize optional user-facing text input into a stripped string."""
    return str(value or "").strip()


def _resolve_nested_text(config: Mapping[str, Any], field_path: tuple[str, ...]) -> str:
    """Read a nested string-like field from a config mapping."""
    current: Any = config
    for key in field_path:
        if not isinstance(current, Mapping):
            return ""
        current = current.get(key)
    return _resolve_optional_text(str(current or ""))


def _set_nested_value(
    config: dict[str, Any],
    field_path: tuple[str, ...],
    value: str,
) -> None:
    """Set one nested config field, creating intermediate mappings as needed."""
    current = config
    for key in field_path[:-1]:
        next_value = current.get(key)
        if isinstance(next_value, dict):
            current = next_value
            continue
        if isinstance(next_value, Mapping):
            next_dict = dict(next_value)
        else:
            next_dict = {}
        current[key] = next_dict
        current = next_dict
    current[field_path[-1]] = value


def _optional_existing_path(path: Path) -> Path | None:
    """Return the resolved path only when the artifact exists."""
    if not path.exists():
        return None
    return path.resolve()


def _hydra_path(path: Path) -> str:
    """Return an absolute path string safe for Hydra override usage."""
    return path.resolve().as_posix()


def _tail_text(text: str, limit: int) -> str:
    """Return a trimmed tail for subprocess failure details."""
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[-limit:]


def _as_mapping(value: Any) -> dict[str, Any]:
    """Coerce config values into a mutable dictionary."""
    if isinstance(value, Mapping):
        return dict(value)
    return {}


__all__ = ["InferenceResult", "infer", "RuntimeBootstrapError"]
