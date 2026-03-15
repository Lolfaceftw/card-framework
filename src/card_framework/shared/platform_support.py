"""Platform capability helpers for packaged runtime support and fallbacks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping
import importlib.util
import platform as platform_module
from typing import Any, Literal
import warnings

type StatusReporter = Callable[[str], None]

_PACKAGED_CPU_TARGETS: frozenset[tuple[str, str]] = frozenset(
    {
        ("windows", "x86_64"),
        ("linux", "x86_64"),
        ("darwin", "arm64"),
    }
)
_PACKAGED_CUDA_TARGETS: frozenset[tuple[str, str]] = frozenset(
    {
        ("windows", "x86_64"),
        ("linux", "x86_64"),
    }
)
_NEMO_SUPPORTED_TARGETS: frozenset[tuple[str, str]] = frozenset(
    {
        ("linux", "x86_64"),
        ("linux", "arm64"),
        ("darwin", "arm64"),
    }
)
_NEMO_DERIVED_DIARIZATION_PROVIDERS: frozenset[str] = frozenset(
    {
        "nemo",
        "nemo_sortformer_offline",
        "sortformer_offline",
        "nemo_sortformer_streaming",
        "sortformer_streaming",
    }
)


class UnsupportedPackagedRuntimeError(RuntimeError):
    """Raise when packaged `infer(...)` is not supported on the current host."""


class RuntimePlatform:
    """Describe one normalized runtime platform and machine combination."""

    def __init__(self, *, system: str, machine: str) -> None:
        self.system = system
        self.machine = machine

    @property
    def system_key(self) -> str:
        """Return the normalized lowercase operating-system key."""
        return self.system.strip().lower() or "unknown"

    @property
    def machine_key(self) -> str:
        """Return the normalized lowercase machine key."""
        return normalize_machine(self.machine)

    @property
    def display_label(self) -> str:
        """Return a user-facing platform description."""
        system_label = {
            "darwin": "macOS",
            "linux": "Linux",
            "windows": "Windows",
        }.get(self.system_key, self.system or "Unknown")
        return f"{system_label} {self.machine_key}"


def normalize_machine(machine: str) -> str:
    """Normalize architecture names into stable keys."""
    normalized = machine.strip().lower()
    if normalized in {"amd64", "x64", "x86_64"}:
        return "x86_64"
    if normalized in {"aarch64", "arm64"}:
        return "arm64"
    return normalized or "unknown"


def detect_runtime_platform(
    *,
    system: str | None = None,
    machine: str | None = None,
) -> RuntimePlatform:
    """Return the current normalized runtime platform."""
    return RuntimePlatform(
        system=(system or platform_module.system() or "Unknown").strip(),
        machine=(machine or platform_module.machine() or "unknown").strip(),
    )


def supported_packaged_runtime_targets_text(
    *, device: Literal["cpu", "cuda"]
) -> str:
    """Return the supported packaged-runtime targets for one device class."""
    if device == "cpu":
        return "Windows x86_64, Linux x86_64, and macOS arm64"
    return "Windows x86_64 and Linux x86_64"


def is_supported_packaged_runtime(
    runtime_platform: RuntimePlatform,
    *,
    device: Literal["cpu", "cuda"],
) -> bool:
    """Return whether packaged `infer(...)` supports this host and device."""
    supported_targets = (
        _PACKAGED_CUDA_TARGETS if device == "cuda" else _PACKAGED_CPU_TARGETS
    )
    return (
        runtime_platform.system_key,
        runtime_platform.machine_key,
    ) in supported_targets


def ensure_supported_packaged_runtime(
    *,
    device: Literal["cpu", "cuda"],
    runtime_platform: RuntimePlatform | None = None,
) -> RuntimePlatform:
    """Validate packaged `infer(...)` support for the current host."""
    resolved_platform = runtime_platform or detect_runtime_platform()
    if is_supported_packaged_runtime(resolved_platform, device=device):
        return resolved_platform

    message = (
        f"card_framework.infer(..., device='{device}') currently supports "
        "packaged whole-pipeline runs on "
        f"{supported_packaged_runtime_targets_text(device=device)}. "
        f"Detected platform: {resolved_platform.display_label}."
    )
    if device == "cuda" and resolved_platform.system_key == "darwin":
        message += " macOS remains CPU-only for the packaged runtime."
    raise UnsupportedPackagedRuntimeError(message)


def supports_nemo_backend(runtime_platform: RuntimePlatform) -> bool:
    """Return whether NeMo-derived diarization backends are platform-supported."""
    return (
        runtime_platform.system_key,
        runtime_platform.machine_key,
    ) in _NEMO_SUPPORTED_TARGETS


def apply_runtime_platform_fallbacks(
    config: MutableMapping[str, Any],
    *,
    reporter: StatusReporter | None = None,
    runtime_platform: RuntimePlatform | None = None,
) -> bool:
    """Apply runtime-safe platform fallbacks to a mutable config mapping."""
    resolved_platform = runtime_platform or detect_runtime_platform()
    audio_cfg = _as_mapping(config.get("audio"))
    diarization_cfg = _as_mapping(audio_cfg.get("diarization"))
    provider = str(diarization_cfg.get("provider", "nemo")).strip().lower()

    if provider not in _NEMO_DERIVED_DIARIZATION_PROVIDERS:
        return False

    reason = _nemo_diarization_fallback_reason(
        provider=provider,
        runtime_platform=resolved_platform,
    )
    if reason is None:
        return False

    diarization_cfg["provider"] = "single_speaker"
    audio_cfg["diarization"] = diarization_cfg
    config["audio"] = audio_cfg

    message = (
        f"{reason} Falling back to `audio.diarization.provider=single_speaker`."
    )
    if reporter is not None:
        reporter(message)
    else:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    return True


def _nemo_diarization_fallback_reason(
    *,
    provider: str,
    runtime_platform: RuntimePlatform,
) -> str | None:
    """Return a fallback reason for one NeMo-derived diarization provider."""
    if not supports_nemo_backend(runtime_platform):
        return (
            f"Configured audio.diarization.provider='{provider}' is not supported on "
            f"{runtime_platform.display_label}."
        )
    if importlib.util.find_spec("nemo") is None:
        return (
            f"Configured audio.diarization.provider='{provider}' requires the "
            "optional `nemo-toolkit[asr]` runtime, which is not installed."
        )
    return None


def _as_mapping(value: object) -> dict[str, Any]:
    """Convert mapping-like values into plain mutable dictionaries."""
    if isinstance(value, MutableMapping):
        return dict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}
