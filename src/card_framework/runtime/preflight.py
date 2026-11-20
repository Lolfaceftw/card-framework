"""Readiness checks that run before the expensive parts of a CARD run.

Each check returns a :class:`Check` instead of raising, so the same checks can
power both the fail-fast path in ``infer()`` and the ``card-framework doctor``
report.
"""

from __future__ import annotations

import shutil
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from card_framework.shared.platform_support import (
    RuntimePlatform,
    is_supported_packaged_runtime,
    supported_packaged_runtime_targets_text,
)

VLLM_PROVIDER_TARGET = "card_framework.providers.vllm_provider.VLLMProvider"
_STAGE_LLM_NAMES = ("summarizer", "critic", "interjector")
_PROBE_TIMEOUT_SECONDS = 3.0
_RECOMMENDED_FREE_GB = 20

Status = Literal["ok", "warn", "fail"]


@dataclass(slots=True, frozen=True)
class Check:
    """One readiness result: what was checked, how it went, and how to fix it."""

    name: str
    status: Status
    detail: str
    fix: str = ""


class NotReadyError(RuntimeError):
    """Raised when a run cannot start because a prerequisite is missing."""

    def __init__(self, checks: list[Check]) -> None:
        self.checks = checks
        lines = ["CARD can't start this run yet."]
        for check in checks:
            lines.append(f"- {check.name}: {check.detail}")
            if check.fix:
                lines.append(f"  Fix: {check.fix}")
        lines.append("Run `card-framework doctor` to check everything at once.")
        super().__init__("\n".join(lines))


EndpointProbe = Callable[[str, str], tuple[int | None, str]]


def iter_llm_sections(
    config: Mapping[str, Any],
) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    """Yield the shared ``llm`` section and each non-empty per-stage override."""
    llm_cfg = config.get("llm")
    if isinstance(llm_cfg, dict) and llm_cfg:
        yield ("llm",), llm_cfg

    stage_llm_cfg = config.get("stage_llm")
    if not isinstance(stage_llm_cfg, dict):
        return
    for stage_name in _STAGE_LLM_NAMES:
        section_cfg = stage_llm_cfg.get(stage_name)
        if isinstance(section_cfg, dict) and section_cfg:
            yield ("stage_llm", stage_name), section_cfg


def llm_endpoint_checks(
    config: Mapping[str, Any],
    *,
    probe: EndpointProbe | None = None,
) -> list[Check]:
    """Check that every configured vLLM endpoint answers before the run starts.

    API providers such as DeepSeek or Gemini are not probed here; their keys are
    validated by the credential step in ``infer()``.
    """
    probe_endpoint = probe or probe_openai_endpoint
    checks: list[Check] = []
    seen_urls: set[str] = set()
    for _, section_cfg in iter_llm_sections(config):
        if str(section_cfg.get("_target_", "")).strip() != VLLM_PROVIDER_TARGET:
            continue
        base_url = str(section_cfg.get("base_url", "")).strip().rstrip("/")
        if not base_url or base_url in seen_urls:
            continue
        seen_urls.add(base_url)
        status_code, reason = probe_endpoint(base_url, str(section_cfg.get("api_key", "")))
        checks.append(_endpoint_check(base_url, status_code, reason))
    return checks


def probe_openai_endpoint(base_url: str, api_key: str) -> tuple[int | None, str]:
    """Request ``{base_url}/models`` and return the HTTP status, or None if unreachable."""
    request = urllib.request.Request(f"{base_url}/models", method="GET")
    if api_key and api_key != "EMPTY":
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=_PROBE_TIMEOUT_SECONDS) as response:
            return response.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, str(exc.reason)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = getattr(exc, "reason", exc)
        return None, str(reason)


def _endpoint_check(base_url: str, status_code: int | None, reason: str) -> Check:
    if status_code is not None and 200 <= status_code < 300:
        return Check("LLM", "ok", f"answering at {base_url}")
    if status_code in (401, 403):
        return Check(
            "LLM",
            "fail",
            f"{base_url} rejected the API key (HTTP {status_code})",
            "Set CARD_FRAMEWORK_VLLM_API_KEY to the key your server expects.",
        )
    if status_code is not None:
        return Check(
            "LLM",
            "fail",
            f"{base_url} answered HTTP {status_code}, not an OpenAI-compatible API",
            "Point to the server's /v1 address, for example http://host:8000/v1.",
        )
    return Check(
        "LLM",
        "fail",
        f"nothing is answering at {base_url} ({reason})",
        "Start a vLLM server there, pass --llm-url with another address, "
        "or use an API model such as --llm deepseek_chat with DEEPSEEK_API_KEY set.",
    )


def platform_check(platform: RuntimePlatform, device: Literal["cpu", "cuda"]) -> Check:
    """Report whether this OS and CPU architecture can run the packaged pipeline."""
    label = platform.display_label
    if is_supported_packaged_runtime(platform, device=device):
        return Check("Platform", "ok", label)
    return Check(
        "Platform",
        "fail",
        f"{label} can't run the packaged pipeline on {device}",
        f"Use {supported_packaged_runtime_targets_text(device=device)}, "
        "or the Docker workflow.",
    )


def tool_check(name: str, executable: str | None, fix: str) -> Check:
    """Report whether one external command-line tool is available."""
    if executable:
        return Check(name, "ok", executable)
    return Check(name, "fail", "not found", fix)


def ffmpeg_check() -> Check:
    """Find FFmpeg on PATH, then fall back to the bundled imageio-ffmpeg binary."""
    executable = shutil.which("ffmpeg")
    if executable is None:
        try:
            import imageio_ffmpeg

            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            executable = None
    return tool_check(
        "FFmpeg",
        executable,
        "Install FFmpeg and add it to PATH, or `pip install imageio-ffmpeg`.",
    )


def uv_check() -> Check:
    """Find the uv executable used to build the voice-cloning runtime."""
    return tool_check(
        "uv",
        shutil.which("uv"),
        "Install it into this environment: `python -m pip install -U uv`.",
    )


def compute_check(cuda_device_name: str | None, cuda_version: str | None) -> Check:
    """Describe the compute device a run will use."""
    if cuda_device_name:
        return Check("GPU", "ok", f"{cuda_device_name} (CUDA {cuda_version})")
    return Check(
        "GPU",
        "warn",
        "none in use, runs will use the CPU and take much longer",
        "With an NVIDIA GPU, run once with --device cuda to install the CUDA build of PyTorch.",
    )


def speaker_check(diarization_provider: str, fallback_reason: str | None) -> Check:
    """Report the diarization a run will use, warning when it falls back to one speaker.

    Args:
        diarization_provider: The configured diarization provider.
        fallback_reason: Why the run would fall back to single-speaker mode, or
            None when the configured provider can run here.
    """
    if fallback_reason is None:
        return Check("Speakers", "ok", f"{diarization_provider} diarization")
    return Check(
        "Speakers",
        "warn",
        f"every recording will be treated as one speaker. {fallback_reason}",
        "To separate speakers, run on Linux x86_64 (where pip installs nemo-toolkit) "
        "or use the Docker workflow.",
    )


def storage_check(free_bytes: int, location: str) -> Check:
    """Warn when the model cache drive is low on space."""
    free_gb = free_bytes / 1_000_000_000
    detail = f"{free_gb:.0f} GB free for models in {location}"
    if free_gb >= _RECOMMENDED_FREE_GB:
        return Check("Storage", "ok", detail)
    return Check(
        "Storage",
        "warn",
        detail,
        f"The first run downloads several GB of models; keep {_RECOMMENDED_FREE_GB} GB free "
        "or set CARD_FRAMEWORK_HOME to a larger drive.",
    )
