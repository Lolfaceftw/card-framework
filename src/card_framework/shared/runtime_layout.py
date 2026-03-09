"""Resolve writable runtime paths for installed CARD package usage."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from card_framework.shared.paths import PACKAGE_ROOT, VENDOR_INDEX_TTS_DIR

_RUNTIME_HOME_ENV_VAR = "CARD_FRAMEWORK_HOME"


@dataclass(slots=True, frozen=True)
class RuntimeLayout:
    """Describe writable runtime locations for package-managed assets."""

    runtime_home: Path
    vendor_source_dir: Path
    vendor_runtime_dir: Path
    checkpoints_dir: Path
    bootstrap_state_path: Path


def resolve_runtime_home(*, environ: dict[str, str] | None = None) -> Path:
    """Return the writable runtime home for package-managed assets."""
    env = os.environ if environ is None else environ
    configured_home = str(env.get(_RUNTIME_HOME_ENV_VAR, "")).strip()
    if configured_home:
        return Path(configured_home).expanduser().resolve()

    try:
        from platformdirs import user_data_path
    except ImportError as exc:
        raise RuntimeError(
            "platformdirs is required to resolve the CARD runtime home."
        ) from exc
    return user_data_path("card-framework", appauthor=False).resolve()


def resolve_runtime_layout(*, environ: dict[str, str] | None = None) -> RuntimeLayout:
    """Build the writable runtime layout used by the library API."""
    runtime_home = resolve_runtime_home(environ=environ)
    return RuntimeLayout(
        runtime_home=runtime_home,
        vendor_source_dir=VENDOR_INDEX_TTS_DIR.resolve(),
        vendor_runtime_dir=(runtime_home / "vendor" / "index_tts").resolve(),
        checkpoints_dir=(runtime_home / "checkpoints" / "index_tts").resolve(),
        bootstrap_state_path=(runtime_home / "bootstrap" / "state.json").resolve(),
    )


def package_root() -> Path:
    """Return the installed package root for diagnostic callers."""
    return PACKAGE_ROOT
