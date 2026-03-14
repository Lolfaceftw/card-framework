"""Load repository-local environment files when present."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from dotenv import load_dotenv

from card_framework.shared.paths import REPO_ROOT

_LOADED_DOTENV_PATHS: set[Path] = set()
_LOAD_LOCK = Lock()
_ENV_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "HF_ACCESS_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    ),
    ("VLLM_URL", "CARD_FRAMEWORK_VLLM_URL", "VLLM_BASE_URL"),
)


def _propagate_env_aliases() -> None:
    """Mirror equivalent environment variables without overriding existing values."""

    for alias_group in _ENV_ALIAS_GROUPS:
        resolved_value = next(
            (
                candidate
                for env_name in alias_group
                if (candidate := str(os.environ.get(env_name, "")).strip())
            ),
            "",
        )
        if not resolved_value:
            continue
        for env_name in alias_group:
            os.environ.setdefault(env_name, resolved_value)


def load_repo_dotenv(
    repo_root: Path = REPO_ROOT, *, override: bool = False
) -> Path | None:
    """Load the repository-root ``.env`` file once when it exists.

    Args:
        repo_root: Repository root that may contain a ``.env`` file.
        override: Whether values from the file should replace existing process
            environment variables.

    Returns:
        The resolved ``.env`` path when a file was loaded or had already been
        loaded, otherwise ``None``.
    """

    env_path = (repo_root / ".env").resolve()
    if not env_path.is_file():
        return None

    with _LOAD_LOCK:
        if env_path in _LOADED_DOTENV_PATHS:
            _propagate_env_aliases()
            return env_path
        load_dotenv(dotenv_path=env_path, override=override)
        _propagate_env_aliases()
        _LOADED_DOTENV_PATHS.add(env_path)

    return env_path
