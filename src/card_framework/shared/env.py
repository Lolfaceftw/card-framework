"""Load repository-local environment files when present."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

from card_framework.shared.paths import REPO_ROOT

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    def _load_dotenv(*, dotenv_path: Path, override: bool = False) -> bool:
        """Gracefully skip dotenv loading when python-dotenv is unavailable."""
        del dotenv_path, override
        return False

_LOADED_DOTENV_PATHS: set[Path] = set()
_LOAD_LOCK = Lock()
_ENV_ALIAS_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "HF_TOKEN",
        "HF_ACCESS_TOKEN",
        "HF_API_TOKEN",
        "HUGGINGFACE_HUB_TOKEN",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    ),
    ("VLLM_URL", "CARD_FRAMEWORK_VLLM_URL", "VLLM_BASE_URL"),
    ("VLLM_API_KEY", "CARD_FRAMEWORK_VLLM_API_KEY"),
    ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    ("ZAI_API_KEY", "GLM_API_KEY"),
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
    repo_root: Path = REPO_ROOT, *, override: bool = False, search_cwd: bool = False
) -> Path | None:
    """Load project-local ``.env`` files once when they exist.

    Args:
        repo_root: Repository root that may contain a ``.env`` file.
        override: Whether values from the file should replace existing process
            environment variables.
        search_cwd: Whether to also load ``Path.cwd() / ".env"`` after the
            repo-root file when it exists.

    Returns:
        The first resolved ``.env`` path that was loaded or had already been
        loaded, otherwise ``None``.
    """
    candidate_paths: list[Path] = [(repo_root / ".env").resolve()]
    if search_cwd:
        cwd_env_path = (Path.cwd() / ".env").resolve()
        if cwd_env_path not in candidate_paths:
            candidate_paths.append(cwd_env_path)

    loaded_path: Path | None = None
    with _LOAD_LOCK:
        for env_path in candidate_paths:
            if not env_path.is_file():
                continue
            if env_path not in _LOADED_DOTENV_PATHS:
                _load_dotenv(dotenv_path=env_path, override=override)
                _LOADED_DOTENV_PATHS.add(env_path)
            if loaded_path is None:
                loaded_path = env_path
        _propagate_env_aliases()

    return loaded_path
