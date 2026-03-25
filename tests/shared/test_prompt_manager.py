from __future__ import annotations

import pytest

try:
    import jinja2 as _jinja2
except Exception:
    _jinja2 = None

if _jinja2 is None or getattr(_jinja2, "__spec__", None) is None:
    pytest.skip(
        "jinja2 is required for prompt manager rendering tests.",
        allow_module_level=True,
    )

from card_framework.shared.prompt_manager import PromptManager


def setup_function(function: object) -> None:
    """Reset the singleton between tests for isolation."""
    del function
    PromptManager._instance = None
    PromptManager._env = None


def test_get_prompt_recreates_environment_when_singleton_survives() -> None:
    """Recover if callers clear the Jinja environment without resetting the instance."""
    PromptManager()
    assert PromptManager._instance is not None
    assert PromptManager._env is not None

    PromptManager._env = None

    prompt = PromptManager.get_prompt(
        "critic_system",
        target_seconds=30,
        target_minutes=0.5,
        duration_tolerance_ratio=0.05,
        duration_tolerance_percent=5.0,
        min_seconds=28.5,
        max_seconds=31.5,
        is_embedding_enabled=False,
    )

    assert "SHORT-FORM COVERAGE AND PROGRESSION" in prompt
    assert PromptManager._env is not None
