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


def test_interjector_system_prompt_prefers_listener_safe_echoes() -> None:
    prompt = PromptManager.get_prompt(
        "interjector_system",
        max_interjection_words=5,
        min_host_progress_percent=35,
        echo_min_host_progress_percent=5,
        max_host_progress_percent=90,
    )

    assert "prefer `echo_agreement` over a generic `backchannel`" in prompt
    assert "most quotable content words" in prompt
    assert "valid 5%-90% portion" in prompt
    assert "dam broke" in prompt
    assert "back to the music" in prompt
    assert "quoted callback" in prompt
    assert "She handled it perfectly." in prompt
    assert "one weak word" in prompt


def test_interjector_repair_prompt_keeps_echo_style_grounded_in_anchor() -> None:
    prompt = PromptManager.get_prompt(
        "interjector_repair",
        host_turn_text="The host said something vivid.",
        next_turn_text="The next speaker responds.",
        interjection_style="echo_agreement",
        anchor_text="the dam broke and everything came pouring out",
        current_text="the dam broke and everything came pouring out",
        available_overlap_ms=900,
        measured_duration_ms=1200,
        attempt_index=1,
        max_interjection_words=5,
        target_word_budget=3,
    )

    assert "keep the most vivid listener-safe content words from the anchor" in prompt
    assert "generic backchannel" in prompt
    assert 'Good revision: `dam broke`' in prompt
    assert 'Good revision: `back to the music`' in prompt
    assert 'Good revision: `handled everything perfectly`' in prompt
