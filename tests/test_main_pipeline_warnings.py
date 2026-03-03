"""Tests for runtime pipeline warning message helpers in main module."""

from __future__ import annotations

import pytest

from pipeline_plan import PipelineStagePlan

pytest.importorskip("a2a")
import main as app_main


def test_build_critic_skip_warning_messages_for_summarizer_stop() -> None:
    warnings = app_main._build_critic_skip_warning_messages(
        stage_plan=PipelineStagePlan(start_stage="transcript", stop_stage="summarizer"),
        voice_clone_enabled=True,
    )

    assert len(warnings) == 3
    assert "disables Critic" in warnings[0]
    assert "pipeline.stop_stage=critic" in warnings[1]
    assert "directly to voice clone" in warnings[2]


def test_build_critic_skip_warning_messages_empty_for_critic_stop() -> None:
    warnings = app_main._build_critic_skip_warning_messages(
        stage_plan=PipelineStagePlan(start_stage="transcript", stop_stage="critic"),
        voice_clone_enabled=True,
    )

    assert warnings == ()


def test_build_critic_skip_warning_messages_empty_for_draft_shortcut() -> None:
    warnings = app_main._build_critic_skip_warning_messages(
        stage_plan=PipelineStagePlan(start_stage="draft", stop_stage="summarizer"),
        voice_clone_enabled=True,
    )

    assert warnings == ()
