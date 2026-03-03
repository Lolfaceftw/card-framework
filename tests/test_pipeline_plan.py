from pathlib import Path

import pytest

from pipeline_plan import build_pipeline_stage_plan


def test_build_pipeline_stage_plan_defaults_to_full(tmp_path: Path) -> None:
    plan = build_pipeline_stage_plan({}, project_root=tmp_path)

    assert plan.start_stage == "audio"
    assert plan.stop_stage == "critic"
    assert plan.run_audio_stage is True
    assert plan.run_summarizer_stage is True
    assert plan.run_critic_stage is True


def test_build_pipeline_stage_plan_supports_summarizer_only(tmp_path: Path) -> None:
    plan = build_pipeline_stage_plan(
        {"start_stage": "transcript", "stop_stage": "summarizer"},
        project_root=tmp_path,
    )

    assert plan.start_stage == "transcript"
    assert plan.stop_stage == "summarizer"
    assert plan.run_audio_stage is False
    assert plan.run_summarizer_stage is True
    assert plan.run_critic_stage is False


def test_build_pipeline_stage_plan_resolves_draft_path(tmp_path: Path) -> None:
    plan = build_pipeline_stage_plan(
        {"start_stage": "draft", "stop_stage": "critic", "draft_path": "draft.md"},
        project_root=tmp_path,
    )

    assert plan.draft_path == (tmp_path / "draft.md").resolve()
    assert plan.run_summarizer_stage is False
    assert plan.run_critic_stage is True


def test_build_pipeline_stage_plan_validates_draft_requirements(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_pipeline_stage_plan(
            {"start_stage": "draft", "stop_stage": "critic", "draft_path": ""},
            project_root=tmp_path,
        )


def test_build_pipeline_stage_plan_supports_draft_to_voiceclone_mode(
    tmp_path: Path,
) -> None:
    plan = build_pipeline_stage_plan(
        {"start_stage": "draft", "stop_stage": "summarizer", "draft_path": "summary.xml"},
        project_root=tmp_path,
    )

    assert plan.start_stage == "draft"
    assert plan.stop_stage == "summarizer"
    assert plan.draft_path == (tmp_path / "summary.xml").resolve()
    assert plan.run_summarizer_stage is False
    assert plan.run_critic_stage is False
