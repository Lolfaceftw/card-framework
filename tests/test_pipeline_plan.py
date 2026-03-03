from pathlib import Path

import pytest

from pipeline_plan import build_pipeline_stage_plan


def test_build_pipeline_stage_plan_defaults_to_full(tmp_path: Path) -> None:
    plan = build_pipeline_stage_plan({}, project_root=tmp_path)

    assert plan.start_stage == "stage-1"
    assert plan.run_audio_stage is True
    assert plan.run_summarizer_stage is True
    assert plan.run_critic_stage is True
    assert plan.requires_retrieval_tools is True
    assert plan.final_summary_path is None


def test_build_pipeline_stage_plan_supports_stage_two_start(tmp_path: Path) -> None:
    plan = build_pipeline_stage_plan(
        {"start_stage": "stage-2"},
        project_root=tmp_path,
    )

    assert plan.start_stage == "stage-2"
    assert plan.run_audio_stage is False
    assert plan.run_summarizer_stage is True
    assert plan.run_critic_stage is True
    assert plan.requires_retrieval_tools is True
    assert plan.final_summary_path is None


def test_build_pipeline_stage_plan_resolves_stage_three_final_summary_path(
    tmp_path: Path,
) -> None:
    plan = build_pipeline_stage_plan(
        {"start_stage": "stage-3", "final_summary_path": "summary.xml"},
        project_root=tmp_path,
    )

    assert plan.start_stage == "stage-3"
    assert plan.final_summary_path == (tmp_path / "summary.xml").resolve()
    assert plan.run_audio_stage is False
    assert plan.run_summarizer_stage is False
    assert plan.run_critic_stage is False
    assert plan.requires_retrieval_tools is False


def test_build_pipeline_stage_plan_validates_stage_three_final_summary_requirements(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        build_pipeline_stage_plan(
            {"start_stage": "stage-3", "final_summary_path": ""},
            project_root=tmp_path,
        )


def test_build_pipeline_stage_plan_rejects_unknown_start_stage(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        build_pipeline_stage_plan(
            {"start_stage": "transcript"},
            project_root=tmp_path,
        )
