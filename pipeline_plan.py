"""Typed stage-plan configuration for pipeline execution modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias, cast

PipelineStartStage: TypeAlias = Literal["audio", "transcript", "draft"]
PipelineStopStage: TypeAlias = Literal["summarizer", "critic"]


@dataclass(slots=True, frozen=True)
class PipelineStagePlan:
    """
    Immutable execution plan derived from user-facing pipeline config.

    Attributes:
        start_stage: Stage where the pipeline starts.
        stop_stage: Stage where the pipeline stops.
        draft_path: Optional draft path used by ``start_stage='draft'``.
    """

    start_stage: PipelineStartStage
    stop_stage: PipelineStopStage
    draft_path: Path | None = None

    @property
    def run_audio_stage(self) -> bool:
        """Return whether stage 1 (audio-to-script) should execute."""
        return self.start_stage == "audio"

    @property
    def run_summarizer_stage(self) -> bool:
        """Return whether summarizer service is required."""
        return self.start_stage in {"audio", "transcript"}

    @property
    def run_critic_stage(self) -> bool:
        """Return whether critic service is required."""
        return self.stop_stage == "critic"

    @property
    def requires_retrieval_tools(self) -> bool:
        """
        Return whether retrieval/indexing tools may be used.

        Retrieval is relevant for summarizer and critic stages.
        """
        return self.run_summarizer_stage or self.run_critic_stage


def build_pipeline_stage_plan(
    pipeline_cfg: Mapping[str, Any],
    *,
    project_root: Path,
) -> PipelineStagePlan:
    """
    Build validated stage plan from config.

    Args:
        pipeline_cfg: ``pipeline`` configuration mapping.
        project_root: Repository root used for relative path resolution.

    Returns:
        Validated immutable stage plan.

    Raises:
        ValueError: If configuration combination is invalid.
    """
    start_stage = _parse_start_stage(pipeline_cfg.get("start_stage", "audio"))
    stop_stage = _parse_stop_stage(pipeline_cfg.get("stop_stage", "critic"))
    draft_path_value = str(pipeline_cfg.get("draft_path", "")).strip()

    if start_stage == "draft":
        if stop_stage not in {"summarizer", "critic"}:
            raise ValueError(
                "pipeline.start_stage=draft requires pipeline.stop_stage in {summarizer, critic}."
            )
        if not draft_path_value:
            raise ValueError(
                "pipeline.draft_path is required when pipeline.start_stage=draft."
            )
        return PipelineStagePlan(
            start_stage=start_stage,
            stop_stage=stop_stage,
            draft_path=_resolve_path(draft_path_value, base_dir=project_root),
        )

    return PipelineStagePlan(
        start_stage=start_stage,
        stop_stage=stop_stage,
        draft_path=None,
    )


def _parse_start_stage(value: Any) -> PipelineStartStage:
    """Parse pipeline start stage with explicit error message."""
    normalized = str(value).strip().lower()
    if normalized not in {"audio", "transcript", "draft"}:
        raise ValueError(
            "pipeline.start_stage must be one of: audio, transcript, draft"
        )
    return cast(PipelineStartStage, normalized)


def _parse_stop_stage(value: Any) -> PipelineStopStage:
    """Parse pipeline stop stage with explicit error message."""
    normalized = str(value).strip().lower()
    if normalized not in {"summarizer", "critic"}:
        raise ValueError("pipeline.stop_stage must be one of: summarizer, critic")
    return cast(PipelineStopStage, normalized)


def _resolve_path(path_value: str, *, base_dir: Path) -> Path:
    """Resolve relative path against project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
