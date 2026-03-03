"""Typed stage-plan configuration for pipeline execution modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, TypeAlias, cast

PipelineStartStage: TypeAlias = Literal["stage-1", "stage-2", "stage-3"]


@dataclass(slots=True, frozen=True)
class PipelineStagePlan:
    """
    Immutable execution plan derived from user-facing pipeline config.

    Attributes:
        start_stage: Stage where the pipeline starts.
        final_summary_path: Optional summary XML path used by ``start_stage='stage-3'``.
    """

    start_stage: PipelineStartStage
    final_summary_path: Path | None = None

    @property
    def run_audio_stage(self) -> bool:
        """Return whether stage-1 (audio separation/transcription/diarization) should run."""
        return self.start_stage == "stage-1"

    @property
    def run_summarizer_stage(self) -> bool:
        """Return whether stage-2 summarizer stage should run."""
        return self.start_stage in {"stage-1", "stage-2"}

    @property
    def run_critic_stage(self) -> bool:
        """Return whether stage-2 critic stage should run."""
        return self.start_stage in {"stage-1", "stage-2"}

    @property
    def requires_retrieval_tools(self) -> bool:
        """Return whether retrieval/indexing tools may be used."""
        return self.start_stage in {"stage-1", "stage-2"}


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
    start_stage = _parse_start_stage(pipeline_cfg.get("start_stage", "stage-1"))
    final_summary_path_value = str(pipeline_cfg.get("final_summary_path", "")).strip()

    if start_stage == "stage-3":
        if not final_summary_path_value:
            raise ValueError(
                "pipeline.final_summary_path is required when pipeline.start_stage=stage-3."
            )
        return PipelineStagePlan(
            start_stage=start_stage,
            final_summary_path=_resolve_path(
                final_summary_path_value,
                base_dir=project_root,
            ),
        )

    return PipelineStagePlan(start_stage=start_stage, final_summary_path=None)


def _parse_start_stage(value: Any) -> PipelineStartStage:
    """Parse pipeline start stage with explicit error message."""
    normalized = str(value).strip().lower()
    if normalized not in {"stage-1", "stage-2", "stage-3"}:
        raise ValueError(
            "pipeline.start_stage must be one of: stage-1, stage-2, stage-3"
        )
    return cast(PipelineStartStage, normalized)


def _resolve_path(path_value: str, *, base_dir: Path) -> Path:
    """Resolve relative path against project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()
