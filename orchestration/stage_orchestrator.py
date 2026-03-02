"""Use-case orchestration for transcript indexing, drafting, and critique stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from events import event_bus
from orchestrator import Orchestrator
from pipeline_plan import PipelineStagePlan
from summary_output import write_summary_xml_to_workspace
from orchestration.transcript import TranscriptLike, coerce_transcript


@dataclass(slots=True, frozen=True)
class StageOrchestrator:
    """Coordinate start/stop stage behavior for the summarization workflow."""

    orchestrator: Orchestrator
    stage_plan: PipelineStagePlan
    project_root: Path
    min_words: int
    max_words: int
    max_iterations: int

    async def run(
        self,
        *,
        transcript: TranscriptLike,
        retrieval_enabled: bool,
    ) -> None:
        """Execute the configured pipeline stages."""
        transcript_dto = coerce_transcript(transcript)
        full_text = ""
        if retrieval_enabled:
            await self.orchestrator.index_transcript(transcript_dto)
        elif self.stage_plan.requires_retrieval_tools:
            full_text = transcript_dto.to_full_text()

        if self.stage_plan.start_stage == "draft":
            await self._run_draft_stage(full_text=full_text)
            return

        if self.stage_plan.stop_stage == "summarizer":
            await self._run_summarizer_stage(full_text=full_text)
            return

        await self._run_full_loop(full_text=full_text)

    async def _run_draft_stage(self, *, full_text: str) -> None:
        """Run critic-only evaluation for a pre-existing draft file."""
        if self.stage_plan.draft_path is None:
            raise ValueError("pipeline.draft_path must be set when start_stage=draft.")
        if not self.stage_plan.draft_path.exists():
            raise ValueError(f"Draft file not found: {self.stage_plan.draft_path}")

        draft = self.stage_plan.draft_path.read_text(encoding="utf-8").strip()
        if not draft:
            raise ValueError(f"Draft file is empty: {self.stage_plan.draft_path}")

        verdict = await self.orchestrator.run_critic_once(
            draft=draft,
            min_words=self.min_words,
            max_words=self.max_words,
            full_transcript_text=full_text,
        )
        event_bus.publish(
            "status_message",
            f"Critic verdict: status={verdict.status}, word_count={verdict.word_count}",
        )
        event_bus.publish("agent_message", "Critic Feedback", verdict.feedback)
        if verdict.status == "pass":
            summary_path = write_summary_xml_to_workspace(draft, self.project_root)
            event_bus.publish(
                "status_message",
                f"Saved critic-approved draft to {summary_path}",
            )

    async def _run_summarizer_stage(self, *, full_text: str) -> None:
        """Run one summarizer pass and emit the draft result."""
        summary = await self.orchestrator.run_summarizer_once(
            min_words=self.min_words,
            max_words=self.max_words,
            full_transcript_text=full_text,
        )
        event_bus.publish(
            "agent_message",
            "Summarizer",
            f"Single-pass Summary:\n```xml\n{summary}\n```",
        )

    async def _run_full_loop(self, *, full_text: str) -> None:
        """Run the full summarizer-critic loop until convergence or max iterations."""
        result = await self.orchestrator.run_loop(
            min_words=self.min_words,
            max_words=self.max_words,
            max_iterations=self.max_iterations,
            full_transcript_text=full_text,
        )
        if not result:
            return

        event_bus.publish(
            "agent_message",
            "Orchestrator",
            f"Final Summary:\n```xml\n{result}\n```",
        )
        summary_path = write_summary_xml_to_workspace(result, self.project_root)
        event_bus.publish(
            "status_message",
            f"Saved final summary to {summary_path}",
        )
