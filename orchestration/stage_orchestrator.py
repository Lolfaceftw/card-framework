"""Use-case orchestration for stage-2 summary loop and stage-3 voice cloning."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import threading
import time

from audio_pipeline.gpu_heartbeat import VoiceCloneGpuHeartbeatService
from audio_pipeline.voice_clone_orchestrator import VoiceCloneOrchestrator
from audio_pipeline.eta import (
    DynamicEtaTracker,
    StageEtaStrategy,
    StageProgressCallback,
    StageProgressUpdate,
    UnitStageEtaLearner,
    UnitStageEtaStrategy,
    format_eta_seconds,
)
from events import event_bus
from orchestrator import Orchestrator
from pipeline_plan import PipelineStagePlan
from summary_output import write_summary_xml_to_workspace
from orchestration.transcript import Transcript, TranscriptLike, coerce_transcript


@dataclass(slots=True, frozen=True)
class StageOrchestrator:
    """Coordinate stage-1/stage-2/stage-3 execution for the summarization workflow."""

    orchestrator: Orchestrator
    stage_plan: PipelineStagePlan
    project_root: Path
    min_words: int
    max_words: int
    max_iterations: int
    voice_clone_orchestrator: VoiceCloneOrchestrator | None = None
    eta_strategy: StageEtaStrategy | None = None
    eta_update_interval_seconds: float = 10.0
    eta_progress_smoothing: float = 0.25
    eta_overrun_factor: float = 1.15
    eta_headroom_seconds: float = 1.0
    voice_clone_gpu_heartbeat: VoiceCloneGpuHeartbeatService | None = None

    async def run(
        self,
        *,
        transcript: TranscriptLike,
        retrieval_enabled: bool,
    ) -> None:
        """Execute the configured pipeline stages."""
        transcript_dto = coerce_transcript(transcript)
        full_text = ""
        if self.stage_plan.requires_retrieval_tools and retrieval_enabled:
            await self.orchestrator.index_transcript(transcript_dto)
        elif self.stage_plan.requires_retrieval_tools:
            full_text = transcript_dto.to_full_text()

        if self.stage_plan.start_stage == "stage-3":
            self._run_stage_three_only(transcript=transcript_dto)
            return

        await self._run_full_loop(full_text=full_text, transcript=transcript_dto)

    def _run_stage_three_only(self, *, transcript: Transcript) -> None:
        """Run stage-3 voice cloning using a pre-existing summary XML file."""
        if self.stage_plan.final_summary_path is None:
            raise ValueError(
                "pipeline.final_summary_path must be set when pipeline.start_stage=stage-3."
            )
        if not self.stage_plan.final_summary_path.exists():
            raise ValueError(
                f"Final summary file not found: {self.stage_plan.final_summary_path}"
            )

        final_summary = self.stage_plan.final_summary_path.read_text(
            encoding="utf-8"
        ).strip()
        if not final_summary:
            raise ValueError(
                f"Final summary file is empty: {self.stage_plan.final_summary_path}"
            )

        summary_path = write_summary_xml_to_workspace(final_summary, self.project_root)
        event_bus.publish(
            "status_message",
            f"Saved summary to {summary_path}",
        )
        event_bus.publish(
            "status_message",
            (
                "Using existing summary XML for stage-3 voice cloning from "
                f"{self.stage_plan.final_summary_path}"
            ),
        )
        self._run_voice_clone_stage(summary_xml=final_summary, transcript=transcript)

    async def _run_full_loop(self, *, full_text: str, transcript: Transcript) -> None:
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
        self._run_voice_clone_stage(summary_xml=result, transcript=transcript)

    def _run_voice_clone_stage(
        self,
        *,
        summary_xml: str,
        transcript: Transcript,
    ) -> None:
        """Run post-summary voice cloning when configured."""
        if self.voice_clone_orchestrator is None:
            return
        manifest_path_value = str(
            transcript.metadata.get("speaker_samples_manifest_path", "")
        ).strip()
        if not manifest_path_value:
            raise ValueError(
                "Speaker sample manifest is required for voice cloning. "
                "Expected transcript metadata key: speaker_samples_manifest_path."
            )
        manifest_path = Path(manifest_path_value)
        if not manifest_path.is_absolute():
            manifest_path = (self.project_root / manifest_path).resolve()
        expected_turn_count = _count_summary_turns(summary_xml)
        eta_tracker: DynamicEtaTracker | None = None
        if (
            expected_turn_count > 0
            and isinstance(self.eta_strategy, UnitStageEtaStrategy)
        ):
            estimated_total_seconds = self.eta_strategy.estimate_unit_stage_total_seconds(
                stage="voice_clone",
                total_units=expected_turn_count,
            )
            if estimated_total_seconds is not None:
                eta_tracker = DynamicEtaTracker(
                    initial_total_seconds=estimated_total_seconds,
                    progress_smoothing=self.eta_progress_smoothing,
                    overrun_factor=self.eta_overrun_factor,
                    headroom_seconds=self.eta_headroom_seconds,
                )
        if eta_tracker is None:
            event_bus.publish(
                "system_message",
                (
                    "Running voice cloning from speaker samples using manifest "
                    f"{manifest_path}"
                ),
            )
        else:
            event_bus.publish(
                "system_message",
                (
                    "Running voice cloning from speaker samples using manifest "
                    f"{manifest_path} (estimated time left "
                    f"{format_eta_seconds(eta_tracker.initial_total_seconds)})"
                ),
            )

        started_at = time.monotonic()
        tracker_lock = threading.Lock()
        stop_event = threading.Event()
        ticker_thread: threading.Thread | None = None

        progress_callback: StageProgressCallback | None = None
        if eta_tracker is not None:
            def _on_progress(update: StageProgressUpdate) -> None:
                elapsed_seconds = max(0.0, time.monotonic() - started_at)
                with tracker_lock:
                    eta_tracker.observe_progress(
                        elapsed_seconds=elapsed_seconds,
                        update=update,
                    )

            progress_callback = _on_progress

            if self.eta_update_interval_seconds > 0:
                def _ticker() -> None:
                    while not stop_event.wait(self.eta_update_interval_seconds):
                        elapsed_seconds = max(0.0, time.monotonic() - started_at)
                        with tracker_lock:
                            remaining_seconds = eta_tracker.estimate_signed_remaining_seconds(
                                elapsed_seconds=elapsed_seconds
                            )
                        if remaining_seconds >= 0:
                            event_bus.publish(
                                "status_message",
                                (
                                    "Voice clone stage: estimated time left "
                                    f"{format_eta_seconds(remaining_seconds)}"
                                ),
                                inline=True,
                            )
                            continue
                        event_bus.publish(
                            "status_message",
                            (
                                "Voice clone stage: running longer than estimate by "
                                f"{format_eta_seconds(abs(remaining_seconds))}"
                            ),
                            inline=True,
                        )

                ticker_thread = threading.Thread(target=_ticker, daemon=True)
                ticker_thread.start()

        try:
            if self.voice_clone_gpu_heartbeat is not None:
                self.voice_clone_gpu_heartbeat.start(pipeline_root_pid=os.getpid())
            result = self.voice_clone_orchestrator.run(
                summary_xml=summary_xml,
                speaker_samples_manifest_path=manifest_path,
                progress_callback=progress_callback,
            )
        finally:
            stop_event.set()
            if ticker_thread is not None:
                ticker_thread.join(timeout=0.2)
            if self.voice_clone_gpu_heartbeat is not None:
                self.voice_clone_gpu_heartbeat.stop()

        elapsed_seconds = max(0.0, time.monotonic() - started_at)
        if (
            isinstance(self.eta_strategy, UnitStageEtaLearner)
            and len(result.artifacts) > 0
        ):
            try:
                self.eta_strategy.observe_unit_stage_duration(
                    stage="voice_clone",
                    total_units=len(result.artifacts),
                    elapsed_seconds=elapsed_seconds,
                )
            except Exception:
                event_bus.publish(
                    "system_message",
                    "Voice clone stage: ETA learning update skipped.",
                )
        event_bus.publish(
            "status_message",
            (
                f"Voice cloning complete: {len(result.artifacts)} artifacts "
                f"written to {result.output_dir} in {format_eta_seconds(elapsed_seconds)}"
            ),
        )


def _count_summary_turns(summary_xml: str) -> int:
    """Return number of speaker-tagged turns in summary XML."""
    return len(
        re.findall(
            r"<([A-Za-z0-9_.-]+)>.*?</\1>",
            summary_xml,
            flags=re.DOTALL,
        )
    )
