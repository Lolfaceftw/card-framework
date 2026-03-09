"""Use-case orchestration for stage-2, stage-3, and stage-4 workflow steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import threading
import time

from card_framework.audio_pipeline.gpu_heartbeat import VoiceCloneGpuHeartbeatService
from card_framework.audio_pipeline.live_draft_voice_clone import LiveDraftVoiceCloneSession
from card_framework.audio_pipeline.interjector import InterjectorOrchestrator, InterjectorRunResult
from card_framework.audio_pipeline.eta import (
    DynamicEtaTracker,
    EtaProfilePersistence,
    StageEtaStrategy,
    StageProgressCallback,
    StageProgressUpdate,
    UnitStageEtaLearner,
    UnitStageEtaHistory,
    UnitStageEtaStrategy,
    format_eta_seconds,
)
from card_framework.audio_pipeline.voice_clone_orchestrator import (
    VoiceCloneOrchestrator,
    VoiceCloneRunResult,
)
from card_framework.shared.events import event_bus
from card_framework.runtime.loop_orchestrator import Orchestrator
from card_framework.runtime.pipeline_plan import PipelineStagePlan
from card_framework.shared.summary_xml import count_summary_turns, parse_summary_xml
from card_framework.shared.summary_output import write_summary_xml_to_workspace
from card_framework.orchestration.transcript import Transcript, TranscriptLike, coerce_transcript


@dataclass(slots=True, frozen=True)
class StageOrchestrator:
    """Coordinate stage-1 through stage-4 execution for the summarization workflow."""

    orchestrator: Orchestrator
    stage_plan: PipelineStagePlan
    project_root: Path
    target_seconds: int
    duration_tolerance_ratio: float
    max_iterations: int
    voice_clone_orchestrator: VoiceCloneOrchestrator | None = None
    interjector_orchestrator: InterjectorOrchestrator | None = None
    speaker_sample_preparer: Callable[[Transcript], Transcript] | None = None
    eta_strategy: StageEtaStrategy | None = None
    eta_profile_path: Path | None = None
    eta_profile_context: dict[str, str] | None = None
    eta_update_interval_seconds: float = 10.0
    eta_progress_smoothing: float = 0.25
    eta_overrun_factor: float = 1.15
    eta_headroom_seconds: float = 1.0
    voice_clone_gpu_heartbeat: VoiceCloneGpuHeartbeatService | None = None
    loop_memory_artifact_path: Path | None = None
    live_draft_audio_enabled: bool = False

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

        if self.stage_plan.start_stage == "stage-4":
            self._run_stage_four_only()
            return
        if self.stage_plan.start_stage == "stage-3":
            self._run_stage_three_only(transcript=transcript_dto)
            return

        await self._run_full_loop(full_text=full_text, transcript=transcript_dto)

    def _run_stage_three_only(self, *, transcript: Transcript) -> None:
        """Run stage-3 voice cloning using a pre-existing summary XML file."""
        if self.voice_clone_orchestrator is None:
            raise ValueError(
                "audio.voice_clone.enabled must be true when "
                "pipeline.start_stage=stage-3."
            )
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
        if self.live_draft_audio_enabled:
            voice_clone_result = self._run_live_batch_voice_clone_stage(
                summary_xml=final_summary,
                transcript=transcript,
            )
        else:
            voice_clone_result = self._run_voice_clone_stage(
                summary_xml=final_summary,
                transcript=transcript,
            )
        self._run_interjector_stage(
            summary_xml=final_summary,
            voice_clone_result=voice_clone_result,
        )

    def _run_stage_four_only(self) -> None:
        """Run stage-4 interjection mixing from existing stage-3 artifacts."""
        if self.interjector_orchestrator is None:
            raise ValueError(
                "audio.interjector.enabled must be true when "
                "pipeline.start_stage=stage-4."
            )
        if self.stage_plan.final_summary_path is None:
            raise ValueError(
                "pipeline.final_summary_path must be set when "
                "pipeline.start_stage=stage-4."
            )
        if self.stage_plan.voice_clone_manifest_path is None:
            raise ValueError(
                "pipeline.voice_clone_manifest_path must be set when "
                "pipeline.start_stage=stage-4."
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
        event_bus.publish("status_message", f"Saved summary to {summary_path}")
        event_bus.publish(
            "status_message",
            (
                "Using existing summary XML and voice-clone manifest for stage-4 "
                f"interjection from {self.stage_plan.final_summary_path}"
            ),
        )
        self._run_interjector_stage(
            summary_xml=final_summary,
            voice_clone_manifest_path=self.stage_plan.voice_clone_manifest_path,
        )

    async def _run_full_loop(self, *, full_text: str, transcript: Transcript) -> None:
        """Run the full summarizer-critic loop until convergence or max iterations."""
        prepared_transcript = transcript
        speaker_samples_manifest_path: Path | None = None
        draft_audio_state_path: Path | None = None
        if self.live_draft_audio_enabled and self.voice_clone_orchestrator is not None:
            prepared_transcript, speaker_samples_manifest_path = (
                self._prepare_transcript_for_live_draft_voice_clone(transcript=transcript)
            )
            draft_audio_state_path = self._build_draft_audio_state_path(
                transcript=prepared_transcript
            )

        run_loop_with_diagnostics = getattr(
            self.orchestrator,
            "run_loop_with_diagnostics",
            None,
        )
        if callable(run_loop_with_diagnostics):
            diagnostics = await run_loop_with_diagnostics(
                target_seconds=self.target_seconds,
                duration_tolerance_ratio=self.duration_tolerance_ratio,
                max_iterations=self.max_iterations,
                full_transcript_text=full_text,
                loop_memory_artifact_path=self.loop_memory_artifact_path,
                loop_memory_context=self._build_loop_memory_context(
                    transcript=prepared_transcript
                ),
                speaker_samples_manifest_path=speaker_samples_manifest_path,
                draft_audio_state_path=draft_audio_state_path,
            )
        else:
            result = await self.orchestrator.run_loop(
                target_seconds=self.target_seconds,
                duration_tolerance_ratio=self.duration_tolerance_ratio,
                max_iterations=self.max_iterations,
                full_transcript_text=full_text,
                loop_memory_artifact_path=self.loop_memory_artifact_path,
                loop_memory_context=self._build_loop_memory_context(
                    transcript=prepared_transcript
                ),
                speaker_samples_manifest_path=speaker_samples_manifest_path,
                draft_audio_state_path=draft_audio_state_path,
            )
            diagnostics = {"converged": bool(result), "draft": result}
        result = diagnostics["draft"] if diagnostics["converged"] else ""
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
        if (
            self.live_draft_audio_enabled
            and self.voice_clone_orchestrator is not None
            and draft_audio_state_path is not None
            and speaker_samples_manifest_path is not None
        ):
            event_bus.publish(
                "system_message",
                (
                    "Finalizing live draft voice-clone manifest from rendered turn "
                    f"audio in {draft_audio_state_path}"
                ),
            )
            live_session = LiveDraftVoiceCloneSession.from_orchestrator(
                orchestrator=self.voice_clone_orchestrator,
                state_path=draft_audio_state_path,
                speaker_samples_manifest_path=speaker_samples_manifest_path,
            )
            voice_clone_result = live_session.finalize()
            event_bus.publish(
                "status_message",
                (
                    f"Voice cloning complete: {len(voice_clone_result.artifacts)} artifacts "
                    f"written to {voice_clone_result.output_dir}"
                ),
            )
        else:
            voice_clone_result = self._run_voice_clone_stage(
                summary_xml=result,
                transcript=prepared_transcript,
            )
        self._run_interjector_stage(
            summary_xml=result,
            voice_clone_result=voice_clone_result,
        )

    def _build_loop_memory_context(self, *, transcript: Transcript) -> dict[str, str]:
        """Build a stable persistence scope for summarizer loop memory artifacts."""
        transcript_hash = hashlib.sha256(
            transcript.to_full_text().encode("utf-8")
        ).hexdigest()
        return {
            "transcript_sha256": transcript_hash,
            "target_seconds": str(self.target_seconds),
            "duration_tolerance_ratio": f"{self.duration_tolerance_ratio:.6f}",
        }

    def _build_draft_audio_state_path(self, *, transcript: Transcript) -> Path:
        """Build the persisted live-draft state path for one transcript/target pair."""
        if self.voice_clone_orchestrator is None:
            raise ValueError("voice_clone_orchestrator is required for live drafting.")
        transcript_hash = hashlib.sha256(
            transcript.to_full_text().encode("utf-8")
        ).hexdigest()[:12]
        filename = f"live_draft_{transcript_hash}_{self.target_seconds}s.state.json"
        return (self.voice_clone_orchestrator.output_dir / filename).resolve()

    def _prepare_transcript_for_live_draft_voice_clone(
        self,
        *,
        transcript: Transcript,
    ) -> tuple[Transcript, Path]:
        """Ensure speaker samples exist before live drafting begins."""
        prepared_transcript = transcript
        manifest_path_value = str(
            prepared_transcript.metadata.get("speaker_samples_manifest_path", "")
        ).strip()
        if not manifest_path_value and self.speaker_sample_preparer is not None:
            event_bus.publish(
                "system_message",
                "Preparing speaker samples before live drafting...",
            )
            prepared_transcript = self.speaker_sample_preparer(prepared_transcript)
            manifest_path_value = str(
                prepared_transcript.metadata.get("speaker_samples_manifest_path", "")
            ).strip()
        if not manifest_path_value:
            raise ValueError(
                "Speaker sample manifest is required for live drafting. "
                "Expected transcript metadata key: speaker_samples_manifest_path."
            )
        manifest_path = Path(manifest_path_value)
        if not manifest_path.is_absolute():
            manifest_path = (self.project_root / manifest_path).resolve()
        return prepared_transcript, manifest_path

    def _run_live_batch_voice_clone_stage(
        self,
        *,
        summary_xml: str,
        transcript: Transcript,
    ) -> VoiceCloneRunResult:
        """Render stage-3 audio through the live-draft session in one batch."""
        if self.voice_clone_orchestrator is None:
            raise ValueError("voice_clone_orchestrator is required for stage-3 audio.")
        prepared_transcript, manifest_path = (
            self._prepare_transcript_for_live_draft_voice_clone(transcript=transcript)
        )
        state_path = self._build_draft_audio_state_path(transcript=prepared_transcript)
        live_session = LiveDraftVoiceCloneSession.from_orchestrator(
            orchestrator=self.voice_clone_orchestrator,
            state_path=state_path,
            speaker_samples_manifest_path=manifest_path,
        )
        snapshot = [
            {
                "line": index,
                "turn_id": f"stage3-turn-{index:03d}",
                "speaker_id": turn.speaker,
                "content": turn.text,
                "emo_preset": turn.emo_preset,
            }
            for index, turn in enumerate(parse_summary_xml(summary_xml), start=1)
        ]
        if not snapshot:
            raise ValueError("Summary XML is empty; cannot run stage-3 voice cloning.")

        event_bus.publish(
            "system_message",
            (
                "Running stage-3 voice cloning through the live draft renderer using "
                f"manifest {manifest_path}"
            ),
        )

        started_at = time.monotonic()
        try:
            live_session.clear()
            if self.voice_clone_gpu_heartbeat is not None:
                self.voice_clone_gpu_heartbeat.start(pipeline_root_pid=os.getpid())
            live_session.bootstrap_from_snapshot(snapshot)
            result = live_session.finalize()
        finally:
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
                self._save_eta_profile(
                    failure_message="Voice clone stage: ETA profile save failed.",
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
        return result

    def _run_voice_clone_stage(
        self,
        *,
        summary_xml: str,
        transcript: Transcript,
    ) -> VoiceCloneRunResult | None:
        """Run post-summary voice cloning when configured."""
        if self.voice_clone_orchestrator is None:
            return None
        prepared_transcript = transcript
        manifest_path_value = str(
            prepared_transcript.metadata.get("speaker_samples_manifest_path", "")
        ).strip()
        if not manifest_path_value and self.speaker_sample_preparer is not None:
            event_bus.publish(
                "system_message",
                "Preparing deferred speaker samples for voice clone stage...",
            )
            prepared_transcript = self.speaker_sample_preparer(prepared_transcript)
            manifest_path_value = str(
                prepared_transcript.metadata.get("speaker_samples_manifest_path", "")
            ).strip()
        if not manifest_path_value:
            raise ValueError(
                "Speaker sample manifest is required for voice cloning. "
                "Expected transcript metadata key: speaker_samples_manifest_path."
            )
        manifest_path = Path(manifest_path_value)
        if not manifest_path.is_absolute():
            manifest_path = (self.project_root / manifest_path).resolve()
        expected_turn_count = count_summary_turns(summary_xml)
        eta_tracker: DynamicEtaTracker | None = None
        if (
            expected_turn_count > 0
            and isinstance(self.eta_strategy, UnitStageEtaStrategy)
            and (
                not isinstance(self.eta_strategy, UnitStageEtaHistory)
                or self.eta_strategy.has_unit_stage_history(stage="voice_clone")
            )
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
                self._save_eta_profile(
                    failure_message="Voice clone stage: ETA profile save failed.",
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
        return result

    def _save_eta_profile(self, *, failure_message: str) -> None:
        """Persist learned ETA state when the configured strategy supports it."""
        if self.eta_profile_path is None or not isinstance(
            self.eta_strategy,
            EtaProfilePersistence,
        ):
            return
        try:
            self.eta_strategy.save_profile(
                self.eta_profile_path,
                context=self.eta_profile_context or {},
            )
        except Exception:
            event_bus.publish("system_message", failure_message)

    def _run_interjector_stage(
        self,
        *,
        summary_xml: str,
        voice_clone_result: VoiceCloneRunResult | None = None,
        voice_clone_manifest_path: Path | None = None,
    ) -> InterjectorRunResult | None:
        """Run Stage-4 interjection mixing when configured."""
        if self.interjector_orchestrator is None:
            return None

        manifest_path = voice_clone_manifest_path
        if manifest_path is None and voice_clone_result is not None:
            manifest_path = voice_clone_result.manifest_path
        if manifest_path is None:
            raise ValueError(
                "Voice clone manifest is required for stage-4 interjection. "
                "Run stage-3 voice cloning first or set "
                "pipeline.voice_clone_manifest_path when pipeline.start_stage=stage-4."
            )
        if not manifest_path.is_absolute():
            manifest_path = (self.project_root / manifest_path).resolve()

        event_bus.publish(
            "system_message",
            f"Running stage-4 interjection from voice-clone manifest {manifest_path}",
        )
        result = self.interjector_orchestrator.run(
            summary_xml=summary_xml,
            voice_clone_manifest_path=manifest_path,
        )
        event_bus.publish(
            "status_message",
            (
                f"Stage-4 interjection complete: {len(result.artifacts)} artifacts "
                f"written to {result.output_dir}"
            ),
        )
        return result

