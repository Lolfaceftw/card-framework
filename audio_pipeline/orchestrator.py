"""Use-case orchestrator for audio-to-transcript generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import threading
import time
from typing import Any, Callable, TypeVar

from audio_pipeline.alignment import align_segments_with_speakers
from audio_pipeline.contracts import (
    SourceSeparator,
    SpeakerDiarizer,
    SpeechTranscriber,
    TranscriptPayload,
)
from audio_pipeline.eta import (
    AudioStageName,
    DynamicEtaTracker,
    EtaProfilePersistence,
    StageProgressCallback,
    StageProgressUpdate,
    StageEtaLearner,
    StageEtaStrategy,
    default_stage_eta_strategy,
    format_eta_seconds,
)
from audio_pipeline.io import build_transcript_payload, write_transcript_atomic
from audio_pipeline.runtime import probe_audio_duration_ms, utc_now_iso
from events import event_bus

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class AudioStageOutput:
    """Output object for completed audio transcription stage."""

    transcript_path: Path
    transcript_payload: TranscriptPayload
    segment_count: int
    warnings: list[str]


@dataclass(slots=True)
class AudioToScriptOrchestrator:
    """
    Coordinate separation, transcription, diarization, and transcript emission.

    This class is the use-case layer. Concrete model/tool adapters are injected
    through protocol ports for low coupling and testability.
    """

    separator: SourceSeparator
    transcriber: SpeechTranscriber
    diarizer: SpeakerDiarizer
    merge_gap_ms: int = 800
    default_speaker: str = "SPEAKER_00"
    eta_strategy: StageEtaStrategy = field(default_factory=default_stage_eta_strategy)
    eta_update_interval_seconds: float = 10.0
    eta_progress_smoothing: float = 0.25
    eta_overrun_factor: float = 1.15
    eta_headroom_seconds: float = 1.0
    eta_profile_filename: str = "eta_profile.json"

    def run(
        self,
        *,
        input_audio_path: Path,
        output_transcript_path: Path,
        work_dir: Path,
        device: str,
        metadata_overrides: dict[str, Any] | None = None,
    ) -> AudioStageOutput:
        """
        Execute the audio-to-script workflow end-to-end.

        Args:
            input_audio_path: Source audio file.
            output_transcript_path: Destination transcript JSON.
            work_dir: Scratch/artifact directory for intermediate outputs.
            device: Runtime device.
            metadata_overrides: Optional extra metadata fields.

        Returns:
            Structured output including transcript path and payload.
        """
        warnings: list[str] = []
        eta_profile_context = self._build_eta_profile_context(device=device)
        eta_profile_path = work_dir / self.eta_profile_filename
        self._load_eta_profile(
            profile_path=eta_profile_path,
            context=eta_profile_context,
        )
        source_duration_ms = probe_audio_duration_ms(input_audio_path)
        if source_duration_ms is None:
            event_bus.publish(
                "system_message",
                "Audio stage: duration probe unavailable; ETA updates may be reduced.",
            )
        try:
            vocals_path = self._run_stage_with_eta(
                stage="separation",
                description=f"separating sources from {input_audio_path}",
                audio_duration_ms=source_duration_ms,
                device=device,
                operation=lambda progress_callback: self.separator.separate_vocals(
                    input_audio_path=input_audio_path,
                    output_dir=work_dir / "separation",
                    device=device,
                    progress_callback=progress_callback,
                ),
            )

            vocals_duration_ms = probe_audio_duration_ms(vocals_path) or source_duration_ms
            asr_segments = self._run_stage_with_eta(
                stage="transcription",
                description=f"transcribing vocals {vocals_path}",
                audio_duration_ms=vocals_duration_ms,
                device=device,
                operation=lambda progress_callback: self.transcriber.transcribe(
                    vocals_path,
                    device=device,
                    progress_callback=progress_callback,
                ),
            )

            diarization_turns = self._run_stage_with_eta(
                stage="diarization",
                description="running NeMo diarization",
                audio_duration_ms=vocals_duration_ms,
                device=device,
                operation=lambda progress_callback: self.diarizer.diarize(
                    audio_path=vocals_path,
                    output_dir=work_dir / "diarization",
                    device=device,
                    progress_callback=progress_callback,
                ),
            )
            if len(diarization_turns) <= 1:
                warnings.append(
                    "Diarization produced a single speaker timeline; verify NeMo setup for multi-speaker audio."
                )

            aligned_segments = align_segments_with_speakers(
                asr_segments=asr_segments,
                diarization_turns=diarization_turns,
                default_speaker=self.default_speaker,
                merge_gap_ms=self.merge_gap_ms,
            )

            metadata = {
                "source_audio_path": str(input_audio_path),
                "vocals_audio_path": str(vocals_path),
                "device": device,
                "generated_at_utc": utc_now_iso(),
                "warnings": warnings,
            }
            if metadata_overrides:
                metadata.update(metadata_overrides)

            payload = build_transcript_payload(
                segments=aligned_segments,
                metadata=metadata,
            )
            write_transcript_atomic(payload, output_transcript_path)

            event_bus.publish(
                "status_message",
                f"Audio stage complete: {len(aligned_segments)} segments written to {output_transcript_path}",
            )
            return AudioStageOutput(
                transcript_path=output_transcript_path,
                transcript_payload=payload,
                segment_count=len(aligned_segments),
                warnings=warnings,
            )
        finally:
            self._save_eta_profile(
                profile_path=eta_profile_path,
                context=eta_profile_context,
            )

    def _run_stage_with_eta(
        self,
        *,
        stage: AudioStageName,
        description: str,
        audio_duration_ms: int | None,
        device: str,
        operation: Callable[[StageProgressCallback | None], T],
    ) -> T:
        """
        Execute one stage while publishing periodic ETA updates.

        Args:
            stage: Stage key (`separation`, `transcription`, or `diarization`).
            description: Human-readable stage message body.
            audio_duration_ms: Audio duration to estimate from.
            device: Runtime device.
            operation: Callable that executes the stage.

        Returns:
            Stage result returned by ``operation``.

        Raises:
            Exception: Re-raises any exception from ``operation``.
        """
        estimated_total_seconds = self._estimate_stage_total_seconds(
            stage=stage,
            audio_duration_ms=audio_duration_ms,
            device=device,
        )
        started_at = time.monotonic()
        eta_tracker = self._build_stage_eta_tracker(
            estimated_total_seconds=estimated_total_seconds,
            audio_duration_ms=audio_duration_ms,
        )
        tracker_lock = threading.Lock()

        def _progress_callback(update: StageProgressUpdate) -> None:
            if eta_tracker is None:
                return
            elapsed_seconds = max(0.0, time.monotonic() - started_at)
            with tracker_lock:
                try:
                    eta_tracker.observe_progress(
                        elapsed_seconds=elapsed_seconds,
                        update=update,
                    )
                except Exception:
                    return

        if eta_tracker is None:
            event_bus.publish("system_message", f"Audio stage: {description}")
        else:
            initial_remaining_seconds = eta_tracker.initial_total_seconds
            event_bus.publish(
                "system_message",
                (
                    f"Audio stage: {description} "
                    f"(estimated time left {format_eta_seconds(initial_remaining_seconds)})"
                ),
            )

        stop_event = threading.Event()
        ticker_thread: threading.Thread | None = None
        if (
            eta_tracker is not None
            and self.eta_update_interval_seconds > 0
        ):
            ticker_thread = threading.Thread(
                target=self._publish_eta_updates,
                kwargs={
                    "stage": stage,
                    "eta_tracker": eta_tracker,
                    "tracker_lock": tracker_lock,
                    "started_at": started_at,
                    "stop_event": stop_event,
                },
                daemon=True,
            )
            ticker_thread.start()

        try:
            result = operation(_progress_callback if eta_tracker is not None else None)
        except Exception:
            self._stop_eta_updates(stop_event=stop_event, ticker_thread=ticker_thread)
            elapsed_seconds = time.monotonic() - started_at
            event_bus.publish(
                "error_message",
                f"Audio stage: {stage} failed after {format_eta_seconds(elapsed_seconds)}",
            )
            raise

        self._stop_eta_updates(stop_event=stop_event, ticker_thread=ticker_thread)
        elapsed_seconds = time.monotonic() - started_at
        self._observe_stage_duration(
            stage=stage,
            audio_duration_ms=audio_duration_ms,
            elapsed_seconds=elapsed_seconds,
            device=device,
        )
        event_bus.publish(
            "status_message",
            f"Audio stage: {stage} finished in {format_eta_seconds(elapsed_seconds)}",
        )
        return result

    def _estimate_stage_total_seconds(
        self,
        *,
        stage: AudioStageName,
        audio_duration_ms: int | None,
        device: str,
    ) -> float | None:
        """Estimate stage duration from strategy and known audio length."""
        if audio_duration_ms is None:
            return None
        return self.eta_strategy.estimate_total_seconds(
            stage=stage,
            audio_duration_ms=audio_duration_ms,
            device=device,
        )

    def _publish_eta_updates(
        self,
        *,
        stage: AudioStageName,
        eta_tracker: DynamicEtaTracker,
        tracker_lock: threading.Lock,
        started_at: float,
        stop_event: threading.Event,
    ) -> None:
        """Emit periodic ETA updates until stage completion."""
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
                        f"Audio stage: {stage} estimated time left "
                        f"{format_eta_seconds(remaining_seconds)}"
                    ),
                    inline=True,
                )
                continue
            event_bus.publish(
                "status_message",
                (
                    f"Audio stage: {stage} running longer than estimate by "
                    f"{format_eta_seconds(abs(remaining_seconds))}"
                ),
                inline=True,
            )

    def _build_stage_eta_tracker(
        self,
        *,
        estimated_total_seconds: float | None,
        audio_duration_ms: int | None,
    ) -> DynamicEtaTracker | None:
        """Build dynamic ETA tracker for one stage run when baseline estimate exists."""
        if estimated_total_seconds is None:
            return None
        total_audio_ms = audio_duration_ms if audio_duration_ms and audio_duration_ms > 0 else None
        return DynamicEtaTracker(
            initial_total_seconds=estimated_total_seconds,
            progress_smoothing=self.eta_progress_smoothing,
            overrun_factor=self.eta_overrun_factor,
            headroom_seconds=self.eta_headroom_seconds,
            total_audio_ms=total_audio_ms,
        )

    def _stop_eta_updates(
        self,
        *,
        stop_event: threading.Event,
        ticker_thread: threading.Thread | None,
    ) -> None:
        """Stop ETA ticker thread safely."""
        stop_event.set()
        if ticker_thread is not None:
            ticker_thread.join(timeout=0.2)

    def _observe_stage_duration(
        self,
        *,
        stage: AudioStageName,
        audio_duration_ms: int | None,
        elapsed_seconds: float,
        device: str,
    ) -> None:
        """Feed observed stage duration back into adaptive ETA strategy."""
        if audio_duration_ms is None or not isinstance(self.eta_strategy, StageEtaLearner):
            return
        try:
            self.eta_strategy.observe_stage_duration(
                stage=stage,
                audio_duration_ms=audio_duration_ms,
                elapsed_seconds=elapsed_seconds,
                device=device,
            )
        except Exception:
            event_bus.publish(
                "system_message",
                f"Audio stage: ETA learning update skipped for {stage}.",
            )

    def _load_eta_profile(
        self,
        *,
        profile_path: Path,
        context: dict[str, str],
    ) -> None:
        """Load persisted ETA profile when supported by the strategy."""
        if not isinstance(self.eta_strategy, EtaProfilePersistence):
            return
        try:
            self.eta_strategy.load_profile(profile_path, context=context)
        except Exception:
            event_bus.publish(
                "system_message",
                "Audio stage: ETA profile load failed; using defaults.",
            )

    def _save_eta_profile(
        self,
        *,
        profile_path: Path,
        context: dict[str, str],
    ) -> None:
        """Persist ETA profile when supported by the strategy."""
        if not isinstance(self.eta_strategy, EtaProfilePersistence):
            return
        try:
            self.eta_strategy.save_profile(profile_path, context=context)
        except Exception:
            event_bus.publish(
                "system_message",
                "Audio stage: ETA profile save failed.",
            )

    def _build_eta_profile_context(self, *, device: str) -> dict[str, str]:
        """Build context key used to scope persisted ETA profiles."""
        return {
            "device": device.strip().lower(),
            "separator": type(self.separator).__name__,
            "separator_model": str(getattr(self.separator, "model_name", "")),
            "transcriber": type(self.transcriber).__name__,
            "transcriber_model": str(getattr(self.transcriber, "model_name", "")),
            "diarizer": type(self.diarizer).__name__,
            "diarizer_model": str(
                getattr(
                    self.diarizer,
                    "msdd_model",
                    getattr(self.diarizer, "model_name", ""),
                )
            ),
        }
