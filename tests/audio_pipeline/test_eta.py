from pathlib import Path
import json

import pytest

from card_framework.audio_pipeline.eta import (
    DynamicEtaTracker,
    LinearStageEtaStrategy,
    StageSpeedProfile,
    StageProgressUpdate,
    format_eta_seconds,
)
from card_framework.audio_pipeline.contracts import (
    DiarizationTurn,
    TimedTextSegment,
    TranscriptSegment,
    TranscriptionResult,
)
from card_framework.audio_pipeline.orchestrator import AudioToScriptOrchestrator


def test_linear_eta_strategy_uses_device_multiplier() -> None:
    strategy = LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=4.0, cuda=1.0),
        transcription=StageSpeedProfile(cpu=2.0, cuda=0.5),
        diarization=StageSpeedProfile(cpu=3.0, cuda=0.75),
    )

    estimated_cpu = strategy.estimate_total_seconds(
        stage="separation",
        audio_duration_ms=1000,
        device="cpu",
    )
    estimated_cuda = strategy.estimate_total_seconds(
        stage="separation",
        audio_duration_ms=1000,
        device="cuda",
    )

    assert estimated_cpu == 4.0
    assert estimated_cuda == 1.0


def test_format_eta_seconds_renders_expected_units() -> None:
    assert format_eta_seconds(None) == "unknown"
    assert format_eta_seconds(4.2) == "4s"
    assert format_eta_seconds(90) == "1m 30s"
    assert format_eta_seconds(3723) == "1h 02m 03s"


def test_linear_eta_strategy_refits_from_observed_throughput() -> None:
    strategy = LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=1.0, cuda=1.0),
        transcription=StageSpeedProfile(cpu=1.0, cuda=1.0),
        diarization=StageSpeedProfile(cpu=1.0, cuda=1.0),
    )

    baseline = strategy.estimate_total_seconds(
        stage="transcription",
        audio_duration_ms=10_000,
        device="cuda",
    )
    assert baseline == 10.0

    strategy.observe_stage_duration(
        stage="transcription",
        audio_duration_ms=10_000,
        elapsed_seconds=20.0,
        device="cuda",
    )
    adapted = strategy.estimate_total_seconds(
        stage="transcription",
        audio_duration_ms=10_000,
        device="cuda",
    )

    assert adapted == 20.0


def test_linear_eta_strategy_estimates_unit_stage_from_defaults() -> None:
    strategy = LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=1.0, cuda=1.0),
        transcription=StageSpeedProfile(cpu=1.0, cuda=1.0),
        diarization=StageSpeedProfile(cpu=1.0, cuda=1.0),
    )

    estimated = strategy.estimate_unit_stage_total_seconds(
        stage="speaker_samples",
        total_units=3,
    )
    assert estimated == 24.0


def test_linear_eta_strategy_refits_unit_stage_throughput() -> None:
    strategy = LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=1.0, cuda=1.0),
        transcription=StageSpeedProfile(cpu=1.0, cuda=1.0),
        diarization=StageSpeedProfile(cpu=1.0, cuda=1.0),
    )
    baseline = strategy.estimate_unit_stage_total_seconds(
        stage="voice_clone",
        total_units=2,
    )
    assert baseline == 40.0

    strategy.observe_unit_stage_duration(
        stage="voice_clone",
        total_units=2,
        elapsed_seconds=60.0,
    )
    adapted = strategy.estimate_unit_stage_total_seconds(
        stage="voice_clone",
        total_units=2,
    )
    assert adapted == 60.0


def test_linear_eta_strategy_profile_round_trip(tmp_path: Path) -> None:
    strategy = LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=1.0, cuda=1.0),
        transcription=StageSpeedProfile(cpu=1.0, cuda=1.0),
        diarization=StageSpeedProfile(cpu=1.0, cuda=1.0),
    )
    strategy.observe_stage_duration(
        stage="separation",
        audio_duration_ms=5_000,
        elapsed_seconds=15.0,
        device="cpu",
    )
    strategy.observe_unit_stage_duration(
        stage="speaker_samples",
        total_units=3,
        elapsed_seconds=21.0,
    )

    profile_path = tmp_path / "eta_profile.json"
    strategy.save_profile(profile_path)

    reloaded = LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=1.0, cuda=1.0),
        transcription=StageSpeedProfile(cpu=1.0, cuda=1.0),
        diarization=StageSpeedProfile(cpu=1.0, cuda=1.0),
    )
    reloaded.load_profile(profile_path)

    estimate = reloaded.estimate_total_seconds(
        stage="separation",
        audio_duration_ms=5_000,
        device="cpu",
    )
    unit_estimate = reloaded.estimate_unit_stage_total_seconds(
        stage="speaker_samples",
        total_units=3,
    )
    assert estimate == 15.0
    assert unit_estimate == 21.0


def test_linear_eta_strategy_ignores_context_mismatch(tmp_path: Path) -> None:
    strategy = LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=1.0, cuda=1.0),
        transcription=StageSpeedProfile(cpu=1.0, cuda=1.0),
        diarization=StageSpeedProfile(cpu=1.0, cuda=1.0),
    )
    profile_path = tmp_path / "eta_profile.json"
    profile_path.write_text(
        json.dumps(
            {
                "version": 1,
                "context": {"transcriber_model": "large-v3"},
                "stages": {
                    "transcription": {
                        "cuda": {"multiplier": 9.0, "samples": 4}
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    strategy.load_profile(
        profile_path,
        context={"transcriber_model": "distil-large-v3"},
    )

    estimate = strategy.estimate_total_seconds(
        stage="transcription",
        audio_duration_ms=1_000,
        device="cuda",
    )
    assert estimate == 1.0


def test_linear_eta_strategy_validates_learning_bounds() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        LinearStageEtaStrategy(
            separation=StageSpeedProfile(cpu=1.0, cuda=1.0),
            transcription=StageSpeedProfile(cpu=1.0, cuda=1.0),
            diarization=StageSpeedProfile(cpu=1.0, cuda=1.0),
            learning_rate=1.5,
        )


def test_dynamic_eta_tracker_reestimates_total_from_progress() -> None:
    tracker = DynamicEtaTracker(
        initial_total_seconds=100.0,
        progress_smoothing=1.0,
    )
    tracker.observe_progress(
        elapsed_seconds=10.0,
        update=StageProgressUpdate(completed_units=2, total_units=10),
    )
    remaining = tracker.estimate_signed_remaining_seconds(elapsed_seconds=10.0)
    assert remaining == 40.0


def test_dynamic_eta_tracker_contracts_quickly_after_real_progress() -> None:
    tracker = DynamicEtaTracker(
        initial_total_seconds=3_600.0,
        progress_smoothing=0.25,
    )
    tracker.observe_progress(
        elapsed_seconds=24.0,
        update=StageProgressUpdate(completed_units=1, total_units=10),
    )

    assert tracker.estimate_total_seconds(elapsed_seconds=24.0) == pytest.approx(240.0)
    assert tracker.estimate_signed_remaining_seconds(elapsed_seconds=24.0) == pytest.approx(
        216.0
    )


def test_dynamic_eta_tracker_expands_gradually_when_runtime_slows() -> None:
    tracker = DynamicEtaTracker(
        initial_total_seconds=100.0,
        progress_smoothing=0.25,
    )
    tracker.observe_progress(
        elapsed_seconds=60.0,
        update=StageProgressUpdate(completed_units=1, total_units=2),
    )

    assert tracker.estimate_total_seconds(elapsed_seconds=60.0) == pytest.approx(105.0)


def test_dynamic_eta_tracker_inflates_after_overrun() -> None:
    tracker = DynamicEtaTracker(
        initial_total_seconds=5.0,
        overrun_factor=1.2,
    )
    remaining = tracker.estimate_signed_remaining_seconds(elapsed_seconds=10.0)
    assert remaining < 0
    assert tracker.estimate_total_seconds(elapsed_seconds=10.0) == 12.0


class _FixedEtaStrategy:
    """Return one fixed ETA estimate for orchestrator progress tests."""

    def estimate_total_seconds(
        self,
        *,
        stage: str,
        audio_duration_ms: int,
        device: str,
    ) -> float | None:
        del stage, audio_duration_ms, device
        return 120.0

    def has_stage_history(self, *, stage: str, device: str) -> bool:
        """Report available history so the orchestrator emits ETA-bearing updates."""
        del stage, device
        return True


def test_audio_orchestrator_transcription_progress_includes_segment_eta() -> None:
    """Publish inline segment progress with ETA during chunked transcription."""
    orchestrator = AudioToScriptOrchestrator(
        separator=object(),
        transcriber=object(),
        diarizer=object(),
        eta_strategy=_FixedEtaStrategy(),
        eta_update_interval_seconds=0.0,
    )
    status_messages: list[str] = []

    def _capture(message: str, inline: bool = False) -> None:
        del inline
        status_messages.append(message)

    from card_framework.shared.events import event_bus

    event_bus.subscribe("status_message", _capture)
    try:
        result = orchestrator._run_stage_with_eta(
            stage="transcription",
            description="transcribing vocals sample.wav",
            audio_duration_ms=60_000,
            device="cpu",
            operation=lambda progress_callback: (
                progress_callback(
                    StageProgressUpdate(
                        completed_units=1,
                        total_units=3,
                        processed_audio_ms=30_000,
                        note="segment 1/3",
                    )
                ),
                "ok",
            )[1],
        )
    finally:
        event_bus.unsubscribe("status_message", _capture)

    assert result == "ok"
    assert any(
        message.startswith(
            "Audio stage: transcription segment 1/3; estimated time left "
        )
        for message in status_messages
    )


def test_audio_orchestrator_separation_progress_emits_inline_and_log_updates() -> None:
    """Publish separation chunk progress to both inline status and log events."""
    orchestrator = AudioToScriptOrchestrator(
        separator=object(),
        transcriber=object(),
        diarizer=object(),
        eta_strategy=_FixedEtaStrategy(),
        eta_update_interval_seconds=0.0,
    )
    inline_messages: list[tuple[str, bool]] = []
    log_messages: list[str] = []

    def _capture_status(message: str, inline: bool = False) -> None:
        inline_messages.append((message, inline))

    def _capture_log(message: str) -> None:
        log_messages.append(message)

    from card_framework.shared.events import event_bus

    event_bus.subscribe("status_message", _capture_status)
    event_bus.subscribe("status_log_message", _capture_log)
    try:
        result = orchestrator._run_stage_with_eta(
            stage="separation",
            description="separating sources from sample.wav",
            audio_duration_ms=60_000,
            device="cpu",
            operation=lambda progress_callback: (
                progress_callback(
                    StageProgressUpdate(
                        completed_units=1,
                        total_units=2,
                        processed_audio_ms=30_000,
                        note="chunk 1/2",
                    )
                ),
                "ok",
            )[1],
        )
    finally:
        event_bus.unsubscribe("status_message", _capture_status)
        event_bus.unsubscribe("status_log_message", _capture_log)

    assert result == "ok"
    assert any(
        message.startswith(
            "Audio stage: separation chunk 1/2; estimated time left "
        )
        and inline
        for message, inline in inline_messages
    )
    assert any(
        message.startswith(
            "Audio stage: separation chunk 1/2; estimated time left "
        )
        for message in log_messages
    )


@pytest.mark.parametrize(
    ("stage", "description", "note"),
    [
        ("separation", "separating sources from sample.wav", "chunk 1/2"),
        ("transcription", "transcribing vocals sample.wav", "segment 1/2"),
    ],
)
def test_audio_orchestrator_uses_default_eta_without_history(
    stage: str,
    description: str,
    note: str,
) -> None:
    """Use configured stage multipliers for audio ETA before any history exists."""
    strategy = LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=4.0, cuda=0.8),
        transcription=StageSpeedProfile(cpu=1.5, cuda=0.35),
        diarization=StageSpeedProfile(cpu=2.5, cuda=0.9),
    )
    assert not strategy.has_stage_history(stage=stage, device="cpu")

    orchestrator = AudioToScriptOrchestrator(
        separator=object(),
        transcriber=object(),
        diarizer=object(),
        eta_strategy=strategy,
        eta_update_interval_seconds=0.0,
    )
    system_messages: list[str] = []
    status_messages: list[str] = []

    def _capture_system(message: str) -> None:
        system_messages.append(message)

    def _capture_status(message: str, inline: bool = False) -> None:
        del inline
        status_messages.append(message)

    from card_framework.shared.events import event_bus

    event_bus.subscribe("system_message", _capture_system)
    event_bus.subscribe("status_message", _capture_status)
    try:
        result = orchestrator._run_stage_with_eta(
            stage=stage,
            description=description,
            audio_duration_ms=60_000,
            device="cpu",
            operation=lambda progress_callback: (
                progress_callback(
                    StageProgressUpdate(
                        completed_units=1,
                        total_units=2,
                        processed_audio_ms=30_000,
                        note=note,
                    )
                ),
                "ok",
            )[1],
        )
    finally:
        event_bus.unsubscribe("system_message", _capture_system)
        event_bus.unsubscribe("status_message", _capture_status)

    assert result == "ok"
    assert any(
        message.startswith(f"Audio stage: {description} (estimated time left ")
        for message in system_messages
    )
    assert any(
        message.startswith(f"Audio stage: {stage} {note}; estimated time left ")
        for message in status_messages
    )


def test_audio_orchestrator_releases_stage_models_between_transcription_and_diarization(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Release transcriber resources before diarization and diarizer resources after use."""

    class _FakeSeparator:
        def __init__(self, vocals_path: Path) -> None:
            self.vocals_path = vocals_path

        def separate_vocals(
            self,
            *,
            input_audio_path: Path,
            output_dir: Path,
            device: str,
            progress_callback=None,
        ) -> Path:
            del input_audio_path, output_dir, device, progress_callback
            self.vocals_path.parent.mkdir(parents=True, exist_ok=True)
            self.vocals_path.write_bytes(b"vocals")
            return self.vocals_path

    class _FakeTranscriber:
        def __init__(self, release_order: list[str]) -> None:
            self.release_order = release_order

        def transcribe(
            self,
            audio_path: Path,
            *,
            device: str,
            progress_callback=None,
        ) -> TranscriptionResult:
            del audio_path, device, progress_callback
            return TranscriptionResult(
                segments=[
                    TimedTextSegment(
                        start_time_ms=0,
                        end_time_ms=1000,
                        text="hello world",
                    )
                ],
                word_timestamps=[],
                language="en",
            )

        def release_cached_resources(self, *, device: str) -> None:
            self.release_order.append(f"transcriber:{device}")

    class _FakeDiarizer:
        def __init__(self, release_order: list[str]) -> None:
            self.release_order = release_order

        def diarize(
            self,
            audio_path: Path,
            output_dir: Path,
            *,
            device: str,
            progress_callback=None,
        ) -> list[DiarizationTurn]:
            del audio_path, output_dir, device, progress_callback
            assert self.release_order == ["transcriber:cuda"]
            return [
                DiarizationTurn(
                    speaker="SPEAKER_00",
                    start_time_ms=0,
                    end_time_ms=1000,
                )
            ]

        def release_cached_resources(self, *, device: str) -> None:
            self.release_order.append(f"diarizer:{device}")

    release_order: list[str] = []
    input_audio_path = tmp_path / "input.wav"
    input_audio_path.write_bytes(b"RIFF")
    output_transcript_path = tmp_path / "transcript.json"
    vocals_path = tmp_path / "work" / "separation" / "vocals.wav"
    orchestrator = AudioToScriptOrchestrator(
        separator=_FakeSeparator(vocals_path),
        transcriber=_FakeTranscriber(release_order),
        diarizer=_FakeDiarizer(release_order),
        eta_update_interval_seconds=0.0,
    )

    monkeypatch.setattr(
        "card_framework.audio_pipeline.orchestrator.probe_audio_duration_ms",
        lambda path: 1000,
    )
    monkeypatch.setattr(
        AudioToScriptOrchestrator,
        "_align_transcript_segments",
        lambda self, *, asr_result, diarization_turns, warnings: (
            [
                TranscriptSegment(
                    speaker="SPEAKER_00",
                    start_time=0,
                    end_time=1000,
                    text="hello world",
                )
            ],
            True,
        ),
    )

    result = orchestrator.run(
        input_audio_path=input_audio_path,
        output_transcript_path=output_transcript_path,
        work_dir=tmp_path / "work",
        device="cuda",
    )

    assert result.segment_count == 1
    assert release_order[:2] == ["transcriber:cuda", "diarizer:cuda"]

