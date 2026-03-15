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

