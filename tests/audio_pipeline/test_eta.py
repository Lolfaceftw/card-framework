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

