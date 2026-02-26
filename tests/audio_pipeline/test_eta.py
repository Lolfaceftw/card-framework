from pathlib import Path
import json

import pytest

from audio_pipeline.eta import (
    LinearStageEtaStrategy,
    StageSpeedProfile,
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
    assert estimate == 15.0


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
