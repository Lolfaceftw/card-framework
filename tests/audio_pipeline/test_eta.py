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
