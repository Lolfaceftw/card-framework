import pytest

from audio_pipeline.factory import build_audio_to_script_orchestrator
from audio_pipeline.eta import LinearStageEtaStrategy
from audio_pipeline.gateways.demucs_gateway import DemucsSourceSeparator
from audio_pipeline.gateways.faster_whisper_gateway import FasterWhisperTranscriber
from audio_pipeline.gateways.fallback_gateways import (
    PassthroughSourceSeparator,
    SingleSpeakerDiarizer,
)
from audio_pipeline.gateways.nemo_diarizer_gateway import NemoSpeakerDiarizer


def test_factory_builds_default_strategies() -> None:
    orchestrator = build_audio_to_script_orchestrator(
        {
            "separation": {"provider": "demucs", "model": "htdemucs"},
            "asr": {"provider": "faster_whisper", "model": "large-v3"},
            "diarization": {"provider": "nemo"},
        }
    )

    assert isinstance(orchestrator.separator, DemucsSourceSeparator)
    assert isinstance(orchestrator.transcriber, FasterWhisperTranscriber)
    assert isinstance(orchestrator.diarizer, NemoSpeakerDiarizer)
    assert isinstance(orchestrator.eta_strategy, LinearStageEtaStrategy)


def test_factory_builds_eta_strategy_from_config() -> None:
    orchestrator = build_audio_to_script_orchestrator(
        {
            "separation": {"provider": "demucs", "model": "htdemucs"},
            "asr": {"provider": "faster_whisper", "model": "large-v3"},
            "diarization": {"provider": "nemo"},
            "eta": {
                "update_interval_seconds": 5,
                "adaptive": {
                    "learning_rate": 0.5,
                    "min_multiplier": 0.1,
                    "max_multiplier": 10.0,
                },
                "stage_multipliers": {
                    "separation": {"cpu": 6.0, "cuda": 1.0},
                    "transcription": {"cpu": 3.0, "cuda": 0.5},
                    "diarization": {"cpu": 4.0, "cuda": 0.75},
                },
            },
        }
    )

    assert isinstance(orchestrator.eta_strategy, LinearStageEtaStrategy)
    assert orchestrator.eta_update_interval_seconds == 5.0
    assert orchestrator.eta_strategy.learning_rate == 0.5
    assert orchestrator.eta_strategy.min_multiplier == 0.1
    assert orchestrator.eta_strategy.max_multiplier == 10.0
    estimated = orchestrator.eta_strategy.estimate_total_seconds(
        stage="transcription",
        audio_duration_ms=2000,
        device="cpu",
    )
    assert estimated == 6.0


def test_factory_supports_fallback_strategies() -> None:
    orchestrator = build_audio_to_script_orchestrator(
        {
            "separation": {"provider": "passthrough"},
            "asr": {"provider": "faster_whisper", "model": "tiny"},
            "diarization": {"provider": "single_speaker"},
        }
    )

    assert isinstance(orchestrator.separator, PassthroughSourceSeparator)
    assert isinstance(orchestrator.diarizer, SingleSpeakerDiarizer)


def test_factory_raises_for_unknown_provider() -> None:
    with pytest.raises(ValueError):
        build_audio_to_script_orchestrator(
            {
                "separation": {"provider": "unknown_provider"},
                "asr": {"provider": "faster_whisper"},
                "diarization": {"provider": "nemo"},
            }
        )


def test_factory_validates_eta_adaptive_bounds() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        build_audio_to_script_orchestrator(
            {
                "separation": {"provider": "demucs"},
                "asr": {"provider": "faster_whisper"},
                "diarization": {"provider": "nemo"},
                "eta": {
                    "adaptive": {
                        "learning_rate": -0.1,
                        "min_multiplier": 0.1,
                        "max_multiplier": 5.0,
                    }
                },
            }
        )
