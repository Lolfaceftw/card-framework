from pathlib import Path

import pytest

from audio_pipeline.factory import (
    build_audio_to_script_orchestrator,
    build_speaker_sample_generator,
    build_voice_clone_orchestrator,
)
from audio_pipeline.eta import LinearStageEtaStrategy
from audio_pipeline.gateways.fallback_voice_clone_gateway import (
    PassthroughVoiceCloneGateway,
)
from audio_pipeline.gateways.indextts_voice_clone_gateway import (
    IndexTTSVoiceCloneGateway,
)
from audio_pipeline.gateways.demucs_gateway import DemucsSourceSeparator
from audio_pipeline.gateways.faster_whisper_gateway import FasterWhisperTranscriber
from audio_pipeline.gateways.fallback_gateways import (
    PassthroughSourceSeparator,
    SingleSpeakerDiarizer,
)
from audio_pipeline.gateways.nemo_diarizer_gateway import NemoSpeakerDiarizer
from audio_pipeline.gateways.speaker_sample_gateway import FfmpegSpeakerSampleExporter


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
                "dynamic": {
                    "progress_smoothing": 0.5,
                    "overrun_factor": 1.3,
                    "headroom_seconds": 2.0,
                },
                "adaptive": {
                    "learning_rate": 0.5,
                    "min_multiplier": 0.1,
                    "max_multiplier": 10.0,
                },
                "unit_bootstrap_seconds_per_unit": {
                    "speaker_samples": 9.0,
                    "voice_clone": 30.0,
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
    assert orchestrator.eta_progress_smoothing == 0.5
    assert orchestrator.eta_overrun_factor == 1.3
    assert orchestrator.eta_headroom_seconds == 2.0
    assert orchestrator.eta_strategy.unit_stage_defaults["speaker_samples"] == 9.0
    assert orchestrator.eta_strategy.unit_stage_defaults["voice_clone"] == 30.0
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


def test_factory_validates_eta_dynamic_bounds() -> None:
    with pytest.raises(ValueError, match="progress_smoothing"):
        build_audio_to_script_orchestrator(
            {
                "separation": {"provider": "demucs"},
                "asr": {"provider": "faster_whisper"},
                "diarization": {"provider": "nemo"},
                "eta": {
                    "dynamic": {
                        "progress_smoothing": 0.0,
                        "overrun_factor": 1.2,
                        "headroom_seconds": 0.0,
                    }
                },
            }
        )


def test_factory_builds_speaker_sample_generator_defaults() -> None:
    generator = build_speaker_sample_generator({})

    assert isinstance(generator.exporter, FfmpegSpeakerSampleExporter)
    assert generator.target_duration_seconds == 30
    assert generator.sample_rate_hz == 16000
    assert generator.channels == 1
    assert generator.clip_method == "concat_turns"
    assert generator.short_speaker_policy == "export_shorter"


def test_factory_builds_voice_clone_orchestrator_when_enabled(tmp_path: Path) -> None:
    orchestrator = build_voice_clone_orchestrator(
        {
            "work_dir": "artifacts/audio_stage",
            "voice_clone": {
                "enabled": True,
                "provider": "passthrough",
                "output_dir_name": "voice_clone",
            },
        },
        project_root=tmp_path,
    )

    assert orchestrator is not None
    assert isinstance(orchestrator.provider, PassthroughVoiceCloneGateway)
    assert orchestrator.output_dir == (
        tmp_path / "artifacts" / "audio_stage" / "voice_clone"
    ).resolve()


def test_factory_disables_voice_clone_orchestrator_by_default(tmp_path: Path) -> None:
    orchestrator = build_voice_clone_orchestrator({}, project_root=tmp_path)

    assert orchestrator is None


def test_factory_raises_for_unknown_voice_clone_provider(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported voice clone provider"):
        build_voice_clone_orchestrator(
            {
                "voice_clone": {
                    "enabled": True,
                    "provider": "unknown",
                }
            },
            project_root=tmp_path,
        )


def test_factory_builds_indextts_subprocess_provider_defaults(tmp_path: Path) -> None:
    orchestrator = build_voice_clone_orchestrator(
        {
            "voice_clone": {
                "enabled": True,
                "provider": "indextts",
            }
        },
        project_root=tmp_path,
    )

    assert orchestrator is not None
    provider = orchestrator.provider
    assert isinstance(provider, IndexTTSVoiceCloneGateway)
    assert provider.execution_backend == "subprocess"
    assert provider.stream_subprocess_output is True
    assert provider.runner_project_dir == (tmp_path / "third_party" / "index_tts").resolve()
    assert provider.cfg_path == (
        tmp_path / "third_party" / "index_tts" / "checkpoints" / "config.yaml"
    ).resolve()
    assert provider.model_dir == (
        tmp_path / "third_party" / "index_tts" / "checkpoints"
    ).resolve()


def test_factory_respects_indextts_stream_subprocess_override(tmp_path: Path) -> None:
    orchestrator = build_voice_clone_orchestrator(
        {
            "voice_clone": {
                "enabled": True,
                "provider": "indextts",
                "stream_subprocess_output": False,
            }
        },
        project_root=tmp_path,
    )

    assert orchestrator is not None
    provider = orchestrator.provider
    assert isinstance(provider, IndexTTSVoiceCloneGateway)
    assert provider.stream_subprocess_output is False
