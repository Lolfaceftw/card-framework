from pathlib import Path

import pytest

from card_framework.audio_pipeline.factory import (
    build_audio_to_script_orchestrator,
    build_interjector_orchestrator,
    build_speaker_diarizer,
    build_speaker_sample_generator,
    build_voice_clone_orchestrator,
)
from card_framework.audio_pipeline.eta import LinearStageEtaStrategy
from card_framework.audio_pipeline.gateways.fallback_voice_clone_gateway import (
    PassthroughVoiceCloneGateway,
)
from card_framework.audio_pipeline.gateways.indextts_voice_clone_gateway import (
    IndexTTSVoiceCloneGateway,
)
from card_framework.audio_pipeline.gateways.demucs_gateway import DemucsSourceSeparator
from card_framework.audio_pipeline.gateways.faster_whisper_gateway import FasterWhisperTranscriber
from card_framework.audio_pipeline.gateways.fallback_gateways import (
    PassthroughSourceSeparator,
    SingleSpeakerDiarizer,
)
from card_framework.audio_pipeline.gateways.nemo_diarizer_gateway import NemoSpeakerDiarizer
from card_framework.audio_pipeline.gateways.pyannote_diarizer_gateway import PyannoteSpeakerDiarizer
from card_framework.audio_pipeline.gateways.speaker_sample_gateway import FfmpegSpeakerSampleExporter
from card_framework.audio_pipeline.gateways.sortformer_diarizer_gateway import SortformerSpeakerDiarizer
from card_framework.shared.llm_provider import LLMProvider


class _FakeLLMProvider(LLMProvider):
    """Minimal LLM provider used for factory wiring tests."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        del system_prompt, user_prompt, max_tokens
        return '{"decisions": []}'

    def chat(
        self,
        messages,
        tools=None,
        tool_choice=None,
        max_tokens: int | None = None,
    ):
        del messages, tools, tool_choice, max_tokens
        return {}


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
    assert orchestrator.diarizer.allow_single_speaker_fallback is False
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


def test_factory_allows_explicit_single_speaker_fallback_opt_in() -> None:
    orchestrator = build_audio_to_script_orchestrator(
        {
            "separation": {"provider": "demucs", "model": "htdemucs"},
            "asr": {"provider": "faster_whisper", "model": "large-v3"},
            "diarization": {
                "provider": "nemo",
                "allow_single_speaker_fallback": True,
            },
        }
    )

    assert isinstance(orchestrator.diarizer, NemoSpeakerDiarizer)
    assert orchestrator.diarizer.allow_single_speaker_fallback is True


def test_factory_builds_pyannote_community1_diarizer() -> None:
    diarizer = build_speaker_diarizer(
        {
            "diarization": {
                "provider": "pyannote_community1",
                "min_speakers": 2,
                "max_speakers": 4,
                "pyannote": {
                    "pipeline_name": "pyannote/speaker-diarization-community-1",
                    "auth_token_env": "HF_TOKEN",
                    "use_exclusive_diarization": True,
                },
            }
        }
    )

    assert isinstance(diarizer, PyannoteSpeakerDiarizer)
    assert diarizer.pipeline_name == "pyannote/speaker-diarization-community-1"
    assert diarizer.auth_token_env == "HF_TOKEN"
    assert diarizer.use_exclusive_diarization is True
    assert diarizer.min_speakers == 2
    assert diarizer.max_speakers == 4


def test_factory_builds_sortformer_diarizers() -> None:
    offline = build_speaker_diarizer(
        {
            "diarization": {
                "provider": "nemo_sortformer_offline",
                "sortformer_offline": {
                    "model_name": "nvidia/diar_sortformer_4spk-v1",
                    "batch_size": 2,
                },
            }
        }
    )
    streaming = build_speaker_diarizer(
        {
            "diarization": {
                "provider": "nemo_sortformer_streaming",
                "sortformer_streaming": {
                    "model_name": "nvidia/diar_streaming_sortformer_4spk-v2",
                    "batch_size": 1,
                    "chunk_len": 124,
                    "chunk_right_context": 1,
                    "fifo_len": 124,
                    "spkcache_update_period": 124,
                    "spkcache_len": 188,
                },
            }
        }
    )

    assert isinstance(offline, SortformerSpeakerDiarizer)
    assert offline.streaming_mode is False
    assert offline.batch_size == 2
    assert isinstance(streaming, SortformerSpeakerDiarizer)
    assert streaming.streaming_mode is True
    assert streaming.chunk_len == 124
    assert streaming.chunk_right_context == 1
    assert streaming.fifo_len == 124
    assert streaming.spkcache_update_period == 124
    assert streaming.spkcache_len == 188


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
    assert generator.sample_rate_hz == 44100
    assert generator.channels == 1
    assert generator.edge_fade_ms == 20
    assert generator.min_slice_duration_ms == 1200
    assert generator.max_slices == 6
    assert generator.selection_order == "chronological"
    assert generator.low_data_policy == "quality_first_shorter"
    assert generator.audio_codec == "pcm_s24le"
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
    assert orchestrator.merge_segments is True
    assert orchestrator.merged_output_filename == "voice_cloned.wav"


def test_factory_disables_voice_clone_orchestrator_by_default(tmp_path: Path) -> None:
    orchestrator = build_voice_clone_orchestrator({}, project_root=tmp_path)

    assert orchestrator is None


def test_factory_builds_interjector_orchestrator_when_enabled(tmp_path: Path) -> None:
    orchestrator = build_interjector_orchestrator(
        {
            "work_dir": "artifacts/audio_stage",
            "voice_clone": {
                "provider": "passthrough",
            },
            "interjector": {
                "enabled": True,
                "output_dir_name": "interjector",
            },
        },
        llm=_FakeLLMProvider(),
        project_root=tmp_path,
    )

    assert orchestrator is not None
    assert isinstance(orchestrator.provider, PassthroughVoiceCloneGateway)
    assert orchestrator.output_dir == (
        tmp_path / "artifacts" / "audio_stage" / "interjector"
    ).resolve()
    assert orchestrator.merged_output_filename == "voice_cloned_interjected.wav"


def test_factory_disables_interjector_orchestrator_by_default(tmp_path: Path) -> None:
    orchestrator = build_interjector_orchestrator(
        {},
        llm=_FakeLLMProvider(),
        project_root=tmp_path,
    )

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
    assert provider.num_beams == 1
    assert provider.runner_project_dir == (
        tmp_path / "src" / "card_framework" / "_vendor" / "index_tts"
    ).resolve()
    assert provider.cfg_path == (
        tmp_path / "checkpoints" / "index_tts" / "config.yaml"
    ).resolve()
    assert provider.model_dir == (
        tmp_path / "checkpoints" / "index_tts"
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


def test_factory_respects_indextts_generation_overrides(tmp_path: Path) -> None:
    orchestrator = build_voice_clone_orchestrator(
        {
            "voice_clone": {
                "enabled": True,
                "provider": "indextts",
                "num_beams": 2,
                "top_p": 0.9,
                "top_k": 40,
                "temperature": 0.7,
                "length_penalty": 0.2,
                "repetition_penalty": 8.0,
                "max_mel_tokens": 1200,
            }
        },
        project_root=tmp_path,
    )

    assert orchestrator is not None
    provider = orchestrator.provider
    assert isinstance(provider, IndexTTSVoiceCloneGateway)
    assert provider.num_beams == 2
    assert provider.top_p == 0.9
    assert provider.top_k == 40
    assert provider.temperature == 0.7
    assert provider.length_penalty == 0.2
    assert provider.repetition_penalty == 8.0
    assert provider.max_mel_tokens == 1200


def test_factory_respects_voice_clone_merged_output_override(tmp_path: Path) -> None:
    orchestrator = build_voice_clone_orchestrator(
        {
            "voice_clone": {
                "enabled": True,
                "provider": "passthrough",
                "merge_segments": True,
                "merged_output_filename": "custom_voice_mix.wav",
            }
        },
        project_root=tmp_path,
    )

    assert orchestrator is not None
    assert orchestrator.merge_segments is True
    assert orchestrator.merged_output_filename == "custom_voice_mix.wav"


def test_factory_builds_interjector_with_indextts_defaults(tmp_path: Path) -> None:
    orchestrator = build_interjector_orchestrator(
        {
            "interjector": {
                "enabled": True,
            },
            "voice_clone": {
                "provider": "indextts",
            },
        },
        llm=_FakeLLMProvider(),
        project_root=tmp_path,
    )

    assert orchestrator is not None
    provider = orchestrator.provider
    assert isinstance(provider, IndexTTSVoiceCloneGateway)
    assert provider.runner_project_dir == (
        tmp_path / "src" / "card_framework" / "_vendor" / "index_tts"
    ).resolve()

