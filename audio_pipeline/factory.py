"""Composition root for audio pipeline adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from audio_pipeline.contracts import SourceSeparator, SpeakerDiarizer, SpeechTranscriber
from audio_pipeline.eta import (
    EtaUnitStageName,
    LinearStageEtaStrategy,
    StageEtaStrategy,
    StageSpeedProfile,
    default_stage_eta_strategy,
)
from audio_pipeline.gateways.demucs_gateway import DemucsSourceSeparator
from audio_pipeline.gateways.fallback_gateways import (
    PassthroughSourceSeparator,
    SingleSpeakerDiarizer,
)
from audio_pipeline.gateways.faster_whisper_gateway import FasterWhisperTranscriber
from audio_pipeline.gateways.nemo_diarizer_gateway import NemoSpeakerDiarizer
from audio_pipeline.gateways.speaker_sample_gateway import FfmpegSpeakerSampleExporter
from audio_pipeline.interjector import (
    InterjectorOrchestrator,
    LLMInterjectionPlanner,
)
from audio_pipeline.gateways.fallback_voice_clone_gateway import (
    PassthroughVoiceCloneGateway,
)
from audio_pipeline.gateways.indextts_voice_clone_gateway import (
    IndexTTSVoiceCloneGateway,
)
from audio_pipeline.orchestrator import AudioToScriptOrchestrator
from audio_pipeline.runtime import resolve_device, resolve_path
from audio_pipeline.speaker_samples import SpeakerSampleGenerator
from audio_pipeline.voice_clone_contracts import VoiceCloneProvider
from audio_pipeline.voice_clone_orchestrator import VoiceCloneOrchestrator
from llm_provider import LLMProvider


def build_audio_to_script_orchestrator(
    audio_cfg: Mapping[str, Any],
) -> AudioToScriptOrchestrator:
    """
    Build audio-stage orchestrator from config mapping.

    Args:
        audio_cfg: ``audio`` config section.

    Returns:
        Wired ``AudioToScriptOrchestrator`` instance.
    """
    separation_cfg = _as_mapping(audio_cfg.get("separation", {}))
    asr_cfg = _as_mapping(audio_cfg.get("asr", {}))
    diarization_cfg = _as_mapping(audio_cfg.get("diarization", {}))
    retry_cfg = _as_mapping(audio_cfg.get("retry", {}))
    eta_cfg = _as_mapping(audio_cfg.get("eta", {}))
    dynamic_eta_cfg = _as_mapping(eta_cfg.get("dynamic", {}))
    eta_progress_smoothing = float(dynamic_eta_cfg.get("progress_smoothing", 0.25))
    eta_overrun_factor = float(dynamic_eta_cfg.get("overrun_factor", 1.15))
    eta_headroom_seconds = float(dynamic_eta_cfg.get("headroom_seconds", 1.0))
    _validate_eta_dynamic_config(
        progress_smoothing=eta_progress_smoothing,
        overrun_factor=eta_overrun_factor,
        headroom_seconds=eta_headroom_seconds,
    )

    separator = _build_separator(
        separation_cfg=separation_cfg,
        retry_cfg=retry_cfg,
    )
    transcriber = _build_transcriber(asr_cfg=asr_cfg)
    diarizer = _build_diarizer(diarization_cfg=diarization_cfg)
    eta_strategy = _build_eta_strategy(eta_cfg=eta_cfg)
    return AudioToScriptOrchestrator(
        separator=separator,
        transcriber=transcriber,
        diarizer=diarizer,
        merge_gap_ms=int(audio_cfg.get("merge_gap_ms", 800)),
        default_speaker=str(audio_cfg.get("default_speaker", "SPEAKER_00")),
        eta_strategy=eta_strategy,
        eta_update_interval_seconds=float(
            eta_cfg.get("update_interval_seconds", 10.0)
        ),
        eta_progress_smoothing=eta_progress_smoothing,
        eta_overrun_factor=eta_overrun_factor,
        eta_headroom_seconds=eta_headroom_seconds,
    )


def build_speaker_sample_generator(audio_cfg: Mapping[str, Any]) -> SpeakerSampleGenerator:
    """
    Build post-transcript speaker-sample generator from audio config mapping.

    Args:
        audio_cfg: ``audio`` config section.

    Returns:
        Wired ``SpeakerSampleGenerator`` instance.
    """
    speaker_samples_cfg = _as_mapping(audio_cfg.get("speaker_samples", {}))
    clip_method = str(speaker_samples_cfg.get("clip_method", "concat_turns"))
    short_policy = str(
        speaker_samples_cfg.get("short_speaker_policy", "export_shorter")
    )
    selection_order = str(speaker_samples_cfg.get("selection_order", "chronological"))
    low_data_policy = str(
        speaker_samples_cfg.get("low_data_policy", "quality_first_shorter")
    )
    audio_codec = str(speaker_samples_cfg.get("audio_codec", "pcm_s24le"))
    if clip_method != "concat_turns":
        raise ValueError("audio.speaker_samples.clip_method must be 'concat_turns'.")
    if short_policy != "export_shorter":
        raise ValueError(
            "audio.speaker_samples.short_speaker_policy must be 'export_shorter'."
        )
    if selection_order not in {"chronological", "longest_first"}:
        raise ValueError(
            "audio.speaker_samples.selection_order must be one of: "
            "chronological, longest_first."
        )
    if low_data_policy not in {
        "quality_first_shorter",
        "backfill_to_target",
        "fail_speaker",
    }:
        raise ValueError(
            "audio.speaker_samples.low_data_policy must be one of: "
            "quality_first_shorter, backfill_to_target, fail_speaker."
        )
    if audio_codec != "pcm_s24le":
        raise ValueError("audio.speaker_samples.audio_codec must be 'pcm_s24le'.")
    min_slice_duration_ms = int(speaker_samples_cfg.get("min_slice_duration_ms", 1200))
    max_slices = int(speaker_samples_cfg.get("max_slices", 6))
    edge_fade_ms = int(speaker_samples_cfg.get("edge_fade_ms", 20))
    if min_slice_duration_ms <= 0:
        raise ValueError("audio.speaker_samples.min_slice_duration_ms must be > 0.")
    if max_slices <= 0:
        raise ValueError("audio.speaker_samples.max_slices must be > 0.")
    if edge_fade_ms < 0:
        raise ValueError("audio.speaker_samples.edge_fade_ms must be >= 0.")
    return SpeakerSampleGenerator(
        exporter=FfmpegSpeakerSampleExporter(),
        target_duration_seconds=int(
            speaker_samples_cfg.get("target_duration_seconds", 30)
        ),
        sample_rate_hz=int(speaker_samples_cfg.get("sample_rate_hz", 44100)),
        channels=int(speaker_samples_cfg.get("channels", 1)),
        edge_fade_ms=edge_fade_ms,
        audio_codec=cast(Literal["pcm_s24le"], audio_codec),
        min_slice_duration_ms=min_slice_duration_ms,
        max_slices=max_slices,
        selection_order=cast(
            Literal["chronological", "longest_first"],
            selection_order,
        ),
        low_data_policy=cast(
            Literal["quality_first_shorter", "backfill_to_target", "fail_speaker"],
            low_data_policy,
        ),
        clip_method=cast(Literal["concat_turns"], clip_method),
        short_speaker_policy=cast(
            Literal["export_shorter"],
            short_policy,
        ),
        manifest_filename=str(speaker_samples_cfg.get("manifest_filename", "manifest.json")),
    )


def build_voice_clone_orchestrator(
    audio_cfg: Mapping[str, Any],
    *,
    project_root: Path,
) -> VoiceCloneOrchestrator | None:
    """
    Build post-summary voice-cloning orchestrator from config mapping.

    Args:
        audio_cfg: ``audio`` config section.
        project_root: Repository root used for path resolution.

    Returns:
        Wired ``VoiceCloneOrchestrator`` when enabled, otherwise ``None``.
    """
    voice_clone_cfg = _as_mapping(audio_cfg.get("voice_clone", {}))
    if not bool(voice_clone_cfg.get("enabled", False)):
        return None

    work_dir = resolve_path(
        str(audio_cfg.get("work_dir", "artifacts/audio_stage")),
        base_dir=project_root,
    )
    output_dir = resolve_path(
        str(voice_clone_cfg.get("output_dir_name", "voice_clone")),
        base_dir=work_dir,
    )
    provider = build_voice_clone_provider(audio_cfg, project_root=project_root)
    return VoiceCloneOrchestrator(
        provider=provider,
        output_dir=output_dir,
        fail_on_error=bool(voice_clone_cfg.get("fail_on_error", True)),
        manifest_filename=str(voice_clone_cfg.get("manifest_filename", "manifest.json")),
        merge_segments=bool(voice_clone_cfg.get("merge_segments", True)),
        merged_output_filename=str(
            voice_clone_cfg.get("merged_output_filename", "voice_cloned.wav")
        ),
        merge_timeout_seconds=int(voice_clone_cfg.get("merge_timeout_seconds", 300)),
        emo_preset_catalog={
            str(name).strip(): str(emo_text).strip()
            for name, emo_text in _as_mapping(voice_clone_cfg.get("emo_presets", {})).items()
            if str(name).strip() and str(emo_text).strip()
        },
    )


def build_interjector_orchestrator(
    audio_cfg: Mapping[str, Any],
    *,
    llm: LLMProvider,
    project_root: Path,
) -> InterjectorOrchestrator | None:
    """
    Build Stage-4 interjector orchestrator from audio config mapping.

    Args:
        audio_cfg: ``audio`` config section.
        llm: LLM provider used for interjection planning.
        project_root: Repository root used for path resolution.

    Returns:
        Wired ``InterjectorOrchestrator`` when enabled, otherwise ``None``.
    """
    interjector_cfg = _as_mapping(audio_cfg.get("interjector", {}))
    if not bool(interjector_cfg.get("enabled", False)):
        return None

    work_dir = resolve_path(
        str(audio_cfg.get("work_dir", "artifacts/audio_stage")),
        base_dir=project_root,
    )
    output_dir_name = str(interjector_cfg.get("output_dir_name", "interjector"))
    if not output_dir_name.strip():
        raise ValueError("audio.interjector.output_dir_name must be non-empty.")
    output_dir = resolve_path(output_dir_name, base_dir=work_dir)
    max_interjection_words = int(interjector_cfg.get("max_interjection_words", 5))
    return InterjectorOrchestrator(
        planner=LLMInterjectionPlanner(
            llm=llm,
            max_tokens=int(interjector_cfg.get("analysis_max_tokens", 900)),
            max_interjection_words=max_interjection_words,
        ),
        provider=build_voice_clone_provider(audio_cfg, project_root=project_root),
        output_dir=output_dir,
        manifest_filename=str(
            interjector_cfg.get("manifest_filename", "interjector_manifest.json")
        ),
        merged_output_filename=str(
            interjector_cfg.get(
                "merged_output_filename",
                "voice_cloned_interjected.wav",
            )
        ),
        max_interjection_words=max_interjection_words,
        max_interjections_per_host_turn=int(
            interjector_cfg.get("max_interjections_per_host_turn", 1)
        ),
        backchannel_reaction_latency_ms=int(
            interjector_cfg.get("backchannel_reaction_latency_ms", 120)
        ),
        echo_alignment_offset_ms=int(
            interjector_cfg.get("echo_alignment_offset_ms", 20)
        ),
        min_host_progress_ratio=float(
            interjector_cfg.get("min_host_progress_ratio", 0.35)
        ),
        max_host_progress_ratio=float(
            interjector_cfg.get("max_host_progress_ratio", 0.90)
        ),
        min_available_overlap_ms=int(
            interjector_cfg.get("min_available_overlap_ms", 120)
        ),
        turn_end_guard_ms=int(interjector_cfg.get("turn_end_guard_ms", 180)),
        mix_timeout_seconds=int(interjector_cfg.get("mix_timeout_seconds", 300)),
    )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Coerce nested config values into plain mapping."""
    if isinstance(value, Mapping):
        return value
    return {}


def _optional_int(value: Any) -> int | None:
    """Return int when configured, otherwise None."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return int(value)


def _build_separator(
    *,
    separation_cfg: Mapping[str, Any],
    retry_cfg: Mapping[str, Any],
) -> SourceSeparator:
    """Factory method for source-separation strategy."""
    provider = str(separation_cfg.get("provider", "demucs")).strip().lower()
    if provider == "demucs":
        return DemucsSourceSeparator(
            model_name=str(separation_cfg.get("model", "htdemucs")),
            timeout_seconds=int(separation_cfg.get("timeout_seconds", 1800)),
            max_retries=int(retry_cfg.get("attempts", 2)),
            retry_base_seconds=float(retry_cfg.get("base_delay_seconds", 1.0)),
        )
    if provider in {"passthrough", "none"}:
        return PassthroughSourceSeparator()
    raise ValueError(f"Unsupported separation provider: {provider}")


def _build_transcriber(*, asr_cfg: Mapping[str, Any]) -> SpeechTranscriber:
    """Factory method for ASR strategy."""
    provider = str(asr_cfg.get("provider", "faster_whisper")).strip().lower()
    if provider == "faster_whisper":
        forced_alignment_cfg = _as_mapping(asr_cfg.get("forced_alignment", {}))
        return FasterWhisperTranscriber(
            model_name=str(asr_cfg.get("model", "large-v3")),
            compute_type_cuda=str(asr_cfg.get("compute_type_cuda", "int8_float16")),
            compute_type_cpu=str(asr_cfg.get("compute_type_cpu", "int8")),
            beam_size=int(asr_cfg.get("beam_size", 5)),
            vad_filter=bool(asr_cfg.get("vad_filter", True)),
            batch_size=int(asr_cfg.get("batch_size", 8)),
            language=(
                str(asr_cfg.get("language", "")).strip() or None
            ),
            suppress_numerals=bool(asr_cfg.get("suppress_numerals", False)),
            enable_forced_alignment=bool(
                forced_alignment_cfg.get("enabled", True)
            ),
            require_forced_alignment=bool(
                forced_alignment_cfg.get("required", True)
            ),
            forced_alignment_batch_size=int(
                forced_alignment_cfg.get("batch_size", 8)
            ),
        )
    raise ValueError(f"Unsupported ASR provider: {provider}")


def _build_diarizer(*, diarization_cfg: Mapping[str, Any]) -> SpeakerDiarizer:
    """Factory method for diarization strategy."""
    provider = str(diarization_cfg.get("provider", "nemo")).strip().lower()
    if provider == "nemo":
        return NemoSpeakerDiarizer(
            msdd_model=str(diarization_cfg.get("nemo_model", "diar_msdd_telephonic")),
            vad_model=str(
                diarization_cfg.get("vad_model", "vad_multilingual_marblenet")
            ),
            speaker_embedding_model=str(
                diarization_cfg.get("speaker_embedding_model", "titanet_large")
            ),
            min_speakers=_optional_int(diarization_cfg.get("min_speakers")),
            max_speakers=_optional_int(diarization_cfg.get("max_speakers")),
            allow_single_speaker_fallback=bool(
                diarization_cfg.get("allow_single_speaker_fallback", False)
            ),
        )
    if provider in {"single_speaker", "none"}:
        return SingleSpeakerDiarizer(
            speaker_label=str(diarization_cfg.get("fallback_speaker", "SPEAKER_00"))
        )
    raise ValueError(f"Unsupported diarization provider: {provider}")


def _build_eta_strategy(*, eta_cfg: Mapping[str, Any]) -> StageEtaStrategy:
    """Factory method for stage ETA strategy."""
    defaults = default_stage_eta_strategy()
    adaptive_cfg = _as_mapping(eta_cfg.get("adaptive", {}))
    learning_rate = float(adaptive_cfg.get("learning_rate", defaults.learning_rate))
    min_multiplier = float(adaptive_cfg.get("min_multiplier", defaults.min_multiplier))
    max_multiplier = float(adaptive_cfg.get("max_multiplier", defaults.max_multiplier))
    _validate_eta_adaptive_config(
        learning_rate=learning_rate,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
    )

    stage_multipliers_cfg = _as_mapping(eta_cfg.get("stage_multipliers", {}))
    unit_defaults_cfg = _as_mapping(eta_cfg.get("unit_bootstrap_seconds_per_unit", {}))
    separation_cfg = _as_mapping(stage_multipliers_cfg.get("separation", {}))
    transcription_cfg = _as_mapping(stage_multipliers_cfg.get("transcription", {}))
    diarization_cfg = _as_mapping(stage_multipliers_cfg.get("diarization", {}))
    unit_defaults = _resolve_unit_stage_defaults(
        configured=unit_defaults_cfg,
        defaults=defaults.unit_stage_defaults,
    )

    return LinearStageEtaStrategy(
        separation=_resolve_stage_profile(
            stage_cfg=separation_cfg,
            default_profile=defaults.separation,
        ),
        transcription=_resolve_stage_profile(
            stage_cfg=transcription_cfg,
            default_profile=defaults.transcription,
        ),
        diarization=_resolve_stage_profile(
            stage_cfg=diarization_cfg,
            default_profile=defaults.diarization,
        ),
        learning_rate=learning_rate,
        min_multiplier=min_multiplier,
        max_multiplier=max_multiplier,
        unit_stage_defaults=unit_defaults,
    )


def _resolve_stage_profile(
    *,
    stage_cfg: Mapping[str, Any],
    default_profile: StageSpeedProfile,
) -> StageSpeedProfile:
    """Parse stage ETA multipliers while preserving defaults."""
    return StageSpeedProfile(
        cpu=float(stage_cfg.get("cpu", default_profile.cpu)),
        cuda=float(stage_cfg.get("cuda", default_profile.cuda)),
    )


def _validate_eta_adaptive_config(
    *,
    learning_rate: float,
    min_multiplier: float,
    max_multiplier: float,
) -> None:
    """Validate adaptive ETA config bounds."""
    if not 0.0 <= learning_rate <= 1.0:
        raise ValueError("audio.eta.adaptive.learning_rate must be within [0.0, 1.0].")
    if min_multiplier <= 0:
        raise ValueError("audio.eta.adaptive.min_multiplier must be > 0.")
    if max_multiplier < min_multiplier:
        raise ValueError(
            "audio.eta.adaptive.max_multiplier must be >= min_multiplier."
        )


def _resolve_unit_stage_defaults(
    *,
    configured: Mapping[str, Any],
    defaults: Mapping[EtaUnitStageName, float],
) -> dict[EtaUnitStageName, float]:
    """Resolve baseline seconds-per-unit defaults for unit stages."""
    resolved: dict[EtaUnitStageName, float] = {}
    for stage_name in ("speaker_samples", "voice_clone"):
        fallback = float(defaults[stage_name])
        resolved[stage_name] = float(configured.get(stage_name, fallback))
        if resolved[stage_name] <= 0:
            raise ValueError(
                f"audio.eta.unit_bootstrap_seconds_per_unit.{stage_name} must be > 0."
            )
    return resolved


def _validate_eta_dynamic_config(
    *,
    progress_smoothing: float,
    overrun_factor: float,
    headroom_seconds: float,
) -> None:
    """Validate runtime dynamic ETA tuning parameters."""
    if not 0.0 < progress_smoothing <= 1.0:
        raise ValueError("audio.eta.dynamic.progress_smoothing must be within (0.0, 1.0].")
    if overrun_factor <= 1.0:
        raise ValueError("audio.eta.dynamic.overrun_factor must be > 1.0.")
    if headroom_seconds < 0:
        raise ValueError("audio.eta.dynamic.headroom_seconds must be >= 0.")


def build_voice_clone_provider(
    audio_cfg: Mapping[str, Any],
    *,
    project_root: Path,
) -> VoiceCloneProvider:
    """Factory method for voice-cloning strategy providers."""
    voice_clone_cfg = _as_mapping(audio_cfg.get("voice_clone", {}))
    provider = str(voice_clone_cfg.get("provider", "indextts")).strip().lower()
    if provider == "indextts":
        execution_backend = str(
            voice_clone_cfg.get("execution_backend", "subprocess")
        )
        runner_project_dir = resolve_path(
            str(voice_clone_cfg.get("runner_project_dir", "third_party/index_tts")),
            base_dir=project_root,
        )
        cfg_default = "checkpoints/config.yaml"
        model_default = "checkpoints"
        if execution_backend.strip().lower() == "subprocess":
            cfg_default = "third_party/index_tts/checkpoints/config.yaml"
            model_default = "third_party/index_tts/checkpoints"
        cfg_path = resolve_path(
            str(voice_clone_cfg.get("cfg_path", cfg_default)),
            base_dir=project_root,
        )
        model_dir = resolve_path(
            str(voice_clone_cfg.get("model_dir", model_default)),
            base_dir=project_root,
        )
        return IndexTTSVoiceCloneGateway(
            cfg_path=cfg_path,
            model_dir=model_dir,
            device=resolve_device(str(voice_clone_cfg.get("device", "auto"))),
            use_fp16=bool(voice_clone_cfg.get("use_fp16", False)),
            use_cuda_kernel=bool(voice_clone_cfg.get("use_cuda_kernel", False)),
            use_deepspeed=bool(voice_clone_cfg.get("use_deepspeed", False)),
            use_accel=bool(voice_clone_cfg.get("use_accel", False)),
            use_torch_compile=bool(voice_clone_cfg.get("use_torch_compile", False)),
            verbose=bool(voice_clone_cfg.get("verbose", False)),
            max_text_tokens_per_segment=int(
                voice_clone_cfg.get("max_text_tokens_per_segment", 120)
            ),
            do_sample=bool(voice_clone_cfg.get("do_sample", True)),
            top_p=float(voice_clone_cfg.get("top_p", 0.8)),
            top_k=int(voice_clone_cfg.get("top_k", 30)),
            temperature=float(voice_clone_cfg.get("temperature", 0.8)),
            length_penalty=float(voice_clone_cfg.get("length_penalty", 0.0)),
            num_beams=int(voice_clone_cfg.get("num_beams", 1)),
            repetition_penalty=float(
                voice_clone_cfg.get("repetition_penalty", 10.0)
            ),
            max_mel_tokens=int(voice_clone_cfg.get("max_mel_tokens", 1500)),
            execution_backend=execution_backend,
            runner_project_dir=runner_project_dir,
            uv_executable=str(voice_clone_cfg.get("uv_executable", "uv")),
            stream_subprocess_output=bool(
                voice_clone_cfg.get("stream_subprocess_output", True)
            ),
        )
    if provider in {"passthrough", "none"}:
        return PassthroughVoiceCloneGateway()
    raise ValueError(f"Unsupported voice clone provider: {provider}")
