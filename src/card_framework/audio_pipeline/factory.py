"""Composition root for audio pipeline adapters."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from card_framework.audio_pipeline.affect_analysis import (
    AffectPresetMappingConfig,
    AffectProfilingConfig,
    EmotionVectorMappingConfig,
    IndexTtsEmotionVectorMapper,
    TranscriptAffectAnalyzer,
)
from card_framework.audio_pipeline.audio_normalization import AudioNormalizationPolicy
from card_framework.audio_pipeline.contracts import SourceSeparator, SpeakerDiarizer, SpeechTranscriber
from card_framework.audio_pipeline.eta import (
    EtaUnitStageName,
    LinearStageEtaStrategy,
    StageEtaStrategy,
    StageSpeedProfile,
    default_stage_eta_strategy,
)
from card_framework.audio_pipeline.gateways.audeering_affect_gateway import (
    AudeeringMspDimAffectGateway,
)
from card_framework.audio_pipeline.gateways.demucs_gateway import DemucsSourceSeparator
from card_framework.audio_pipeline.gateways.emotion2vec_affect_gateway import (
    Emotion2VecPlusAffectGateway,
)
from card_framework.audio_pipeline.gateways.fallback_gateways import (
    PassthroughSourceSeparator,
    SingleSpeakerDiarizer,
)
from card_framework.audio_pipeline.gateways.faster_whisper_gateway import FasterWhisperTranscriber
from card_framework.audio_pipeline.gateways.granite_speech_gateway import GraniteSpeechTranscriber
from card_framework.audio_pipeline.gateways.nemo_diarizer_gateway import NemoSpeakerDiarizer
from card_framework.audio_pipeline.gateways.pyannote_diarizer_gateway import PyannoteSpeakerDiarizer
from card_framework.audio_pipeline.gateways.speaker_sample_gateway import FfmpegSpeakerSampleExporter
from card_framework.audio_pipeline.gateways.sortformer_diarizer_gateway import SortformerSpeakerDiarizer
from card_framework.audio_pipeline.interjector import (
    InterjectorOrchestrator,
    LLMInterjectionPlanner,
)
from card_framework.audio_pipeline.summary_turn_emotion_linker import (
    SummaryTurnEmotionLinker,
    SummaryTurnEmotionLinkingConfig,
)
from card_framework.audio_pipeline.gateways.fallback_voice_clone_gateway import (
    PassthroughVoiceCloneGateway,
)
from card_framework.audio_pipeline.gateways.indextts_voice_clone_gateway import (
    IndexTTSVoiceCloneGateway,
)
from card_framework.audio_pipeline.orchestrator import AudioToScriptOrchestrator
from card_framework.audio_pipeline.runtime import resolve_device, resolve_path
from card_framework.audio_pipeline.speaker_samples import SpeakerSampleGenerator
from card_framework.audio_pipeline.turn_emotion_resolution import (
    SourceLinkedTurnEmotionResolutionStrategy,
    TurnEmotionResolutionConfig,
)
from card_framework.audio_pipeline.voice_clone_contracts import VoiceCloneProvider
from card_framework.audio_pipeline.voice_clone_orchestrator import VoiceCloneOrchestrator
from card_framework.shared.llm_provider import LLMProvider


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
    diarizer = build_speaker_diarizer(audio_cfg)
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


def build_source_separator(audio_cfg: Mapping[str, Any]) -> SourceSeparator:
    """
    Build source separator from config mapping.

    Args:
        audio_cfg: ``audio`` config section.

    Returns:
        Wired ``SourceSeparator`` instance.
    """
    separation_cfg = _as_mapping(audio_cfg.get("separation", {}))
    retry_cfg = _as_mapping(audio_cfg.get("retry", {}))
    return _build_separator(
        separation_cfg=separation_cfg,
        retry_cfg=retry_cfg,
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


def build_speaker_diarizer(audio_cfg: Mapping[str, Any]) -> SpeakerDiarizer:
    """
    Build speaker diarizer from audio config mapping.

    Args:
        audio_cfg: ``audio`` config section.

    Returns:
        Wired ``SpeakerDiarizer`` instance.
    """
    diarization_cfg = _as_mapping(audio_cfg.get("diarization", {}))
    return _build_diarizer(diarization_cfg=diarization_cfg)


def build_transcript_affect_analyzer(
    audio_cfg: Mapping[str, Any],
    *,
    project_root: Path,
) -> TranscriptAffectAnalyzer | None:
    """Build post-transcript affect analyzer from audio config mapping."""
    affect_cfg = _as_mapping(audio_cfg.get("affect_analysis", {}))
    if not bool(affect_cfg.get("enabled", False)):
        return None

    work_dir = resolve_path(
        str(audio_cfg.get("work_dir", "artifacts/audio_stage")),
        base_dir=project_root,
    )
    output_dir_name = str(affect_cfg.get("output_dir_name", "affect_analysis")).strip()
    if not output_dir_name:
        raise ValueError("audio.affect_analysis.output_dir_name must be non-empty.")
    output_dir = resolve_path(output_dir_name, base_dir=work_dir)

    categorical_cfg = _as_mapping(affect_cfg.get("categorical", {}))
    dimensional_cfg = _as_mapping(affect_cfg.get("dimensional", {}))
    categorical_provider = _build_categorical_affect_provider(categorical_cfg)
    dimensional_provider = _build_dimensional_affect_provider(dimensional_cfg)
    if categorical_provider is None and dimensional_provider is None:
        raise ValueError(
            "audio.affect_analysis.enabled=true requires at least one enabled provider."
        )

    voice_clone_cfg = _as_mapping(audio_cfg.get("voice_clone", {}))
    mapping_cfg = _as_mapping(affect_cfg.get("mapping", {}))
    profiling_cfg = _as_mapping(affect_cfg.get("profiling", {}))
    return TranscriptAffectAnalyzer(
        output_dir=output_dir,
        configured_audio_path=str(audio_cfg.get("audio_path", "")),
        source_audio_mode=str(affect_cfg.get("source_audio", "auto")),
        target_sample_rate_hz=int(affect_cfg.get("target_sample_rate_hz", 16000)),
        context_padding_ms=int(affect_cfg.get("context_padding_ms", 150)),
        min_segment_duration_ms=int(affect_cfg.get("min_segment_duration_ms", 1200)),
        prompt_max_hint_chars=int(affect_cfg.get("prompt_max_hint_chars", 160)),
        top_emotions_for_hint=int(affect_cfg.get("top_emotions_for_hint", 2)),
        verbose_logging=bool(affect_cfg.get("verbose_logging", True)),
        mapping=AffectPresetMappingConfig(
            allowed_presets=_resolve_emo_preset_names(voice_clone_cfg),
            default_preset=str(mapping_cfg.get("default_preset", "neutral")).strip()
            or "neutral",
            urgent_emotions=tuple(
                str(value).strip().lower()
                for value in list(mapping_cfg.get("urgent_emotions", ["angry", "fearful"]))
                if str(value).strip()
            ),
            upbeat_emotions=tuple(
                str(value).strip().lower()
                for value in list(mapping_cfg.get("upbeat_emotions", ["happy", "surprised"]))
                if str(value).strip()
            ),
            urgent_arousal_threshold=float(
                mapping_cfg.get("urgent_arousal_threshold", 0.65)
            ),
            upbeat_valence_threshold=float(
                mapping_cfg.get("upbeat_valence_threshold", 0.60)
            ),
            warm_valence_threshold=float(
                mapping_cfg.get("warm_valence_threshold", 0.48)
            ),
            low_arousal_threshold=float(mapping_cfg.get("low_arousal_threshold", 0.45)),
            engaged_arousal_threshold=float(
                mapping_cfg.get("engaged_arousal_threshold", 0.45)
            ),
        ),
        emotion_vector_mapping=IndexTtsEmotionVectorMapper(
            EmotionVectorMappingConfig(
                enabled=bool(mapping_cfg.get("emotion_vectors_enabled", True)),
                categorical_weight=float(mapping_cfg.get("categorical_weight", 0.65)),
                dimensional_weight=float(mapping_cfg.get("dimensional_weight", 0.35)),
                calm_floor=float(mapping_cfg.get("calm_floor", 0.12)),
                positive_surprise_boost=float(
                    mapping_cfg.get("positive_surprise_boost", 0.35)
                ),
                negative_anger_bias=float(
                    mapping_cfg.get("negative_anger_bias", 0.85)
                ),
                negative_fear_bias=float(mapping_cfg.get("negative_fear_bias", 0.9)),
                melancholic_bias=float(mapping_cfg.get("melancholic_bias", 0.7)),
                disgust_bias=float(mapping_cfg.get("disgust_bias", 0.35)),
                confidence_floor=float(mapping_cfg.get("confidence_floor", 0.15)),
                dominant_emotion_cap=float(
                    mapping_cfg.get("dominant_emotion_cap", 0.38)
                ),
                max_sum=float(mapping_cfg.get("emotion_vector_max_sum", 0.8)),
                apply_indextts_bias=bool(
                    mapping_cfg.get("apply_indextts_emotion_bias", True)
                ),
            )
        ),
        profiling=AffectProfilingConfig(
            enabled=bool(profiling_cfg.get("enabled", False)),
            sample_interval_seconds=float(
                profiling_cfg.get("sample_interval_seconds", 0.2)
            ),
            command_timeout_seconds=float(
                profiling_cfg.get("command_timeout_seconds", 3.0)
            ),
            profile_filename=str(profiling_cfg.get("profile_filename", "profile.json")),
        ),
        categorical_provider=categorical_provider,
        dimensional_provider=dimensional_provider,
        manifest_filename=str(affect_cfg.get("manifest_filename", "manifest.json")),
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
        merged_output_normalization=_build_audio_normalization_policy(
            _as_mapping(voice_clone_cfg.get("normalization", {}))
        ),
    )


def build_summary_turn_emotion_linker(
    audio_cfg: Mapping[str, Any],
) -> SummaryTurnEmotionLinker | None:
    """Build the optional source-grounded summary-turn emotion linker."""
    linking_cfg = _as_mapping(audio_cfg.get("summary_turn_linking", {}))
    if not bool(linking_cfg.get("enabled", False)):
        return None

    voice_clone_cfg = _as_mapping(audio_cfg.get("voice_clone", {}))
    return SummaryTurnEmotionLinker(
        SummaryTurnEmotionLinkingConfig(
            enabled=True,
            top_k=int(linking_cfg.get("top_k", 3)),
            min_support_score=float(linking_cfg.get("min_support_score", 0.08)),
            same_speaker_bonus=float(linking_cfg.get("same_speaker_bonus", 1.2)),
            default_emotion_confidence=float(
                linking_cfg.get("default_emotion_confidence", 0.5)
            ),
            min_emotion_confidence=float(
                linking_cfg.get("min_emotion_confidence", 0.15)
            ),
            lexical_similarity_weight=float(
                linking_cfg.get("lexical_similarity_weight", 0.45)
            ),
            coverage_similarity_weight=float(
                linking_cfg.get("coverage_similarity_weight", 0.25)
            ),
            sequence_similarity_weight=float(
                linking_cfg.get("sequence_similarity_weight", 0.15)
            ),
            position_similarity_weight=float(
                linking_cfg.get("position_similarity_weight", 0.15)
            ),
            min_sequence_similarity_without_overlap=float(
                linking_cfg.get("min_sequence_similarity_without_overlap", 0.6)
            ),
            require_emotion_vector=bool(
                linking_cfg.get("require_emotion_vector", True)
            ),
            apply_indextts_bias=bool(
                linking_cfg.get("apply_indextts_bias", False)
            ),
            max_sum=float(
                linking_cfg.get(
                    "max_sum",
                    voice_clone_cfg.get("emotion_vector_max_sum", 0.8),
                )
            ),
            preserve_existing_explicit_vectors=bool(
                linking_cfg.get("preserve_existing_explicit_vectors", True)
            ),
            force_live_draft_rerender=bool(
                linking_cfg.get("force_live_draft_rerender", False)
            ),
        )
    )


def build_turn_emotion_resolution_strategy(
    audio_cfg: Mapping[str, Any],
) -> SourceLinkedTurnEmotionResolutionStrategy:
    """Build the live-draft turn-emotion resolution strategy from config."""
    summary_turn_emotion_cfg = _as_mapping(audio_cfg.get("summary_turn_emotion", {}))
    raw_fallback_order = summary_turn_emotion_cfg.get(
        "fallback_order",
        ["source_linked", "explicit"],
    )
    if isinstance(raw_fallback_order, str):
        fallback_order = tuple(
            step.strip()
            for step in raw_fallback_order.split(",")
            if step.strip()
        )
    else:
        fallback_order = tuple(
            str(step).strip()
            for step in list(raw_fallback_order)
            if str(step).strip()
        )
    return SourceLinkedTurnEmotionResolutionStrategy(
        config=TurnEmotionResolutionConfig(
            prelink_enabled=bool(
                summary_turn_emotion_cfg.get("prelink_enabled", True)
            ),
            strategy_name=str(
                summary_turn_emotion_cfg.get("prelink_strategy", "source_linked")
            ).strip()
            or "source_linked",
            fallback_order=fallback_order,
        ),
        linker=build_summary_turn_emotion_linker(audio_cfg),
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
    min_host_progress_ratio = float(
        interjector_cfg.get("min_host_progress_ratio", 0.35)
    )
    max_host_progress_ratio = float(
        interjector_cfg.get("max_host_progress_ratio", 0.90)
    )
    timing_refinement_cfg = _as_mapping(interjector_cfg.get("timing_refinement", {}))
    repair_cfg = _as_mapping(interjector_cfg.get("repair", {}))
    leading_silence_trim_cfg = _as_mapping(
        interjector_cfg.get("leading_silence_trim", {})
    )
    echo_pacing_fit_cfg = _as_mapping(interjector_cfg.get("echo_pacing_fit", {}))
    return InterjectorOrchestrator(
        planner=LLMInterjectionPlanner(
            llm=llm,
            max_tokens=int(interjector_cfg.get("analysis_max_tokens", 900)),
            repair_max_tokens=int(repair_cfg.get("prompt_max_tokens", 220)),
            max_interjection_words=max_interjection_words,
            min_host_progress_ratio=min_host_progress_ratio,
            max_host_progress_ratio=max_host_progress_ratio,
            prefer_tool_call_planning=bool(
                interjector_cfg.get("tool_call_planning_enabled", True)
            ),
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
            interjector_cfg.get("echo_alignment_offset_ms", 0)
        ),
        min_host_progress_ratio=min_host_progress_ratio,
        max_host_progress_ratio=max_host_progress_ratio,
        turn_end_guard_ms=int(interjector_cfg.get("turn_end_guard_ms", 180)),
        mix_timeout_seconds=int(interjector_cfg.get("mix_timeout_seconds", 300)),
        merged_output_normalization=_build_audio_normalization_policy(
            _as_mapping(interjector_cfg.get("normalization", {}))
        ),
        timing_refinement_enabled=bool(
            timing_refinement_cfg.get("enabled", True)
        ),
        fallback_anchor_realignment_enabled=bool(
            timing_refinement_cfg.get("fallback_anchor_realignment_enabled", True)
        ),
        repair_enabled=bool(repair_cfg.get("enabled", True)),
        repair_max_attempts=_optional_int(repair_cfg.get("max_attempts")),
        trim_leading_silence_enabled=bool(
            leading_silence_trim_cfg.get("enabled", True)
        ),
        trim_leading_silence_threshold_db=float(
            leading_silence_trim_cfg.get("threshold_db", -50.0)
        ),
        trim_leading_silence_min_duration_ms=int(
            leading_silence_trim_cfg.get("min_duration_ms", 0)
        ),
        trim_leading_silence_preserve_ms=int(
            leading_silence_trim_cfg.get("preserve_ms", 5)
        ),
        echo_pacing_fit_enabled=bool(
            echo_pacing_fit_cfg.get("enabled", True)
        ),
        echo_pacing_timing_backend=str(
            echo_pacing_fit_cfg.get("timing_backend", "faster_whisper_native")
        ),
        echo_pacing_model_name=str(
            echo_pacing_fit_cfg.get("model_name", "tiny.en")
        ),
        echo_pacing_device=str(
            echo_pacing_fit_cfg.get("device", "cpu")
        ),
        echo_pacing_compute_type_cpu=str(
            echo_pacing_fit_cfg.get("compute_type_cpu", "int8")
        ),
        echo_pacing_compute_type_cuda=str(
            echo_pacing_fit_cfg.get("compute_type_cuda", "int8_float16")
        ),
        echo_pacing_beam_size=int(
            echo_pacing_fit_cfg.get("beam_size", 1)
        ),
        echo_pacing_target_host_fill_ratio=float(
            echo_pacing_fit_cfg.get("target_host_fill_ratio", 0.95)
        ),
        echo_pacing_min_overflow_ratio=float(
            echo_pacing_fit_cfg.get("min_overflow_ratio", 1.05)
        ),
        echo_pacing_max_tempo=float(
            echo_pacing_fit_cfg.get("max_tempo", 1.75)
        ),
        echo_pacing_min_phrase_match_similarity=float(
            echo_pacing_fit_cfg.get("min_phrase_match_similarity", 0.75)
        ),
        echo_pacing_max_window_delta=int(
            echo_pacing_fit_cfg.get("max_window_delta", 1)
        ),
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


def _optional_float(value: Any) -> float | None:
    """Return float when configured, otherwise None."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return float(value)


def _configured_float(value: Any, *, default: float) -> float:
    """Return configured float or a supplied default when omitted."""
    parsed = _optional_float(value)
    if parsed is None:
        return default
    return parsed


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
            window_length_seconds=int(
                separation_cfg.get("window_length_seconds", 600)
            ),
            window_context_seconds=int(
                separation_cfg.get("window_context_seconds", 5)
            ),
        )
    if provider in {"passthrough", "none"}:
        return PassthroughSourceSeparator()
    raise ValueError(f"Unsupported separation provider: {provider}")


def _build_transcriber(*, asr_cfg: Mapping[str, Any]) -> SpeechTranscriber:
    """Factory method for ASR strategy."""
    provider = str(asr_cfg.get("provider", "granite_speech")).strip().lower()
    forced_alignment_cfg = _as_mapping(asr_cfg.get("forced_alignment", {}))
    if provider == "granite_speech":
        return GraniteSpeechTranscriber(
            model_name=str(
                asr_cfg.get("model", "ibm-granite/granite-4.0-1b-speech")
            ),
            language=(str(asr_cfg.get("language", "en")).strip() or "en"),
            chunk_length_seconds=int(asr_cfg.get("chunk_length_seconds", 25)),
            chunk_overlap_seconds=int(asr_cfg.get("chunk_overlap_seconds", 1)),
            max_new_tokens=int(asr_cfg.get("max_new_tokens", 96)),
            repetition_penalty=_configured_float(
                asr_cfg.get("repetition_penalty"),
                default=1.25,
            ),
            enable_forced_alignment=bool(forced_alignment_cfg.get("enabled", True)),
            require_forced_alignment=bool(forced_alignment_cfg.get("required", True)),
            forced_alignment_batch_size=int(forced_alignment_cfg.get("batch_size", 8)),
        )
    if provider == "faster_whisper":
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
    pyannote_cfg = _as_mapping(diarization_cfg.get("pyannote", {}))
    sortformer_offline_cfg = _as_mapping(diarization_cfg.get("sortformer_offline", {}))
    sortformer_streaming_cfg = _as_mapping(
        diarization_cfg.get("sortformer_streaming", {})
    )
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
            max_speakers=_optional_int(diarization_cfg.get("max_speakers", 10)),
            allow_single_speaker_fallback=bool(
                diarization_cfg.get("allow_single_speaker_fallback", False)
            ),
            vad_onset=_configured_float(diarization_cfg.get("vad_onset"), default=0.8),
            vad_offset=_configured_float(diarization_cfg.get("vad_offset"), default=0.6),
            vad_pad_onset=_configured_float(
                diarization_cfg.get("vad_pad_onset"),
                default=0.1,
            ),
            vad_pad_offset=_configured_float(
                diarization_cfg.get("vad_pad_offset"),
                default=0.0,
            ),
            vad_min_duration_on=_configured_float(
                diarization_cfg.get("vad_min_duration_on"),
                default=0.2,
            ),
            vad_min_duration_off=_configured_float(
                diarization_cfg.get("vad_min_duration_off"),
                default=0.6,
            ),
            clustering_max_rp_threshold=_configured_float(
                diarization_cfg.get("clustering_max_rp_threshold"),
                default=0.4,
            ),
            clustering_sparse_search_volume=int(
                diarization_cfg.get("clustering_sparse_search_volume", 60)
            ),
            clustering_maj_vote_spk_count=bool(
                diarization_cfg.get("clustering_maj_vote_spk_count", True)
            ),
        )
    if provider in {"pyannote", "pyannote_community1"}:
        return PyannoteSpeakerDiarizer(
            pipeline_name=str(
                pyannote_cfg.get(
                    "pipeline_name",
                    "pyannote/speaker-diarization-community-1",
                )
            ),
            auth_token=(
                str(pyannote_cfg.get("auth_token", "")).strip() or None
            ),
            auth_token_env=str(
                pyannote_cfg.get("auth_token_env", "HUGGINGFACE_TOKEN")
            ),
            use_exclusive_diarization=bool(
                pyannote_cfg.get("use_exclusive_diarization", True)
            ),
            min_speakers=_optional_int(diarization_cfg.get("min_speakers")),
            max_speakers=_optional_int(diarization_cfg.get("max_speakers")),
        )
    if provider in {"nemo_sortformer_offline", "sortformer_offline"}:
        return SortformerSpeakerDiarizer(
            model_name=str(
                sortformer_offline_cfg.get(
                    "model_name",
                    "nvidia/diar_sortformer_4spk-v1",
                )
            ),
            checkpoint_path=(
                str(sortformer_offline_cfg.get("checkpoint_path", "")).strip()
                or None
            ),
            streaming_mode=False,
            batch_size=int(sortformer_offline_cfg.get("batch_size", 1)),
        )
    if provider in {"nemo_sortformer_streaming", "sortformer_streaming"}:
        return SortformerSpeakerDiarizer(
            model_name=str(
                sortformer_streaming_cfg.get(
                    "model_name",
                    "nvidia/diar_streaming_sortformer_4spk-v2",
                )
            ),
            checkpoint_path=(
                str(sortformer_streaming_cfg.get("checkpoint_path", "")).strip()
                or None
            ),
            streaming_mode=True,
            batch_size=int(sortformer_streaming_cfg.get("batch_size", 1)),
            chunk_len=int(sortformer_streaming_cfg.get("chunk_len", 340)),
            chunk_right_context=int(
                sortformer_streaming_cfg.get("chunk_right_context", 40)
            ),
            fifo_len=int(sortformer_streaming_cfg.get("fifo_len", 40)),
            spkcache_update_period=int(
                sortformer_streaming_cfg.get("spkcache_update_period", 300)
            ),
            spkcache_len=int(sortformer_streaming_cfg.get("spkcache_len", 188)),
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


def _build_categorical_affect_provider(
    affect_cfg: Mapping[str, Any],
) -> Emotion2VecPlusAffectGateway | None:
    """Build the configured categorical affect provider when enabled."""
    if not bool(affect_cfg.get("enabled", True)):
        return None
    provider = str(affect_cfg.get("provider", "emotion2vec_plus")).strip().lower()
    resolved_device = resolve_device(str(affect_cfg.get("device", "auto")))
    if provider == "emotion2vec_plus":
        return Emotion2VecPlusAffectGateway(
            model_name=str(affect_cfg.get("model_name", "emotion2vec_plus_large")),
            device=resolved_device,
            batch_size=int(affect_cfg.get("batch_size", 4)),
            hub=str(affect_cfg.get("hub", "hf")),
            granularity=str(affect_cfg.get("granularity", "utterance")),
            extract_embedding=bool(affect_cfg.get("extract_embedding", True)),
            disable_update_check=bool(affect_cfg.get("disable_update_check", True)),
        )
    raise ValueError(
        "audio.affect_analysis.categorical.provider must be 'emotion2vec_plus'."
    )


def _build_dimensional_affect_provider(
    affect_cfg: Mapping[str, Any],
) -> AudeeringMspDimAffectGateway | None:
    """Build the configured dimensional affect provider when enabled."""
    if not bool(affect_cfg.get("enabled", True)):
        return None
    provider = str(affect_cfg.get("provider", "audeering_msp_dim")).strip().lower()
    resolved_device = resolve_device(str(affect_cfg.get("device", "auto")))
    if provider == "audeering_msp_dim":
        return AudeeringMspDimAffectGateway(
            model_name=str(
                affect_cfg.get(
                    "model_name",
                    "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                )
            ),
            device=resolved_device,
            use_fp16=bool(affect_cfg.get("use_fp16", True)),
            batch_size=int(affect_cfg.get("batch_size", 4)),
        )
    raise ValueError(
        "audio.affect_analysis.dimensional.provider must be 'audeering_msp_dim'."
    )


def _resolve_emo_preset_names(voice_clone_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """Resolve the configured voice-clone preset names for affect mapping."""
    raw_catalog = _as_mapping(voice_clone_cfg.get("emo_presets", {}))
    presets = tuple(
        str(name).strip()
        for name, emo_text in raw_catalog.items()
        if str(name).strip() and str(emo_text).strip()
    )
    if presets:
        return presets
    return ("neutral", "warm", "engaged", "upbeat", "urgent")


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
        default_runner_project_dir = _default_vendor_index_tts_dir(project_root)
        runner_project_dir = resolve_path(
            str(
                voice_clone_cfg.get(
                    "runner_project_dir",
                    str(default_runner_project_dir),
                )
            ),
            base_dir=project_root,
        )
        default_checkpoints_dir = _default_index_tts_checkpoints_dir(
            project_root=project_root,
            runner_project_dir=runner_project_dir,
        )
        cfg_default = str(default_checkpoints_dir / "config.yaml")
        model_default = str(default_checkpoints_dir)
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
            normalize_emotion_vector=bool(
                voice_clone_cfg.get("normalize_emotion_vector", True)
            ),
            emotion_vector_apply_bias=bool(
                voice_clone_cfg.get("emotion_vector_apply_bias", True)
            ),
            emotion_vector_max_sum=float(
                voice_clone_cfg.get("emotion_vector_max_sum", 0.8)
            ),
        )
    if provider in {"passthrough", "none"}:
        return PassthroughVoiceCloneGateway()
    raise ValueError(f"Unsupported voice clone provider: {provider}")


def _build_audio_normalization_policy(
    normalization_cfg: Mapping[str, Any],
) -> AudioNormalizationPolicy | None:
    """Build one optional audio-normalization policy from config."""
    if not normalization_cfg:
        return None
    return AudioNormalizationPolicy(
        enabled=bool(normalization_cfg.get("enabled", False)),
        target_lufs=float(normalization_cfg.get("target_lufs", -16.0)),
        true_peak_db=float(normalization_cfg.get("true_peak_db", -1.5)),
        loudness_range_lu=float(normalization_cfg.get("loudness_range_lu", 11.0)),
        linear=bool(normalization_cfg.get("linear", True)),
        dual_mono=bool(normalization_cfg.get("dual_mono", False)),
        timeout_seconds=int(normalization_cfg.get("timeout_seconds", 300)),
    )


def _default_vendor_index_tts_dir(project_root: Path) -> Path:
    """Return the default vendored IndexTTS source directory for one project root."""
    return (
        project_root / "src" / "card_framework" / "_vendor" / "index_tts"
    ).resolve()


def _default_index_tts_checkpoints_dir(
    *,
    project_root: Path,
    runner_project_dir: Path,
) -> Path:
    """Return the default IndexTTS checkpoints directory for one project root."""
    del runner_project_dir
    return (project_root / "checkpoints" / "index_tts").resolve()

