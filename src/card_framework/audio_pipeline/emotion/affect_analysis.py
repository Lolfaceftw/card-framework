"""Transcript-affect analysis orchestration and contracts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from card_framework.audio_pipeline.emotion.affect_profiling import (
    ResourceProfile,
    ResourceUsageMonitor,
)
from card_framework.audio_pipeline.emotion.emotion_vectors import (
    EmotionVector,
    INDEX_TTS_EMOTION_ORDER,
)
from card_framework.audio_pipeline.errors import NonRetryableAudioStageError
from card_framework.audio_pipeline.runtime import utc_now_iso
from card_framework.audio_pipeline.speaker_samples import resolve_sample_source_audio_path
from card_framework.orchestration.transcript import Transcript, TranscriptSegment
from card_framework.shared.events import event_bus
from card_framework.shared.perf import report_loading_duration


@dataclass(slots=True, frozen=True)
class AffectScore:
    """One scored categorical affect label."""

    label: str
    score: float

    def to_payload(self) -> dict[str, Any]:
        """Serialize this score as a JSON-friendly mapping."""
        return {
            "label": self.label,
            "score": round(float(self.score), 6),
        }


@dataclass(slots=True, frozen=True)
class AffectClip:
    """Prepared segment audio clip ready for model inference."""

    segment_index: int
    speaker: str
    text: str
    start_time_ms: int
    end_time_ms: int
    samples: np.ndarray
    sample_rate_hz: int


@dataclass(slots=True, frozen=True)
class SegmentAffectUpdate:
    """Partial affect outputs produced by one provider."""

    segment_index: int
    emotion_scores: tuple[AffectScore, ...] = ()
    arousal: float | None = None
    dominance: float | None = None
    valence: float | None = None
    embedding_dimension: int | None = None
    provider_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class SegmentAffectRecord:
    """Merged affect view for one transcript segment."""

    segment_index: int
    speaker: str
    text: str
    start_time_ms: int
    end_time_ms: int
    emotion_scores: tuple[AffectScore, ...] = ()
    arousal: float | None = None
    dominance: float | None = None
    valence: float | None = None
    embedding_dimension: int | None = None
    suggested_emo_preset: str = "neutral"
    hint: str = ""
    emotion_vector: EmotionVector | None = None
    emotion_vector_source: str = ""
    emotion_vector_confidence: float | None = None
    provider_details: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Serialize this record as a JSON-friendly mapping."""
        payload: dict[str, Any] = {
            "segment_index": self.segment_index,
            "speaker": self.speaker,
            "text": self.text,
            "start_time_ms": self.start_time_ms,
            "end_time_ms": self.end_time_ms,
            "emotion_scores": [score.to_payload() for score in self.emotion_scores],
            "suggested_emo_preset": self.suggested_emo_preset,
            "hint": self.hint,
            "provider_details": dict(self.provider_details),
        }
        if self.emotion_vector is not None:
            payload["emotion_vector"] = self.emotion_vector.to_list()
        if self.emotion_vector_source:
            payload["emotion_vector_source"] = self.emotion_vector_source
        if self.emotion_vector_confidence is not None:
            payload["emotion_vector_confidence"] = round(
                float(self.emotion_vector_confidence), 6
            )
        if self.arousal is not None:
            payload["arousal"] = round(float(self.arousal), 6)
        if self.dominance is not None:
            payload["dominance"] = round(float(self.dominance), 6)
        if self.valence is not None:
            payload["valence"] = round(float(self.valence), 6)
        if self.embedding_dimension is not None:
            payload["embedding_dimension"] = int(self.embedding_dimension)
        return payload


@dataclass(slots=True, frozen=True)
class AffectPresetMappingConfig:
    """Heuristic mapping from affect outputs into repo emotion presets."""

    allowed_presets: tuple[str, ...]
    default_preset: str = "neutral"
    urgent_emotions: tuple[str, ...] = ("angry", "fearful")
    upbeat_emotions: tuple[str, ...] = ("happy", "surprised")
    urgent_arousal_threshold: float = 0.65
    upbeat_valence_threshold: float = 0.60
    warm_valence_threshold: float = 0.48
    low_arousal_threshold: float = 0.45
    engaged_arousal_threshold: float = 0.45


@dataclass(slots=True, frozen=True)
class AffectProfilingConfig:
    """Runtime profiling options for affect analysis."""

    enabled: bool = False
    sample_interval_seconds: float = 0.2
    command_timeout_seconds: float = 3.0
    profile_filename: str = "profile.json"


@dataclass(slots=True, frozen=True)
class EmotionVectorMappingConfig:
    """Heuristic mapping controls for IndexTTS emotion vectors."""

    enabled: bool = True
    categorical_weight: float = 0.65
    dimensional_weight: float = 0.35
    calm_floor: float = 0.12
    positive_surprise_boost: float = 0.35
    negative_anger_bias: float = 0.85
    negative_fear_bias: float = 0.9
    melancholic_bias: float = 0.7
    disgust_bias: float = 0.35
    confidence_floor: float = 0.15
    dominant_emotion_cap: float = 0.38
    max_sum: float = 0.8
    apply_indextts_bias: bool = True


@dataclass(slots=True, frozen=True)
class EmotionVectorMappingResult:
    """Structured result from affect-to-vector mapping."""

    vector: EmotionVector
    source: str
    confidence: float


@dataclass(slots=True, frozen=True)
class AffectAnalysisRunResult:
    """Completed transcript-affect analysis result bundle."""

    transcript: Transcript
    output_dir: Path
    manifest_path: Path
    profile_path: Path | None
    generated_at_utc: str
    source_audio_path: Path
    records: tuple[SegmentAffectRecord, ...]
    provider_profiles: dict[str, ResourceProfile]


@runtime_checkable
class SegmentAffectProvider(Protocol):
    """Provider interface for one affect-analysis backend."""

    @property
    def provider_name(self) -> str:
        """Return a stable provider identifier."""

    @property
    def device(self) -> str:
        """Return the configured execution device."""

    def analyze(self, clips: Sequence[AffectClip]) -> list[SegmentAffectUpdate]:
        """Return per-segment affect updates."""

    def release_cached_resources(self) -> None:
        """Release cached resources after one run."""


@runtime_checkable
class AffectEmotionVectorMapper(Protocol):
    """Strategy interface for affect-to-emotion-vector conversion."""

    def map(
        self,
        *,
        emotion_scores: Sequence[AffectScore],
        arousal: float | None,
        dominance: float | None,
        valence: float | None,
    ) -> EmotionVectorMappingResult | None:
        """Return one optional IndexTTS emotion vector."""


@dataclass(slots=True, frozen=True)
class IndexTtsEmotionVectorMapper:
    """Default heuristic mapper from affect outputs into IndexTTS vectors."""

    config: EmotionVectorMappingConfig

    def map(
        self,
        *,
        emotion_scores: Sequence[AffectScore],
        arousal: float | None,
        dominance: float | None,
        valence: float | None,
    ) -> EmotionVectorMappingResult | None:
        """Combine categorical and dimensional affect into a conservative vector."""
        if not self.config.enabled:
            return None
        vector_by_name = {name: 0.0 for name in INDEX_TTS_EMOTION_ORDER}
        confidence = 0.0
        for score in emotion_scores:
            mapped_name = _map_affect_label_to_indextts(score.label)
            if not mapped_name:
                continue
            vector_by_name[mapped_name] += float(score.score) * self.config.categorical_weight
            confidence = max(confidence, float(score.score))

        aro = _clamp_unit(arousal)
        dom = _clamp_unit(dominance)
        val = _clamp_unit(valence)
        if aro is not None and val is not None:
            pos = val
            neg = 1.0 - val
            vector_by_name["happy"] += (
                pos * (0.45 + 0.55 * aro) * self.config.dimensional_weight
            )
            vector_by_name["surprised"] += (
                pos
                * aro
                * self.config.positive_surprise_boost
                * self.config.dimensional_weight
            )
            vector_by_name["calm"] += (
                max(0.0, 1.0 - aro) * (0.3 + 0.7 * pos) * self.config.dimensional_weight
            )
            vector_by_name["melancholic"] += (
                neg
                * max(0.0, 1.0 - aro)
                * self.config.melancholic_bias
                * self.config.dimensional_weight
            )
            vector_by_name["sad"] += (
                neg
                * max(0.0, 1.0 - aro)
                * (0.45 + 0.35 * (1.0 - (dom if dom is not None else 0.5)))
                * self.config.dimensional_weight
            )
            resolved_dom = dom if dom is not None else 0.5
            vector_by_name["angry"] += (
                neg
                * aro
                * resolved_dom
                * self.config.negative_anger_bias
                * self.config.dimensional_weight
            )
            vector_by_name["afraid"] += (
                neg
                * aro
                * (1.0 - resolved_dom)
                * self.config.negative_fear_bias
                * self.config.dimensional_weight
            )
            vector_by_name["disgusted"] += (
                neg
                * (0.25 + 0.75 * resolved_dom)
                * self.config.disgust_bias
                * self.config.dimensional_weight
            )
            confidence = max(confidence, 0.5 + 0.5 * abs(0.5 - val))

        vector_by_name["calm"] += self.config.calm_floor
        values = [max(0.0, vector_by_name[name]) for name in INDEX_TTS_EMOTION_ORDER]
        max_value = max(values) if values else 0.0
        if max_value > self.config.dominant_emotion_cap > 0.0:
            scale = self.config.dominant_emotion_cap / max_value
            values = [value * scale for value in values]
        vector = EmotionVector(tuple(values)).with_index_tts_normalization(
            apply_bias=self.config.apply_indextts_bias,
            max_sum=self.config.max_sum,
        )
        resolved_confidence = max(self.config.confidence_floor, min(confidence, 1.0))
        return EmotionVectorMappingResult(
            vector=vector,
            source="affect_analysis",
            confidence=resolved_confidence,
        )


class TranscriptAffectAnalyzer:
    """Enrich a transcript with audio-grounded affect metadata."""

    def __init__(
        self,
        *,
        output_dir: Path,
        configured_audio_path: str,
        source_audio_mode: str,
        target_sample_rate_hz: int,
        context_padding_ms: int,
        min_segment_duration_ms: int,
        prompt_max_hint_chars: int,
        top_emotions_for_hint: int,
        verbose_logging: bool,
        mapping: AffectPresetMappingConfig,
        profiling: AffectProfilingConfig,
        emotion_vector_mapping: AffectEmotionVectorMapper | None = None,
        categorical_provider: SegmentAffectProvider | None = None,
        dimensional_provider: SegmentAffectProvider | None = None,
        manifest_filename: str = "manifest.json",
    ) -> None:
        if target_sample_rate_hz <= 0:
            raise ValueError("target_sample_rate_hz must be > 0.")
        if context_padding_ms < 0:
            raise ValueError("context_padding_ms must be >= 0.")
        if min_segment_duration_ms <= 0:
            raise ValueError("min_segment_duration_ms must be > 0.")
        if prompt_max_hint_chars <= 0:
            raise ValueError("prompt_max_hint_chars must be > 0.")
        if top_emotions_for_hint <= 0:
            raise ValueError("top_emotions_for_hint must be > 0.")
        if not manifest_filename.strip():
            raise ValueError("manifest_filename must be non-empty.")
        if categorical_provider is None and dimensional_provider is None:
            raise ValueError("At least one affect provider must be configured.")
        self.output_dir = output_dir.resolve()
        self.configured_audio_path = str(configured_audio_path or "").strip()
        self.source_audio_mode = str(source_audio_mode or "auto").strip().lower()
        self.target_sample_rate_hz = int(target_sample_rate_hz)
        self.context_padding_ms = int(context_padding_ms)
        self.min_segment_duration_ms = int(min_segment_duration_ms)
        self.prompt_max_hint_chars = int(prompt_max_hint_chars)
        self.top_emotions_for_hint = int(top_emotions_for_hint)
        self.verbose_logging = bool(verbose_logging)
        self.mapping = mapping
        self.emotion_vector_mapping = emotion_vector_mapping or IndexTtsEmotionVectorMapper(
            EmotionVectorMappingConfig()
        )
        self.profiling = profiling
        self.categorical_provider = categorical_provider
        self.dimensional_provider = dimensional_provider
        self.manifest_filename = manifest_filename.strip()

    def analyze_transcript(
        self,
        *,
        transcript: Transcript,
        base_dir: Path,
    ) -> AffectAnalysisRunResult:
        """Analyze transcript segments against source audio and persist a manifest."""
        self._emit(
            "Affect analysis: resolving source audio and preparing segment clips..."
        )
        source_audio_path = resolve_sample_source_audio_path(
            source_mode=self.source_audio_mode,
            transcript_metadata=transcript.metadata,
            configured_audio_path=self.configured_audio_path,
            base_dir=base_dir,
        )
        clips = self._build_clips(transcript=transcript, source_audio_path=source_audio_path)
        if not clips:
            raise NonRetryableAudioStageError(
                "Affect analysis could not build any usable audio clips from transcript segments."
            )
        self._emit(
            (
                "Affect analysis: prepared "
                f"{len(clips)} clip(s) from {source_audio_path}."
            )
        )

        provider_profiles: dict[str, ResourceProfile] = {}
        updates_by_segment: dict[int, SegmentAffectUpdate] = {}
        for provider in (self.categorical_provider, self.dimensional_provider):
            if provider is None:
                continue
            self._emit(
                (
                    "Affect analysis: running provider "
                    f"{provider.provider_name} on {len(clips)} clip(s) "
                    f"using device={provider.device}."
                )
            )
            updates, profile = self._run_provider(provider=provider, clips=clips)
            for update in updates:
                previous = updates_by_segment.get(update.segment_index)
                updates_by_segment[update.segment_index] = _merge_updates(previous, update)
            if profile is not None:
                provider_profiles[provider.provider_name] = profile

        records = tuple(
            self._build_record(
                clip=clip,
                update=updates_by_segment.get(clip.segment_index),
            )
            for clip in clips
        )
        preset_counts = Counter(record.suggested_emo_preset for record in records)
        if preset_counts:
            self._emit(
                "Affect analysis: suggested preset distribution "
                + ", ".join(
                    f"{preset}={count}"
                    for preset, count in sorted(preset_counts.items())
                )
            )
        generated_at_utc = utc_now_iso()
        manifest_path = (self.output_dir / self.manifest_filename).resolve()
        profile_path = self._write_profile_bundle(
            provider_profiles=provider_profiles,
            generated_at_utc=generated_at_utc,
            source_audio_path=source_audio_path,
            segment_count=len(records),
        )
        updated_transcript = self._apply_records_to_transcript(
            transcript=transcript,
            records=records,
            manifest_path=manifest_path,
            source_audio_path=source_audio_path,
        )
        metadata = dict(updated_transcript.metadata)
        metadata.update(
            {
                "affect_analysis_manifest_path": str(manifest_path),
                "affect_analysis_generated_at_utc": generated_at_utc,
                "affect_analysis_source_audio_path": str(source_audio_path),
            }
        )
        if profile_path is not None:
            metadata["affect_analysis_profile_path"] = str(profile_path)
        updated_transcript = updated_transcript.with_metadata(metadata)
        manifest_path = self._write_manifest(
            transcript=updated_transcript,
            source_audio_path=source_audio_path,
            records=records,
            provider_profiles=provider_profiles,
            generated_at_utc=generated_at_utc,
        )
        event_bus.publish(
            "status_message",
            (
                "Affect analysis complete for "
                f"{len(records)} segment(s); manifest written to {manifest_path}"
            ),
        )
        if profile_path is not None:
            self._emit(f"Affect analysis: resource profile written to {profile_path}")
        return AffectAnalysisRunResult(
            transcript=updated_transcript,
            output_dir=self.output_dir,
            manifest_path=manifest_path,
            profile_path=profile_path,
            generated_at_utc=generated_at_utc,
            source_audio_path=source_audio_path,
            records=records,
            provider_profiles=provider_profiles,
        )

    def _build_clips(
        self,
        *,
        transcript: Transcript,
        source_audio_path: Path,
    ) -> list[AffectClip]:
        """Load source audio once and slice it into segment clips."""
        try:
            import torch
            import torchaudio
        except Exception as exc:
            raise NonRetryableAudioStageError(
                "Affect analysis requires the 'torch' and 'torchaudio' runtimes."
            ) from exc

        with report_loading_duration(
            f"affect-analysis source audio '{source_audio_path.name}'",
            reporter=lambda message: event_bus.publish("system_message", message),
        ):
            try:
                waveform, sample_rate_hz = torchaudio.load(str(source_audio_path))
            except Exception as exc:
                raise NonRetryableAudioStageError(
                    f"Failed to load affect-analysis source audio: {source_audio_path}"
                ) from exc

        if waveform.ndim != 2 or waveform.shape[-1] == 0:
            raise NonRetryableAudioStageError(
                f"Affect-analysis source audio is empty: {source_audio_path}"
            )
        if waveform.shape[0] > 1:
            mono = waveform.mean(dim=0, keepdim=True)
        else:
            mono = waveform
        if int(sample_rate_hz) != self.target_sample_rate_hz:
            try:
                mono = torchaudio.functional.resample(
                    mono,
                    orig_freq=int(sample_rate_hz),
                    new_freq=self.target_sample_rate_hz,
                )
            except Exception as exc:
                raise NonRetryableAudioStageError(
                    "Failed to resample affect-analysis source audio."
                ) from exc
            sample_rate_hz = self.target_sample_rate_hz
        samples = mono.squeeze(0).detach().cpu()
        total_samples = int(samples.shape[-1])
        total_duration_ms = int(round((total_samples / float(sample_rate_hz)) * 1000.0))
        self._emit(
            (
                "Affect analysis: loaded source audio at "
                f"{sample_rate_hz} Hz with total duration {total_duration_ms} ms."
            )
        )

        clips: list[AffectClip] = []
        for index, segment in enumerate(transcript.segments):
            clip_start_ms, clip_end_ms = _expand_segment_bounds(
                start_time_ms=segment.start_time,
                end_time_ms=segment.end_time,
                context_padding_ms=self.context_padding_ms,
                min_duration_ms=self.min_segment_duration_ms,
                max_duration_ms=total_duration_ms,
            )
            start_sample = _milliseconds_to_samples(clip_start_ms, sample_rate_hz)
            end_sample = _milliseconds_to_samples(clip_end_ms, sample_rate_hz)
            if end_sample <= start_sample:
                continue
            clip_samples = samples[start_sample:end_sample]
            if int(clip_samples.numel()) <= 0:
                continue
            clips.append(
                AffectClip(
                    segment_index=index,
                    speaker=segment.speaker,
                    text=segment.text,
                    start_time_ms=segment.start_time,
                    end_time_ms=segment.end_time,
                    samples=clip_samples.numpy().astype(np.float32, copy=False),
                    sample_rate_hz=int(sample_rate_hz),
                )
            )
        del torch
        return clips

    def _run_provider(
        self,
        *,
        provider: SegmentAffectProvider,
        clips: Sequence[AffectClip],
    ) -> tuple[list[SegmentAffectUpdate], ResourceProfile | None]:
        """Run one provider with optional profiling."""
        monitor: ResourceUsageMonitor | None = None
        if self.profiling.enabled:
            monitor = ResourceUsageMonitor(
                pid=os.getpid(),
                device=provider.device,
                sample_interval_seconds=self.profiling.sample_interval_seconds,
                command_timeout_seconds=self.profiling.command_timeout_seconds,
            )
            monitor.start()
        updates: list[SegmentAffectUpdate] = []
        profile: ResourceProfile | None = None
        try:
            updates = provider.analyze(clips)
        except Exception as exc:
            if monitor is not None:
                profile = monitor.stop(label=provider.provider_name)
            try:
                provider.release_cached_resources()
            finally:
                raise exc.with_traceback(exc.__traceback__)
        try:
            provider.release_cached_resources()
        finally:
            profile = (
                monitor.stop(label=provider.provider_name)
                if monitor is not None
                else None
            )
        if profile is not None:
            self._emit("Affect analysis: " + _format_resource_profile_summary(profile))
        return updates, profile

    def _build_record(
        self,
        *,
        clip: AffectClip,
        update: SegmentAffectUpdate | None,
    ) -> SegmentAffectRecord:
        """Merge provider outputs into one prompt-friendly record."""
        emotion_scores = update.emotion_scores if update is not None else ()
        top_scores = tuple(sorted(emotion_scores, key=lambda item: item.score, reverse=True))
        arousal = update.arousal if update is not None else None
        dominance = update.dominance if update is not None else None
        valence = update.valence if update is not None else None
        suggested_emo_preset = _suggest_emo_preset(
            emotion_scores=top_scores,
            arousal=arousal,
            valence=valence,
            mapping=self.mapping,
        )
        emotion_vector_result = (
            self.emotion_vector_mapping.map(
                emotion_scores=top_scores,
                arousal=arousal,
                dominance=dominance,
                valence=valence,
            )
            if self.emotion_vector_mapping is not None
            else None
        )
        hint = _build_affect_hint(
            preset=suggested_emo_preset,
            emotion_scores=top_scores[: self.top_emotions_for_hint],
            arousal=arousal,
            dominance=dominance,
            valence=valence,
            max_chars=self.prompt_max_hint_chars,
            emotion_vector=(
                emotion_vector_result.vector
                if emotion_vector_result is not None
                else None
            ),
        )
        return SegmentAffectRecord(
            segment_index=clip.segment_index,
            speaker=clip.speaker,
            text=clip.text,
            start_time_ms=clip.start_time_ms,
            end_time_ms=clip.end_time_ms,
            emotion_scores=top_scores,
            arousal=arousal,
            dominance=dominance,
            valence=valence,
            embedding_dimension=(update.embedding_dimension if update is not None else None),
            suggested_emo_preset=suggested_emo_preset,
            hint=hint,
            emotion_vector=(
                emotion_vector_result.vector
                if emotion_vector_result is not None
                else None
            ),
            emotion_vector_source=(
                emotion_vector_result.source if emotion_vector_result is not None else ""
            ),
            emotion_vector_confidence=(
                emotion_vector_result.confidence
                if emotion_vector_result is not None
                else None
            ),
            provider_details=dict(update.provider_details) if update is not None else {},
        )

    def _apply_records_to_transcript(
        self,
        *,
        transcript: Transcript,
        records: Sequence[SegmentAffectRecord],
        manifest_path: Path,
        source_audio_path: Path,
    ) -> Transcript:
        """Write prompt-facing affect metadata back onto transcript segments."""
        records_by_index = {record.segment_index: record for record in records}
        updated_segments: list[TranscriptSegment] = []
        for index, segment in enumerate(transcript.segments):
            record = records_by_index.get(index)
            extras = dict(segment.extras)
            if record is not None:
                extras["affect"] = record.to_payload()
                extras["affect_hint"] = record.hint
                extras["suggested_emo_preset"] = record.suggested_emo_preset
                if record.emotion_vector is not None:
                    extras["emotion_vector"] = record.emotion_vector.to_list()
                if record.emotion_vector_source:
                    extras["emotion_vector_source"] = record.emotion_vector_source
                if record.emotion_vector_confidence is not None:
                    extras["emotion_vector_confidence"] = round(
                        float(record.emotion_vector_confidence),
                        6,
                    )
            updated_segments.append(
                TranscriptSegment(
                    speaker=segment.speaker,
                    text=segment.text,
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    extras=extras,
                )
            )
        metadata = dict(transcript.metadata)
        metadata.update(
            {
                "affect_analysis_manifest_path": str(manifest_path),
                "affect_analysis_source_audio_path": str(source_audio_path),
                "affect_analysis_segment_count": len(records),
            }
        )
        return Transcript(
            segments=tuple(updated_segments),
            metadata=metadata,
            extras=dict(transcript.extras),
        )

    def _write_manifest(
        self,
        *,
        transcript: Transcript,
        source_audio_path: Path,
        records: Sequence[SegmentAffectRecord],
        provider_profiles: Mapping[str, ResourceProfile],
        generated_at_utc: str,
    ) -> Path:
        """Persist one manifest describing the affect-analysis run."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = (self.output_dir / self.manifest_filename).resolve()
        payload = {
            "generated_at_utc": generated_at_utc,
            "source_audio_path": str(source_audio_path),
            "segment_count": len(records),
            "providers": sorted(
                name
                for name in (
                    self.categorical_provider.provider_name
                    if self.categorical_provider is not None
                    else None,
                    self.dimensional_provider.provider_name
                    if self.dimensional_provider is not None
                    else None,
                )
                if name
            ),
            "profiling_enabled": self.profiling.enabled,
            "provider_profiles": {
                name: profile.to_payload() for name, profile in provider_profiles.items()
            },
            "records": [record.to_payload() for record in records],
            "transcript_metadata": dict(transcript.metadata),
        }
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest_path

    def _write_profile_bundle(
        self,
        *,
        provider_profiles: Mapping[str, ResourceProfile],
        generated_at_utc: str,
        source_audio_path: Path,
        segment_count: int,
    ) -> Path | None:
        """Persist a combined resource-profile report when profiling is enabled."""
        if not self.profiling.enabled or not provider_profiles:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        profile_path = (self.output_dir / self.profiling.profile_filename).resolve()
        payload = {
            "generated_at_utc": generated_at_utc,
            "source_audio_path": str(source_audio_path),
            "segment_count": int(segment_count),
            "profiles": {
                name: profile.to_payload() for name, profile in provider_profiles.items()
            },
        }
        profile_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return profile_path

    def _emit(self, message: str) -> None:
        """Publish one verbose operator-facing affect-analysis log line."""
        if not self.verbose_logging:
            return
        event_bus.publish("system_message", message)


def _merge_updates(
    previous: SegmentAffectUpdate | None,
    current: SegmentAffectUpdate,
) -> SegmentAffectUpdate:
    """Merge two provider updates for the same segment."""
    if previous is None:
        return current
    return SegmentAffectUpdate(
        segment_index=current.segment_index,
        emotion_scores=(
            current.emotion_scores
            if current.emotion_scores
            else previous.emotion_scores
        ),
        arousal=current.arousal if current.arousal is not None else previous.arousal,
        dominance=(
            current.dominance if current.dominance is not None else previous.dominance
        ),
        valence=current.valence if current.valence is not None else previous.valence,
        embedding_dimension=(
            current.embedding_dimension
            if current.embedding_dimension is not None
            else previous.embedding_dimension
        ),
        provider_details={
            **dict(previous.provider_details),
            **dict(current.provider_details),
        },
    )


def _suggest_emo_preset(
    *,
    emotion_scores: Sequence[AffectScore],
    arousal: float | None,
    valence: float | None,
    mapping: AffectPresetMappingConfig,
) -> str:
    """Map affect scores into one configured voice-clone preset."""
    top_label = emotion_scores[0].label if emotion_scores else ""
    allowed = set(mapping.allowed_presets)
    fallback = (
        mapping.default_preset
        if mapping.default_preset in allowed
        else (mapping.allowed_presets[0] if mapping.allowed_presets else "neutral")
    )
    if (
        "urgent" in allowed
        and (
            top_label in mapping.urgent_emotions
            or (arousal is not None and arousal >= mapping.urgent_arousal_threshold)
        )
    ):
        return "urgent"
    if (
        "upbeat" in allowed
        and (
            top_label in mapping.upbeat_emotions
            or (valence is not None and valence >= mapping.upbeat_valence_threshold)
        )
    ):
        return "upbeat"
    if (
        "warm" in allowed
        and valence is not None
        and arousal is not None
        and valence >= mapping.warm_valence_threshold
        and arousal <= mapping.low_arousal_threshold
    ):
        return "warm"
    if "engaged" in allowed and arousal is not None and arousal >= mapping.engaged_arousal_threshold:
        return "engaged"
    return fallback


def _build_affect_hint(
    *,
    preset: str,
    emotion_scores: Sequence[AffectScore],
    arousal: float | None,
    dominance: float | None,
    valence: float | None,
    max_chars: int,
    emotion_vector: EmotionVector | None = None,
) -> str:
    """Render one compact prompt-facing affect hint."""
    parts = [f"preset={preset}"]
    if emotion_scores:
        emotion_text = ", ".join(
            f"{score.label}:{score.score:.2f}" for score in emotion_scores
        )
        parts.append(f"top={emotion_text}")
    avd_parts = [
        value
        for value in (
            f"a={arousal:.2f}" if arousal is not None else "",
            f"d={dominance:.2f}" if dominance is not None else "",
            f"v={valence:.2f}" if valence is not None else "",
        )
        if value
    ]
    if avd_parts:
        parts.append("avd=" + ", ".join(avd_parts))
    if emotion_vector is not None:
        parts.append(
            "emotion_vector="
            + ",".join(f"{value:.3f}" for value in emotion_vector.to_list())
        )
    hint = "; ".join(parts)
    if len(hint) <= max_chars:
        return hint
    return hint[: max(0, max_chars - 3)].rstrip() + "..."


def _clamp_unit(value: float | None) -> float | None:
    """Clamp one optional scalar into the inclusive unit interval."""
    if value is None:
        return None
    return max(0.0, min(float(value), 1.0))


def _map_affect_label_to_indextts(label: str) -> str:
    """Map provider-specific affect labels onto the IndexTTS emotion basis."""
    normalized = str(label).strip().lower()
    alias_map = {
        "happy": "happy",
        "joy": "happy",
        "excited": "happy",
        "angry": "angry",
        "anger": "angry",
        "annoyed": "angry",
        "sad": "sad",
        "sorrow": "sad",
        "fear": "afraid",
        "fearful": "afraid",
        "afraid": "afraid",
        "anxious": "afraid",
        "disgust": "disgusted",
        "disgusted": "disgusted",
        "melancholic": "melancholic",
        "melancholy": "melancholic",
        "depressed": "melancholic",
        "surprise": "surprised",
        "surprised": "surprised",
        "calm": "calm",
        "neutral": "calm",
        "content": "calm",
        "relaxed": "calm",
    }
    return alias_map.get(normalized, "")


def _format_resource_profile_summary(profile: ResourceProfile) -> str:
    """Render one compact operator-facing resource profile summary."""
    parts = [
        f"provider={profile.label}",
        f"elapsed={profile.elapsed_seconds:.3f}s",
        f"samples={profile.sample_count}",
    ]
    if profile.peak_rss_mb is not None:
        parts.append(f"peak_rss_mb={profile.peak_rss_mb:.1f}")
    if profile.peak_gpu_dedicated_mb is not None:
        parts.append(f"peak_gpu_dedicated_mb={profile.peak_gpu_dedicated_mb:.1f}")
    if profile.peak_gpu_shared_mb is not None:
        parts.append(f"peak_gpu_shared_mb={profile.peak_gpu_shared_mb:.1f}")
    if profile.torch_peak_allocated_mb is not None:
        parts.append(f"torch_peak_allocated_mb={profile.torch_peak_allocated_mb:.1f}")
    if profile.torch_peak_reserved_mb is not None:
        parts.append(f"torch_peak_reserved_mb={profile.torch_peak_reserved_mb:.1f}")
    return "; ".join(parts)


def _milliseconds_to_samples(milliseconds: int, sample_rate_hz: int) -> int:
    """Convert one millisecond timestamp into a sample index."""
    return int(round((milliseconds / 1000.0) * sample_rate_hz))


def _expand_segment_bounds(
    *,
    start_time_ms: int,
    end_time_ms: int,
    context_padding_ms: int,
    min_duration_ms: int,
    max_duration_ms: int,
) -> tuple[int, int]:
    """Expand one segment boundary with padding and a minimum duration floor."""
    clip_start_ms = max(0, start_time_ms - context_padding_ms)
    clip_end_ms = min(max_duration_ms, end_time_ms + context_padding_ms)
    if clip_end_ms - clip_start_ms >= min_duration_ms:
        return clip_start_ms, clip_end_ms
    shortfall = min_duration_ms - (clip_end_ms - clip_start_ms)
    expand_before = shortfall // 2
    expand_after = shortfall - expand_before
    clip_start_ms = max(0, clip_start_ms - expand_before)
    clip_end_ms = min(max_duration_ms, clip_end_ms + expand_after)
    if clip_end_ms - clip_start_ms >= min_duration_ms:
        return clip_start_ms, clip_end_ms
    if clip_start_ms == 0:
        clip_end_ms = min(max_duration_ms, clip_start_ms + min_duration_ms)
    else:
        clip_start_ms = max(0, clip_end_ms - min_duration_ms)
    return clip_start_ms, clip_end_ms

