"""Tests for transcript affect-analysis orchestration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from card_framework.audio_pipeline.emotion.affect_analysis import (
    AffectPresetMappingConfig,
    AffectProfilingConfig,
    AffectScore,
    SegmentAffectUpdate,
    TranscriptAffectAnalyzer,
)
from card_framework.orchestration.transcript import Transcript


class _FakeCategoricalProvider:
    """Deterministic categorical affect provider for analyzer tests."""

    provider_name = "fake_categorical"
    device = "cpu"

    def analyze(self, clips):
        assert len(clips) == 2
        return [
            SegmentAffectUpdate(
                segment_index=0,
                emotion_scores=(
                    AffectScore(label="angry", score=0.75),
                    AffectScore(label="neutral", score=0.20),
                ),
            ),
            SegmentAffectUpdate(
                segment_index=1,
                emotion_scores=(
                    AffectScore(label="happy", score=0.65),
                    AffectScore(label="neutral", score=0.25),
                ),
                embedding_dimension=1024,
            ),
        ]

    def release_cached_resources(self) -> None:
        return None


class _FakeDimensionalProvider:
    """Deterministic dimensional affect provider for analyzer tests."""

    provider_name = "fake_dimensional"
    device = "cpu"

    def analyze(self, clips):
        assert len(clips) == 2
        return [
            SegmentAffectUpdate(
                segment_index=0,
                arousal=0.82,
                dominance=0.66,
                valence=0.22,
            ),
            SegmentAffectUpdate(
                segment_index=1,
                arousal=0.30,
                dominance=0.40,
                valence=0.78,
            ),
        ]

    def release_cached_resources(self) -> None:
        return None


def test_transcript_affect_analyzer_enriches_transcript_and_writes_manifest(
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "input.wav"
    sample_rate_hz = 16_000
    waveform = np.concatenate(
        [
            np.full(sample_rate_hz, 0.1, dtype=np.float32),
            np.full(sample_rate_hz, -0.1, dtype=np.float32),
        ]
    )
    sf.write(str(audio_path), waveform, sample_rate_hz)
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "start_time": 0,
                    "end_time": 900,
                    "text": "First segment.",
                },
                {
                    "speaker": "SPEAKER_01",
                    "start_time": 1000,
                    "end_time": 1800,
                    "text": "Second segment.",
                },
            ],
            "metadata": {"source_audio_path": str(audio_path)},
        }
    )
    analyzer = TranscriptAffectAnalyzer(
        output_dir=tmp_path / "affect_analysis",
        configured_audio_path="",
        source_audio_mode="auto",
        target_sample_rate_hz=sample_rate_hz,
        context_padding_ms=0,
        min_segment_duration_ms=200,
        prompt_max_hint_chars=160,
        top_emotions_for_hint=2,
        verbose_logging=False,
        mapping=AffectPresetMappingConfig(
            allowed_presets=("neutral", "warm", "engaged", "upbeat", "urgent"),
        ),
        profiling=AffectProfilingConfig(enabled=False),
        categorical_provider=_FakeCategoricalProvider(),
        dimensional_provider=_FakeDimensionalProvider(),
    )

    result = analyzer.analyze_transcript(transcript=transcript, base_dir=tmp_path)

    first_extras = result.transcript.segments[0].extras
    second_extras = result.transcript.segments[1].extras
    assert first_extras["suggested_emo_preset"] == "urgent"
    assert second_extras["suggested_emo_preset"] == "upbeat"
    assert "affect_hint" in first_extras
    assert "angry" in first_extras["affect_hint"]
    assert first_extras["emotion_vector"][1] > 0.0
    assert first_extras["emotion_vector_source"] == "affect_analysis"
    assert first_extras["emotion_vector_confidence"] >= 0.15
    assert result.manifest_path.is_file()
    assert result.profile_path is None
    assert result.transcript.metadata["affect_analysis_manifest_path"] == str(
        result.manifest_path
    )
    assert result.transcript.metadata["affect_analysis_source_audio_path"] == str(
        audio_path
    )

