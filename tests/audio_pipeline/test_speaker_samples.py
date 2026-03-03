import json
from pathlib import Path

import pytest

from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.speaker_samples import (
    AudioSlice,
    SpeakerSampleGenerator,
    build_speaker_sample_plans,
    resolve_sample_source_audio_path,
)


def test_build_speaker_sample_plans_truncates_to_target_and_marks_shorter() -> None:
    transcript_payload = {
        "segments": [
            {"speaker": "SPEAKER_00", "start_time": 0, "end_time": 20_000, "text": "a"},
            {"speaker": "SPEAKER_01", "start_time": 5_000, "end_time": 15_000, "text": "b"},
            {
                "speaker": "SPEAKER_00",
                "start_time": 25_000,
                "end_time": 45_000,
                "text": "c",
            },
        ]
    }

    plans = build_speaker_sample_plans(
        transcript_payload=transcript_payload,
        target_duration_ms=30_000,
    )

    assert [plan.speaker for plan in plans] == ["SPEAKER_00", "SPEAKER_01"]
    assert plans[0].actual_duration_ms == 30_000
    assert plans[0].status == "ok"
    assert plans[0].slices == (
        AudioSlice(start_time_ms=0, end_time_ms=20_000),
        AudioSlice(start_time_ms=25_000, end_time_ms=35_000),
    )
    assert plans[1].actual_duration_ms == 10_000
    assert plans[1].status == "shorter_than_requested"
    assert plans[1].slices == (AudioSlice(start_time_ms=5_000, end_time_ms=15_000),)


def test_speaker_sample_generator_writes_manifest_and_audio_files(tmp_path: Path) -> None:
    class _StubExporter:
        def export(
            self,
            *,
            source_audio_path: Path,
            slices: list[AudioSlice] | tuple[AudioSlice, ...],
            output_path: Path,
            sample_rate_hz: int,
            channels: int,
        ) -> None:
            del source_audio_path, slices, sample_rate_hz, channels
            output_path.write_bytes(b"wav")

    source_audio_path = tmp_path / "vocals.wav"
    source_audio_path.write_bytes(b"source")
    transcript_payload = {
        "segments": [
            {
                "speaker": "SPEAKER/00",
                "start_time": 0,
                "end_time": 10_000,
                "text": "hello",
            },
            {
                "speaker": "SPEAKER_00",
                "start_time": 11_000,
                "end_time": 21_000,
                "text": "world",
            },
        ]
    }
    generator = SpeakerSampleGenerator(
        exporter=_StubExporter(),
        target_duration_seconds=30,
        manifest_filename="samples.manifest.json",
    )

    result = generator.generate(
        transcript_payload=transcript_payload,
        source_audio_path=source_audio_path,
        output_dir=tmp_path / "speaker_samples",
    )

    assert result.manifest_path.exists()
    assert len(result.artifacts) == 2
    assert all(artifact.path.exists() for artifact in result.artifacts)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["target_duration_ms"] == 30_000
    assert manifest["clip_method"] == "concat_turns"
    assert manifest["short_speaker_policy"] == "export_shorter"
    assert len(manifest["samples"]) == 2


def test_resolve_sample_source_audio_path_supports_modes_and_fallback(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"source")
    vocals = tmp_path / "vocals.wav"
    vocals.write_bytes(b"vocals")

    metadata = {
        "source_audio_path": str(source),
        "vocals_audio_path": str(vocals),
    }

    resolved_vocals = resolve_sample_source_audio_path(
        source_mode="vocals",
        transcript_metadata=metadata,
        configured_audio_path="",
        base_dir=tmp_path,
    )
    assert resolved_vocals == vocals

    resolved_source = resolve_sample_source_audio_path(
        source_mode="source",
        transcript_metadata={"source_audio_path": str(source)},
        configured_audio_path="",
        base_dir=tmp_path,
    )
    assert resolved_source == source

    fallback_source = resolve_sample_source_audio_path(
        source_mode="source",
        transcript_metadata={},
        configured_audio_path=str(source),
        base_dir=tmp_path,
    )
    assert fallback_source == source

    with pytest.raises(NonRetryableAudioStageError):
        resolve_sample_source_audio_path(
            source_mode="vocals",
            transcript_metadata={},
            configured_audio_path="",
            base_dir=tmp_path,
        )


def test_speaker_sample_generator_propagates_export_failure_without_manifest(
    tmp_path: Path,
) -> None:
    class _FailingExporter:
        def export(
            self,
            *,
            source_audio_path: Path,
            slices: list[AudioSlice] | tuple[AudioSlice, ...],
            output_path: Path,
            sample_rate_hz: int,
            channels: int,
        ) -> None:
            del source_audio_path, slices, output_path, sample_rate_hz, channels
            raise NonRetryableAudioStageError("Failed to export speaker sample with ffmpeg")

    source_audio_path = tmp_path / "vocals.wav"
    source_audio_path.write_bytes(b"source")
    transcript_payload = {
        "segments": [
            {
                "speaker": "SPEAKER_00",
                "start_time": 0,
                "end_time": 10_000,
                "text": "hello",
            }
        ]
    }
    output_dir = tmp_path / "speaker_samples"
    generator = SpeakerSampleGenerator(
        exporter=_FailingExporter(),
        target_duration_seconds=30,
        manifest_filename="samples.manifest.json",
    )

    with pytest.raises(NonRetryableAudioStageError, match="Failed to export speaker sample"):
        generator.generate(
            transcript_payload=transcript_payload,
            source_audio_path=source_audio_path,
            output_dir=output_dir,
        )

    assert not (output_dir / "samples.manifest.json").exists()
    assert list(output_dir.glob("*.wav")) == []


def test_speaker_sample_generator_emits_progress_updates(tmp_path: Path) -> None:
    class _StubExporter:
        def export(
            self,
            *,
            source_audio_path: Path,
            slices: list[AudioSlice] | tuple[AudioSlice, ...],
            output_path: Path,
            sample_rate_hz: int,
            channels: int,
        ) -> None:
            del source_audio_path, slices, sample_rate_hz, channels
            output_path.write_bytes(b"wav")

    source_audio_path = tmp_path / "vocals.wav"
    source_audio_path.write_bytes(b"source")
    transcript_payload = {
        "segments": [
            {"speaker": "SPEAKER_00", "start_time": 0, "end_time": 10_000, "text": "hello"},
            {"speaker": "SPEAKER_01", "start_time": 10_000, "end_time": 20_000, "text": "world"},
        ]
    }
    updates: list[tuple[int | None, int | None]] = []
    generator = SpeakerSampleGenerator(exporter=_StubExporter(), target_duration_seconds=10)

    result = generator.generate(
        transcript_payload=transcript_payload,
        source_audio_path=source_audio_path,
        output_dir=tmp_path / "speaker_samples",
        progress_callback=lambda update: updates.append(
            (update.completed_units, update.total_units)
        ),
    )

    assert len(result.artifacts) == 2
    assert updates[0] == (0, 2)
    assert updates[-1] == (2, 2)
