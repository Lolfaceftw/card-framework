from __future__ import annotations

import json
from pathlib import Path

import pytest

from card_framework.audio_pipeline.speaker_samples import (
    AudioSlice,
    SpeakerSampleArtifact,
    SpeakerSampleGenerationResult,
)
from card_framework.benchmark.summarizer_critic.sporc.sampled_segment_speaker_prep import (
    _match_sampled_file_to_manifest,
    _parse_sampled_segment_file,
    _resolve_bootstrap_speaker_mapping,
    SampledSegmentSpeakerPrepError,
    main,
)
from card_framework.orchestration.transcript import Transcript


def _write_manifest(tmp_path: Path, *, source_audio_path: Path) -> Path:
    transcript_path = (tmp_path / "source.transcript.json").resolve()
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "text": "Hello there",
                        "start_time": 0,
                        "end_time": 2_000,
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "text": "General Kenobi",
                        "start_time": 2_500,
                        "end_time": 5_000,
                    },
                ],
                "metadata": {
                    "episode_title": "Cloud Atlas (2012)",
                    "source_audio_url": str(source_audio_path),
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = (tmp_path / "manifest.json").resolve()
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_sample",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(tmp_path / "speakerTurnData.jsonl.gz"),
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "sporc_00077_https_example_com_cloud_atlas_mp3",
                        "dataset": "sporc_full",
                        "episode_key": "https://example.com/cloud-atlas.mp3",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "A summary",
                        "target_seconds": 60,
                        "metadata": {
                            "episode_title": "Cloud Atlas (2012)",
                            "source_audio_url": str(source_audio_path),
                        },
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_parse_sampled_segment_file_extracts_headers(tmp_path: Path) -> None:
    sampled_path = (tmp_path / "ep1.txt").resolve()
    sampled_path.write_text(
        "Episode: Cloud Atlas (2012)\n"
        "Sample ID: sporc_00077_https_example_com_cloud_atlas_mp3\n\n"
        "1. [SPEAKER_00] Hello there.\n",
        encoding="utf-8",
    )

    parsed = _parse_sampled_segment_file(sampled_path)

    assert parsed.episode_title == "Cloud Atlas (2012)"
    assert parsed.sample_id == "sporc_00077_https_example_com_cloud_atlas_mp3"


def test_match_sampled_file_prefers_sample_id(tmp_path: Path) -> None:
    source_audio_path = (tmp_path / "source.wav").resolve()
    source_audio_path.write_bytes(b"RIFF")
    manifest_path = _write_manifest(tmp_path, source_audio_path=source_audio_path)
    sampled_path = (tmp_path / "ep1.txt").resolve()
    sampled_path.write_text(
        "Episode: Some Other Title\n"
        "Sample ID: sporc_00077_https_example_com_cloud_atlas_mp3\n",
        encoding="utf-8",
    )

    from card_framework.benchmark.summarizer_critic.sporc.sampled_segment_speaker_prep import (
        _load_manifest_matches,
    )

    match = _match_sampled_file_to_manifest(
        sampled_file=_parse_sampled_segment_file(sampled_path),
        manifest_matches=_load_manifest_matches(manifest_path),
    )

    assert match.sample.sample_id == "sporc_00077_https_example_com_cloud_atlas_mp3"


def test_match_sampled_file_recovers_from_unique_prefix_sample_id(
    tmp_path: Path,
) -> None:
    source_audio_path = (tmp_path / "source.wav").resolve()
    source_audio_path.write_bytes(b"RIFF")
    manifest_path = _write_manifest(tmp_path, source_audio_path=source_audio_path)
    sampled_path = (tmp_path / "ep1.txt").resolve()
    sampled_path.write_text(
        "Episode: Some Other Title\n"
        "Sample ID: sporc_00077_https_example_com_cloud_atlas\n",
        encoding="utf-8",
    )

    from card_framework.benchmark.summarizer_critic.sporc.sampled_segment_speaker_prep import (
        _load_manifest_matches,
    )

    match = _match_sampled_file_to_manifest(
        sampled_file=_parse_sampled_segment_file(sampled_path),
        manifest_matches=_load_manifest_matches(manifest_path),
    )

    assert match.sample.sample_id == "sporc_00077_https_example_com_cloud_atlas_mp3"


def test_resolve_bootstrap_speaker_mapping_explains_missing_clean_speaker() -> None:
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "hello",
                    "start_time": 0,
                    "end_time": 1_000,
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": "world",
                    "start_time": 1_000,
                    "end_time": 2_000,
                },
            ]
        }
    )
    bootstrap_transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "BOOT_00",
                    "text": "hello",
                    "start_time": 0,
                    "end_time": 1_000,
                }
            ]
        }
    )

    with pytest.raises(
        SampledSegmentSpeakerPrepError,
        match="overlap-safe sample extraction",
    ):
        _resolve_bootstrap_speaker_mapping(
            transcript=transcript,
            bootstrap_transcript=bootstrap_transcript,
        )


def test_main_prepares_episode_audio_and_speaker_samples(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_audio_path = (tmp_path / "source.wav").resolve()
    source_audio_path.write_bytes(b"RIFF")
    manifest_path = _write_manifest(tmp_path, source_audio_path=source_audio_path)
    segments_dir = (tmp_path / "sampled_segments").resolve()
    segments_dir.mkdir()
    sampled_path = (segments_dir / "ep1.txt").resolve()
    sampled_path.write_text(
        "Episode: Cloud Atlas (2012)\n"
        "Sample ID: sporc_00077_https_example_com_cloud_atlas_mp3\n",
        encoding="utf-8",
    )
    output_root = (segments_dir / "episodes").resolve()
    config_path = (tmp_path / "config.yaml").resolve()
    config_path.write_text("audio:\n  speaker_samples: {}\n", encoding="utf-8")

    def _fake_load_audio_config(project_root: Path, *, config_path: Path | None = None):
        del project_root, config_path
        return {"speaker_samples": {"target_duration_seconds": 30}}

    class _FakeBootstrapResult:
        def __init__(self, transcript_path: Path, output_dir: Path):
            sample_audio_path_0 = (output_dir / "speaker_00.wav").resolve()
            sample_audio_path_1 = (output_dir / "speaker_01.wav").resolve()
            manifest_path = (output_dir / "manifest.json").resolve()
            self.transcript_path = transcript_path.resolve()
            self.generation_result = SpeakerSampleGenerationResult(
                output_dir=output_dir.resolve(),
                manifest_path=manifest_path,
                generated_at_utc="2026-03-20T21:00:00Z",
                artifacts=(
                    SpeakerSampleArtifact(
                        speaker="BOOT_00",
                        path=sample_audio_path_0,
                        requested_duration_ms=30_000,
                        actual_duration_ms=2_000,
                        status="shorter_than_requested",
                        dropped_slice_count=0,
                        ranges_ms=(AudioSlice(start_time_ms=0, end_time_ms=2_000),),
                    ),
                    SpeakerSampleArtifact(
                        speaker="BOOT_01",
                        path=sample_audio_path_1,
                        requested_duration_ms=30_000,
                        actual_duration_ms=2_500,
                        status="shorter_than_requested",
                        dropped_slice_count=0,
                        ranges_ms=(AudioSlice(start_time_ms=2_500, end_time_ms=5_000),),
                    ),
                ),
            )

    def _fake_bootstrap_speaker_samples_from_diarization(
        *,
        project_root: Path,
        audio_cfg,
        audio_path: Path,
        bootstrap_transcript_path: Path,
        bootstrap_audio_work_dir: Path,
        speaker_samples_output_dir: Path,
        status_reporter=None,
    ):
        del project_root, audio_cfg, bootstrap_audio_work_dir, status_reporter
        speaker_samples_output_dir.mkdir(parents=True, exist_ok=True)
        sample_audio_path_0 = (speaker_samples_output_dir / "speaker_00.wav").resolve()
        sample_audio_path_1 = (speaker_samples_output_dir / "speaker_01.wav").resolve()
        sample_audio_path_0.write_bytes(audio_path.read_bytes())
        sample_audio_path_1.write_bytes(audio_path.read_bytes())
        manifest_path = (speaker_samples_output_dir / "manifest.json").resolve()
        manifest_path.write_text(
            json.dumps(
                {
                    "source_audio_path": str(audio_path),
                    "samples": [
                        {"speaker": "BOOT_00"},
                        {"speaker": "BOOT_01"},
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        bootstrap_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_transcript_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker": "BOOT_00",
                            "text": "Hello there",
                            "start_time": 0,
                            "end_time": 2_000,
                        },
                        {
                            "speaker": "BOOT_01",
                            "text": "General Kenobi",
                            "start_time": 2_500,
                            "end_time": 5_000,
                        },
                    ],
                    "metadata": {"source_audio_path": str(audio_path)},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return _FakeBootstrapResult(bootstrap_transcript_path, speaker_samples_output_dir)

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.sampled_segment_speaker_prep.load_audio_config",
        _fake_load_audio_config,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.sampled_segment_speaker_prep.bootstrap_speaker_samples_from_diarization",
        _fake_bootstrap_speaker_samples_from_diarization,
    )

    exit_code = main(
        [
            "--segments-dir",
            str(segments_dir),
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--config-file",
            str(config_path),
        ]
    )

    prepared_transcript_path = (
        output_root / "0001_cloud_atlas_2012" / "inputs" / "prepared.transcript.json"
    ).resolve()
    prepared_payload = json.loads(prepared_transcript_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert Path(prepared_payload["metadata"]["source_audio_path"]).exists()
    assert Path(
        prepared_payload["metadata"]["speaker_samples_manifest_path"]
    ).exists()
    assert prepared_payload["metadata"]["speaker_sample_bootstrap_transcript_path"]
    manifest_payload = json.loads(
        Path(prepared_payload["metadata"]["speaker_samples_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert [sample["speaker"] for sample in manifest_payload["samples"]] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert [sample["bootstrap_speaker"] for sample in manifest_payload["samples"]] == [
        "BOOT_00",
        "BOOT_01",
    ]
    assert (output_root / "speaker_sample_prep_report.json").exists()
