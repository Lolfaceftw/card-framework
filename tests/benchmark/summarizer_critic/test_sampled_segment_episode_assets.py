"""Coverage for sampled-segment episode asset preparation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets import (
    _build_episode_asset_plans,
    _parse_sampled_segment_file,
    _resolve_bootstrap_speaker_mapping,
    _selected_speaker_labels_from_paths,
    _ensure_sampled_speakers_have_clean_samples,
    main,
    SampledSegmentPreparationError,
    SampledSegmentFile,
)
from card_framework.orchestration.transcript import Transcript


def _write_manifest(tmp_path: Path) -> Path:
    transcript_path = tmp_path / "episode.transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0,
                        "end_time": 1200,
                        "text": "hello world",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 1200,
                        "end_time": 2400,
                        "text": "general kenobi",
                    },
                ],
                "metadata": {
                    "episode_title": "Episode Alpha",
                    "source_audio_url": "https://example.com/audio.mp3",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(tmp_path / "speakerTurnData.jsonl.gz"),
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "sporc_00001",
                        "dataset": "sporc_full",
                        "episode_key": "https://example.com/audio.mp3",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "Reference summary.",
                        "target_seconds": 45,
                        "metadata": {
                            "episode_title": "Episode Alpha",
                            "source_audio_url": "https://example.com/audio.mp3",
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest_path


def test_parse_sampled_segment_file_reads_headers(tmp_path: Path) -> None:
    segment_path = tmp_path / "ep1.txt"
    segment_path.write_text(
        "Episode: Episode Alpha\nSample ID: sporc_00001\n\n1. [SPEAKER_00] hello\n",
        encoding="utf-8",
    )

    parsed = _parse_sampled_segment_file(segment_path)

    assert parsed.episode_title == "Episode Alpha"
    assert parsed.sample_id == "sporc_00001"


def test_build_episode_asset_plans_groups_duplicate_sample_ids(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    segment_files = (
        SampledSegmentFile(
            path=(tmp_path / "ep1.txt").resolve(),
            episode_title="Episode Alpha",
            sample_id="sporc_00001",
        ),
        SampledSegmentFile(
            path=(tmp_path / "ep2.txt").resolve(),
            episode_title="Episode Alpha",
            sample_id="sporc_00001",
        ),
    )

    plans = _build_episode_asset_plans(
        manifest_path=manifest_path,
        segment_files=segment_files,
    )

    assert len(plans) == 1
    assert plans[0].sample.sample_id == "sporc_00001"
    assert plans[0].segment_files == (
        (tmp_path / "ep1.txt").resolve(),
        (tmp_path / "ep2.txt").resolve(),
    )


def test_build_episode_asset_plans_falls_back_to_unique_title_when_sample_id_misses(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    segment_files = (
        SampledSegmentFile(
            path=(tmp_path / "ep1.txt").resolve(),
            episode_title="Episode Alpha",
            sample_id="sporc_99999_truncated_or_stale",
        ),
    )

    plans = _build_episode_asset_plans(
        manifest_path=manifest_path,
        segment_files=segment_files,
    )

    assert len(plans) == 1
    assert plans[0].sample.sample_id == "sporc_00001"


def test_resolve_bootstrap_speaker_mapping_allows_partial_clean_speaker_coverage() -> None:
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
                    "text": "",
                    "start_time": 0,
                    "end_time": 1_000,
                }
            ]
        }
    )

    mapping = _resolve_bootstrap_speaker_mapping(
        transcript=transcript,
        bootstrap_transcript=bootstrap_transcript,
    )

    assert mapping == {"BOOT_00": "SPEAKER_00"}


def test_resolve_bootstrap_speaker_mapping_prioritizes_sampled_speakers() -> None:
    transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "SPEAKER_00",
                    "text": "host",
                    "start_time": 0,
                    "end_time": 100,
                },
                {
                    "speaker": "SPEAKER_01",
                    "text": "required sampled speaker",
                    "start_time": 100,
                    "end_time": 198,
                },
                {
                    "speaker": "SPEAKER_02",
                    "text": "extra speaker",
                    "start_time": 198,
                    "end_time": 297,
                },
            ]
        }
    )
    bootstrap_transcript = Transcript.from_mapping(
        {
            "segments": [
                {
                    "speaker": "BOOT_00",
                    "text": "",
                    "start_time": 0,
                    "end_time": 100,
                },
                {
                    "speaker": "BOOT_01",
                    "text": "",
                    "start_time": 100,
                    "end_time": 297,
                },
            ]
        }
    )

    mapping = _resolve_bootstrap_speaker_mapping(
        transcript=transcript,
        bootstrap_transcript=bootstrap_transcript,
        required_transcript_speakers=("SPEAKER_01",),
    )

    assert mapping == {
        "BOOT_00": "SPEAKER_00",
        "BOOT_01": "SPEAKER_01",
    }


def test_sampled_speaker_coverage_only_fails_for_missing_sampled_speakers(
    tmp_path: Path,
) -> None:
    segment_path = (tmp_path / "ep1.txt").resolve()
    segment_path.write_text(
        "Episode: Episode Alpha\n"
        "1. [SPEAKER_00] hello there\n"
        "2. [SPEAKER_00] general kenobi\n",
        encoding="utf-8",
    )

    sampled_speakers = _selected_speaker_labels_from_paths((segment_path,))
    _ensure_sampled_speakers_have_clean_samples(
        sampled_speakers=sampled_speakers,
        available_sample_speakers={"SPEAKER_00"},
    )

    with pytest.raises(
        SampledSegmentPreparationError,
        match="Missing sampled speakers",
    ):
        _ensure_sampled_speakers_have_clean_samples(
            sampled_speakers=("SPEAKER_00", "SPEAKER_01"),
            available_sample_speakers={"SPEAKER_00"},
        )


def test_main_prepares_episode_assets_and_summary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    segments_dir = tmp_path / "sampled_segments"
    segments_dir.mkdir()
    segment_path = segments_dir / "ep1.txt"
    segment_path.write_text(
        (
            "Episode: Episode Alpha\n"
            "Sample ID: sporc_00001\n\n"
            "1. [SPEAKER_00] this sampled text does not exist in the transcript\n"
        ),
        encoding="utf-8",
    )

    def fake_load_audio_config(project_root: Path, *, config_path: Path | None = None):
        del project_root, config_path
        return {
            "speaker_samples": {
                "target_duration_seconds": 30,
                "sample_rate_hz": 44100,
                "channels": 1,
                "edge_fade_ms": 20,
                "audio_codec": "pcm_s24le",
                "min_slice_duration_ms": 1000,
                "max_slices": 6,
                "selection_order": "chronological",
                "low_data_policy": "quality_first_shorter",
                "clip_method": "concat_turns",
                "short_speaker_policy": "export_shorter",
                "manifest_filename": "manifest.json",
            }
        }

    class FakeGenerationResult:
        def __init__(self, output_dir: Path):
            self.output_dir = output_dir
            self.manifest_path = output_dir / "manifest.json"
            self.generated_at_utc = "2026-03-20T00:00:00+00:00"
            self.artifacts = ("speaker0", "speaker1")

    class FakeBootstrapResult:
        def __init__(self, transcript_path: Path, output_dir: Path):
            self.transcript_path = transcript_path
            self.generation_result = FakeGenerationResult(output_dir)

    def fake_bootstrap_speaker_samples_from_diarization(
        *,
        project_root: Path,
        audio_cfg,
        audio_path: Path,
        bootstrap_transcript_path: Path,
        bootstrap_audio_work_dir: Path,
        speaker_samples_output_dir: Path,
        status_reporter=None,
    ):
        del project_root, bootstrap_audio_work_dir, status_reporter
        assert audio_cfg["speaker_samples"]["source_audio"] == "vocals"
        bootstrap_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_transcript_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker": "BOOT_00",
                            "start_time": 0,
                            "end_time": 1200,
                            "text": "hello world",
                        },
                        {
                            "speaker": "BOOT_01",
                            "start_time": 1200,
                            "end_time": 2400,
                            "text": "general kenobi",
                        },
                    ],
                    "metadata": {
                        "source_audio_path": str(audio_path),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        speaker_samples_output_dir.mkdir(parents=True, exist_ok=True)
        (speaker_samples_output_dir / "speaker_00.wav").write_bytes(b"RIFF")
        (speaker_samples_output_dir / "speaker_01.wav").write_bytes(b"RIFF")
        (speaker_samples_output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {"speaker": "BOOT_00"},
                        {"speaker": "BOOT_01"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return FakeBootstrapResult(bootstrap_transcript_path, speaker_samples_output_dir)

    def fake_download_binary_to_path(
        *,
        source_audio_url: str,
        destination: Path,
        force: bool,
    ) -> Path:
        del source_audio_url, force
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"ID3")
        return destination.resolve()

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets.load_audio_config",
        fake_load_audio_config,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets._download_binary_to_path",
        fake_download_binary_to_path,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets.bootstrap_speaker_samples_from_diarization",
        fake_bootstrap_speaker_samples_from_diarization,
    )

    output_root = tmp_path / "episode_assets"
    exit_code = main(
        [
            "--segments-dir",
            str(segments_dir),
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--config-file",
            str(tmp_path / "config.yaml"),
        ]
    )

    assert exit_code == 0
    summary_path = output_root / "prepared_episodes_manifest.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["episode_count"] == 1
    episode_payload = payload["episodes"][0]
    assert episode_payload["sample_id"] == "sporc_00001"
    assert Path(episode_payload["source_audio_path"]).exists()
    transcript_payload = json.loads(
        Path(episode_payload["transcript_path"]).read_text(encoding="utf-8")
    )
    assert (
        transcript_payload["metadata"]["speaker_samples_manifest_path"]
        == episode_payload["speaker_samples_manifest_path"]
    )
    manifest_payload = json.loads(
        Path(episode_payload["speaker_samples_manifest_path"]).read_text(
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
    assert transcript_payload["segments"][0]["text"] == (
        "this sampled text does not exist in the transcript"
    )
    assert transcript_payload["metadata"]["sampled_segments_files"] == [
        str(
            output_root
            / "sporc_00001__Episode_Alpha"
            / "sampled_segments"
            / "ep1.txt"
        )
    ]


def test_main_prefers_mapping_that_covers_sampled_speakers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    transcript_path = tmp_path / "episode.transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0,
                        "end_time": 100,
                        "text": "host",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 100,
                        "end_time": 198,
                        "text": "required sampled speaker",
                    },
                    {
                        "speaker": "SPEAKER_02",
                        "start_time": 198,
                        "end_time": 297,
                        "text": "extra speaker",
                    },
                ],
                "metadata": {
                    "episode_title": "Episode Beta",
                    "source_audio_url": "https://example.com/audio-beta.mp3",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(tmp_path / "speakerTurnData.jsonl.gz"),
                "sample_count": 1,
                "samples": [
                    {
                        "sample_id": "sporc_00002",
                        "dataset": "sporc_full",
                        "episode_key": "https://example.com/audio-beta.mp3",
                        "transcript_path": str(transcript_path),
                        "reference_summary": "Reference summary.",
                        "target_seconds": 45,
                        "metadata": {
                            "episode_title": "Episode Beta",
                            "source_audio_url": "https://example.com/audio-beta.mp3",
                        },
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    segments_dir = tmp_path / "sampled_segments"
    segments_dir.mkdir()
    segment_path = segments_dir / "ep1.txt"
    segment_path.write_text(
        (
            "Episode: Episode Beta\n"
            "Sample ID: sporc_00002\n\n"
            "1. [SPEAKER_01] sampled speaker must survive\n"
        ),
        encoding="utf-8",
    )

    def fake_load_audio_config(project_root: Path, *, config_path: Path | None = None):
        del project_root, config_path
        return {
            "speaker_samples": {
                "target_duration_seconds": 30,
                "sample_rate_hz": 44100,
                "channels": 1,
                "edge_fade_ms": 20,
                "audio_codec": "pcm_s24le",
                "min_slice_duration_ms": 1000,
                "max_slices": 6,
                "selection_order": "chronological",
                "low_data_policy": "quality_first_shorter",
                "clip_method": "concat_turns",
                "short_speaker_policy": "export_shorter",
                "manifest_filename": "manifest.json",
            }
        }

    class FakeGenerationResult:
        def __init__(self, output_dir: Path):
            self.output_dir = output_dir
            self.manifest_path = output_dir / "manifest.json"
            self.generated_at_utc = "2026-03-21T00:00:00+00:00"
            self.artifacts = ("speaker0", "speaker1")

    class FakeBootstrapResult:
        def __init__(self, transcript_path: Path, output_dir: Path):
            self.transcript_path = transcript_path
            self.generation_result = FakeGenerationResult(output_dir)

    def fake_bootstrap_speaker_samples_from_diarization(
        *,
        project_root: Path,
        audio_cfg,
        audio_path: Path,
        bootstrap_transcript_path: Path,
        bootstrap_audio_work_dir: Path,
        speaker_samples_output_dir: Path,
        status_reporter=None,
    ):
        del project_root, bootstrap_audio_work_dir, status_reporter
        assert audio_cfg["speaker_samples"]["source_audio"] == "vocals"
        bootstrap_transcript_path.parent.mkdir(parents=True, exist_ok=True)
        bootstrap_transcript_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker": "BOOT_00",
                            "start_time": 0,
                            "end_time": 100,
                            "text": "",
                        },
                        {
                            "speaker": "BOOT_01",
                            "start_time": 100,
                            "end_time": 297,
                            "text": "",
                        },
                    ],
                    "metadata": {
                        "source_audio_path": str(audio_path),
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        speaker_samples_output_dir.mkdir(parents=True, exist_ok=True)
        (speaker_samples_output_dir / "speaker_00.wav").write_bytes(b"RIFF")
        (speaker_samples_output_dir / "speaker_01.wav").write_bytes(b"RIFF")
        (speaker_samples_output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {"speaker": "BOOT_00"},
                        {"speaker": "BOOT_01"},
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return FakeBootstrapResult(bootstrap_transcript_path, speaker_samples_output_dir)

    def fake_download_binary_to_path(
        *,
        source_audio_url: str,
        destination: Path,
        force: bool,
    ) -> Path:
        del source_audio_url, force
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"ID3")
        return destination.resolve()

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets.load_audio_config",
        fake_load_audio_config,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets._download_binary_to_path",
        fake_download_binary_to_path,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.sampled_segments.prepare_episode_assets.bootstrap_speaker_samples_from_diarization",
        fake_bootstrap_speaker_samples_from_diarization,
    )

    output_root = tmp_path / "episode_assets"
    exit_code = main(
        [
            "--segments-dir",
            str(segments_dir),
            "--dataset-manifest",
            str(manifest_path),
            "--output-root",
            str(output_root),
            "--config-file",
            str(tmp_path / "config.yaml"),
        ]
    )

    assert exit_code == 0
    episode_payload = json.loads(
        (output_root / "prepared_episodes_manifest.json").read_text(encoding="utf-8")
    )["episodes"][0]
    manifest_payload = json.loads(
        Path(episode_payload["speaker_samples_manifest_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert [sample["speaker"] for sample in manifest_payload["samples"]] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
