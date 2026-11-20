"""Coverage for the sampled-transcript stage-2 pipeline helper."""

from __future__ import annotations

import json
from pathlib import Path

from card_framework.benchmark.summarizer_critic.sporc.sampled_pipeline import (
    _build_episode_catalog,
    _build_synthetic_segments,
    _load_sampled_rows,
    _materialize_episode,
    _parse_episode_selection_spec,
    SampledSegmentRow,
    main,
)


def _write_manifest(
    tmp_path: Path,
    *,
    include_speaker_manifest: bool = True,
    include_source_audio: bool = True,
) -> Path:
    transcripts_root = tmp_path / "transcripts"
    transcripts_root.mkdir(parents=True, exist_ok=True)
    manifest_samples: list[dict[str, object]] = []
    for index in range(1, 3):
        transcript_path = transcripts_root / f"episode_{index}.transcript.json"
        metadata: dict[str, object] = {
            "episode_title": f"Episode {index}",
        }
        if include_source_audio:
            source_audio_path = tmp_path / f"episode_{index}.wav"
            source_audio_path.write_bytes(b"RIFF")
            metadata["source_audio_path"] = str(source_audio_path)
        if include_speaker_manifest:
            speaker_manifest_path = tmp_path / f"speaker_manifest_{index}.json"
            speaker_manifest_path.write_text('{"samples": []}\n', encoding="utf-8")
            metadata["speaker_samples_manifest_path"] = str(speaker_manifest_path)
        transcript_path.write_text(
            json.dumps(
                {
                    "segments": [
                        {
                            "speaker": "SPEAKER_00",
                            "start_time": 0,
                            "end_time": 1000,
                            "text": f"episode {index} alpha beta gamma",
                        }
                    ],
                    "metadata": metadata,
                }
            ),
            encoding="utf-8",
        )
        manifest_samples.append(
            {
                "sample_id": f"sporc_{index:05d}",
                "dataset": "sporc_full",
                "episode_key": f"episode://{index}",
                "transcript_path": str(transcript_path),
                "reference_summary": "Reference summary.",
                "target_seconds": 45,
                "metadata": {"episode_title": f"Episode {index}"},
            }
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
                "sample_count": len(manifest_samples),
                "samples": manifest_samples,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def _write_selected_samples(tmp_path: Path) -> Path:
    selected_samples_path = tmp_path / "selected_samples.jsonl"
    rows = [
        {
            "sample_id": "sporc_00001",
            "episode_key": "episode://1",
            "episode_title": "Episode 1",
            "selection_order": 1,
            "speaker_label": "SPEAKER_00",
            "segment_text": "alpha beta gamma delta",
            "actual_words": 4,
            "estimated_words": 4,
            "rationale": "Opening beat.",
        },
        {
            "sample_id": "sporc_00002",
            "episode_key": "episode://2",
            "episode_title": "Episode 2",
            "selection_order": 1,
            "speaker_label": "SPEAKER_01",
            "segment_text": "epsilon zeta eta theta",
            "actual_words": 4,
            "estimated_words": 4,
            "rationale": "Follow-up beat.",
        },
    ]
    selected_samples_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return selected_samples_path


def test_parse_episode_selection_spec_supports_lists_and_ranges() -> None:
    valid_positions = {1, 2, 3, 4, 5}

    assert _parse_episode_selection_spec("1,3-4", valid_positions=valid_positions) == (
        1,
        3,
        4,
    )
    assert _parse_episode_selection_spec("[2,5]", valid_positions=valid_positions) == (
        2,
        3,
        4,
        5,
    )


def test_build_episode_catalog_matches_manifest_rows_with_selected_samples(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    selected_samples_path = _write_selected_samples(tmp_path)

    sampled_rows = _load_sampled_rows(selected_samples_path)
    catalog = _build_episode_catalog(
        manifest_path=manifest_path,
        sampled_rows=sampled_rows,
    )

    assert [entry.manifest_episode_number for entry in catalog] == [1, 2]
    assert [entry.sample.sample_id for entry in catalog] == [
        "sporc_00001",
        "sporc_00002",
    ]
    assert catalog[0].episode_title == "Episode 1"
    assert catalog[1].sampled_rows[0].speaker_label == "SPEAKER_01"


def test_build_synthetic_segments_uses_sequential_synthetic_timestamps() -> None:
    segments = _build_synthetic_segments(
        (
            SampledSegmentRow(
                sample_id="sample-1",
                episode_key="episode://1",
                episode_title="Episode 1",
                selection_order=1,
                speaker_label="SPEAKER_00",
                segment_text="alpha beta gamma delta",
                rationale="Opening.",
                actual_words=4,
            ),
            SampledSegmentRow(
                sample_id="sample-1",
                episode_key="episode://1",
                episode_title="Episode 1",
                selection_order=2,
                speaker_label="SPEAKER_01",
                segment_text="epsilon zeta eta theta",
                rationale="Reply.",
                actual_words=4,
            ),
        ),
        words_per_minute=170.0,
        min_segment_ms=1000,
        segment_gap_ms=250,
    )

    assert [segment.start_time for segment in segments] == [0, 1662]
    assert segments[0].end_time == 1412
    assert segments[1].end_time == 3074
    assert segments[0].extras["selection_order"] == 1


def test_materialize_episode_preserves_metadata_and_warns_when_reuse_is_missing(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(
        tmp_path,
        include_speaker_manifest=False,
        include_source_audio=False,
    )
    selected_samples_path = _write_selected_samples(tmp_path)
    sampled_rows = _load_sampled_rows(selected_samples_path)
    catalog = _build_episode_catalog(
        manifest_path=manifest_path,
        sampled_rows=sampled_rows,
    )

    materialized = _materialize_episode(
        entry=catalog[0],
        selected_samples_path=selected_samples_path,
        manifest_path=manifest_path,
        output_root=tmp_path / "outputs",
        words_per_minute=120.0,
        min_segment_ms=500,
        segment_gap_ms=100,
    )

    payload = json.loads(materialized.transcript_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["sample_id"] == "sporc_00001"
    assert (
        payload["metadata"]["sampled_transcript_timing_mode"]
        == "synthetic_sequential"
    )
    assert payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert payload["segments"][0]["start_time"] == 0
    assert payload["segments"][0]["end_time"] >= 500
    assert materialized.reused_speaker_manifest_path is None
    assert materialized.source_audio_path is None
    assert len(materialized.warnings) == 2


def test_main_prepare_only_writes_one_selected_episode_summary(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path)
    selected_samples_path = _write_selected_samples(tmp_path)
    output_root = tmp_path / "outputs"

    exit_code = main(
        [
            "--dataset-manifest",
            str(manifest_path),
            "--selected-samples-path",
            str(selected_samples_path),
            "--output-root",
            str(output_root),
            "--select-episodes",
            "2",
            "--prepare-only",
        ]
    )

    assert exit_code == 0
    summary_payload = json.loads(
        (output_root / "run_summary.json").read_text(encoding="utf-8")
    )
    assert summary_payload["selected_episode_positions"] == [2]
    assert summary_payload["episode_count"] == 1
    assert summary_payload["episodes"][0]["sample_id"] == "sporc_00002"
    transcript_path = Path(summary_payload["episodes"][0]["transcript_path"])
    assert transcript_path.exists()
