"""Dataset preparation coverage for the SPoRC benchmark."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    DatasetPreparationError,
    build_prepared_dataset_manifest,
    curate_longform_sporc_manifest,
    derive_sporc_target_seconds,
    derive_target_seconds,
    iter_prepared_dataset_samples,
    load_prepared_dataset_manifest,
)


def _write_jsonl_gz(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _segment(
    *,
    speaker: str,
    text: str,
    start_time: int,
    end_time: int,
) -> dict[str, object]:
    return {
        "speaker": speaker,
        "text": text,
        "start_time": start_time,
        "end_time": end_time,
    }


def _write_transcript(
    path: Path,
    *,
    segments: list[dict[str, object]],
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "segments": segments,
                "metadata": {
                    "dataset": "sporc",
                    "episode_title": title,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_prepared_manifest(
    path: Path,
    *,
    samples: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(path.parent / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(path.parent / "speakerTurnData.jsonl.gz"),
                "sample_count": len(samples),
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )


def test_derive_target_seconds_uses_reference_summary_length() -> None:
    assert derive_target_seconds("one two three four five", reference_wpm=150.0) == 2


def test_derive_sporc_target_seconds_uses_hybrid_floor_for_short_descriptions() -> None:
    assert (
        derive_sporc_target_seconds(
            "short teaser",
            transcript_word_count=125,
            reference_wpm=150.0,
        )
        == 12
    )


def test_build_prepared_dataset_manifest_uses_sporc_episode_and_turn_files(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episodeLevelDataSample.jsonl.gz"
    turn_path = tmp_path / "speakerTurnDataSample.jsonl.gz"
    episode_key = "podcasts/example_episode.mp3"
    _write_jsonl_gz(
        episode_path,
        [
            {
                "potentialOutPath": episode_key,
                "podDescription": "A short podcast description used as the target summary.",
                "episodeTitle": "Example Episode",
                "hostSpeakerLabels": {"Alex Host": ["SPEAKER_00"]},
                "guestSpeakerLabels": {"Blair Guest": ["SPEAKER_01"]},
            }
        ],
    )
    _write_jsonl_gz(
        turn_path,
        [
            {
                "potentialOutPath": episode_key,
                "start": 0.0,
                "end": 3.0,
                "speakers": ["SPEAKER_00"],
                "content": "Welcome back to the show.",
            },
            {
                "potentialOutPath": episode_key,
                "start": 3.0,
                "end": 8.0,
                "speakers": ["SPEAKER_01"],
                "content": "Thanks for having me to talk about the launch.",
            },
        ],
    )

    manifest = build_prepared_dataset_manifest(
        episode_data_path=episode_path,
        turn_data_path=turn_path,
        prepared_root=tmp_path / "prepared",
        dataset_variant="sample",
    )

    assert manifest.sample_count == 1
    sample = manifest.samples[0]
    assert sample.dataset == "sporc_sample"
    assert sample.reference_summary.startswith("A short podcast description")
    transcript_payload = json.loads(
        Path(sample.transcript_path).read_text(encoding="utf-8")
    )
    assert transcript_payload["segments"][0]["speaker"] == "SPEAKER_00"
    assert transcript_payload["segments"][1]["end_time"] == 8000
    assert (
        transcript_payload["metadata"]["speaker_alias_map"]["Alex Host"] == "SPEAKER_00"
    )
    assert transcript_payload["metadata"]["source_audio_url"] == episode_key
    assert sample.target_seconds == 12
    assert sample.metadata["longform_curation_candidate"]["segment_count"] == 2
    assert sample.metadata["longform_curation_candidate"]["unique_speaker_count"] == 2


def test_build_prepared_dataset_manifest_supports_real_sporc_field_names(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episodeLevelDataSample.jsonl.gz"
    turn_path = tmp_path / "speakerTurnDataSample.jsonl.gz"
    episode_key = "https://example.com/podcast.mp3"
    _write_jsonl_gz(
        episode_path,
        [
            {
                "mp3url": episode_key,
                "epDescription": "<p>Episode level summary text.</p>",
                "epTitle": "Real Schema Episode",
                "hostSpeakerLabels": {"Alex Host": ["SPEAKER_00"]},
            }
        ],
    )
    _write_jsonl_gz(
        turn_path,
        [
            {
                "mp3url": episode_key,
                "startTime": 0.0,
                "endTime": 5.0,
                "speaker": ["SPEAKER_00"],
                "turnText": "Welcome back to the show.",
            }
        ],
    )

    manifest = build_prepared_dataset_manifest(
        episode_data_path=episode_path,
        turn_data_path=turn_path,
        prepared_root=tmp_path / "prepared_real_schema",
        dataset_variant="sample",
    )

    sample = manifest.samples[0]
    assert sample.reference_summary == "Episode level summary text."
    transcript_payload = json.loads(
        Path(sample.transcript_path).read_text(encoding="utf-8")
    )
    assert transcript_payload["metadata"]["episode_title"] == "Real Schema Episode"
    assert transcript_payload["segments"][0]["text"] == "Welcome back to the show."
    assert transcript_payload["segments"][0]["end_time"] == 5000
    assert transcript_payload["metadata"]["source_audio_url"] == episode_key
    assert sample.target_seconds == 12


def test_build_prepared_dataset_manifest_full_variant_uses_sidecar_and_episode_order(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episodeLevelData.jsonl.gz"
    turn_path = tmp_path / "speakerTurnData.jsonl.gz"
    episode_key_a = "https://example.com/episode-a.mp3"
    episode_key_b = "https://example.com/episode-b.mp3"
    _write_jsonl_gz(
        episode_path,
        [
            {
                "mp3url": episode_key_a,
                "epDescription": "Episode A summary.",
                "epTitle": "Episode A",
            },
            {
                "mp3url": episode_key_b,
                "epDescription": "Episode B summary.",
                "epTitle": "Episode B",
            },
        ],
    )
    _write_jsonl_gz(
        turn_path,
        [
            {
                "mp3url": episode_key_b,
                "startTime": 0.0,
                "endTime": 5.0,
                "speaker": ["SPEAKER_01"],
                "turnText": "This is episode B.",
            },
            {
                "mp3url": episode_key_a,
                "startTime": 0.0,
                "endTime": 5.0,
                "speaker": ["SPEAKER_00"],
                "turnText": "This is episode A.",
            },
        ],
    )

    manifest = build_prepared_dataset_manifest(
        episode_data_path=episode_path,
        turn_data_path=turn_path,
        prepared_root=tmp_path / "prepared_full",
        dataset_variant="full",
    )

    assert manifest.sample_count == 2
    assert manifest.samples == ()
    assert manifest.samples_path is not None
    assert manifest.samples_path.exists()
    loaded_samples = list(iter_prepared_dataset_samples(manifest.manifest_path))
    assert [sample.sample_id for sample in loaded_samples] == [
        "sporc_00001_https_example_com_episode_a_mp3",
        "sporc_00002_https_example_com_episode_b_mp3",
    ]
    assert (
        loaded_samples[0].metadata["longform_curation_candidate"]["segment_count"] == 1
    )
    payload = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
    assert payload["samples_path"] == str(manifest.samples_path)
    assert "samples" not in payload


def test_build_prepared_dataset_manifest_rejects_repeated_turn_groups(
    tmp_path: Path,
) -> None:
    episode_path = tmp_path / "episodeLevelData.jsonl.gz"
    turn_path = tmp_path / "speakerTurnData.jsonl.gz"
    episode_key_a = "https://example.com/episode-a.mp3"
    episode_key_b = "https://example.com/episode-b.mp3"
    _write_jsonl_gz(
        episode_path,
        [
            {
                "mp3url": episode_key_a,
                "epDescription": "Episode A summary.",
                "epTitle": "Episode A",
            },
            {
                "mp3url": episode_key_b,
                "epDescription": "Episode B summary.",
                "epTitle": "Episode B",
            },
        ],
    )
    _write_jsonl_gz(
        turn_path,
        [
            {
                "mp3url": episode_key_a,
                "startTime": 0.0,
                "endTime": 5.0,
                "speaker": ["SPEAKER_00"],
                "turnText": "First A block.",
            },
            {
                "mp3url": episode_key_b,
                "startTime": 0.0,
                "endTime": 5.0,
                "speaker": ["SPEAKER_01"],
                "turnText": "B block.",
            },
            {
                "mp3url": episode_key_a,
                "startTime": 5.0,
                "endTime": 10.0,
                "speaker": ["SPEAKER_00"],
                "turnText": "Second A block.",
            },
        ],
    )

    try:
        build_prepared_dataset_manifest(
            episode_data_path=episode_path,
            turn_data_path=turn_path,
            prepared_root=tmp_path / "prepared_full",
            dataset_variant="full",
        )
    except DatasetPreparationError as exc:
        assert "repeated episode key" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected DatasetPreparationError")


def test_load_prepared_dataset_manifest_supports_bounded_sidecar_loading(
    tmp_path: Path,
) -> None:
    prepared_root = tmp_path / "prepared_full"
    prepared_root.mkdir(parents=True, exist_ok=True)
    samples_path = prepared_root / "samples.jsonl.gz"
    with gzip.open(samples_path, "wt", encoding="utf-8") as handle:
        for index in range(3):
            handle.write(
                json.dumps(
                    {
                        "sample_id": f"sample_{index}",
                        "dataset": "sporc_full",
                        "episode_key": f"episode://{index}",
                        "transcript_path": str(prepared_root / f"{index}.json"),
                        "reference_summary": "reference",
                        "target_seconds": 45,
                        "metadata": {},
                    }
                )
            )
            handle.write("\n")
    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(tmp_path / "speakerTurnData.jsonl.gz"),
                "sample_count": 3,
                "samples_path": str(samples_path),
            }
        ),
        encoding="utf-8",
    )

    manifest = load_prepared_dataset_manifest(
        manifest_path,
        max_samples=2,
    )
    iterated = list(iter_prepared_dataset_samples(manifest_path, max_samples=2))

    assert manifest.sample_count == 3
    assert manifest.samples_path == samples_path.resolve()
    assert [sample.sample_id for sample in manifest.samples] == [
        "sample_0",
        "sample_1",
    ]
    assert [sample.sample_id for sample in iterated] == [
        "sample_0",
        "sample_1",
    ]


def test_curate_longform_sporc_manifest_filters_to_strict_multi_speaker_examples(
    tmp_path: Path,
) -> None:
    transcript_root = tmp_path / "transcripts"
    passing_transcript = transcript_root / "passing.transcript.json"
    music_transcript = transcript_root / "music.transcript.json"
    single_speaker_transcript = transcript_root / "single.transcript.json"
    dominant_speaker_transcript = transcript_root / "dominant.transcript.json"

    passing_segments: list[dict[str, object]] = []
    current_start = 0
    for index in range(150):
        passing_segments.append(
            _segment(
                speaker="SPEAKER_00" if index % 2 == 0 else "SPEAKER_01",
                text=" ".join(["dialogue"] * 110),
                start_time=current_start,
                end_time=current_start + 60_000,
            )
        )
        current_start += 60_000
    _write_transcript(
        passing_transcript,
        segments=passing_segments,
        title="Long Form Dialogue",
    )

    music_segments: list[dict[str, object]] = []
    current_start = 0
    for index in range(150):
        music_segments.append(
            _segment(
                speaker="SPEAKER_00" if index % 2 == 0 else "SPEAKER_01",
                text=" ".join(["music"] * 120),
                start_time=current_start,
                end_time=current_start + 60_000,
            )
        )
        current_start += 60_000
    _write_transcript(
        music_transcript,
        segments=music_segments,
        title="Mostly Music",
    )

    single_speaker_segments: list[dict[str, object]] = []
    current_start = 0
    for _ in range(160):
        single_speaker_segments.append(
            _segment(
                speaker="SPEAKER_00",
                text=" ".join(["monologue"] * 100),
                start_time=current_start,
                end_time=current_start + 60_000,
            )
        )
        current_start += 60_000
    _write_transcript(
        single_speaker_transcript,
        segments=single_speaker_segments,
        title="Collapsed Diarization",
    )

    dominant_speaker_segments: list[dict[str, object]] = []
    current_start = 0
    for _ in range(151):
        dominant_speaker_segments.append(
            _segment(
                speaker="SPEAKER_00",
                text=" ".join(["host"] * 100),
                start_time=current_start,
                end_time=current_start + 60_000,
            )
        )
        current_start += 60_000
    for _ in range(20):
        dominant_speaker_segments.append(
            _segment(
                speaker="SPEAKER_01",
                text=" ".join(["guest"] * 50),
                start_time=current_start,
                end_time=current_start + 60_000,
            )
        )
        current_start += 60_000
    _write_transcript(
        dominant_speaker_transcript,
        segments=dominant_speaker_segments,
        title="Host Dominant",
    )

    source_manifest_path = tmp_path / "prepared_full" / "manifest.json"
    _write_prepared_manifest(
        source_manifest_path,
        samples=[
            {
                "sample_id": "passing",
                "dataset": "sporc_full",
                "episode_key": "episode://passing",
                "transcript_path": str(passing_transcript),
                "reference_summary": "Passing long-form summary.",
                "target_seconds": 45,
                "metadata": {},
            },
            {
                "sample_id": "music",
                "dataset": "sporc_full",
                "episode_key": "episode://music",
                "transcript_path": str(music_transcript),
                "reference_summary": "Music-heavy example.",
                "target_seconds": 45,
                "metadata": {},
            },
            {
                "sample_id": "single",
                "dataset": "sporc_full",
                "episode_key": "episode://single",
                "transcript_path": str(single_speaker_transcript),
                "reference_summary": "Single speaker example.",
                "target_seconds": 45,
                "metadata": {},
            },
            {
                "sample_id": "dominant",
                "dataset": "sporc_full",
                "episode_key": "episode://dominant",
                "transcript_path": str(dominant_speaker_transcript),
                "reference_summary": "Dominant host example.",
                "target_seconds": 45,
                "metadata": {},
            },
        ],
    )

    curated_manifest, audit_path = curate_longform_sporc_manifest(
        source_manifest_path=source_manifest_path,
        output_manifest_path=tmp_path / "prepared_full" / "benchmark_longform_15k_manifest.json",
        audit_path=tmp_path / "prepared_full" / "benchmark_longform_15k_audit.json",
    )

    assert curated_manifest.sample_count == 1
    assert curated_manifest.samples[0].sample_id == "passing"
    assert (
        curated_manifest.samples[0].metadata["longform_curation"]["spoken_word_count"]
        == 16_500
    )
    assert (
        curated_manifest.samples[0].metadata["longform_curation"][
            "unique_speaker_count"
        ]
        == 2
    )
    assert audit_path.exists()
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_payload["selected_sample_count"] == 1
    audit_rows = {
        row["sample_id"]: row for row in audit_payload["samples"]
    }
    assert (
        "non_lexical_word_ratio>0.10"
        in audit_rows["music"]["excluded_reasons"]
    )
    assert (
        "unique_speaker_count<2"
        in audit_rows["single"]["excluded_reasons"]
    )
    assert (
        "dominant_speaker_word_share>0.90"
        in audit_rows["dominant"]["excluded_reasons"]
    )


def test_curate_longform_sporc_manifest_reuses_precomputed_stats_and_reports_status_progress(
    tmp_path: Path,
    monkeypatch,
) -> None:
    samples: list[dict[str, object]] = []
    for sample_index in range(2):
        samples.append(
            {
                "sample_id": f"passing_{sample_index}",
                "dataset": "sporc_full",
                "episode_key": f"episode://passing_{sample_index}",
                "transcript_path": str(
                    tmp_path / "transcripts" / f"passing_{sample_index}.transcript.json"
                ),
                "reference_summary": "Passing long-form summary.",
                "target_seconds": 45,
                "metadata": {
                    "longform_curation_candidate": {
                        "profile": "longform_15k_v1",
                        "raw_word_count": 16_500,
                        "spoken_word_count": 16_500,
                        "non_lexical_word_count": 0,
                        "non_lexical_word_ratio": 0.0,
                        "segment_count": 150,
                        "unique_speaker_count": 2,
                        "strong_speaker_count": 2,
                        "dominant_speaker_word_share": 0.5,
                        "timestamp_span_seconds": 9000.0,
                        "speaker_stats": {
                            "SPEAKER_00": {
                                "raw_word_count": 8250,
                                "spoken_word_count": 8250,
                                "segment_count": 75,
                            },
                            "SPEAKER_01": {
                                "raw_word_count": 8250,
                                "spoken_word_count": 8250,
                                "segment_count": 75,
                            },
                        },
                    }
                },
            }
        )

    source_manifest_path = tmp_path / "prepared_full" / "manifest.json"
    _write_prepared_manifest(source_manifest_path, samples=samples)
    status_messages: list[str] = []
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.dataset._LONGFORM_CURATION_PROGRESS_INTERVAL",
        1,
    )
    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.dataset._load_transcript_payload",
        lambda transcript_path: (_ for _ in ()).throw(
            AssertionError("precomputed curation stats should avoid transcript reloads")
        ),
    )

    curate_longform_sporc_manifest(
        source_manifest_path=source_manifest_path,
        output_manifest_path=tmp_path / "prepared_full" / "benchmark_longform_15k_manifest.json",
        audit_path=tmp_path / "prepared_full" / "benchmark_longform_15k_audit.json",
        status_reporter=status_messages.append,
    )

    assert any(
        message == "Loading prepared SPoRC source manifest..."
        for message in status_messages
    )
    assert any(
        message.startswith("Loaded prepared SPoRC source manifest in ")
        for message in status_messages
    )
    assert any(
        message == "Loading SPoRC long-form curation sample stats..."
        for message in status_messages
    )
    assert any(
        "evaluated 1 / 2 prepared samples; selected 1 so far; transcript fallbacks 0."
        in message
        for message in status_messages
    )
    assert any(
        "evaluated 2 / 2 prepared samples; selected 2 so far; transcript fallbacks 0."
        in message
        for message in status_messages
    )
    assert any(
        message.startswith("Loaded SPoRC long-form curation sample stats in ")
        for message in status_messages
    )
    assert any(
        message == "Loading curated SPoRC long-form manifest..."
        for message in status_messages
    )
    assert any(
        message == "Loading SPoRC long-form curation audit..."
        for message in status_messages
    )


def test_curated_longform_manifest_loads_with_existing_schema(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcripts" / "passing.transcript.json"
    segments: list[dict[str, object]] = []
    current_start = 0
    for index in range(150):
        segments.append(
            _segment(
                speaker="SPEAKER_00" if index % 2 == 0 else "SPEAKER_01",
                text=" ".join(["dialogue"] * 110),
                start_time=current_start,
                end_time=current_start + 60_000,
            )
        )
        current_start += 60_000
    _write_transcript(
        transcript_path,
        segments=segments,
        title="Long Form Dialogue",
    )
    source_manifest_path = tmp_path / "prepared_full" / "manifest.json"
    _write_prepared_manifest(
        source_manifest_path,
        samples=[
            {
                "sample_id": "passing",
                "dataset": "sporc_full",
                "episode_key": "episode://passing",
                "transcript_path": str(transcript_path),
                "reference_summary": "Passing long-form summary.",
                "target_seconds": 45,
                "metadata": {},
            }
        ],
    )

    curated_manifest, _ = curate_longform_sporc_manifest(
        source_manifest_path=source_manifest_path,
        output_manifest_path=tmp_path / "prepared_full" / "benchmark_longform_15k_manifest.json",
        audit_path=tmp_path / "prepared_full" / "benchmark_longform_15k_audit.json",
    )

    reloaded_manifest = load_prepared_dataset_manifest(curated_manifest.manifest_path)
    reloaded_payload = json.loads(
        Path(curated_manifest.manifest_path).read_text(encoding="utf-8")
    )
    assert reloaded_manifest.sample_count == 1
    assert reloaded_manifest.samples[0].sample_id == "passing"
    assert reloaded_payload["dataset_id"] == "sporc_full_longform_15k"
    assert reloaded_payload["sample_count"] == 1
    assert (
        reloaded_payload["samples"][0]["metadata"]["longform_curation"]["profile"]
        == "longform_15k_v1"
    )
