"""Coverage for the SPoRC manifest dedupe helper."""

from __future__ import annotations

import json
from pathlib import Path

from card_framework.benchmark.summarizer_critic.sporc.dedupe_manifest import (
    dedupe_sporc_manifest_in_place,
)


def _write_transcript(
    path: Path,
    *,
    title: str,
    text: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0,
                        "end_time": 1000,
                        "text": text,
                    }
                ],
                "metadata": {
                    "dataset": "sporc",
                    "episode_title": title,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_manifest(path: Path) -> Path:
    transcripts_root = path.parent / "transcripts"
    transcript_a = transcripts_root / "duplicate_a.transcript.json"
    transcript_b = transcripts_root / "duplicate_b.transcript.json"
    transcript_c = transcripts_root / "same_show_other_episode.transcript.json"
    transcript_d = transcripts_root / "unique_episode.transcript.json"
    duplicate_text = "alpha beta gamma delta epsilon"
    _write_transcript(
        transcript_a,
        title="Episode 140",
        text=duplicate_text,
    )
    _write_transcript(
        transcript_b,
        title="Episode 140",
        text=duplicate_text,
    )
    _write_transcript(
        transcript_c,
        title="Episode 141",
        text="same show but different episode text",
    )
    _write_transcript(
        transcript_d,
        title="Completely Different Episode",
        text="unique transcript text",
    )
    samples = [
        {
            "sample_id": "sample_a",
            "dataset": "sporc_full",
            "episode_key": "https://example.com/a.mp3",
            "transcript_path": str(transcript_a),
            "reference_summary": "Shared duplicate summary.",
            "target_seconds": 45,
            "metadata": {
                "episode_title": "Episode 140",
                "reference_word_count": 3,
                "source_audio_url": "https://example.com/a.mp3",
                "longform_curation": {
                    "spoken_word_count": 5,
                    "segment_count": 1,
                },
            },
        },
        {
            "sample_id": "sample_b",
            "dataset": "sporc_full",
            "episode_key": "https://example.com/b.mp3",
            "transcript_path": str(transcript_b),
            "reference_summary": "Shared duplicate summary.",
            "target_seconds": 45,
            "metadata": {
                "episode_title": "Episode 140",
                "reference_word_count": 3,
                "source_audio_url": "https://example.com/b.mp3",
                "longform_curation": {
                    "spoken_word_count": 5,
                    "segment_count": 1,
                },
            },
        },
        {
            "sample_id": "sample_c",
            "dataset": "sporc_full",
            "episode_key": "https://example.com/c.mp3",
            "transcript_path": str(transcript_c),
            "reference_summary": "Different episode summary.",
            "target_seconds": 45,
            "metadata": {
                "episode_title": "Episode 141",
                "reference_word_count": 3,
                "source_audio_url": "https://example.com/c.mp3",
                "longform_curation": {
                    "spoken_word_count": 6,
                    "segment_count": 1,
                },
            },
        },
        {
            "sample_id": "sample_d",
            "dataset": "sporc_full",
            "episode_key": "https://example.com/d.mp3",
            "transcript_path": str(transcript_d),
            "reference_summary": "Unique summary.",
            "target_seconds": 45,
            "metadata": {
                "episode_title": "Completely Different Episode",
                "reference_word_count": 2,
                "source_audio_url": "https://example.com/d.mp3",
                "longform_curation": {
                    "spoken_word_count": 3,
                    "segment_count": 1,
                },
            },
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
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
    return path


def test_dedupe_sporc_manifest_in_place_rewrites_same_manifest_fast(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "benchmark_longform_15k_manifest.json")
    audit_path = tmp_path / "benchmark_longform_15k_manifest.dedupe_audit.json"

    result = dedupe_sporc_manifest_in_place(
        manifest_path=manifest_path,
        audit_path=audit_path,
    )

    assert result.manifest_path == manifest_path.resolve()
    assert result.audit_path == audit_path.resolve()
    assert result.original_sample_count == 4
    assert result.deduped_sample_count == 3
    assert result.removed_sample_count == 1
    assert result.candidate_group_count == 1
    assert result.transcript_signature_reads == 2

    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["sample_count"] == 3
    assert [sample["sample_id"] for sample in manifest_payload["samples"]] == [
        "sample_a",
        "sample_c",
        "sample_d",
    ]
    assert manifest_payload["dedupe"]["removed_sample_count"] == 1

    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_payload["removed_sample_count"] == 1
    assert audit_payload["removed_samples"][0]["kept_sample_id"] == "sample_a"
    assert audit_payload["removed_samples"][0]["removed_sample_id"] == "sample_b"
