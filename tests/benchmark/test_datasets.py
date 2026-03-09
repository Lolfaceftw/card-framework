from __future__ import annotations

import json
from pathlib import Path

import pytest

from card_framework.benchmark.datasets import (
    ManifestError,
    load_manifest,
    prepare_manifest,
)


def test_load_manifest_resolves_relative_path(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript.json"
    transcript_path.write_text(json.dumps({"segments": []}), encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "sample_1",
                        "dataset": "local",
                        "transcript_path": "transcript.json",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    samples = load_manifest(manifest_path, tmp_path)

    assert len(samples) == 1
    assert samples[0].sample_id == "sample_1"
    assert Path(samples[0].transcript_path).exists()


def test_load_manifest_raises_for_missing_samples(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": []}), encoding="utf-8")

    with pytest.raises(ManifestError):
        load_manifest(manifest_path, tmp_path)


def test_load_manifest_auto_resolves_local_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "artifacts" / "transcripts" / "sample.transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(json.dumps({"segments": []}), encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "local_summary",
                        "dataset": "local",
                        "transcript_path": "auto",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    samples = load_manifest(manifest_path, tmp_path)

    assert len(samples) == 1
    assert Path(samples[0].transcript_path) == transcript_path.resolve()


def test_prepare_manifest_auto_discovers_local_transcript(tmp_path: Path) -> None:
    transcript_path = tmp_path / "artifacts" / "transcripts" / "sample.transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.write_text(
        json.dumps({"segments": [{"speaker": "SPEAKER_00", "text": "hello world"}]}),
        encoding="utf-8",
    )

    manifest = prepare_manifest(
        repo_root=tmp_path,
        output_path=tmp_path / "manifest.json",
        sources=["local"],
        num_samples=1,
        local_transcript_path="auto",
    )

    assert manifest["samples"][0]["transcript_path"] == str(transcript_path.resolve())


def test_prepare_manifest_raises_when_no_local_transcript_exists(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="No reusable local transcript was found"):
        prepare_manifest(
            repo_root=tmp_path,
            output_path=tmp_path / "manifest.json",
            sources=["local"],
            num_samples=1,
            local_transcript_path="auto",
        )

