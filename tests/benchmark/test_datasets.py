from __future__ import annotations

import json
from pathlib import Path

import pytest

from card_framework.benchmark.datasets import ManifestError, load_manifest


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

