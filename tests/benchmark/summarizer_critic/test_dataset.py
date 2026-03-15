"""Dataset preparation coverage for the summarizer-critic benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import zipfile

import pytest

from card_framework.benchmark.summarizer_critic.qmsum import dataset as qmsum_dataset
from card_framework.benchmark.summarizer_critic.dataset import (
    DatasetPreparationError,
    build_prepared_dataset_manifest,
    derive_target_seconds,
    ensure_official_qmsum_repo,
)


def _write_qmsum_meeting(root: Path, *, meeting_id: str, answer: str) -> None:
    payload = {
        "meeting_id": meeting_id,
        "meeting_transcripts": [
            {
                "speaker": "Alice",
                "content": "We launched the pilot and collected feedback.",
            },
            {"speaker": "Bob", "content": "We also agreed on the next sprint scope."},
        ],
        "general_query_list": [
            {"query": "Summarize the meeting.", "answer": answer},
        ],
    }
    target_dir = root / "data" / "ALL" / "test"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / f"{meeting_id}.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def test_derive_target_seconds_uses_reference_summary_length() -> None:
    assert derive_target_seconds("one two three four five", reference_wpm=150.0) == 2


def test_build_prepared_dataset_manifest_uses_all_general_queries(
    tmp_path: Path,
) -> None:
    repo_dir = tmp_path / "QMSum"
    _write_qmsum_meeting(
        repo_dir, meeting_id="meeting_a", answer="short reference summary"
    )
    _write_qmsum_meeting(
        repo_dir,
        meeting_id="meeting_b",
        answer="another reference summary with a few more words",
    )

    manifest = build_prepared_dataset_manifest(
        qmsum_repo_dir=repo_dir,
        prepared_root=tmp_path / "prepared",
    )

    assert manifest.sample_count == 2
    assert [sample.sample_id for sample in manifest.samples] == [
        "meeting_a__general_01",
        "meeting_b__general_01",
    ]
    transcript_path = Path(manifest.samples[0].transcript_path)
    assert transcript_path.exists()


def test_ensure_official_qmsum_repo_rejects_paths_outside_artifacts_root(
    tmp_path: Path,
) -> None:
    with pytest.raises(DatasetPreparationError, match="artifacts root"):
        ensure_official_qmsum_repo(repo_dir=tmp_path / "QMSum")


def test_ensure_official_qmsum_repo_rejects_zip_path_traversal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    approved_root = tmp_path / "artifacts"
    repo_dir = approved_root / "external" / "QMSum"
    malicious_zip = tmp_path / "malicious.zip"
    malicious_target = approved_root / "escape.txt"

    approved_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(malicious_zip, "w") as archive:
        archive.writestr("../escape.txt", "escape")

    monkeypatch.setattr(qmsum_dataset, "APPROVED_QMSUM_ROOT", approved_root)
    monkeypatch.setattr(
        qmsum_dataset.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stderr="git clone failed",
            stdout="",
        ),
    )

    def _copy_zip(_url: str, destination: Path) -> tuple[str, None]:
        shutil.copyfile(malicious_zip, destination)
        return str(destination), None

    monkeypatch.setattr(qmsum_dataset.urllib.request, "urlretrieve", _copy_zip)

    with pytest.raises(DatasetPreparationError, match="escapes extraction root"):
        ensure_official_qmsum_repo(repo_dir=repo_dir)

    assert not malicious_target.exists()
