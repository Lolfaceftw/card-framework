"""Dataset preparation coverage for the summarizer-critic benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from card_framework.benchmark.summarizer_critic.dataset import (
    build_prepared_dataset_manifest,
    derive_target_seconds,
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
