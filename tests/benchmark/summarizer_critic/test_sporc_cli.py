"""CLI helper coverage for the SPoRC benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path

from card_framework.benchmark.summarizer_critic.sporc.cli import (
    _child_environment,
    _parse_duration_preset_list,
    _render_duration_sweep_menu,
    _resolve_prepare_root,
    main,
)
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_PREPARED_FULL_ROOT,
    DEFAULT_PREPARED_ROOT,
)
from card_framework.shared.paths import REPO_ROOT


def test_sporc_child_environment_enables_unbuffered_python_output(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "existing-path")

    env = _child_environment()

    assert env["PYTHONUNBUFFERED"] == "1"
    pythonpath_entries = env["PYTHONPATH"].split(os.pathsep)
    assert str(REPO_ROOT / "src") in pythonpath_entries
    assert "existing-path" in pythonpath_entries


def test_render_duration_sweep_menu_lists_presets() -> None:
    menu = _render_duration_sweep_menu()

    assert "30s" in menu
    assert "batch-all" in menu


def test_parse_duration_preset_list_splits_and_validates() -> None:
    assert _parse_duration_preset_list("30s,1m,5m") == ["30s", "1m", "5m"]


def test_resolve_prepare_root_defaults_by_dataset_variant() -> None:
    assert _resolve_prepare_root(dataset_variant="sample", prepared_root="") == (
        DEFAULT_PREPARED_ROOT.resolve()
    )
    assert _resolve_prepare_root(dataset_variant="full", prepared_root="") == (
        DEFAULT_PREPARED_FULL_ROOT.resolve()
    )


def test_main_curate_longform_writes_manifest_and_audit(
    tmp_path: Path,
    capsys,
) -> None:
    transcript_path = tmp_path / "transcripts" / "passing.transcript.json"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    segments = []
    current_start = 0
    for index in range(150):
        segments.append(
            {
                "speaker": "SPEAKER_00" if index % 2 == 0 else "SPEAKER_01",
                "text": " ".join(["dialogue"] * 110),
                "start_time": current_start,
                "end_time": current_start + 60_000,
            }
        )
        current_start += 60_000
    transcript_path.write_text(
        json.dumps(
            {
                "segments": segments,
                "metadata": {"dataset": "sporc", "episode_title": "Long Dialogue"},
            }
        ),
        encoding="utf-8",
    )
    source_manifest_path = tmp_path / "prepared_full" / "manifest.json"
    source_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    source_manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(tmp_path / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(tmp_path / "speakerTurnData.jsonl.gz"),
                "sample_count": 1,
                "samples": [
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
            }
        ),
        encoding="utf-8",
    )
    output_manifest_path = tmp_path / "prepared_full" / "benchmark_longform_15k_manifest.json"
    audit_path = tmp_path / "prepared_full" / "benchmark_longform_15k_audit.json"

    exit_code = main(
        [
            "curate-longform",
            "--source-manifest",
            str(source_manifest_path),
            "--output-manifest",
            str(output_manifest_path),
            "--audit-path",
            str(audit_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert Path(payload["dataset_manifest"]).exists()
    assert Path(payload["audit_path"]).exists()
