"""Coverage for the SPoRC transcript export helper."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from card_framework.benchmark.summarizer_critic.sporc.export_transcripts import (
    export_manifest_transcripts,
)


def _write_transcript(
    path: Path,
    *,
    episode_title: str,
    first_text: str,
    second_text: str,
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
                        "text": first_text,
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "start_time": 1000,
                        "end_time": 2000,
                        "text": second_text,
                    },
                ],
                "metadata": {
                    "dataset": "sporc",
                    "episode_title": episode_title,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_turn_data(path: Path, *, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_turn_lines(path: Path, *, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def _read_gzip_lines(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.readlines()


def _write_inline_manifest(path: Path) -> Path:
    transcripts_root = path.parent / "transcripts"
    samples: list[dict[str, object]] = []
    turn_rows: list[dict[str, object]] = []
    turn_data_path = path.parent / "speakerTurnData.jsonl.gz"
    for index in range(1, 4):
        transcript_path = transcripts_root / f"episode_{index}.transcript.json"
        _write_transcript(
            transcript_path,
            episode_title=f"Episode {index}",
            first_text=f"prepared episode {index} first line",
            second_text=f"prepared episode {index} second line",
        )
        episode_key = f"episode://{index}"
        turn_rows.extend(
            [
                {
                    "potentialOutPath": episode_key,
                    "startTime": 0.0,
                    "endTime": 1.0,
                    "speaker": ["SPEAKER_00"],
                    "turnText": f"dataset episode {index} first line",
                },
                {
                    "potentialOutPath": episode_key,
                    "startTime": 1.0,
                    "endTime": 2.0,
                    "speaker": ["SPEAKER_01"],
                    "turnText": f"dataset episode {index} second line",
                },
            ]
        )
        samples.append(
            {
                "sample_id": f"sample_{index}",
                "dataset": "sporc_full",
                "episode_key": episode_key,
                "transcript_path": str(transcript_path),
                "reference_summary": "Reference summary.",
                "target_seconds": 45,
                "metadata": {"episode_title": f"Episode {index}"},
            }
        )
    _write_turn_data(turn_data_path, rows=turn_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(path.parent / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(turn_data_path),
                "sample_count": 3,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_manifest_with_actual_episode_numbers(
    path: Path,
    *,
    sidecar: bool,
    actual_episode_numbers: list[int],
) -> Path:
    transcripts_root = path.parent / "transcripts"
    samples: list[dict[str, object]] = []
    sidecar_rows: list[str] = []
    turn_rows: list[dict[str, object]] = []
    turn_data_path = path.parent / "speakerTurnData.actual.jsonl.gz"
    samples_path = path.parent / "actual_samples.jsonl.gz"
    for manifest_position, actual_episode_number in enumerate(
        actual_episode_numbers,
        start=1,
    ):
        episode_key = f"episode://actual/{actual_episode_number}"
        transcript_path = (
            transcripts_root
            / f"{actual_episode_number:05d}_episode_{manifest_position}.transcript.json"
        )
        _write_transcript(
            transcript_path,
            episode_title=f"Actual Episode {actual_episode_number}",
            first_text=f"prepared actual {actual_episode_number} first line",
            second_text=f"prepared actual {actual_episode_number} second line",
        )
        sample_payload = {
            "sample_id": f"sporc_{actual_episode_number:05d}_episode_{manifest_position}",
            "dataset": "sporc_full",
            "episode_key": episode_key,
            "transcript_path": str(transcript_path),
            "reference_summary": "Reference summary.",
            "target_seconds": 45,
            "metadata": {
                "episode_title": f"Actual Episode {actual_episode_number}",
            },
        }
        turn_rows.extend(
            [
                {
                    "potentialOutPath": episode_key,
                    "startTime": 0.0,
                    "endTime": 1.0,
                    "speaker": ["SPEAKER_00"],
                    "turnText": (
                        f"dataset actual {actual_episode_number} first line"
                    ),
                },
                {
                    "potentialOutPath": episode_key,
                    "startTime": 1.0,
                    "endTime": 2.0,
                    "speaker": ["SPEAKER_01"],
                    "turnText": (
                        f"dataset actual {actual_episode_number} second line"
                    ),
                },
            ]
        )
        if sidecar:
            sidecar_rows.append(json.dumps(sample_payload))
        else:
            samples.append(sample_payload)

    _write_turn_data(turn_data_path, rows=turn_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest_payload: dict[str, object] = {
        "dataset_id": "sporc_full_longform_15k",
        "sporc_dataset_id": "blitt/SPoRC",
        "dataset_variant": "full",
        "episode_data_path": str(path.parent / "episodeLevelData.jsonl.gz"),
        "turn_data_path": str(turn_data_path),
        "sample_count": len(actual_episode_numbers),
    }
    if sidecar:
        with gzip.open(samples_path, "wt", encoding="utf-8") as handle:
            for row in sidecar_rows:
                handle.write(row + "\n")
        manifest_payload["samples_path"] = str(samples_path)
    else:
        manifest_payload["samples"] = samples
    path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    return path


def _write_sidecar_manifest(path: Path) -> Path:
    transcripts_root = path.parent / "transcripts"
    samples_path = path.parent / "samples.jsonl.gz"
    turn_data_path = path.parent / "speakerTurnData.jsonl.gz"
    rows: list[str] = []
    turn_rows: list[dict[str, object]] = []
    for index in range(1, 4):
        transcript_path = transcripts_root / f"sidecar_episode_{index}.transcript.json"
        _write_transcript(
            transcript_path,
            episode_title=f"Sidecar Episode {index}",
            first_text=f"prepared sidecar {index} first line",
            second_text=f"prepared sidecar {index} second line",
        )
        episode_key = f"episode://sidecar/{index}"
        turn_rows.extend(
            [
                {
                    "potentialOutPath": episode_key,
                    "startTime": 0.0,
                    "endTime": 1.0,
                    "speaker": ["SPEAKER_00"],
                    "turnText": f"dataset sidecar {index} first line",
                },
                {
                    "potentialOutPath": episode_key,
                    "startTime": 1.0,
                    "endTime": 2.0,
                    "speaker": ["SPEAKER_01"],
                    "turnText": f"dataset sidecar {index} second line",
                },
            ]
        )
        rows.append(
            json.dumps(
                {
                    "sample_id": f"sidecar_sample_{index}",
                    "dataset": "sporc_full",
                    "episode_key": episode_key,
                    "transcript_path": str(transcript_path),
                    "reference_summary": "Reference summary.",
                    "target_seconds": 45,
                    "metadata": {"episode_title": f"Sidecar Episode {index}"},
                }
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(samples_path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row + "\n")
    _write_turn_data(turn_data_path, rows=turn_rows)
    path.write_text(
        json.dumps(
            {
                "dataset_id": "sporc_full_longform_15k",
                "sporc_dataset_id": "blitt/SPoRC",
                "dataset_variant": "full",
                "episode_data_path": str(path.parent / "episodeLevelData.jsonl.gz"),
                "turn_data_path": str(turn_data_path),
                "sample_count": 3,
                "samples_path": str(samples_path),
            }
        ),
        encoding="utf-8",
    )
    return path


def test_export_manifest_transcripts_writes_combined_txt_for_inline_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = _write_inline_manifest(tmp_path / "manifest.json")
    output_path = tmp_path / "combined.txt"

    result = export_manifest_transcripts(
        manifest_path=manifest_path,
        episode_spec="1,3",
        output_path=output_path,
    )

    assert result.output_path == output_path.resolve()
    assert result.episode_positions == (1, 3)
    assert result.episode_count == 2
    exported_text = output_path.read_text(encoding="utf-8")
    assert "===== Episode 1 =====" in exported_text
    assert "===== Episode 3 =====" in exported_text
    assert "Episode Title: Episode 1" in exported_text
    assert "Episode Title: Episode 3" in exported_text
    assert "[00:00:00-00:00:01] [SPEAKER_00] dataset episode 1 first line" in exported_text
    assert "[00:00:01-00:00:02] [SPEAKER_01] dataset episode 3 second line" in exported_text
    assert "prepared episode 1 first line" not in exported_text
    assert "Episode Title: Episode 2" not in exported_text


def test_export_manifest_transcripts_supports_sidecar_manifest_and_bracket_range(
    tmp_path: Path,
) -> None:
    manifest_path = _write_sidecar_manifest(tmp_path / "manifest_sidecar.json")
    output_path = tmp_path / "sidecar_combined.txt"

    result = export_manifest_transcripts(
        manifest_path=manifest_path,
        episode_spec="[2,3]",
        output_path=output_path,
    )

    assert result.episode_positions == (2, 3)
    exported_text = output_path.read_text(encoding="utf-8")
    assert "===== Episode 2 =====" in exported_text
    assert "===== Episode 3 =====" in exported_text
    assert "dataset sidecar 2 first line" in exported_text
    assert "prepared sidecar 2 first line" not in exported_text
    assert "Sidecar Episode 1" not in exported_text


def test_export_manifest_transcripts_stops_after_last_selected_episode_group(
    tmp_path: Path,
) -> None:
    manifest_path = _write_inline_manifest(tmp_path / "manifest.json")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    turn_data_path = Path(str(manifest_payload["turn_data_path"]))
    _write_turn_lines(
        turn_data_path,
        lines=[
            json.dumps(
                {
                    "potentialOutPath": "episode://1",
                    "startTime": 0.0,
                    "endTime": 1.0,
                    "speaker": ["SPEAKER_00"],
                    "turnText": "dataset episode 1 first line",
                }
            ),
            json.dumps(
                {
                    "potentialOutPath": "episode://1",
                    "startTime": 1.0,
                    "endTime": 2.0,
                    "speaker": ["SPEAKER_01"],
                    "turnText": "dataset episode 1 second line",
                }
            ),
            json.dumps(
                {
                    "potentialOutPath": "episode://2",
                    "startTime": 0.0,
                    "endTime": 1.0,
                    "speaker": ["SPEAKER_00"],
                    "turnText": "dataset episode 2 first line",
                }
            ),
            json.dumps(
                {
                    "potentialOutPath": "episode://2",
                    "startTime": 1.0,
                    "endTime": 2.0,
                    "speaker": ["SPEAKER_01"],
                    "turnText": "dataset episode 2 second line",
                }
            ),
            json.dumps(
                {
                    "potentialOutPath": "episode://3",
                    "startTime": 0.0,
                    "endTime": 1.0,
                    "speaker": ["SPEAKER_00"],
                    "turnText": "dataset episode 3 first line",
                }
            ),
            "{not valid json}",
        ],
    )

    output_path = tmp_path / "selected.txt"
    result = export_manifest_transcripts(
        manifest_path=manifest_path,
        episode_spec="1-2",
        output_path=output_path,
    )

    assert result.episode_positions == (1, 2)
    exported_text = output_path.read_text(encoding="utf-8")
    assert "dataset episode 1 first line" in exported_text
    assert "dataset episode 2 second line" in exported_text
    assert "dataset episode 3 first line" not in exported_text


def test_export_manifest_transcripts_backwards_compat_keeps_longform_manifest_positions(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest_with_actual_episode_numbers(
        tmp_path / "compat_inline_manifest.json",
        sidecar=False,
        actual_episode_numbers=[10, 20, 30],
    )
    output_path = tmp_path / "compat_inline.txt"

    result = export_manifest_transcripts(
        manifest_path=manifest_path,
        episode_spec="1-2",
        output_path=output_path,
        backwards_compat=True,
    )

    assert result.episode_positions == (1, 2)
    exported_text = output_path.read_text(encoding="utf-8")
    assert "===== Episode 1 =====" in exported_text
    assert "===== Episode 2 =====" in exported_text
    assert "dataset actual 10 first line" in exported_text
    assert "dataset actual 20 second line" in exported_text
    assert "dataset actual 30 first line" not in exported_text


def test_export_manifest_transcripts_backwards_compat_maps_longform_positions_onto_sidecar_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compatibility_manifest_path = _write_manifest_with_actual_episode_numbers(
        tmp_path / "compatibility_longform_manifest.json",
        sidecar=False,
        actual_episode_numbers=[10, 20, 30],
    )
    target_manifest_path = _write_manifest_with_actual_episode_numbers(
        tmp_path / "target_original_manifest.json",
        sidecar=True,
        actual_episode_numbers=[5, 10, 20, 25, 30],
    )
    output_path = tmp_path / "compat_sidecar.txt"

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.export_transcripts.DEFAULT_LONGFORM_MANIFEST_PATH",
        compatibility_manifest_path,
    )

    result = export_manifest_transcripts(
        manifest_path=target_manifest_path,
        episode_spec="1-2",
        output_path=output_path,
        backwards_compat=True,
    )

    assert result.episode_positions == (1, 2)
    exported_text = output_path.read_text(encoding="utf-8")
    assert "===== Episode 1 =====" in exported_text
    assert "===== Episode 2 =====" in exported_text
    assert "dataset actual 10 first line" in exported_text
    assert "dataset actual 20 second line" in exported_text
    assert "dataset actual 5 first line" not in exported_text
    assert "dataset actual 30 first line" not in exported_text


def test_export_manifest_transcripts_raw_writes_exact_dataset_rows(
    tmp_path: Path,
) -> None:
    manifest_path = _write_inline_manifest(tmp_path / "manifest.json")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    turn_data_path = Path(str(manifest_payload["turn_data_path"]))
    output_path = tmp_path / "raw_rows.jsonl"

    result = export_manifest_transcripts(
        manifest_path=manifest_path,
        episode_spec="1,3",
        output_path=output_path,
        raw=True,
    )

    expected_lines = [
        line
        for line in _read_gzip_lines(turn_data_path)
        if '"potentialOutPath": "episode://1"' in line
        or '"potentialOutPath": "episode://3"' in line
    ]
    assert result.line_count == len(expected_lines)
    assert output_path.read_text(encoding="utf-8").splitlines(keepends=True) == expected_lines
    assert "===== Episode" not in output_path.read_text(encoding="utf-8")


def test_export_manifest_transcripts_raw_backwards_compat_writes_exact_dataset_rows_from_target_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compatibility_manifest_path = _write_manifest_with_actual_episode_numbers(
        tmp_path / "compatibility_longform_manifest.json",
        sidecar=False,
        actual_episode_numbers=[10, 20, 30],
    )
    target_manifest_path = _write_manifest_with_actual_episode_numbers(
        tmp_path / "target_original_manifest.json",
        sidecar=True,
        actual_episode_numbers=[5, 10, 20, 25, 30],
    )
    target_manifest_payload = json.loads(target_manifest_path.read_text(encoding="utf-8"))
    turn_data_path = Path(str(target_manifest_payload["turn_data_path"]))
    output_path = tmp_path / "compat_raw_rows.jsonl"

    monkeypatch.setattr(
        "card_framework.benchmark.summarizer_critic.sporc.export_transcripts.DEFAULT_LONGFORM_MANIFEST_PATH",
        compatibility_manifest_path,
    )

    result = export_manifest_transcripts(
        manifest_path=target_manifest_path,
        episode_spec="1-2",
        output_path=output_path,
        backwards_compat=True,
        raw=True,
    )

    expected_lines = [
        line
        for line in _read_gzip_lines(turn_data_path)
        if '"potentialOutPath": "episode://actual/10"' in line
        or '"potentialOutPath": "episode://actual/20"' in line
    ]
    assert result.line_count == len(expected_lines)
    assert output_path.read_text(encoding="utf-8").splitlines(keepends=True) == expected_lines
