"""Coverage for SPoRC sweep bottleneck analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from card_framework.benchmark.summarizer_critic.sporc.sweep_analysis import (
    analyze_sweep,
)


def _write_log(path: Path) -> None:
    rows = [
        {
            "sporc_progress_event": "run_start",
            "duration_preset": "30s",
            "run_id": "run-1",
            "sample_count": 1,
        },
        {
            "sporc_progress_event": "sample_start",
            "duration_preset": "30s",
            "sample_index": 1,
            "sample_count": 1,
            "sample_id": "sample-1",
            "target_seconds": 30,
        },
        {
            "sporc_progress_event": "sample_heartbeat",
            "duration_preset": "30s",
            "sample_index": 1,
            "sample_count": 1,
            "sample_id": "sample-1",
            "target_seconds": 30,
            "stage": "index",
            "iteration": None,
            "started_at_utc": "2026-03-19T02:00:00+00:00",
            "last_activity_utc": "2026-03-19T02:00:10+00:00",
            "elapsed_sample_seconds": 10.0,
            "last_activity_age_seconds": 1.0,
        },
        {
            "sporc_progress_event": "sample_heartbeat",
            "duration_preset": "30s",
            "sample_index": 1,
            "sample_count": 1,
            "sample_id": "sample-1",
            "target_seconds": 30,
            "stage": "summarizer",
            "iteration": 1,
            "started_at_utc": "2026-03-19T02:00:00+00:00",
            "last_activity_utc": "2026-03-19T02:00:40+00:00",
            "elapsed_sample_seconds": 40.0,
            "last_activity_age_seconds": 2.0,
        },
        {
            "sporc_progress_event": "sample_heartbeat",
            "duration_preset": "30s",
            "sample_index": 1,
            "sample_count": 1,
            "sample_id": "sample-1",
            "target_seconds": 30,
            "stage": "critic",
            "iteration": 1,
            "started_at_utc": "2026-03-19T02:00:00+00:00",
            "last_activity_utc": "2026-03-19T02:01:10+00:00",
            "elapsed_sample_seconds": 70.0,
            "last_activity_age_seconds": 3.0,
        },
        {
            "sporc_progress_event": "sample_done",
            "duration_preset": "30s",
            "sample_index": 1,
            "sample_count": 1,
            "sample_id": "sample-1",
            "final_status": "pass",
            "counts_toward_target": True,
            "composite": 0.8,
            "duration_seconds": 100.0,
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )


def _write_checkpoint(path: Path) -> None:
    payload = {
        "completed_results": [
            {
                "sample_id": "sample-1",
                "dataset": "sporc_full",
                "episode_key": "episode-1",
                "status": "completed",
                "target_seconds": 30,
                "candidate_seconds": 28.5,
                "converged": True,
                "iterations_run": 1,
                "final_status": "pass",
                "duration_seconds": 100.0,
                "retrieval_events": 1,
                "tool_invocations": 2,
                "metrics": {"composite": 0.8},
                "counts_toward_target": True,
            }
        ],
        "active_sample": None,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_queue_db(path: Path, transcript_path: Path) -> None:
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            """
            CREATE TABLE sample_queue (
                sample_id TEXT PRIMARY KEY,
                sample_position INTEGER NOT NULL,
                sample_payload_json TEXT,
                status TEXT NOT NULL,
                owner_label TEXT,
                owner_pid INTEGER,
                final_status TEXT,
                counts_toward_target INTEGER NOT NULL DEFAULT 0,
                replacement_reason TEXT,
                updated_at_utc TEXT NOT NULL
            )
            """
        )
        payload = {
            "sample_id": "sample-1",
            "dataset": "sporc_full",
            "episode_key": "episode-1",
            "transcript_path": str(transcript_path),
            "reference_summary": "reference",
            "target_seconds": 30,
            "metadata": {},
        }
        connection.execute(
            """
            INSERT INTO sample_queue(
                sample_id,
                sample_position,
                sample_payload_json,
                status,
                owner_label,
                owner_pid,
                final_status,
                counts_toward_target,
                replacement_reason,
                updated_at_utc
            ) VALUES (?, ?, ?, 'completed', NULL, NULL, 'pass', 1, NULL, ?)
            """,
            (
                "sample-1",
                1,
                json.dumps(payload),
                "2026-03-19T02:02:00+00:00",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def test_analyze_sweep_emits_graphs_and_quantitative_exports(tmp_path: Path) -> None:
    sweep_root = tmp_path / "sweep"
    logs_root = sweep_root / "logs"
    checkpoints_root = sweep_root / "checkpoints"
    queues_root = sweep_root / "queues" / "30s"
    logs_root.mkdir(parents=True)
    checkpoints_root.mkdir(parents=True)
    queues_root.mkdir(parents=True)

    transcript_path = tmp_path / "sample-1.transcript.json"
    transcript_path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "text": "hello world from the benchmark transcript",
                    },
                    {
                        "speaker": "SPEAKER_01",
                        "text": "second speaker line",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    _write_log(logs_root / "30s.shard-1-of-1.log")
    _write_checkpoint(checkpoints_root / "30s.shard-1-of-1.checkpoint.json")
    _write_queue_db(queues_root / "sample_queue.sqlite3", transcript_path)

    status_messages: list[str] = []
    result = analyze_sweep(
        sweep_root=sweep_root,
        output_dir=sweep_root / "analysis",
        reporter=status_messages.append,
    )

    assert Path(result.summary_markdown_path).exists()
    assert Path(result.summary_json_path).exists()
    assert all(Path(path).exists() for path in result.graphs)
    assert all(Path(path).exists() for path in result.csv_exports)

    summary_json = json.loads(Path(result.summary_json_path).read_text(encoding="utf-8"))
    assert summary_json["sample_records"][0]["sample_id"] == "sample-1"
    assert summary_json["sample_records"][0]["transcript_words"] == 9
    assert summary_json["sample_records"][0]["index_duration_seconds"] > 0
    assert any(
        message.startswith("Loading SPoRC sweep logs")
        for message in status_messages
    )
    assert any(
        message.startswith("Loaded analysis summaries")
        for message in status_messages
    )
