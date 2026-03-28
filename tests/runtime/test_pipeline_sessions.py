from __future__ import annotations

import json
from pathlib import Path
import sqlite3

from card_framework.runtime.pipeline_sessions import PipelineSessionStore


def _write_transcript(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "segments": [
                    {
                        "speaker": "SPEAKER_00",
                        "start_time": 0,
                        "end_time": 1000,
                        "text": "hello world",
                    }
                ],
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_summary(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("<SPEAKER_00>Hello world</SPEAKER_00>\n", encoding="utf-8")


def _write_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path = path.parent / "turn_001.wav"
    merged_output_path = path.parent / "merged.wav"
    artifact_path.write_bytes(b"RIFF")
    merged_output_path.write_bytes(b"RIFF")
    path.write_text(
        json.dumps(
            {
                "artifacts": [{"output_audio_path": artifact_path.name}],
                "merged_output_audio_path": merged_output_path.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_compute_resume_plan_uses_first_missing_required_stage(tmp_path: Path) -> None:
    store = PipelineSessionStore(tmp_path / "sessions.sqlite3")
    work_dir = tmp_path / "artifacts" / "audio_stage" / "runs" / "session-1"
    transcript_path = tmp_path / "artifacts" / "transcripts" / "session-1.transcript.json"
    summary_path = tmp_path / "artifacts" / "pipeline_sessions" / "session-1" / "summary.xml"
    voice_clone_manifest_path = work_dir / "voice_clone" / "manifest.json"
    interjector_manifest_path = work_dir / "interjector" / "interjector_manifest.json"
    store.create_or_replace_session(
        session_id="session-1",
        requested_start_stage="stage-1",
        prepared_overrides=["pipeline.start_stage=stage-1"],
        artifacts={
            "work_dir": str(work_dir),
            "transcript_path": str(transcript_path),
            "workspace_summary_path": str(tmp_path / "summary.xml"),
            "session_summary_path": str(summary_path),
            "voice_clone_manifest_path": str(voice_clone_manifest_path),
            "interjector_manifest_path": str(interjector_manifest_path),
        },
        options={
            "voice_clone_enabled": True,
            "interjector_enabled": True,
        },
    )
    _write_transcript(transcript_path)
    _write_summary(summary_path)

    plan = store.compute_resume_plan(session_id="session-1")

    assert plan.resume_stage == "stage-3"
    assert plan.stage_validity == {
        "stage-1": True,
        "stage-2": True,
        "stage-3": False,
        "stage-4": False,
    }


def test_cleanup_for_stage_restart_removes_stage_specific_outputs(tmp_path: Path) -> None:
    store = PipelineSessionStore(tmp_path / "sessions.sqlite3")
    work_dir = tmp_path / "artifacts" / "audio_stage" / "runs" / "session-2"
    transcript_path = tmp_path / "artifacts" / "transcripts" / "session-2.transcript.json"
    summary_path = tmp_path / "artifacts" / "pipeline_sessions" / "session-2" / "summary.xml"
    voice_clone_manifest_path = work_dir / "voice_clone" / "manifest.json"
    interjector_manifest_path = work_dir / "interjector" / "interjector_manifest.json"
    store.create_or_replace_session(
        session_id="session-2",
        requested_start_stage="stage-1",
        prepared_overrides=["pipeline.start_stage=stage-1"],
        artifacts={
            "work_dir": str(work_dir),
            "transcript_path": str(transcript_path),
            "workspace_summary_path": str(tmp_path / "summary.xml"),
            "session_summary_path": str(summary_path),
            "voice_clone_manifest_path": str(voice_clone_manifest_path),
            "interjector_manifest_path": str(interjector_manifest_path),
        },
        options={
            "voice_clone_enabled": True,
            "interjector_enabled": True,
        },
    )
    _write_transcript(transcript_path)
    _write_summary(summary_path)
    _write_manifest(voice_clone_manifest_path)
    _write_manifest(interjector_manifest_path)

    store.cleanup_for_stage_restart(session_id="session-2", stage_name="stage-2")

    assert transcript_path.exists()
    assert not summary_path.exists()
    assert not voice_clone_manifest_path.parent.exists()
    assert not interjector_manifest_path.parent.exists()


def test_mark_session_failed_infers_current_stage_for_attempt_update(
    tmp_path: Path,
) -> None:
    store = PipelineSessionStore(tmp_path / "sessions.sqlite3")
    work_dir = tmp_path / "artifacts" / "audio_stage" / "runs" / "session-3"
    transcript_path = tmp_path / "artifacts" / "transcripts" / "session-3.transcript.json"
    summary_path = tmp_path / "artifacts" / "pipeline_sessions" / "session-3" / "summary.xml"
    voice_clone_manifest_path = work_dir / "voice_clone" / "manifest.json"
    interjector_manifest_path = work_dir / "interjector" / "interjector_manifest.json"
    store.create_or_replace_session(
        session_id="session-3",
        requested_start_stage="stage-1",
        prepared_overrides=["pipeline.start_stage=stage-1"],
        artifacts={
            "work_dir": str(work_dir),
            "transcript_path": str(transcript_path),
            "workspace_summary_path": str(tmp_path / "summary.xml"),
            "session_summary_path": str(summary_path),
            "voice_clone_manifest_path": str(voice_clone_manifest_path),
            "interjector_manifest_path": str(interjector_manifest_path),
        },
        options={
            "voice_clone_enabled": True,
            "interjector_enabled": False,
        },
    )
    store.mark_stage_started(session_id="session-3", stage_name="stage-3")

    store.mark_session_failed(
        session_id="session-3",
        current_stage=None,
        reason="voice clone crashed",
    )

    record = store.get_session("session-3")
    assert record is not None
    assert record.status == "failed"
    assert record.resume_stage == "stage-3"
    assert record.last_error == "voice clone crashed"

    with sqlite3.connect(tmp_path / "sessions.sqlite3") as connection:
        row = connection.execute(
            """
            SELECT status, error_text
            FROM stage_attempts
            WHERE session_id = ?
            ORDER BY attempt_id DESC
            LIMIT 1
            """,
            ("session-3",),
        ).fetchone()
    assert row == ("failed", "voice clone crashed")
