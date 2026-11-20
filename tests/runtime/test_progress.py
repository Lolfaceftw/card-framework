"""Tests for following a pipeline run through the files each stage writes."""

from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

from card_framework.runtime.progress import (
    RunState,
    count_completed,
    pipeline_stages,
    run_with_progress,
    stage_note,
)
from card_framework.shared.stage_view import PlainStageView

_FULL_CONFIG = {
    "audio": {
        "voice_clone": {"enabled": True},
        "interjector": {"enabled": True},
    }
}


def test_pipeline_stages_follow_the_config(tmp_path: Path) -> None:
    """List only the stages the config will run, each with its output file."""
    full = pipeline_stages(tmp_path, _FULL_CONFIG)
    text_only = pipeline_stages(tmp_path, {"audio": {"voice_clone": {"enabled": False}}})

    assert [stage.artifact.name for stage in full] == [
        "transcript.json",
        "summary.xml",
        "voice_cloned.wav",
        "voice_cloned_interjected.wav",
    ]
    assert full[2].artifact == tmp_path / "audio_stage" / "voice_clone" / "voice_cloned.wav"
    assert [stage.key for stage in text_only] == ["stage-1", "stage-2"]


def test_count_completed_ignores_files_from_earlier_runs(tmp_path: Path) -> None:
    """Count a stage only when its file was written during this run, in order."""
    stages = pipeline_stages(tmp_path, _FULL_CONFIG)
    transcript = stages[0].artifact
    summary = stages[1].artifact
    transcript.write_text("{}", encoding="utf-8")
    summary.write_text("<summary/>", encoding="utf-8")
    old = time.time() - 3600
    os.utime(transcript, (old, old))

    since = time.time() - 60

    assert count_completed(stages, since) == 0
    transcript.touch()
    assert count_completed(stages, since) == 2


def test_stage_notes_describe_transcript_and_summary(tmp_path: Path) -> None:
    """Summarise finished stages the way the stage list shows them."""
    stages = pipeline_stages(tmp_path, _FULL_CONFIG)
    stages[0].artifact.write_text(
        json.dumps(
            {
                "segments": [
                    {"speaker": "SPEAKER_00", "start_time": 0, "end_time": 4000},
                    {"speaker": "SPEAKER_01", "start_time": 4000, "end_time": 103000},
                ]
            }
        ),
        encoding="utf-8",
    )
    stages[1].artifact.write_text(
        '<SPEAKER_00 emotion_categories="calm">Hi.</SPEAKER_00>\n'
        "<SPEAKER_01>Hello.</SPEAKER_01>\n",
        encoding="utf-8",
    )

    assert stage_note(stages[0], 30) == "2 speakers, 1:43 of speech"
    assert stage_note(stages[1], 30) == "2 turns"
    assert stage_note(stages[2], 30) == ""


class _RecordingView:
    def __init__(self) -> None:
        self.completed: list[int] = []
        self.closed: bool | None = None

    def update(self, state: RunState) -> None:
        self.completed.append(state.completed)

    def close(self, state: RunState, *, succeeded: bool) -> None:
        self.closed = succeeded


def test_run_with_progress_logs_output_and_tracks_stages(tmp_path: Path) -> None:
    """Send pipeline output to the log while stages tick off as files appear."""
    stages = pipeline_stages(tmp_path, {"audio": {"voice_clone": {"enabled": False}}})
    script = (
        "import pathlib, time\n"
        f"root = pathlib.Path({str(tmp_path)!r})\n"
        "print('loading models')\n"
        "(root / 'transcript.json').write_text('{}')\n"
        "time.sleep(0.3)\n"
        "(root / 'summary.xml').write_text('<summary/>')\n"
        "print('summary written')\n"
    )
    state = RunState(stages=stages, target_seconds=30, log_path=tmp_path / "card.log")
    view = _RecordingView()

    returncode = run_with_progress(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=dict(os.environ),
        state=state,
        view=view,
        poll_seconds=0.05,
    )

    assert returncode == 0
    assert state.completed == 2
    assert view.completed[-1] == 2
    log_text = (tmp_path / "card.log").read_text(encoding="utf-8")
    assert "loading models" in log_text and "summary written" in log_text


def test_run_with_progress_returns_the_failure_exit_code(tmp_path: Path) -> None:
    """Report a failed pipeline through its exit code without raising."""
    state = RunState(
        stages=pipeline_stages(tmp_path, {}),
        target_seconds=30,
        log_path=tmp_path / "card.log",
    )

    returncode = run_with_progress(
        [sys.executable, "-c", "import sys; print('boom'); sys.exit(3)"],
        cwd=tmp_path,
        env=dict(os.environ),
        state=state,
        view=_RecordingView(),
        poll_seconds=0.05,
    )

    assert returncode == 3
    assert state.completed == 0


def test_plain_view_prints_one_ascii_line_per_change(tmp_path: Path) -> None:
    """Write line-based progress that survives pipes and non-UTF-8 consoles."""
    stages = pipeline_stages(tmp_path, {})
    log_path = tmp_path / "card.log"
    log_path.write_text(
        "step one\nTraceback (most recent call last):\nRuntimeError: summarizer got HTTP 400\nexit\n",
        encoding="utf-8",
    )
    state = RunState(stages=stages, target_seconds=30, log_path=log_path)
    stream = io.StringIO()
    view = PlainStageView(stream)

    view.update(state)
    state.setup_message = "Loading IndexTTS runtime"
    view.update(state)
    state.setup_done_at = state.started_at + 2
    state.stage_finished_at.append(state.setup_done_at + 5)
    state.notes["stage-1"] = "2 speakers"
    view.update(state)
    view.close(state, succeeded=False)

    lines = stream.getvalue().splitlines()
    assert lines[0] == "card-framework: setup: Checking your setup"
    assert lines[1] == "card-framework: setup: Loading IndexTTS runtime"
    assert lines[2] == "card-framework: setup done in 0:02"
    assert lines[3] == "card-framework: stage-1 done: transcript.json (2 speakers) in 0:05"
    assert "card-framework: cause: RuntimeError: summarizer got HTTP 400" in lines
    assert "card-framework:   | Traceback (most recent call last):" in lines
    assert lines[-1] == f"card-framework: full log: {log_path}"
    assert stream.getvalue().isascii()
