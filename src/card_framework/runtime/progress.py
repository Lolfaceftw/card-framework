"""Follow a running pipeline through the files each stage writes.

The pipeline runs in a subprocess and doesn't publish structured progress
events. Every stage does finish by writing one known file, so progress is the
number of those files that appeared after the run started. That keeps this
module independent of the pipeline internals.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from card_framework.shared.durations import format_duration

_SPEAKER_TAG = re.compile(r"<SPEAKER_[^\s>/]+")
_EXCEPTION_LINE = re.compile(r"^(?:[A-Za-z_][\w]*\.)*[A-Za-z_]\w*(?:Error|Exception): ")
_MAX_CAUSE_LENGTH = 300


@dataclass(slots=True, frozen=True)
class Stage:
    """One pipeline stage and the file it writes when it finishes."""

    key: str
    artifact: Path
    activity: str


@dataclass(slots=True)
class RunState:
    """Everything a progress view needs to draw the current moment of a run."""

    stages: tuple[Stage, ...]
    target_seconds: int
    log_path: Path | None = None
    started_at: float = field(default_factory=time.monotonic)
    setup_message: str = "Checking your setup"
    setup_done_at: float | None = None
    stage_finished_at: list[float] = field(default_factory=list)
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def completed(self) -> int:
        """Return how many stages have finished, in order."""
        return len(self.stage_finished_at)

    def stage_started_at(self, index: int) -> float | None:
        """Return when stage ``index`` started, or None if it hasn't."""
        if self.setup_done_at is None or index > self.completed:
            return None
        if index == 0:
            return self.setup_done_at
        return self.stage_finished_at[index - 1]


class StageView(Protocol):
    """Anything that can draw a :class:`RunState` for a person watching."""

    def update(self, state: RunState) -> None:
        """Redraw for the latest state."""

    def close(self, state: RunState, *, succeeded: bool) -> None:
        """Draw the final state and release the terminal."""


def pipeline_stages(output_dir: Path, config: Mapping[str, Any]) -> tuple[Stage, ...]:
    """Return the stages this config will run, each with the file it writes."""
    audio_cfg = _mapping(config.get("audio"))
    voice_clone_cfg = _mapping(audio_cfg.get("voice_clone"))
    interjector_cfg = _mapping(audio_cfg.get("interjector"))
    work_dir = output_dir / "audio_stage"

    stages = [
        Stage("stage-1", output_dir / "transcript.json", "Transcribing and finding speakers"),
        Stage("stage-2", output_dir / "summary.xml", "Writing a summary that fits the time"),
    ]
    if voice_clone_cfg.get("enabled", False):
        stages.append(
            Stage(
                "stage-3",
                work_dir
                / str(voice_clone_cfg.get("output_dir_name", "voice_clone"))
                / str(voice_clone_cfg.get("merged_output_filename", "voice_cloned.wav")),
                "Speaking each turn in the original voice",
            )
        )
    if voice_clone_cfg.get("enabled", False) and interjector_cfg.get("enabled", False):
        stages.append(
            Stage(
                "stage-4",
                work_dir
                / str(interjector_cfg.get("output_dir_name", "interjector"))
                / str(
                    interjector_cfg.get(
                        "merged_output_filename", "voice_cloned_interjected.wav"
                    )
                ),
                "Adding backchannels and overlap",
            )
        )
    return tuple(stages)


def count_completed(stages: Sequence[Stage], since_epoch: float) -> int:
    """Return how many stages, in order, wrote their file at or after ``since_epoch``."""
    completed = 0
    for stage in stages:
        try:
            if stage.artifact.stat().st_mtime < since_epoch:
                break
        except OSError:
            break
        completed += 1
    return completed


def stage_note(stage: Stage, target_seconds: int) -> str:
    """Return a short human note about a finished stage's file, or "" if unreadable."""
    try:
        if stage.key == "stage-1":
            return _transcript_note(stage.artifact)
        if stage.key == "stage-2":
            turns = len(_SPEAKER_TAG.findall(stage.artifact.read_text(encoding="utf-8")))
            return f"{turns} turns" if turns else ""
        return _audio_note(stage.artifact, target_seconds)
    except (OSError, ValueError, RuntimeError):
        return ""


def run_with_progress(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    state: RunState,
    view: StageView,
    poll_seconds: float = 0.5,
) -> int:
    """Run the pipeline subprocess, logging to ``state.log_path`` while redrawing ``view``.

    Returns:
        The subprocess exit code.
    """
    if state.log_path is None:
        raise ValueError("run_with_progress needs state.log_path for the pipeline log.")
    state.log_path.parent.mkdir(parents=True, exist_ok=True)
    since_epoch = time.time()
    if state.setup_done_at is None:
        state.setup_done_at = time.monotonic()
    returncodes: list[int] = []
    errors: list[BaseException] = []

    with state.log_path.open("w", encoding="utf-8", errors="replace") as log_file:

        def _run() -> None:
            try:
                completed = subprocess.run(
                    list(command),
                    cwd=cwd,
                    check=False,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=dict(env),
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            except BaseException as exc:  # re-raised in the calling thread below
                errors.append(exc)
            else:
                returncodes.append(completed.returncode)

        worker = threading.Thread(target=_run, name="card-pipeline", daemon=True)
        worker.start()
        while worker.is_alive():
            _record_progress(state, since_epoch)
            view.update(state)
            worker.join(poll_seconds)

    if errors:
        raise errors[0]
    _record_progress(state, since_epoch)
    view.update(state)
    return returncodes[0]


def tail_lines(path: Path, count: int = 15) -> list[str]:
    """Return the last ``count`` non-empty lines of a text file."""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    return [line for line in lines if line.strip()][-count:]


def failure_cause(path: Path) -> str:
    """Return the last ``SomeError: message`` line in a log, or "" if there is none."""
    for line in reversed(tail_lines(path, count=400)):
        stripped = line.strip()
        if _EXCEPTION_LINE.match(stripped):
            return stripped[:_MAX_CAUSE_LENGTH]
    return ""


def _record_progress(state: RunState, since_epoch: float) -> None:
    completed = count_completed(state.stages, since_epoch)
    while state.completed < completed:
        stage = state.stages[state.completed]
        state.notes[stage.key] = stage_note(stage, state.target_seconds)
        state.stage_finished_at.append(time.monotonic())


def _transcript_note(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    segments = payload.get("segments", []) if isinstance(payload, dict) else []
    speakers = {str(segment.get("speaker", "")) for segment in segments if segment.get("speaker")}
    end_ms = max((int(segment.get("end_time", 0)) for segment in segments), default=0)
    parts = []
    if speakers:
        parts.append(f"{len(speakers)} speaker" + ("" if len(speakers) == 1 else "s"))
    if end_ms:
        parts.append(f"{format_duration(end_ms / 1000)} of speech")
    return ", ".join(parts)


def _audio_note(path: Path, target_seconds: int) -> str:
    import soundfile

    seconds = soundfile.info(str(path)).duration
    return f"{format_duration(seconds)} for a {format_duration(target_seconds)} target"


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
