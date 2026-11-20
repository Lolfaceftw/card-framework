"""Interactive human transcript sampling workflow for SPoRC research sessions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.prompt import FloatPrompt, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from card_framework.agents.utils import count_words
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_HUMAN_SAMPLING_ROOT,
    DEFAULT_LONGFORM_MANIFEST_PATH,
    DURATION_PRESET_SECONDS,
    PreparedBenchmarkSample,
)
from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    iter_prepared_dataset_samples,
    load_prepared_dataset_manifest,
)
from card_framework.orchestration.transcript import Transcript

_DEFAULT_EPISODES = 20
_DEFAULT_WORDS_PER_MINUTE = 170.0
_DEFAULT_SAMPLE_FACTOR = 5.0
_DEFAULT_MAX_WINDOWS_PER_EPISODE = 10
_SESSION_DATABASE_FILENAME = "session.sqlite3"
_SESSION_SUMMARY_FILENAME = "session_summary.json"
_SELECTED_SAMPLES_JSONL_FILENAME = "selected_samples.jsonl"
_SELECTED_SAMPLES_CSV_FILENAME = "selected_samples.csv"
_OVERRIDE_PATTERN = re.compile(r"^\[(?P<speaker>[^\]]+)\]\s*(?P<text>.+)$")


@dataclass(slots=True, frozen=True)
class TranscriptLine:
    """Represent one transcript line shown to a human sampler."""

    line_number: int
    speaker: str
    text: str
    word_count: int
    start_time: int
    end_time: int


@dataclass(slots=True, frozen=True)
class WindowPlan:
    """Describe one contiguous transcript window for a sampling step."""

    window_index: int
    start_line_number: int
    end_line_number: int
    word_count: int
    max_samples: int


@dataclass(slots=True, frozen=True)
class EpisodePlan:
    """Store the precomputed sampling plan for one manifest episode."""

    sample_id: str
    episode_key: str
    episode_title: str
    transcript_path: Path
    transcript_word_count: int
    segment_count: int
    target_words: int
    target_samples: int
    windows: tuple[WindowPlan, ...]


@dataclass(slots=True, frozen=True)
class _SessionPaths:
    session_root: Path
    database_path: Path
    summary_path: Path
    selected_samples_jsonl_path: Path
    selected_samples_csv_path: Path


@dataclass(slots=True, frozen=True)
class ParsedSelectionToken:
    """Represent one parsed line-number or override selection."""

    selection_type: str
    line_number: int | None
    speaker: str
    text: str
    word_count: int
    start_time: int | None
    end_time: int | None


def run_human_sampling_session(
    args: argparse.Namespace,
    *,
    console: Console | None = None,
) -> int:
    """Run the interactive Rich-based transcript sampling workflow."""
    active_console = console if console is not None else Console()
    sampling_root = _resolve_sampling_root(str(getattr(args, "sampling_root", "") or ""))
    resume_session = str(getattr(args, "resume_session", "") or "").strip()
    connection: sqlite3.Connection | None = None
    try:
        if resume_session:
            session_paths = _session_paths(
                sampling_root=sampling_root,
                session_id=resume_session,
            )
            connection = _open_session_connection(
                session_paths.database_path,
                initialize_schema=False,
            )
            if _load_session_row(connection, resume_session) is None:
                raise RuntimeError(
                    f"SPoRC human sampling session does not exist: {resume_session}"
                )
            _set_session_status(connection, session_id=resume_session, status="active")
            _write_session_exports(
                connection,
                session_id=resume_session,
                session_paths=session_paths,
            )
            _render_resume_banner(
                console=active_console,
                session_id=resume_session,
                session_paths=session_paths,
            )
            return _run_sampling_loop(
                connection,
                session_id=resume_session,
                session_paths=session_paths,
                console=active_console,
            )

        session_id, session_paths, connection = _create_new_session(
            args,
            console=active_console,
            sampling_root=sampling_root,
        )
        _render_session_created_banner(
            console=active_console,
            session_id=session_id,
            session_paths=session_paths,
            connection=connection,
        )
        return _run_sampling_loop(
            connection,
            session_id=session_id,
            session_paths=session_paths,
            console=active_console,
        )
    finally:
        if connection is not None:
            connection.close()


def _create_new_session(
    args: argparse.Namespace,
    *,
    console: Console,
    sampling_root: Path,
) -> tuple[str, _SessionPaths, sqlite3.Connection]:
    manifest_path = Path(
        str(getattr(args, "dataset_manifest", "") or DEFAULT_LONGFORM_MANIFEST_PATH)
    ).resolve()
    if not manifest_path.exists():
        raise RuntimeError(
            "SPoRC human sampling requires an existing manifest. "
            f"Expected: {manifest_path}. Run `curate-longform` first or pass "
            "`--dataset-manifest`."
        )

    preset = _resolve_duration_preset(args, console=console)
    episodes_requested = _resolve_episode_count(args, console=console)
    words_per_minute = _resolve_words_per_minute(args, console=console)
    sample_factor = _resolve_sample_factor(args, console=console)
    max_windows_per_episode = _resolve_max_windows_per_episode(args, console=console)
    preset_seconds = DURATION_PRESET_SECONDS[preset]
    per_episode_target_words = _compute_target_words(
        preset_seconds,
        words_per_minute=words_per_minute,
        sample_factor=sample_factor,
    )
    manifest = load_prepared_dataset_manifest(
        manifest_path,
        include_samples=False,
    )
    if episodes_requested > int(manifest.sample_count):
        raise RuntimeError(
            "Requested episode prefix exceeds the available manifest length: "
            f"{episodes_requested} > {manifest.sample_count}."
        )
    prepared_samples = tuple(
        iter_prepared_dataset_samples(
            manifest_path,
            max_samples=episodes_requested,
        )
    )
    if len(prepared_samples) != episodes_requested:
        raise RuntimeError(
            "Unable to load the requested SPoRC episode prefix from the manifest."
        )

    session_id = str(uuid4())
    session_paths = _session_paths(sampling_root=sampling_root, session_id=session_id)
    session_paths.session_root.mkdir(parents=True, exist_ok=True)
    connection = _open_session_connection(
        session_paths.database_path,
        initialize_schema=True,
    )
    created_at_utc = _utc_now()
    with connection:
        connection.execute(
            """
            INSERT INTO session_config (
                session_id,
                created_at_utc,
                updated_at_utc,
                status,
                manifest_path,
                preset,
                preset_seconds,
                episodes_requested,
                words_per_minute,
                sample_factor,
                per_episode_target_words,
                max_windows_per_episode,
                sampling_root
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                created_at_utc,
                created_at_utc,
                "active",
                str(manifest_path),
                preset,
                preset_seconds,
                episodes_requested,
                words_per_minute,
                sample_factor,
                per_episode_target_words,
                max_windows_per_episode,
                str(sampling_root),
            ),
        )
        for episode_order, sample in enumerate(prepared_samples, start=1):
            episode_plan = _build_episode_plan(
                sample,
                target_words=per_episode_target_words,
                max_windows_per_episode=max_windows_per_episode,
            )
            connection.execute(
                """
                INSERT INTO episodes (
                    session_id,
                    episode_order,
                    sample_id,
                    episode_key,
                    episode_title,
                    transcript_path,
                    transcript_word_count,
                    segment_count,
                    target_words,
                    target_samples,
                    window_count,
                    status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    episode_order,
                    episode_plan.sample_id,
                    episode_plan.episode_key,
                    episode_plan.episode_title,
                    str(episode_plan.transcript_path),
                    episode_plan.transcript_word_count,
                    episode_plan.segment_count,
                    episode_plan.target_words,
                    episode_plan.target_samples,
                    len(episode_plan.windows),
                    "pending",
                ),
            )
            for window in episode_plan.windows:
                connection.execute(
                    """
                    INSERT INTO windows (
                        session_id,
                        episode_order,
                        window_index,
                        start_line_number,
                        end_line_number,
                        word_count,
                        max_samples,
                        status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        episode_order,
                        window.window_index,
                        window.start_line_number,
                        window.end_line_number,
                        window.word_count,
                        window.max_samples,
                        "pending",
                    ),
                )
    _write_session_exports(connection, session_id=session_id, session_paths=session_paths)
    return session_id, session_paths, connection


def _build_episode_plan(
    sample: PreparedBenchmarkSample,
    *,
    target_words: int,
    max_windows_per_episode: int,
) -> EpisodePlan:
    transcript_path = Path(sample.transcript_path).resolve()
    lines = _load_transcript_lines(transcript_path)
    if not lines:
        raise RuntimeError(f"Transcript contains no usable lines: {transcript_path}")
    transcript_word_count = sum(line.word_count for line in lines)
    target_samples = _estimate_target_sample_count(lines, target_words=target_words)
    window_count = max(
        1,
        min(
            len(lines),
            target_samples,
            max(1, int(max_windows_per_episode)),
        ),
    )
    window_ranges = _build_word_balanced_windows(lines, window_count=window_count)
    window_sample_limits = _distribute_evenly(target_samples, bucket_count=window_count)
    windows = tuple(
        WindowPlan(
            window_index=index,
            start_line_number=window_range[0].line_number,
            end_line_number=window_range[-1].line_number,
            word_count=sum(line.word_count for line in window_range),
            max_samples=window_sample_limits[index - 1],
        )
        for index, window_range in enumerate(window_ranges, start=1)
    )
    episode_title = str(sample.metadata.get("episode_title", "") or "").strip()
    if not episode_title:
        episode_title = sample.sample_id
    return EpisodePlan(
        sample_id=sample.sample_id,
        episode_key=sample.episode_key,
        episode_title=episode_title,
        transcript_path=transcript_path,
        transcript_word_count=transcript_word_count,
        segment_count=len(lines),
        target_words=target_words,
        target_samples=target_samples,
        windows=windows,
    )


def _resolve_sampling_root(raw_value: str) -> Path:
    candidate = raw_value.strip()
    if candidate:
        return Path(candidate).resolve()
    return DEFAULT_HUMAN_SAMPLING_ROOT.resolve()


def _session_paths(*, sampling_root: Path, session_id: str) -> _SessionPaths:
    session_root = (sampling_root / session_id).resolve()
    return _SessionPaths(
        session_root=session_root,
        database_path=(session_root / _SESSION_DATABASE_FILENAME).resolve(),
        summary_path=(session_root / _SESSION_SUMMARY_FILENAME).resolve(),
        selected_samples_jsonl_path=(
            session_root / _SELECTED_SAMPLES_JSONL_FILENAME
        ).resolve(),
        selected_samples_csv_path=(
            session_root / _SELECTED_SAMPLES_CSV_FILENAME
        ).resolve(),
    )


def _open_session_connection(
    database_path: Path,
    *,
    initialize_schema: bool,
) -> sqlite3.Connection:
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if initialize_schema:
        _initialize_schema(connection)
    return connection


def _initialize_schema(connection: sqlite3.Connection) -> None:
    with connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS session_config (
                session_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL,
                status TEXT NOT NULL,
                manifest_path TEXT NOT NULL,
                preset TEXT NOT NULL,
                preset_seconds INTEGER NOT NULL,
                episodes_requested INTEGER NOT NULL,
                words_per_minute REAL NOT NULL,
                sample_factor REAL NOT NULL,
                per_episode_target_words INTEGER NOT NULL,
                max_windows_per_episode INTEGER NOT NULL,
                sampling_root TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS episodes (
                session_id TEXT NOT NULL,
                episode_order INTEGER NOT NULL,
                sample_id TEXT NOT NULL,
                episode_key TEXT NOT NULL,
                episode_title TEXT NOT NULL,
                transcript_path TEXT NOT NULL,
                transcript_word_count INTEGER NOT NULL,
                segment_count INTEGER NOT NULL,
                target_words INTEGER NOT NULL,
                target_samples INTEGER NOT NULL,
                window_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (session_id, episode_order),
                FOREIGN KEY (session_id) REFERENCES session_config(session_id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS windows (
                session_id TEXT NOT NULL,
                episode_order INTEGER NOT NULL,
                window_index INTEGER NOT NULL,
                start_line_number INTEGER NOT NULL,
                end_line_number INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                max_samples INTEGER NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (session_id, episode_order, window_index),
                FOREIGN KEY (session_id, episode_order)
                    REFERENCES episodes(session_id, episode_order)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS selections (
                selection_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                episode_order INTEGER NOT NULL,
                window_index INTEGER NOT NULL,
                selection_order INTEGER NOT NULL,
                selection_type TEXT NOT NULL,
                line_number INTEGER,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                word_count INTEGER NOT NULL,
                start_time INTEGER,
                end_time INTEGER,
                created_at_utc TEXT NOT NULL,
                UNIQUE (session_id, episode_order, window_index, selection_order),
                FOREIGN KEY (session_id, episode_order, window_index)
                    REFERENCES windows(session_id, episode_order, window_index)
                    ON DELETE CASCADE
            );
            """
        )


def _resolve_duration_preset(args: argparse.Namespace, *, console: Console) -> str:
    raw_preset = str(getattr(args, "duration_preset", "") or "").strip()
    if raw_preset:
        if raw_preset not in DURATION_PRESET_SECONDS:
            raise RuntimeError(
                f"Unsupported human sampling duration preset: {raw_preset}"
            )
        return raw_preset
    return Prompt.ask(
        "Duration preset",
        choices=list(DURATION_PRESET_SECONDS.keys()),
        default="30s",
        console=console,
    )


def _resolve_episode_count(args: argparse.Namespace, *, console: Console) -> int:
    raw_value = int(getattr(args, "episodes", 0) or 0)
    if raw_value > 0:
        return raw_value
    return int(
        IntPrompt.ask(
            "Episode count from the manifest prefix",
            default=_DEFAULT_EPISODES,
            console=console,
        )
    )


def _resolve_words_per_minute(args: argparse.Namespace, *, console: Console) -> float:
    raw_value = float(getattr(args, "words_per_minute", 0.0) or 0.0)
    if raw_value > 0:
        return raw_value
    return float(
        FloatPrompt.ask(
            "Words per minute",
            default=_DEFAULT_WORDS_PER_MINUTE,
            console=console,
        )
    )


def _resolve_sample_factor(args: argparse.Namespace, *, console: Console) -> float:
    raw_value = float(getattr(args, "sample_factor", 0.0) or 0.0)
    if raw_value > 0:
        return raw_value
    return float(
        FloatPrompt.ask(
            "Sampling factor multiplier",
            default=_DEFAULT_SAMPLE_FACTOR,
            console=console,
        )
    )


def _resolve_max_windows_per_episode(
    args: argparse.Namespace,
    *,
    console: Console,
) -> int:
    raw_value = int(getattr(args, "max_windows_per_episode", 0) or 0)
    if raw_value > 0:
        return raw_value
    return int(
        IntPrompt.ask(
            "Maximum windows per episode",
            default=_DEFAULT_MAX_WINDOWS_PER_EPISODE,
            console=console,
        )
    )


def _compute_target_words(
    preset_seconds: int,
    *,
    words_per_minute: float,
    sample_factor: float,
) -> int:
    return max(
        1,
        int(round(words_per_minute * (preset_seconds / 60.0) * sample_factor)),
    )


def _estimate_target_sample_count(
    lines: tuple[TranscriptLine, ...],
    *,
    target_words: int,
) -> int:
    usable_word_counts = [line.word_count for line in lines if line.word_count > 0]
    average_words_per_line = (
        sum(usable_word_counts) / len(usable_word_counts)
        if usable_word_counts
        else 1.0
    )
    estimated = int(round(target_words / average_words_per_line))
    return max(1, min(len(lines), estimated))


def _build_word_balanced_windows(
    lines: tuple[TranscriptLine, ...],
    *,
    window_count: int,
) -> tuple[tuple[TranscriptLine, ...], ...]:
    if window_count <= 0:
        raise ValueError("window_count must be greater than zero.")
    if window_count > len(lines):
        raise ValueError("window_count cannot exceed the number of transcript lines.")

    windows: list[tuple[TranscriptLine, ...]] = []
    remaining_effective_words = sum(max(line.word_count, 1) for line in lines)
    start_index = 0
    for windows_left in range(window_count, 0, -1):
        if windows_left == 1:
            window_lines = lines[start_index:]
            windows.append(window_lines)
            break

        target_words = remaining_effective_words / windows_left
        end_index = start_index
        current_effective_words = 0
        max_end_exclusive = len(lines) - (windows_left - 1)
        while end_index < max_end_exclusive:
            current_effective_words += max(lines[end_index].word_count, 1)
            end_index += 1
            if current_effective_words >= target_words:
                break
        if end_index <= start_index:
            end_index = start_index + 1
        window_lines = lines[start_index:end_index]
        windows.append(window_lines)
        remaining_effective_words -= sum(
            max(line.word_count, 1) for line in window_lines
        )
        start_index = end_index
    return tuple(windows)


def _distribute_evenly(total: int, *, bucket_count: int) -> tuple[int, ...]:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be greater than zero.")
    base = total // bucket_count
    remainder = total % bucket_count
    return tuple(base + (1 if index < remainder else 0) for index in range(bucket_count))


def _load_transcript_lines(transcript_path: Path) -> tuple[TranscriptLine, ...]:
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    transcript = Transcript.from_mapping(payload)
    lines: list[TranscriptLine] = []
    for line_number, segment in enumerate(transcript.segments, start=1):
        text = str(segment.text or "").strip()
        if not text:
            continue
        lines.append(
            TranscriptLine(
                line_number=line_number,
                speaker=segment.speaker,
                text=text,
                word_count=count_words(text),
                start_time=segment.start_time,
                end_time=segment.end_time,
            )
        )
    return tuple(lines)


def _run_sampling_loop(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    session_paths: _SessionPaths,
    console: Console,
) -> int:
    transcript_cache: dict[int, tuple[TranscriptLine, ...]] = {}
    while True:
        session_row = _require_session_row(connection, session_id)
        pending_window = _load_next_pending_window(connection, session_id=session_id)
        if pending_window is None:
            _set_session_status(connection, session_id=session_id, status="completed")
            _write_session_exports(
                connection,
                session_id=session_id,
                session_paths=session_paths,
            )
            _render_completed_banner(
                console=console,
                session_id=session_id,
                session_paths=session_paths,
                connection=connection,
            )
            return 0

        episode_order = int(pending_window["episode_order"])
        episode_row = _load_episode_row(
            connection,
            session_id=session_id,
            episode_order=episode_order,
        )
        if episode_row is None:
            raise RuntimeError(
                "SPoRC human sampling session is missing an episode row for a pending window."
            )
        lines = transcript_cache.get(episode_order)
        if lines is None:
            lines = _load_transcript_lines(Path(str(episode_row["transcript_path"])))
            transcript_cache[episode_order] = lines
        window_lines = tuple(
            line
            for line in lines
            if int(pending_window["start_line_number"])
            <= line.line_number
            <= int(pending_window["end_line_number"])
        )
        if not window_lines:
            raise RuntimeError(
                "SPoRC human sampling session contains an empty transcript window."
            )

        progress = _load_progress_snapshot(connection, session_id=session_id)
        episode_progress = _load_episode_progress(
            connection,
            session_id=session_id,
            episode_order=episode_order,
        )
        if console.is_terminal:
            console.clear()
        _render_window_screen(
            console=console,
            session_row=session_row,
            episode_row=episode_row,
            window_row=pending_window,
            progress=progress,
            episode_progress=episode_progress,
            window_lines=window_lines,
        )
        try:
            raw_selection = Prompt.ask(
                "Selections",
                console=console,
            )
        except KeyboardInterrupt:
            _set_session_status(connection, session_id=session_id, status="paused")
            _write_session_exports(
                connection,
                session_id=session_id,
                session_paths=session_paths,
            )
            _render_paused_banner(
                console=console,
                session_id=session_id,
                session_paths=session_paths,
            )
            return 130
        try:
            parsed_tokens = _parse_selection_input(
                raw_selection,
                max_samples=int(pending_window["max_samples"]),
                window_lines=window_lines,
            )
        except ValueError as exc:
            console.print(f"[bold red]Invalid selection:[/bold red] {exc}")
            continue

        if parsed_tokens == "__HELP__":
            _render_help_panel(console)
            continue
        if parsed_tokens == "__SKIP__":
            _persist_window_completion(
                connection,
                session_id=session_id,
                episode_order=episode_order,
                window_index=int(pending_window["window_index"]),
                selections=(),
            )
        else:
            _persist_window_completion(
                connection,
                session_id=session_id,
                episode_order=episode_order,
                window_index=int(pending_window["window_index"]),
                selections=tuple(parsed_tokens),
            )
        _write_session_exports(
            connection,
            session_id=session_id,
            session_paths=session_paths,
        )


def _load_session_row(connection: sqlite3.Connection, session_id: str) -> sqlite3.Row | None:
    return connection.execute(
        "SELECT * FROM session_config WHERE session_id = ?",
        (session_id,),
    ).fetchone()


def _require_session_row(connection: sqlite3.Connection, session_id: str) -> sqlite3.Row:
    row = _load_session_row(connection, session_id)
    if row is None:
        raise RuntimeError(f"SPoRC human sampling session does not exist: {session_id}")
    return row


def _load_episode_row(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    episode_order: int,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT *
        FROM episodes
        WHERE session_id = ? AND episode_order = ?
        """,
        (session_id, episode_order),
    ).fetchone()


def _load_next_pending_window(
    connection: sqlite3.Connection,
    *,
    session_id: str,
) -> sqlite3.Row | None:
    return connection.execute(
        """
        SELECT *
        FROM windows
        WHERE session_id = ? AND status = 'pending'
        ORDER BY episode_order ASC, window_index ASC
        LIMIT 1
        """,
        (session_id,),
    ).fetchone()


def _load_progress_snapshot(
    connection: sqlite3.Connection,
    *,
    session_id: str,
) -> dict[str, int]:
    total_windows = int(
        connection.execute(
            "SELECT COUNT(*) FROM windows WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
    )
    completed_windows = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM windows
            WHERE session_id = ? AND status = 'completed'
            """,
            (session_id,),
        ).fetchone()[0]
    )
    selected_samples = int(
        connection.execute(
            "SELECT COUNT(*) FROM selections WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
    )
    total_episodes = int(
        connection.execute(
            "SELECT COUNT(*) FROM episodes WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
    )
    completed_episodes = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM episodes
            WHERE session_id = ? AND status = 'completed'
            """,
            (session_id,),
        ).fetchone()[0]
    )
    return {
        "total_windows": total_windows,
        "completed_windows": completed_windows,
        "selected_samples": selected_samples,
        "total_episodes": total_episodes,
        "completed_episodes": completed_episodes,
    }


def _load_episode_progress(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    episode_order: int,
) -> dict[str, int]:
    completed_windows = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM windows
            WHERE session_id = ?
              AND episode_order = ?
              AND status = 'completed'
            """,
            (session_id, episode_order),
        ).fetchone()[0]
    )
    selected_samples = int(
        connection.execute(
            """
            SELECT COUNT(*)
            FROM selections
            WHERE session_id = ?
              AND episode_order = ?
            """,
            (session_id, episode_order),
        ).fetchone()[0]
    )
    return {
        "completed_windows": completed_windows,
        "selected_samples": selected_samples,
    }


def _parse_selection_input(
    raw_selection: str,
    *,
    max_samples: int,
    window_lines: tuple[TranscriptLine, ...],
) -> tuple[ParsedSelectionToken, ...] | str:
    raw_text = raw_selection.strip()
    if not raw_text:
        raise ValueError("Enter at least one line number, /override, /skip, or /help.")
    lowered = raw_text.lower()
    if lowered == "/help":
        return "__HELP__"
    if lowered == "/skip":
        return "__SKIP__"

    window_line_numbers = {line.line_number: line for line in window_lines}
    try:
        tokens = next(csv.reader([raw_text], skipinitialspace=True))
    except csv.Error as exc:
        raise ValueError(f"Unable to parse the selection list: {exc}") from exc
    normalized_tokens = [token.strip() for token in tokens if token.strip()]
    if not normalized_tokens:
        raise ValueError("Enter at least one line number or override segment.")
    if len(normalized_tokens) > max_samples:
        raise ValueError(
            f"This window accepts at most {max_samples} sample(s), not {len(normalized_tokens)}."
        )

    parsed_tokens: list[ParsedSelectionToken] = []
    seen_line_numbers: set[int] = set()
    for token in normalized_tokens:
        if token.lower() in {"/help", "/skip"}:
            raise ValueError("/help and /skip must be entered by themselves.")
        if token.startswith("/override"):
            parsed_tokens.append(_parse_override_token(token))
            continue
        try:
            line_number = int(token)
        except ValueError as exc:
            raise ValueError(
                "Selections must be transcript line numbers or /override <segment>."
            ) from exc
        if line_number in seen_line_numbers:
            raise ValueError(f"Line {line_number} is listed more than once.")
        line = window_line_numbers.get(line_number)
        if line is None:
            raise ValueError(
                f"Line {line_number} is not inside the current window "
                f"({window_lines[0].line_number}-{window_lines[-1].line_number})."
            )
        seen_line_numbers.add(line_number)
        parsed_tokens.append(
            ParsedSelectionToken(
                selection_type="line",
                line_number=line.line_number,
                speaker=line.speaker,
                text=line.text,
                word_count=line.word_count,
                start_time=line.start_time,
                end_time=line.end_time,
            )
        )
    return tuple(parsed_tokens)


def _parse_override_token(token: str) -> ParsedSelectionToken:
    raw_override = token[len("/override") :].strip()
    if not raw_override:
        raise ValueError("Override selections must include replacement text.")
    match = _OVERRIDE_PATTERN.match(raw_override)
    if match is None:
        speaker = "OVERRIDE"
        text = raw_override
    else:
        speaker = str(match.group("speaker")).strip() or "OVERRIDE"
        text = str(match.group("text")).strip()
    if not text:
        raise ValueError("Override selections must include replacement text.")
    return ParsedSelectionToken(
        selection_type="override",
        line_number=None,
        speaker=speaker,
        text=text,
        word_count=count_words(text),
        start_time=None,
        end_time=None,
    )


def _persist_window_completion(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    episode_order: int,
    window_index: int,
    selections: tuple[ParsedSelectionToken, ...],
) -> None:
    created_at_utc = _utc_now()
    with connection:
        connection.execute(
            """
            UPDATE windows
            SET status = 'completed'
            WHERE session_id = ?
              AND episode_order = ?
              AND window_index = ?
            """,
            (session_id, episode_order, window_index),
        )
        for selection_order, selection in enumerate(selections, start=1):
            connection.execute(
                """
                INSERT INTO selections (
                    session_id,
                    episode_order,
                    window_index,
                    selection_order,
                    selection_type,
                    line_number,
                    speaker,
                    text,
                    word_count,
                    start_time,
                    end_time,
                    created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    episode_order,
                    window_index,
                    selection_order,
                    selection.selection_type,
                    selection.line_number,
                    selection.speaker,
                    selection.text,
                    selection.word_count,
                    selection.start_time,
                    selection.end_time,
                    created_at_utc,
                ),
            )
        remaining_windows = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM windows
                WHERE session_id = ?
                  AND episode_order = ?
                  AND status = 'pending'
                """,
                (session_id, episode_order),
            ).fetchone()[0]
        )
        connection.execute(
            """
            UPDATE episodes
            SET status = ?
            WHERE session_id = ?
              AND episode_order = ?
            """,
            ("completed" if remaining_windows == 0 else "active", session_id, episode_order),
        )
        connection.execute(
            """
            UPDATE session_config
            SET updated_at_utc = ?,
                status = 'active'
            WHERE session_id = ?
            """,
            (created_at_utc, session_id),
        )


def _set_session_status(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    status: str,
) -> None:
    with connection:
        connection.execute(
            """
            UPDATE session_config
            SET status = ?,
                updated_at_utc = ?
            WHERE session_id = ?
            """,
            (status, _utc_now(), session_id),
        )


def _write_session_exports(
    connection: sqlite3.Connection,
    *,
    session_id: str,
    session_paths: _SessionPaths,
) -> None:
    summary_payload = _build_session_summary_payload(connection, session_id=session_id)
    session_paths.summary_path.write_text(
        json.dumps(summary_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    selection_rows = _load_selection_export_rows(connection, session_id=session_id)
    with session_paths.selected_samples_jsonl_path.open("w", encoding="utf-8") as handle:
        for row in selection_rows:
            handle.write(json.dumps(row) + "\n")
    fieldnames = list(selection_rows[0].keys()) if selection_rows else _selection_export_fieldnames()
    with session_paths.selected_samples_csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selection_rows:
            writer.writerow(row)


def _build_session_summary_payload(
    connection: sqlite3.Connection,
    *,
    session_id: str,
) -> dict[str, Any]:
    session_row = _require_session_row(connection, session_id)
    progress = _load_progress_snapshot(connection, session_id=session_id)
    episode_rows = connection.execute(
        """
        SELECT
            e.*,
            COALESCE(COUNT(s.selection_id), 0) AS selected_samples
        FROM episodes AS e
        LEFT JOIN selections AS s
          ON s.session_id = e.session_id
         AND s.episode_order = e.episode_order
        WHERE e.session_id = ?
        GROUP BY
            e.session_id,
            e.episode_order,
            e.sample_id,
            e.episode_key,
            e.episode_title,
            e.transcript_path,
            e.transcript_word_count,
            e.segment_count,
            e.target_words,
            e.target_samples,
            e.window_count,
            e.status
        ORDER BY e.episode_order ASC
        """,
        (session_id,),
    ).fetchall()
    return {
        "session_id": session_id,
        "status": str(session_row["status"]),
        "created_at_utc": str(session_row["created_at_utc"]),
        "updated_at_utc": str(session_row["updated_at_utc"]),
        "manifest_path": str(session_row["manifest_path"]),
        "preset": str(session_row["preset"]),
        "preset_seconds": int(session_row["preset_seconds"]),
        "episodes_requested": int(session_row["episodes_requested"]),
        "words_per_minute": float(session_row["words_per_minute"]),
        "sample_factor": float(session_row["sample_factor"]),
        "per_episode_target_words": int(session_row["per_episode_target_words"]),
        "max_windows_per_episode": int(session_row["max_windows_per_episode"]),
        "selected_sample_count": progress["selected_samples"],
        "completed_windows": progress["completed_windows"],
        "total_windows": progress["total_windows"],
        "completed_episodes": progress["completed_episodes"],
        "total_episodes": progress["total_episodes"],
        "episodes": [
            {
                "episode_order": int(row["episode_order"]),
                "sample_id": str(row["sample_id"]),
                "episode_key": str(row["episode_key"]),
                "episode_title": str(row["episode_title"]),
                "transcript_path": str(row["transcript_path"]),
                "transcript_word_count": int(row["transcript_word_count"]),
                "segment_count": int(row["segment_count"]),
                "target_words": int(row["target_words"]),
                "target_samples": int(row["target_samples"]),
                "window_count": int(row["window_count"]),
                "status": str(row["status"]),
                "selected_samples": int(row["selected_samples"]),
            }
            for row in episode_rows
        ],
    }


def _selection_export_fieldnames() -> list[str]:
    return [
        "session_id",
        "preset",
        "per_episode_target_words",
        "episode_order",
        "sample_id",
        "episode_title",
        "episode_key",
        "transcript_path",
        "window_index",
        "window_max_samples",
        "selection_order",
        "selection_type",
        "line_number",
        "speaker",
        "text",
        "word_count",
        "start_time",
        "end_time",
        "created_at_utc",
    ]


def _load_selection_export_rows(
    connection: sqlite3.Connection,
    *,
    session_id: str,
) -> list[dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT
            c.session_id,
            c.preset,
            c.per_episode_target_words,
            e.episode_order,
            e.sample_id,
            e.episode_title,
            e.episode_key,
            e.transcript_path,
            w.window_index,
            w.max_samples AS window_max_samples,
            s.selection_order,
            s.selection_type,
            s.line_number,
            s.speaker,
            s.text,
            s.word_count,
            s.start_time,
            s.end_time,
            s.created_at_utc
        FROM selections AS s
        INNER JOIN session_config AS c
            ON c.session_id = s.session_id
        INNER JOIN episodes AS e
            ON e.session_id = s.session_id
           AND e.episode_order = s.episode_order
        INNER JOIN windows AS w
            ON w.session_id = s.session_id
           AND w.episode_order = s.episode_order
           AND w.window_index = s.window_index
        WHERE s.session_id = ?
        ORDER BY
            s.episode_order ASC,
            s.window_index ASC,
            s.selection_order ASC
        """,
        (session_id,),
    ).fetchall()
    return [
        {
            "session_id": str(row["session_id"]),
            "preset": str(row["preset"]),
            "per_episode_target_words": int(row["per_episode_target_words"]),
            "episode_order": int(row["episode_order"]),
            "sample_id": str(row["sample_id"]),
            "episode_title": str(row["episode_title"]),
            "episode_key": str(row["episode_key"]),
            "transcript_path": str(row["transcript_path"]),
            "window_index": int(row["window_index"]),
            "window_max_samples": int(row["window_max_samples"]),
            "selection_order": int(row["selection_order"]),
            "selection_type": str(row["selection_type"]),
            "line_number": (
                int(row["line_number"]) if row["line_number"] is not None else None
            ),
            "speaker": str(row["speaker"]),
            "text": str(row["text"]),
            "word_count": int(row["word_count"]),
            "start_time": int(row["start_time"]) if row["start_time"] is not None else None,
            "end_time": int(row["end_time"]) if row["end_time"] is not None else None,
            "created_at_utc": str(row["created_at_utc"]),
        }
        for row in rows
    ]


def _render_session_created_banner(
    *,
    console: Console,
    session_id: str,
    session_paths: _SessionPaths,
    connection: sqlite3.Connection,
) -> None:
    session_row = _require_session_row(connection, session_id)
    summary = _build_session_summary_payload(connection, session_id=session_id)
    preset_seconds = int(session_row["preset_seconds"])
    formula_minutes = preset_seconds / 60.0
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Session ID:[/bold] {session_id}",
                    f"[bold]Manifest:[/bold] {session_row['manifest_path']}",
                    (
                        "[bold]Per-episode target:[/bold] "
                        f"{session_row['words_per_minute']:.1f} WPM x {formula_minutes:.2f} min x "
                        f"{session_row['sample_factor']:.2f} = "
                        f"{session_row['per_episode_target_words']} sampled words"
                    ),
                    (
                        f"[bold]Episode prefix:[/bold] 1..{summary['episodes_requested']} "
                        f"from the curated manifest"
                    ),
                    f"[bold]Database:[/bold] {session_paths.database_path}",
                ]
            ),
            title="SPoRC Human Sampler",
            border_style="green",
        )
    )


def _render_resume_banner(
    *,
    console: Console,
    session_id: str,
    session_paths: _SessionPaths,
) -> None:
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Resuming session:[/bold] {session_id}",
                    f"[bold]Database:[/bold] {session_paths.database_path}",
                    f"[bold]Summary:[/bold] {session_paths.summary_path}",
                ]
            ),
            title="Resume SPoRC Human Sampler",
            border_style="cyan",
        )
    )


def _render_paused_banner(
    *,
    console: Console,
    session_id: str,
    session_paths: _SessionPaths,
) -> None:
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Session paused:[/bold] {session_id}",
                    (
                        "[bold]Resume command:[/bold] "
                        "uv run python -m card_framework.benchmark.summarizer_critic.sporc "
                        f"human-sample --resume-session {session_id}"
                    ),
                    f"[bold]Summary:[/bold] {session_paths.summary_path}",
                    f"[bold]Selections CSV:[/bold] {session_paths.selected_samples_csv_path}",
                ]
            ),
            title="Sampling Paused",
            border_style="yellow",
        )
    )


def _render_completed_banner(
    *,
    console: Console,
    session_id: str,
    session_paths: _SessionPaths,
    connection: sqlite3.Connection,
) -> None:
    progress = _load_progress_snapshot(connection, session_id=session_id)
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"[bold]Session completed:[/bold] {session_id}",
                    (
                        f"[bold]Selections captured:[/bold] "
                        f"{progress['selected_samples']} sample(s)"
                    ),
                    f"[bold]Summary:[/bold] {session_paths.summary_path}",
                    f"[bold]Selections JSONL:[/bold] {session_paths.selected_samples_jsonl_path}",
                    f"[bold]Selections CSV:[/bold] {session_paths.selected_samples_csv_path}",
                ]
            ),
            title="Sampling Complete",
            border_style="green",
        )
    )


def _render_help_panel(console: Console) -> None:
    console.print(
        Panel.fit(
            "\n".join(
                [
                    "Enter comma-separated transcript line numbers, for example `12,13`.",
                    "Overrides use `/override <segment>`, for example "
                    "`/override [SPEAKER_04] Replacement text`.",
                    "If the override text contains commas, wrap that token in quotes.",
                    "Use `/skip` to leave the current window empty.",
                    "Press `Ctrl+C` at any time to pause and save the session.",
                ]
            ),
            title="Sampler Help",
            border_style="blue",
        )
    )


def _render_window_screen(
    *,
    console: Console,
    session_row: sqlite3.Row,
    episode_row: sqlite3.Row,
    window_row: sqlite3.Row,
    progress: dict[str, int],
    episode_progress: dict[str, int],
    window_lines: tuple[TranscriptLine, ...],
) -> None:
    preset_seconds = int(session_row["preset_seconds"])
    session_table = Table(title="Session Progress", show_header=False, box=None)
    session_table.add_row(
        "Episodes",
        (
            f"{progress['completed_episodes']}/{progress['total_episodes']} completed"
        ),
    )
    session_table.add_row(
        "Windows",
        f"{progress['completed_windows']}/{progress['total_windows']} completed",
    )
    session_table.add_row("Selections", str(progress["selected_samples"]))
    session_table.add_row(
        "Per-episode target",
        (
            f"{int(session_row['per_episode_target_words'])} words "
            f"({session_row['preset']} = {preset_seconds}s)"
        ),
    )

    episode_table = Table(title="Episode Plan", show_header=False, box=None)
    episode_table.add_row(
        "Episode",
        f"{int(episode_row['episode_order'])}: {episode_row['episode_title']}",
    )
    episode_table.add_row("Sample ID", str(episode_row["sample_id"]))
    episode_table.add_row("Transcript words", str(int(episode_row["transcript_word_count"])))
    episode_table.add_row("Transcript lines", str(int(episode_row["segment_count"])))
    episode_table.add_row("Target words", str(int(episode_row["target_words"])))
    episode_table.add_row("Estimated samples", str(int(episode_row["target_samples"])))
    episode_table.add_row(
        "Episode progress",
        (
            f"{episode_progress['completed_windows']}/{int(episode_row['window_count'])} windows, "
            f"{episode_progress['selected_samples']} saved sample(s)"
        ),
    )

    window_title = (
        f"Window {int(window_row['window_index'])}/{int(episode_row['window_count'])} "
        f"(lines {int(window_row['start_line_number'])}-{int(window_row['end_line_number'])})"
    )
    lines_table = Table(title=window_title, header_style="bold magenta")
    lines_table.add_column("#", style="cyan", no_wrap=True)
    lines_table.add_column("Speaker", style="green", no_wrap=True)
    lines_table.add_column("Words", justify="right", no_wrap=True)
    lines_table.add_column("Text", overflow="fold")
    for line in window_lines:
        lines_table.add_row(
            str(line.line_number),
            line.speaker,
            str(line.word_count),
            line.text,
        )

    instruction = Text()
    instruction.append(
        (
            f"Enter up to {int(window_row['max_samples'])} sample(s) from this window. "
            "Use line numbers or /override. Enter /help for syntax."
        ),
        style="bold",
    )
    console.print(Panel.fit(session_table, border_style="cyan"))
    console.print(Panel.fit(episode_table, border_style="magenta"))
    console.print(lines_table)
    console.print(instruction)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
