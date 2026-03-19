"""Analyze SPoRC duration-sweep artifacts for throughput bottlenecks."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from card_framework.shared.paths import REPO_ROOT
from card_framework.shared.perf import StatusReporter, report_loading_duration


_DEFAULT_SWEEP_ROOT = (
    REPO_ROOT / "artifacts" / "summarizer_critic_benchmark" / "sweeps"
)
_STAGE_ORDER = ("index", "summarizer", "critic", "clear", "sample_start", "unknown")
_STAGE_COLORS = {
    "index": "#4c78a8",
    "summarizer": "#f58518",
    "critic": "#54a24b",
    "clear": "#e45756",
    "sample_start": "#b279a2",
    "unknown": "#9d9da1",
}
_PRESET_COLORS = {
    "30s": "#4c78a8",
    "1m": "#f58518",
    "5m": "#54a24b",
    "15m": "#e45756",
    "unknown": "#9d9da1",
}
_STATUS_MARKERS = {
    "completed": "o",
    "failed": "x",
    "active": "^",
}
_ISSUE_PATTERNS = (
    (
        "checkpoint_permission_error",
        re.compile(r"PermissionError: .*WinError 32", re.IGNORECASE),
    ),
    (
        "connection_reset_error",
        re.compile(r"ConnectionResetError|WinError 10054", re.IGNORECASE),
    ),
    (
        "child_exit_nonzero",
        re.compile(r"\[Sweep\] Child exited with code (?!0\b)\d+", re.IGNORECASE),
    ),
    (
        "traceback",
        re.compile(r"^Traceback \(most recent call last\):", re.IGNORECASE),
    ),
)


@dataclass(slots=True)
class WorkerIssue:
    """One non-JSON issue detected in a worker log."""

    worker_label: str
    preset: str
    category: str
    message: str


@dataclass(slots=True)
class StageSegment:
    """One observed sample-stage interval from a worker log."""

    worker_label: str
    preset: str
    sample_id: str
    target_seconds: int | None
    stage: str
    start_elapsed_seconds: float
    end_elapsed_seconds: float
    duration_seconds: float
    start_utc: str | None
    end_utc: str | None
    iteration: int | None = None
    is_active: bool = False
    max_activity_age_seconds: float = 0.0


@dataclass(slots=True)
class ActiveWorkerState:
    """Current active-sample snapshot for one worker."""

    worker_label: str
    preset: str
    sample_id: str
    target_seconds: int | None
    stage: str
    iteration: int | None
    elapsed_sample_seconds: float
    last_activity_age_seconds: float | None
    started_at_utc: str | None


@dataclass(slots=True)
class SampleRecord:
    """Merged per-sample analysis row."""

    worker_label: str
    preset: str
    sample_id: str
    target_seconds: int | None
    status: str
    final_status: str
    observed_duration_seconds: float
    completed_duration_seconds: float | None
    candidate_seconds: float | None
    converged: bool | None
    iterations_run: int | None
    retrieval_events: int | None
    tool_invocations: int | None
    counts_toward_target: bool | None
    replacement_reason: str | None
    error_message: str | None
    active_stage: str | None
    active_iteration: int | None
    last_activity_age_seconds: float | None
    transcript_path: str | None
    transcript_words: int | None
    transcript_segments: int | None
    transcript_speakers: int | None
    index_duration_seconds: float = 0.0
    summarizer_duration_seconds: float = 0.0
    critic_duration_seconds: float = 0.0
    clear_duration_seconds: float = 0.0
    sample_start_duration_seconds: float = 0.0
    unknown_duration_seconds: float = 0.0
    composite_score: float | None = None


@dataclass(slots=True)
class AnalysisResult:
    """Manifest of generated analysis artifacts."""

    sweep_root: str
    output_dir: str
    generated_at_utc: str
    graphs: list[str]
    csv_exports: list[str]
    summary_markdown_path: str
    summary_json_path: str


@dataclass(slots=True)
class _SampleProgress:
    """Mutable state while parsing one worker log."""

    worker_label: str
    preset: str
    sample_id: str
    target_seconds: int | None
    sample_index: int | None
    sample_count: int | None
    started_at_utc: str | None = None
    current_stage: str | None = None
    current_iteration: int | None = None
    current_stage_start_elapsed_seconds: float = 0.0
    last_elapsed_seconds: float = 0.0
    last_activity_age_seconds: float | None = None
    current_stage_max_activity_age_seconds: float = 0.0


def _parse_iso8601(raw_value: str | None) -> datetime | None:
    if not raw_value:
        return None
    try:
        return datetime.fromisoformat(str(raw_value))
    except ValueError:
        return None


def _isoformat_or_none(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _find_latest_sweep_root(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No sweep directories found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunk_values(values: list[str], chunk_size: int = 500) -> list[list[str]]:
    return [
        values[index : index + chunk_size]
        for index in range(0, len(values), chunk_size)
    ]


def _stage_sort_key(stage: str) -> tuple[int, str]:
    try:
        return (_STAGE_ORDER.index(stage), stage)
    except ValueError:
        return (len(_STAGE_ORDER), stage)


def _close_stage_segment(
    *,
    progress: _SampleProgress,
    stage_segments: list[StageSegment],
    end_elapsed_seconds: float,
    is_active: bool,
) -> None:
    stage = str(progress.current_stage or "unknown")
    start_elapsed_seconds = max(0.0, float(progress.current_stage_start_elapsed_seconds))
    end_elapsed_seconds = max(start_elapsed_seconds, float(end_elapsed_seconds))
    duration_seconds = max(0.0, end_elapsed_seconds - start_elapsed_seconds)
    started_at = _parse_iso8601(progress.started_at_utc)
    if started_at is None:
        start_utc = None
        end_utc = None
    else:
        start_utc = _isoformat_or_none(started_at + timedelta(seconds=start_elapsed_seconds))
        end_utc = _isoformat_or_none(started_at + timedelta(seconds=end_elapsed_seconds))
    stage_segments.append(
        StageSegment(
            worker_label=progress.worker_label,
            preset=progress.preset,
            sample_id=progress.sample_id,
            target_seconds=progress.target_seconds,
            stage=stage,
            start_elapsed_seconds=round(start_elapsed_seconds, 3),
            end_elapsed_seconds=round(end_elapsed_seconds, 3),
            duration_seconds=round(duration_seconds, 3),
            start_utc=start_utc,
            end_utc=end_utc,
            iteration=progress.current_iteration,
            is_active=is_active,
            max_activity_age_seconds=round(
                max(
                    progress.current_stage_max_activity_age_seconds,
                    _safe_float(progress.last_activity_age_seconds, 0.0),
                ),
                3,
            ),
        )
    )


def _parse_worker_log(
    log_path: Path,
) -> tuple[list[StageSegment], ActiveWorkerState | None, list[WorkerIssue]]:
    worker_label = log_path.stem
    preset = (
        worker_label.split(".shard-", maxsplit=1)[0]
        if ".shard-" in worker_label
        else "unknown"
    )
    issues: list[WorkerIssue] = []
    seen_issues: set[tuple[str, str]] = set()
    stage_segments: list[StageSegment] = []
    active_progress: _SampleProgress | None = None
    active_state: ActiveWorkerState | None = None

    with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.rstrip()
            for category, pattern in _ISSUE_PATTERNS:
                if not pattern.search(line):
                    continue
                issue_key = (category, line)
                if issue_key in seen_issues:
                    continue
                issues.append(
                    WorkerIssue(
                        worker_label=worker_label,
                        preset=preset,
                        category=category,
                        message=line,
                    )
                )
                seen_issues.add(issue_key)

            line = raw_line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_name = payload.get("sporc_progress_event")
            if event_name == "sample_start":
                if active_progress is not None and active_progress.sample_id != str(payload.get("sample_id") or ""):
                    _close_stage_segment(
                        progress=active_progress,
                        stage_segments=stage_segments,
                        end_elapsed_seconds=max(active_progress.last_elapsed_seconds, 0.0),
                        is_active=True,
                    )
                active_progress = _SampleProgress(
                    worker_label=worker_label,
                    preset=preset,
                    sample_id=str(payload.get("sample_id") or ""),
                    target_seconds=_safe_int(payload.get("target_seconds")),
                    sample_index=_safe_int(payload.get("sample_index")),
                    sample_count=_safe_int(payload.get("sample_count")),
                )
                continue
            if event_name == "sample_heartbeat":
                sample_id = str(payload.get("sample_id") or "")
                if active_progress is None or active_progress.sample_id != sample_id:
                    active_progress = _SampleProgress(
                        worker_label=worker_label,
                        preset=preset,
                        sample_id=sample_id,
                        target_seconds=_safe_int(payload.get("target_seconds")),
                        sample_index=_safe_int(payload.get("sample_index")),
                        sample_count=_safe_int(payload.get("sample_count")),
                    )
                started_at_utc = str(payload.get("started_at_utc") or "").strip()
                if started_at_utc:
                    active_progress.started_at_utc = started_at_utc
                stage = str(payload.get("stage") or "unknown")
                elapsed_sample_seconds = max(0.0, _safe_float(payload.get("elapsed_sample_seconds")))
                iteration = _safe_int(payload.get("iteration"))
                last_activity_age_seconds = _safe_float(payload.get("last_activity_age_seconds"), 0.0)
                if active_progress.current_stage is None:
                    active_progress.current_stage = stage
                    active_progress.current_iteration = iteration
                    active_progress.current_stage_start_elapsed_seconds = 0.0
                    active_progress.current_stage_max_activity_age_seconds = last_activity_age_seconds
                elif stage != active_progress.current_stage:
                    _close_stage_segment(
                        progress=active_progress,
                        stage_segments=stage_segments,
                        end_elapsed_seconds=elapsed_sample_seconds,
                        is_active=False,
                    )
                    active_progress.current_stage = stage
                    active_progress.current_iteration = iteration
                    active_progress.current_stage_start_elapsed_seconds = elapsed_sample_seconds
                    active_progress.current_stage_max_activity_age_seconds = last_activity_age_seconds
                else:
                    active_progress.current_stage_max_activity_age_seconds = max(
                        active_progress.current_stage_max_activity_age_seconds,
                        last_activity_age_seconds,
                    )
                active_progress.last_elapsed_seconds = max(
                    active_progress.last_elapsed_seconds,
                    elapsed_sample_seconds,
                )
                active_progress.last_activity_age_seconds = last_activity_age_seconds
                active_state = ActiveWorkerState(
                    worker_label=worker_label,
                    preset=preset,
                    sample_id=sample_id,
                    target_seconds=_safe_int(payload.get("target_seconds")),
                    stage=stage,
                    iteration=iteration,
                    elapsed_sample_seconds=round(elapsed_sample_seconds, 3),
                    last_activity_age_seconds=round(last_activity_age_seconds, 3),
                    started_at_utc=active_progress.started_at_utc,
                )
                continue
            if event_name == "sample_done":
                sample_id = str(payload.get("sample_id") or "")
                if active_progress is None or active_progress.sample_id != sample_id:
                    continue
                end_elapsed_seconds = max(
                    _safe_float(payload.get("duration_seconds")),
                    active_progress.last_elapsed_seconds,
                )
                _close_stage_segment(
                    progress=active_progress,
                    stage_segments=stage_segments,
                    end_elapsed_seconds=end_elapsed_seconds,
                    is_active=False,
                )
                active_progress = None
                active_state = None

    if active_progress is not None:
        _close_stage_segment(
            progress=active_progress,
            stage_segments=stage_segments,
            end_elapsed_seconds=max(active_progress.last_elapsed_seconds, 0.0),
            is_active=True,
        )
        if active_state is None:
            active_state = ActiveWorkerState(
                worker_label=active_progress.worker_label,
                preset=active_progress.preset,
                sample_id=active_progress.sample_id,
                target_seconds=active_progress.target_seconds,
                stage=str(active_progress.current_stage or "unknown"),
                iteration=active_progress.current_iteration,
                elapsed_sample_seconds=round(active_progress.last_elapsed_seconds, 3),
                last_activity_age_seconds=(
                    round(active_progress.last_activity_age_seconds, 3)
                    if active_progress.last_activity_age_seconds is not None
                    else None
                ),
                started_at_utc=active_progress.started_at_utc,
            )
    return stage_segments, active_state, issues


def _load_checkpoint_rows(
    checkpoints_root: Path,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    completed_results: dict[tuple[str, str], dict[str, Any]] = {}
    active_samples: dict[str, dict[str, Any]] = {}
    if not checkpoints_root.exists():
        return completed_results, active_samples
    for checkpoint_path in sorted(checkpoints_root.glob("*.checkpoint.json")):
        worker_label = checkpoint_path.name.removesuffix(".checkpoint.json")
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if isinstance(payload.get("completed_results"), list):
            for item in payload["completed_results"]:
                if not isinstance(item, dict):
                    continue
                sample_id = str(item.get("sample_id") or "").strip()
                if sample_id:
                    completed_results[(worker_label, sample_id)] = dict(item)
        if isinstance(payload.get("active_sample"), dict) and payload["active_sample"].get("sample_id"):
            active_samples[worker_label] = dict(payload["active_sample"])
    return completed_results, active_samples


def _load_sample_payloads(
    queues_root: Path,
    *,
    sample_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    sample_payloads: dict[str, dict[str, Any]] = {}
    if not queues_root.exists():
        return sample_payloads
    requested_ids = sorted(sample_id for sample_id in (sample_ids or set()) if sample_id)
    for queue_db in sorted(queues_root.rglob("sample_queue.sqlite3")):
        connection = sqlite3.connect(str(queue_db))
        try:
            if requested_ids:
                rows: list[tuple[str, str]] = []
                for sample_id_chunk in _chunk_values(requested_ids):
                    placeholders = ",".join("?" for _ in sample_id_chunk)
                    rows.extend(
                        connection.execute(
                            f"""
                            SELECT sample_id, sample_payload_json
                            FROM sample_queue
                            WHERE sample_payload_json IS NOT NULL
                              AND sample_id IN ({placeholders})
                            """,
                            sample_id_chunk,
                        ).fetchall()
                    )
            else:
                rows = connection.execute(
                    """
                    SELECT sample_id, sample_payload_json
                    FROM sample_queue
                    WHERE sample_payload_json IS NOT NULL
                    """
                ).fetchall()
        finally:
            connection.close()
        for sample_id, raw_payload in rows:
            if str(sample_id) in sample_payloads:
                continue
            try:
                payload = json.loads(str(raw_payload))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                sample_payloads[str(sample_id)] = payload
    return sample_payloads


def _compute_transcript_complexity(
    sample_payloads: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    complexity_by_sample: dict[str, dict[str, Any]] = {}
    for sample_id, payload in sample_payloads.items():
        transcript_path = str(payload.get("transcript_path") or "").strip()
        if not transcript_path:
            complexity_by_sample[sample_id] = {
                "transcript_path": None,
                "transcript_words": None,
                "transcript_segments": None,
                "transcript_speakers": None,
            }
            continue
        transcript_file = Path(transcript_path)
        if not transcript_file.exists():
            complexity_by_sample[sample_id] = {
                "transcript_path": str(transcript_file),
                "transcript_words": None,
                "transcript_segments": None,
                "transcript_speakers": None,
            }
            continue
        payload_json = json.loads(transcript_file.read_text(encoding="utf-8"))
        raw_segments = payload_json.get("segments")
        if not isinstance(raw_segments, list):
            raw_segments = []
        total_words = 0
        speakers: set[str] = set()
        for raw_segment in raw_segments:
            if not isinstance(raw_segment, dict):
                continue
            speakers.add(str(raw_segment.get("speaker", "UNKNOWN")))
            total_words += len(
                re.sub(r"<[^>]+>", " ", str(raw_segment.get("text", ""))).split()
            )
        complexity_by_sample[sample_id] = {
            "transcript_path": str(transcript_file),
            "transcript_words": len(raw_segments) and int(total_words) or 0,
            "transcript_segments": int(len(raw_segments)),
            "transcript_speakers": int(len(speakers)),
        }
    return complexity_by_sample


def _merge_sample_records(
    *,
    stage_segments: list[StageSegment],
    active_states: dict[str, ActiveWorkerState],
    checkpoint_results: dict[tuple[str, str], dict[str, Any]],
    checkpoint_active_samples: dict[str, dict[str, Any]],
    sample_payloads: dict[str, dict[str, Any]],
    complexity_by_sample: dict[str, dict[str, Any]],
    worker_issues: list[WorkerIssue],
) -> list[SampleRecord]:
    stage_durations_by_key: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for segment in stage_segments:
        stage_durations_by_key[(segment.worker_label, segment.sample_id)][
            segment.stage
        ] += max(0.0, segment.duration_seconds)

    issue_categories_by_worker: dict[str, list[str]] = defaultdict(list)
    for issue in worker_issues:
        issue_categories_by_worker[issue.worker_label].append(issue.category)

    sample_keys: set[tuple[str, str]] = set(checkpoint_results)
    sample_keys.update(stage_durations_by_key)
    sample_keys.update(
        (worker_label, str(payload.get("sample_id") or ""))
        for worker_label, payload in checkpoint_active_samples.items()
        if str(payload.get("sample_id") or "").strip()
    )
    sample_keys.update(
        (worker_label, state.sample_id) for worker_label, state in active_states.items()
    )

    sample_records: list[SampleRecord] = []
    for worker_label, sample_id in sorted(sample_keys):
        checkpoint_row = checkpoint_results.get((worker_label, sample_id), {})
        active_state = active_states.get(worker_label)
        if active_state is not None and active_state.sample_id != sample_id:
            active_state = None
        checkpoint_active = checkpoint_active_samples.get(worker_label)
        if checkpoint_active is not None and str(
            checkpoint_active.get("sample_id") or ""
        ) != sample_id:
            checkpoint_active = None
        preset = (
            worker_label.split(".shard-", maxsplit=1)[0]
            if ".shard-" in worker_label
            else "unknown"
        )
        sample_payload = sample_payloads.get(sample_id, {})
        complexity = complexity_by_sample.get(sample_id, {})
        stage_durations = stage_durations_by_key.get((worker_label, sample_id), {})
        checkpoint_status = str(checkpoint_row.get("status") or "").strip()
        status = checkpoint_status or (
            "active" if active_state or checkpoint_active else "unknown"
        )
        final_status = str(checkpoint_row.get("final_status") or "").strip()
        if not final_status and status == "active":
            final_status = "active"
        active_stage = None
        active_iteration = None
        last_activity_age_seconds = None
        observed_duration_seconds = _safe_float(checkpoint_row.get("duration_seconds"))
        if active_state is not None:
            active_stage = active_state.stage
            active_iteration = active_state.iteration
            last_activity_age_seconds = active_state.last_activity_age_seconds
            observed_duration_seconds = max(
                observed_duration_seconds,
                active_state.elapsed_sample_seconds,
            )
        elif checkpoint_active is not None:
            active_stage = str(checkpoint_active.get("stage") or "").strip() or None
            active_iteration = _safe_int(checkpoint_active.get("iteration"))
            last_activity_age_seconds = _safe_float(
                checkpoint_active.get("last_activity_age_seconds"),
                0.0,
            )
            observed_duration_seconds = max(
                observed_duration_seconds,
                _safe_float(checkpoint_active.get("elapsed_sample_seconds")),
            )
        if status == "unknown" and issue_categories_by_worker.get(worker_label):
            status = "failed"
            if not final_status:
                final_status = issue_categories_by_worker[worker_label][0]
        metrics = checkpoint_row.get("metrics")
        composite_score = None
        if isinstance(metrics, dict) and metrics.get("composite") is not None:
            composite_score = round(_safe_float(metrics.get("composite")), 4)
        sample_records.append(
            SampleRecord(
                worker_label=worker_label,
                preset=preset,
                sample_id=sample_id,
                target_seconds=_safe_int(
                    checkpoint_row.get("target_seconds")
                    if checkpoint_row
                    else sample_payload.get("target_seconds")
                ),
                status=status,
                final_status=final_status,
                observed_duration_seconds=round(observed_duration_seconds, 3),
                completed_duration_seconds=(
                    round(_safe_float(checkpoint_row.get("duration_seconds")), 3)
                    if checkpoint_row
                    else None
                ),
                candidate_seconds=(
                    round(_safe_float(checkpoint_row.get("candidate_seconds")), 3)
                    if checkpoint_row
                    else None
                ),
                converged=bool(checkpoint_row.get("converged"))
                if checkpoint_row
                else None,
                iterations_run=_safe_int(checkpoint_row.get("iterations_run")),
                retrieval_events=_safe_int(checkpoint_row.get("retrieval_events")),
                tool_invocations=_safe_int(checkpoint_row.get("tool_invocations")),
                counts_toward_target=(
                    bool(checkpoint_row.get("counts_toward_target"))
                    if checkpoint_row
                    else None
                ),
                replacement_reason=(
                    str(checkpoint_row.get("replacement_reason"))
                    if checkpoint_row.get("replacement_reason") is not None
                    else None
                ),
                error_message=(
                    str(checkpoint_row.get("error_message"))
                    if checkpoint_row.get("error_message") is not None
                    else None
                ),
                active_stage=active_stage,
                active_iteration=active_iteration,
                last_activity_age_seconds=(
                    round(last_activity_age_seconds, 3)
                    if last_activity_age_seconds is not None
                    else None
                ),
                transcript_path=(
                    str(complexity.get("transcript_path"))
                    if complexity.get("transcript_path") is not None
                    else None
                ),
                transcript_words=_safe_int(complexity.get("transcript_words")),
                transcript_segments=_safe_int(complexity.get("transcript_segments")),
                transcript_speakers=_safe_int(complexity.get("transcript_speakers")),
                index_duration_seconds=round(stage_durations.get("index", 0.0), 3),
                summarizer_duration_seconds=round(
                    stage_durations.get("summarizer", 0.0),
                    3,
                ),
                critic_duration_seconds=round(stage_durations.get("critic", 0.0), 3),
                clear_duration_seconds=round(stage_durations.get("clear", 0.0), 3),
                sample_start_duration_seconds=round(
                    stage_durations.get("sample_start", 0.0),
                    3,
                ),
                unknown_duration_seconds=round(
                    stage_durations.get("unknown", 0.0),
                    3,
                ),
                composite_score=composite_score,
            )
        )
    return sample_records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stage_duration_fields(sample: SampleRecord) -> list[tuple[str, float]]:
    return [
        ("index", sample.index_duration_seconds),
        ("summarizer", sample.summarizer_duration_seconds),
        ("critic", sample.critic_duration_seconds),
        ("clear", sample.clear_duration_seconds),
        ("sample_start", sample.sample_start_duration_seconds),
        ("unknown", sample.unknown_duration_seconds),
    ]


def _build_concurrency_series(
    stage_segments: list[StageSegment],
) -> tuple[datetime | None, dict[str, list[tuple[float, int]]]]:
    parsed_segments_by_stage: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    baseline: datetime | None = None
    for segment in stage_segments:
        if segment.start_utc is None or segment.end_utc is None:
            continue
        start_dt = _parse_iso8601(segment.start_utc)
        end_dt = _parse_iso8601(segment.end_utc)
        if start_dt is None or end_dt is None:
            continue
        parsed_segments_by_stage[segment.stage].append((start_dt, end_dt))
        baseline = start_dt if baseline is None else min(baseline, start_dt)
    if baseline is None:
        return None, {}
    series: dict[str, list[tuple[float, int]]] = {}
    for stage in sorted(parsed_segments_by_stage, key=_stage_sort_key):
        events: list[tuple[float, int]] = []
        for start_dt, end_dt in parsed_segments_by_stage[stage]:
            start_seconds = (start_dt - baseline).total_seconds()
            end_seconds = max(start_seconds, (end_dt - baseline).total_seconds())
            events.append((start_seconds, 1))
            events.append((end_seconds, -1))
        events.sort(key=lambda item: (item[0], -item[1]))
        points: list[tuple[float, int]] = [(0.0, 0)]
        current = 0
        for offset_seconds, delta in events:
            points.append((offset_seconds, current))
            current += delta
            points.append((offset_seconds, current))
        series[stage] = points
    return baseline, series


def _write_empty_figure(
    *,
    output_path: Path,
    title: str,
    message: str,
    figsize: tuple[float, float] = (10, 4),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title)
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=12,
        wrap=True,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_worker_gantt(
    *,
    stage_segments: list[StageSegment],
    output_path: Path,
) -> None:
    parsed_segments: list[tuple[StageSegment, datetime, datetime]] = []
    for segment in stage_segments:
        if segment.start_utc is None or segment.end_utc is None:
            continue
        start_dt = _parse_iso8601(segment.start_utc)
        end_dt = _parse_iso8601(segment.end_utc)
        if start_dt is None or end_dt is None:
            continue
        parsed_segments.append((segment, start_dt, end_dt))
    if not parsed_segments:
        _write_empty_figure(
            output_path=output_path,
            title="Worker Stage Timeline",
            message="No stage timing data was found in the sweep logs.",
        )
        return
    baseline = min(start_dt for _, start_dt, _ in parsed_segments)
    worker_labels = sorted({segment.worker_label for segment, _, _ in parsed_segments})
    y_positions = {worker_label: index for index, worker_label in enumerate(worker_labels)}

    fig_height = max(4.5, 0.6 * len(worker_labels) + 2.0)
    fig, ax = plt.subplots(figsize=(16, fig_height))
    for segment, start_dt, end_dt in parsed_segments:
        start_minutes = (start_dt - baseline).total_seconds() / 60.0
        duration_minutes = max(0.01, (end_dt - start_dt).total_seconds() / 60.0)
        ax.barh(
            y_positions[segment.worker_label],
            duration_minutes,
            left=start_minutes,
            color=_STAGE_COLORS.get(segment.stage, _STAGE_COLORS["unknown"]),
            alpha=0.65 if segment.is_active else 0.9,
            edgecolor="black" if segment.is_active else "none",
            linewidth=0.8 if segment.is_active else 0.0,
            hatch="//" if segment.is_active else None,
            height=0.7,
        )
    ax.set_yticks(list(y_positions.values()), worker_labels)
    ax.set_xlabel("Minutes Since First Observed Stage Start")
    ax.set_ylabel("Worker")
    ax.set_title("Worker Stage Timeline")
    legend_handles = [
        Patch(facecolor=_STAGE_COLORS[stage], label=stage)
        for stage in _STAGE_ORDER
        if any(segment.stage == stage for segment, _, _ in parsed_segments)
    ]
    if any(segment.is_active for segment, _, _ in parsed_segments):
        legend_handles.append(
            Patch(facecolor="white", edgecolor="black", hatch="//", label="active")
        )
    ax.legend(handles=legend_handles, loc="upper right")
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_preset_stage_breakdown(
    *,
    sample_records: list[SampleRecord],
    output_path: Path,
) -> None:
    presets = sorted(
        {record.preset for record in sample_records},
        key=lambda value: (value not in _PRESET_COLORS, value),
    )
    if not presets:
        _write_empty_figure(
            output_path=output_path,
            title="Observed Worker Time By Stage And Preset",
            message="No sample records were available for preset breakdown analysis.",
        )
        return
    stage_totals: dict[str, dict[str, float]] = {
        preset: defaultdict(float) for preset in presets
    }
    for record in sample_records:
        for stage, duration_seconds in _stage_duration_fields(record):
            stage_totals[record.preset][stage] += max(0.0, duration_seconds)

    fig, ax = plt.subplots(figsize=(12, 6))
    bottoms = [0.0 for _ in presets]
    stages = sorted(
        {stage for totals in stage_totals.values() for stage in totals},
        key=_stage_sort_key,
    )
    for stage in stages:
        values = [stage_totals[preset].get(stage, 0.0) / 3600.0 for preset in presets]
        ax.bar(
            presets,
            values,
            bottom=bottoms,
            color=_STAGE_COLORS.get(stage, _STAGE_COLORS["unknown"]),
            label=stage,
        )
        bottoms = [bottom + value for bottom, value in zip(bottoms, values, strict=True)]
    for index, total_hours in enumerate(bottoms):
        ax.text(index, total_hours + 0.02, f"{total_hours:.2f}h", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Observed Worker-Hours")
    ax.set_title("Observed Worker Time By Stage And Preset")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_stage_concurrency(
    *,
    stage_segments: list[StageSegment],
    output_path: Path,
) -> None:
    _, series = _build_concurrency_series(stage_segments)
    if not series:
        _write_empty_figure(
            output_path=output_path,
            title="Stage Concurrency Over Time",
            message="No overlapping stage timing data was found in the sweep logs.",
        )
        return
    fig, ax = plt.subplots(figsize=(14, 6))
    for stage in sorted(series, key=_stage_sort_key):
        points = series[stage]
        x_minutes = [offset_seconds / 60.0 for offset_seconds, _ in points]
        y_values = [count for _, count in points]
        ax.step(
            x_minutes,
            y_values,
            where="post",
            label=stage,
            color=_STAGE_COLORS.get(stage, _STAGE_COLORS["unknown"]),
            linewidth=2.0,
        )
    ax.set_xlabel("Minutes Since First Observed Stage Start")
    ax.set_ylabel("Workers In Stage")
    ax.set_title("Stage Concurrency Over Time")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _build_failure_rows(
    *,
    sample_records: list[SampleRecord],
    worker_issues: list[WorkerIssue],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for record in sample_records:
        if record.status == "completed" and record.final_status == "pass":
            continue
        if record.status == "active":
            continue
        failures.append(
            {
                "category": record.final_status or record.status,
                "count": 1,
                "wasted_seconds": max(0.0, record.observed_duration_seconds),
            }
        )
    for issue in worker_issues:
        failures.append(
            {"category": issue.category, "count": 1, "wasted_seconds": 0.0}
        )
    aggregate: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"category": "", "count": 0, "wasted_seconds": 0.0}
    )
    for row in failures:
        category = str(row["category"])
        aggregate[category]["category"] = category
        aggregate[category]["count"] += int(row["count"])
        aggregate[category]["wasted_seconds"] += float(row["wasted_seconds"])
    return sorted(
        aggregate.values(),
        key=lambda row: (
            -float(row["wasted_seconds"]),
            -int(row["count"]),
            str(row["category"]),
        ),
    )


def _plot_failure_waste(
    *,
    failure_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    if not failure_rows:
        _write_empty_figure(
            output_path=output_path,
            title="Failure And Waste Summary",
            message="No failed samples or worker-log issues were detected.",
        )
        return
    categories = [str(row["category"]) for row in failure_rows]
    wasted_minutes = [float(row["wasted_seconds"]) / 60.0 for row in failure_rows]
    counts = [int(row["count"]) for row in failure_rows]

    fig, (ax_time, ax_count) = plt.subplots(
        2,
        1,
        figsize=(12, max(6, 0.45 * len(categories) + 3.0)),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    ax_time.barh(categories, wasted_minutes, color="#e45756")
    ax_time.set_title("Failure And Waste Summary")
    ax_time.set_xlabel("Wasted Worker-Minutes")
    ax_time.invert_yaxis()
    for idx, value in enumerate(wasted_minutes):
        ax_time.text(value + 0.05, idx, f"{value:.1f}", va="center", fontsize=9)

    ax_count.barh(categories, counts, color="#4c78a8")
    ax_count.set_xlabel("Count")
    ax_count.invert_yaxis()
    for idx, value in enumerate(counts):
        ax_count.text(value + 0.05, idx, str(value), va="center", fontsize=9)

    for axis in (ax_time, ax_count):
        axis.grid(axis="x", alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_complexity_scatter(
    *,
    sample_records: list[SampleRecord],
    output_path: Path,
) -> None:
    plotted = [
        record
        for record in sample_records
        if record.transcript_words is not None and record.observed_duration_seconds > 0
    ]
    if not plotted:
        _write_empty_figure(
            output_path=output_path,
            title="Transcript Complexity Vs Observed Wall Time",
            message="No completed sample rows had both transcript complexity and wall-time data.",
        )
        return
    fig, ax = plt.subplots(figsize=(12, 7))
    for preset in sorted(
        {record.preset for record in plotted},
        key=lambda value: (value not in _PRESET_COLORS, value),
    ):
        for status in ("completed", "failed", "active"):
            rows = [
                record
                for record in plotted
                if record.preset == preset and record.status == status
            ]
            if not rows:
                continue
            ax.scatter(
                [int(record.transcript_words or 0) for record in rows],
                [record.observed_duration_seconds / 60.0 for record in rows],
                color=_PRESET_COLORS.get(preset, _PRESET_COLORS["unknown"]),
                marker=_STATUS_MARKERS[status],
                alpha=0.8,
                s=70 if status != "active" else 85,
                linewidths=1.5 if status == "failed" else 0.8,
                edgecolors="black" if status == "active" else None,
            )
    slowest_rows = sorted(
        plotted,
        key=lambda record: record.observed_duration_seconds,
        reverse=True,
    )[:5]
    for record in slowest_rows:
        ax.annotate(
            record.preset,
            (int(record.transcript_words or 0), record.observed_duration_seconds / 60.0),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    legend_handles = [
        Patch(facecolor=_PRESET_COLORS.get(preset, _PRESET_COLORS["unknown"]), label=preset)
        for preset in sorted(
            {record.preset for record in plotted},
            key=lambda value: (value not in _PRESET_COLORS, value),
        )
    ]
    status_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker=_STATUS_MARKERS[status],
            linestyle="None",
            markersize=8,
            label=status,
        )
        for status in ("completed", "failed", "active")
        if any(record.status == status for record in plotted)
    ]
    ax.legend(handles=legend_handles + status_handles, loc="upper left", ncol=2)
    ax.set_xlabel("Transcript Words")
    ax.set_ylabel("Observed Sample Wall Time (minutes)")
    ax.set_title("Transcript Complexity Vs Observed Wall Time")
    ax.grid(alpha=0.25, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _stage_share_summary(sample_records: list[SampleRecord]) -> list[tuple[str, float]]:
    totals = defaultdict(float)
    for record in sample_records:
        for stage, duration_seconds in _stage_duration_fields(record):
            totals[stage] += max(0.0, duration_seconds)
    grand_total = sum(totals.values())
    if grand_total <= 0:
        return []
    return sorted(
        (
            (stage, duration_seconds / grand_total)
            for stage, duration_seconds in totals.items()
            if duration_seconds > 0
        ),
        key=lambda item: (-item[1], _stage_sort_key(item[0])),
    )


def _build_summary_document(
    *,
    sweep_root: Path,
    output_dir: Path,
    sample_records: list[SampleRecord],
    active_states: dict[str, ActiveWorkerState],
    worker_issues: list[WorkerIssue],
    failure_rows: list[dict[str, Any]],
    analysis_result: AnalysisResult,
) -> str:
    completed_count = 0
    pass_count = 0
    active_count = 0
    preset_buckets: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in sample_records:
        preset_buckets[record.preset].append(record)
        if record.status == "completed":
            completed_count += 1
            if record.final_status == "pass":
                pass_count += 1
        elif record.status == "active":
            active_count += 1

    preset_rows = []
    for preset in sorted(
        preset_buckets,
        key=lambda value: (value not in _PRESET_COLORS, value),
    ):
        rows = preset_buckets[preset]
        preset_rows.append(
            {
                "preset": preset,
                "completed": sum(
                    1 for record in rows if record.status == "completed"
                ),
                "pass": sum(
                    1
                    for record in rows
                    if record.status == "completed" and record.final_status == "pass"
                ),
                "active": sum(1 for record in rows if record.status == "active"),
                "observed_worker_hours": round(
                    sum(record.observed_duration_seconds for record in rows) / 3600.0,
                    3,
                ),
                "index_hours": round(
                    sum(record.index_duration_seconds for record in rows) / 3600.0,
                    3,
                ),
                "summarizer_hours": round(
                    sum(record.summarizer_duration_seconds for record in rows) / 3600.0,
                    3,
                ),
                "critic_hours": round(
                    sum(record.critic_duration_seconds for record in rows) / 3600.0,
                    3,
                ),
            }
        )

    active_stage_counts = Counter(
        state.stage for state in active_states.values() if state.stage
    )
    share_summary = _stage_share_summary(sample_records)
    longest_rows = sorted(
        sample_records,
        key=lambda record: record.observed_duration_seconds,
        reverse=True,
    )[:10]

    lines = [
        "# SPoRC Sweep Bottleneck Analysis",
        "",
        f"- Generated at: `{analysis_result.generated_at_utc}`",
        f"- Sweep root: `{sweep_root}`",
        f"- Output dir: `{output_dir}`",
        f"- Completed samples observed: `{completed_count}`",
        f"- Passing completed samples: `{pass_count}`",
        f"- Active samples observed: `{active_count}`",
        f"- Worker issues detected: `{len(worker_issues)}`",
        "",
        "## Outputs",
        "",
        "- Graphs:",
    ]
    lines.extend(f"  - `{Path(path).name}`" for path in analysis_result.graphs)
    lines.append("- CSV exports:")
    lines.extend(f"  - `{Path(path).name}`" for path in analysis_result.csv_exports)
    lines.extend(
        [
            "",
            "## Preset Summary",
            "",
            "| preset | completed | pass | active | observed_worker_hours | index_hours | summarizer_hours | critic_hours |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        (
            f"| `{row['preset']}` | `{row['completed']}` | `{row['pass']}` | "
            f"`{row['active']}` | `{row['observed_worker_hours']:.3f}` | "
            f"`{row['index_hours']:.3f}` | `{row['summarizer_hours']:.3f}` | "
            f"`{row['critic_hours']:.3f}` |"
        )
        for row in preset_rows
    )
    lines.extend(
        [
            "",
            "## Current Active Workers",
            "",
            "| worker | preset | sample_id | stage | elapsed_seconds | last_activity_age_seconds |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    active_rows = sorted(
        active_states.values(),
        key=lambda item: (item.preset, item.worker_label),
    )
    if active_rows:
        lines.extend(
            (
                f"| `{row.worker_label}` | `{row.preset}` | `{row.sample_id}` | "
                f"`{row.stage}` | `{row.elapsed_sample_seconds:.3f}` | "
                f"`{row.last_activity_age_seconds if row.last_activity_age_seconds is not None else 'n/a'}` |"
            )
            for row in active_rows
        )
    else:
        lines.append("| `-` | `-` | `-` | `-` | `0` | `0` |")

    lines.extend(["", "## Stage Share", ""])
    if share_summary:
        lines.extend(
            f"- `{stage}`: `{share * 100:.1f}%` of observed worker time"
            for stage, share in share_summary[:5]
        )
    else:
        lines.append("- No observed stage time yet.")

    lines.extend(["", "## Active Stage Counts", ""])
    if active_stage_counts:
        lines.extend(
            f"- `{stage}`: `{count}` worker(s)"
            for stage, count in sorted(
                active_stage_counts.items(),
                key=lambda item: _stage_sort_key(item[0]),
            )
        )
    else:
        lines.append("- No active workers.")

    lines.extend(
        [
            "",
            "## Failure And Waste",
            "",
            "| category | count | wasted_seconds |",
            "| --- | ---: | ---: |",
        ]
    )
    if failure_rows:
        lines.extend(
            f"| `{row['category']}` | `{int(row['count'])}` | `{float(row['wasted_seconds']):.3f}` |"
            for row in failure_rows
        )
    else:
        lines.append("| `none` | `0` | `0.000` |")

    lines.extend(
        [
            "",
            "## Longest Observed Samples",
            "",
            "| sample_id | preset | status | observed_seconds | index_seconds | summarizer_seconds | critic_seconds | transcript_words |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        (
            f"| `{row.sample_id}` | `{row.preset}` | `{row.final_status or row.status}` | "
            f"`{row.observed_duration_seconds:.3f}` | `{row.index_duration_seconds:.3f}` | "
            f"`{row.summarizer_duration_seconds:.3f}` | `{row.critic_duration_seconds:.3f}` | "
            f"`{row.transcript_words if row.transcript_words is not None else 'n/a'}` |"
        )
        for row in longest_rows
    )
    return "\n".join(lines) + "\n"


def _write_summary_json(
    *,
    path: Path,
    sample_records: list[SampleRecord],
    stage_segments: list[StageSegment],
    active_states: dict[str, ActiveWorkerState],
    worker_issues: list[WorkerIssue],
    analysis_result: AnalysisResult,
    failure_rows: list[dict[str, Any]],
) -> None:
    payload = {
        "analysis": asdict(analysis_result),
        "active_workers": [
            asdict(state)
            for state in sorted(
                active_states.values(),
                key=lambda item: (item.preset, item.worker_label),
            )
        ],
        "worker_issues": [asdict(issue) for issue in worker_issues],
        "failure_rows": failure_rows,
        "sample_records": [asdict(record) for record in sample_records],
        "stage_segments": [asdict(segment) for segment in stage_segments],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_sweep(
    *,
    sweep_root: Path,
    output_dir: Path | None = None,
    reporter: StatusReporter | None = None,
) -> AnalysisResult:
    """Analyze one SPoRC duration sweep and emit graphs plus text summaries."""
    sweep_root = sweep_root.resolve()
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = sweep_root / "analysis" / stamp
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if reporter is not None:
        reporter(
            f"Analyzing SPoRC sweep '{sweep_root.name}' into '{output_dir.name}'..."
        )

    with report_loading_duration("SPoRC sweep analysis", reporter=reporter):
        all_stage_segments: list[StageSegment] = []
        active_states: dict[str, ActiveWorkerState] = {}
        worker_issues: list[WorkerIssue] = []
        log_paths = sorted((sweep_root / "logs").glob("*.log"))
        with report_loading_duration("SPoRC sweep logs", reporter=reporter):
            for log_path in log_paths:
                stage_segments, active_state, issues = _parse_worker_log(log_path)
                all_stage_segments.extend(stage_segments)
                if active_state is not None:
                    active_states[active_state.worker_label] = active_state
                worker_issues.extend(issues)
        if reporter is not None:
            reporter(
                "Parsed "
                f"{len(log_paths)} worker log(s), {len(all_stage_segments)} stage segment(s), "
                f"and {len(worker_issues)} worker issue(s)."
            )

        with report_loading_duration("SPoRC sweep checkpoints", reporter=reporter):
            checkpoint_results, checkpoint_active_samples = _load_checkpoint_rows(
                sweep_root / "checkpoints"
            )
        if reporter is not None:
            reporter(
                "Loaded "
                f"{len(checkpoint_results)} completed checkpoint row(s) and "
                f"{len(checkpoint_active_samples)} active checkpoint sample(s)."
            )

        relevant_sample_ids = {
            sample_id
            for _, sample_id in checkpoint_results
            if sample_id
        }
        relevant_sample_ids.update(
            segment.sample_id for segment in all_stage_segments if segment.sample_id
        )
        relevant_sample_ids.update(
            state.sample_id for state in active_states.values() if state.sample_id
        )
        relevant_sample_ids.update(
            str(payload.get("sample_id") or "")
            for payload in checkpoint_active_samples.values()
            if str(payload.get("sample_id") or "").strip()
        )

        with report_loading_duration("SPoRC sample payload cache", reporter=reporter):
            sample_payloads = _load_sample_payloads(
                sweep_root / "queues",
                sample_ids=relevant_sample_ids,
            )
        if reporter is not None:
            reporter(
                "Loaded "
                f"{len(sample_payloads)} unique sample payload(s) for "
                f"{len(relevant_sample_ids)} observed sample id(s)."
            )

        with report_loading_duration(
            "transcript complexity metrics",
            reporter=reporter,
        ):
            complexity_by_sample = _compute_transcript_complexity(sample_payloads)

        with report_loading_duration("merged sample analysis rows", reporter=reporter):
            sample_records = _merge_sample_records(
                stage_segments=all_stage_segments,
                active_states=active_states,
                checkpoint_results=checkpoint_results,
                checkpoint_active_samples=checkpoint_active_samples,
                sample_payloads=sample_payloads,
                complexity_by_sample=complexity_by_sample,
                worker_issues=worker_issues,
            )
            failure_rows = _build_failure_rows(
                sample_records=sample_records,
                worker_issues=worker_issues,
            )
        if reporter is not None:
            reporter(
                f"Merged {len(sample_records)} sample row(s) with "
                f"{len(failure_rows)} failure/waste bucket(s)."
            )

        graph_paths = [
            output_dir / "01_worker_gantt.png",
            output_dir / "02_preset_worker_time_breakdown.png",
            output_dir / "03_stage_concurrency.png",
            output_dir / "04_failure_waste_waterfall.png",
            output_dir / "05_sample_complexity_scatter.png",
        ]
        with report_loading_duration("analysis graphs", reporter=reporter):
            _plot_worker_gantt(
                stage_segments=all_stage_segments,
                output_path=graph_paths[0],
            )
            _plot_preset_stage_breakdown(
                sample_records=sample_records,
                output_path=graph_paths[1],
            )
            _plot_stage_concurrency(
                stage_segments=all_stage_segments,
                output_path=graph_paths[2],
            )
            _plot_failure_waste(
                failure_rows=failure_rows,
                output_path=graph_paths[3],
            )
            _plot_complexity_scatter(
                sample_records=sample_records,
                output_path=graph_paths[4],
            )

        csv_paths = [
            output_dir / "samples.csv",
            output_dir / "stage_segments.csv",
            output_dir / "active_workers.csv",
            output_dir / "failures.csv",
        ]
        with report_loading_duration("analysis CSV exports", reporter=reporter):
            _write_csv(csv_paths[0], [asdict(record) for record in sample_records])
            _write_csv(
                csv_paths[1],
                [asdict(segment) for segment in all_stage_segments],
            )
            _write_csv(
                csv_paths[2],
                [
                    asdict(state)
                    for state in sorted(
                        active_states.values(),
                        key=lambda item: (item.preset, item.worker_label),
                    )
                ],
            )
            _write_csv(csv_paths[3], failure_rows)

        analysis_result = AnalysisResult(
            sweep_root=str(sweep_root),
            output_dir=str(output_dir),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
            graphs=[str(path) for path in graph_paths],
            csv_exports=[str(path) for path in csv_paths],
            summary_markdown_path=str(output_dir / "summary.md"),
            summary_json_path=str(output_dir / "summary.json"),
        )
        with report_loading_duration("analysis summaries", reporter=reporter):
            summary_markdown = _build_summary_document(
                sweep_root=sweep_root,
                output_dir=output_dir,
                sample_records=sample_records,
                active_states=active_states,
                worker_issues=worker_issues,
                failure_rows=failure_rows,
                analysis_result=analysis_result,
            )
            Path(analysis_result.summary_markdown_path).write_text(
                summary_markdown,
                encoding="utf-8",
            )
            _write_summary_json(
                path=Path(analysis_result.summary_json_path),
                sample_records=sample_records,
                stage_segments=all_stage_segments,
                active_states=active_states,
                worker_issues=worker_issues,
                analysis_result=analysis_result,
                failure_rows=failure_rows,
            )
    return analysis_result


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for sweep bottleneck analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze SPoRC duration-sweep logs and emit bottleneck graphs.",
    )
    parser.add_argument(
        "--sweep-root",
        default="",
        help="Explicit sweep root. Defaults to the latest SPoRC sweep artifact.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Optional output directory. Defaults to <sweep>/analysis/<timestamp>/.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the sweep analysis helper."""
    parser = build_parser()
    args = parser.parse_args(argv)
    sweep_root = (
        Path(str(args.sweep_root)).resolve()
        if str(args.sweep_root).strip()
        else _find_latest_sweep_root(_DEFAULT_SWEEP_ROOT)
    )
    output_dir = (
        Path(str(args.output_dir)).resolve()
        if str(args.output_dir).strip()
        else None
    )
    result = analyze_sweep(
        sweep_root=sweep_root,
        output_dir=output_dir,
        reporter=lambda message: print(message, flush=True),
    )
    print(json.dumps(asdict(result), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
