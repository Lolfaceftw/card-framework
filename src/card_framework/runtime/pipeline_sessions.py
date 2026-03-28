"""SQLite-backed stage-level pipeline session persistence and resume planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
from typing import Any, Literal, cast

from card_framework.shared.summary_xml import parse_summary_xml

PipelineStageName = Literal["stage-1", "stage-2", "stage-3", "stage-4"]
PipelineSessionStatus = Literal[
    "prepared",
    "running",
    "paused",
    "failed",
    "completed",
]
StageAttemptStatus = Literal["running", "completed", "paused", "failed"]
_VALID_STAGE_NAMES: tuple[PipelineStageName, ...] = (
    "stage-1",
    "stage-2",
    "stage-3",
    "stage-4",
)


@dataclass(slots=True, frozen=True)
class PipelineSessionRecord:
    """Represent one persisted pipeline session row."""

    session_id: str
    created_at_utc: str
    updated_at_utc: str
    status: PipelineSessionStatus
    requested_start_stage: PipelineStageName
    resume_stage: PipelineStageName | None
    current_stage: PipelineStageName | None
    prepared_overrides: tuple[str, ...]
    artifacts: dict[str, str]
    options: dict[str, Any]
    last_error: str
    last_exit_reason: str


@dataclass(slots=True, frozen=True)
class PipelineResumePlan:
    """Describe where one stored session should resume."""

    session_id: str
    status: PipelineSessionStatus
    requested_start_stage: PipelineStageName
    resume_stage: PipelineStageName | None
    current_stage: PipelineStageName | None
    prepared_overrides: tuple[str, ...]
    artifacts: dict[str, str]
    options: dict[str, Any]
    stage_validity: dict[PipelineStageName, bool]
    last_error: str
    last_exit_reason: str

    @property
    def is_completed(self) -> bool:
        """Return whether the session has already finished every required stage."""
        return self.resume_stage is None


class PipelineSessionStore:
    """Persist pipeline session state and resolve stage-level resume plans."""

    def __init__(self, database_path: Path) -> None:
        """Initialize the store and ensure its schema exists."""
        self.database_path = database_path.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    def create_or_replace_session(
        self,
        *,
        session_id: str,
        requested_start_stage: PipelineStageName,
        prepared_overrides: list[str],
        artifacts: dict[str, str],
        options: dict[str, Any],
    ) -> None:
        """Create or replace one prepared pipeline session."""
        now_iso = _utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO pipeline_sessions (
                    session_id,
                    created_at_utc,
                    updated_at_utc,
                    status,
                    requested_start_stage,
                    resume_stage,
                    current_stage,
                    prepared_overrides_json,
                    artifacts_json,
                    options_json,
                    last_error,
                    last_exit_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at_utc=excluded.updated_at_utc,
                    status=excluded.status,
                    requested_start_stage=excluded.requested_start_stage,
                    resume_stage=excluded.resume_stage,
                    current_stage=excluded.current_stage,
                    prepared_overrides_json=excluded.prepared_overrides_json,
                    artifacts_json=excluded.artifacts_json,
                    options_json=excluded.options_json,
                    last_error='',
                    last_exit_reason=''
                """,
                (
                    session_id,
                    now_iso,
                    now_iso,
                    "prepared",
                    requested_start_stage,
                    requested_start_stage,
                    "",
                    json.dumps(prepared_overrides, ensure_ascii=True),
                    json.dumps(artifacts, ensure_ascii=True),
                    json.dumps(options, ensure_ascii=True),
                    "",
                    "",
                ),
            )

    def get_session(self, session_id: str) -> PipelineSessionRecord | None:
        """Return one session record when it exists."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pipeline_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def mark_stage_started(
        self,
        *,
        session_id: str,
        stage_name: PipelineStageName,
    ) -> None:
        """Mark one pipeline stage as the active stage for a session."""
        now_iso = _utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    UPDATE pipeline_sessions
                    SET updated_at_utc = ?,
                        status = 'running',
                        resume_stage = ?,
                        current_stage = ?,
                        last_error = '',
                        last_exit_reason = ''
                    WHERE session_id = ?
                    """,
                    (now_iso, stage_name, stage_name, session_id),
                )
                connection.execute(
                    """
                    INSERT INTO stage_attempts (
                        session_id,
                        stage_name,
                        status,
                        started_at_utc,
                        finished_at_utc,
                        detail_json,
                        error_text
                    ) VALUES (?, ?, ?, ?, NULL, ?, '')
                    """,
                    (
                        session_id,
                        stage_name,
                        "running",
                        now_iso,
                        "{}",
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def mark_stage_completed(
        self,
        *,
        session_id: str,
        stage_name: PipelineStageName,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """Mark one pipeline stage complete."""
        now_iso = _utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    UPDATE pipeline_sessions
                    SET updated_at_utc = ?,
                        current_stage = '',
                        last_error = '',
                        last_exit_reason = ''
                    WHERE session_id = ?
                    """,
                    (now_iso, session_id),
                )
                connection.execute(
                    """
                    UPDATE stage_attempts
                    SET status = 'completed',
                        finished_at_utc = ?,
                        detail_json = ?,
                        error_text = ''
                    WHERE attempt_id = (
                        SELECT attempt_id
                        FROM stage_attempts
                        WHERE session_id = ? AND stage_name = ?
                        ORDER BY attempt_id DESC
                        LIMIT 1
                    )
                    """,
                    (
                        now_iso,
                        json.dumps(detail or {}, ensure_ascii=True),
                        session_id,
                        stage_name,
                    ),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def mark_session_completed(self, *, session_id: str) -> None:
        """Mark a session fully completed."""
        now_iso = _utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE pipeline_sessions
                SET updated_at_utc = ?,
                    status = 'completed',
                    resume_stage = '',
                    current_stage = '',
                    last_error = '',
                    last_exit_reason = 'completed'
                WHERE session_id = ?
                """,
                (now_iso, session_id),
            )

    def mark_session_paused(
        self,
        *,
        session_id: str,
        current_stage: PipelineStageName | None,
        reason: str,
    ) -> None:
        """Mark a session paused, typically after Ctrl+C."""
        self._mark_session_terminal_state(
            session_id=session_id,
            status="paused",
            current_stage=current_stage,
            reason=reason,
        )

    def mark_session_failed(
        self,
        *,
        session_id: str,
        current_stage: PipelineStageName | None,
        reason: str,
    ) -> None:
        """Mark a session failed because the runtime crashed or raised."""
        self._mark_session_terminal_state(
            session_id=session_id,
            status="failed",
            current_stage=current_stage,
            reason=reason,
        )

    def compute_resume_plan(self, *, session_id: str) -> PipelineResumePlan:
        """Validate stored checkpoints and return the next stage to run."""
        record = self.get_session(session_id)
        if record is None:
            raise RuntimeError(f"Pipeline session does not exist: {session_id}")
        stage_validity = {
            "stage-1": _is_valid_transcript_path(record.artifacts.get("transcript_path", "")),
            "stage-2": _is_valid_summary_path(record.artifacts.get("session_summary_path", "")),
            "stage-3": _is_valid_voice_clone_manifest_path(
                record.artifacts.get("voice_clone_manifest_path", "")
            ),
            "stage-4": _is_valid_interjector_manifest_path(
                record.artifacts.get("interjector_manifest_path", "")
            ),
        }
        resume_stage = _resolve_resume_stage(record=record, stage_validity=stage_validity)
        return PipelineResumePlan(
            session_id=record.session_id,
            status=record.status,
            requested_start_stage=record.requested_start_stage,
            resume_stage=resume_stage,
            current_stage=record.current_stage,
            prepared_overrides=record.prepared_overrides,
            artifacts=dict(record.artifacts),
            options=dict(record.options),
            stage_validity=stage_validity,
            last_error=record.last_error,
            last_exit_reason=record.last_exit_reason,
        )

    def cleanup_for_stage_restart(
        self,
        *,
        session_id: str,
        stage_name: PipelineStageName | None = None,
    ) -> None:
        """Delete stale artifacts for the stage that will be rerun."""
        record = self.get_session(session_id)
        if record is None:
            raise RuntimeError(f"Pipeline session does not exist: {session_id}")
        resolved_stage_name = stage_name
        if resolved_stage_name is None:
            resolved_stage_name = self.compute_resume_plan(
                session_id=session_id
            ).resume_stage
        if resolved_stage_name is None:
            return

        artifacts = record.artifacts
        transcript_path = _resolve_optional_path(artifacts.get("transcript_path", ""))
        session_summary_path = _resolve_optional_path(
            artifacts.get("session_summary_path", "")
        )
        work_dir = _resolve_optional_path(artifacts.get("work_dir", ""))
        voice_clone_dir = _resolve_parent_dir(
            artifacts.get("voice_clone_manifest_path", "")
        )
        interjector_dir = _resolve_parent_dir(
            artifacts.get("interjector_manifest_path", "")
        )

        if resolved_stage_name == "stage-1":
            _delete_path(transcript_path)
            _delete_path(session_summary_path)
            _delete_path(voice_clone_dir, is_dir=True)
            _delete_path(interjector_dir, is_dir=True)
            _delete_path(work_dir, is_dir=True)
            return
        if resolved_stage_name == "stage-2":
            _delete_path(session_summary_path)
            _delete_path(voice_clone_dir, is_dir=True)
            _delete_path(interjector_dir, is_dir=True)
            return
        if resolved_stage_name == "stage-3":
            _delete_path(voice_clone_dir, is_dir=True)
            _delete_path(interjector_dir, is_dir=True)
            return
        _delete_path(interjector_dir, is_dir=True)

    def _mark_session_terminal_state(
        self,
        *,
        session_id: str,
        status: Literal["paused", "failed"],
        current_stage: PipelineStageName | None,
        reason: str,
    ) -> None:
        """Persist one paused or failed session state."""
        now_iso = _utc_now_iso()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                resolved_current_stage = current_stage or self._resolve_session_current_stage(
                    connection=connection,
                    session_id=session_id,
                )
                resume_stage = resolved_current_stage or self._resolve_session_resume_stage(
                    connection=connection,
                    session_id=session_id,
                )
                connection.execute(
                    """
                    UPDATE pipeline_sessions
                    SET updated_at_utc = ?,
                        status = ?,
                        resume_stage = ?,
                        current_stage = '',
                        last_error = ?,
                        last_exit_reason = ?
                    WHERE session_id = ?
                    """,
                    (
                        now_iso,
                        status,
                        resume_stage or "",
                        reason,
                        status,
                        session_id,
                    ),
                )
                if resolved_current_stage is not None:
                    connection.execute(
                        """
                        UPDATE stage_attempts
                        SET status = ?,
                            finished_at_utc = ?,
                            error_text = ?
                        WHERE attempt_id = (
                            SELECT attempt_id
                            FROM stage_attempts
                            WHERE session_id = ? AND stage_name = ?
                            ORDER BY attempt_id DESC
                            LIMIT 1
                        )
                        """,
                        (
                            status,
                            now_iso,
                            reason,
                            session_id,
                            resolved_current_stage,
                        ),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def _resolve_session_current_stage(
        self,
        *,
        connection: sqlite3.Connection,
        session_id: str,
    ) -> PipelineStageName | None:
        """Resolve the current in-flight stage stored on the session row."""
        row = connection.execute(
            "SELECT current_stage FROM pipeline_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        raw_current_stage = str(row["current_stage"] or "").strip()
        if raw_current_stage in _VALID_STAGE_NAMES:
            return cast(PipelineStageName, raw_current_stage)
        return None

    def _resolve_session_resume_stage(
        self,
        *,
        connection: sqlite3.Connection,
        session_id: str,
    ) -> PipelineStageName | None:
        """Resolve the current stored resume stage directly from the session row."""
        row = connection.execute(
            "SELECT resume_stage, requested_start_stage FROM pipeline_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        raw_resume_stage = str(row["resume_stage"] or "").strip()
        if raw_resume_stage in _VALID_STAGE_NAMES:
            return cast(PipelineStageName, raw_resume_stage)
        raw_requested_start_stage = str(row["requested_start_stage"] or "").strip()
        if raw_requested_start_stage in _VALID_STAGE_NAMES:
            return cast(PipelineStageName, raw_requested_start_stage)
        return None

    def _connect(self) -> sqlite3.Connection:
        """Open one SQLite connection for store operations."""
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _initialize_schema(self) -> None:
        """Ensure the store schema exists."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS pipeline_sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requested_start_stage TEXT NOT NULL,
                    resume_stage TEXT NOT NULL,
                    current_stage TEXT NOT NULL,
                    prepared_overrides_json TEXT NOT NULL,
                    artifacts_json TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    last_error TEXT NOT NULL,
                    last_exit_reason TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS stage_attempts (
                    attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    stage_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at_utc TEXT NOT NULL,
                    finished_at_utc TEXT,
                    detail_json TEXT NOT NULL,
                    error_text TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES pipeline_sessions(session_id)
                        ON DELETE CASCADE
                );
                """
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> PipelineSessionRecord:
        """Convert one SQLite row into a typed session record."""
        raw_resume_stage = str(row["resume_stage"] or "").strip()
        resume_stage = (
            cast(PipelineStageName, raw_resume_stage)
            if raw_resume_stage in _VALID_STAGE_NAMES
            else None
        )
        raw_current_stage = str(row["current_stage"] or "").strip()
        current_stage = (
            cast(PipelineStageName, raw_current_stage)
            if raw_current_stage in _VALID_STAGE_NAMES
            else None
        )
        return PipelineSessionRecord(
            session_id=str(row["session_id"]),
            created_at_utc=str(row["created_at_utc"]),
            updated_at_utc=str(row["updated_at_utc"]),
            status=cast(PipelineSessionStatus, str(row["status"])),
            requested_start_stage=cast(
                PipelineStageName,
                str(row["requested_start_stage"]),
            ),
            resume_stage=resume_stage,
            current_stage=current_stage,
            prepared_overrides=tuple(
                item
                for item in json.loads(str(row["prepared_overrides_json"]))
                if isinstance(item, str)
            ),
            artifacts={
                str(key): str(value)
                for key, value in json.loads(str(row["artifacts_json"])).items()
            },
            options=dict(json.loads(str(row["options_json"]))),
            last_error=str(row["last_error"] or ""),
            last_exit_reason=str(row["last_exit_reason"] or ""),
        )


def _resolve_resume_stage(
    *,
    record: PipelineSessionRecord,
    stage_validity: dict[PipelineStageName, bool],
) -> PipelineStageName | None:
    """Resolve the next stage to run from validated checkpoints."""
    start_stage = record.requested_start_stage
    if start_stage == "stage-1":
        if not stage_validity["stage-1"]:
            return "stage-1"
        if not stage_validity["stage-2"]:
            return "stage-2"
        if bool(record.options.get("voice_clone_enabled", False)) and not stage_validity["stage-3"]:
            return "stage-3"
        if bool(record.options.get("interjector_enabled", False)) and not stage_validity["stage-4"]:
            return "stage-4"
        return None
    if start_stage == "stage-2":
        if not stage_validity["stage-2"]:
            return "stage-2"
        if bool(record.options.get("voice_clone_enabled", False)) and not stage_validity["stage-3"]:
            return "stage-3"
        if bool(record.options.get("interjector_enabled", False)) and not stage_validity["stage-4"]:
            return "stage-4"
        return None
    if start_stage == "stage-3":
        if bool(record.options.get("voice_clone_enabled", True)) and not stage_validity["stage-3"]:
            return "stage-3"
        if bool(record.options.get("interjector_enabled", False)) and not stage_validity["stage-4"]:
            return "stage-4"
        return None
    if not stage_validity["stage-4"]:
        return "stage-4"
    return None


def _is_valid_transcript_path(path_value: str) -> bool:
    """Return whether one transcript JSON checkpoint exists and parses."""
    path = _resolve_optional_path(path_value)
    if path is None or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    return isinstance(payload, dict) and isinstance(payload.get("segments", []), list)


def _is_valid_summary_path(path_value: str) -> bool:
    """Return whether one summary XML checkpoint exists and parses."""
    path = _resolve_optional_path(path_value)
    if path is None or not path.is_file():
        return False
    try:
        parse_summary_xml(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return True


def _is_valid_voice_clone_manifest_path(path_value: str) -> bool:
    """Return whether one stage-3 voice-clone manifest exists and is usable."""
    return _is_valid_artifact_manifest_path(
        path_value=path_value,
        merged_output_key="merged_output_audio_path",
        artifact_output_key="output_audio_path",
    )


def _is_valid_interjector_manifest_path(path_value: str) -> bool:
    """Return whether one stage-4 interjector manifest exists and is usable."""
    return _is_valid_artifact_manifest_path(
        path_value=path_value,
        merged_output_key="merged_output_audio_path",
        artifact_output_key="output_audio_path",
    )


def _is_valid_artifact_manifest_path(
    *,
    path_value: str,
    merged_output_key: str,
    artifact_output_key: str,
) -> bool:
    """Return whether one manifest exists and its referenced artifacts exist."""
    path = _resolve_optional_path(path_value)
    if path is None or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    raw_artifacts = payload.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        return False
    merged_output_value = str(payload.get(merged_output_key, "") or "").strip()
    if merged_output_value:
        merged_output_path = _resolve_manifest_relative_path(
            manifest_path=path,
            path_value=merged_output_value,
        )
        if not merged_output_path.is_file():
            return False
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            return False
        output_path_value = str(raw_artifact.get(artifact_output_key, "") or "").strip()
        if not output_path_value:
            return False
        resolved_output_path = _resolve_manifest_relative_path(
            manifest_path=path,
            path_value=output_path_value,
        )
        if not resolved_output_path.is_file():
            return False
    return True


def _resolve_manifest_relative_path(*, manifest_path: Path, path_value: str) -> Path:
    """Resolve one manifest-relative artifact path."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def _resolve_optional_path(path_value: str) -> Path | None:
    """Resolve one optional path string into a ``Path`` when present."""
    normalized = str(path_value).strip()
    if not normalized:
        return None
    return Path(normalized).expanduser().resolve()


def _resolve_parent_dir(path_value: str) -> Path | None:
    """Resolve the parent directory for one optional artifact path."""
    path = _resolve_optional_path(path_value)
    if path is None:
        return None
    return path.parent


def _delete_path(path: Path | None, *, is_dir: bool = False) -> None:
    """Best-effort removal for stale session artifacts before stage restart."""
    if path is None or not path.exists():
        return
    if is_dir:
        shutil.rmtree(path, ignore_errors=True)
        return
    try:
        path.unlink()
    except FileNotFoundError:
        return


def _utc_now_iso() -> str:
    """Return the current UTC timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
