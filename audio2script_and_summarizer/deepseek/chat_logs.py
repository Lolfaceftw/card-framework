"""DeepSeek chat trace persistence helpers."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TextIO

DEEPSEEK_CHAT_LOG_FILE = "trace.ndjson"
DEEPSEEK_CHAT_LOG_META_FILE = "run_meta.json"
DEEPSEEK_CHAT_LOG_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
DEEPSEEK_CHAT_LOG_STREAM_FLUSH_INTERVAL_SECONDS = 10.0
EndpointMode = Literal["beta", "stable"]


@dataclass(slots=True)
class DeepSeekChatLogWriter:
    """Persist per-call and global DeepSeek trace events as line-delimited JSON."""

    run_directory: Path
    log_path: Path
    _log_handle: TextIO
    _open_call_ids: set[str]
    stream_flush_interval_seconds: float = (
        DEEPSEEK_CHAT_LOG_STREAM_FLUSH_INTERVAL_SECONDS
    )
    _next_call_index: int = 0

    @classmethod
    def create(
        cls,
        root_dir: str | Path,
        *,
        stream_flush_interval_seconds: float = (
            DEEPSEEK_CHAT_LOG_STREAM_FLUSH_INTERVAL_SECONDS
        ),
    ) -> DeepSeekChatLogWriter:
        """Create a timestamped DeepSeek chat-log run directory."""
        if (
            not math.isfinite(stream_flush_interval_seconds)
            or stream_flush_interval_seconds <= 0
        ):
            raise ValueError("stream_flush_interval_seconds must be positive and finite.")
        resolved_root = Path(root_dir).resolve()
        resolved_root.mkdir(parents=True, exist_ok=True)
        run_directory = _build_unique_chat_log_run_directory(resolved_root)
        run_directory.mkdir(parents=True, exist_ok=False)
        log_path = run_directory / DEEPSEEK_CHAT_LOG_FILE
        log_handle = log_path.open("a", encoding="utf-8")
        writer = cls(
            run_directory=run_directory,
            log_path=log_path,
            _log_handle=log_handle,
            _open_call_ids=set(),
            stream_flush_interval_seconds=stream_flush_interval_seconds,
        )
        writer._write_run_metadata()
        writer.write_global_event(
            {
                "event": "run_start",
                "run_directory": str(run_directory),
            }
        )
        return writer

    def write_global_event(self, payload: dict[str, object]) -> None:
        """Append one event to the single run NDJSON trace log."""
        record = {
            "ts_utc": _utc_now_iso(),
            **payload,
        }
        self._write_json_line(self._log_handle, record)

    def start_call(
        self,
        *,
        call_type: str,
        endpoint_mode: EndpointMode,
        model: str,
        metadata: dict[str, object] | None = None,
    ) -> str:
        """Register one call and write a call-start event."""
        self._next_call_index += 1
        call_id = f"call_{self._next_call_index:04d}"
        self._open_call_ids.add(call_id)
        start_payload: dict[str, object] = {
            "event": "call_start",
            "call_id": call_id,
            "call_type": call_type,
            "endpoint_mode": endpoint_mode,
            "model": model,
        }
        if metadata:
            start_payload["metadata"] = metadata
        self.write_call_event(call_id, start_payload)
        return call_id

    def write_call_event(self, call_id: str, payload: dict[str, object]) -> None:
        """Append one call-scoped event to the single run NDJSON trace log."""
        record = {
            "ts_utc": _utc_now_iso(),
            "call_id": call_id,
            **payload,
        }
        self._write_json_line(self._log_handle, record)

    def finish_call(
        self,
        call_id: str,
        *,
        status: Literal["ok", "error"],
        finish_reason: str | None = None,
        usage_total_tokens: int | None = None,
        error: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Mark one call closed with a final completion event."""
        done_payload: dict[str, object] = {
            "event": "call_done",
            "status": status,
        }
        if finish_reason is not None:
            done_payload["finish_reason"] = finish_reason
        if usage_total_tokens is not None:
            done_payload["usage_total_tokens"] = usage_total_tokens
        if error is not None:
            done_payload["error"] = error
        if metadata:
            done_payload["metadata"] = metadata
        self.write_call_event(call_id, done_payload)
        self._open_call_ids.discard(call_id)

    def close(self) -> None:
        """Flush and close the run log file handle."""
        self.write_global_event({"event": "run_done"})
        for call_id in list(self._open_call_ids):
            self.write_call_event(
                call_id,
                {
                    "event": "call_done",
                    "status": "error",
                    "error": "call closed without explicit finish_call",
                },
            )
            self._open_call_ids.discard(call_id)
        self._log_handle.close()

    def _write_run_metadata(self) -> None:
        """Persist static run metadata once per execution run."""
        metadata_path = self.run_directory / DEEPSEEK_CHAT_LOG_META_FILE
        metadata_payload = {
            "run_started_at_utc": _utc_now_iso(),
            "run_directory": str(self.run_directory),
            "log_path": str(self.log_path),
            "stream_flush_interval_seconds": self.stream_flush_interval_seconds,
        }
        with metadata_path.open("w", encoding="utf-8") as metadata_file:
            json.dump(metadata_payload, metadata_file, indent=2, ensure_ascii=True)
            metadata_file.write("\n")

    @staticmethod
    def _write_json_line(handle: TextIO, payload: dict[str, object]) -> None:
        """Write one JSON line and flush immediately for continuous visibility."""
        handle.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        )
        handle.flush()


def _build_unique_chat_log_run_directory(root_dir: Path) -> Path:
    """Return a unique timestamped chat-log run directory path."""
    timestamp = datetime.now(timezone.utc).strftime(DEEPSEEK_CHAT_LOG_TIMESTAMP_FORMAT)
    candidate = root_dir / timestamp
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        suffixed = root_dir / f"{timestamp}_{suffix}"
        if not suffixed.exists():
            return suffixed
        suffix += 1


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

