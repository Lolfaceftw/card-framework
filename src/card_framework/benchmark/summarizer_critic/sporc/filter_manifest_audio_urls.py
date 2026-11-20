"""Filter prepared SPoRC manifests down to reachable source-audio URLs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import threading
import time
from typing import Any
from urllib.parse import urlparse

import requests

from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_LONGFORM_MANIFEST_PATH,
)
from card_framework.shared.perf import StatusReporter, report_loading_duration

_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_MAX_WORKERS = 8
_DEFAULT_MAX_WORKERS_PER_HOST = 2
_DEFAULT_MAX_RETRIES = 1
_DEFAULT_RETRY_BACKOFF_SECONDS = 0.5
_MAX_PENDING_VALIDATION_TASKS_MULTIPLIER = 4
_AUDIO_URL_FILTER_AUDIT_SUFFIX = ".audio_url_audit.json"
_SOURCE_AUDIO_HEADERS = {"User-Agent": "card-framework-sporc-manifest-filter/1.0"}
_SOURCE_AUDIO_GET_HEADERS = {**_SOURCE_AUDIO_HEADERS, "Range": "bytes=0-0"}
_HEAD_FALLBACK_STATUS_CODES = {400, 401, 403, 405, 429, 500, 501, 502, 503, 504}
_TRANSIENT_HTTP_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
_PROGRESS_BAR_WIDTH = 20
_PROGRESS_REPORT_EVERY_ROWS = 25
_PROGRESS_REPORT_INTERVAL_SECONDS = 5.0


@dataclass(slots=True, frozen=True)
class AudioUrlProbeResult:
    is_valid: bool
    url: str
    reason: str
    method: str
    status_code: int | None = None
    resolved_url: str | None = None


@dataclass(slots=True, frozen=True)
class ManifestAudioUrlFilterResult:
    manifest_path: Path
    audit_path: Path
    original_sample_count: int
    kept_sample_count: int
    removed_sample_count: int


@dataclass(slots=True, frozen=True)
class _ValidatedSampleRow:
    raw_row: object
    source_audio_url: str
    probe: AudioUrlProbeResult


@dataclass(slots=True)
class _ProbeCacheEntry:
    event: threading.Event
    result: AudioUrlProbeResult | None = None


def filter_sporc_manifest_audio_urls(
    *,
    manifest_path: Path = DEFAULT_LONGFORM_MANIFEST_PATH,
    output_manifest_path: Path | None = None,
    audit_path: Path | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    max_workers_per_host: int = _DEFAULT_MAX_WORKERS_PER_HOST,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS,
    status_reporter: StatusReporter | None = None,
    session: requests.Session | None = None,
) -> ManifestAudioUrlFilterResult:
    """Rewrite a prepared SPoRC manifest with only reachable source-audio rows."""
    resolved_manifest_path = manifest_path.resolve()
    resolved_output_manifest_path = (
        output_manifest_path.resolve()
        if output_manifest_path is not None
        else resolved_manifest_path
    )
    resolved_audit_path = (
        audit_path.resolve()
        if audit_path is not None
        else resolved_output_manifest_path.with_name(
            f"{resolved_output_manifest_path.stem}{_AUDIO_URL_FILTER_AUDIT_SUFFIX}"
        )
    )
    with report_loading_duration(
        "SPoRC manifest for audio-url filtering",
        reporter=status_reporter,
    ):
        payload = _load_manifest_payload(resolved_manifest_path)
        uses_inline_samples = isinstance(payload.get("samples"), list)
        inline_samples = _load_inline_raw_samples(payload) if uses_inline_samples else None
        total_samples = _resolve_total_sample_count(
            payload,
            manifest_path=resolved_manifest_path,
            inline_samples=inline_samples,
        )

    effective_max_workers = 1 if session is not None else max(int(max_workers), 1)
    effective_max_workers_per_host = 1 if session is not None else max(
        int(max_workers_per_host), 1
    )
    effective_max_retries = max(int(max_retries), 0)
    effective_retry_backoff_seconds = max(float(retry_backoff_seconds), 0.0)

    kept_samples: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    output_samples_path: Path | None = None
    audit_rows_path: Path | None = None

    with report_loading_duration(
        "SPoRC source-audio URL validation",
        reporter=status_reporter,
    ):
        validation_started = time.perf_counter()
        last_progress_report_at = validation_started
        validated_count = 0
        kept_count = 0
        removed_count = 0
        _emit_audio_url_validation_progress(
            validated_count=0,
            total_count=total_samples,
            kept_count=0,
            removed_count=0,
            elapsed_seconds=0.0,
            status_reporter=status_reporter,
        )
        if uses_inline_samples:
            assert inline_samples is not None
            for validated_row in _iter_validated_rows(
                rows=inline_samples,
                timeout_seconds=timeout_seconds,
                max_workers=effective_max_workers,
                max_workers_per_host=effective_max_workers_per_host,
                max_retries=effective_max_retries,
                retry_backoff_seconds=effective_retry_backoff_seconds,
                session=session,
            ):
                kept_count, removed_count = _record_inline_row(
                    validated_row=validated_row,
                    kept_samples=kept_samples,
                    removed_rows=removed_rows,
                    kept_count=kept_count,
                    removed_count=removed_count,
                )
                validated_count, last_progress_report_at = _maybe_report_progress(
                    validated_count=validated_count + 1,
                    total_count=total_samples,
                    kept_count=kept_count,
                    removed_count=removed_count,
                    validation_started=validation_started,
                    last_report_at=last_progress_report_at,
                    status_reporter=status_reporter,
                )
        else:
            output_samples_path = _resolve_output_samples_path(
                payload=payload,
                source_manifest_path=resolved_manifest_path,
                output_manifest_path=resolved_output_manifest_path,
            )
            kept_count, removed_count, audit_rows_path = _filter_compact_rows(
                payload=payload,
                manifest_path=resolved_manifest_path,
                total_samples=total_samples,
                output_samples_path=output_samples_path,
                timeout_seconds=timeout_seconds,
                max_workers=effective_max_workers,
                max_workers_per_host=effective_max_workers_per_host,
                max_retries=effective_max_retries,
                retry_backoff_seconds=effective_retry_backoff_seconds,
                validation_started=validation_started,
                last_report_at=last_progress_report_at,
                status_reporter=status_reporter,
                session=session,
            )
    updated_payload = dict(payload)
    updated_payload["sample_count"] = kept_count
    updated_payload["audio_url_filter"] = {
        "performed_at_utc": _utc_now(),
        "source_manifest_path": str(resolved_manifest_path),
        "original_sample_count": total_samples,
        "kept_sample_count": kept_count,
        "removed_sample_count": removed_count,
        "audit_path": str(resolved_audit_path),
    }
    if uses_inline_samples:
        updated_payload["samples"] = kept_samples
        updated_payload.pop("samples_path", None)
    else:
        updated_payload.pop("samples", None)
        updated_payload["samples_path"] = str(output_samples_path)

    with report_loading_duration(
        "filtered SPoRC manifest rewrite",
        reporter=status_reporter,
    ):
        _write_json_payload(resolved_output_manifest_path, updated_payload)
    with report_loading_duration(
        "SPoRC audio-url filter audit",
        reporter=status_reporter,
    ):
        if uses_inline_samples:
            _write_json_payload(
                resolved_audit_path,
                _audit_payload(
                    source_manifest_path=resolved_manifest_path,
                    output_manifest_path=resolved_output_manifest_path,
                    original_sample_count=total_samples,
                    kept_sample_count=kept_count,
                    removed_sample_count=removed_count,
                    removed_rows=removed_rows,
                ),
            )
        else:
            assert audit_rows_path is not None
            _write_streamed_audit_payload(
                audit_path=resolved_audit_path,
                source_manifest_path=resolved_manifest_path,
                output_manifest_path=resolved_output_manifest_path,
                original_sample_count=total_samples,
                kept_sample_count=kept_count,
                removed_sample_count=removed_count,
                audit_rows_path=audit_rows_path,
            )
    return ManifestAudioUrlFilterResult(
        manifest_path=resolved_output_manifest_path,
        audit_path=resolved_audit_path,
        original_sample_count=total_samples,
        kept_sample_count=kept_count,
        removed_sample_count=removed_count,
    )


def run_filter_manifest_audio_urls_cli(
    manifest_path: str,
    *,
    output_manifest_path: str = "",
    audit_path: str = "",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    max_workers_per_host: int = _DEFAULT_MAX_WORKERS_PER_HOST,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS,
    status_reporter: StatusReporter | None = None,
) -> ManifestAudioUrlFilterResult:
    resolved_manifest_path = Path(manifest_path or DEFAULT_LONGFORM_MANIFEST_PATH).resolve()
    resolved_output_manifest = (
        Path(output_manifest_path).resolve() if output_manifest_path else None
    )
    resolved_audit_path = Path(audit_path).resolve() if audit_path else None
    return filter_sporc_manifest_audio_urls(
        manifest_path=resolved_manifest_path,
        output_manifest_path=resolved_output_manifest,
        audit_path=resolved_audit_path,
        timeout_seconds=timeout_seconds,
        max_workers=max_workers,
        max_workers_per_host=max_workers_per_host,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        status_reporter=status_reporter,
    )


def run_filter_audio_urls_cli(
    manifest_path: str,
    *,
    output_manifest: str = "",
    audit_path: str = "",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    max_workers: int = _DEFAULT_MAX_WORKERS,
    max_workers_per_host: int = _DEFAULT_MAX_WORKERS_PER_HOST,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS,
    status_reporter: StatusReporter | None = None,
) -> ManifestAudioUrlFilterResult:
    return run_filter_manifest_audio_urls_cli(
        manifest_path,
        output_manifest_path=output_manifest,
        audit_path=audit_path,
        timeout_seconds=timeout_seconds,
        max_workers=max_workers,
        max_workers_per_host=max_workers_per_host,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        status_reporter=status_reporter,
    )


def _load_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise RuntimeError(f"Prepared dataset manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Prepared dataset manifest must contain an object.")
    return payload


def _load_inline_raw_samples(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise RuntimeError("Prepared dataset manifest must contain inline samples.")
    return [row for row in raw_samples if isinstance(row, dict)]


def _iter_compact_raw_rows(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
) -> Iterator[dict[str, Any]]:
    samples_path = _resolve_samples_path(payload, manifest_path=manifest_path)
    if samples_path is None or not samples_path.exists():
        raise RuntimeError(
            "Prepared dataset manifest must contain inline samples or a valid samples_path."
        )
    with gzip.open(samples_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                yield row


def _resolve_total_sample_count(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
    inline_samples: list[dict[str, Any]] | None,
) -> int:
    raw_sample_count = payload.get("sample_count")
    if isinstance(raw_sample_count, int):
        return raw_sample_count
    if isinstance(raw_sample_count, float):
        return int(raw_sample_count)
    if inline_samples is not None:
        return len(inline_samples)
    samples_path = _resolve_samples_path(payload, manifest_path=manifest_path)
    if samples_path is None or not samples_path.exists():
        raise RuntimeError(
            "Prepared dataset manifest must contain inline samples or a valid samples_path."
        )
    count = 0
    with gzip.open(samples_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            if raw_line.strip():
                count += 1
    return count


def _resolve_samples_path(payload: dict[str, Any], *, manifest_path: Path) -> Path | None:
    raw_samples_path = str(payload.get("samples_path") or "").strip()
    if not raw_samples_path:
        return None
    candidate = Path(raw_samples_path)
    return (manifest_path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()


def _resolve_output_samples_path(
    *,
    payload: dict[str, Any],
    source_manifest_path: Path,
    output_manifest_path: Path,
) -> Path:
    existing_samples_path = _resolve_samples_path(payload, manifest_path=source_manifest_path)
    if existing_samples_path is not None and output_manifest_path == source_manifest_path:
        return existing_samples_path
    return output_manifest_path.with_name(f"{output_manifest_path.stem}.samples.jsonl.gz")


def _record_inline_row(
    *,
    validated_row: _ValidatedSampleRow,
    kept_samples: list[dict[str, Any]],
    removed_rows: list[dict[str, Any]],
    kept_count: int,
    removed_count: int,
) -> tuple[int, int]:
    if isinstance(validated_row.raw_row, dict) and validated_row.probe.is_valid:
        kept_samples.append(validated_row.raw_row)
        return kept_count + 1, removed_count
    removed_rows.append(_removed_row_payload(validated_row))
    return kept_count, removed_count + 1


def _maybe_report_progress(
    *,
    validated_count: int,
    total_count: int,
    kept_count: int,
    removed_count: int,
    validation_started: float,
    last_report_at: float,
    status_reporter: StatusReporter | None,
) -> tuple[int, float]:
    now = time.perf_counter()
    if _should_report_audio_url_validation_progress(
        validated_count=validated_count,
        total_count=total_count,
        last_report_at=last_report_at,
        now=now,
    ):
        _emit_audio_url_validation_progress(
            validated_count=validated_count,
            total_count=total_count,
            kept_count=kept_count,
            removed_count=removed_count,
            elapsed_seconds=max(now - validation_started, 0.0),
            status_reporter=status_reporter,
        )
        return validated_count, now
    return validated_count, last_report_at


def _filter_compact_rows(
    *,
    payload: dict[str, Any],
    manifest_path: Path,
    total_samples: int,
    output_samples_path: Path,
    timeout_seconds: float,
    max_workers: int,
    max_workers_per_host: int,
    max_retries: int,
    retry_backoff_seconds: float,
    validation_started: float,
    last_report_at: float,
    status_reporter: StatusReporter | None,
    session: requests.Session | None,
) -> tuple[int, int, Path]:
    kept_samples_temp = output_samples_path.with_name(f"{output_samples_path.name}.tmp")
    audit_rows_path = output_samples_path.with_name(f"{output_samples_path.stem}.audit_rows.jsonl")
    kept_samples_temp.unlink(missing_ok=True)
    audit_rows_path.unlink(missing_ok=True)
    kept_count = 0
    removed_count = 0
    validated_count = 0
    output_samples_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            gzip.open(kept_samples_temp, "wt", encoding="utf-8") as kept_handle,
            audit_rows_path.open("w", encoding="utf-8") as audit_rows_handle,
        ):
            for validated_row in _iter_validated_rows(
                rows=_iter_compact_raw_rows(payload, manifest_path=manifest_path),
                timeout_seconds=timeout_seconds,
                max_workers=max_workers,
                max_workers_per_host=max_workers_per_host,
                max_retries=max_retries,
                retry_backoff_seconds=retry_backoff_seconds,
                session=session,
            ):
                if isinstance(validated_row.raw_row, dict) and validated_row.probe.is_valid:
                    kept_handle.write(
                        json.dumps(
                            validated_row.raw_row,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    kept_handle.write("\n")
                    kept_count += 1
                else:
                    audit_rows_handle.write(
                        json.dumps(
                            _removed_row_payload(validated_row),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )
                    audit_rows_handle.write("\n")
                    removed_count += 1
                validated_count, last_report_at = _maybe_report_progress(
                    validated_count=validated_count + 1,
                    total_count=total_samples,
                    kept_count=kept_count,
                    removed_count=removed_count,
                    validation_started=validation_started,
                    last_report_at=last_report_at,
                    status_reporter=status_reporter,
                )
        kept_samples_temp.replace(output_samples_path)
    except Exception:
        kept_samples_temp.unlink(missing_ok=True)
        audit_rows_path.unlink(missing_ok=True)
        raise
    return kept_count, removed_count, audit_rows_path


def _iter_validated_rows(
    *,
    rows: Iterable[dict[str, Any]],
    timeout_seconds: float,
    max_workers: int,
    max_workers_per_host: int,
    max_retries: int,
    retry_backoff_seconds: float,
    session: requests.Session | None,
) -> Iterator[_ValidatedSampleRow]:
    probe_cache: dict[str, _ProbeCacheEntry] = {}
    probe_cache_lock = threading.Lock()
    host_limiters: dict[str, threading.BoundedSemaphore] = {}
    host_limiters_lock = threading.Lock()
    session_pool = _ThreadLocalSessionPool(base_session=session)
    try:
        if max_workers <= 1:
            for row in rows:
                yield _validate_row(
                    row=row,
                    timeout_seconds=timeout_seconds,
                    max_workers_per_host=max_workers_per_host,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    probe_cache=probe_cache,
                    probe_cache_lock=probe_cache_lock,
                    host_limiters=host_limiters,
                    host_limiters_lock=host_limiters_lock,
                    session_pool=session_pool,
                )
            return
        max_pending_tasks = max(max_workers * _MAX_PENDING_VALIDATION_TASKS_MULTIPLIER, 1)
        pending_futures: dict[int, Future[_ValidatedSampleRow]] = {}
        next_to_emit = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for row_index, row in enumerate(rows):
                pending_futures[row_index] = executor.submit(
                    _validate_row,
                    row=row,
                    timeout_seconds=timeout_seconds,
                    max_workers_per_host=max_workers_per_host,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    probe_cache=probe_cache,
                    probe_cache_lock=probe_cache_lock,
                    host_limiters=host_limiters,
                    host_limiters_lock=host_limiters_lock,
                    session_pool=session_pool,
                )
                if len(pending_futures) >= max_pending_tasks:
                    yield from _drain_pending_validated_rows(
                        pending_futures=pending_futures,
                        next_to_emit=next_to_emit,
                        wait_for_oldest=True,
                    )
                    next_to_emit += 1
            while pending_futures:
                yield from _drain_pending_validated_rows(
                    pending_futures=pending_futures,
                    next_to_emit=next_to_emit,
                    wait_for_oldest=True,
                )
                next_to_emit += 1
    finally:
        session_pool.close_all()


def _drain_pending_validated_rows(
    *,
    pending_futures: dict[int, Future[_ValidatedSampleRow]],
    next_to_emit: int,
    wait_for_oldest: bool = False,
) -> Iterator[_ValidatedSampleRow]:
    if not pending_futures:
        return
    if wait_for_oldest:
        if next_to_emit in pending_futures:
            yield pending_futures.pop(next_to_emit).result()
        return
    future = pending_futures.get(next_to_emit)
    if future is not None and future.done():
        yield pending_futures.pop(next_to_emit).result()


def _validate_row(
    *,
    row: dict[str, Any],
    timeout_seconds: float,
    max_workers_per_host: int,
    max_retries: int,
    retry_backoff_seconds: float,
    probe_cache: dict[str, _ProbeCacheEntry],
    probe_cache_lock: threading.Lock,
    host_limiters: dict[str, threading.BoundedSemaphore],
    host_limiters_lock: threading.Lock,
    session_pool: "_ThreadLocalSessionPool",
) -> _ValidatedSampleRow:
    source_audio_url = _resolve_sample_source_audio_url(row)
    probe = _probe_with_cache(
        source_audio_url=source_audio_url,
        timeout_seconds=timeout_seconds,
        max_workers_per_host=max_workers_per_host,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        probe_cache=probe_cache,
        probe_cache_lock=probe_cache_lock,
        host_limiters=host_limiters,
        host_limiters_lock=host_limiters_lock,
        session_pool=session_pool,
    )
    return _ValidatedSampleRow(
        raw_row=row,
        source_audio_url=source_audio_url,
        probe=probe,
    )


def _probe_with_cache(
    *,
    source_audio_url: str,
    timeout_seconds: float,
    max_workers_per_host: int,
    max_retries: int,
    retry_backoff_seconds: float,
    probe_cache: dict[str, _ProbeCacheEntry],
    probe_cache_lock: threading.Lock,
    host_limiters: dict[str, threading.BoundedSemaphore],
    host_limiters_lock: threading.Lock,
    session_pool: "_ThreadLocalSessionPool",
) -> AudioUrlProbeResult:
    cache_key = _normalize_source_audio_target(source_audio_url)
    if not cache_key:
        return _probe_source_audio_target(
            source_audio_url,
            session=session_pool.get_session(),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    should_compute = False
    with probe_cache_lock:
        entry = probe_cache.get(cache_key)
        if entry is None:
            entry = _ProbeCacheEntry(event=threading.Event())
            probe_cache[cache_key] = entry
            should_compute = True
    if not should_compute:
        entry.event.wait()
        return entry.result or AudioUrlProbeResult(
            is_valid=False,
            url=source_audio_url,
            reason="cache_result_missing",
            method="none",
        )
    try:
        effective_max_retries = max(int(max_retries), 0)
        result: AudioUrlProbeResult | None = None
        for attempt_index in range(effective_max_retries + 1):
            limiter = (
                _remote_host_limiter(
                    source_audio_url,
                    max_workers_per_host=max_workers_per_host,
                    host_limiters=host_limiters,
                    host_limiters_lock=host_limiters_lock,
                )
                if _is_remote_source_audio_target(source_audio_url)
                else nullcontext()
            )
            with limiter:
                result = _probe_source_audio_target(
                    source_audio_url,
                    session=session_pool.get_session(),
                    timeout_seconds=timeout_seconds,
                    max_retries=0,
                    retry_backoff_seconds=0.0,
            )
            if result.is_valid or not _is_transient_probe_result(result):
                break
            if attempt_index < effective_max_retries:
                time.sleep(retry_backoff_seconds * (2**attempt_index))
    except Exception as exc:  # pragma: no cover - defensive fallback
        result = AudioUrlProbeResult(
            is_valid=False,
            url=source_audio_url,
            reason=f"unexpected_{type(exc).__name__}",
            method="none",
        )
    entry.result = result
    entry.event.set()
    return result


def _remote_host_limiter(
    source_audio_url: str,
    *,
    max_workers_per_host: int,
    host_limiters: dict[str, threading.BoundedSemaphore],
    host_limiters_lock: threading.Lock,
) -> threading.BoundedSemaphore:
    host_key = _remote_host_key(source_audio_url)
    with host_limiters_lock:
        limiter = host_limiters.get(host_key)
        if limiter is None:
            limiter = threading.BoundedSemaphore(max_workers_per_host)
            host_limiters[host_key] = limiter
        return limiter


class _ThreadLocalSessionPool:
    def __init__(self, *, base_session: requests.Session | None) -> None:
        self._base_session = base_session
        self._local = threading.local()
        self._sessions: list[requests.Session] = []
        self._lock = threading.Lock()

    def get_session(self) -> requests.Session:
        if self._base_session is not None:
            return self._base_session
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            with self._lock:
                self._sessions.append(session)
            self._local.session = session
        return session

    def close_all(self) -> None:
        if self._base_session is not None:
            return
        with self._lock:
            sessions = tuple(self._sessions)
            self._sessions.clear()
        for session in sessions:
            session.close()


def _resolve_sample_source_audio_url(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, dict) else {}
    for candidate in (
        metadata_mapping.get("source_audio_url"),
        sample.get("episode_key"),
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _normalize_source_audio_target(source_audio_url: str) -> str:
    candidate = str(source_audio_url or "").strip()
    if not candidate:
        return ""
    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"}:
        normalized_query = f"?{parsed.query}" if parsed.query else ""
        normalized_fragment = f"#{parsed.fragment}" if parsed.fragment else ""
        return (
            f"{parsed.scheme.lower()}://{parsed.netloc.casefold()}"
            f"{parsed.path or ''}{normalized_query}{normalized_fragment}"
        )
    return str(Path(candidate).expanduser())


def _removed_row_payload(validated_row: _ValidatedSampleRow) -> dict[str, Any]:
    row = validated_row.raw_row
    if not isinstance(row, dict):
        return {
            "sample_id": "",
            "episode_key": "",
            "source_audio_url": validated_row.source_audio_url,
            "reason": validated_row.probe.reason,
            "probe_method": validated_row.probe.method,
            "status_code": validated_row.probe.status_code,
        }
    return {
        "sample_id": str(row.get("sample_id") or "").strip(),
        "episode_key": str(row.get("episode_key") or "").strip(),
        "source_audio_url": validated_row.source_audio_url,
        "reason": validated_row.probe.reason,
        "probe_method": validated_row.probe.method,
        "status_code": validated_row.probe.status_code,
    }


def _audit_payload(
    *,
    source_manifest_path: Path,
    output_manifest_path: Path,
    original_sample_count: int,
    kept_sample_count: int,
    removed_sample_count: int,
    removed_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at_utc": _utc_now(),
        "source_manifest_path": str(source_manifest_path),
        "output_manifest_path": str(output_manifest_path),
        "original_sample_count": original_sample_count,
        "kept_sample_count": kept_sample_count,
        "removed_sample_count": removed_sample_count,
        "removed_samples": removed_rows,
    }


def _write_streamed_audit_payload(
    *,
    audit_path: Path,
    source_manifest_path: Path,
    output_manifest_path: Path,
    original_sample_count: int,
    kept_sample_count: int,
    removed_sample_count: int,
    audit_rows_path: Path,
) -> None:
    temp_path = audit_path.with_name(f"{audit_path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write("{\n")
        header_fields = (
            ("created_at_utc", _utc_now()),
            ("source_manifest_path", str(source_manifest_path)),
            ("output_manifest_path", str(output_manifest_path)),
            ("original_sample_count", original_sample_count),
            ("kept_sample_count", kept_sample_count),
            ("removed_sample_count", removed_sample_count),
        )
        for field_name, field_value in header_fields:
            handle.write(f"  {json.dumps(field_name)}: {json.dumps(field_value)},\n")
        handle.write('  "removed_samples": [\n')
        first_row = True
        with audit_rows_path.open("r", encoding="utf-8") as rows_handle:
            for raw_line in rows_handle:
                line = raw_line.strip()
                if not line:
                    continue
                if not first_row:
                    handle.write(",\n")
                handle.write("    ")
                handle.write(line)
                first_row = False
        if not first_row:
            handle.write("\n")
        handle.write("  ]\n")
        handle.write("}\n")
    temp_path.replace(audit_path)
    audit_rows_path.unlink(missing_ok=True)


def _format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "n/a"
    rounded_seconds = int(round(seconds))
    hours, remainder = divmod(rounded_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours}h{minutes:02d}m"
    if minutes > 0:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _validation_progress_snapshot(
    *,
    validated_count: int,
    total_count: int,
    elapsed_seconds: float,
) -> tuple[str, str]:
    if validated_count <= 0 or elapsed_seconds <= 0:
        average_rows_per_minute = None
    else:
        average_rows_per_minute = validated_count / (elapsed_seconds / 60.0)
    average_text = (
        f"{average_rows_per_minute:.2f} rows/min"
        if average_rows_per_minute is not None
        else "n/a"
    )
    remaining_rows = max(total_count - validated_count, 0)
    if remaining_rows <= 0:
        eta_text = "0s"
    elif average_rows_per_minute is None or average_rows_per_minute <= 0:
        eta_text = "n/a"
    else:
        eta_text = _format_eta((remaining_rows / average_rows_per_minute) * 60.0)
    return average_text, eta_text


def _render_validation_progress_bar(
    *,
    validated_count: int,
    total_count: int,
) -> str:
    if total_count <= 0:
        filled_width = _PROGRESS_BAR_WIDTH
    else:
        filled_width = int(
            round((validated_count / float(total_count)) * _PROGRESS_BAR_WIDTH)
        )
    filled_width = min(max(filled_width, 0), _PROGRESS_BAR_WIDTH)
    return "[" + ("#" * filled_width) + ("-" * (_PROGRESS_BAR_WIDTH - filled_width)) + "]"


def _emit_audio_url_validation_progress(
    *,
    validated_count: int,
    total_count: int,
    kept_count: int,
    removed_count: int,
    elapsed_seconds: float,
    status_reporter: StatusReporter | None,
) -> None:
    if status_reporter is None:
        return
    average_text, eta_text = _validation_progress_snapshot(
        validated_count=validated_count,
        total_count=total_count,
        elapsed_seconds=elapsed_seconds,
    )
    progress_bar = _render_validation_progress_bar(
        validated_count=validated_count,
        total_count=total_count,
    )
    progress_ratio = (
        (validated_count / float(total_count)) if total_count > 0 else 1.0
    )
    status_reporter(
        "Loading SPoRC source-audio URL validation... validated "
        f"{validated_count:,} / {total_count:,} rows {progress_bar} "
        f"{progress_ratio * 100.0:5.1f}%; kept {kept_count:,}; removed "
        f"{removed_count:,}; rolling_avg={average_text}; eta={eta_text}."
    )


def _should_report_audio_url_validation_progress(
    *,
    validated_count: int,
    total_count: int,
    last_report_at: float,
    now: float,
) -> bool:
    if total_count <= 0:
        return False
    if validated_count >= total_count:
        return True
    if validated_count <= 0:
        return False
    if validated_count % _PROGRESS_REPORT_EVERY_ROWS == 0:
        return True
    return (now - last_report_at) >= _PROGRESS_REPORT_INTERVAL_SECONDS


def _is_remote_source_audio_target(source_audio_url: str) -> bool:
    return urlparse(str(source_audio_url or "").strip()).scheme in {"http", "https"}


def _remote_host_key(source_audio_url: str) -> str:
    parsed = urlparse(str(source_audio_url or "").strip())
    return parsed.netloc.casefold() or "__unknown__"


def _probe_source_audio_target(
    source_audio_url: str,
    *,
    session: requests.Session,
    timeout_seconds: float,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_backoff_seconds: float = _DEFAULT_RETRY_BACKOFF_SECONDS,
) -> AudioUrlProbeResult:
    candidate = str(source_audio_url or "").strip()
    if not candidate:
        return AudioUrlProbeResult(
            is_valid=False,
            url="",
            reason="missing_source_audio_url",
            method="none",
        )
    parsed = urlparse(candidate)
    if parsed.scheme in {"http", "https"}:
        return _probe_remote_source_audio_url(
            candidate,
            session=session,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            retry_backoff_seconds=retry_backoff_seconds,
        )
    local_path = Path(candidate).expanduser()
    if local_path.exists() and local_path.is_file():
        return AudioUrlProbeResult(
            is_valid=True,
            url=candidate,
            reason="local_file_exists",
            method="local",
            resolved_url=str(local_path.resolve()),
        )
    return AudioUrlProbeResult(
        is_valid=False,
        url=candidate,
        reason="unsupported_or_missing_local_path",
        method="local",
    )


def _probe_remote_source_audio_url(
    source_audio_url: str,
    *,
    session: requests.Session,
    timeout_seconds: float,
    max_retries: int,
    retry_backoff_seconds: float,
) -> AudioUrlProbeResult:
    attempts = max(int(max_retries), 0) + 1
    last_result: AudioUrlProbeResult | None = None
    for attempt_index in range(attempts):
        last_result = _probe_remote_source_audio_url_once(
            source_audio_url,
            session=session,
            timeout_seconds=timeout_seconds,
        )
        if last_result.is_valid or not _is_transient_probe_result(last_result):
            return last_result
        if attempt_index + 1 >= attempts:
            return last_result
        time.sleep(max(float(retry_backoff_seconds), 0.0) * (2**attempt_index))
    return last_result or AudioUrlProbeResult(
        is_valid=False,
        url=source_audio_url,
        reason="request_failed",
        method="none",
    )


def _probe_remote_source_audio_url_once(
    source_audio_url: str,
    *,
    session: requests.Session,
    timeout_seconds: float,
) -> AudioUrlProbeResult:
    request_timeout = (10.0, max(float(timeout_seconds), 1.0))
    head_error: str | None = None
    try:
        response = session.head(
            source_audio_url,
            allow_redirects=True,
            timeout=request_timeout,
            headers=_SOURCE_AUDIO_HEADERS,
        )
        response.close()
        if response.status_code < 400:
            return AudioUrlProbeResult(
                is_valid=True,
                url=source_audio_url,
                reason="reachable",
                method="HEAD",
                status_code=int(response.status_code),
                resolved_url=str(response.url or source_audio_url),
            )
        if int(response.status_code) not in _HEAD_FALLBACK_STATUS_CODES:
            return AudioUrlProbeResult(
                is_valid=False,
                url=source_audio_url,
                reason=f"http_{int(response.status_code)}",
                method="HEAD",
                status_code=int(response.status_code),
                resolved_url=str(response.url or source_audio_url),
            )
        head_error = f"http_{int(response.status_code)}"
    except requests.RequestException as exc:
        head_error = f"request_exception:{type(exc).__name__}"
    try:
        with session.get(
            source_audio_url,
            allow_redirects=True,
            stream=True,
            timeout=request_timeout,
            headers=_SOURCE_AUDIO_GET_HEADERS,
        ) as response:
            if response.status_code >= 400:
                return AudioUrlProbeResult(
                    is_valid=False,
                    url=source_audio_url,
                    reason=f"http_{int(response.status_code)}",
                    method="GET",
                    status_code=int(response.status_code),
                    resolved_url=str(response.url or source_audio_url),
                )
            for _chunk in response.iter_content(chunk_size=1):
                break
            return AudioUrlProbeResult(
                is_valid=True,
                url=source_audio_url,
                reason="reachable",
                method="GET",
                status_code=int(response.status_code),
                resolved_url=str(response.url or source_audio_url),
            )
    except requests.RequestException as exc:
        return AudioUrlProbeResult(
            is_valid=False,
            url=source_audio_url,
            reason=f"{head_error or 'request_failed'}:{type(exc).__name__}",
            method="GET",
        )


def _is_transient_probe_result(probe: AudioUrlProbeResult) -> bool:
    if probe.is_valid:
        return False
    if probe.status_code is not None and probe.status_code in _TRANSIENT_HTTP_STATUS_CODES:
        return True
    return "request_exception:" in probe.reason


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
