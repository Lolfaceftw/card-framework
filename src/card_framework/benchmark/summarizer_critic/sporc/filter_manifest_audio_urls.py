"""Filter prepared SPoRC manifests down to reachable source-audio URLs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlparse

import requests

from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_LONGFORM_MANIFEST_PATH,
)
from card_framework.shared.perf import StatusReporter, report_loading_duration

_DEFAULT_TIMEOUT_SECONDS = 10.0
_AUDIO_URL_FILTER_AUDIT_SUFFIX = ".audio_url_audit.json"
_SOURCE_AUDIO_HEADERS = {"User-Agent": "card-framework-sporc-manifest-filter/1.0"}
_SOURCE_AUDIO_GET_HEADERS = {
    **_SOURCE_AUDIO_HEADERS,
    "Range": "bytes=0-0",
}
_HEAD_FALLBACK_STATUS_CODES = {400, 401, 403, 405, 429, 500, 501, 502, 503, 504}
_PROGRESS_BAR_WIDTH = 20
_PROGRESS_REPORT_EVERY_ROWS = 25
_PROGRESS_REPORT_INTERVAL_SECONDS = 5.0


@dataclass(slots=True, frozen=True)
class AudioUrlProbeResult:
    """Describe one manifest source-audio URL probe."""

    is_valid: bool
    url: str
    reason: str
    method: str
    status_code: int | None = None
    resolved_url: str | None = None


@dataclass(slots=True, frozen=True)
class ManifestAudioUrlFilterResult:
    """Summarize one SPoRC manifest audio-URL filtering pass."""

    manifest_path: Path
    audit_path: Path
    original_sample_count: int
    kept_sample_count: int
    removed_sample_count: int


def filter_sporc_manifest_audio_urls(
    *,
    manifest_path: Path = DEFAULT_LONGFORM_MANIFEST_PATH,
    output_manifest_path: Path | None = None,
    audit_path: Path | None = None,
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
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
        raw_samples = _load_raw_samples(payload, manifest_path=resolved_manifest_path)

    owned_session = session is None
    active_session = session or requests.Session()
    try:
        kept_samples: list[dict[str, Any]] = []
        removed_rows: list[dict[str, Any]] = []
        total_samples = len(raw_samples)
        validation_started = time.perf_counter()
        last_progress_report_at = validation_started

        with report_loading_duration(
            "SPoRC source-audio URL validation",
            reporter=status_reporter,
        ):
            _emit_audio_url_validation_progress(
                validated_count=0,
                total_count=total_samples,
                kept_count=0,
                removed_count=0,
                elapsed_seconds=0.0,
                status_reporter=status_reporter,
            )
            for sample in raw_samples:
                if not isinstance(sample, dict):
                    removed_rows.append(
                        {
                            "sample_id": "",
                            "episode_key": "",
                            "source_audio_url": "",
                            "reason": "invalid_sample_row",
                            "probe_method": "none",
                            "status_code": None,
                        }
                    )
                    validated_count = len(kept_samples) + len(removed_rows)
                    now = time.perf_counter()
                    if _should_report_audio_url_validation_progress(
                        validated_count=validated_count,
                        total_count=total_samples,
                        last_report_at=last_progress_report_at,
                        now=now,
                    ):
                        _emit_audio_url_validation_progress(
                            validated_count=validated_count,
                            total_count=total_samples,
                            kept_count=len(kept_samples),
                            removed_count=len(removed_rows),
                            elapsed_seconds=max(now - validation_started, 0.0),
                            status_reporter=status_reporter,
                        )
                        last_progress_report_at = now
                    continue
                source_audio_url = _resolve_sample_source_audio_url(sample)
                probe = _probe_source_audio_target(
                    source_audio_url,
                    session=active_session,
                    timeout_seconds=timeout_seconds,
                )
                if probe.is_valid:
                    kept_samples.append(sample)
                    continue
                removed_rows.append(
                    {
                        "sample_id": str(sample.get("sample_id") or "").strip(),
                        "episode_key": str(sample.get("episode_key") or "").strip(),
                        "source_audio_url": source_audio_url,
                        "reason": probe.reason,
                        "probe_method": probe.method,
                        "status_code": probe.status_code,
                    }
                )
                validated_count = len(kept_samples) + len(removed_rows)
                now = time.perf_counter()
                if _should_report_audio_url_validation_progress(
                    validated_count=validated_count,
                    total_count=total_samples,
                    last_report_at=last_progress_report_at,
                    now=now,
                ):
                    _emit_audio_url_validation_progress(
                        validated_count=validated_count,
                        total_count=total_samples,
                        kept_count=len(kept_samples),
                        removed_count=len(removed_rows),
                        elapsed_seconds=max(now - validation_started, 0.0),
                        status_reporter=status_reporter,
                    )
                    last_progress_report_at = now
    finally:
        if owned_session:
            active_session.close()

    updated_payload = dict(payload)
    updated_payload["sample_count"] = len(kept_samples)
    updated_payload["audio_url_filter"] = {
        "performed_at_utc": _utc_now(),
        "source_manifest_path": str(resolved_manifest_path),
        "original_sample_count": len(raw_samples),
        "kept_sample_count": len(kept_samples),
        "removed_sample_count": len(removed_rows),
        "audit_path": str(resolved_audit_path),
    }

    output_samples_path: Path | None = None
    if uses_inline_samples:
        updated_payload["samples"] = kept_samples
        updated_payload.pop("samples_path", None)
    else:
        output_samples_path = _resolve_output_samples_path(
            payload=payload,
            source_manifest_path=resolved_manifest_path,
            output_manifest_path=resolved_output_manifest_path,
        )
        updated_payload.pop("samples", None)
        updated_payload["samples_path"] = str(output_samples_path)

    audit_payload = {
        "created_at_utc": _utc_now(),
        "source_manifest_path": str(resolved_manifest_path),
        "output_manifest_path": str(resolved_output_manifest_path),
        "original_sample_count": len(raw_samples),
        "kept_sample_count": len(kept_samples),
        "removed_sample_count": len(removed_rows),
        "removed_samples": removed_rows,
    }

    with report_loading_duration(
        "filtered SPoRC manifest rewrite",
        reporter=status_reporter,
    ):
        if output_samples_path is not None:
            _write_gzip_jsonl_rows(output_samples_path, kept_samples)
        _write_json_payload(resolved_output_manifest_path, updated_payload)
    with report_loading_duration(
        "SPoRC audio-url filter audit",
        reporter=status_reporter,
    ):
        _write_json_payload(resolved_audit_path, audit_payload)

    return ManifestAudioUrlFilterResult(
        manifest_path=resolved_output_manifest_path,
        audit_path=resolved_audit_path,
        original_sample_count=len(raw_samples),
        kept_sample_count=len(kept_samples),
        removed_sample_count=len(removed_rows),
    )


def run_filter_manifest_audio_urls_cli(
    manifest_path: str,
    *,
    output_manifest_path: str = "",
    audit_path: str = "",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    status_reporter: StatusReporter | None = None,
) -> ManifestAudioUrlFilterResult:
    """Run manifest audio-url filtering from CLI-style string arguments."""
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
        status_reporter=status_reporter,
    )


def run_filter_audio_urls_cli(
    manifest_path: str,
    *,
    output_manifest: str = "",
    audit_path: str = "",
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    status_reporter: StatusReporter | None = None,
) -> ManifestAudioUrlFilterResult:
    """Backward-compatible wrapper for older helper naming."""
    return run_filter_manifest_audio_urls_cli(
        manifest_path,
        output_manifest_path=output_manifest,
        audit_path=audit_path,
        timeout_seconds=timeout_seconds,
        status_reporter=status_reporter,
    )


def _load_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    """Load one prepared SPoRC manifest payload from disk."""
    if not manifest_path.exists():
        raise RuntimeError(f"Prepared dataset manifest does not exist: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Prepared dataset manifest must contain an object.")
    return payload


def _load_raw_samples(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    """Load raw manifest sample rows while preserving their original shape."""
    raw_samples = payload.get("samples")
    if isinstance(raw_samples, list):
        return [row for row in raw_samples if isinstance(row, dict)]

    samples_path = _resolve_samples_path(payload, manifest_path=manifest_path)
    if samples_path is None or not samples_path.exists():
        raise RuntimeError(
            "Prepared dataset manifest must contain inline samples or a valid samples_path."
        )
    rows: list[dict[str, Any]] = []
    with gzip.open(samples_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _resolve_samples_path(payload: dict[str, Any], *, manifest_path: Path) -> Path | None:
    raw_samples_path = str(payload.get("samples_path") or "").strip()
    if not raw_samples_path:
        return None
    candidate = Path(raw_samples_path)
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _resolve_output_samples_path(
    *,
    payload: dict[str, Any],
    source_manifest_path: Path,
    output_manifest_path: Path,
) -> Path:
    existing_samples_path = _resolve_samples_path(payload, manifest_path=source_manifest_path)
    if (
        existing_samples_path is not None
        and output_manifest_path == source_manifest_path
    ):
        return existing_samples_path
    return output_manifest_path.with_name(
        f"{output_manifest_path.stem}.samples.jsonl.gz"
    )


def _format_eta(seconds: float | None) -> str:
    """Format one compact ETA string."""
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
    """Return rolling-throughput and ETA strings for URL validation."""
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
    """Render one compact text progress bar for operator updates."""
    if total_count <= 0:
        filled_width = _PROGRESS_BAR_WIDTH
    else:
        filled_width = int(
            round((validated_count / float(total_count)) * _PROGRESS_BAR_WIDTH)
        )
    filled_width = min(max(filled_width, 0), _PROGRESS_BAR_WIDTH)
    return (
        "["
        + ("#" * filled_width)
        + ("-" * (_PROGRESS_BAR_WIDTH - filled_width))
        + "]"
    )


def _emit_audio_url_validation_progress(
    *,
    validated_count: int,
    total_count: int,
    kept_count: int,
    removed_count: int,
    elapsed_seconds: float,
    status_reporter: StatusReporter | None,
) -> None:
    """Emit one operator-facing progress update for URL validation."""
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
    """Return whether the URL filter should emit another progress update."""
    if total_count <= 0:
        return False
    if validated_count >= total_count:
        return True
    if validated_count <= 0:
        return False
    if validated_count % _PROGRESS_REPORT_EVERY_ROWS == 0:
        return True
    return (now - last_report_at) >= _PROGRESS_REPORT_INTERVAL_SECONDS


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


def _probe_source_audio_target(
    source_audio_url: str,
    *,
    session: requests.Session,
    timeout_seconds: float,
) -> AudioUrlProbeResult:
    """Return whether one manifest source-audio target appears reachable."""
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
) -> AudioUrlProbeResult:
    """Probe one remote source-audio URL with a cheap HEAD/GET fallback path."""
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
        head_error = type(exc).__name__

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
        final_reason = f"{head_error or 'request_failed'}:{type(exc).__name__}"
        return AudioUrlProbeResult(
            is_valid=False,
            url=source_audio_url,
            reason=final_reason,
            method="GET",
        )


def _write_gzip_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one JSONL gzip sidecar atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
        temp_path.replace(path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp_path.replace(path)


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 form."""
    return datetime.now(timezone.utc).isoformat()
