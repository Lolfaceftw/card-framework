"""Dedupe prepared SPoRC manifests in place."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_LONGFORM_MANIFEST_PATH,
)
from card_framework.shared.perf import StatusReporter, report_loading_duration


@dataclass(slots=True, frozen=True)
class ManifestDedupeResult:
    """Summarize one manifest dedupe pass."""

    manifest_path: Path
    audit_path: Path
    original_sample_count: int
    deduped_sample_count: int
    removed_sample_count: int
    candidate_group_count: int
    transcript_signature_reads: int


def dedupe_sporc_manifest_in_place(
    *,
    manifest_path: Path = DEFAULT_LONGFORM_MANIFEST_PATH,
    audit_path: Path | None = None,
    status_reporter: StatusReporter | None = None,
) -> ManifestDedupeResult:
    """Dedupe one inline SPoRC manifest and rewrite it in place.

    Args:
        manifest_path: Prepared or curated SPoRC manifest to dedupe.
        audit_path: Optional path for the emitted audit JSON. Defaults to a
            sibling file next to the manifest.
        status_reporter: Optional status line callback for operator-visible
            progress updates.

    Returns:
        A summary of the dedupe pass and emitted artifact paths.

    Raises:
        RuntimeError: The manifest shape is unsupported or a transcript payload
            needed for duplicate confirmation is invalid.
    """
    resolved_manifest_path = manifest_path.resolve()
    resolved_audit_path = (
        audit_path.resolve()
        if audit_path is not None
        else resolved_manifest_path.with_name(
            f"{resolved_manifest_path.stem}.dedupe_audit.json"
        )
    )

    with report_loading_duration(
        "SPoRC manifest for dedupe",
        reporter=status_reporter,
    ):
        payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("SPoRC dedupe requires the manifest root to be a JSON object.")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list):
        raise RuntimeError(
            "SPoRC dedupe currently supports manifests with inline `samples` only."
        )

    candidate_groups = _build_candidate_groups(raw_samples)
    candidate_group_count = sum(1 for group in candidate_groups.values() if len(group) > 1)
    transcript_signature_cache: dict[Path, str] = {}
    transcript_signature_reads = 0
    kept_indices: list[int] = []
    removed_rows: list[dict[str, Any]] = []

    with report_loading_duration(
        "SPoRC manifest duplicate confirmation",
        reporter=status_reporter,
    ):
        for group_indices in _iter_group_indices(candidate_groups):
            if len(group_indices) == 1:
                kept_indices.append(group_indices[0])
                continue

            seen_signatures: dict[str, int] = {}
            for sample_index in group_indices:
                sample = raw_samples[sample_index]
                transcript_signature, was_cache_hit = _transcript_signature(
                    sample,
                    manifest_dir=resolved_manifest_path.parent,
                    cache=transcript_signature_cache,
                )
                if not was_cache_hit:
                    transcript_signature_reads += 1
                kept_index = seen_signatures.get(transcript_signature)
                if kept_index is None:
                    seen_signatures[transcript_signature] = sample_index
                    kept_indices.append(sample_index)
                    continue
                removed_rows.append(
                    _build_removed_row(
                        removed_sample=sample,
                        kept_sample=raw_samples[kept_index],
                        transcript_signature=transcript_signature,
                    )
                )

    kept_indices.sort()
    deduped_samples = [raw_samples[index] for index in kept_indices]
    payload["samples"] = deduped_samples
    payload["sample_count"] = len(deduped_samples)
    payload["dedupe"] = {
        "performed_at_utc": _utc_now(),
        "original_sample_count": len(raw_samples),
        "deduped_sample_count": len(deduped_samples),
        "removed_sample_count": len(removed_rows),
        "candidate_group_count": candidate_group_count,
        "transcript_signature_reads": transcript_signature_reads,
        "audit_path": str(resolved_audit_path),
    }

    audit_payload = {
        "created_at_utc": _utc_now(),
        "manifest_path": str(resolved_manifest_path),
        "original_sample_count": len(raw_samples),
        "deduped_sample_count": len(deduped_samples),
        "removed_sample_count": len(removed_rows),
        "candidate_group_count": candidate_group_count,
        "transcript_signature_reads": transcript_signature_reads,
        "removed_samples": removed_rows,
    }

    with report_loading_duration(
        "deduped SPoRC manifest rewrite",
        reporter=status_reporter,
    ):
        _atomic_write_json(resolved_manifest_path, payload)
    with report_loading_duration(
        "SPoRC dedupe audit",
        reporter=status_reporter,
    ):
        _atomic_write_json(resolved_audit_path, audit_payload)

    return ManifestDedupeResult(
        manifest_path=resolved_manifest_path,
        audit_path=resolved_audit_path,
        original_sample_count=len(raw_samples),
        deduped_sample_count=len(deduped_samples),
        removed_sample_count=len(removed_rows),
        candidate_group_count=candidate_group_count,
        transcript_signature_reads=transcript_signature_reads,
    )


def run_dedupe_manifest_cli(
    manifest_path: str,
    *,
    audit_path: str = "",
    status_reporter: StatusReporter | None = None,
) -> ManifestDedupeResult:
    """Run manifest dedupe from CLI-style string arguments."""
    resolved_manifest_path = Path(manifest_path or DEFAULT_LONGFORM_MANIFEST_PATH).resolve()
    resolved_audit_path = Path(audit_path).resolve() if audit_path else None
    return dedupe_sporc_manifest_in_place(
        manifest_path=resolved_manifest_path,
        audit_path=resolved_audit_path,
        status_reporter=status_reporter,
    )


def _build_candidate_groups(
    raw_samples: list[object],
) -> dict[str, list[int]]:
    """Group sample indices by a cheap metadata-level candidate fingerprint."""
    groups: dict[str, list[int]] = defaultdict(list)
    for index, raw_sample in enumerate(raw_samples):
        if not isinstance(raw_sample, dict):
            continue
        groups[_candidate_fingerprint(raw_sample)].append(index)
    return groups


def _iter_group_indices(
    candidate_groups: dict[str, list[int]],
) -> Iterable[list[int]]:
    """Yield grouped indices in stable manifest order."""
    for group_indices in candidate_groups.values():
        yield group_indices


def _candidate_fingerprint(sample: dict[str, Any]) -> str:
    """Build a cheap metadata fingerprint for duplicate-candidate grouping."""
    metadata = sample.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, dict) else {}
    longform = metadata_mapping.get("longform_curation")
    longform_mapping = longform if isinstance(longform, dict) else {}
    title = _normalize_key_text(metadata_mapping.get("episode_title"))
    reference_summary = _normalize_key_text(sample.get("reference_summary"))
    spoken_word_count = int(longform_mapping.get("spoken_word_count") or 0)
    segment_count = int(longform_mapping.get("segment_count") or 0)
    reference_word_count = int(metadata_mapping.get("reference_word_count") or 0)
    candidate_key = "\n".join(
        [
            title,
            reference_summary,
            str(reference_word_count),
            str(spoken_word_count),
            str(segment_count),
        ]
    )
    return hashlib.sha1(candidate_key.encode("utf-8")).hexdigest()


def _normalize_key_text(value: object) -> str:
    """Normalize free text into a comparison-friendly key string."""
    return " ".join(str(value or "").split()).strip().casefold()


def _transcript_signature(
    sample: dict[str, Any],
    *,
    manifest_dir: Path,
    cache: dict[Path, str],
) -> tuple[str, bool]:
    """Return a canonical transcript signature for exact duplicate confirmation."""
    transcript_path = Path(str(sample.get("transcript_path") or ""))
    if not transcript_path.is_absolute():
        transcript_path = (manifest_dir / transcript_path).resolve()
    else:
        transcript_path = transcript_path.resolve()
    if transcript_path in cache:
        return cache[transcript_path], True

    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Transcript payload must be a JSON object: {transcript_path}"
        )
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise RuntimeError(
            f"Transcript payload must contain a `segments` list: {transcript_path}"
        )

    signature_builder = hashlib.sha256()
    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        speaker = str(raw_segment.get("speaker") or "").strip()
        text = " ".join(str(raw_segment.get("text") or "").split()).strip()
        start_time = str(raw_segment.get("start_time") or "")
        end_time = str(raw_segment.get("end_time") or "")
        signature_builder.update(speaker.encode("utf-8"))
        signature_builder.update(b"\x1f")
        signature_builder.update(start_time.encode("utf-8"))
        signature_builder.update(b"\x1f")
        signature_builder.update(end_time.encode("utf-8"))
        signature_builder.update(b"\x1f")
        signature_builder.update(text.encode("utf-8"))
        signature_builder.update(b"\x1e")
    signature = signature_builder.hexdigest()
    cache[transcript_path] = signature
    return signature, False


def _build_removed_row(
    *,
    removed_sample: dict[str, Any],
    kept_sample: dict[str, Any],
    transcript_signature: str,
) -> dict[str, Any]:
    """Build one audit row for a removed duplicate sample."""
    removed_metadata = (
        removed_sample.get("metadata")
        if isinstance(removed_sample.get("metadata"), dict)
        else {}
    )
    kept_metadata = (
        kept_sample.get("metadata")
        if isinstance(kept_sample.get("metadata"), dict)
        else {}
    )
    return {
        "reason": "duplicate_transcript_signature_with_matching_metadata",
        "kept_sample_id": str(kept_sample.get("sample_id") or ""),
        "removed_sample_id": str(removed_sample.get("sample_id") or ""),
        "kept_episode_key": str(kept_sample.get("episode_key") or ""),
        "removed_episode_key": str(removed_sample.get("episode_key") or ""),
        "episode_title": str(
            removed_metadata.get("episode_title") or kept_metadata.get("episode_title") or ""
        ),
        "transcript_signature": transcript_signature,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically to the target path."""
    temp_path = path.with_name(f"{path.name}.tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _utc_now() -> str:
    """Return the current UTC timestamp as ISO 8601."""
    return datetime.now(timezone.utc).isoformat()
