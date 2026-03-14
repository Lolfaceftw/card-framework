"""Prepare the official QMSum general-summary benchmark slice."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile

from card_framework.agents.utils import count_words
from card_framework.benchmark.artifacts import utc_now_iso
from card_framework.benchmark.summarizer_critic.config import (
    DEFAULT_DATASET_MANIFEST_PATH,
    DEFAULT_PREPARED_ROOT,
    DEFAULT_QMSUM_REPO_DIR,
    DEFAULT_REFERENCE_WPM,
    PreparedBenchmarkSample,
    PreparedDatasetManifest,
    QMSUM_REPO_URL,
)


class DatasetPreparationError(RuntimeError):
    """Raise when the QMSum benchmark slice cannot be prepared."""


def derive_target_seconds(
    reference_summary: str,
    *,
    reference_wpm: float = DEFAULT_REFERENCE_WPM,
) -> int:
    """Derive a duration target from the human reference summary length."""
    if reference_wpm <= 0:
        raise ValueError("reference_wpm must be greater than zero.")
    reference_words = max(1, count_words(reference_summary))
    return max(1, int(round((reference_words / reference_wpm) * 60.0)))


def ensure_official_qmsum_repo(
    *,
    repo_dir: Path = DEFAULT_QMSUM_REPO_DIR,
    force_download: bool = False,
) -> Path:
    """Ensure the official QMSum repository is available locally."""
    data_dir = repo_dir / "data" / "ALL" / "test"
    if data_dir.exists() and not force_download:
        return repo_dir.resolve()

    if force_download and repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    clone_result = subprocess.run(
        ["git", "clone", "--depth", "1", QMSUM_REPO_URL, str(repo_dir)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if clone_result.returncode == 0 and data_dir.exists():
        return repo_dir.resolve()

    branch_names = ("main", "master")
    for branch_name in branch_names:
        archive_url = (
            f"https://github.com/Yale-LILY/QMSum/archive/refs/heads/{branch_name}.zip"
        )
        archive_path = repo_dir.parent / f"qmsum_{branch_name}.zip"
        extract_root = repo_dir.parent / f"QMSum-{branch_name}"
        try:
            urllib.request.urlretrieve(archive_url, archive_path)
        except (urllib.error.URLError, OSError):
            continue
        try:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(repo_dir.parent)
        finally:
            archive_path.unlink(missing_ok=True)
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        if extract_root.exists():
            shutil.move(str(extract_root), str(repo_dir))
        if data_dir.exists():
            return repo_dir.resolve()

    clone_detail = (clone_result.stderr or clone_result.stdout).strip()
    raise DatasetPreparationError(
        "Unable to fetch the official QMSum repository. "
        f"Git clone detail: {clone_detail or 'no output'}"
    )


def _read_qmsum_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise DatasetPreparationError(f"QMSum file must contain an object: {path}")
    return payload


def _coerce_transcript_segments(payload: dict[str, object]) -> list[dict[str, object]]:
    raw_turns = payload.get("meeting_transcripts", [])
    if not isinstance(raw_turns, list) or not raw_turns:
        raise DatasetPreparationError("QMSum meeting is missing meeting_transcripts.")
    segments: list[dict[str, object]] = []
    for index, raw_turn in enumerate(raw_turns):
        if not isinstance(raw_turn, dict):
            continue
        speaker = str(raw_turn.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
        text = str(
            raw_turn.get("content", raw_turn.get("utterance", raw_turn.get("text", "")))
        ).strip()
        if not text:
            continue
        segments.append(
            {
                "speaker": speaker,
                "text": text,
                "start_time": index,
                "end_time": index + 1,
            }
        )
    if not segments:
        raise DatasetPreparationError(
            "QMSum meeting does not contain usable transcript text."
        )
    return segments


def _coerce_general_queries(payload: dict[str, object]) -> list[dict[str, str]]:
    raw_queries = payload.get("general_query_list", [])
    if not isinstance(raw_queries, list) or not raw_queries:
        raise DatasetPreparationError("QMSum meeting is missing general_query_list.")
    queries: list[dict[str, str]] = []
    for raw_query in raw_queries:
        if not isinstance(raw_query, dict):
            continue
        query_text = str(raw_query.get("query", "Summarize the meeting.")).strip()
        answer = str(raw_query.get("answer", "")).strip()
        if not answer:
            continue
        queries.append(
            {"query": query_text or "Summarize the meeting.", "answer": answer}
        )
    if not queries:
        raise DatasetPreparationError(
            "QMSum meeting does not contain usable general answers."
        )
    return queries


def build_prepared_dataset_manifest(
    *,
    qmsum_repo_dir: Path,
    prepared_root: Path = DEFAULT_PREPARED_ROOT,
    reference_wpm: float = DEFAULT_REFERENCE_WPM,
) -> PreparedDatasetManifest:
    """Build the prepared benchmark manifest from an official QMSum checkout."""
    source_root = qmsum_repo_dir / "data" / "ALL" / "test"
    if not source_root.exists():
        raise DatasetPreparationError(
            f"Official QMSum test split was not found at: {source_root}"
        )

    prepared_root.mkdir(parents=True, exist_ok=True)
    transcript_root = prepared_root / "transcripts"
    transcript_root.mkdir(parents=True, exist_ok=True)

    samples: list[PreparedBenchmarkSample] = []
    transcript_cache: dict[str, Path] = {}
    for meeting_path in sorted(source_root.glob("*.json")):
        payload = _read_qmsum_json(meeting_path)
        meeting_id = (
            str(payload.get("meeting_id", meeting_path.stem)).strip()
            or meeting_path.stem
        )
        if meeting_id not in transcript_cache:
            transcript_payload = {
                "segments": _coerce_transcript_segments(payload),
                "metadata": {
                    "dataset": "qmsum",
                    "meeting_id": meeting_id,
                    "source_json_path": str(meeting_path.resolve()),
                },
            }
            transcript_path = transcript_root / f"{meeting_id}.transcript.json"
            transcript_path.write_text(
                json.dumps(transcript_payload, indent=2),
                encoding="utf-8",
            )
            transcript_cache[meeting_id] = transcript_path.resolve()

        general_queries = _coerce_general_queries(payload)
        for query_index, query_payload in enumerate(general_queries, start=1):
            reference_summary = query_payload["answer"]
            target_seconds = derive_target_seconds(
                reference_summary,
                reference_wpm=reference_wpm,
            )
            samples.append(
                PreparedBenchmarkSample(
                    sample_id=f"{meeting_id}__general_{query_index:02d}",
                    dataset="qmsum_general_test",
                    meeting_id=meeting_id,
                    transcript_path=str(transcript_cache[meeting_id]),
                    reference_summary=reference_summary,
                    query_text=query_payload["query"],
                    target_seconds=target_seconds,
                    metadata={
                        "general_query_index": query_index,
                        "reference_word_count": count_words(reference_summary),
                        "source_json_path": str(meeting_path.resolve()),
                    },
                )
            )

    if not samples:
        raise DatasetPreparationError(
            "No QMSum general-summary examples were prepared."
        )

    manifest_path = prepared_root / DEFAULT_DATASET_MANIFEST_PATH.name
    manifest_payload = {
        "dataset_id": "qmsum_general_test",
        "created_at_utc": utc_now_iso(),
        "qmsum_repo_dir": str(qmsum_repo_dir.resolve()),
        "reference_wpm": reference_wpm,
        "sample_count": len(samples),
        "samples": [asdict(sample) for sample in samples],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return PreparedDatasetManifest(
        manifest_path=manifest_path.resolve(),
        dataset_id="qmsum_general_test",
        qmsum_repo_dir=qmsum_repo_dir.resolve(),
        sample_count=len(samples),
        samples=tuple(samples),
    )


def prepare_official_qmsum_benchmark(
    *,
    qmsum_repo_dir: Path = DEFAULT_QMSUM_REPO_DIR,
    prepared_root: Path = DEFAULT_PREPARED_ROOT,
    reference_wpm: float = DEFAULT_REFERENCE_WPM,
    force_download: bool = False,
) -> PreparedDatasetManifest:
    """Fetch QMSum when needed and prepare the reproducible benchmark slice."""
    repo_dir = ensure_official_qmsum_repo(
        repo_dir=qmsum_repo_dir,
        force_download=force_download,
    )
    return build_prepared_dataset_manifest(
        qmsum_repo_dir=repo_dir,
        prepared_root=prepared_root,
        reference_wpm=reference_wpm,
    )


def load_prepared_dataset_manifest(manifest_path: Path) -> PreparedDatasetManifest:
    """Load a previously prepared dataset manifest from disk."""
    if not manifest_path.exists():
        raise DatasetPreparationError(
            f"Prepared dataset manifest does not exist: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetPreparationError(
            "Prepared dataset manifest must contain an object."
        )
    raw_samples = payload.get("samples", [])
    if not isinstance(raw_samples, list) or not raw_samples:
        raise DatasetPreparationError(
            "Prepared dataset manifest must contain a non-empty samples list."
        )
    samples = tuple(
        PreparedBenchmarkSample(
            sample_id=str(raw_sample.get("sample_id", "")).strip(),
            dataset=str(raw_sample.get("dataset", "")).strip(),
            meeting_id=str(raw_sample.get("meeting_id", "")).strip(),
            transcript_path=str(raw_sample.get("transcript_path", "")).strip(),
            reference_summary=str(raw_sample.get("reference_summary", "")).strip(),
            query_text=str(raw_sample.get("query_text", "")).strip(),
            target_seconds=int(raw_sample.get("target_seconds", 0)),
            metadata=(
                dict(raw_sample.get("metadata", {}))
                if isinstance(raw_sample.get("metadata", {}), dict)
                else {}
            ),
        )
        for raw_sample in raw_samples
    )
    return PreparedDatasetManifest(
        manifest_path=manifest_path.resolve(),
        dataset_id=str(payload.get("dataset_id", "qmsum_general_test")).strip()
        or "qmsum_general_test",
        qmsum_repo_dir=Path(str(payload.get("qmsum_repo_dir", ""))).resolve(),
        sample_count=int(payload.get("sample_count", len(samples))),
        samples=samples,
    )
