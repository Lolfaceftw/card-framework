"""Dataset manifest utilities for reproducible benchmark runs."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from card_framework.benchmark.artifacts import utc_now_iso
from card_framework.benchmark.types import BenchmarkSample


class ManifestError(RuntimeError):
    """Raised when benchmark manifest data is invalid."""


def _normalize_transcript_path(raw_path: str, repo_root: Path) -> str:
    path = Path(raw_path)
    if path.is_absolute():
        return str(path)
    return str((repo_root / path).resolve())


def _segment_stats(transcript: dict[str, Any]) -> dict[str, int]:
    segments = transcript.get("segments", [])
    unique_speakers = {
        str(segment.get("speaker", "UNKNOWN")) for segment in segments if segment
    }
    approx_words = sum(
        len(str(segment.get("text", "")).split()) for segment in segments if segment
    )
    return {
        "segment_count": len(segments),
        "speaker_count": len(unique_speakers),
        "approx_words": approx_words,
    }


def load_manifest(manifest_path: Path, repo_root: Path) -> list[BenchmarkSample]:
    """Load and validate a benchmark manifest."""
    if not manifest_path.exists():
        raise ManifestError(f"Manifest does not exist: {manifest_path}")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ManifestError("Manifest must contain a non-empty 'samples' array")

    samples: list[BenchmarkSample] = []
    for raw_sample in raw_samples:
        if not isinstance(raw_sample, dict):
            raise ManifestError("Each sample must be an object")

        sample_id = str(raw_sample.get("sample_id", "")).strip()
        dataset = str(raw_sample.get("dataset", "")).strip()
        transcript_path_raw = str(raw_sample.get("transcript_path", "")).strip()
        if not sample_id or not dataset or not transcript_path_raw:
            raise ManifestError(
                "Each sample requires sample_id, dataset, and transcript_path"
            )

        transcript_path = _normalize_transcript_path(transcript_path_raw, repo_root)
        if not Path(transcript_path).exists():
            raise ManifestError(
                f"Sample '{sample_id}' transcript path does not exist: {transcript_path}"
            )

        metadata = raw_sample.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                dataset=dataset,
                transcript_path=transcript_path,
                metadata=metadata,
            )
        )

    return samples


def load_transcript(sample: BenchmarkSample) -> dict[str, Any]:
    """Load transcript JSON for one sample."""
    transcript_path = Path(sample.transcript_path)
    with transcript_path.open("r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ManifestError(f"Transcript payload must be object: {transcript_path}")
    return payload


def prepare_manifest(
    *,
    repo_root: Path,
    output_path: Path,
    sources: list[str],
    num_samples: int,
    local_transcript_path: str,
) -> dict[str, Any]:
    """Prepare a benchmark manifest from selected sources."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []

    normalized_sources = [source.strip().lower() for source in sources if source.strip()]

    if "local" in normalized_sources:
        transcript_path = _normalize_transcript_path(local_transcript_path, repo_root)
        transcript = json.loads(Path(transcript_path).read_text(encoding="utf-8-sig"))
        stats = _segment_stats(transcript)
        samples.append(
            {
                "sample_id": "local_summary",
                "dataset": "local",
                "transcript_path": transcript_path,
                "metadata": stats,
            }
        )

    if "qmsum" in normalized_sources or "ami" in normalized_sources:
        try:
            from datasets import load_dataset
        except Exception as exc:  # pragma: no cover - optional dependency at runtime
            raise ManifestError(
                "Hugging Face datasets extras are required for qmsum/ami source"
            ) from exc

        data_dir = output_path.parent / "prepared_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        if "qmsum" in normalized_sources:
            qmsum = load_dataset("Yale-LILY/QMSum", split="test", streaming=True)
            for index, example in enumerate(qmsum):
                if index >= num_samples:
                    break
                segments = [
                    {
                        "speaker": turn.get("speaker", "UNKNOWN"),
                        "text": turn.get("utterance", ""),
                    }
                    for turn in example.get("meeting_transcripts", [])
                ]
                if not segments:
                    continue
                sample_payload = {"segments": segments}
                sample_path = data_dir / f"qmsum_{index}.json"
                sample_path.write_text(
                    json.dumps(sample_payload, indent=2),
                    encoding="utf-8",
                )
                stats = _segment_stats(sample_payload)
                samples.append(
                    {
                        "sample_id": f"qmsum_{index}",
                        "dataset": "qmsum",
                        "transcript_path": str(sample_path),
                        "metadata": {
                            **stats,
                            "query_id": example.get("query_id", f"qmsum_{index}"),
                        },
                    }
                )

        if "ami" in normalized_sources:
            ami = load_dataset("edinburghcstr/ami", "ihm", split="test", streaming=True)
            ami_iter = iter(ami)
            for index in range(num_samples):
                segments: list[dict[str, str]] = []
                for _ in range(50):
                    try:
                        row = next(ami_iter)
                    except StopIteration:
                        break
                    text = str(row.get("text", ""))
                    if text:
                        segments.append(
                            {
                                "speaker": str(row.get("speaker_id", "UNKNOWN")),
                                "text": text,
                            }
                        )
                if not segments:
                    continue
                sample_payload = {"segments": segments}
                sample_path = data_dir / f"ami_{index}.json"
                sample_path.write_text(
                    json.dumps(sample_payload, indent=2),
                    encoding="utf-8",
                )
                stats = _segment_stats(sample_payload)
                samples.append(
                    {
                        "sample_id": f"ami_{index}",
                        "dataset": "ami",
                        "transcript_path": str(sample_path),
                        "metadata": stats,
                    }
                )

    if not samples:
        raise ManifestError("No samples were prepared. Check sources and inputs.")

    manifest = {
        "manifest_version": "1",
        "created_at_utc": utc_now_iso(),
        "sources": normalized_sources,
        "samples": samples,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Re-load through typed parser to guarantee validity and normalize absolute paths.
    loaded_samples = load_manifest(output_path, repo_root)
    manifest["samples"] = [asdict(sample) for sample in loaded_samples]
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

