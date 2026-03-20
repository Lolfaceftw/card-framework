"""Prepare per-episode source audio, speaker samples, and sampled transcripts."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any
from urllib.parse import urlparse

import requests
from rich.console import Console

from card_framework.agents.utils import count_words
from card_framework.audio_pipeline.factory import build_speaker_sample_generator
from card_framework.audio_pipeline.io import write_transcript_atomic
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_LONGFORM_MANIFEST_PATH,
    PreparedBenchmarkSample,
)
from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    load_prepared_dataset_manifest,
)
from card_framework.cli.calibrate import load_audio_config
from card_framework.orchestration.transcript import Transcript, TranscriptSegment
from card_framework.shared.paths import DEFAULT_CONFIG_PATH, REPO_ROOT

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_OUTPUT_ROOT = (_SCRIPT_DIR / "episode_assets").resolve()
_SOURCE_AUDIO_HEADERS = {"User-Agent": "card-framework-sporc-sampled-segments/1.0"}
_SOURCE_AUDIO_TIMEOUT = (10.0, 300.0)
_DEFAULT_WORDS_PER_MINUTE = 170.0
_DEFAULT_MIN_SEGMENT_MS = 1_000
_DEFAULT_SEGMENT_GAP_MS = 250
_EPISODE_HEADER_PATTERN = re.compile(r"^\s*Episode:\s*(?P<value>.+?)\s*$")
_SAMPLE_ID_HEADER_PATTERN = re.compile(r"^\s*Sample ID:\s*(?P<value>.+?)\s*$")
_SEGMENT_LINE_PATTERN = re.compile(
    r"^\s*(?P<order>\d+)\.\s+\[(?P<speaker>[^\]]+)\]\s+(?P<text>.+?)\s*$"
)


class SampledSegmentPreparationError(RuntimeError):
    """Raise when sampled-segment episode assets cannot be prepared."""


@dataclass(slots=True, frozen=True)
class SampledSegmentFile:
    """Describe one sampled-segment text file."""

    path: Path
    episode_title: str
    sample_id: str | None = None
    segments: tuple["SelectedSegment", ...] = ()


@dataclass(slots=True, frozen=True)
class SelectedSegment:
    """Represent one selected sampled transcript segment."""

    selection_order: int
    speaker_label: str
    segment_text: str
    rationale: str


@dataclass(slots=True, frozen=True)
class EpisodeAssetPlan:
    """Group one SPoRC sample with all matching sampled-segment files."""

    sample: PreparedBenchmarkSample
    episode_title: str
    segment_files: tuple[Path, ...]


@dataclass(slots=True, frozen=True)
class EpisodeAssetResult:
    """Summarize one prepared episode asset bundle."""

    sample_id: str
    episode_title: str
    episode_root: Path
    source_transcript_path: Path
    sampled_transcript_path: Path
    source_audio_url: str
    source_audio_path: Path
    speaker_samples_manifest_path: Path
    segment_files: tuple[Path, ...]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for sampled-segment episode preparation."""
    parser = argparse.ArgumentParser(
        description=(
            "Match sampled-segment episode files back to the prepared SPoRC "
            "manifest, download each episode's source audio, and generate "
            "speaker samples using the prepared full transcript."
        )
    )
    parser.add_argument(
        "segment_files",
        nargs="*",
        help=(
            "Optional sampled-segment text files. When omitted, all *.txt files "
            "in the sampled_segments folder are used."
        ),
    )
    parser.add_argument(
        "--segments-dir",
        default=str(_SCRIPT_DIR),
        help=(
            "Directory that contains the sampled-segment text files. Defaults to "
            "the sampled_segments folder beside this script."
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        default=str(DEFAULT_LONGFORM_MANIFEST_PATH),
        help="Prepared SPoRC manifest JSON path.",
    )
    parser.add_argument(
        "--output-root",
        default=str(_DEFAULT_OUTPUT_ROOT),
        help=(
            "Directory where per-episode audio, transcript, and speaker-sample "
            "artifacts will be written."
        ),
    )
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG_PATH),
        help="Runtime config file used to resolve speaker-sample settings.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload audio and regenerate speaker samples even when present.",
    )
    parser.add_argument(
        "--words-per-minute",
        type=float,
        default=_DEFAULT_WORDS_PER_MINUTE,
        help="Synthetic speaking rate used to timestamp the sampled transcript.",
    )
    parser.add_argument(
        "--min-segment-ms",
        type=int,
        default=_DEFAULT_MIN_SEGMENT_MS,
        help="Minimum synthetic duration per sampled transcript segment.",
    )
    parser.add_argument(
        "--segment-gap-ms",
        type=int,
        default=_DEFAULT_SEGMENT_GAP_MS,
        help="Gap inserted between synthetic sampled transcript segments.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare local audio and speaker-sample assets for sampled segments."""
    args = build_parser().parse_args(argv)
    console = Console()
    segments_dir = Path(args.segments_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    manifest_path = Path(args.dataset_manifest).expanduser().resolve()
    config_path = Path(args.config_file).expanduser().resolve()

    segment_paths = _resolve_segment_files(
        segments_dir=segments_dir,
        requested_paths=args.segment_files,
    )
    segment_files = tuple(
        _parse_sampled_segment_file(path) for path in segment_paths
    )
    plans = _build_episode_asset_plans(
        manifest_path=manifest_path,
        segment_files=segment_files,
    )
    audio_cfg = load_audio_config(REPO_ROOT, config_path=config_path)
    speaker_samples_cfg = dict(audio_cfg.get("speaker_samples", {}))
    speaker_samples_cfg["source_audio"] = "source"
    audio_cfg = dict(audio_cfg)
    audio_cfg["speaker_samples"] = speaker_samples_cfg

    output_root.mkdir(parents=True, exist_ok=True)
    results = [
        _prepare_episode_assets(
            plan=plan,
            audio_cfg=audio_cfg,
            output_root=output_root,
            force=args.force,
            words_per_minute=float(args.words_per_minute),
            min_segment_ms=int(args.min_segment_ms),
            segment_gap_ms=int(args.segment_gap_ms),
            console=console,
        )
        for plan in plans
    ]
    summary_path = output_root / "prepared_episodes_manifest.json"
    summary_path.write_text(
        json.dumps(
            {
                "dataset_manifest_path": str(manifest_path),
                "config_path": str(config_path),
                "output_root": str(output_root),
                "episode_count": len(results),
                "episodes": [_result_to_payload(result) for result in results],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    console.print(f"Wrote episode asset summary: {summary_path}")
    return 0


def _resolve_segment_files(
    *,
    segments_dir: Path,
    requested_paths: list[str],
) -> tuple[Path, ...]:
    if requested_paths:
        resolved: list[Path] = []
        for raw_value in requested_paths:
            candidate = Path(raw_value).expanduser()
            if not candidate.is_absolute():
                candidate = (segments_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()
            if not candidate.is_file():
                raise SampledSegmentPreparationError(
                    f"Sampled-segment file does not exist: {candidate}"
                )
            resolved.append(candidate)
        return tuple(resolved)

    discovered = tuple(sorted(path.resolve() for path in segments_dir.glob("*.txt")))
    if not discovered:
        raise SampledSegmentPreparationError(
            f"No sampled-segment .txt files were found in {segments_dir}"
        )
    return discovered


def _parse_sampled_segment_file(path: Path) -> SampledSegmentFile:
    episode_title = ""
    sample_id: str | None = None
    segments: list[SelectedSegment] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        episode_match = _EPISODE_HEADER_PATTERN.match(raw_line)
        if episode_match is not None and not episode_title:
            episode_title = episode_match.group("value").strip()
            continue
        sample_id_match = _SAMPLE_ID_HEADER_PATTERN.match(raw_line)
        if sample_id_match is not None and sample_id is None:
            candidate = sample_id_match.group("value").strip()
            if candidate:
                sample_id = candidate
            continue
        rationale_line = raw_line.strip()
        if rationale_line.startswith("Rationale:"):
            if not segments:
                raise SampledSegmentPreparationError(
                    f"Sampled-segment file has a rationale before a segment: {path}"
                )
            previous = segments[-1]
            segments[-1] = SelectedSegment(
                selection_order=previous.selection_order,
                speaker_label=previous.speaker_label,
                segment_text=previous.segment_text,
                rationale=rationale_line.split(":", 1)[1].strip(),
            )
            continue
        segment_match = _SEGMENT_LINE_PATTERN.match(raw_line)
        if segment_match is not None:
            segments.append(
                SelectedSegment(
                    selection_order=int(segment_match.group("order")),
                    speaker_label=segment_match.group("speaker").strip(),
                    segment_text=segment_match.group("text").strip(),
                    rationale="",
                )
            )
    if not episode_title:
        raise SampledSegmentPreparationError(
            f"Sampled-segment file is missing an Episode header: {path}"
        )
    if not segments:
        raise SampledSegmentPreparationError(
            f"Sampled-segment file does not contain numbered transcript segments: {path}"
        )
    return SampledSegmentFile(
        path=path.resolve(),
        episode_title=episode_title,
        sample_id=sample_id,
        segments=tuple(segments),
    )


def _build_episode_asset_plans(
    *,
    manifest_path: Path,
    segment_files: tuple[SampledSegmentFile, ...],
) -> tuple[EpisodeAssetPlan, ...]:
    manifest = load_prepared_dataset_manifest(manifest_path)
    samples_by_id = {sample.sample_id: sample for sample in manifest.samples}
    samples_by_title: dict[str, list[PreparedBenchmarkSample]] = defaultdict(list)
    for sample in manifest.samples:
        samples_by_title[_normalize_title_key(_sample_episode_title(sample))].append(
            sample
        )

    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    grouped_titles: dict[str, str] = {}
    grouped_samples: dict[str, PreparedBenchmarkSample] = {}
    for segment_file in segment_files:
        sample = _match_segment_file_to_sample(
            segment_file=segment_file,
            samples_by_id=samples_by_id,
            samples_by_title=samples_by_title,
        )
        grouped_paths[sample.sample_id].append(segment_file.path)
        grouped_titles[sample.sample_id] = _sample_episode_title(sample)
        grouped_samples[sample.sample_id] = sample

    plans = [
        EpisodeAssetPlan(
            sample=grouped_samples[sample_id],
            episode_title=grouped_titles[sample_id],
            segment_files=tuple(sorted(paths)),
        )
        for sample_id, paths in sorted(grouped_paths.items())
    ]
    return tuple(plans)


def _match_segment_file_to_sample(
    *,
    segment_file: SampledSegmentFile,
    samples_by_id: dict[str, PreparedBenchmarkSample],
    samples_by_title: dict[str, list[PreparedBenchmarkSample]],
) -> PreparedBenchmarkSample:
    normalized_title = _normalize_title_key(segment_file.episode_title)
    title_matches = samples_by_title.get(normalized_title, [])
    if segment_file.sample_id:
        sample = samples_by_id.get(segment_file.sample_id)
        if sample is not None:
            return sample
        prefix_matches = [
            candidate
            for candidate_id, candidate in samples_by_id.items()
            if candidate_id.startswith(segment_file.sample_id)
            or segment_file.sample_id.startswith(candidate_id)
        ]
        if len(prefix_matches) == 1:
            return prefix_matches[0]
        if len(title_matches) == 1:
            return title_matches[0]
        raise SampledSegmentPreparationError(
            "Sampled-segment file references a sample_id that is missing from "
            "the prepared manifest, and the episode title was not specific "
            f"enough to recover automatically: {segment_file.sample_id}"
        )

    if not title_matches:
        raise SampledSegmentPreparationError(
            "Unable to match sampled-segment episode title to the prepared "
            f"manifest: '{segment_file.episode_title}' ({segment_file.path})"
        )
    if len(title_matches) > 1:
        raise SampledSegmentPreparationError(
            "Multiple prepared SPoRC samples matched the sampled-segment title "
            f"'{segment_file.episode_title}'. Add the Sample ID header to "
            f"{segment_file.path} to disambiguate."
        )
    return title_matches[0]


def _prepare_episode_assets(
    *,
    plan: EpisodeAssetPlan,
    audio_cfg: dict[str, Any],
    output_root: Path,
    force: bool,
    words_per_minute: float,
    min_segment_ms: int,
    segment_gap_ms: int,
    console: Console,
) -> EpisodeAssetResult:
    episode_root = (
        output_root / f"{_slugify(plan.sample.sample_id)}__{_slugify(plan.episode_title)}"
    ).resolve()
    episode_root.mkdir(parents=True, exist_ok=True)
    copied_segment_files = _copy_segment_files(
        segment_files=plan.segment_files,
        destination_dir=episode_root / "sampled_segments",
    )

    source_transcript = _load_transcript(Path(plan.sample.transcript_path))
    source_audio_url = _normalize_source_audio_url(plan.sample, source_transcript)
    source_audio_path = _resolve_source_audio(
        source_audio_url=source_audio_url,
        destination=_source_audio_destination(
            episode_root=episode_root,
            sample=plan.sample,
            source_audio_url=source_audio_url,
        ),
        force=force,
    )
    source_transcript = _update_transcript_metadata(
        transcript=source_transcript,
        metadata_updates={
            "sample_id": plan.sample.sample_id,
            "episode_title": plan.episode_title,
            "source_audio_url": source_audio_url,
            "source_audio_path": str(source_audio_path),
            "sampled_segments_files": [str(path) for path in copied_segment_files],
        },
    )

    speaker_samples_dir = episode_root / "speaker_samples"
    local_manifest_path = speaker_samples_dir / "manifest.json"
    if force or not local_manifest_path.exists():
        console.print(
            f"Preparing speaker samples for {plan.sample.sample_id}: {plan.episode_title}"
        )
        generator = build_speaker_sample_generator(audio_cfg)
        generation_result = generator.generate(
            transcript_payload=source_transcript.to_payload(),
            source_audio_path=source_audio_path,
            output_dir=speaker_samples_dir,
            progress_callback=None,
        )
        speaker_manifest_path = generation_result.manifest_path.resolve()
        source_transcript = _update_transcript_metadata(
            transcript=source_transcript,
            metadata_updates={
                "speaker_samples_manifest_path": str(speaker_manifest_path),
                "speaker_samples_dir": str(generation_result.output_dir.resolve()),
                "speaker_sample_count": len(generation_result.artifacts),
                "speaker_samples_generated_at_utc": generation_result.generated_at_utc,
            },
        )
    else:
        speaker_manifest_path = local_manifest_path.resolve()
        source_transcript = _update_transcript_metadata(
            transcript=source_transcript,
            metadata_updates={
                "speaker_samples_manifest_path": str(speaker_manifest_path),
                "speaker_samples_dir": str(speaker_samples_dir.resolve()),
            },
        )
        console.print(
            f"Reusing speaker samples for {plan.sample.sample_id}: {speaker_manifest_path}"
        )

    source_transcript_path = episode_root / "prepared_full.transcript.json"
    write_transcript_atomic(source_transcript.to_payload(), source_transcript_path)
    sampled_transcript = _build_sampled_transcript(
        plan=plan,
        sampled_segment_paths=tuple(copied_segment_files),
        source_transcript=source_transcript,
        source_transcript_path=source_transcript_path,
        source_audio_url=source_audio_url,
        source_audio_path=source_audio_path,
        speaker_manifest_path=speaker_manifest_path,
        words_per_minute=words_per_minute,
        min_segment_ms=min_segment_ms,
        segment_gap_ms=segment_gap_ms,
    )
    sampled_transcript_path = episode_root / "sampled.transcript.json"
    write_transcript_atomic(sampled_transcript.to_payload(), sampled_transcript_path)
    episode_info_path = episode_root / "episode_info.json"
    episode_info_path.write_text(
        json.dumps(
            {
                "sample_id": plan.sample.sample_id,
                "episode_title": plan.episode_title,
                "source_audio_url": source_audio_url,
                "source_audio_path": str(source_audio_path),
                "speaker_samples_manifest_path": str(speaker_manifest_path),
                "source_transcript_path": str(source_transcript_path),
                "sampled_transcript_path": str(sampled_transcript_path),
                "segment_files": [str(path) for path in copied_segment_files],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return EpisodeAssetResult(
        sample_id=plan.sample.sample_id,
        episode_title=plan.episode_title,
        episode_root=episode_root,
        source_transcript_path=source_transcript_path.resolve(),
        sampled_transcript_path=sampled_transcript_path.resolve(),
        source_audio_url=source_audio_url,
        source_audio_path=source_audio_path,
        speaker_samples_manifest_path=speaker_manifest_path,
        segment_files=tuple(copied_segment_files),
    )


def _copy_segment_files(
    *,
    segment_files: tuple[Path, ...],
    destination_dir: Path,
) -> tuple[Path, ...]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[Path] = []
    for segment_file in segment_files:
        destination = (destination_dir / segment_file.name).resolve()
        if segment_file.resolve() != destination:
            shutil.copy2(segment_file, destination)
        copied_paths.append(destination)
    return tuple(copied_paths)


def _selected_segments_from_paths(
    segment_files: tuple[Path, ...],
) -> tuple[tuple[Path, SelectedSegment], ...]:
    selected: list[tuple[Path, SelectedSegment]] = []
    for segment_file_path in segment_files:
        parsed = _parse_sampled_segment_file(segment_file_path)
        for segment in parsed.segments:
            selected.append((parsed.path, segment))
    selected.sort(
        key=lambda item: (
            item[0].name.lower(),
            item[1].selection_order,
            item[1].speaker_label.lower(),
            item[1].segment_text.lower(),
        )
    )
    unique: list[tuple[Path, SelectedSegment]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for segment_file_path, segment in selected:
        signature = (
            str(segment_file_path),
            segment.selection_order,
            segment.speaker_label,
            segment.segment_text,
        )
        if signature in seen:
            continue
        seen.add(signature)
        unique.append((segment_file_path, segment))
    return tuple(unique)


def _load_transcript(path: Path) -> Transcript:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise SampledSegmentPreparationError(
            f"Failed to load prepared transcript JSON: {path}"
        ) from exc
    return Transcript.from_mapping(payload)


def _update_transcript_metadata(
    *,
    transcript: Transcript,
    metadata_updates: dict[str, Any],
) -> Transcript:
    updated_metadata = dict(transcript.metadata)
    updated_metadata.update(metadata_updates)
    return transcript.with_metadata(updated_metadata)


def _normalize_source_audio_url(
    sample: PreparedBenchmarkSample,
    transcript: Transcript,
) -> str:
    candidates = (
        str(transcript.metadata.get("source_audio_url", "")).strip(),
        str(sample.metadata.get("source_audio_url", "")).strip(),
        str(sample.episode_key).strip(),
    )
    for value in candidates:
        if value:
            return value
    raise SampledSegmentPreparationError(
        f"SPoRC sample '{sample.sample_id}' is missing a source audio URL or path."
    )


def _resolve_source_audio(
    *,
    source_audio_url: str,
    destination: Path,
    force: bool,
) -> Path:
    parsed = urlparse(source_audio_url)
    if parsed.scheme in {"http", "https"}:
        return _download_binary_to_path(
            source_audio_url=source_audio_url,
            destination=destination,
            force=force,
        )

    candidate = Path(source_audio_url).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise SampledSegmentPreparationError(
            f"Source audio path does not exist: {candidate}"
        )
    return candidate


def _source_audio_destination(
    *,
    episode_root: Path,
    sample: PreparedBenchmarkSample,
    source_audio_url: str,
) -> Path:
    suffix = Path(urlparse(source_audio_url).path).suffix or ".mp3"
    return (episode_root / "source_audio" / f"{sample.sample_id}{suffix}").resolve()


def _download_binary_to_path(
    *,
    source_audio_url: str,
    destination: Path,
    force: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination.resolve()
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    temp_path.unlink(missing_ok=True)
    try:
        with requests.get(
            source_audio_url,
            stream=True,
            timeout=_SOURCE_AUDIO_TIMEOUT,
            headers=_SOURCE_AUDIO_HEADERS,
        ) as response:
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        temp_path.replace(destination)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise SampledSegmentPreparationError(
            "Failed to download source audio "
            f"'{source_audio_url}' to '{destination}'."
        ) from exc
    return destination.resolve()


def _build_sampled_transcript(
    *,
    plan: EpisodeAssetPlan,
    sampled_segment_paths: tuple[Path, ...],
    source_transcript: Transcript,
    source_transcript_path: Path,
    source_audio_url: str,
    source_audio_path: Path,
    speaker_manifest_path: Path,
    words_per_minute: float,
    min_segment_ms: int,
    segment_gap_ms: int,
) -> Transcript:
    if words_per_minute <= 0:
        raise ValueError("words_per_minute must be greater than zero.")
    if min_segment_ms <= 0:
        raise ValueError("min_segment_ms must be greater than zero.")
    if segment_gap_ms < 0:
        raise ValueError("segment_gap_ms must be >= 0.")

    current_start_ms = 0
    sampled_segments: list[TranscriptSegment] = []
    for segment_file_path, selected_segment in _selected_segments_from_paths(
        sampled_segment_paths
    ):
        word_count = max(1, count_words(selected_segment.segment_text))
        estimated_duration_ms = int(
            round((word_count / words_per_minute) * 60.0 * 1000.0)
        )
        duration_ms = max(min_segment_ms, estimated_duration_ms)
        segment_end_ms = current_start_ms + duration_ms
        sampled_segments.append(
            TranscriptSegment(
                speaker=selected_segment.speaker_label or "UNKNOWN",
                text=selected_segment.segment_text,
                start_time=current_start_ms,
                end_time=segment_end_ms,
                extras={
                    "selection_order": selected_segment.selection_order,
                    "rationale": selected_segment.rationale,
                    "sampled_segment_source_file": str(segment_file_path),
                    "synthetic_duration_ms": duration_ms,
                    "synthetic_word_count": word_count,
                },
            )
        )
        current_start_ms = segment_end_ms + segment_gap_ms

    metadata = dict(source_transcript.metadata)
    metadata.update(
        {
            "sample_id": plan.sample.sample_id,
            "episode_title": plan.episode_title,
            "source_audio_url": source_audio_url,
            "source_audio_path": str(source_audio_path),
            "speaker_samples_manifest_path": str(speaker_manifest_path),
            "source_transcript_path": str(source_transcript_path),
            "sampled_segments_files": [str(path) for path in sampled_segment_paths],
            "sampled_segment_count": len(sampled_segments),
            "sampled_transcript_timing_mode": "synthetic_sequential",
            "sampled_transcript_generated_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
        }
    )
    return Transcript(
        segments=tuple(sampled_segments),
        metadata=metadata,
        extras=dict(source_transcript.extras),
    )


def _sample_episode_title(sample: PreparedBenchmarkSample) -> str:
    title = str(sample.metadata.get("episode_title", "")).strip()
    return title or sample.sample_id


def _normalize_title_key(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9]+", " ", value).strip().lower()
    return re.sub(r"\s+", " ", collapsed)


def _slugify(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return collapsed[:80] or "episode"


def _result_to_payload(result: EpisodeAssetResult) -> dict[str, Any]:
    return {
        "sample_id": result.sample_id,
        "episode_title": result.episode_title,
        "episode_root": str(result.episode_root),
        "transcript_path": str(result.sampled_transcript_path),
        "source_transcript_path": str(result.source_transcript_path),
        "sampled_transcript_path": str(result.sampled_transcript_path),
        "source_audio_url": result.source_audio_url,
        "source_audio_path": str(result.source_audio_path),
        "speaker_samples_manifest_path": str(result.speaker_samples_manifest_path),
        "segment_files": [str(path) for path in result.segment_files],
    }


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
