"""Prepare per-episode source audio and speaker samples for sampled segments."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any
from urllib.parse import urlparse

import requests

from card_framework.audio_pipeline.factory import build_speaker_sample_generator
from card_framework.audio_pipeline.io import write_transcript_atomic
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_LONGFORM_MANIFEST_PATH,
    PreparedBenchmarkSample,
)
from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    iter_prepared_dataset_samples,
    load_prepared_dataset_manifest,
)
from card_framework.cli.calibrate import load_audio_config
from card_framework.orchestration.transcript import Transcript
from card_framework.shared.paths import DEFAULT_CONFIG_PATH, REPO_ROOT

_DEFAULT_SEGMENTS_DIR = (
    REPO_ROOT
    / "src"
    / "card_framework"
    / "benchmark"
    / "summarizer_critic"
    / "sporc"
    / "sampled_segments"
).resolve()
_DEFAULT_OUTPUT_DIRNAME = "episodes"
_SOURCE_AUDIO_TIMEOUT = (10.0, 300.0)
_SOURCE_AUDIO_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    ),
    "Accept": "audio/*,*/*;q=0.8",
    "Referer": "https://www.buzzsprout.com/",
}
_EPISODE_HEADER_PATTERN = re.compile(r"^\s*Episode:\s*(?P<value>.+?)\s*$", re.IGNORECASE)
_SAMPLE_ID_HEADER_PATTERN = re.compile(
    r"^\s*Sample ID:\s*(?P<value>.+?)\s*$",
    re.IGNORECASE,
)
_NORMALIZE_TEXT_PATTERN = re.compile(r"[^a-z0-9]+")


class SampledSegmentSpeakerPrepError(RuntimeError):
    """Raise when sampled-segment speaker-sample preparation fails."""


@dataclass(slots=True, frozen=True)
class SampledSegmentFile:
    """Represent one sampled-segment text file to prepare."""

    path: Path
    episode_title: str
    sample_id: str


@dataclass(slots=True, frozen=True)
class ManifestEpisodeMatch:
    """Bundle one manifest entry together with its manifest order."""

    manifest_episode_number: int
    sample: PreparedBenchmarkSample


@dataclass(slots=True, frozen=True)
class PreparedEpisodeResult:
    """Describe one prepared episode artifact bundle."""

    manifest_episode_number: int
    episode_title: str
    sample_id: str
    episode_dir: Path
    sampled_segments_path: Path
    transcript_path: Path
    source_audio_path: Path
    speaker_samples_manifest_path: Path


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for sampled-segment speaker-sample preparation."""

    parser = argparse.ArgumentParser(
        description=(
            "Read sampled-segment text files, match them to SPoRC manifest episodes, "
            "download or copy each episode's source audio into a per-episode folder, "
            "and generate speaker samples using the same post-transcript path as the "
            "main pipeline."
        )
    )
    parser.add_argument(
        "--segments-dir",
        default=str(_DEFAULT_SEGMENTS_DIR),
        help=(
            "Directory that contains sampled-segment .txt files. Defaults to the repo "
            "sampled_segments folder."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="",
        help=(
            "Output root for prepared per-episode folders. Defaults to "
            "<segments-dir>/episodes."
        ),
    )
    parser.add_argument(
        "--dataset-manifest",
        default=str(DEFAULT_LONGFORM_MANIFEST_PATH),
        help="Prepared manifest JSON path used to resolve SPoRC episodes.",
    )
    parser.add_argument(
        "--config-file",
        default=str(DEFAULT_CONFIG_PATH),
        help="Runtime config file used to build the speaker-sample generator.",
    )
    parser.add_argument(
        "--segment-file",
        action="append",
        default=[],
        help=(
            "Optional sampled-segment filename or path to prepare. Repeat to process "
            "a subset. Defaults to every .txt file in --segments-dir."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload remote audio even when the local per-episode copy exists.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help=(
            "Regenerate speaker samples even when the per-episode transcript and "
            "speaker manifest already exist."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare source audio and speaker samples for sampled-segment episodes."""

    args = build_parser().parse_args(argv)
    segments_dir = Path(str(args.segments_dir)).expanduser().resolve()
    output_root = (
        Path(str(args.output_root)).expanduser().resolve()
        if str(args.output_root).strip()
        else (segments_dir / _DEFAULT_OUTPUT_DIRNAME).resolve()
    )
    manifest_path = Path(str(args.dataset_manifest)).expanduser().resolve()
    config_path = Path(str(args.config_file)).expanduser().resolve()

    if not segments_dir.exists():
        raise SampledSegmentSpeakerPrepError(
            f"Sampled-segment directory does not exist: {segments_dir}"
        )
    if not manifest_path.exists():
        raise SampledSegmentSpeakerPrepError(
            f"Prepared SPoRC manifest does not exist: {manifest_path}"
        )
    if not config_path.exists():
        raise SampledSegmentSpeakerPrepError(
            f"Runtime config file does not exist: {config_path}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    sampled_files = _discover_segment_files(
        segments_dir=segments_dir,
        selected_files=tuple(str(item) for item in args.segment_file),
    )
    manifest_matches = _load_manifest_matches(manifest_path)
    audio_cfg = _load_audio_cfg(config_path)

    results: list[PreparedEpisodeResult] = []
    for sampled_file in sampled_files:
        manifest_match = _match_sampled_file_to_manifest(
            sampled_file=sampled_file,
            manifest_matches=manifest_matches,
        )
        results.append(
            _prepare_episode(
                sampled_file=sampled_file,
                manifest_match=manifest_match,
                output_root=output_root,
                audio_cfg=audio_cfg,
                force_download=bool(args.force_download),
                force_regenerate=bool(args.force_regenerate),
            )
        )

    report_path = (output_root / "speaker_sample_prep_report.json").resolve()
    report_payload = {
        "segments_dir": str(segments_dir),
        "output_root": str(output_root),
        "dataset_manifest": str(manifest_path),
        "config_file": str(config_path),
        "episode_count": len(results),
        "episodes": [
            {
                "manifest_episode_number": result.manifest_episode_number,
                "episode_title": result.episode_title,
                "sample_id": result.sample_id,
                "episode_dir": str(result.episode_dir),
                "sampled_segments_path": str(result.sampled_segments_path),
                "transcript_path": str(result.transcript_path),
                "source_audio_path": str(result.source_audio_path),
                "speaker_samples_manifest_path": str(
                    result.speaker_samples_manifest_path
                ),
            }
            for result in results
        ],
    }
    report_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for result in results:
        print(
            f"[{result.manifest_episode_number:04d}] {result.episode_title}\n"
            f"  sample_id: {result.sample_id}\n"
            f"  episode_dir: {result.episode_dir}\n"
            f"  source_audio: {result.source_audio_path}\n"
            f"  speaker_manifest: {result.speaker_samples_manifest_path}"
        )
    print(f"Report: {report_path}")
    return 0


def _discover_segment_files(
    *,
    segments_dir: Path,
    selected_files: tuple[str, ...],
) -> tuple[SampledSegmentFile, ...]:
    if selected_files:
        candidate_paths: list[Path] = []
        for raw_value in selected_files:
            raw_path = Path(raw_value).expanduser()
            candidate_paths.append(
                raw_path.resolve()
                if raw_path.is_absolute()
                else (segments_dir / raw_path).resolve()
            )
    else:
        candidate_paths = sorted(segments_dir.glob("*.txt"))

    if not candidate_paths:
        raise SampledSegmentSpeakerPrepError(
            f"No sampled-segment .txt files were found in: {segments_dir}"
        )
    return tuple(_parse_sampled_segment_file(path) for path in candidate_paths)


def _parse_sampled_segment_file(path: Path) -> SampledSegmentFile:
    if not path.exists():
        raise SampledSegmentSpeakerPrepError(
            f"Sampled-segment file does not exist: {path}"
        )
    episode_title = ""
    sample_id = ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not episode_title:
            match = _EPISODE_HEADER_PATTERN.match(raw_line)
            if match is not None:
                episode_title = match.group("value").strip()
                continue
        if not sample_id:
            match = _SAMPLE_ID_HEADER_PATTERN.match(raw_line)
            if match is not None:
                sample_id = match.group("value").strip()
                continue
        if episode_title and sample_id:
            break
    if not sample_id:
        raise SampledSegmentSpeakerPrepError(
            f"Sampled-segment file is missing a 'Sample ID:' header: {path}"
        )
    if not episode_title:
        episode_title = path.stem
    return SampledSegmentFile(
        path=path.resolve(),
        episode_title=episode_title,
        sample_id=sample_id,
    )


def _load_manifest_matches(manifest_path: Path) -> tuple[ManifestEpisodeMatch, ...]:
    load_prepared_dataset_manifest(manifest_path, include_samples=False)
    return tuple(
        ManifestEpisodeMatch(
            manifest_episode_number=manifest_episode_number,
            sample=sample,
        )
        for manifest_episode_number, sample in enumerate(
            iter_prepared_dataset_samples(manifest_path),
            start=1,
        )
    )


def _match_sampled_file_to_manifest(
    *,
    sampled_file: SampledSegmentFile,
    manifest_matches: tuple[ManifestEpisodeMatch, ...],
) -> ManifestEpisodeMatch:
    for match in manifest_matches:
        if match.sample.sample_id == sampled_file.sample_id:
            return match

    normalized_target_title = _normalize_match_text(sampled_file.episode_title)
    for match in manifest_matches:
        episode_title = str(match.sample.metadata.get("episode_title") or "").strip()
        if (
            episode_title
            and _normalize_match_text(episode_title) == normalized_target_title
        ):
            return match

    raise SampledSegmentSpeakerPrepError(
        "Could not match sampled-segment file to the prepared manifest: "
        f"{sampled_file.path} (sample_id={sampled_file.sample_id!r}, "
        f"episode_title={sampled_file.episode_title!r})"
    )


def _normalize_match_text(value: str) -> str:
    normalized = _NORMALIZE_TEXT_PATTERN.sub(" ", value.strip().lower())
    return " ".join(part for part in normalized.split() if part)


def _load_audio_cfg(config_path: Path) -> dict[str, Any]:
    audio_cfg = load_audio_config(REPO_ROOT, config_path=config_path)
    resolved = deepcopy(audio_cfg)
    speaker_samples_cfg = dict(resolved.get("speaker_samples", {}))
    speaker_samples_cfg["enabled"] = True
    speaker_samples_cfg["source_audio"] = "source"
    resolved["speaker_samples"] = speaker_samples_cfg
    return resolved


def _prepare_episode(
    *,
    sampled_file: SampledSegmentFile,
    manifest_match: ManifestEpisodeMatch,
    output_root: Path,
    audio_cfg: dict[str, Any],
    force_download: bool,
    force_regenerate: bool,
) -> PreparedEpisodeResult:
    sample = manifest_match.sample
    episode_title = (
        str(sample.metadata.get("episode_title") or "").strip()
        or sampled_file.episode_title
    )
    episode_dir = (
        output_root
        / f"{manifest_match.manifest_episode_number:04d}_{_slugify(episode_title)}"
    ).resolve()
    inputs_dir = (episode_dir / "inputs").resolve()
    source_audio_dir = (episode_dir / "source_audio").resolve()
    speaker_samples_dir = (episode_dir / "speaker_samples").resolve()
    inputs_dir.mkdir(parents=True, exist_ok=True)
    source_audio_dir.mkdir(parents=True, exist_ok=True)
    speaker_samples_dir.mkdir(parents=True, exist_ok=True)

    copied_sampled_segments_path = (inputs_dir / sampled_file.path.name).resolve()
    shutil.copy2(sampled_file.path, copied_sampled_segments_path)

    transcript = Transcript.from_mapping(
        json.loads(Path(sample.transcript_path).read_text(encoding="utf-8"))
    )
    transcript_output_path = (inputs_dir / "prepared.transcript.json").resolve()
    speaker_manifest_path = (speaker_samples_dir / "manifest.json").resolve()
    if (
        transcript_output_path.exists()
        and speaker_manifest_path.exists()
        and not force_regenerate
    ):
        prepared_payload = json.loads(
            transcript_output_path.read_text(encoding="utf-8")
        )
        prepared_transcript = Transcript.from_mapping(prepared_payload)
        source_audio_value = str(
            prepared_transcript.metadata.get("source_audio_path") or ""
        ).strip()
        if not source_audio_value:
            raise SampledSegmentSpeakerPrepError(
                f"Prepared transcript is missing source_audio_path: {transcript_output_path}"
            )
        source_audio_path = Path(source_audio_value).expanduser().resolve()
        return PreparedEpisodeResult(
            manifest_episode_number=manifest_match.manifest_episode_number,
            episode_title=episode_title,
            sample_id=sample.sample_id,
            episode_dir=episode_dir,
            sampled_segments_path=copied_sampled_segments_path,
            transcript_path=transcript_output_path,
            source_audio_path=source_audio_path,
            speaker_samples_manifest_path=speaker_manifest_path,
        )

    updated_transcript, source_audio_path = _materialize_local_source_audio(
        transcript=transcript,
        sample=sample,
        source_audio_dir=source_audio_dir,
        force_download=force_download,
    )
    generator = build_speaker_sample_generator(audio_cfg)
    generation_result = generator.generate(
        transcript_payload=updated_transcript.to_payload(),
        source_audio_path=source_audio_path,
        output_dir=speaker_samples_dir,
        progress_callback=None,
    )
    metadata = dict(updated_transcript.metadata)
    metadata.update(
        {
            "sampled_segments_path": str(copied_sampled_segments_path),
            "speaker_samples_manifest_path": str(
                generation_result.manifest_path.resolve()
            ),
            "speaker_samples_dir": str(generation_result.output_dir.resolve()),
            "speaker_sample_count": len(generation_result.artifacts),
            "speaker_samples_generated_at_utc": generation_result.generated_at_utc,
        }
    )
    prepared_transcript = updated_transcript.with_metadata(metadata)
    write_transcript_atomic(prepared_transcript.to_payload(), transcript_output_path)
    return PreparedEpisodeResult(
        manifest_episode_number=manifest_match.manifest_episode_number,
        episode_title=episode_title,
        sample_id=sample.sample_id,
        episode_dir=episode_dir,
        sampled_segments_path=copied_sampled_segments_path,
        transcript_path=transcript_output_path,
        source_audio_path=source_audio_path,
        speaker_samples_manifest_path=generation_result.manifest_path.resolve(),
    )


def _materialize_local_source_audio(
    *,
    transcript: Transcript,
    sample: PreparedBenchmarkSample,
    source_audio_dir: Path,
    force_download: bool,
) -> tuple[Transcript, Path]:
    metadata = dict(transcript.metadata)
    source_audio_url = (
        str(metadata.get("source_audio_url") or "").strip()
        or str(sample.metadata.get("source_audio_url") or "").strip()
        or str(sample.episode_key).strip()
    )
    if not source_audio_url:
        raise SampledSegmentSpeakerPrepError(
            f"Prepared SPoRC sample is missing source audio metadata: {sample.sample_id}"
        )
    resolved_source_audio_path = _copy_or_download_source_audio(
        source_audio_url=source_audio_url,
        source_audio_dir=source_audio_dir,
        sample_id=sample.sample_id,
        existing_source_audio_path=str(metadata.get("source_audio_path") or "").strip(),
        force_download=force_download,
    )
    metadata["source_audio_url"] = source_audio_url
    metadata["source_audio_path"] = str(resolved_source_audio_path)
    return transcript.with_metadata(metadata), resolved_source_audio_path


def _copy_or_download_source_audio(
    *,
    source_audio_url: str,
    source_audio_dir: Path,
    sample_id: str,
    existing_source_audio_path: str,
    force_download: bool,
) -> Path:
    if existing_source_audio_path:
        existing_candidate = _resolve_existing_local_path(existing_source_audio_path)
        if existing_candidate is not None:
            destination = source_audio_dir / existing_candidate.name
            if destination.exists() and not force_download:
                return destination.resolve()
            shutil.copy2(existing_candidate, destination)
            return destination.resolve()

    parsed = urlparse(source_audio_url)
    suffix = Path(parsed.path).suffix or ".mp3"
    destination = (source_audio_dir / f"{sample_id}{suffix}").resolve()
    if parsed.scheme in {"http", "https"}:
        if destination.exists() and destination.stat().st_size > 0 and not force_download:
            return destination
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
            raise SampledSegmentSpeakerPrepError(
                "Failed to download source audio "
                f"for {sample_id!r} from {source_audio_url!r}: {exc}"
            ) from exc
        return destination

    candidate = _resolve_existing_local_path(source_audio_url)
    if candidate is None:
        raise SampledSegmentSpeakerPrepError(
            f"Source audio path does not exist for {sample_id}: {source_audio_url}"
        )
    if destination.exists() and not force_download:
        return destination.resolve()
    shutil.copy2(candidate, destination)
    return destination.resolve()


def _resolve_existing_local_path(raw_value: str) -> Path | None:
    candidate = Path(raw_value).expanduser()
    if not candidate.is_absolute():
        candidate = (REPO_ROOT / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.exists():
        return candidate
    return None


def _slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip()).strip("_").lower()
    return normalized or "episode"


if __name__ == "__main__":
    raise SystemExit(main())
