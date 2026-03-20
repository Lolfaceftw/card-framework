"""Build sampled stage-2 transcripts from sampler exports and run the pipeline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any
from uuid import uuid4

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from card_framework.agents.utils import count_words
from card_framework.audio_pipeline.io import write_transcript_atomic
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_LONGFORM_MANIFEST_PATH,
    PreparedBenchmarkSample,
)
from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    iter_prepared_dataset_samples,
    load_prepared_dataset_manifest,
)
from card_framework.orchestration.transcript import Transcript, TranscriptSegment
from card_framework.shared.paths import REPO_ROOT

_DEFAULT_ARTIFACTS_ROOT = (
    REPO_ROOT / "artifacts" / "sampled_transcript_pipeline"
).resolve()
_DEFAULT_PAGE_SIZE = 10
_DEFAULT_TITLE_WIDTH = 72
_DEFAULT_WORDS_PER_MINUTE = 170.0
_DEFAULT_MIN_SEGMENT_MS = 1_000
_DEFAULT_SEGMENT_GAP_MS = 250
_PATH_METADATA_KEYS = (
    "source_audio_path",
    "speaker_samples_manifest_path",
    "vocals_audio_path",
)
_EXPLICIT_RANGE_PATTERN = re.compile(r"^\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*$")


@dataclass(slots=True, frozen=True)
class SampledSegmentRow:
    """Represent one sampled segment export row."""

    sample_id: str
    episode_key: str
    episode_title: str
    selection_order: int
    speaker_label: str
    segment_text: str
    rationale: str
    actual_words: int
    estimated_words: int | None = None
    source_start_time: int | None = None
    source_end_time: int | None = None


@dataclass(slots=True, frozen=True)
class EpisodeCatalogEntry:
    """Describe one manifest episode that has sampled segments available."""

    manifest_episode_number: int
    sample: PreparedBenchmarkSample
    episode_title: str
    sampled_rows: tuple[SampledSegmentRow, ...]


@dataclass(slots=True, frozen=True)
class MaterializedEpisodeResult:
    """Record one prepared episode transcript and optional pipeline run."""

    manifest_episode_number: int
    sample_id: str
    episode_title: str
    transcript_path: Path
    output_root: Path
    sampled_segment_count: int
    reused_speaker_manifest_path: str | None
    source_audio_path: str | None
    pipeline_exit_code: int | None
    warnings: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the sampled-transcript helper."""
    parser = argparse.ArgumentParser(
        description=(
            "Select manifest episodes with existing sampled segments, materialize "
            "stage-2 transcript JSON artifacts from those sampled segments, and "
            "optionally pass them into the normal pipeline wrapper."
        )
    )
    parser.add_argument(
        "--dataset-manifest",
        default=str(DEFAULT_LONGFORM_MANIFEST_PATH),
        help="Prepared manifest JSON path. Defaults to the curated long-form manifest.",
    )
    sample_group = parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument(
        "--selected-samples-path",
        help="JSONL export path such as sporc_llm_sampling/<run>/selected_samples.jsonl.",
    )
    sample_group.add_argument(
        "--llm-sample-run-root",
        help="Sampler run directory that contains selected_samples.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        help=(
            "Output directory for the sampled transcripts and optional pipeline "
            "runs. Defaults to artifacts/sampled_transcript_pipeline/<run-id>."
        ),
    )
    parser.add_argument(
        "--select-episodes",
        default="",
        help=(
            "Optional non-interactive selection spec using manifest positions, "
            "for example 1,3-5 or [7,10]. When omitted, the interactive menu opens."
        ),
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=_DEFAULT_PAGE_SIZE,
        help="Interactive menu page size. Defaults to 10.",
    )
    parser.add_argument(
        "--title-width",
        type=int,
        default=_DEFAULT_TITLE_WIDTH,
        help="Maximum rendered episode-title width before truncation.",
    )
    parser.add_argument(
        "--words-per-minute",
        type=float,
        default=_DEFAULT_WORDS_PER_MINUTE,
        help="Synthetic transcript timing rate used for sampled segments.",
    )
    parser.add_argument(
        "--min-segment-ms",
        type=int,
        default=_DEFAULT_MIN_SEGMENT_MS,
        help="Minimum synthetic duration per sampled segment.",
    )
    parser.add_argument(
        "--segment-gap-ms",
        type=int,
        default=_DEFAULT_SEGMENT_GAP_MS,
        help="Gap inserted between synthetic sampled transcript segments.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write sampled transcript artifacts only and skip setup_and_run.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first child pipeline failure.",
    )
    parser.add_argument(
        "--setup-override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional setup_and_run override. Repeat for multiple values.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the sampled-transcript materialization helper."""
    args = build_parser().parse_args(argv)
    console = Console()
    selected_samples_path = _resolve_selected_samples_path(args)
    output_root = _resolve_output_root(args)
    output_root.mkdir(parents=True, exist_ok=True)

    sampled_rows = _load_sampled_rows(selected_samples_path)
    manifest_path = Path(args.dataset_manifest).resolve()
    catalog = _build_episode_catalog(
        manifest_path=manifest_path,
        sampled_rows=sampled_rows,
    )
    if not catalog:
        raise RuntimeError(
            "No manifest episodes matched the sampled-segment export. "
            "Check that the manifest and selected_samples JSONL come from the same dataset."
        )

    manifest_positions = {entry.manifest_episode_number for entry in catalog}
    if args.select_episodes:
        selected_positions = _parse_episode_selection_spec(
            args.select_episodes,
            valid_positions=manifest_positions,
        )
    else:
        selected_positions = _interactive_episode_selection(
            console=console,
            catalog=catalog,
            page_size=int(args.page_size),
            title_width=int(args.title_width),
        )

    selected_entries = [
        entry
        for entry in catalog
        if entry.manifest_episode_number in selected_positions
    ]
    _render_selection_banner(console=console, selected_entries=selected_entries)

    summary_records: list[dict[str, Any]] = []
    exit_code = 0
    for entry in selected_entries:
        materialized = _materialize_episode(
            entry=entry,
            selected_samples_path=selected_samples_path,
            manifest_path=manifest_path,
            output_root=output_root,
            words_per_minute=float(args.words_per_minute),
            min_segment_ms=int(args.min_segment_ms),
            segment_gap_ms=int(args.segment_gap_ms),
        )
        if not args.prepare_only:
            pipeline_exit_code = _run_setup_and_run(
                transcript_path=materialized.transcript_path,
                episode_output_root=materialized.output_root,
                setup_overrides=[
                    str(item) for item in args.setup_override if str(item).strip()
                ],
                console=console,
            )
            materialized = MaterializedEpisodeResult(
                manifest_episode_number=materialized.manifest_episode_number,
                sample_id=materialized.sample_id,
                episode_title=materialized.episode_title,
                transcript_path=materialized.transcript_path,
                output_root=materialized.output_root,
                sampled_segment_count=materialized.sampled_segment_count,
                reused_speaker_manifest_path=materialized.reused_speaker_manifest_path,
                source_audio_path=materialized.source_audio_path,
                pipeline_exit_code=pipeline_exit_code,
                warnings=materialized.warnings,
            )
            if pipeline_exit_code != 0:
                exit_code = 1
                if args.stop_on_error:
                    summary_records.append(_result_to_summary_record(materialized))
                    break
        summary_records.append(_result_to_summary_record(materialized))

    summary_path = (output_root / "run_summary.json").resolve()
    summary_payload = {
        "run_id": output_root.name,
        "created_at_utc": _utc_now_iso(),
        "manifest_path": str(manifest_path),
        "selected_samples_path": str(selected_samples_path),
        "prepare_only": bool(args.prepare_only),
        "selected_episode_positions": list(selected_positions),
        "episode_count": len(summary_records),
        "episodes": summary_records,
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _render_completion_banner(
        console=console,
        summary_path=summary_path,
        episode_count=len(summary_records),
        prepare_only=bool(args.prepare_only),
        exit_code=exit_code,
    )
    return exit_code


def _resolve_selected_samples_path(args: argparse.Namespace) -> Path:
    if args.selected_samples_path:
        return Path(str(args.selected_samples_path)).expanduser().resolve()
    run_root = Path(str(args.llm_sample_run_root)).expanduser().resolve()
    candidate = (run_root / "selected_samples.jsonl").resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"Sampler run root does not contain selected_samples.jsonl: {candidate}"
        )
    return candidate


def _resolve_output_root(args: argparse.Namespace) -> Path:
    if args.output_root:
        return Path(str(args.output_root)).expanduser().resolve()
    run_id = (
        datetime.now(timezone.utc).strftime("sampled_pipeline_%Y%m%dT%H%M%SZ_")
        + uuid4().hex[:8]
    )
    return (_DEFAULT_ARTIFACTS_ROOT / run_id).resolve()


def _load_sampled_rows(selected_samples_path: Path) -> tuple[SampledSegmentRow, ...]:
    rows: list[SampledSegmentRow] = []
    with selected_samples_path.open("r", encoding="utf-8") as handle:
        for index, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                continue
            rows.append(_parse_sampled_row(payload, default_order=index))
    if not rows:
        raise RuntimeError(
            f"Sampled-segment export did not contain any usable rows: {selected_samples_path}"
        )
    return tuple(rows)


def _parse_sampled_row(
    payload: dict[str, Any],
    *,
    default_order: int,
) -> SampledSegmentRow:
    sample_id = str(payload.get("sample_id") or "").strip()
    episode_key = str(payload.get("episode_key") or "").strip()
    if not sample_id and not episode_key:
        raise RuntimeError(
            "Each sampled-segment row must include sample_id or episode_key."
        )
    segment_text = str(
        payload.get("segment_text") or payload.get("text") or ""
    ).strip()
    if not segment_text:
        raise RuntimeError("Sampled-segment rows require non-empty segment text.")
    speaker_label = str(
        payload.get("speaker_label") or payload.get("speaker") or "UNKNOWN"
    ).strip() or "UNKNOWN"
    actual_words = int(payload.get("actual_words") or count_words(segment_text))
    estimated_words_raw = payload.get("estimated_words")
    estimated_words = (
        int(estimated_words_raw) if estimated_words_raw is not None else None
    )
    selection_order_raw = payload.get("selection_order")
    selection_order = (
        int(selection_order_raw) if selection_order_raw is not None else default_order
    )
    return SampledSegmentRow(
        sample_id=sample_id,
        episode_key=episode_key,
        episode_title=str(payload.get("episode_title") or "").strip(),
        selection_order=selection_order,
        speaker_label=speaker_label,
        segment_text=segment_text,
        rationale=str(payload.get("rationale") or "").strip(),
        actual_words=actual_words,
        estimated_words=estimated_words,
        source_start_time=_optional_int(payload.get("start_time")),
        source_end_time=_optional_int(payload.get("end_time")),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return int(round(float(stripped)))
        except ValueError:
            return None
    return None


def _build_episode_catalog(
    *,
    manifest_path: Path,
    sampled_rows: tuple[SampledSegmentRow, ...],
) -> tuple[EpisodeCatalogEntry, ...]:
    load_prepared_dataset_manifest(manifest_path, include_samples=False)
    rows_by_sample_id: dict[str, list[SampledSegmentRow]] = {}
    rows_by_episode_key: dict[str, list[SampledSegmentRow]] = {}
    for row in sampled_rows:
        if row.sample_id:
            rows_by_sample_id.setdefault(row.sample_id, []).append(row)
        if row.episode_key:
            rows_by_episode_key.setdefault(row.episode_key, []).append(row)

    catalog: list[EpisodeCatalogEntry] = []
    for manifest_episode_number, sample in enumerate(
        iter_prepared_dataset_samples(manifest_path),
        start=1,
    ):
        matching_rows = rows_by_sample_id.get(sample.sample_id)
        if not matching_rows:
            matching_rows = rows_by_episode_key.get(sample.episode_key, [])
        if not matching_rows:
            continue
        ordered_rows = tuple(
            sorted(
                matching_rows,
                key=lambda item: (
                    item.selection_order,
                    item.actual_words,
                    item.segment_text,
                ),
            )
        )
        episode_title = (
            str(sample.metadata.get("episode_title") or "").strip()
            or ordered_rows[0].episode_title
            or sample.sample_id
        )
        catalog.append(
            EpisodeCatalogEntry(
                manifest_episode_number=manifest_episode_number,
                sample=sample,
                episode_title=episode_title,
                sampled_rows=ordered_rows,
            )
        )
    return tuple(catalog)


def _interactive_episode_selection(
    *,
    console: Console,
    catalog: tuple[EpisodeCatalogEntry, ...],
    page_size: int,
    title_width: int,
) -> tuple[int, ...]:
    if page_size <= 0:
        raise ValueError("page_size must be greater than zero.")
    if title_width <= 10:
        raise ValueError("title_width must be greater than ten.")

    selected_positions: set[int] = set()
    page_index = 0
    total_pages = max(1, (len(catalog) + page_size - 1) // page_size)
    valid_positions = {entry.manifest_episode_number for entry in catalog}

    while True:
        _render_episode_page(
            console=console,
            catalog=catalog,
            page_index=page_index,
            page_size=page_size,
            title_width=title_width,
            selected_positions=selected_positions,
        )
        raw_selection = Prompt.ask(
            "Enter episode numbers, `n`, `p`, `all`, `clear`, or `done`",
            console=console,
        ).strip()
        normalized = raw_selection.lower()
        if normalized in {"n", "next"}:
            page_index = min(page_index + 1, total_pages - 1)
            continue
        if normalized in {"p", "prev", "previous"}:
            page_index = max(page_index - 1, 0)
            continue
        if normalized == "all":
            selected_positions = set(valid_positions)
            continue
        if normalized == "clear":
            selected_positions.clear()
            continue
        if normalized in {"done", "d"}:
            if not selected_positions:
                console.print(
                    Panel.fit(
                        "Select at least one episode before finishing.",
                        title="Selection Required",
                        border_style="yellow",
                    )
                )
                continue
            return tuple(sorted(selected_positions))

        selected_batch = _parse_episode_selection_spec(
            raw_selection,
            valid_positions=valid_positions,
        )
        for position in selected_batch:
            if position in selected_positions:
                selected_positions.remove(position)
            else:
                selected_positions.add(position)


def _parse_episode_selection_spec(
    raw_value: str,
    *,
    valid_positions: set[int],
) -> tuple[int, ...]:
    normalized = str(raw_value or "").strip()
    if not normalized:
        raise RuntimeError("Episode selection cannot be empty.")

    explicit_range = _EXPLICIT_RANGE_PATTERN.fullmatch(normalized)
    if explicit_range is not None:
        start = int(explicit_range.group(1))
        end = int(explicit_range.group(2))
        return _validate_episode_positions(range(start, end + 1), valid_positions)

    selected_positions: list[int] = []
    for raw_token in normalized.split(","):
        token = raw_token.strip()
        if not token:
            continue
        if "-" in token:
            parts = [part.strip() for part in token.split("-", maxsplit=1)]
            if len(parts) != 2 or not parts[0].isdigit() or not parts[1].isdigit():
                raise RuntimeError(
                    "Episode ranges must look like `7-10` or `[7,10]`."
                )
            start = int(parts[0])
            end = int(parts[1])
            if end < start:
                raise RuntimeError(
                    "Episode-range end must be greater than or equal to the start."
                )
            selected_positions.extend(range(start, end + 1))
            continue
        if not token.isdigit():
            raise RuntimeError(
                "Episode selections must be positive integers or ranges."
            )
        selected_positions.append(int(token))

    if not selected_positions:
        raise RuntimeError("Episode selection did not resolve any manifest positions.")
    return _validate_episode_positions(selected_positions, valid_positions)


def _validate_episode_positions(
    positions: Any,
    valid_positions: set[int],
) -> tuple[int, ...]:
    validated = sorted({int(position) for position in positions})
    invalid = [position for position in validated if position not in valid_positions]
    if invalid:
        available_hint = ", ".join(str(item) for item in sorted(valid_positions)[:20])
        raise RuntimeError(
            "Episode selection includes manifest positions without sampled segments: "
            f"{invalid}. Available sampled positions start with: {available_hint}"
        )
    return tuple(validated)


def _render_episode_page(
    *,
    console: Console,
    catalog: tuple[EpisodeCatalogEntry, ...],
    page_index: int,
    page_size: int,
    title_width: int,
    selected_positions: set[int],
) -> None:
    total_pages = max(1, (len(catalog) + page_size - 1) // page_size)
    page_start = page_index * page_size
    page_entries = catalog[page_start : page_start + page_size]

    table = Table(
        title=f"Sampled Episodes Page {page_index + 1}/{total_pages}",
        header_style="bold magenta",
    )
    table.add_column("Manifest #", justify="right", style="cyan", no_wrap=True)
    table.add_column("Selected", justify="center", no_wrap=True)
    table.add_column("Title", overflow="fold")
    table.add_column("Samples", justify="right", no_wrap=True)
    for entry in page_entries:
        table.add_row(
            str(entry.manifest_episode_number),
            "x" if entry.manifest_episode_number in selected_positions else "",
            _truncate_title(entry.episode_title, width=title_width),
            str(len(entry.sampled_rows)),
        )
    console.print(table)
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Available sampled episodes: {len(catalog)}",
                    (
                        "Selected manifest positions: "
                        + (
                            ", ".join(str(item) for item in sorted(selected_positions))
                            if selected_positions
                            else "(none)"
                        )
                    ),
                ]
            ),
            title="Selection State",
            border_style="cyan",
        )
    )


def _truncate_title(title: str, *, width: int) -> str:
    cleaned = " ".join(str(title or "").split())
    if len(cleaned) <= width:
        return cleaned
    if width <= 3:
        return cleaned[:width]
    return cleaned[: width - 3].rstrip() + "..."


def _render_selection_banner(
    *,
    console: Console,
    selected_entries: list[EpisodeCatalogEntry],
) -> None:
    lines = [
        f"{entry.manifest_episode_number}: {entry.episode_title}"
        for entry in selected_entries
    ]
    console.print(
        Panel.fit(
            "\n".join(lines),
            title="Selected Episodes",
            border_style="green",
        )
    )


def _materialize_episode(
    *,
    entry: EpisodeCatalogEntry,
    selected_samples_path: Path,
    manifest_path: Path,
    output_root: Path,
    words_per_minute: float,
    min_segment_ms: int,
    segment_gap_ms: int,
) -> MaterializedEpisodeResult:
    source_transcript_path = Path(entry.sample.transcript_path).resolve()
    source_payload = json.loads(source_transcript_path.read_text(encoding="utf-8-sig"))
    source_transcript = Transcript.from_mapping(source_payload)
    normalized_metadata = _normalize_path_metadata(
        dict(source_transcript.metadata),
        source_transcript_path=source_transcript_path,
    )
    episode_root = (
        output_root
        / "episodes"
        / f"{entry.manifest_episode_number:05d}_{_slugify(entry.sample.sample_id)}"
    ).resolve()
    inputs_root = (episode_root / "inputs").resolve()
    transcript_path = (inputs_root / "sampled.transcript.json").resolve()

    synthetic_segments = _build_synthetic_segments(
        entry.sampled_rows,
        words_per_minute=words_per_minute,
        min_segment_ms=min_segment_ms,
        segment_gap_ms=segment_gap_ms,
    )
    metadata = dict(normalized_metadata)
    metadata.update(
        {
            "episode_title": entry.episode_title,
            "sample_id": entry.sample.sample_id,
            "episode_key": entry.sample.episode_key,
            "sampled_transcript_created_at_utc": _utc_now_iso(),
            "sampled_transcript_selected_samples_path": str(selected_samples_path),
            "sampled_transcript_source_manifest_path": str(manifest_path),
            "sampled_transcript_source_transcript_path": str(source_transcript_path),
            "sampled_transcript_manifest_episode_number": entry.manifest_episode_number,
            "sampled_transcript_segment_count": len(synthetic_segments),
            "sampled_transcript_word_rate_wpm": words_per_minute,
            "sampled_transcript_timing_mode": "synthetic_sequential",
            "sampled_transcript_speakers": sorted(
                {segment.speaker for segment in synthetic_segments}
            ),
        }
    )
    transcript = Transcript(
        segments=tuple(synthetic_segments),
        metadata=metadata,
        extras=dict(source_transcript.extras),
    )
    write_transcript_atomic(transcript.to_payload(), transcript_path)

    warnings = _build_episode_warnings(metadata)
    return MaterializedEpisodeResult(
        manifest_episode_number=entry.manifest_episode_number,
        sample_id=entry.sample.sample_id,
        episode_title=entry.episode_title,
        transcript_path=transcript_path,
        output_root=episode_root,
        sampled_segment_count=len(synthetic_segments),
        reused_speaker_manifest_path=(
            str(metadata["speaker_samples_manifest_path"])
            if str(metadata.get("speaker_samples_manifest_path") or "").strip()
            else None
        ),
        source_audio_path=(
            str(metadata["source_audio_path"])
            if str(metadata.get("source_audio_path") or "").strip()
            else None
        ),
        pipeline_exit_code=None,
        warnings=tuple(warnings),
    )


def _normalize_path_metadata(
    metadata: dict[str, Any],
    *,
    source_transcript_path: Path,
) -> dict[str, Any]:
    normalized = dict(metadata)
    for key in _PATH_METADATA_KEYS:
        raw_value = str(normalized.get(key) or "").strip()
        if not raw_value:
            continue
        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = (REPO_ROOT / candidate).resolve()
        else:
            candidate = candidate.resolve()
        if candidate.exists():
            normalized[key] = str(candidate)
            continue
        source_candidate = (source_transcript_path.parent / raw_value).resolve()
        if source_candidate.exists():
            normalized[key] = str(source_candidate)
    return normalized


def _build_synthetic_segments(
    sampled_rows: tuple[SampledSegmentRow, ...],
    *,
    words_per_minute: float,
    min_segment_ms: int,
    segment_gap_ms: int,
) -> list[TranscriptSegment]:
    if words_per_minute <= 0:
        raise ValueError("words_per_minute must be greater than zero.")
    if min_segment_ms <= 0:
        raise ValueError("min_segment_ms must be greater than zero.")
    if segment_gap_ms < 0:
        raise ValueError("segment_gap_ms cannot be negative.")

    current_time_ms = 0
    segments: list[TranscriptSegment] = []
    for row in sampled_rows:
        actual_words = max(1, row.actual_words)
        duration_ms = max(
            min_segment_ms,
            int(round((actual_words / words_per_minute) * 60_000)),
        )
        extras: dict[str, Any] = {
            "selection_order": row.selection_order,
            "actual_words": row.actual_words,
        }
        if row.estimated_words is not None:
            extras["estimated_words"] = row.estimated_words
        if row.rationale:
            extras["selection_rationale"] = row.rationale
        if row.source_start_time is not None:
            extras["source_start_time"] = row.source_start_time
        if row.source_end_time is not None:
            extras["source_end_time"] = row.source_end_time
        segments.append(
            TranscriptSegment(
                speaker=row.speaker_label,
                text=row.segment_text,
                start_time=current_time_ms,
                end_time=current_time_ms + duration_ms,
                extras=extras,
            )
        )
        current_time_ms += duration_ms + segment_gap_ms
    return segments


def _build_episode_warnings(metadata: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    speaker_manifest_path = str(metadata.get("speaker_samples_manifest_path") or "").strip()
    source_audio_path = str(metadata.get("source_audio_path") or "").strip()
    if not speaker_manifest_path:
        warnings.append(
            "No reusable speaker_samples_manifest_path was found in the source "
            "transcript metadata. If downstream voice cloning or interjector "
            "stages stay enabled, the runtime may need speaker-sample bootstrap work."
        )
    if not source_audio_path:
        warnings.append(
            "No local source_audio_path was found in the source transcript metadata. "
            "If speaker-sample bootstrap is required, setup_and_run may prompt for audio "
            "or fail fast depending on the active config."
        )
    return warnings


def _run_setup_and_run(
    *,
    transcript_path: Path,
    episode_output_root: Path,
    setup_overrides: list[str],
    console: Console,
) -> int:
    command = [
        sys.executable,
        "-m",
        "card_framework.cli.setup_and_run",
        "--output-root",
        str(episode_output_root),
        "--override",
        "pipeline.start_stage=stage-2",
        "--override",
        f"transcript_path={transcript_path.as_posix()}",
    ]
    for override in setup_overrides:
        command.extend(["--override", override])
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Episode output root: {episode_output_root}",
                    f"Transcript override: {transcript_path}",
                ]
            ),
            title="Running setup_and_run",
            border_style="cyan",
        )
    )
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        check=False,
        text=True,
    )
    return int(completed.returncode)


def _result_to_summary_record(result: MaterializedEpisodeResult) -> dict[str, Any]:
    return {
        "manifest_episode_number": result.manifest_episode_number,
        "sample_id": result.sample_id,
        "episode_title": result.episode_title,
        "transcript_path": str(result.transcript_path),
        "output_root": str(result.output_root),
        "sampled_segment_count": result.sampled_segment_count,
        "reused_speaker_manifest_path": result.reused_speaker_manifest_path,
        "source_audio_path": result.source_audio_path,
        "pipeline_exit_code": result.pipeline_exit_code,
        "warnings": list(result.warnings),
    }


def _render_completion_banner(
    *,
    console: Console,
    summary_path: Path,
    episode_count: int,
    prepare_only: bool,
    exit_code: int,
) -> None:
    title = (
        "Sampled Pipeline Ready"
        if exit_code == 0
        else "Sampled Pipeline Finished With Failures"
    )
    border_style = "green" if exit_code == 0 else "yellow"
    console.print(
        Panel.fit(
            "\n".join(
                [
                    f"Episodes processed: {episode_count}",
                    f"Mode: {'prepare-only' if prepare_only else 'prepare-and-run'}",
                    f"Summary: {summary_path}",
                ]
            ),
            title=title,
            border_style=border_style,
        )
    )


def _slugify(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "")).strip("_")
    return collapsed[:80] or "episode"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
