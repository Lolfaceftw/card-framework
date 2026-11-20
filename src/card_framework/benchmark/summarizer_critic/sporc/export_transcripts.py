"""Export combined raw transcript text for selected SPoRC manifest episodes."""

from __future__ import annotations

import argparse
import gzip
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_LONGFORM_MANIFEST_PATH,
)
from card_framework.shared.perf import StatusReporter, report_loading_duration

_BRACKET_RANGE_PATTERN = re.compile(r"^\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*$")
_RANGE_TOKEN_PATTERN = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_EPISODE_KEY_FIELDS = (
    "mp3url",
    "mp3Url",
    "mp3_url",
    "episodeMp3Url",
    "episode_mp3_url",
    "fullPotentialOutPath",
    "potentialOutPath",
    "episodeUrl",
    "episode_url",
)
_TURN_TEXT_FIELDS = ("turnText", "content", "text", "utterance", "transcript")
_TURN_SPEAKER_FIELDS = ("speakers", "speaker", "speakerLabel", "speaker_label")
_TURN_STREAM_PROGRESS_INTERVAL = 500_000


@dataclass(slots=True, frozen=True)
class TranscriptExportResult:
    """Summarize one transcript export run."""

    manifest_path: Path
    output_path: Path
    episode_positions: tuple[int, ...]
    episode_count: int
    line_count: int


def export_manifest_transcripts(
    *,
    manifest_path: Path = DEFAULT_LONGFORM_MANIFEST_PATH,
    episode_spec: str,
    output_path: Path | None = None,
    backwards_compat: bool = False,
    raw: bool = False,
    status_reporter: StatusReporter | None = None,
) -> TranscriptExportResult:
    """Export selected manifest episodes into one combined raw transcript text file."""
    resolved_manifest_path = manifest_path.resolve()
    episode_positions = _parse_episode_spec(episode_spec)
    if not episode_positions:
        raise RuntimeError("At least one manifest episode position is required.")

    resolved_output_path = (
        output_path.resolve()
        if output_path is not None
        else resolved_manifest_path.with_name(
            f"{resolved_manifest_path.stem}.episodes_{'_'.join(str(value) for value in episode_positions)}"
            f"{'.jsonl' if raw else '.txt'}"
        )
    )

    with report_loading_duration(
        "SPoRC manifest for transcript export",
        reporter=status_reporter,
    ):
        manifest_payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, dict):
        raise RuntimeError("SPoRC transcript export requires a JSON object manifest.")

    samples = _select_manifest_samples(
        manifest_payload,
        manifest_path=resolved_manifest_path,
        episode_positions=episode_positions,
        backwards_compat=backwards_compat,
    )
    if raw:
        with report_loading_duration(
            "selected raw SPoRC transcript rows for export",
            reporter=status_reporter,
        ):
            line_count = _write_selected_raw_turn_rows(
                manifest_payload,
                manifest_path=resolved_manifest_path,
                samples=samples,
                output_path=resolved_output_path,
                status_reporter=status_reporter,
            )
    else:
        with report_loading_duration(
            "selected raw SPoRC turns for transcript export",
            reporter=status_reporter,
        ):
            selected_turns = _load_selected_turns(
                manifest_payload,
                manifest_path=resolved_manifest_path,
                samples=samples,
                status_reporter=status_reporter,
            )

        line_count = 0
        with report_loading_duration(
            "selected SPoRC transcript export",
            reporter=status_reporter,
        ):
            resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
            with resolved_output_path.open("w", encoding="utf-8") as handle:
                for manifest_episode_number, sample in samples:
                    episode_title = _sample_episode_title(sample)
                    handle.write(f"===== Episode {manifest_episode_number} =====\n")
                    handle.write(f"Sample ID: {sample['sample_id']}\n")
                    handle.write(f"Episode Title: {episode_title}\n")
                    handle.write(f"Episode Key: {sample['episode_key']}\n")
                    handle.write("\n")
                    episode_turns = selected_turns.get(
                        str(sample.get("episode_key") or ""),
                        [],
                    )
                    if not episode_turns:
                        raise RuntimeError(
                            "No raw dataset turns were found for manifest episode "
                            f"{manifest_episode_number}: {sample.get('episode_key')}"
                        )
                    for turn in episode_turns:
                        handle.write(
                            (
                                f"[{_format_timestamp(turn['start_time_ms'])}-"
                                f"{_format_timestamp(turn['end_time_ms'])}] "
                                f"[{turn['speaker_label']}] {turn['text']}\n"
                            )
                        )
                        line_count += 1
                    handle.write("\n")

    return TranscriptExportResult(
        manifest_path=resolved_manifest_path,
        output_path=resolved_output_path,
        episode_positions=episode_positions,
        episode_count=len(samples),
        line_count=line_count,
    )


def run_export_transcripts_cli(
    args: argparse.Namespace,
    *,
    status_reporter: StatusReporter | None = None,
) -> TranscriptExportResult:
    """Run transcript export from parsed CLI arguments."""
    return export_manifest_transcripts(
        manifest_path=Path(str(args.manifest_path or DEFAULT_LONGFORM_MANIFEST_PATH)),
        episode_spec=str(args.episodes or ""),
        output_path=Path(str(args.output_path)).resolve()
        if str(args.output_path or "").strip()
        else None,
        backwards_compat=bool(getattr(args, "backwards_compat", False)),
        raw=bool(getattr(args, "raw", False)),
        status_reporter=status_reporter,
    )


def _parse_episode_spec(raw_value: str) -> tuple[int, ...]:
    """Parse a mixed list of manifest positions and ranges."""
    normalized = str(raw_value or "").strip()
    if not normalized:
        return ()

    bracket_match = _BRACKET_RANGE_PATTERN.fullmatch(normalized)
    if bracket_match is not None:
        start_value = int(bracket_match.group(1))
        end_value = int(bracket_match.group(2))
        if end_value < start_value:
            raise RuntimeError("Episode range end must be >= start.")
        return tuple(range(start_value, end_value + 1))

    positions: list[int] = []
    for token in normalized.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        range_match = _RANGE_TOKEN_PATTERN.fullmatch(stripped)
        if range_match is not None:
            start_value = int(range_match.group(1))
            end_value = int(range_match.group(2))
            if end_value < start_value:
                raise RuntimeError("Episode range end must be >= start.")
            positions.extend(range(start_value, end_value + 1))
            continue
        positions.append(int(stripped))

    deduped_positions: list[int] = []
    seen: set[int] = set()
    for position in positions:
        if position <= 0:
            raise RuntimeError("Episode positions must be greater than zero.")
        if position in seen:
            continue
        seen.add(position)
        deduped_positions.append(position)
    return tuple(deduped_positions)


def _select_manifest_samples(
    manifest_payload: dict[str, Any],
    *,
    manifest_path: Path,
    episode_positions: tuple[int, ...],
    backwards_compat: bool = False,
) -> list[tuple[int, dict[str, Any]]]:
    """Return selected raw sample rows in the requested selection order."""
    if backwards_compat:
        return _select_manifest_samples_backwards_compat(
            manifest_payload,
            manifest_path=manifest_path,
            episode_positions=episode_positions,
        )

    raw_sample_count = int(manifest_payload.get("sample_count") or 0)
    max_position = max(episode_positions)
    if raw_sample_count > 0 and max_position > raw_sample_count:
        raise RuntimeError(
            "Requested manifest episode exceeds the available sample count: "
            f"{max_position} > {raw_sample_count}."
        )

    raw_samples = manifest_payload.get("samples")
    if isinstance(raw_samples, list):
        selected: list[tuple[int, dict[str, Any]]] = []
        for position in episode_positions:
            try:
                raw_sample = raw_samples[position - 1]
            except IndexError as exc:
                raise RuntimeError(
                    "Requested manifest episode exceeds the available inline sample list: "
                    f"{position}."
                ) from exc
            if not isinstance(raw_sample, dict):
                raise RuntimeError(
                    f"Manifest sample at position {position} must be an object."
                )
            selected.append((position, raw_sample))
        return selected

    samples_path = _resolve_samples_path(manifest_payload, manifest_path=manifest_path)
    if samples_path is None or not samples_path.exists():
        raise RuntimeError(
            "Manifest must contain inline `samples` or a valid `samples_path` sidecar."
        )

    wanted_positions = set(episode_positions)
    selected_by_position: dict[int, dict[str, Any]] = {}
    open_fn = gzip.open if samples_path.suffix == ".gz" else open
    with open_fn(samples_path, "rt", encoding="utf-8") as handle:
        for position, raw_line in enumerate(handle, start=1):
            if position > max_position:
                break
            if position not in wanted_positions:
                continue
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload_line = json.loads(stripped)
            if not isinstance(payload_line, dict):
                raise RuntimeError(
                    f"Manifest sidecar sample at position {position} must be an object."
                )
            selected_by_position[position] = payload_line

    missing_positions = [value for value in episode_positions if value not in selected_by_position]
    if missing_positions:
        raise RuntimeError(
            "Unable to load manifest episodes from the sidecar: "
            + ", ".join(str(value) for value in missing_positions)
        )
    return [(position, selected_by_position[position]) for position in episode_positions]


def _select_manifest_samples_backwards_compat(
    manifest_payload: dict[str, Any],
    *,
    manifest_path: Path,
    episode_positions: tuple[int, ...],
) -> list[tuple[int, dict[str, Any]]]:
    """Map curated longform manifest positions onto the target manifest rows."""
    compatibility_manifest_path, compatibility_manifest_payload = (
        _resolve_backwards_compat_manifest(
            target_manifest_payload=manifest_payload,
            target_manifest_path=manifest_path,
        )
    )
    compatibility_samples = _select_manifest_samples(
        compatibility_manifest_payload,
        manifest_path=compatibility_manifest_path,
        episode_positions=episode_positions,
        backwards_compat=False,
    )
    if compatibility_manifest_path == manifest_path:
        return compatibility_samples

    episode_keys = tuple(
        _require_sample_episode_key(sample, selection_number=selection_number)
        for selection_number, sample in compatibility_samples
    )
    return _select_manifest_samples_by_episode_keys(
        manifest_payload,
        manifest_path=manifest_path,
        selection_numbers=episode_positions,
        episode_keys=episode_keys,
    )


def _resolve_samples_path(
    manifest_payload: dict[str, Any],
    *,
    manifest_path: Path,
) -> Path | None:
    """Resolve an optional manifest `samples_path` sidecar."""
    raw_samples_path = manifest_payload.get("samples_path")
    if not raw_samples_path:
        return None
    candidate = Path(str(raw_samples_path))
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_path.parent / candidate).resolve()


def _resolve_backwards_compat_manifest(
    *,
    target_manifest_payload: dict[str, Any],
    target_manifest_path: Path,
) -> tuple[Path, dict[str, Any]]:
    """Return the manifest that defines backwards-compatible export positions."""
    raw_samples = target_manifest_payload.get("samples")
    if isinstance(raw_samples, list):
        return target_manifest_path, target_manifest_payload

    compatibility_manifest_path = DEFAULT_LONGFORM_MANIFEST_PATH.resolve()
    compatibility_payload = json.loads(
        compatibility_manifest_path.read_text(encoding="utf-8")
    )
    if not isinstance(compatibility_payload, dict):
        raise RuntimeError(
            "Backwards-compat transcript export requires the compatibility "
            "manifest root to be a JSON object."
        )
    compatibility_samples = compatibility_payload.get("samples")
    if not isinstance(compatibility_samples, list):
        raise RuntimeError(
            "Backwards-compat transcript export requires an inline-sample "
            "compatibility manifest."
        )
    return compatibility_manifest_path, compatibility_payload


def _require_sample_episode_key(
    sample: dict[str, Any],
    *,
    selection_number: int,
) -> str:
    """Return the stable episode key required for compatibility mapping."""
    episode_key = str(sample.get("episode_key") or "").strip()
    if episode_key:
        return episode_key
    raise RuntimeError(
        "Selected compatibility-manifest sample is missing an `episode_key`: "
        f"{selection_number}."
    )


def _select_manifest_samples_by_episode_keys(
    manifest_payload: dict[str, Any],
    *,
    manifest_path: Path,
    selection_numbers: tuple[int, ...],
    episode_keys: tuple[str, ...],
) -> list[tuple[int, dict[str, Any]]]:
    """Return target-manifest samples keyed by stable SPoRC episode key."""
    raw_samples = manifest_payload.get("samples")
    if isinstance(raw_samples, list):
        selected_by_episode_key: dict[str, dict[str, Any]] = {}
        for raw_sample in raw_samples:
            if not isinstance(raw_sample, dict):
                continue
            episode_key = str(raw_sample.get("episode_key") or "").strip()
            if not episode_key or episode_key not in episode_keys:
                continue
            if episode_key in selected_by_episode_key:
                raise RuntimeError(
                    "Manifest contained duplicate episode keys in backwards-compat "
                    f"mode: {episode_key}."
                )
            selected_by_episode_key[episode_key] = raw_sample
        missing_episode_keys = [
            episode_key
            for episode_key in episode_keys
            if episode_key not in selected_by_episode_key
        ]
        if missing_episode_keys:
            raise RuntimeError(
                "Unable to resolve backwards-compatible episode selections from "
                "the inline manifest."
            )
        return [
            (selection_number, selected_by_episode_key[episode_key])
            for selection_number, episode_key in zip(
                selection_numbers,
                episode_keys,
                strict=True,
            )
        ]

    samples_path = _resolve_samples_path(manifest_payload, manifest_path=manifest_path)
    if samples_path is None or not samples_path.exists():
        raise RuntimeError(
            "Manifest must contain inline `samples` or a valid `samples_path` sidecar."
        )

    wanted_episode_keys = set(episode_keys)
    selected_by_episode_key: dict[str, dict[str, Any]] = {}
    open_fn = gzip.open if samples_path.suffix == ".gz" else open
    with open_fn(samples_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload_line = json.loads(stripped)
            if not isinstance(payload_line, dict):
                raise RuntimeError(
                    "Manifest sidecar sample in backwards-compat mode must be an object."
                )
            episode_key = str(payload_line.get("episode_key") or "").strip()
            if episode_key not in wanted_episode_keys:
                continue
            if episode_key in selected_by_episode_key:
                raise RuntimeError(
                    "Manifest sidecar contained duplicate episode keys in "
                    f"backwards-compat mode: {episode_key}."
                )
            selected_by_episode_key[episode_key] = payload_line
            if len(selected_by_episode_key) == len(wanted_episode_keys):
                break

    missing_episode_keys = [
        episode_key
        for episode_key in episode_keys
        if episode_key not in selected_by_episode_key
    ]
    if missing_episode_keys:
        raise RuntimeError(
            "Unable to resolve backwards-compatible episode selections from "
            "the sidecar manifest."
        )
    return [
        (selection_number, selected_by_episode_key[episode_key])
        for selection_number, episode_key in zip(
            selection_numbers,
            episode_keys,
            strict=True,
        )
    ]


def _resolve_turn_data_path(
    *,
    manifest_path: Path,
    manifest_payload: dict[str, Any],
) -> Path:
    """Resolve the manifest's raw SPoRC turn-data path."""
    raw_turn_data_path = manifest_payload.get("turn_data_path")
    if not raw_turn_data_path:
        raise RuntimeError("Manifest must define `turn_data_path` for raw transcript export.")
    candidate = Path(str(raw_turn_data_path))
    if candidate.is_absolute():
        return candidate.resolve()
    return (manifest_path.parent / candidate).resolve()


def _sample_episode_title(sample: dict[str, Any]) -> str:
    """Return the best available episode title for one raw sample row."""
    metadata = sample.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, dict) else {}
    return str(metadata_mapping.get("episode_title") or sample.get("sample_id") or "").strip()


def _load_selected_turns(
    manifest_payload: dict[str, Any],
    *,
    manifest_path: Path,
    samples: list[tuple[int, dict[str, Any]]],
    status_reporter: StatusReporter | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Load raw dataset turns for the selected manifest samples in one streaming pass."""
    turn_data_path = _resolve_turn_data_path(
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
    )
    if not turn_data_path.exists():
        raise RuntimeError(f"Raw SPoRC turn data file was not found: {turn_data_path}")

    selected_episode_keys = {
        str(sample.get("episode_key") or "")
        for _, sample in samples
    }
    selected_turns: dict[str, list[dict[str, Any]]] = {
        episode_key: [] for episode_key in selected_episode_keys
    }
    remaining_episode_keys = set(selected_episode_keys)
    current_episode_key = ""
    current_turns: list[dict[str, Any]] | None = None
    streamed_row_count = 0

    open_fn = gzip.open if turn_data_path.suffix == ".gz" else open
    with open_fn(turn_data_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            streamed_row_count += 1
            if (
                status_reporter is not None
                and streamed_row_count % _TURN_STREAM_PROGRESS_INTERVAL == 0
            ):
                status_reporter(
                    "Loading selected raw SPoRC turns... streamed "
                    f"{streamed_row_count:,} rows so far."
                )
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                continue
            episode_key = _turn_episode_key(payload)
            if not episode_key:
                continue

            if not current_episode_key:
                current_episode_key = episode_key
                current_turns = [] if episode_key in selected_episode_keys else None
            elif episode_key != current_episode_key:
                if current_turns is not None:
                    current_turns.sort(
                        key=lambda item: (
                            item["start_time_ms"],
                            item["end_time_ms"],
                        )
                    )
                    selected_turns[current_episode_key] = current_turns
                    remaining_episode_keys.discard(current_episode_key)
                    if not remaining_episode_keys:
                        break
                current_episode_key = episode_key
                current_turns = [] if episode_key in selected_episode_keys else None

            if current_turns is None:
                continue

            text = _turn_text(payload)
            speakers = _turn_speakers(payload)
            if not text or not speakers:
                continue
            current_turns.append(
                {
                    "start_time_ms": _seconds_to_milliseconds(
                        _turn_float(payload, "start", "startTime", "start_time")
                    ),
                    "end_time_ms": _seconds_to_milliseconds(
                        _turn_float(payload, "end", "endTime", "end_time")
                    ),
                    "speaker_label": ",".join(speakers),
                    "text": text,
                }
            )

    if current_turns is not None and current_episode_key in remaining_episode_keys:
        current_turns.sort(
            key=lambda item: (item["start_time_ms"], item["end_time_ms"])
        )
        selected_turns[current_episode_key] = current_turns
        remaining_episode_keys.discard(current_episode_key)
    if remaining_episode_keys:
        missing_episode_keys = sorted(remaining_episode_keys)
        raise RuntimeError(
            "No raw dataset turns were found for selected manifest episodes: "
            + ", ".join(missing_episode_keys)
        )
    return selected_turns


def _write_selected_raw_turn_rows(
    manifest_payload: dict[str, Any],
    *,
    manifest_path: Path,
    samples: list[tuple[int, dict[str, Any]]],
    output_path: Path,
    status_reporter: StatusReporter | None = None,
) -> int:
    """Write exact raw dataset JSONL rows for the selected episodes."""
    turn_data_path = _resolve_turn_data_path(
        manifest_path=manifest_path,
        manifest_payload=manifest_payload,
    )
    if not turn_data_path.exists():
        raise RuntimeError(f"Raw SPoRC turn data file was not found: {turn_data_path}")

    selected_episode_keys = {
        str(sample.get("episode_key") or "")
        for _, sample in samples
    }
    remaining_episode_keys = set(selected_episode_keys)
    current_episode_key = ""
    streamed_row_count = 0
    written_row_count = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    open_fn = gzip.open if turn_data_path.suffix == ".gz" else open
    with open_fn(turn_data_path, "rt", encoding="utf-8") as source_handle:
        with output_path.open("w", encoding="utf-8") as output_handle:
            for raw_line in source_handle:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                streamed_row_count += 1
                if (
                    status_reporter is not None
                    and streamed_row_count % _TURN_STREAM_PROGRESS_INTERVAL == 0
                ):
                    status_reporter(
                        "Loading selected raw SPoRC rows... streamed "
                        f"{streamed_row_count:,} rows so far."
                    )
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    continue
                episode_key = _turn_episode_key(payload)
                if not episode_key:
                    continue

                if not current_episode_key:
                    current_episode_key = episode_key
                elif episode_key != current_episode_key:
                    if current_episode_key in remaining_episode_keys:
                        remaining_episode_keys.discard(current_episode_key)
                        if not remaining_episode_keys:
                            break
                    current_episode_key = episode_key

                if episode_key not in selected_episode_keys:
                    continue
                output_handle.write(raw_line)
                written_row_count += 1

    if current_episode_key in remaining_episode_keys:
        remaining_episode_keys.discard(current_episode_key)
    if remaining_episode_keys:
        missing_episode_keys = sorted(remaining_episode_keys)
        raise RuntimeError(
            "No raw dataset turns were found for selected manifest episodes: "
            + ", ".join(missing_episode_keys)
        )
    return written_row_count


def _turn_episode_key(record: dict[str, Any]) -> str:
    """Return the raw dataset episode key for one speaker-turn row."""
    for field_name in _EPISODE_KEY_FIELDS:
        value = str(record.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _turn_text(record: dict[str, Any]) -> str:
    """Return raw transcript text for one speaker-turn row."""
    for field_name in _TURN_TEXT_FIELDS:
        value = record.get(field_name)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _turn_speakers(record: dict[str, Any]) -> list[str]:
    """Return the raw speaker labels for one speaker-turn row."""
    for field_name in _TURN_SPEAKER_FIELDS:
        raw_value = record.get(field_name)
        values = _coerce_list_of_strings(raw_value)
        if values:
            return values
    return []


def _coerce_list_of_strings(value: Any) -> list[str]:
    """Coerce a turn-record field into a list of strings."""
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [text]


def _turn_float(record: dict[str, Any], *field_names: str) -> float:
    """Return the first parseable float among the provided turn fields."""
    for field_name in field_names:
        value = record.get(field_name)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            try:
                return float(stripped)
            except ValueError:
                continue
    return 0.0


def _seconds_to_milliseconds(value: float) -> int:
    """Convert one seconds value into rounded milliseconds."""
    return max(0, int(round(float(value) * 1000.0)))


def _format_timestamp(milliseconds: int) -> str:
    """Format one millisecond timestamp into `HH:MM:SS`."""
    total_seconds = max(0, int(round(milliseconds / 1000.0)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
