"""Prepare the SPoRC benchmark slice for summarizer-critic evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
import gzip
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
from typing import Any

from card_framework.agents.utils import count_words
from card_framework.benchmark.artifacts import utc_now_iso
from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_DATASET_MANIFEST_PATH,
    DEFAULT_DATASET_VARIANT,
    DEFAULT_LONGFORM_AUDIT_PATH,
    DEFAULT_LONGFORM_MANIFEST_PATH,
    DEFAULT_PREPARED_ROOT,
    DEFAULT_REFERENCE_WPM,
    DEFAULT_SPORC_ROOT,
    LONGFORM_MAX_DOMINANT_SPEAKER_SHARE,
    LONGFORM_MAX_NON_LEXICAL_RATIO,
    LONGFORM_MIN_SEGMENTS,
    LONGFORM_MIN_SPEAKERS,
    LONGFORM_MIN_SPEAKER_SEGMENTS,
    LONGFORM_MIN_SPEAKER_WORDS,
    LONGFORM_MIN_SPOKEN_WORDS,
    PreparedBenchmarkSample,
    PreparedDatasetManifest,
    SPORC_DATASET_ID,
    SPORC_VARIANT_FILES,
)


class DatasetPreparationError(RuntimeError):
    """Raise when the SPoRC benchmark slice cannot be prepared."""


_SPORC_TRANSCRIPT_SUMMARY_RATIO = 0.25
_SPORC_MIN_TARGET_SECONDS = 12
_SPORC_MAX_TARGET_SECONDS = 45
_LONGFORM_CURATION_PROFILE = "longform_15k_v1"
_EPISODE_METADATA_CACHE_FILENAME = "sporc_episode_metadata_v1.sqlite3"
_FULL_SAMPLES_FILENAME = "samples.jsonl.gz"
_EPISODE_CACHE_SCHEMA_VERSION = "sporc_episode_metadata_v1"
_EPISODE_INDEX_PROGRESS_INTERVAL = 50_000
_TURN_STREAM_PROGRESS_INTERVAL = 500_000
_WORD_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")
_NON_LEXICAL_TOKENS = frozenset(
    {
        "ad",
        "ads",
        "applause",
        "laugh",
        "laughed",
        "laughing",
        "laughter",
        "music",
        "noise",
        "silence",
    }
)
LOGGER = logging.getLogger(__name__)


def derive_target_seconds(
    reference_summary: str,
    *,
    reference_wpm: float = DEFAULT_REFERENCE_WPM,
) -> int:
    """Derive a duration target from the reference summary length."""
    if reference_wpm <= 0:
        raise ValueError("reference_wpm must be greater than zero.")
    reference_words = max(1, count_words(reference_summary))
    return max(1, int(round((reference_words / reference_wpm) * 60.0)))


def derive_sporc_target_seconds(
    reference_summary: str,
    *,
    transcript_word_count: int,
    reference_wpm: float = DEFAULT_REFERENCE_WPM,
    transcript_summary_ratio: float = _SPORC_TRANSCRIPT_SUMMARY_RATIO,
    min_target_seconds: int = _SPORC_MIN_TARGET_SECONDS,
    max_target_seconds: int = _SPORC_MAX_TARGET_SECONDS,
) -> int:
    """Derive a practical SPoRC duration target from summary and transcript size.

    SPoRC episode descriptions are often teaser-length and can be too short to pair
    cleanly with the repo's coverage-sensitive summarizer-critic loop. This hybrid
    target keeps the reference-summary signal while enforcing a podcast-friendly
    lower bound based on transcript size and a bounded target range.
    """
    if transcript_summary_ratio <= 0:
        raise ValueError("transcript_summary_ratio must be greater than zero.")
    if min_target_seconds <= 0:
        raise ValueError("min_target_seconds must be greater than zero.")
    if max_target_seconds < min_target_seconds:
        raise ValueError("max_target_seconds must be >= min_target_seconds.")

    reference_seconds = derive_target_seconds(
        reference_summary,
        reference_wpm=reference_wpm,
    )
    transcript_words = max(1, int(transcript_word_count))
    transcript_seconds = int(
        round(((transcript_words * transcript_summary_ratio) / reference_wpm) * 60.0)
    )
    bounded = max(reference_seconds, transcript_seconds, min_target_seconds)
    return min(max_target_seconds, bounded)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    return ""


def _coerce_list_of_strings(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in (_coerce_text(item) for item in value) if item]
    if isinstance(value, tuple):
        return [item for item in (_coerce_text(item) for item in value) if item]
    text = _coerce_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [item for item in (_coerce_text(item) for item in parsed) if item]
    return [text]


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): raw_value for key, raw_value in value.items()}
    text = _coerce_text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {str(key): raw_value for key, raw_value in parsed.items()}


def _episode_key(record: dict[str, Any]) -> str:
    candidates = (
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
    for field_name in candidates:
        value = _coerce_text(record.get(field_name))
        if value:
            return value
    return ""


def _episode_reference_summary(record: dict[str, Any]) -> str:
    candidates = (
        "epDescription",
        "podDescription",
        "episodeDescription",
        "description",
        "summary",
        "episodeSummary",
    )
    for field_name in candidates:
        value = _coerce_text(record.get(field_name))
        if value:
            without_html = re.sub(r"<[^>]+>", " ", value)
            return re.sub(r"\s+", " ", without_html).strip()
    return ""


def _episode_title(record: dict[str, Any]) -> str:
    candidates = ("epTitle", "episodeTitle", "title", "episodeName", "podcastName")
    for field_name in candidates:
        value = _coerce_text(record.get(field_name))
        if value:
            return value
    return ""


def _episode_source_audio_url(record: dict[str, Any], *, episode_key: str) -> str:
    """Resolve a source-audio URL or path for one SPoRC episode row."""
    candidates = ("mp3url", "mp3Url", "mp3_url", "episodeMp3Url", "episode_url")
    for field_name in candidates:
        value = _coerce_text(record.get(field_name))
        if value:
            return value
    return episode_key


def _turn_text(record: dict[str, Any]) -> str:
    candidates = ("turnText", "content", "text", "utterance", "transcript")
    for field_name in candidates:
        value = _coerce_text(record.get(field_name))
        if value:
            return re.sub(r"\s+", " ", value).strip()
    return ""


def _turn_speakers(record: dict[str, Any]) -> list[str]:
    candidates = ("speakers", "speaker", "speakerLabel", "speaker_label")
    for field_name in candidates:
        values = _coerce_list_of_strings(record.get(field_name))
        if values:
            return values
    return []


def _turn_float(record: dict[str, Any], *field_names: str) -> float:
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


def _slugify(value: str) -> str:
    collapsed = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return collapsed[:80] or "episode"


def _word_tokens(text: str) -> list[str]:
    return [token for token in _WORD_TOKEN_PATTERN.findall(text.lower()) if token]


def _timestamp_milliseconds(payload: dict[str, Any], *field_names: str) -> int | None:
    for field_name in field_names:
        value = payload.get(field_name)
        if value is None:
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(round(value))
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                continue
            try:
                return int(round(float(stripped)))
            except ValueError:
                continue
    return None


def _speaker_alias_map(record: dict[str, Any], speakers: list[str]) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for speaker in speakers:
        alias_map[speaker] = speaker
        alias_map[speaker.lower()] = speaker
    for field_name in ("hostSpeakerLabels", "guestSpeakerLabels"):
        for name, raw_labels in _coerce_mapping(record.get(field_name)).items():
            labels = _coerce_list_of_strings(raw_labels)
            for label in labels:
                if label:
                    alias_map[name] = label
                    alias_map[name.lower()] = label
    return alias_map


def _speaker_notes(record: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    for field_name in (
        "hostPredictedNames",
        "guestPredictedNames",
        "neitherPredictedNames",
        "mainEpSpeakers",
    ):
        values = _coerce_list_of_strings(record.get(field_name))
        if values:
            notes.append(f"{field_name}: {', '.join(values)}")
    for field_name in ("hostSpeakerLabels", "guestSpeakerLabels"):
        mapping = _coerce_mapping(record.get(field_name))
        if mapping:
            notes.append(f"{field_name}: {json.dumps(mapping, sort_keys=True)}")
    return notes


def _iter_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _episode_metadata_cache_path(prepared_root: Path) -> Path:
    return (prepared_root / ".cache" / _EPISODE_METADATA_CACHE_FILENAME).resolve()


def _episode_cache_signature(episode_data_path: Path) -> str:
    stat_result = episode_data_path.stat()
    return json.dumps(
        {
            "schema_version": _EPISODE_CACHE_SCHEMA_VERSION,
            "episode_data_path": str(episode_data_path.resolve()),
            "size_bytes": stat_result.st_size,
            "mtime_ns": stat_result.st_mtime_ns,
        },
        sort_keys=True,
    )


def _open_episode_metadata_connection(cache_path: Path) -> sqlite3.Connection:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(cache_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _initialize_episode_metadata_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS episodes (
            episode_key TEXT PRIMARY KEY,
            episode_index INTEGER NOT NULL,
            reference_summary TEXT NOT NULL,
            episode_title TEXT NOT NULL,
            source_audio_url TEXT NOT NULL,
            speaker_alias_map_json TEXT NOT NULL,
            speaker_notes_json TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS episodes_episode_index_idx
        ON episodes (episode_index);
        """
    )


def _prepare_sample_spool_table(connection: sqlite3.Connection) -> None:
    connection.execute("DROP TABLE IF EXISTS prepared_samples_spool")
    connection.execute(
        """
        CREATE TABLE prepared_samples_spool (
            episode_index INTEGER PRIMARY KEY,
            sample_json TEXT NOT NULL
        )
        """
    )


def _metadata_value(connection: sqlite3.Connection, key: str) -> str | None:
    row = connection.execute(
        "SELECT value FROM metadata WHERE key = ?",
        (key,),
    ).fetchone()
    if row is None:
        return None
    return _coerce_text(row["value"])


def _set_metadata_value(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        """
        INSERT INTO metadata (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )


def _clear_episode_metadata_cache(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM episodes")
    connection.execute("DELETE FROM metadata")


def _ensure_episode_metadata_cache(
    connection: sqlite3.Connection,
    *,
    episode_data_path: Path,
) -> None:
    _initialize_episode_metadata_schema(connection)
    expected_signature = _episode_cache_signature(episode_data_path)
    cached_signature = _metadata_value(connection, "episode_source_signature")
    cached_count = int(
        connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    )
    if cached_signature == expected_signature and cached_count > 0:
        LOGGER.info(
            "Reusing cached SPoRC episode metadata index with %s rows from %s.",
            cached_count,
            episode_data_path,
        )
        return

    LOGGER.info("Building SPoRC episode metadata index from %s.", episode_data_path)
    _clear_episode_metadata_cache(connection)
    rows_since_commit = 0
    with connection:
        for episode_index, episode_record in enumerate(
            _iter_jsonl_gz(episode_data_path),
            start=1,
        ):
            episode_key = _episode_key(episode_record)
            if not episode_key:
                continue
            try:
                connection.execute(
                    """
                    INSERT INTO episodes (
                        episode_key,
                        episode_index,
                        reference_summary,
                        episode_title,
                        source_audio_url,
                        speaker_alias_map_json,
                        speaker_notes_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        episode_key,
                        episode_index,
                        _episode_reference_summary(episode_record),
                        _episode_title(episode_record),
                        _episode_source_audio_url(
                            episode_record,
                            episode_key=episode_key,
                        ),
                        json.dumps(
                            _speaker_alias_map(episode_record, []),
                            sort_keys=True,
                        ),
                        json.dumps(_speaker_notes(episode_record)),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DatasetPreparationError(
                    "SPoRC episode metadata contained a duplicate episode key. "
                    f"Unable to build a stable cache for {episode_key!r}."
                ) from exc
            rows_since_commit += 1
            if episode_index % _EPISODE_INDEX_PROGRESS_INTERVAL == 0:
                LOGGER.info(
                    "Indexed %s SPoRC episode rows into %s.",
                    episode_index,
                    connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                )
            if rows_since_commit >= _EPISODE_INDEX_PROGRESS_INTERVAL:
                connection.commit()
                rows_since_commit = 0
        _set_metadata_value(connection, "episode_source_signature", expected_signature)
    LOGGER.info(
        "Finished SPoRC episode metadata index with %s rows.",
        connection.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
    )


def _iter_turn_groups(
    turn_data_path: Path,
) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    current_episode_key = ""
    current_turns: list[dict[str, Any]] = []
    completed_episode_keys: set[str] = set()
    turn_row_count = 0
    group_count = 0
    for record in _iter_jsonl_gz(turn_data_path):
        turn_row_count += 1
        if turn_row_count % _TURN_STREAM_PROGRESS_INTERVAL == 0:
            LOGGER.info(
                "Streamed %s SPoRC turn rows across %s contiguous episode groups.",
                turn_row_count,
                group_count,
            )
        episode_key = _episode_key(record)
        if not episode_key:
            continue
        if not current_episode_key:
            current_episode_key = episode_key
            current_turns = [record]
            continue
        if episode_key == current_episode_key:
            current_turns.append(record)
            continue
        if episode_key in completed_episode_keys:
            raise DatasetPreparationError(
                "SPoRC turn rows are no longer grouped contiguously by episode. "
                f"Encountered a repeated episode key after its block ended: {episode_key!r}."
            )
        yield current_episode_key, current_turns
        completed_episode_keys.add(current_episode_key)
        group_count += 1
        current_episode_key = episode_key
        current_turns = [record]
    if current_episode_key:
        yield current_episode_key, current_turns


def _augment_speaker_alias_map(
    *,
    base_alias_map: dict[str, str],
    unique_speakers: list[str],
) -> dict[str, str]:
    alias_map = dict(base_alias_map)
    for speaker in unique_speakers:
        alias_map[speaker] = speaker
        alias_map[speaker.lower()] = speaker
    return alias_map


def _parse_prepared_sample(raw_sample: dict[str, Any]) -> PreparedBenchmarkSample:
    return PreparedBenchmarkSample(
        sample_id=_coerce_text(raw_sample.get("sample_id")),
        dataset=_coerce_text(raw_sample.get("dataset")),
        episode_key=_coerce_text(raw_sample.get("episode_key")),
        transcript_path=_coerce_text(raw_sample.get("transcript_path")),
        reference_summary=_coerce_text(raw_sample.get("reference_summary")),
        target_seconds=int(raw_sample.get("target_seconds", 0)),
        metadata=(
            dict(raw_sample.get("metadata", {}))
            if isinstance(raw_sample.get("metadata"), dict)
            else {}
        ),
    )


def _build_prepared_sample(
    *,
    episode_index: int,
    episode_key: str,
    reference_summary: str,
    episode_title: str,
    source_audio_url: str,
    base_speaker_alias_map: dict[str, str],
    speaker_notes: list[str],
    raw_turns: list[dict[str, Any]],
    transcript_root: Path,
    dataset_variant: str,
    reference_wpm: float,
    episode_data_path: Path,
    turn_data_path: Path,
) -> PreparedBenchmarkSample | None:
    if not reference_summary:
        return None

    segments: list[dict[str, Any]] = []
    unique_speakers: list[str] = []
    seen_speakers: set[str] = set()
    for turn_index, raw_turn in enumerate(
        sorted(
            raw_turns,
            key=lambda item: _turn_float(item, "start", "startTime", "start_time"),
        ),
        start=1,
    ):
        text = _turn_text(raw_turn)
        speakers = _turn_speakers(raw_turn)
        if not text or not speakers:
            continue
        canonical_speaker = speakers[0]
        if canonical_speaker not in seen_speakers:
            unique_speakers.append(canonical_speaker)
            seen_speakers.add(canonical_speaker)
        overlap_speakers = [speaker for speaker in speakers[1:] if speaker]
        start_time_seconds = _turn_float(raw_turn, "start", "startTime", "start_time")
        end_time_seconds = _turn_float(raw_turn, "end", "endTime", "end_time")
        if end_time_seconds <= start_time_seconds:
            end_time_seconds = start_time_seconds + 1.0
        start_time = max(0, int(round(start_time_seconds * 1000.0)))
        end_time = max(start_time + 1, int(round(end_time_seconds * 1000.0)))
        segments.append(
            {
                "speaker": canonical_speaker,
                "text": text,
                "start_time": start_time,
                "end_time": end_time,
                "overlap_speakers": overlap_speakers,
                "source_turn_index": turn_index,
            }
        )
    if not segments:
        return None

    speaker_alias_map = _augment_speaker_alias_map(
        base_alias_map=base_speaker_alias_map,
        unique_speakers=unique_speakers,
    )
    transcript_word_count = sum(count_words(str(segment["text"])) for segment in segments)
    transcript_payload = {
        "segments": segments,
        "metadata": {
            "dataset": "sporc",
            "episode_key": episode_key,
            "episode_title": episode_title,
            "speaker_alias_map": speaker_alias_map,
            "speaker_notes": speaker_notes,
            "source_audio_url": source_audio_url,
            "source_episode_data_path": str(episode_data_path),
            "source_turn_data_path": str(turn_data_path),
        },
    }
    transcript_path = (
        transcript_root / f"{episode_index:05d}_{_slugify(episode_key)}.transcript.json"
    ).resolve()
    transcript_path.write_text(
        json.dumps(transcript_payload, indent=2),
        encoding="utf-8",
    )
    return PreparedBenchmarkSample(
        sample_id=f"sporc_{episode_index:05d}_{_slugify(episode_key)}",
        dataset=f"sporc_{dataset_variant}",
        episode_key=episode_key,
        transcript_path=str(transcript_path),
        reference_summary=reference_summary,
        target_seconds=derive_sporc_target_seconds(
            reference_summary,
            transcript_word_count=transcript_word_count,
            reference_wpm=reference_wpm,
        ),
        metadata={
            "episode_title": episode_title,
            "reference_word_count": count_words(reference_summary),
            "source_audio_url": source_audio_url,
            "speaker_alias_map": speaker_alias_map,
            "speaker_notes": speaker_notes,
        },
    )


def resolve_sporc_source_files(
    *,
    dataset_root: Path = DEFAULT_SPORC_ROOT,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    episode_data_path: str = "",
    turn_data_path: str = "",
    force_download: bool = False,
) -> tuple[Path, Path]:
    """Resolve local or gated-download SPoRC source files."""
    explicit_episode = Path(episode_data_path).resolve() if episode_data_path else None
    explicit_turn = Path(turn_data_path).resolve() if turn_data_path else None
    if explicit_episode is not None and explicit_turn is not None:
        if not explicit_episode.exists():
            raise DatasetPreparationError(
                f"SPoRC episode data file does not exist: {explicit_episode}"
            )
        if not explicit_turn.exists():
            raise DatasetPreparationError(
                f"SPoRC turn data file does not exist: {explicit_turn}"
            )
        return explicit_episode, explicit_turn

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - import guard
        raise DatasetPreparationError(
            "huggingface_hub is required to download gated SPoRC files. "
            "Install repo dependencies with `uv sync --dev` or pass explicit "
            "--episode-data-path and --turn-data-path values."
        ) from exc

    variant_files = SPORC_VARIANT_FILES.get(dataset_variant)
    if variant_files is None:
        raise DatasetPreparationError(
            f"Unsupported SPoRC dataset variant: {dataset_variant}"
        )

    dataset_root.mkdir(parents=True, exist_ok=True)
    token = (
        os.getenv("HF_ACCESS_TOKEN", "").strip()
        or os.getenv("HF_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_TOKEN", "").strip()
        or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
        or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
        or None
    )
    try:
        episode_path = Path(
            hf_hub_download(
                repo_id=SPORC_DATASET_ID,
                repo_type="dataset",
                filename=variant_files["episode"],
                local_dir=str(dataset_root / dataset_variant),
                token=token,
                force_download=force_download,
            )
        ).resolve()
        turn_path = Path(
            hf_hub_download(
                repo_id=SPORC_DATASET_ID,
                repo_type="dataset",
                filename=variant_files["turn"],
                local_dir=str(dataset_root / dataset_variant),
                token=token,
                force_download=force_download,
            )
        ).resolve()
    except Exception as exc:
        raise DatasetPreparationError(
            "Unable to download gated SPoRC files. Accept the dataset terms on "
            "Hugging Face, set `HF_ACCESS_TOKEN`, `HF_TOKEN`, or "
            "`HUGGINGFACE_HUB_TOKEN`, or pass "
            "local `--episode-data-path` and `--turn-data-path` values."
        ) from exc
    return episode_path, turn_path


def build_prepared_dataset_manifest(
    *,
    episode_data_path: Path,
    turn_data_path: Path,
    prepared_root: Path = DEFAULT_PREPARED_ROOT,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    reference_wpm: float = DEFAULT_REFERENCE_WPM,
) -> PreparedDatasetManifest:
    """Build the prepared SPoRC benchmark manifest from raw episode and turn files."""
    if not episode_data_path.exists():
        raise DatasetPreparationError(
            f"SPoRC episode data file does not exist: {episode_data_path}"
        )
    if not turn_data_path.exists():
        raise DatasetPreparationError(
            f"SPoRC turn data file does not exist: {turn_data_path}"
        )

    resolved_episode_path = episode_data_path.resolve()
    resolved_turn_path = turn_data_path.resolve()
    prepared_root = prepared_root.resolve()
    prepared_root.mkdir(parents=True, exist_ok=True)
    transcript_root = (prepared_root / "transcripts").resolve()
    transcript_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (prepared_root / DEFAULT_DATASET_MANIFEST_PATH.name).resolve()
    samples_path = (
        (prepared_root / _FULL_SAMPLES_FILENAME).resolve()
        if dataset_variant == "full"
        else None
    )
    cache_path = _episode_metadata_cache_path(prepared_root)
    samples: tuple[PreparedBenchmarkSample, ...] = ()

    with _open_episode_metadata_connection(cache_path) as connection:
        _ensure_episode_metadata_cache(
            connection,
            episode_data_path=resolved_episode_path,
        )
        _prepare_sample_spool_table(connection)
        try:
            for episode_key, raw_turns in _iter_turn_groups(resolved_turn_path):
                episode_row = connection.execute(
                    """
                    SELECT
                        episode_index,
                        reference_summary,
                        episode_title,
                        source_audio_url,
                        speaker_alias_map_json,
                        speaker_notes_json
                    FROM episodes
                    WHERE episode_key = ?
                    """,
                    (episode_key,),
                ).fetchone()
                if episode_row is None:
                    continue
                sample = _build_prepared_sample(
                    episode_index=int(episode_row["episode_index"]),
                    episode_key=episode_key,
                    reference_summary=_coerce_text(episode_row["reference_summary"]),
                    episode_title=_coerce_text(episode_row["episode_title"]),
                    source_audio_url=_coerce_text(episode_row["source_audio_url"]),
                    base_speaker_alias_map=(
                        json.loads(_coerce_text(episode_row["speaker_alias_map_json"]))
                        if _coerce_text(episode_row["speaker_alias_map_json"])
                        else {}
                    ),
                    speaker_notes=(
                        json.loads(_coerce_text(episode_row["speaker_notes_json"]))
                        if _coerce_text(episode_row["speaker_notes_json"])
                        else []
                    ),
                    raw_turns=raw_turns,
                    transcript_root=transcript_root,
                    dataset_variant=dataset_variant,
                    reference_wpm=reference_wpm,
                    episode_data_path=resolved_episode_path,
                    turn_data_path=resolved_turn_path,
                )
                if sample is None:
                    continue
                connection.execute(
                    """
                    INSERT INTO prepared_samples_spool (episode_index, sample_json)
                    VALUES (?, ?)
                    """,
                    (
                        int(episode_row["episode_index"]),
                        json.dumps(asdict(sample), separators=(",", ":")),
                    ),
                )

            sample_count = int(
                connection.execute(
                    "SELECT COUNT(*) FROM prepared_samples_spool"
                ).fetchone()[0]
            )
            if sample_count <= 0:
                raise DatasetPreparationError(
                    "No usable SPoRC samples were prepared from the supplied files."
                )

            if samples_path is not None:
                with gzip.open(samples_path, "wt", encoding="utf-8") as handle:
                    for row in connection.execute(
                        """
                        SELECT sample_json
                        FROM prepared_samples_spool
                        ORDER BY episode_index
                        """
                    ):
                        handle.write(_coerce_text(row["sample_json"]))
                        handle.write("\n")
            else:
                samples = tuple(
                    _parse_prepared_sample(json.loads(_coerce_text(row["sample_json"])))
                    for row in connection.execute(
                        """
                        SELECT sample_json
                        FROM prepared_samples_spool
                        ORDER BY episode_index
                        """
                    )
                )
        finally:
            connection.execute("DROP TABLE IF EXISTS prepared_samples_spool")
            connection.commit()

    manifest_payload = {
        "dataset_id": f"sporc_{dataset_variant}",
        "created_at_utc": utc_now_iso(),
        "sporc_dataset_id": SPORC_DATASET_ID,
        "dataset_variant": dataset_variant,
        "episode_data_path": str(resolved_episode_path),
        "turn_data_path": str(resolved_turn_path),
        "reference_wpm": reference_wpm,
        "sample_count": sample_count,
    }
    if samples_path is not None:
        manifest_payload["samples_path"] = str(samples_path)
    else:
        manifest_payload["samples"] = [asdict(sample) for sample in samples]

    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return PreparedDatasetManifest(
        manifest_path=manifest_path,
        dataset_id=f"sporc_{dataset_variant}",
        sporc_dataset_id=SPORC_DATASET_ID,
        dataset_variant=dataset_variant,
        episode_data_path=resolved_episode_path,
        turn_data_path=resolved_turn_path,
        sample_count=sample_count,
        samples=samples,
        samples_path=samples_path,
    )


def _longform_curation_thresholds() -> dict[str, int | float | str]:
    return {
        "profile": _LONGFORM_CURATION_PROFILE,
        "min_spoken_word_count": LONGFORM_MIN_SPOKEN_WORDS,
        "min_segment_count": LONGFORM_MIN_SEGMENTS,
        "min_unique_speaker_count": LONGFORM_MIN_SPEAKERS,
        "min_speaker_word_count": LONGFORM_MIN_SPEAKER_WORDS,
        "min_speaker_segment_count": LONGFORM_MIN_SPEAKER_SEGMENTS,
        "max_dominant_speaker_word_share": LONGFORM_MAX_DOMINANT_SPEAKER_SHARE,
        "max_non_lexical_word_ratio": LONGFORM_MAX_NON_LEXICAL_RATIO,
    }


def _collect_longform_curation_stats(
    transcript_payload: dict[str, Any],
) -> dict[str, Any]:
    raw_segments = transcript_payload.get("segments", [])
    speaker_stats: dict[str, dict[str, int]] = {}
    raw_word_count = 0
    spoken_word_count = 0
    non_lexical_word_count = 0
    segment_count = 0
    min_start_time: int | None = None
    max_end_time: int | None = None

    for raw_segment in raw_segments:
        if not isinstance(raw_segment, dict):
            continue
        text = _coerce_text(raw_segment.get("text"))
        if not text:
            continue
        speaker = _coerce_text(raw_segment.get("speaker")) or "UNKNOWN"
        tokens = _word_tokens(text)
        raw_segment_word_count = len(tokens)
        non_lexical_segment_word_count = sum(
            1 for token in tokens if token in _NON_LEXICAL_TOKENS
        )
        spoken_segment_word_count = raw_segment_word_count - non_lexical_segment_word_count
        stats = speaker_stats.setdefault(
            speaker,
            {
                "raw_word_count": 0,
                "spoken_word_count": 0,
                "segment_count": 0,
            },
        )
        stats["raw_word_count"] += raw_segment_word_count
        stats["spoken_word_count"] += spoken_segment_word_count
        stats["segment_count"] += 1
        raw_word_count += raw_segment_word_count
        spoken_word_count += spoken_segment_word_count
        non_lexical_word_count += non_lexical_segment_word_count
        segment_count += 1
        start_time = _timestamp_milliseconds(
            raw_segment,
            "start_time",
            "startTime",
        )
        end_time = _timestamp_milliseconds(
            raw_segment,
            "end_time",
            "endTime",
        )
        if start_time is not None:
            min_start_time = (
                start_time
                if min_start_time is None
                else min(min_start_time, start_time)
            )
        if end_time is not None:
            max_end_time = (
                end_time if max_end_time is None else max(max_end_time, end_time)
            )

    dominant_speaker_word_share = 0.0
    if spoken_word_count > 0 and speaker_stats:
        dominant_speaker_word_share = max(
            speaker["spoken_word_count"] for speaker in speaker_stats.values()
        ) / float(spoken_word_count)
    non_lexical_word_ratio = (
        non_lexical_word_count / float(raw_word_count) if raw_word_count > 0 else 0.0
    )
    strong_speaker_count = sum(
        1
        for speaker in speaker_stats.values()
        if speaker["spoken_word_count"] >= LONGFORM_MIN_SPEAKER_WORDS
        and speaker["segment_count"] >= LONGFORM_MIN_SPEAKER_SEGMENTS
    )
    timestamp_span_seconds = 0.0
    if min_start_time is not None and max_end_time is not None and max_end_time > min_start_time:
        timestamp_span_seconds = (max_end_time - min_start_time) / 1000.0

    ordered_speaker_stats = {
        speaker: {
            "raw_word_count": values["raw_word_count"],
            "spoken_word_count": values["spoken_word_count"],
            "segment_count": values["segment_count"],
        }
        for speaker, values in sorted(
            speaker_stats.items(),
            key=lambda item: (-item[1]["spoken_word_count"], item[0]),
        )
    }
    return {
        "profile": _LONGFORM_CURATION_PROFILE,
        "raw_word_count": raw_word_count,
        "spoken_word_count": spoken_word_count,
        "non_lexical_word_count": non_lexical_word_count,
        "non_lexical_word_ratio": round(non_lexical_word_ratio, 4),
        "segment_count": segment_count,
        "unique_speaker_count": len(speaker_stats),
        "strong_speaker_count": strong_speaker_count,
        "dominant_speaker_word_share": round(dominant_speaker_word_share, 4),
        "timestamp_span_seconds": round(timestamp_span_seconds, 3),
        "speaker_stats": ordered_speaker_stats,
    }


def _evaluate_longform_curation_stats(
    stats: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if int(stats.get("spoken_word_count", 0)) < LONGFORM_MIN_SPOKEN_WORDS:
        failures.append(f"spoken_word_count<{LONGFORM_MIN_SPOKEN_WORDS}")
    if int(stats.get("segment_count", 0)) < LONGFORM_MIN_SEGMENTS:
        failures.append(f"segment_count<{LONGFORM_MIN_SEGMENTS}")
    if int(stats.get("unique_speaker_count", 0)) < LONGFORM_MIN_SPEAKERS:
        failures.append(f"unique_speaker_count<{LONGFORM_MIN_SPEAKERS}")
    if int(stats.get("strong_speaker_count", 0)) < LONGFORM_MIN_SPEAKERS:
        failures.append(
            f"strong_speaker_count<{LONGFORM_MIN_SPEAKERS}"
        )
    if (
        float(stats.get("dominant_speaker_word_share", 0.0))
        > LONGFORM_MAX_DOMINANT_SPEAKER_SHARE
    ):
        failures.append(
            f"dominant_speaker_word_share>{LONGFORM_MAX_DOMINANT_SPEAKER_SHARE:.2f}"
        )
    if float(stats.get("non_lexical_word_ratio", 0.0)) > LONGFORM_MAX_NON_LEXICAL_RATIO:
        failures.append(
            f"non_lexical_word_ratio>{LONGFORM_MAX_NON_LEXICAL_RATIO:.2f}"
        )
    return failures


def _load_transcript_payload(transcript_path: Path) -> dict[str, Any]:
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetPreparationError(
            f"Prepared transcript payload must be an object: {transcript_path}"
        )
    return payload


def curate_longform_sporc_manifest(
    *,
    source_manifest_path: Path,
    output_manifest_path: Path = DEFAULT_LONGFORM_MANIFEST_PATH,
    audit_path: Path = DEFAULT_LONGFORM_AUDIT_PATH,
) -> tuple[PreparedDatasetManifest, Path]:
    """Filter a prepared SPoRC manifest down to strict long-form dialogue episodes."""
    source_manifest = load_prepared_dataset_manifest(
        source_manifest_path,
        include_samples=False,
    )
    output_manifest_path = output_manifest_path.resolve()
    audit_path = audit_path.resolve()
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    selected_samples: list[PreparedBenchmarkSample] = []
    audit_rows: list[dict[str, Any]] = []
    for sample in iter_prepared_dataset_samples(source_manifest.manifest_path):
        transcript_path = Path(sample.transcript_path).resolve()
        transcript_payload = _load_transcript_payload(transcript_path)
        stats = _collect_longform_curation_stats(transcript_payload)
        excluded_reasons = _evaluate_longform_curation_stats(stats)
        audit_rows.append(
            {
                "sample_id": sample.sample_id,
                "episode_key": sample.episode_key,
                "transcript_path": str(transcript_path),
                "included": not excluded_reasons,
                "excluded_reasons": excluded_reasons,
                "curation_stats": stats,
            }
        )
        if excluded_reasons:
            continue
        metadata = dict(sample.metadata)
        metadata["longform_curation"] = stats
        selected_samples.append(
            PreparedBenchmarkSample(
                sample_id=sample.sample_id,
                dataset=sample.dataset,
                episode_key=sample.episode_key,
                transcript_path=sample.transcript_path,
                reference_summary=sample.reference_summary,
                target_seconds=sample.target_seconds,
                metadata=metadata,
            )
        )

    if not selected_samples:
        raise DatasetPreparationError(
            "No SPoRC samples satisfied the long-form curation thresholds."
        )

    manifest_payload = {
        "dataset_id": f"{source_manifest.dataset_id}_longform_15k",
        "created_at_utc": utc_now_iso(),
        "source_manifest_path": str(source_manifest.manifest_path),
        "sporc_dataset_id": source_manifest.sporc_dataset_id,
        "dataset_variant": source_manifest.dataset_variant,
        "episode_data_path": str(source_manifest.episode_data_path),
        "turn_data_path": str(source_manifest.turn_data_path),
        "sample_count": len(selected_samples),
        "curation_profile": _LONGFORM_CURATION_PROFILE,
        "curation_thresholds": _longform_curation_thresholds(),
        "samples": [asdict(sample) for sample in selected_samples],
    }
    output_manifest_path.write_text(
        json.dumps(manifest_payload, indent=2),
        encoding="utf-8",
    )

    audit_payload = {
        "created_at_utc": utc_now_iso(),
        "source_manifest_path": str(source_manifest.manifest_path),
        "output_manifest_path": str(output_manifest_path),
        "curation_profile": _LONGFORM_CURATION_PROFILE,
        "curation_thresholds": _longform_curation_thresholds(),
        "evaluated_sample_count": len(audit_rows),
        "selected_sample_count": len(selected_samples),
        "samples": audit_rows,
    }
    audit_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    return (
        PreparedDatasetManifest(
            manifest_path=output_manifest_path,
            dataset_id=str(manifest_payload["dataset_id"]),
            sporc_dataset_id=source_manifest.sporc_dataset_id,
            dataset_variant=source_manifest.dataset_variant,
            episode_data_path=source_manifest.episode_data_path,
            turn_data_path=source_manifest.turn_data_path,
            sample_count=len(selected_samples),
            samples=tuple(selected_samples),
            samples_path=None,
        ),
        audit_path,
    )


def prepare_sporc_benchmark(
    *,
    prepared_root: Path = DEFAULT_PREPARED_ROOT,
    dataset_variant: str = DEFAULT_DATASET_VARIANT,
    episode_data_path: str = "",
    turn_data_path: str = "",
    reference_wpm: float = DEFAULT_REFERENCE_WPM,
    force_download: bool = False,
) -> PreparedDatasetManifest:
    """Download or reuse SPoRC data, then build a reproducible benchmark slice."""
    resolved_episode_path, resolved_turn_path = resolve_sporc_source_files(
        dataset_root=DEFAULT_SPORC_ROOT,
        dataset_variant=dataset_variant,
        episode_data_path=episode_data_path,
        turn_data_path=turn_data_path,
        force_download=force_download,
    )
    return build_prepared_dataset_manifest(
        episode_data_path=resolved_episode_path,
        turn_data_path=resolved_turn_path,
        prepared_root=prepared_root,
        dataset_variant=dataset_variant,
        reference_wpm=reference_wpm,
    )


def _load_prepared_manifest_payload(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.exists():
        raise DatasetPreparationError(
            f"Prepared dataset manifest does not exist: {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DatasetPreparationError(
            "Prepared dataset manifest must contain an object."
        )
    return payload


def _resolve_manifest_samples_path(
    payload: dict[str, Any],
    *,
    manifest_path: Path,
) -> Path | None:
    raw_samples_path = _coerce_text(payload.get("samples_path"))
    if not raw_samples_path:
        return None
    candidate = Path(raw_samples_path)
    if not candidate.is_absolute():
        candidate = (manifest_path.parent / candidate).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def iter_prepared_dataset_samples(
    manifest_path: Path,
    *,
    max_samples: int | None = None,
) -> Iterator[PreparedBenchmarkSample]:
    """Iterate prepared SPoRC samples without materializing the full manifest."""
    resolved_manifest_path = manifest_path.resolve()
    payload = _load_prepared_manifest_payload(resolved_manifest_path)
    raw_samples = payload.get("samples")
    samples_path = _resolve_manifest_samples_path(
        payload,
        manifest_path=resolved_manifest_path,
    )
    emitted = 0
    limit = None if max_samples is None or int(max_samples) <= 0 else int(max_samples)
    if isinstance(raw_samples, list):
        for raw_sample in raw_samples:
            if limit is not None and emitted >= limit:
                break
            if not isinstance(raw_sample, dict):
                continue
            yield _parse_prepared_sample(raw_sample)
            emitted += 1
        return
    if samples_path is None or not samples_path.exists():
        raise DatasetPreparationError(
            "Prepared dataset manifest must contain inline samples or a valid samples_path."
        )
    with gzip.open(samples_path, "rt", encoding="utf-8") as handle:
        for raw_line in handle:
            if limit is not None and emitted >= limit:
                break
            line = raw_line.strip()
            if not line:
                continue
            payload_line = json.loads(line)
            if not isinstance(payload_line, dict):
                continue
            yield _parse_prepared_sample(payload_line)
            emitted += 1


def load_prepared_dataset_manifest(
    manifest_path: Path,
    *,
    include_samples: bool = True,
    max_samples: int | None = None,
) -> PreparedDatasetManifest:
    """Load a previously prepared SPoRC dataset manifest from disk."""
    resolved_manifest_path = manifest_path.resolve()
    payload = _load_prepared_manifest_payload(resolved_manifest_path)
    raw_samples = payload.get("samples")
    samples_path = _resolve_manifest_samples_path(
        payload,
        manifest_path=resolved_manifest_path,
    )
    if not isinstance(raw_samples, list) and samples_path is None:
        raise DatasetPreparationError(
            "Prepared dataset manifest must contain inline samples or a valid samples_path."
        )

    samples: tuple[PreparedBenchmarkSample, ...] = ()
    if include_samples:
        samples = tuple(
            iter_prepared_dataset_samples(
                resolved_manifest_path,
                max_samples=max_samples,
            )
        )
    resolved_sample_count = payload.get("sample_count")
    if resolved_sample_count is None:
        if isinstance(raw_samples, list):
            resolved_sample_count = len(raw_samples)
        else:
            resolved_sample_count = len(samples)

    return PreparedDatasetManifest(
        manifest_path=resolved_manifest_path,
        dataset_id=_coerce_text(payload.get("dataset_id")) or "sporc_sample",
        sporc_dataset_id=_coerce_text(payload.get("sporc_dataset_id"))
        or SPORC_DATASET_ID,
        dataset_variant=_coerce_text(payload.get("dataset_variant"))
        or DEFAULT_DATASET_VARIANT,
        episode_data_path=Path(
            _coerce_text(payload.get("episode_data_path"))
        ).resolve(),
        turn_data_path=Path(_coerce_text(payload.get("turn_data_path"))).resolve(),
        sample_count=int(resolved_sample_count),
        samples=samples,
        samples_path=samples_path,
    )
