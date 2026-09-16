"""Measure speaking rate and pause allocation for CARD, source, and Descript audio.

Reviewer question Q2 on the ICASSP submission asks for the measured speaking
rate of CARD outputs against the source and against Descript, because the
rendered-duration gate in Eq. (3) constrains only total waveform length. Six of
the fifteen retained listeners described CARD speech as rushed or sped up, so
the open question is whether CARD meets its duration target partly by raising
rate or by shortening pauses.

Descript preserves selected spans of the original speaker recordings, so it
provides an edit-only comparison for CARD's synthesized output. The analysis
does not assume that Descript leaves the source pause distribution unchanged.

Two quantities are reported per artifact, following the distinction standard in
speech-rate work [1]:

- speech rate: words divided by total waveform duration.
- articulation rate: words divided by voiced duration, excluding silence.

One minus voiced duration divided by total duration is the pause fraction. A
system can hold articulation rate fixed and still sound hurried by removing
pauses, and the two measures separate those effects.

Silence is detected with the ffmpeg ``silencedetect`` filter at the
project-specific threshold recorded in the JSON output. The script needs ffmpeg
and ffprobe on PATH and no Python audio dependency.

The full comparison requires five episode-matched CARD, source, and Descript
artifacts. It rejects an input set when a CARD and Descript waveform for the
same episode have the same SHA-256 digest.

Current Literature and Gaps:
    Speech-rate work supports separating pause-inclusive rate from rate over
    detected speech [1], but it does not define a data-integrity rule for an
    edited reference whose media failed to render. The 30-second contiguous-
    silence limit is therefore a conservative project-specific export check
    used only to stop an incomplete five-minute reference from entering the
    outcome statistics. It is not a general model of pausing behavior.

Examples:
    Source rate only, from the transcript slices committed in this repository,
    which needs no audio and no participant folders::

        python scripts/analyze_speaking_rate.py --source-from-slices \\
            --out-dir artifacts/speaking_rate

    Full comparison, once both external artifact sets are available::

        python scripts/analyze_speaking_rate.py \\
            --source-from-slices \\
            --participant-audio-root "D:/Participant Audio v3" \\
            --descript-root "D:/Descript 5m references" \\
            --out-dir artifacts/speaking_rate

References:
    [1] N. H. de Jong and T. Wempe, "Praat script to detect syllable nuclei
    and measure speech rate automatically," Behavior Research Methods, vol. 41,
    no. 2, pp. 385-390, 2009, doi: 10.3758/BRM.41.2.385.
    [2] Student, "The probable error of a mean," Biometrika, vol. 6,
    no. 1, pp. 1-25, Mar. 1908, doi: 10.2307/2331554.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLICE_DIR = (
    REPO_ROOT / "datasets" / "sporc" / "human_eval_assets_10m" / "transcript_slices"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "speaking_rate"

FOLDER_PATTERN = re.compile(r"^Podcast_(?P<podcast>\d{2})_(?P<duration>30s|1m|5m)$")
DESCRIPT_FOLDER_PATTERN = re.compile(r"^Podcast_(?P<podcast>\d{2})_5m$")
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus"}
EXPECTED_EPISODE_IDS = tuple(f"Podcast {number:02d}" for number in range(1, 6))
EXPECTED_PARTICIPANT_FOLDERS = {
    (episode_id, duration)
    for episode_id in EXPECTED_EPISODE_IDS
    for duration in ("30s", "1m", "5m")
}
SLICE_MEASURED_PATTERN = re.compile(r"measured\s+(?P<seconds>\d+(?:\.\d+)?)s")
# Matches "SPEAKER_00:" from CARD exports and "Jalee James:" from a Descript
# export, optionally preceded by a bracketed label. Capped at three capitalized
# words so an ordinary sentence opening is not mistaken for a speaker turn.
SPEAKER_TAG_PATTERN = re.compile(
    r"^[ \t]*(?:\[[^\]]+\][ \t]*)?"
    r"(?:SPEAKER_\d+|[A-Z][\w.'-]*(?:[ \t]+[A-Z][\w.'-]*){0,2})[ \t]*:[ \t]*",
    re.MULTILINE,
)
SILENCE_START_PATTERN = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
SILENCE_END_PATTERN = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")

# Project-specific operational threshold; it is recorded in every JSON result.
SILENCE_NOISE_DB = -35.0
SILENCE_MIN_SECONDS = 0.25
MAX_DESCRIPT_CONTIGUOUS_SILENCE_SECONDS = 30.0
T_CRITICAL_95_BY_DF = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
}
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class RateRow:
    """Store one traceable speaking-rate measurement.

    Attributes:
        system: Origin of the measured artifact, such as ``card`` or
            ``descript``.
        podcast_id: Episode identifier shared by matched artifacts.
        duration_condition: Requested CARD duration or source condition.
        words: Spoken-word count after markup removal.
        total_seconds: Container duration reported by ffprobe.
        voiced_seconds: Duration remaining after detected silence is removed,
            or ``None`` when no waveform was available.
        speech_rate_wps: Words divided by total waveform duration.
        articulation_rate_wps: Words divided by voiced duration, or ``None``
            when no waveform was available.
        pause_fraction: Detected silence divided by total duration, or ``None``
            when no waveform was available.
        longest_silence_seconds: Longest detected silence event, or ``None``
            when only transcript slices were available.
        audio_file: Input audio basename, or ``None`` for transcript slices.
        transcript_file: Input transcript basename.
        audio_sha256: Input audio digest, or ``None`` for transcript slices.
    """

    system: str
    podcast_id: str
    duration_condition: str
    words: int
    total_seconds: float
    voiced_seconds: float | None
    speech_rate_wps: float
    articulation_rate_wps: float | None
    pause_fraction: float | None
    longest_silence_seconds: float | None
    audio_file: str | None
    transcript_file: str
    audio_sha256: str | None


def count_words(text: str) -> int:
    """Count spoken tokens after removing transcript markup.

    XML-like CARD speaker and interjection tags are removed while their spoken
    contents remain. Speaker labels ending in a colon and bracketed or
    parenthetical stage markers are also excluded.

    Args:
        text: Transcript text whose spoken tokens should be counted.

    Returns:
        Number of alphanumeric or apostrophe-bearing spoken tokens.

    Example:
        ``count_words("<SPEAKER_00>Hello, we're here.</SPEAKER_00>")`` returns
        ``3``.
    """
    stripped = re.sub(r"<[^>]+>", " ", text)
    stripped = SPEAKER_TAG_PATTERN.sub(" ", stripped)
    stripped = re.sub(r"\[[^\]]*\]", " ", stripped)
    stripped = re.sub(r"\([^)]*\)", " ", stripped)
    return len([token for token in re.findall(r"[A-Za-z0-9']+", stripped)])


def file_sha256(path: Path) -> str:
    """Compute a SHA-256 digest for one local file.

    Args:
        path: File to read in binary chunks.

    Returns:
        Uppercase hexadecimal SHA-256 digest.

    Raises:
        OSError: If the file cannot be read completely.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def probe_duration_seconds(path: Path) -> float:
    """Read an audio container's duration with ffprobe.

    Args:
        path: Audio file accepted by the local ffprobe installation.

    Returns:
        Positive container duration in seconds.

    Raises:
        RuntimeError: If ffprobe fails, reports an unparseable value, or
            reports a non-positive duration.
    """
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr.strip()}")
    try:
        duration = float(completed.stdout.strip())
    except ValueError as error:
        raise RuntimeError(f"Unparseable ffprobe duration for {path}.") from error
    if duration <= 0:
        raise RuntimeError(f"Non-positive ffprobe duration for {path}: {duration}")
    return duration


def measure_silence(path: Path, total_seconds: float) -> tuple[float, float]:
    """Measure voiced duration and the longest detected silence.

    Uses ffmpeg ``silencedetect``. A trailing ``silence_start`` with no matching
    ``silence_end`` means the file ends in silence, which is closed at the file
    duration. The silence settings are project-specific operational choices and
    are written into the summary output.

    Args:
        path: Audio file accepted by the local ffmpeg installation.
        total_seconds: Positive container duration from ffprobe.

    Returns:
        Pair of positive voiced duration and longest silence duration in
        seconds. The longest silence is zero when no event is detected.

    Raises:
        RuntimeError: If ffmpeg cannot read the file, emits unbalanced silence
            events, or leaves no positive voiced duration.
    """
    completed = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            str(path),
            "-af",
            f"silencedetect=noise={SILENCE_NOISE_DB}dB:d={SILENCE_MIN_SECONDS}",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {path}: {completed.stderr.strip()[-400:]}"
        )
    log = completed.stderr
    starts = [float(value) for value in SILENCE_START_PATTERN.findall(log)]
    ends = [float(value) for value in SILENCE_END_PATTERN.findall(log)]
    if len(starts) == len(ends) + 1:
        ends.append(total_seconds)
    if len(starts) != len(ends):
        raise RuntimeError(f"Unbalanced silencedetect output for {path}.")
    silence_durations = [
        max(0.0, min(end, total_seconds) - max(0.0, start))
        for start, end in zip(starts, ends, strict=True)
    ]
    voiced_seconds = total_seconds - sum(silence_durations)
    if voiced_seconds <= 0:
        raise RuntimeError(f"No positive voiced duration remained for {path}.")
    longest_silence_seconds = max(silence_durations, default=0.0)
    return voiced_seconds, longest_silence_seconds


def build_row(
    *,
    system: str,
    podcast_id: str,
    duration_condition: str,
    words: int,
    total_seconds: float,
    voiced_seconds: float,
    longest_silence_seconds: float,
    audio_file: str,
    transcript_file: str,
    audio_sha256: str,
) -> RateRow:
    """Build one validated measurement row.

    Args:
        system: Artifact origin recorded in the output.
        podcast_id: Episode identifier used for paired comparisons.
        duration_condition: Requested CARD duration or source condition.
        words: Positive spoken-word count.
        total_seconds: Positive total waveform or retained-span duration.
        voiced_seconds: Positive voiced duration no greater than total duration.
        longest_silence_seconds: Longest detected silence.
        audio_file: Input audio basename.
        transcript_file: Input transcript basename.
        audio_sha256: Input audio digest.

    Returns:
        Rounded rates and provenance for the validated artifact.

    Raises:
        ValueError: If the word count or durations cannot define valid rates.
    """
    if words <= 0:
        raise ValueError(f"Word count must be positive for {transcript_file}.")
    if total_seconds <= 0:
        raise ValueError(f"Total duration must be positive for {transcript_file}.")
    if voiced_seconds <= 0 or voiced_seconds > total_seconds:
        raise ValueError(
            f"Voiced duration must be in (0, total] for {transcript_file}: "
            f"{voiced_seconds} of {total_seconds} seconds."
        )
    if not 0 <= longest_silence_seconds <= total_seconds:
        raise ValueError(
            f"Longest silence must be in [0, total] for {transcript_file}: "
            f"{longest_silence_seconds} of {total_seconds} seconds."
        )
    speech_rate = words / total_seconds
    articulation_rate = words / voiced_seconds
    pause_fraction = 1.0 - (voiced_seconds / total_seconds)
    return RateRow(
        system=system,
        podcast_id=podcast_id,
        duration_condition=duration_condition,
        words=words,
        total_seconds=round(total_seconds, 3),
        voiced_seconds=round(voiced_seconds, 3),
        speech_rate_wps=round(speech_rate, 4),
        articulation_rate_wps=round(articulation_rate, 4),
        pause_fraction=round(pause_fraction, 4),
        longest_silence_seconds=round(longest_silence_seconds, 3),
        audio_file=audio_file,
        transcript_file=transcript_file,
        audio_sha256=audio_sha256,
    )


def build_transcript_slice_row(
    *,
    podcast_id: str,
    words: int,
    total_seconds: float,
    transcript_file: str,
) -> RateRow:
    """Build a source-slice row without inventing waveform-derived metrics.

    Args:
        podcast_id: Episode identifier used in grouped summaries.
        words: Positive spoken-word count across the retained spans.
        total_seconds: Positive sum of recorded span durations.
        transcript_file: Input transcript-slice basename.

    Returns:
        Source speech-rate row with waveform-dependent fields set to ``None``.

    Raises:
        ValueError: If the word count or retained duration is non-positive.
    """
    if words <= 0:
        raise ValueError(f"Word count must be positive for {transcript_file}.")
    if total_seconds <= 0:
        raise ValueError(f"Total duration must be positive for {transcript_file}.")
    return RateRow(
        system="source_slices",
        podcast_id=podcast_id,
        duration_condition="10m_source",
        words=words,
        total_seconds=round(total_seconds, 3),
        voiced_seconds=None,
        speech_rate_wps=round(words / total_seconds, 4),
        articulation_rate_wps=None,
        pause_fraction=None,
        longest_silence_seconds=None,
        audio_file=None,
        transcript_file=transcript_file,
        audio_sha256=None,
    )


def select_single_file(
    folder: Path,
    pattern: str,
    label: str,
    *,
    audio_only: bool = False,
) -> Path:
    """Select exactly one file matching an expected artifact role.

    Args:
        folder: Directory containing one episode-condition artifact set.
        pattern: Glob pattern that identifies the requested role.
        label: Human-readable role used in validation errors.
        audio_only: Restrict matches to supported audio suffixes.

    Returns:
        The sole matching file.

    Raises:
        RuntimeError: If zero or multiple files match the requested role.
    """
    matches = sorted(path for path in folder.glob(pattern) if path.is_file())
    if audio_only:
        matches = [path for path in matches if path.suffix.lower() in AUDIO_SUFFIXES]
    if len(matches) != 1:
        names = [path.name for path in matches]
        raise RuntimeError(
            f"Expected one {label} in {folder}, found {len(matches)}: {names}"
        )
    return matches[0]


def measure_artifact(
    *,
    system: str,
    podcast_id: str,
    duration_condition: str,
    transcript_path: Path,
    audio_path: Path,
) -> RateRow:
    """Measure one transcript-audio pair and retain its provenance.

    Args:
        system: Artifact origin recorded in the output.
        podcast_id: Episode identifier used for paired comparisons.
        duration_condition: Requested CARD duration or source condition.
        transcript_path: UTF-8 transcript corresponding to ``audio_path``.
        audio_path: Audio waveform corresponding to ``transcript_path``.

    Returns:
        Validated speaking-rate row for the pair.

    Raises:
        UnicodeDecodeError: If the transcript is not valid UTF-8.
        RuntimeError: If ffprobe or ffmpeg cannot measure the audio, or a
            Descript export exceeds the project-specific contiguous-silence
            integrity limit.
        ValueError: If the measured values cannot define valid rates.
    """
    transcript = transcript_path.read_text(encoding="utf-8")
    total_seconds = probe_duration_seconds(audio_path)
    voiced_seconds, longest_silence_seconds = measure_silence(audio_path, total_seconds)
    if (
        system == "descript"
        and longest_silence_seconds > MAX_DESCRIPT_CONTIGUOUS_SILENCE_SECONDS
    ):
        raise RuntimeError(
            f"Incomplete Descript export {audio_path}: longest detected silence "
            f"is {longest_silence_seconds:.3f} seconds, exceeding the "
            f"{MAX_DESCRIPT_CONTIGUOUS_SILENCE_SECONDS:.1f}-second integrity limit."
        )
    return build_row(
        system=system,
        podcast_id=podcast_id,
        duration_condition=duration_condition,
        words=count_words(transcript),
        total_seconds=total_seconds,
        voiced_seconds=voiced_seconds,
        longest_silence_seconds=longest_silence_seconds,
        audio_file=audio_path.name,
        transcript_file=transcript_path.name,
        audio_sha256=file_sha256(audio_path),
    )


def rows_from_transcript_slices(slice_dir: Path) -> list[RateRow]:
    """Measure source speech rate from the five committed transcript slices.

    Each slice file records the measured seconds of every retained span, so the
    source rate is recoverable without the 10-minute audio. Only speech rate is
    available this way. Articulation rate and pause fraction remain unavailable
    because silence detection requires the waveform.

    Args:
        slice_dir: Directory containing one ``transcript slices.txt`` file for
            each expected episode.

    Returns:
        Five rows ordered by episode.

    Raises:
        FileNotFoundError: If ``slice_dir`` is missing.
        RuntimeError: If the expected five episode files or measured spans are
            incomplete.
        UnicodeDecodeError: If a slice file is not valid UTF-8.
    """
    if not slice_dir.is_dir():
        raise FileNotFoundError(f"Transcript slice directory not found: {slice_dir}")
    slice_files = sorted(slice_dir.glob("*transcript slices.txt"))
    if len(slice_files) != len(EXPECTED_EPISODE_IDS):
        raise RuntimeError(
            f"Expected {len(EXPECTED_EPISODE_IDS)} transcript slice files in "
            f"{slice_dir}, found {len(slice_files)}."
        )
    rows: list[RateRow] = []
    for path in slice_files:
        order_match = re.match(r"(\d{2})\s*-", path.name)
        if order_match is None:
            raise RuntimeError(f"Cannot parse episode number from {path.name}.")
        podcast_id = f"Podcast {int(order_match.group(1)):02d}"
        text = path.read_text(encoding="utf-8")
        blocks = re.split(r"\n(?=\d+\.\s+\[)", text)
        words = 0
        seconds = 0.0
        measured_blocks = 0
        for block in blocks:
            measured = SLICE_MEASURED_PATTERN.search(block)
            if measured is None:
                continue
            measured_blocks += 1
            spoken = re.split(r"\nSource:", block)[0]
            spoken = re.sub(r"^\s*\d+\.\s*", " ", spoken)
            words += count_words(spoken)
            seconds += float(measured.group("seconds"))
        if measured_blocks == 0:
            raise RuntimeError(f"No measured transcript spans found in {path}.")
        rows.append(
            build_transcript_slice_row(
                podcast_id=podcast_id,
                words=words,
                total_seconds=seconds,
                transcript_file=path.name,
            )
        )
    actual_ids = tuple(row.podcast_id for row in rows)
    if actual_ids != EXPECTED_EPISODE_IDS:
        raise RuntimeError(
            f"Expected slice episodes {EXPECTED_EPISODE_IDS}, found {actual_ids}."
        )
    return rows


def rows_from_participant_audio(root: Path) -> list[RateRow]:
    """Measure CARD summaries and one source artifact per episode.

    Expects the ``Podcast_XX_<duration>`` layout used by
    ``scripts/run_participant_audio_gemini_judge.py``: one ``[SOURCE]...10m audio
    slice`` pair and one ``[SUMMARY]...`` pair per folder. CARD is measured at
    all three duration conditions. Source audio is measured from each five-minute
    folder so the repeated copy in the shorter-condition folders is not counted
    more than once.

    Args:
        root: Participant Audio v3 directory containing all fifteen expected
            episode-condition folders.

    Returns:
        Twenty rows: fifteen CARD outputs and five episode source artifacts.

    Raises:
        FileNotFoundError: If ``root`` is missing.
        RuntimeError: If the folder matrix or any transcript-audio pair is
            incomplete or ambiguous.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Participant audio root not found: {root}")
    folders: dict[tuple[str, str], Path] = {}
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        match = FOLDER_PATTERN.fullmatch(folder.name)
        if match is None:
            continue
        episode_id = f"Podcast {int(match.group('podcast')):02d}"
        folders[(episode_id, match.group("duration"))] = folder

    actual_folders = set(folders)
    if actual_folders != EXPECTED_PARTICIPANT_FOLDERS:
        missing = sorted(EXPECTED_PARTICIPANT_FOLDERS - actual_folders)
        unexpected = sorted(actual_folders - EXPECTED_PARTICIPANT_FOLDERS)
        raise RuntimeError(
            f"Participant folder matrix mismatch. Missing: {missing}; "
            f"unexpected: {unexpected}."
        )

    rows: list[RateRow] = []
    for (podcast_id, duration), folder in sorted(folders.items()):
        summary_transcript = select_single_file(
            folder,
            "[[]SUMMARY[]]* - summary.txt",
            "CARD summary transcript",
        )
        summary_audio = select_single_file(
            folder,
            "[[]SUMMARY[]]*",
            "CARD summary audio",
            audio_only=True,
        )
        rows.append(
            measure_artifact(
                system="card",
                podcast_id=podcast_id,
                duration_condition=duration,
                transcript_path=summary_transcript,
                audio_path=summary_audio,
            )
        )
        if duration != "5m":
            continue
        source_transcript = select_single_file(
            folder,
            "[[]SOURCE[]]* - 10m audio slice transcript.txt",
            "source transcript",
        )
        source_audio = select_single_file(
            folder,
            "[[]SOURCE[]]* - 10m audio slice.*",
            "source audio",
            audio_only=True,
        )
        rows.append(
            measure_artifact(
                system="source_audio",
                podcast_id=podcast_id,
                duration_condition="10m_source",
                transcript_path=source_transcript,
                audio_path=source_audio,
            )
        )
    return rows


def rows_from_descript(root: Path) -> list[RateRow]:
    """Measure the five episode-matched Descript references.

    Expects ``Podcast_XX_5m`` subfolders, each holding exactly one ``[SUMMARY]``
    WAV and its ``summary transcript.txt`` export as described in
    ``datasets/sporc/human_eval_assets_10m/descript_5m_sota_reference_prompts.md``.

    Args:
        root: Directory containing the five Descript episode folders.

    Returns:
        Five Descript rows ordered by episode.

    Raises:
        FileNotFoundError: If ``root`` is missing.
        RuntimeError: If the episode set or any transcript-audio pair is
            incomplete or ambiguous.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Descript root not found: {root}")
    folders: dict[str, Path] = {}
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        match = DESCRIPT_FOLDER_PATTERN.fullmatch(folder.name)
        if match is None:
            continue
        podcast_id = f"Podcast {int(match.group('podcast')):02d}"
        folders[podcast_id] = folder
    actual_ids = set(folders)
    expected_ids = set(EXPECTED_EPISODE_IDS)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise RuntimeError(
            f"Descript episode set mismatch. Missing: {missing}; "
            f"unexpected: {unexpected}."
        )

    rows: list[RateRow] = []
    for podcast_id, folder in sorted(folders.items()):
        transcript = select_single_file(
            folder,
            "[[]SUMMARY[]]* - summary transcript.txt",
            "Descript summary transcript",
        )
        audio = select_single_file(
            folder,
            "[[]SUMMARY[]]*.wav",
            "Descript summary WAV",
            audio_only=True,
        )
        rows.append(
            measure_artifact(
                system="descript",
                podcast_id=podcast_id,
                duration_condition="5m",
                transcript_path=transcript,
                audio_path=audio,
            )
        )
    return rows


def index_comparison_rows(
    rows: list[RateRow], system: str, duration_condition: str
) -> dict[str, RateRow]:
    """Index one complete comparison group by episode.

    Args:
        rows: Measurements collected during the full run.
        system: Requested artifact origin.
        duration_condition: Requested duration condition.

    Returns:
        Mapping containing exactly the five expected episode rows.

    Raises:
        RuntimeError: If rows are duplicated or the expected episode set is
            incomplete.
    """
    selected = [
        row
        for row in rows
        if row.system == system and row.duration_condition == duration_condition
    ]
    indexed = {row.podcast_id: row for row in selected}
    if len(indexed) != len(selected):
        raise RuntimeError(
            f"Duplicate {system}/{duration_condition} rows prevent pairing."
        )
    actual_ids = set(indexed)
    expected_ids = set(EXPECTED_EPISODE_IDS)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        raise RuntimeError(
            f"Incomplete {system}/{duration_condition} rows. Missing: {missing}; "
            f"unexpected: {unexpected}."
        )
    return indexed


def validate_distinct_card_descript_audio(rows: list[RateRow]) -> None:
    """Reject episode pairs that reuse one waveform under two system labels.

    Args:
        rows: Full-run measurements containing CARD and Descript five-minute
            rows with SHA-256 provenance.

    Raises:
        RuntimeError: If a matched CARD and Descript pair has the same audio
            digest or lacks a digest.
    """
    card_rows = index_comparison_rows(rows, "card", "5m")
    descript_rows = index_comparison_rows(rows, "descript", "5m")
    duplicate_ids: list[str] = []
    for podcast_id in EXPECTED_EPISODE_IDS:
        card_hash = card_rows[podcast_id].audio_sha256
        descript_hash = descript_rows[podcast_id].audio_sha256
        if card_hash is None or descript_hash is None:
            raise RuntimeError(f"Missing audio digest for {podcast_id}.")
        if card_hash == descript_hash:
            duplicate_ids.append(podcast_id)
    if duplicate_ids:
        raise RuntimeError(
            "CARD and Descript audio are byte-identical for: "
            f"{', '.join(duplicate_ids)}. Supply the actual Descript exports."
        )


def paired_difference(
    rows: list[RateRow],
    left: tuple[str, str],
    right: tuple[str, str],
    field: str,
) -> dict[str, float]:
    """Compute an episode-paired mean difference and 95% t interval.

    Args:
        rows: Full-run measurement rows.
        left: ``(system, duration_condition)`` for the minuend.
        right: ``(system, duration_condition)`` for the subtrahend.
        field: Numeric :class:`RateRow` attribute to compare.

    Returns:
        Sample size, mean paired difference, and two-sided 95% interval.

    Raises:
        KeyError: If the sample size exceeds the tabulated critical values.
        RuntimeError: If either comparison group is incomplete, duplicated, or
            lacks the requested measurement.
    """
    left_by_episode = index_comparison_rows(rows, *left)
    right_by_episode = index_comparison_rows(rows, *right)
    diffs: list[float] = []
    for key in EXPECTED_EPISODE_IDS:
        left_value = getattr(left_by_episode[key], field)
        right_value = getattr(right_by_episode[key], field)
        if left_value is None or right_value is None:
            raise RuntimeError(
                f"Cannot compare {field} for {key}: a measurement is unavailable."
            )
        diffs.append(left_value - right_value)
    mean = statistics.mean(diffs)
    stderr = statistics.stdev(diffs) / math.sqrt(len(diffs))
    # IEEE citation: paired-mean uncertainty follows Student's t method [2].
    critical = T_CRITICAL_95_BY_DF[len(diffs) - 1]
    return {
        "n": len(diffs),
        "mean": round(mean, 4),
        "ci_low": round(mean - critical * stderr, 4),
        "ci_high": round(mean + critical * stderr, 4),
    }


def mean_available(group: list[RateRow], field: str) -> float | None:
    """Average one measurement while preserving unavailable waveform values.

    A group may have every value available or every value unavailable. Mixed
    availability signals incomplete measurement and is rejected so the group
    mean cannot silently represent a changing denominator.

    Args:
        group: Rows belonging to one system and duration condition.
        field: Numeric :class:`RateRow` attribute to average.

    Returns:
        Rounded arithmetic mean when all rows contain the measurement, or
        ``None`` when no row contains it.

    Raises:
        RuntimeError: If only part of the group contains the requested
            measurement.

    Example:
        ``mean_available(source_slice_rows, "pause_fraction")`` returns
        ``None`` because transcript slices do not include waveforms.
    """
    values = [getattr(item, field) for item in group]
    available = [value for value in values if value is not None]
    if not available:
        return None
    if len(available) != len(values):
        raise RuntimeError(
            f"Incomplete {field}: {len(available)} of {len(values)} values available."
        )
    return round(statistics.mean(available), 4)


def summarize(rows: list[RateRow]) -> dict[str, object]:
    """Aggregate means and add strict paired contrasts for a full run.

    Args:
        rows: Measurements collected from the selected input mode.

    Returns:
        JSON-serializable settings, grouped means, and paired contrasts. A
        source-slices-only run has no paired contrasts.

    Raises:
        RuntimeError: If a full comparison group is incomplete or duplicated.
    """
    grouped: dict[str, list[RateRow]] = {}
    for row in rows:
        grouped.setdefault(f"{row.system}/{row.duration_condition}", []).append(row)
    means = {
        key: {
            "n": len(group),
            "speech_rate_wps": round(
                statistics.mean(item.speech_rate_wps for item in group), 4
            ),
            "articulation_rate_wps": mean_available(group, "articulation_rate_wps"),
            "pause_fraction": mean_available(group, "pause_fraction"),
        }
        for key, group in sorted(grouped.items())
    }
    systems = {row.system for row in rows}
    contrasts: dict[str, dict[str, float]] = {}
    if {"card", "source_audio", "descript"}.issubset(systems):
        contrasts = {
            "card_minus_source_speech_rate": paired_difference(
                rows,
                ("card", "5m"),
                ("source_audio", "10m_source"),
                "speech_rate_wps",
            ),
            "card_minus_descript_speech_rate": paired_difference(
                rows,
                ("card", "5m"),
                ("descript", "5m"),
                "speech_rate_wps",
            ),
            "card_minus_source_articulation": paired_difference(
                rows,
                ("card", "5m"),
                ("source_audio", "10m_source"),
                "articulation_rate_wps",
            ),
            "card_minus_descript_articulation": paired_difference(
                rows,
                ("card", "5m"),
                ("descript", "5m"),
                "articulation_rate_wps",
            ),
            "card_minus_source_pause_fraction": paired_difference(
                rows,
                ("card", "5m"),
                ("source_audio", "10m_source"),
                "pause_fraction",
            ),
            "card_minus_descript_pause_fraction": paired_difference(
                rows,
                ("card", "5m"),
                ("descript", "5m"),
                "pause_fraction",
            ),
        }
    return {
        "analysis_settings": {
            "silence_noise_db": SILENCE_NOISE_DB,
            "silence_min_seconds": SILENCE_MIN_SECONDS,
            "max_descript_contiguous_silence_seconds": (
                MAX_DESCRIPT_CONTIGUOUS_SILENCE_SECONDS
            ),
            "interval": "paired two-sided 95% Student t",
            "expected_episodes": list(EXPECTED_EPISODE_IDS),
        },
        "means_by_system_condition": means,
        "paired_contrasts_5m": contrasts,
    }


def parse_args() -> argparse.Namespace:
    """Parse and validate the supported analysis modes.

    Returns:
        Namespace containing the selected input roots and output directory.

    Raises:
        SystemExit: If no input mode is selected or only one side of the full
            participant-versus-Descript comparison is supplied.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--participant-audio-root", type=Path, default=None)
    parser.add_argument("--descript-root", type=Path, default=None)
    parser.add_argument(
        "--source-from-slices",
        action="store_true",
        help="Add source rates from the committed transcript slices.",
    )
    parser.add_argument("--slice-dir", type=Path, default=DEFAULT_SLICE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    artifact_roots = (
        args.participant_audio_root is not None,
        args.descript_root is not None,
    )
    if artifact_roots.count(True) == 1:
        parser.error(
            "--participant-audio-root and --descript-root must be supplied together."
        )
    if not args.source_from_slices and not all(artifact_roots):
        parser.error(
            "Select --source-from-slices, or supply both external artifact roots."
        )
    return args


def main() -> None:
    """Run the selected analysis and write traceable CSV and JSON results.

    The source-slices mode can run alone. External artifact analysis requires
    both Participant Audio v3 and the five Descript exports; before any result
    is written, the run verifies that matched CARD and Descript WAVs are not
    byte-identical.

    Raises:
        FileNotFoundError: If a requested input directory is missing.
        RuntimeError: If inputs are incomplete, ambiguous, contaminated, or
            cannot be measured by ffmpeg and ffprobe.
        OSError: If output files cannot be created or written.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    rows: list[RateRow] = []
    if args.source_from_slices:
        LOGGER.info("Reading transcript slices from %s", args.slice_dir)
        rows.extend(rows_from_transcript_slices(args.slice_dir))
    if args.participant_audio_root is not None:
        LOGGER.info("Probing participant audio under %s", args.participant_audio_root)
        rows.extend(rows_from_participant_audio(args.participant_audio_root))
    if args.descript_root is not None:
        LOGGER.info("Probing Descript references under %s", args.descript_root)
        rows.extend(rows_from_descript(args.descript_root))
    if args.participant_audio_root is not None:
        validate_distinct_card_descript_audio(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "speaking_rate.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    summary = summarize(rows)
    json_path = args.out_dir / "speaking_rate_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info("Wrote %s (%d rows)", csv_path, len(rows))
    LOGGER.info("Wrote %s", json_path)
    means_by_system = cast(dict[str, object], summary["means_by_system_condition"])
    for key, value in means_by_system.items():
        LOGGER.info("  %-26s %s", key, value)


if __name__ == "__main__":
    main()
