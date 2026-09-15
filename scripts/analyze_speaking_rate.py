"""Measure speaking rate and pause allocation for CARD, source, and Descript audio.

Reviewer question Q2 on the ICASSP submission asks for the measured speaking
rate of CARD outputs against the source and against Descript, because the
rendered-duration gate in Eq. (3) constrains only total waveform length. Six of
the fifteen retained listeners described CARD speech as rushed or sped up, so
the open question is whether CARD meets its duration target partly by raising
rate or by shortening pauses.

Descript edits the original speaker audio rather than resynthesizing it, so its
articulation rate is the source rate by construction. That makes it a useful
control: a CARD-minus-source rate gap with no matching Descript-minus-source gap
localizes the effect in Card 3 synthesis rather than in content selection.

Two quantities are reported per artifact, following the distinction standard in
speech-rate work:

- speech rate: words divided by total waveform duration.
- articulation rate: words divided by voiced duration, excluding silence.

Their ratio is the pause fraction. A system can hold articulation rate fixed and
still sound hurried by removing pauses, and the two measures separate those.

Silence is detected with the ffmpeg ``silencedetect`` filter, so this script
needs ffmpeg and ffprobe on PATH and no Python audio dependency.

Examples:
    Source rate only, from the transcript slices committed in this repository,
    which needs no audio and no participant folders::

        python scripts/analyze_speaking_rate.py --source-from-slices \\
            --out-dir artifacts/speaking_rate

    Full comparison, once the participant-audio folders are available::

        python scripts/analyze_speaking_rate.py \\
            --participant-audio-root "D:/Participant Audio v3" \\
            --descript-root "D:/Descript 5m references" \\
            --out-dir artifacts/speaking_rate
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLICE_DIR = (
    REPO_ROOT / "datasets" / "sporc" / "human_eval_assets_10m" / "transcript_slices"
)
DEFAULT_OUT_DIR = REPO_ROOT / "artifacts" / "speaking_rate"

FOLDER_PATTERN = re.compile(r"^Podcast_(?P<podcast>\d{2})_(?P<duration>30s|1m|5m)$")
AUDIO_SUFFIXES = {".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus"}
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

# -35 dBFS over 250 ms marks an inter-phrase pause without cutting stop closures.
SILENCE_NOISE_DB = -35.0
SILENCE_MIN_SECONDS = 0.25


@dataclass(slots=True, frozen=True)
class RateRow:
    """One measured artifact: system, episode, target, and its rate quantities."""

    system: str
    podcast_id: str
    duration_condition: str
    words: int
    total_seconds: float
    voiced_seconds: float
    speech_rate_wps: float
    articulation_rate_wps: float
    pause_fraction: float


def count_words(text: str) -> int:
    """Count spoken words, dropping speaker tags and bracketed stage markers."""
    stripped = SPEAKER_TAG_PATTERN.sub(" ", text)
    stripped = re.sub(r"\[[^\]]*\]", " ", stripped)
    stripped = re.sub(r"\([^)]*\)", " ", stripped)
    return len([token for token in re.findall(r"[A-Za-z0-9']+", stripped)])


def probe_duration_seconds(path: Path) -> float:
    """Return container duration in seconds via ffprobe.

    Raises:
        RuntimeError: If ffprobe fails or reports an unparseable duration.
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
        return float(completed.stdout.strip())
    except ValueError as error:
        raise RuntimeError(f"Unparseable ffprobe duration for {path}.") from error


def measure_voiced_seconds(path: Path, total_seconds: float) -> float:
    """Return voiced duration, total minus detected silence.

    Uses ffmpeg ``silencedetect``. A trailing ``silence_start`` with no matching
    ``silence_end`` means the file ends in silence, which is closed at the file
    duration.

    Raises:
        RuntimeError: If ffmpeg cannot read the file.
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
        raise RuntimeError(f"ffmpeg failed for {path}: {completed.stderr.strip()[-400:]}")
    log = completed.stderr
    starts = [float(value) for value in SILENCE_START_PATTERN.findall(log)]
    ends = [float(value) for value in SILENCE_END_PATTERN.findall(log)]
    if len(starts) == len(ends) + 1:
        ends.append(total_seconds)
    if len(starts) != len(ends):
        raise RuntimeError(f"Unbalanced silencedetect output for {path}.")
    silence = sum(
        max(0.0, min(end, total_seconds) - max(0.0, start))
        for start, end in zip(starts, ends, strict=True)
    )
    return max(0.0, total_seconds - silence)


def build_row(
    *,
    system: str,
    podcast_id: str,
    duration_condition: str,
    words: int,
    total_seconds: float,
    voiced_seconds: float,
) -> RateRow:
    """Assemble one row, deriving the rate quantities from words and durations."""
    speech_rate = words / total_seconds if total_seconds > 0 else 0.0
    articulation_rate = words / voiced_seconds if voiced_seconds > 0 else 0.0
    pause_fraction = (
        1.0 - (voiced_seconds / total_seconds) if total_seconds > 0 else 0.0
    )
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
    )


def rows_from_transcript_slices(slice_dir: Path) -> list[RateRow]:
    """Measure source speech rate from the committed transcript slices.

    Each slice file records the measured seconds of every retained span, so the
    source rate is recoverable without the 10-minute audio. Only the speech rate
    is available this way; voiced duration needs the waveform, and is reported as
    the full span duration, which makes the articulation rate a lower bound.

    Raises:
        FileNotFoundError: If the slice directory is missing.
    """
    if not slice_dir.is_dir():
        raise FileNotFoundError(f"Transcript slice directory not found: {slice_dir}")
    rows: list[RateRow] = []
    for path in sorted(slice_dir.glob("*transcript slices.txt")):
        order_match = re.match(r"(\d{2})\s*-", path.name)
        if order_match is None:
            continue
        podcast_id = f"Podcast {int(order_match.group(1)):02d}"
        text = path.read_text(encoding="utf-8", errors="replace")
        blocks = re.split(r"\n(?=\d+\.\s+\[)", text)
        words = 0
        seconds = 0.0
        for block in blocks:
            measured = SLICE_MEASURED_PATTERN.search(block)
            if measured is None:
                continue
            spoken = re.split(r"\nSource:", block)[0]
            spoken = re.sub(r"^\s*\d+\.\s*", " ", spoken)
            words += count_words(spoken)
            seconds += float(measured.group("seconds"))
        if seconds <= 0:
            continue
        rows.append(
            build_row(
                system="source_slices",
                podcast_id=podcast_id,
                duration_condition="10m_source",
                words=words,
                total_seconds=seconds,
                voiced_seconds=seconds,
            )
        )
    return rows


def rows_from_participant_audio(root: Path) -> list[RateRow]:
    """Measure CARD summary and 10-minute source audio per participant folder.

    Expects the ``Podcast_XX_<duration>`` layout used by
    ``scripts/run_participant_audio_gemini_judge.py``: one ``[SOURCE]...10m audio
    slice`` pair and one ``[SUMMARY]...`` pair per folder.

    Raises:
        FileNotFoundError: If the participant-audio root is missing.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Participant audio root not found: {root}")
    rows: list[RateRow] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        match = FOLDER_PATTERN.match(folder.name)
        if match is None:
            continue
        podcast_id = f"Podcast {int(match.group('podcast')):02d}"
        duration = match.group("duration")
        pairs = (
            ("card", "[[]SUMMARY[]]*summary.txt", "[[]SUMMARY[]]*"),
            (
                "source_audio",
                "[[]SOURCE[]]*10m audio slice transcript.txt",
                "[[]SOURCE[]]*10m audio slice.*",
            ),
        )
        for system, transcript_glob, audio_glob in pairs:
            transcripts = sorted(folder.glob(transcript_glob))
            audios = sorted(
                path
                for path in folder.glob(audio_glob)
                if path.suffix.lower() in AUDIO_SUFFIXES
            )
            if len(transcripts) != 1 or len(audios) != 1:
                print(
                    f"  skip {folder.name} [{system}]: "
                    f"{len(transcripts)} transcript(s), {len(audios)} audio file(s)"
                )
                continue
            total = probe_duration_seconds(audios[0])
            voiced = measure_voiced_seconds(audios[0], total)
            rows.append(
                build_row(
                    system=system,
                    podcast_id=podcast_id,
                    duration_condition=duration if system == "card" else "10m_source",
                    words=count_words(
                        transcripts[0].read_text(encoding="utf-8", errors="replace")
                    ),
                    total_seconds=total,
                    voiced_seconds=voiced,
                )
            )
    return rows


def rows_from_descript(root: Path) -> list[RateRow]:
    """Measure the Descript five-minute references.

    Expects one subfolder per episode whose name carries the episode number, each
    holding the exported WAV and the exported transcript described in
    ``datasets/sporc/human_eval_assets_10m/descript_5m_sota_reference_prompts.md``.

    Raises:
        FileNotFoundError: If the Descript root is missing.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Descript root not found: {root}")
    rows: list[RateRow] = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        number = re.search(r"(\d{1,2})", folder.name)
        if number is None:
            continue
        transcripts = sorted(folder.glob("*.txt"))
        audios = sorted(
            path for path in folder.iterdir() if path.suffix.lower() in AUDIO_SUFFIXES
        )
        if len(transcripts) != 1 or len(audios) != 1:
            print(
                f"  skip {folder.name} [descript]: "
                f"{len(transcripts)} transcript(s), {len(audios)} audio file(s)"
            )
            continue
        total = probe_duration_seconds(audios[0])
        voiced = measure_voiced_seconds(audios[0], total)
        rows.append(
            build_row(
                system="descript",
                podcast_id=f"Podcast {int(number.group(1)):02d}",
                duration_condition="5m",
                words=count_words(
                    transcripts[0].read_text(encoding="utf-8", errors="replace")
                ),
                total_seconds=total,
                voiced_seconds=voiced,
            )
        )
    return rows


def paired_difference(
    rows: list[RateRow], left: str, right: str, field: str
) -> dict[str, float] | None:
    """Return the episode-paired mean difference and its 95% t interval.

    Pairs ``left`` against ``right`` on ``podcast_id``. Returns ``None`` when
    fewer than two episodes are shared, which is when an interval is meaningless.
    """
    left_by_episode = {row.podcast_id: row for row in rows if row.system == left}
    right_by_episode = {row.podcast_id: row for row in rows if row.system == right}
    shared = sorted(set(left_by_episode) & set(right_by_episode))
    if len(shared) < 2:
        return None
    diffs = [
        getattr(left_by_episode[key], field) - getattr(right_by_episode[key], field)
        for key in shared
    ]
    mean = statistics.mean(diffs)
    if len(diffs) < 3:
        return {"n": len(diffs), "mean": round(mean, 4)}
    stderr = statistics.stdev(diffs) / math.sqrt(len(diffs))
    # Two-sided 95% t critical values for small n, so scipy is not required.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}.get(
        len(diffs) - 1, 2.262
    )
    return {
        "n": len(diffs),
        "mean": round(mean, 4),
        "ci_low": round(mean - critical * stderr, 4),
        "ci_high": round(mean + critical * stderr, 4),
    }


def summarize(rows: list[RateRow]) -> dict[str, object]:
    """Group rows by system and condition and add the paired CARD contrasts."""
    grouped: dict[str, list[RateRow]] = {}
    for row in rows:
        grouped.setdefault(f"{row.system}/{row.duration_condition}", []).append(row)
    means = {
        key: {
            "n": len(group),
            "speech_rate_wps": round(
                statistics.mean(item.speech_rate_wps for item in group), 4
            ),
            "articulation_rate_wps": round(
                statistics.mean(item.articulation_rate_wps for item in group), 4
            ),
            "pause_fraction": round(
                statistics.mean(item.pause_fraction for item in group), 4
            ),
        }
        for key, group in sorted(grouped.items())
    }
    five_minute = [
        row for row in rows if row.duration_condition in {"5m", "10m_source"}
    ]
    contrasts = {
        "card_minus_source_articulation": paired_difference(
            five_minute, "card", "source_audio", "articulation_rate_wps"
        ),
        "card_minus_descript_articulation": paired_difference(
            five_minute, "card", "descript", "articulation_rate_wps"
        ),
        "card_minus_source_pause_fraction": paired_difference(
            five_minute, "card", "source_audio", "pause_fraction"
        ),
        "card_minus_descript_pause_fraction": paired_difference(
            five_minute, "card", "descript", "pause_fraction"
        ),
    }
    return {"means_by_system_condition": means, "paired_contrasts_5m": contrasts}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
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
    return parser.parse_args()


def main() -> None:
    """Collect the requested measurements and write the CSV and JSON outputs."""
    args = parse_args()
    rows: list[RateRow] = []
    if args.source_from_slices:
        print(f"Reading transcript slices from {args.slice_dir}")
        rows.extend(rows_from_transcript_slices(args.slice_dir))
    if args.participant_audio_root is not None:
        print(f"Probing participant audio under {args.participant_audio_root}")
        rows.extend(rows_from_participant_audio(args.participant_audio_root))
    if args.descript_root is not None:
        print(f"Probing Descript references under {args.descript_root}")
        rows.extend(rows_from_descript(args.descript_root))
    if not rows:
        raise SystemExit(
            "No inputs selected. Pass --source-from-slices, "
            "--participant-audio-root, or --descript-root."
        )

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
    print(f"\nWrote {csv_path} ({len(rows)} rows)")
    print(f"Wrote {json_path}")
    for key, value in summary["means_by_system_condition"].items():  # type: ignore[union-attr]
        print(f"  {key:<26} {value}")


if __name__ == "__main__":
    main()
