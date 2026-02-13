"""Interactive wizard for generating split prompt/holdout speaker clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydub import AudioSegment

from audio2script_and_summarizer.stage3_voice import (
    discover_summary_json_files,
    load_summary_entries,
    resolve_voice_sample_path,
)
from benchmarks.voice_clone.utils import now_utc_compact

DEFAULT_SEARCH_ROOT: Final[Path] = Path(".")
DEFAULT_PROMPT_BASE_DIR: Final[Path] = Path("benchmarks") / "auto_prompt"
DEFAULT_HOLDOUT_BASE_DIR: Final[Path] = Path("benchmarks") / "auto_holdout"
DEFAULT_TARGET_CLIP_SECONDS: Final[float] = 12.0
DEFAULT_MIN_CLIP_SECONDS: Final[float] = 4.0


@dataclass(slots=True, frozen=True)
class HoldoutSplitResult:
    """Represent generated prompt/holdout split directories."""

    summary_json_path: Path
    prompt_dir: Path
    holdout_dir: Path
    speaker_count: int
    requested_clip_seconds: float
    min_clip_seconds: float


def _prompt_text(prompt: str, *, default: str | None = None) -> str:
    """Prompt user for text input with optional default value."""
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    if raw:
        return raw
    return default or ""


def _prompt_int(
    prompt: str,
    *,
    default: int | None = None,
    allow_empty: bool = False,
) -> int | None:
    """Prompt user for integer input."""
    default_text = "" if default is None else str(default)
    while True:
        raw = _prompt_text(prompt, default=default_text if default is not None else None)
        if not raw and allow_empty:
            return None
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")


def _pick_summary_json(search_root: Path) -> Path:
    """Select a summary JSON path from discovered files or manual input."""
    candidates = discover_summary_json_files(search_root)
    if not candidates:
        manual = _prompt_text(
            "No summary JSON found. Enter summary JSON path manually",
            default="",
        )
        path = Path(manual).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Summary JSON not found: {path}")
        return path

    print("\nDiscovered summary JSON files (newest first):")
    for index, candidate in enumerate(candidates[:10], start=1):
        print(f"  {index}. {candidate}")
    selected_index = _prompt_int(
        "Select summary number",
        default=1,
        allow_empty=False,
    )
    assert selected_index is not None
    bounded_index = max(1, min(selected_index, len(candidates)))
    return candidates[bounded_index - 1]


def _resolve_speaker_source_map(summary_json_path: Path) -> dict[str, Path]:
    """Resolve one source WAV path per speaker from a summary JSON."""
    entries = load_summary_entries(summary_json_path.resolve())
    summary_dir = summary_json_path.resolve().parent
    source_paths: dict[str, Path] = {}
    for entry in entries:
        if entry.speaker in source_paths:
            continue
        source_paths[entry.speaker] = resolve_voice_sample_path(
            entry.voice_sample, summary_dir
        )
    return source_paths


def create_split_prompt_holdout_clips(
    *,
    speaker_sources: dict[str, Path],
    prompt_dir: Path,
    holdout_dir: Path,
    target_clip_seconds: float,
    min_clip_seconds: float,
) -> dict[str, tuple[Path, Path]]:
    """Create split prompt/holdout clips for each speaker.

    Args:
        speaker_sources: Mapping of speaker id to source WAV path.
        prompt_dir: Destination directory for prompt clips.
        holdout_dir: Destination directory for holdout clips.
        target_clip_seconds: Desired duration for each split clip.
        min_clip_seconds: Minimum acceptable clip duration after splitting.

    Returns:
        Mapping of speaker id to ``(prompt_path, holdout_path)``.

    Raises:
        ValueError: Any source clip is too short for configured split bounds.
    """
    if target_clip_seconds <= 0:
        raise ValueError("target_clip_seconds must be > 0.")
    if min_clip_seconds <= 0:
        raise ValueError("min_clip_seconds must be > 0.")
    if min_clip_seconds > target_clip_seconds:
        raise ValueError("min_clip_seconds cannot exceed target_clip_seconds.")

    prompt_dir.mkdir(parents=True, exist_ok=True)
    holdout_dir.mkdir(parents=True, exist_ok=True)

    requested_ms = int(round(target_clip_seconds * 1000.0))
    min_ms = int(round(min_clip_seconds * 1000.0))
    outputs: dict[str, tuple[Path, Path]] = {}

    for speaker_id, source_path in sorted(speaker_sources.items()):
        audio = AudioSegment.from_file(str(source_path))
        max_split_ms = len(audio) // 2
        clip_ms = min(requested_ms, max_split_ms)
        if clip_ms < min_ms:
            raise ValueError(
                f"Speaker {speaker_id} source clip is too short for split holdout "
                f"({len(audio) / 1000.0:.2f}s at {source_path})."
            )

        prompt_path = (prompt_dir / f"{speaker_id}.wav").resolve()
        holdout_path = (holdout_dir / f"{speaker_id}.wav").resolve()
        audio[:clip_ms].export(str(prompt_path), format="wav")
        audio[-clip_ms:].export(str(holdout_path), format="wav")
        outputs[speaker_id] = (prompt_path, holdout_path)

    return outputs


def run_holdout_wizard(*, summary_json_path: Path | None = None) -> HoldoutSplitResult:
    """Run interactive wizard to create split prompt/holdout clip sets.

    Args:
        summary_json_path: Optional pre-selected summary JSON path.

    Returns:
        Metadata describing generated split directories.
    """
    print("Voice Holdout Split Wizard")
    print("- This creates separate prompt and holdout WAVs per speaker.")
    print("- Holdout clips are created from the tail of each source speaker clip.")
    print("")

    selected_summary_path: Path
    if summary_json_path is None:
        search_root_raw = _prompt_text(
            "Search root for *_summary.json files",
            default=str(DEFAULT_SEARCH_ROOT),
        )
        search_root = Path(search_root_raw).expanduser().resolve()
        selected_summary_path = _pick_summary_json(search_root)
    else:
        selected_summary_path = summary_json_path.expanduser().resolve()
        if not selected_summary_path.exists():
            raise FileNotFoundError(f"Summary JSON not found: {selected_summary_path}")
    print(f"\nSelected summary: {selected_summary_path}")

    run_id = now_utc_compact()
    prompt_dir_raw = _prompt_text(
        "Prompt WAV output directory",
        default=str((DEFAULT_PROMPT_BASE_DIR / run_id)),
    )
    holdout_dir_raw = _prompt_text(
        "Holdout WAV output directory",
        default=str((DEFAULT_HOLDOUT_BASE_DIR / run_id)),
    )
    target_clip_raw = _prompt_text(
        "Target clip length in seconds",
        default=f"{DEFAULT_TARGET_CLIP_SECONDS:.1f}",
    )
    min_clip_raw = _prompt_text(
        "Minimum clip length in seconds",
        default=f"{DEFAULT_MIN_CLIP_SECONDS:.1f}",
    )

    prompt_dir = Path(prompt_dir_raw).expanduser().resolve()
    holdout_dir = Path(holdout_dir_raw).expanduser().resolve()
    target_clip_seconds = float(target_clip_raw)
    min_clip_seconds = float(min_clip_raw)

    speaker_sources = _resolve_speaker_source_map(selected_summary_path)
    outputs = create_split_prompt_holdout_clips(
        speaker_sources=speaker_sources,
        prompt_dir=prompt_dir,
        holdout_dir=holdout_dir,
        target_clip_seconds=target_clip_seconds,
        min_clip_seconds=min_clip_seconds,
    )

    print("\nSplit complete.")
    print(f"Speakers processed: {len(outputs)}")
    print(f"Prompt directory: {prompt_dir}")
    print(f"Holdout directory: {holdout_dir}")
    return HoldoutSplitResult(
        summary_json_path=selected_summary_path,
        prompt_dir=prompt_dir,
        holdout_dir=holdout_dir,
        speaker_count=len(outputs),
        requested_clip_seconds=target_clip_seconds,
        min_clip_seconds=min_clip_seconds,
    )
