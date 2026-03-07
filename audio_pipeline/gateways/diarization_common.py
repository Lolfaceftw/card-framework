"""Shared helpers for speaker-diarization gateways and benchmarks."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
import subprocess

from audio_pipeline.contracts import DiarizationTurn
from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.runtime import ensure_command_available


def prepare_diarization_audio(
    *,
    audio_path: Path,
    output_dir: Path,
    output_filename: str = "diarization_input_mono.wav",
) -> Path:
    """Normalize diarization input to mono 16 kHz WAV via ``ffmpeg``."""
    ensure_command_available("ffmpeg")
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = output_dir / output_filename
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-vn",
        str(normalized_path),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise NonRetryableAudioStageError(
            "Failed to prepare mono diarization audio with ffmpeg. "
            f"Command: {' '.join(command)}. "
            f"Stderr: {(exc.stderr or '').strip()}"
        ) from exc
    return normalized_path


def parse_rttm_file(rttm_path: Path) -> list[DiarizationTurn]:
    """Parse RTTM into ordered diarization turns."""
    turns: list[DiarizationTurn] = []
    for line in rttm_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 8 or parts[0] != "SPEAKER":
            continue
        start_seconds = float(parts[3])
        duration_seconds = float(parts[4])
        end_seconds = start_seconds + duration_seconds
        turns.append(
            DiarizationTurn(
                speaker=str(parts[7]).strip(),
                start_time_ms=max(0, int(round(start_seconds * 1000))),
                end_time_ms=max(0, int(round(end_seconds * 1000))),
            )
        )
    turns.sort(key=lambda turn: turn.start_time_ms)
    return turns


def normalize_speaker_labels(
    turns: Iterable[DiarizationTurn],
) -> list[DiarizationTurn]:
    """Normalize arbitrary speaker labels into ``SPEAKER_XX``."""
    mapping: dict[str, str] = {}
    normalized: list[DiarizationTurn] = []
    next_index = 0
    for turn in turns:
        if turn.speaker not in mapping:
            mapping[turn.speaker] = f"SPEAKER_{next_index:02d}"
            next_index += 1
        normalized.append(
            DiarizationTurn(
                speaker=mapping[turn.speaker],
                start_time_ms=turn.start_time_ms,
                end_time_ms=turn.end_time_ms,
            )
        )
    return normalized


def write_rttm_file(
    turns: Iterable[DiarizationTurn],
    output_path: Path,
    *,
    uri: str,
) -> None:
    """Write diarization turns to RTTM."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered_turns = sorted(turns, key=lambda turn: turn.start_time_ms)
    lines: list[str] = []
    for turn in ordered_turns:
        duration_ms = max(0, turn.end_time_ms - turn.start_time_ms)
        if duration_ms <= 0:
            continue
        lines.append(
            "SPEAKER "
            f"{uri} 1 "
            f"{turn.start_time_ms / 1000.0:.3f} "
            f"{duration_ms / 1000.0:.3f} "
            f"<NA> <NA> {turn.speaker} <NA> <NA>"
        )
    output_path.write_text(
        ("\n".join(lines) + "\n") if lines else "",
        encoding="utf-8",
    )
