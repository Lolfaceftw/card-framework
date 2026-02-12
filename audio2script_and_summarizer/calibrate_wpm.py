"""Run one-shot emotion-aware WPM calibration and export JSON results.

Usage:
    uv run --extra audio2script python -m audio2script_and_summarizer.calibrate_wpm \
        --voice-dir "path/to/<audio>_voices" \
        --output "calibrated_wpm.json"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from .logging_utils import configure_logging
from .tts_pacing_calibration import (
    TTSPacingCalibration,
    calibrate_tts_pacing_profiles,
    estimate_weighted_wpm_from_transcript,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_OUTPUT_FILENAME = "calibrated_wpm.json"


class CalibrationOutputPayload(TypedDict):
    """Represent serialized calibration output payload."""

    generated_at_utc: str
    voice_dir: str
    device: str
    presets_path: str
    avg_wpm_neutral: float
    weighted_wpm_transcript_mix: float | None
    per_speaker_wpm_neutral: dict[str, float]
    per_speaker_preset_wpm: dict[str, dict[str, float]]


def _repo_root() -> Path:
    """Return repository root inferred from this module location."""
    return Path(__file__).resolve().parent.parent


def _default_cfg_path() -> Path:
    """Return default IndexTTS2 config path."""
    return _repo_root() / "voice-cloner-and-interjector" / "checkpoints" / "config.yaml"


def _default_model_dir() -> Path:
    """Return default IndexTTS2 checkpoints directory."""
    return _repo_root() / "voice-cloner-and-interjector" / "checkpoints"


def _default_presets_path() -> Path:
    """Return default emotion-preset JSON path."""
    return Path(__file__).resolve().with_name("emotion_pacing_presets.json")


def _resolve_device(requested_device: str | None) -> str:
    """Resolve device string for calibration.

    Args:
        requested_device: Optional user-specified device.

    Returns:
        ``cuda`` when available, otherwise ``cpu`` unless user override is provided.
    """
    if requested_device:
        return requested_device.strip()
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _round_wpm_value(wpm: float) -> float:
    """Round WPM to two decimals for stable JSON output."""
    return round(float(max(1.0, wpm)), 2)


def _build_output_payload(
    *,
    voice_dir: str,
    device: str,
    presets_path: str,
    avg_wpm: float,
    weighted_wpm: float | None,
    per_speaker_wpm: dict[str, float],
    calibration: TTSPacingCalibration,
) -> CalibrationOutputPayload:
    """Build JSON payload for calibrated WPM output.

    Args:
        voice_dir: Input speaker-voice directory used for calibration.
        device: Runtime device used for IndexTTS2.
        presets_path: Preset JSON path used for calibration.
        avg_wpm: Average neutral WPM from calibration.
        weighted_wpm: Optional transcript-weighted WPM.
        per_speaker_wpm: Neutral per-speaker WPM values.
        calibration: Full calibration object including per-preset rates.

    Returns:
        JSON-serializable calibration payload.
    """
    per_speaker_preset_wpm: dict[str, dict[str, float]] = {}
    for speaker, speaker_rates in sorted(
        calibration.seconds_per_word_by_speaker_preset.items()
    ):
        per_speaker_preset_wpm[speaker] = {
            preset: _round_wpm_value((1.0 / seconds_per_word) * 60.0)
            for preset, seconds_per_word in sorted(speaker_rates.items())
            if seconds_per_word > 0.0
        }

    return CalibrationOutputPayload(
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        voice_dir=str(Path(voice_dir).resolve()),
        device=device,
        presets_path=str(Path(presets_path).resolve()),
        avg_wpm_neutral=_round_wpm_value(avg_wpm),
        weighted_wpm_transcript_mix=(
            _round_wpm_value(weighted_wpm) if weighted_wpm is not None else None
        ),
        per_speaker_wpm_neutral={
            speaker: _round_wpm_value(value)
            for speaker, value in sorted(per_speaker_wpm.items())
        },
        per_speaker_preset_wpm=per_speaker_preset_wpm,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for one-shot WPM calibration."""
    parser = argparse.ArgumentParser(
        description="Calibrate emotion-aware TTS WPM and export per-emotion metrics."
    )
    parser.add_argument(
        "--voice-dir",
        required=True,
        help="Directory containing per-speaker WAV files (e.g. <audio>_voices).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run IndexTTS2 on (cuda/cpu). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--cfg-path",
        default=str(_default_cfg_path()),
        help="Path to IndexTTS2 config.yaml.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(_default_model_dir()),
        help="Path to IndexTTS2 checkpoint directory.",
    )
    parser.add_argument(
        "--presets-path",
        default=str(_default_presets_path()),
        help="Path to emotion pacing presets JSON.",
    )
    parser.add_argument(
        "--transcript-json",
        default=None,
        help=(
            "Optional transcript JSON path. When provided, also computes "
            "weighted WPM based on transcript speaker distribution."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_FILENAME,
        help="Output JSON path for calibration results.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite output JSON when it already exists.",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        default=False,
        help="Print the generated calibration JSON to stdout.",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("AUDIO2SCRIPT_LOG_LEVEL", "INFO").upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Console log level.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one-shot calibration and write per-emotion WPM output JSON.

    Args:
        argv: Optional CLI argv override used for tests.

    Returns:
        Process exit code where ``0`` indicates success.
    """
    args = _parse_args(argv)
    configure_logging(
        level=args.log_level,
        component="calibrate_wpm",
        enable_console=True,
    )

    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.force:
        logger.error("Output file already exists: %s. Pass --force to overwrite.", output_path)
        return 1

    device = _resolve_device(args.device)
    logger.info(
        "Starting emotion-aware WPM calibration voice_dir=%s device=%s presets=%s",
        args.voice_dir,
        device,
        args.presets_path,
    )

    try:
        avg_wpm, per_speaker_wpm, calibration = calibrate_tts_pacing_profiles(
            voice_dir=args.voice_dir,
            device=device,
            cfg_path=args.cfg_path,
            model_dir=args.model_dir,
            presets_path=args.presets_path,
            progress_cb=None,
        )
    except Exception:
        logger.exception("Calibration failed.")
        return 1

    weighted_wpm: float | None = None
    if args.transcript_json:
        try:
            weighted_wpm, _ = estimate_weighted_wpm_from_transcript(
                transcript_json_path=args.transcript_json,
                calibration=calibration,
            )
        except Exception:
            logger.exception(
                "Transcript-weighted WPM estimation failed for %s.",
                args.transcript_json,
            )
            return 1

    payload = _build_output_payload(
        voice_dir=args.voice_dir,
        device=device,
        presets_path=args.presets_path,
        avg_wpm=avg_wpm,
        weighted_wpm=weighted_wpm,
        per_speaker_wpm=per_speaker_wpm,
        calibration=calibration,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    logger.info(
        "Calibration completed avg_wpm_neutral=%.2f weighted_wpm=%s output=%s",
        payload["avg_wpm_neutral"],
        (
            f"{payload['weighted_wpm_transcript_mix']:.2f}"
            if payload["weighted_wpm_transcript_mix"] is not None
            else "n/a"
        ),
        output_path,
    )
    if args.print_json:
        print(json.dumps(payload, indent=2, ensure_ascii=True))
    else:
        print(f"Calibration JSON written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
