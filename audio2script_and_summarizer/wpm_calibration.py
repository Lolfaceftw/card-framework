"""Utilities to calibrate IndexTTS2 speaking rate for duration control."""

from __future__ import annotations

import logging
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Literal, Tuple

from pydub import AudioSegment

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

CALIBRATION_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This is a short calibration sample for timing."
)


CalibrationEventType = Literal[
    "model_init_started",
    "model_init_completed",
    "speaker_started",
    "speaker_completed",
    "calibration_completed",
]


@dataclass(slots=True, frozen=True)
class CalibrationEvent:
    """Represent one progress event during WPM calibration.

    Attributes:
        event_type: Event category used by the caller for UI updates.
        speaker_name: Speaker label for speaker-scoped events.
        speaker_index: One-based index of current speaker.
        speaker_count: Total number of speakers to calibrate.
        speaker_wpm: Computed WPM for the current speaker, when available.
        average_wpm: Final average WPM, when available.
    """

    event_type: CalibrationEventType
    speaker_name: str | None = None
    speaker_index: int | None = None
    speaker_count: int | None = None
    speaker_wpm: float | None = None
    average_wpm: float | None = None


def _emit_progress(
    progress_cb: Callable[[CalibrationEvent], None] | None,
    event: CalibrationEvent,
) -> None:
    """Emit one calibration progress event.

    Args:
        progress_cb: Optional consumer callback.
        event: Progress event payload.
    """
    if progress_cb is None:
        return
    try:
        progress_cb(event)
    except Exception as exc:  # noqa: BLE001
        logger.warning("WPM calibration progress callback failed: %s", exc)


def _ensure_indextts_on_path(repo_root: Path) -> None:
    """Ensure voice-cloner-and-interjector is on sys.path."""
    voice_root = repo_root / "voice-cloner-and-interjector"
    if voice_root.exists() and str(voice_root) not in sys.path:
        sys.path.insert(0, str(voice_root))


def _create_tts(device: str, cfg_path: Path, model_dir: Path):
    """Create an IndexTTS2 instance for calibration."""
    _ensure_indextts_on_path(cfg_path.parent.parent)
    from indextts.infer_v2 import IndexTTS2  # pylint: disable=import-error

    use_fp16 = device.startswith("cuda")
    use_cuda_kernel = device.startswith("cuda")
    return IndexTTS2(
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        device=device,
        use_fp16=use_fp16,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=False,
    )


def _measure_duration_seconds(
    audio_path: Path, retries: int = 5, delay: float = 0.2
) -> float:
    """Measure duration of an audio file in seconds.

    Retries are used to handle Windows file locks while TTS is still flushing.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            audio = AudioSegment.from_file(audio_path)
            return max(0.0, len(audio) / 1000.0)
        except OSError as exc:
            last_exc = exc
            if attempt >= retries - 1:
                break
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    return 0.0


def calibrate_voice_wpm(
    voice_dir: str,
    device: str,
    cfg_path: str,
    model_dir: str,
    emo_text: str = "neutral, clear",
    emo_alpha: float = 0.6,
    use_emo_text: bool = True,
    progress_cb: Callable[[CalibrationEvent], None] | None = None,
) -> Tuple[float, Dict[str, float]]:
    """Calibrate speaking rate (WPM) for each speaker sample using IndexTTS2.

    Args:
        voice_dir: Directory containing speaker WAV samples.
        device: Device string for IndexTTS2 (e.g., "cuda", "cpu").
        cfg_path: Path to IndexTTS2 config.yaml.
        model_dir: Path to IndexTTS2 checkpoints directory.
        emo_text: Emotion text used during calibration.
        emo_alpha: Emotion intensity used during calibration.
        use_emo_text: Whether to apply emo_text during calibration.
        progress_cb: Optional callback for progress and speaker-level updates.

    Returns:
        Tuple of (average_wpm, per_speaker_wpm).
    """
    voice_path = Path(voice_dir)
    wav_files = sorted(voice_path.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No speaker wav files found in {voice_dir}")
    speaker_count = len(wav_files)

    cfg_path_obj = Path(cfg_path)
    model_dir_obj = Path(model_dir)
    if not cfg_path_obj.exists():
        raise FileNotFoundError(f"IndexTTS2 config not found: {cfg_path_obj}")
    if not model_dir_obj.exists():
        raise FileNotFoundError(f"IndexTTS2 model dir not found: {model_dir_obj}")

    _emit_progress(
        progress_cb,
        CalibrationEvent(
            event_type="model_init_started",
            speaker_count=speaker_count,
        ),
    )
    logger.info("Initializing IndexTTS2 for WPM calibration...")
    tts = _create_tts(device=device, cfg_path=cfg_path_obj, model_dir=model_dir_obj)
    _emit_progress(
        progress_cb,
        CalibrationEvent(
            event_type="model_init_completed",
            speaker_count=speaker_count,
        ),
    )

    words = max(1, len(CALIBRATION_TEXT.split()))
    per_speaker_wpm: Dict[str, float] = {}

    for speaker_index, wav_path in enumerate(wav_files, start=1):
        speaker_name = wav_path.stem
        _emit_progress(
            progress_cb,
            CalibrationEvent(
                event_type="speaker_started",
                speaker_name=speaker_name,
                speaker_index=speaker_index,
                speaker_count=speaker_count,
            ),
        )
        handle, tmp_path = tempfile.mkstemp(
            prefix=f"calib_{speaker_name}_", suffix=".wav"
        )
        try:
            import os

            os.close(handle)
        except OSError:
            pass
        Path(tmp_path).unlink(missing_ok=True)
        temp_audio = Path(tmp_path)

        try:
            logger.info("Calibrating speaker %s...", speaker_name)
            tts.infer(
                spk_audio_prompt=str(wav_path),
                text=CALIBRATION_TEXT,
                output_path=str(temp_audio),
                emo_alpha=emo_alpha,
                use_emo_text=use_emo_text,
                emo_text=emo_text if use_emo_text else "",
                use_random=False,
                verbose=False,
            )
            duration_seconds = _measure_duration_seconds(temp_audio)
            if duration_seconds <= 0:
                raise ValueError("Zero duration from calibration audio.")

            wpm = (words / duration_seconds) * 60.0
            per_speaker_wpm[speaker_name] = wpm
            logger.info("Speaker %s WPM: %.2f", speaker_name, wpm)
            _emit_progress(
                progress_cb,
                CalibrationEvent(
                    event_type="speaker_completed",
                    speaker_name=speaker_name,
                    speaker_index=speaker_index,
                    speaker_count=speaker_count,
                    speaker_wpm=wpm,
                ),
            )
        finally:
            try:
                temp_audio.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "Could not delete calibration file %s: %s", temp_audio, exc
                )

    if not per_speaker_wpm:
        raise ValueError("No WPM results from calibration.")

    avg_wpm = sum(per_speaker_wpm.values()) / len(per_speaker_wpm)
    avg_wpm = max(1.0, avg_wpm)
    avg_wpm = float(math.ceil(avg_wpm * 100.0) / 100.0)

    logger.info("Average calibrated WPM: %.2f", avg_wpm)
    _emit_progress(
        progress_cb,
        CalibrationEvent(
            event_type="calibration_completed",
            speaker_count=speaker_count,
            average_wpm=avg_wpm,
        ),
    )
    return avg_wpm, per_speaker_wpm
