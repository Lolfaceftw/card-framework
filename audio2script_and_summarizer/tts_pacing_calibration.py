"""Emotion-aware TTS pacing calibration utilities for duration budgeting.

This module calibrates IndexTTS2 speaking pace with emotion presets, then uses
those measured rates to estimate word budgets and expected rendered durations.
"""

from __future__ import annotations

import json
import logging
import math
import os
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, TypedDict

from pydub import AudioSegment

from .speaker_validation import load_transcript_segments

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_PRESETS_FILENAME = "emotion_pacing_presets.json"
DEFAULT_CALIBRATION_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "This timing sample estimates speaking pace."
)
DEFAULT_NEUTRAL_PRESET = "neutral"


class EmotionPacingPresetPayload(TypedDict):
    """Represent one raw preset entry from JSON configuration."""

    name: str
    emo_text: str
    emo_alpha: float
    calibration_text: str
    keywords: list[str]


@dataclass(slots=True, frozen=True)
class EmotionPacingPreset:
    """Represent a normalized emotion pacing preset.

    Attributes:
        name: Stable preset identifier.
        emo_text: Emotion text passed into IndexTTS2.
        emo_alpha: Emotion intensity passed into IndexTTS2.
        calibration_text: Reference text used for pace measurement.
        keywords: Lower-cased keyword match list used for summary-line mapping.
    """

    name: str
    emo_text: str
    emo_alpha: float
    calibration_text: str
    keywords: tuple[str, ...]


CalibrationEventType = Literal[
    "model_init_started",
    "model_init_completed",
    "speaker_started",
    "speaker_completed",
    "calibration_completed",
]


@dataclass(slots=True, frozen=True)
class CalibrationEvent:
    """Represent one progress event during emotion-aware pacing calibration.

    Attributes:
        event_type: Event category consumed by dashboards/callers.
        speaker_name: Speaker label for speaker-scoped events.
        speaker_index: One-based index of current speaker.
        speaker_count: Total number of speakers to calibrate.
        speaker_wpm: Neutral/default WPM for the current speaker.
        average_wpm: Final average WPM value.
    """

    event_type: CalibrationEventType
    speaker_name: str | None = None
    speaker_index: int | None = None
    speaker_count: int | None = None
    speaker_wpm: float | None = None
    average_wpm: float | None = None


@dataclass(slots=True, frozen=True)
class TTSPacingCalibration:
    """Contain measured per-speaker, per-preset speaking rates.

    Attributes:
        presets: Ordered preset mapping keyed by preset name.
        seconds_per_word_by_speaker_preset: Nested mapping indexed by speaker then
            preset name.
        speaker_default_seconds_per_word: Default seconds-per-word per speaker,
            derived from neutral preset when available.
        global_default_seconds_per_word: Fallback seconds-per-word when a speaker
            is unknown.
    """

    presets: dict[str, EmotionPacingPreset]
    seconds_per_word_by_speaker_preset: dict[str, dict[str, float]]
    speaker_default_seconds_per_word: dict[str, float]
    global_default_seconds_per_word: float

    def get_seconds_per_word(self, speaker: str, preset_name: str) -> float:
        """Resolve seconds-per-word with robust fallback order.

        Args:
            speaker: Speaker label.
            preset_name: Preset identifier.

        Returns:
            Positive seconds-per-word value.
        """
        speaker_key = speaker.strip()
        preset_key = preset_name.strip()
        speaker_rates = self.seconds_per_word_by_speaker_preset.get(speaker_key)
        if speaker_rates is not None:
            direct = speaker_rates.get(preset_key)
            if direct is not None and direct > 0.0:
                return direct
        speaker_default = self.speaker_default_seconds_per_word.get(speaker_key)
        if speaker_default is not None and speaker_default > 0.0:
            return speaker_default
        return max(1e-6, self.global_default_seconds_per_word)

    def get_wpm(self, speaker: str, preset_name: str) -> float:
        """Resolve words-per-minute from seconds-per-word."""
        seconds_per_word = self.get_seconds_per_word(
            speaker=speaker,
            preset_name=preset_name,
        )
        return (1.0 / seconds_per_word) * 60.0


DEFAULT_PRESETS: tuple[EmotionPacingPreset, ...] = (
    EmotionPacingPreset(
        name="neutral",
        emo_text="Neutral, clear, steady delivery",
        emo_alpha=0.6,
        calibration_text=(
            "Today we are reviewing a practical topic. "
            "The explanation should sound clear and balanced."
        ),
        keywords=("neutral", "clear", "steady", "balanced"),
    ),
    EmotionPacingPreset(
        name="excited_fast",
        emo_text="Excited, energetic, fast-paced",
        emo_alpha=0.7,
        calibration_text=(
            "This is exciting news and everything is moving quickly. "
            "The energy is high and the pace feels lively."
        ),
        keywords=("excited", "energetic", "fast", "upbeat", "lively"),
    ),
    EmotionPacingPreset(
        name="serious_slow",
        emo_text="Serious, deliberate, measured and slower",
        emo_alpha=0.65,
        calibration_text=(
            "This section requires careful reasoning and a measured pace. "
            "Each point should be delivered deliberately."
        ),
        keywords=("serious", "deliberate", "measured", "slow", "calm"),
    ),
    EmotionPacingPreset(
        name="empathetic",
        emo_text="Empathetic, supportive, warm and thoughtful",
        emo_alpha=0.65,
        calibration_text=(
            "I understand why this situation can feel difficult. "
            "Let us walk through it with a warm and thoughtful tone."
        ),
        keywords=("empathetic", "supportive", "warm", "thoughtful", "gentle"),
    ),
    EmotionPacingPreset(
        name="inquisitive",
        emo_text="Curious, inquisitive, reflective",
        emo_alpha=0.6,
        calibration_text=(
            "That is an interesting question and it is worth exploring. "
            "Let us think through the key details together."
        ),
        keywords=("curious", "inquisitive", "reflective", "questioning"),
    ),
)


def _default_presets_path() -> Path:
    """Return repository-local default preset configuration path."""
    return Path(__file__).resolve().with_name(DEFAULT_PRESETS_FILENAME)


def _emit_progress(
    progress_cb: Callable[[CalibrationEvent], None] | None,
    event: CalibrationEvent,
) -> None:
    """Emit one calibration event to an optional callback."""
    if progress_cb is None:
        return
    try:
        progress_cb(event)
    except Exception as exc:  # noqa: BLE001
        logger.warning("TTS pacing progress callback failed: %s", exc)


def _ensure_indextts_on_path(repo_root: Path) -> None:
    """Ensure voice-cloner-and-interjector package is importable."""
    voice_root = repo_root / "voice-cloner-and-interjector"
    if voice_root.exists() and str(voice_root) not in sys.path:
        sys.path.insert(0, str(voice_root))


def _create_tts(device: str, cfg_path: Path, model_dir: Path):
    """Create an IndexTTS2 instance for pacing calibration."""
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
    audio_path: Path,
    retries: int = 5,
    delay_seconds: float = 0.2,
) -> float:
    """Measure audio duration in seconds with retry for transient file locks."""
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            return max(0.0, len(AudioSegment.from_file(audio_path)) / 1000.0)
        except OSError as exc:
            last_exc = exc
            if attempt >= retries - 1:
                break
            time.sleep(delay_seconds)
    if last_exc is not None:
        raise last_exc
    return 0.0


def load_emotion_pacing_presets(
    presets_path: str | None = None,
) -> dict[str, EmotionPacingPreset]:
    """Load pacing presets from JSON, falling back to built-in defaults.

    Args:
        presets_path: Optional path to presets JSON.

    Returns:
        Ordered mapping of preset name to normalized preset config.
    """
    resolved_path = (
        Path(presets_path).resolve()
        if presets_path and presets_path.strip()
        else _default_presets_path()
    )
    if not resolved_path.exists():
        logger.warning(
            "Emotion preset file not found at %s; using built-in defaults.",
            resolved_path,
        )
        return {preset.name: preset for preset in DEFAULT_PRESETS}

    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "Failed to parse preset file %s (%s); using built-in defaults.",
            resolved_path,
            exc,
        )
        return {preset.name: preset for preset in DEFAULT_PRESETS}

    raw_items: list[object]
    if isinstance(payload, dict):
        candidate = payload.get("presets")
        raw_items = candidate if isinstance(candidate, list) else []
    elif isinstance(payload, list):
        raw_items = payload
    else:
        raw_items = []

    loaded: dict[str, EmotionPacingPreset] = {}
    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue
        name = str(raw_item.get("name", "")).strip().lower()
        if not name:
            continue
        emo_text = str(raw_item.get("emo_text", "")).strip()
        if not emo_text:
            continue
        calibration_text = str(
            raw_item.get("calibration_text", DEFAULT_CALIBRATION_TEXT)
        ).strip()
        calibration_text = calibration_text or DEFAULT_CALIBRATION_TEXT
        try:
            emo_alpha = float(raw_item.get("emo_alpha", 0.6))
        except (TypeError, ValueError):
            emo_alpha = 0.6
        emo_alpha = max(0.0, min(1.0, emo_alpha))
        raw_keywords = raw_item.get("keywords", [])
        keywords: list[str] = []
        if isinstance(raw_keywords, list):
            keywords = [
                str(keyword).strip().lower()
                for keyword in raw_keywords
                if str(keyword).strip()
            ]

        loaded[name] = EmotionPacingPreset(
            name=name,
            emo_text=emo_text,
            emo_alpha=emo_alpha,
            calibration_text=calibration_text,
            keywords=tuple(keywords),
        )

    if not loaded:
        logger.warning(
            "Preset file %s had no valid entries; using built-in defaults.",
            resolved_path,
        )
        return {preset.name: preset for preset in DEFAULT_PRESETS}
    return loaded


def resolve_emotion_preset_name(
    emo_text: str,
    presets: dict[str, EmotionPacingPreset],
    *,
    default_preset_name: str = DEFAULT_NEUTRAL_PRESET,
) -> str:
    """Map free-form emo_text into a configured preset name.

    Args:
        emo_text: Emotion text from summary line.
        presets: Preset mapping.
        default_preset_name: Preferred fallback preset.

    Returns:
        Chosen preset name.
    """
    if not presets:
        return default_preset_name
    lowered = emo_text.strip().lower()
    if lowered:
        for preset_name, preset in presets.items():
            if any(keyword in lowered for keyword in preset.keywords):
                return preset_name
    if default_preset_name in presets:
        return default_preset_name
    return next(iter(presets))


def _round_wpm(wpm: float) -> float:
    """Round WPM upward to two decimals for stable budgeting logs."""
    bounded = max(1.0, wpm)
    return float(math.ceil(bounded * 100.0) / 100.0)


def calibrate_tts_pacing_profiles(
    *,
    voice_dir: str,
    device: str,
    cfg_path: str,
    model_dir: str,
    presets_path: str | None = None,
    progress_cb: Callable[[CalibrationEvent], None] | None = None,
) -> tuple[float, dict[str, float], TTSPacingCalibration]:
    """Calibrate per-speaker speaking pace for each emotion preset.

    Args:
        voice_dir: Directory with per-speaker WAV reference files.
        device: Device string used by IndexTTS2.
        cfg_path: IndexTTS2 config path.
        model_dir: IndexTTS2 checkpoint directory.
        presets_path: Optional preset JSON path override.
        progress_cb: Optional calibration progress callback.

    Returns:
        Tuple of average WPM, per-speaker default WPM mapping, and full
        calibration object.

    Raises:
        FileNotFoundError: Missing speaker WAVs or TTS model files.
        ValueError: Calibration produced no measurable pacing rates.
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

    presets = load_emotion_pacing_presets(presets_path=presets_path)
    if not presets:
        raise ValueError("No emotion pacing presets available for calibration.")

    _emit_progress(
        progress_cb,
        CalibrationEvent(
            event_type="model_init_started",
            speaker_count=speaker_count,
        ),
    )
    logger.info(
        "Initializing IndexTTS2 for emotion-aware pacing calibration. speakers=%d presets=%d",
        speaker_count,
        len(presets),
    )
    tts = _create_tts(device=device, cfg_path=cfg_path_obj, model_dir=model_dir_obj)
    _emit_progress(
        progress_cb,
        CalibrationEvent(
            event_type="model_init_completed",
            speaker_count=speaker_count,
        ),
    )

    seconds_per_word_by_speaker_preset: dict[str, dict[str, float]] = {}

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
        speaker_rates: dict[str, float] = {}
        for preset_name, preset in presets.items():
            handle, temp_path_raw = tempfile.mkstemp(
                prefix=f"calib_{speaker_name}_{preset_name}_",
                suffix=".wav",
            )
            try:
                os.close(handle)
            except OSError:
                pass
            temp_path = Path(temp_path_raw)
            temp_path.unlink(missing_ok=True)
            try:
                tts.infer(
                    spk_audio_prompt=str(wav_path),
                    text=preset.calibration_text,
                    output_path=str(temp_path),
                    emo_alpha=preset.emo_alpha,
                    use_emo_text=True,
                    emo_text=preset.emo_text,
                    use_random=False,
                    verbose=False,
                )
                duration_seconds = _measure_duration_seconds(temp_path)
                word_count = max(1, len(preset.calibration_text.split()))
                if duration_seconds <= 0.0:
                    raise ValueError(
                        "Calibration produced non-positive audio duration."
                    )
                speaker_rates[preset_name] = duration_seconds / float(word_count)
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "Could not delete calibration temp file %s: %s",
                        temp_path,
                        exc,
                    )
        if not speaker_rates:
            continue
        seconds_per_word_by_speaker_preset[speaker_name] = speaker_rates
        neutral_name = (
            DEFAULT_NEUTRAL_PRESET
            if DEFAULT_NEUTRAL_PRESET in speaker_rates
            else next(iter(speaker_rates))
        )
        speaker_wpm = (1.0 / speaker_rates[neutral_name]) * 60.0
        _emit_progress(
            progress_cb,
            CalibrationEvent(
                event_type="speaker_completed",
                speaker_name=speaker_name,
                speaker_index=speaker_index,
                speaker_count=speaker_count,
                speaker_wpm=speaker_wpm,
            ),
        )

    if not seconds_per_word_by_speaker_preset:
        raise ValueError("No speaker pacing rates were produced during calibration.")

    speaker_default_seconds_per_word: dict[str, float] = {}
    for speaker_name, speaker_rates in sorted(seconds_per_word_by_speaker_preset.items()):
        if DEFAULT_NEUTRAL_PRESET in speaker_rates:
            speaker_default_seconds_per_word[speaker_name] = speaker_rates[
                DEFAULT_NEUTRAL_PRESET
            ]
        else:
            speaker_default_seconds_per_word[speaker_name] = statistics.mean(
                speaker_rates.values()
            )

    global_default_seconds_per_word = statistics.mean(
        speaker_default_seconds_per_word.values()
    )
    calibration = TTSPacingCalibration(
        presets=presets,
        seconds_per_word_by_speaker_preset=seconds_per_word_by_speaker_preset,
        speaker_default_seconds_per_word=speaker_default_seconds_per_word,
        global_default_seconds_per_word=global_default_seconds_per_word,
    )

    per_speaker_wpm = {
        speaker: _round_wpm((1.0 / rate) * 60.0)
        for speaker, rate in speaker_default_seconds_per_word.items()
    }
    avg_wpm = _round_wpm(sum(per_speaker_wpm.values()) / len(per_speaker_wpm))
    _emit_progress(
        progress_cb,
        CalibrationEvent(
            event_type="calibration_completed",
            speaker_count=speaker_count,
            average_wpm=avg_wpm,
        ),
    )
    logger.info(
        "TTS pacing calibration complete avg_wpm=%.2f speakers=%d presets=%d",
        avg_wpm,
        len(per_speaker_wpm),
        len(presets),
    )
    return avg_wpm, per_speaker_wpm, calibration


def estimate_weighted_wpm_from_transcript(
    *,
    transcript_json_path: str,
    calibration: TTSPacingCalibration,
) -> tuple[float, dict[str, float]]:
    """Estimate effective WPM using transcript speaker distribution and TTS rates.

    Args:
        transcript_json_path: Path to diarized transcript JSON.
        calibration: Precomputed TTS pacing calibration object.

    Returns:
        Tuple of weighted average WPM and per-speaker default WPM mapping.
    """
    segments = load_transcript_segments(transcript_json_path)
    per_speaker_word_totals: dict[str, int] = {}
    for segment in segments:
        word_count = len(segment.text.split())
        if word_count <= 0:
            continue
        per_speaker_word_totals[segment.speaker] = (
            per_speaker_word_totals.get(segment.speaker, 0) + word_count
        )

    per_speaker_wpm = {
        speaker: _round_wpm((1.0 / rate) * 60.0)
        for speaker, rate in calibration.speaker_default_seconds_per_word.items()
    }
    if not per_speaker_word_totals:
        if per_speaker_wpm:
            fallback = _round_wpm(sum(per_speaker_wpm.values()) / len(per_speaker_wpm))
            return fallback, per_speaker_wpm
        global_wpm = _round_wpm((1.0 / calibration.global_default_seconds_per_word) * 60.0)
        return global_wpm, per_speaker_wpm

    total_words = 0
    weighted_seconds = 0.0
    for speaker, word_total in per_speaker_word_totals.items():
        sec_per_word = calibration.get_seconds_per_word(
            speaker=speaker,
            preset_name=DEFAULT_NEUTRAL_PRESET,
        )
        total_words += word_total
        weighted_seconds += float(word_total) * sec_per_word

    if total_words <= 0 or weighted_seconds <= 0.0:
        global_wpm = _round_wpm((1.0 / calibration.global_default_seconds_per_word) * 60.0)
        return global_wpm, per_speaker_wpm

    weighted_wpm = (float(total_words) / weighted_seconds) * 60.0
    return _round_wpm(weighted_wpm), per_speaker_wpm


def estimate_summary_duration_seconds(
    *,
    summary_json_path: str,
    calibration: TTSPacingCalibration,
    segment_pause_ms: int = 400,
    crossfade_ms: int = 300,
) -> float:
    """Estimate Stage 3 total duration from summary text and pacing calibration.

    Args:
        summary_json_path: Path to Stage 2 summary JSON.
        calibration: Precomputed pacing calibration.
        segment_pause_ms: Pause inserted between segments in Stage 3.
        crossfade_ms: Crossfade duration used in Stage 3.

    Returns:
        Estimated total duration in seconds.
    """
    payload = json.loads(Path(summary_json_path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        return 0.0

    total_seconds = 0.0
    line_count = 0
    for raw_item in payload:
        if not isinstance(raw_item, dict):
            continue
        text = str(raw_item.get("text", "")).strip()
        speaker = str(raw_item.get("speaker", "")).strip()
        emo_text = str(raw_item.get("emo_text", "")).strip()
        if not text:
            continue
        preset_name = resolve_emotion_preset_name(
            emo_text=emo_text,
            presets=calibration.presets,
        )
        sec_per_word = calibration.get_seconds_per_word(
            speaker=speaker,
            preset_name=preset_name,
        )
        total_seconds += float(len(text.split())) * sec_per_word
        line_count += 1

    if line_count <= 1:
        return max(0.0, total_seconds)
    transition_overhead_ms = max(0, int(segment_pause_ms) - int(crossfade_ms))
    total_seconds += ((line_count - 1) * transition_overhead_ms) / 1000.0
    return max(0.0, total_seconds)
