"""Persistent cache helpers for Stage 1.75 TTS pacing calibration.

The cache is keyed by a deterministic fingerprint derived from calibration
inputs, then stores serialized ``TTSPacingCalibration`` payloads on disk.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

from . import tts_pacing_calibration
from .tts_pacing_calibration import TTSPacingCalibration

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

CALIBRATION_CACHE_SCHEMA_VERSION = 1


class _FileSnapshot(TypedDict):
    """Represent one file metadata entry used in fingerprint generation."""

    path: str
    size: int
    mtime_ns: int


class _CalibrationCacheRecord(TypedDict):
    """Represent serialized cache file contents."""

    schema_version: int
    fingerprint: str
    generated_at_utc: str
    calibration: object


def _sha256_text(value: str) -> str:
    """Return SHA-256 hex digest for UTF-8 text."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    """Return SHA-256 hex digest for file bytes.

    Args:
        path: File path to hash.

    Returns:
        Lowercase SHA-256 hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _collect_file_snapshots(
    *,
    files: list[Path],
    root: Path,
) -> list[_FileSnapshot]:
    """Collect deterministic file metadata snapshots.

    Args:
        files: File list to capture.
        root: Root used to derive stable relative paths.

    Returns:
        Sorted metadata entries for hashing payloads.
    """
    snapshots: list[_FileSnapshot] = []
    for path in sorted(files):
        stat_result = path.stat()
        snapshots.append(
            _FileSnapshot(
                path=str(path.resolve().relative_to(root.resolve())),
                size=int(stat_result.st_size),
                mtime_ns=int(stat_result.st_mtime_ns),
            )
        )
    return snapshots


def _default_presets_hash() -> str:
    """Hash built-in default presets for missing preset-file scenarios."""
    payload = [
        {
            "name": preset.name,
            "emo_text": preset.emo_text,
            "emo_alpha": preset.emo_alpha,
            "calibration_text": preset.calibration_text,
            "keywords": list(preset.keywords),
        }
        for preset in tts_pacing_calibration.DEFAULT_PRESETS
    ]
    return _sha256_text(json.dumps(payload, sort_keys=True))


def _tts_module_hash() -> str:
    """Return hash for ``tts_pacing_calibration.py`` contents."""
    module_path = Path(tts_pacing_calibration.__file__).resolve()
    return _sha256_file(module_path)


def cache_path_for_fingerprint(*, cache_dir: str, fingerprint: str) -> Path:
    """Return cache file path for one fingerprint key.

    Args:
        cache_dir: Directory where calibration cache files are stored.
        fingerprint: Fingerprint key generated for calibration inputs.

    Returns:
        Absolute cache JSON path.
    """
    return Path(cache_dir).resolve() / f"{fingerprint}.json"


def build_calibration_fingerprint(
    *,
    voice_dir: str,
    device: str,
    cfg_path: str,
    model_dir: str,
    presets_path: str | None,
) -> str:
    """Build deterministic fingerprint from Stage 1.75 calibration inputs.

    Args:
        voice_dir: Directory containing per-speaker WAV references.
        device: Runtime device used by IndexTTS2.
        cfg_path: Path to IndexTTS2 ``config.yaml``.
        model_dir: Path to IndexTTS2 checkpoint directory.
        presets_path: Optional path to emotion pacing presets JSON.

    Returns:
        SHA-256 hex digest representing this input set.

    Raises:
        FileNotFoundError: Missing required calibration inputs.
    """
    voice_dir_path = Path(voice_dir).resolve()
    cfg_path_obj = Path(cfg_path).resolve()
    model_dir_path = Path(model_dir).resolve()

    wav_files = sorted(voice_dir_path.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No speaker wav files found in {voice_dir_path}")
    if not cfg_path_obj.exists():
        raise FileNotFoundError(f"IndexTTS2 config not found: {cfg_path_obj}")
    if not model_dir_path.exists():
        raise FileNotFoundError(f"IndexTTS2 model dir not found: {model_dir_path}")

    presets_hash = _default_presets_hash()
    resolved_presets_path: str | None = None
    if presets_path is not None and presets_path.strip():
        preset_candidate = Path(presets_path).resolve()
        resolved_presets_path = str(preset_candidate)
        if preset_candidate.exists():
            presets_hash = _sha256_file(preset_candidate)

    model_files = [path for path in model_dir_path.rglob("*") if path.is_file()]
    model_snapshots = _collect_file_snapshots(files=model_files, root=model_dir_path)
    wav_snapshots = _collect_file_snapshots(files=wav_files, root=voice_dir_path)
    payload = {
        "schema_version": CALIBRATION_CACHE_SCHEMA_VERSION,
        "device": device.strip().lower(),
        "cfg_path": str(cfg_path_obj),
        "cfg_hash": _sha256_file(cfg_path_obj),
        "model_dir": str(model_dir_path),
        "model_snapshots": model_snapshots,
        "voice_dir": str(voice_dir_path),
        "voice_snapshots": wav_snapshots,
        "presets_path": resolved_presets_path,
        "presets_hash": presets_hash,
        "tts_pacing_calibration_hash": _tts_module_hash(),
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def load_cached_tts_pacing(
    *,
    cache_dir: str,
    fingerprint: str,
) -> TTSPacingCalibration | None:
    """Load one cached TTS pacing calibration payload.

    Args:
        cache_dir: Directory where cache JSON files are stored.
        fingerprint: Fingerprint key for the desired cache entry.

    Returns:
        Parsed ``TTSPacingCalibration`` when cache is valid, otherwise ``None``.
    """
    cache_path = cache_path_for_fingerprint(cache_dir=cache_dir, fingerprint=fingerprint)
    if not cache_path.exists():
        return None
    try:
        payload_raw = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Unable to parse WPM calibration cache file: %s", cache_path)
        return None
    if not isinstance(payload_raw, dict):
        logger.warning("Invalid WPM calibration cache payload format: %s", cache_path)
        return None

    schema_version = payload_raw.get("schema_version")
    cached_fingerprint = payload_raw.get("fingerprint")
    calibration_payload = payload_raw.get("calibration")
    if schema_version != CALIBRATION_CACHE_SCHEMA_VERSION:
        logger.info("Ignoring stale WPM cache schema in %s", cache_path)
        return None
    if cached_fingerprint != fingerprint:
        logger.warning("Ignoring mismatched WPM cache fingerprint in %s", cache_path)
        return None
    if not isinstance(calibration_payload, dict):
        logger.warning("Ignoring WPM cache missing calibration body: %s", cache_path)
        return None
    try:
        return TTSPacingCalibration.from_dict(calibration_payload)
    except ValueError:
        logger.warning("Ignoring invalid WPM cache calibration payload: %s", cache_path)
        return None


def save_cached_tts_pacing(
    *,
    cache_dir: str,
    fingerprint: str,
    calibration: TTSPacingCalibration,
) -> Path:
    """Persist one TTS pacing calibration payload to cache storage.

    Args:
        cache_dir: Directory where cache JSON files are stored.
        fingerprint: Fingerprint key for this cache entry.
        calibration: Calibration object to serialize.

    Returns:
        Absolute written cache file path.
    """
    cache_path = cache_path_for_fingerprint(cache_dir=cache_dir, fingerprint=fingerprint)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _CalibrationCacheRecord(
        schema_version=CALIBRATION_CACHE_SCHEMA_VERSION,
        fingerprint=fingerprint,
        generated_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        calibration=calibration.to_dict(),
    )
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return cache_path
