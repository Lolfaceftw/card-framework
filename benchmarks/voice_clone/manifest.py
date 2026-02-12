"""Manifest loading and validation for voice clone benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.voice_clone.types import ManifestItem, RawManifestItem
from benchmarks.voice_clone.utils import resolve_existing_path


def load_manifest(manifest_path: Path, *, max_items: int | None = None) -> list[ManifestItem]:
    """Load and validate benchmark manifest rows.

    Args:
        manifest_path: Manifest JSON file path.
        max_items: Optional row limit for smoke runs.

    Returns:
        Validated manifest rows.

    Raises:
        ValueError: Manifest shape or required fields are invalid.
        FileNotFoundError: Referenced audio files do not exist.
    """
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Manifest JSON must be a list.")
    if not payload:
        raise ValueError("Manifest JSON is empty.")

    base_dir = manifest_path.parent
    items: list[ManifestItem] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"Manifest entry {index} must be an object.")
        raw = RawManifestItem(**row)
        speaker_id = str(raw.get("speaker_id", "")).strip()
        prompt_wav_raw = str(raw.get("prompt_wav", "")).strip()
        reference_wav_raw = str(raw.get("reference_wav", "")).strip()
        text = str(raw.get("text", "")).strip()
        if not speaker_id or not prompt_wav_raw or not reference_wav_raw or not text:
            raise ValueError(
                "Manifest entry must include speaker_id, prompt_wav, reference_wav, and text."
            )
        emo_alpha_raw = raw.get("emo_alpha", 0.6)
        try:
            emo_alpha = float(emo_alpha_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid emo_alpha at entry {index}: {emo_alpha_raw}") from exc

        items.append(
            ManifestItem(
                item_id=index,
                speaker_id=speaker_id,
                prompt_wav=resolve_existing_path(prompt_wav_raw, base_dir),
                reference_wav=resolve_existing_path(reference_wav_raw, base_dir),
                text=text,
                use_emo_text=bool(raw.get("use_emo_text", False)),
                emo_text=str(raw.get("emo_text", "")).strip(),
                emo_alpha=max(0.0, min(1.0, emo_alpha)),
            )
        )
    if max_items is not None:
        if max_items < 1:
            raise ValueError("--max-items must be >= 1 when provided.")
        items = items[:max_items]
    if not items:
        raise ValueError("Manifest contains no entries after applying max-items.")
    return items

