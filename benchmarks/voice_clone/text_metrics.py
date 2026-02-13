"""ASR-backed text-fidelity scoring helpers for voice-clone benchmarks."""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from pathlib import Path

from benchmarks.voice_clone.types import ManifestItem, SpeechTranscriberProtocol, TextScoreRow

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim leading/trailing spaces."""
    return " ".join(text.strip().split())


def normalize_text_for_wer(text: str) -> list[str]:
    """Normalize free-form text and return token list for WER computation."""
    lowered = unicodedata.normalize("NFKC", text).lower()
    alnum_only = re.sub(r"[^\w\s]", " ", lowered)
    normalized = _normalize_whitespace(alnum_only)
    if not normalized:
        return []
    return normalized.split(" ")


def normalize_text_for_cer(text: str) -> list[str]:
    """Normalize free-form text and return character sequence for CER."""
    lowered = unicodedata.normalize("NFKC", text).lower()
    alnum_only = re.sub(r"[^\w\s]", "", lowered)
    normalized = _normalize_whitespace(alnum_only).replace(" ", "")
    return list(normalized)


def _levenshtein_distance(left: list[str], right: list[str]) -> int:
    """Compute edit distance between token sequences."""
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous_row = list(range(len(right) + 1))
    for left_index, left_token in enumerate(left, start=1):
        current_row = [left_index]
        for right_index, right_token in enumerate(right, start=1):
            insertion = current_row[right_index - 1] + 1
            deletion = previous_row[right_index] + 1
            substitution = previous_row[right_index - 1] + (
                0 if left_token == right_token else 1
            )
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]


def compute_word_error_rate(reference_text: str, hypothesis_text: str) -> float:
    """Compute WER using normalized word tokens."""
    reference_tokens = normalize_text_for_wer(reference_text)
    hypothesis_tokens = normalize_text_for_wer(hypothesis_text)
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0
    distance = _levenshtein_distance(reference_tokens, hypothesis_tokens)
    return float(distance / len(reference_tokens))


def compute_char_error_rate(reference_text: str, hypothesis_text: str) -> float:
    """Compute CER using normalized character sequences."""
    reference_chars = normalize_text_for_cer(reference_text)
    hypothesis_chars = normalize_text_for_cer(hypothesis_text)
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    distance = _levenshtein_distance(reference_chars, hypothesis_chars)
    return float(distance / len(reference_chars))


class FasterWhisperTranscriber(SpeechTranscriberProtocol):
    """Decode generated speech using faster-whisper."""

    def __init__(self, *, model_size: str, device: str) -> None:
        """Initialize transcriber settings.

        Args:
            model_size: Faster-whisper model identifier, such as ``small``.
            device: Runtime device string (for example ``cpu`` or ``cuda:0``).
        """
        self._model_size = model_size.strip() or "small"
        self._requested_device = device.strip() or "cpu"
        self._model: object | None = None

    def _ensure_loaded(self) -> object:
        """Load whisper model on first use."""
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "Missing optional dependency 'faster-whisper' for text metrics."
            ) from exc

        device_name = self._requested_device
        device_index: int | None = None
        if ":" in self._requested_device:
            maybe_name, maybe_index = self._requested_device.split(":", maxsplit=1)
            device_name = maybe_name
            try:
                device_index = int(maybe_index)
            except ValueError:
                device_index = None
        device_name = device_name.lower()
        compute_type = "float16" if device_name.startswith("cuda") else "int8"

        kwargs: dict[str, object] = {
            "device": device_name,
            "compute_type": compute_type,
        }
        if device_index is not None:
            kwargs["device_index"] = device_index
        self._model = WhisperModel(self._model_size, **kwargs)
        logger.info(
            "text_metrics_transcriber_ready backend=faster_whisper model=%s device=%s",
            self._model_size,
            self._requested_device,
        )
        return self._model

    def transcribe(self, wav_path: Path) -> str:
        """Return decoded transcript for one WAV file."""
        model = self._ensure_loaded()
        segments, _ = model.transcribe(str(wav_path), beam_size=5, vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments if segment.text.strip())


def make_text_score_rows(
    *,
    items: list[ManifestItem],
    generated_paths: dict[int, Path],
    transcriber: SpeechTranscriberProtocol,
) -> list[TextScoreRow]:
    """Build per-item WER/CER rows from ASR transcripts."""
    rows: list[TextScoreRow] = []
    for item in items:
        generated_path = generated_paths[item.item_id]
        hypothesis_text = transcriber.transcribe(generated_path)
        rows.append(
            TextScoreRow(
                item_id=item.item_id,
                speaker_id=item.speaker_id,
                generated_wav=generated_path,
                reference_text=item.text,
                hypothesis_text=hypothesis_text,
                wer=compute_word_error_rate(item.text, hypothesis_text),
                cer=compute_char_error_rate(item.text, hypothesis_text),
            )
        )
    return rows


def write_text_scores_csv(rows: list[TextScoreRow], output_path: Path) -> None:
    """Write per-item text-fidelity scores to CSV."""
    fieldnames = [
        "item_id",
        "speaker_id",
        "generated_wav",
        "reference_text",
        "hypothesis_text",
        "wer",
        "cer",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "item_id": row.item_id,
                    "speaker_id": row.speaker_id,
                    "generated_wav": str(row.generated_wav),
                    "reference_text": row.reference_text,
                    "hypothesis_text": row.hypothesis_text,
                    "wer": f"{row.wer:.8f}",
                    "cer": f"{row.cer:.8f}",
                }
            )
