"""CTC forced-alignment adapter for word-level timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from card_framework.audio_pipeline.contracts import WordTimestamp
from card_framework.audio_pipeline.errors import DependencyMissingError, NonRetryableAudioStageError
from card_framework.audio_pipeline.runtime import ensure_module_available

_LANGS_TO_ISO: dict[str, str] = {
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "ru": "rus",
    "uk": "ukr",
    "zh": "chi",
    "ja": "jpn",
    "ko": "kor",
    "ar": "ara",
    "hi": "hin",
    "tr": "tur",
    "nl": "nld",
    "pl": "pol",
    "sv": "swe",
    "no": "nor",
    "da": "dan",
    "fi": "fin",
    "cs": "ces",
    "el": "ell",
    "id": "ind",
    "vi": "vie",
    "th": "tha",
    "tl": "tgl",
}


@dataclass(slots=True, frozen=True)
class CtcForcedAlignerGateway:
    """Produce aligned word timestamps using the ctc-forced-aligner package."""

    def align_words(
        self,
        *,
        audio_waveform: Any,
        transcript_text: str,
        language: str,
        device: str,
        batch_size: int,
    ) -> list[WordTimestamp]:
        """
        Align words to millisecond timestamps.

        Args:
            audio_waveform: Decoded mono waveform from Faster-Whisper.
            transcript_text: Raw transcript text to align.
            language: Detected language code (e.g., ``en``).
            device: Runtime device (``cpu`` or ``cuda``).
            batch_size: Alignment batch size.

        Returns:
            Ordered word-level timestamps.
        """
        normalized_text = transcript_text.strip()
        if not normalized_text:
            return []

        ensure_module_available("ctc_forced_aligner")
        ensure_module_available("torch")

        try:
            import torch  # type: ignore[import-not-found]
            from ctc_forced_aligner import (  # type: ignore[import-not-found]
                generate_emissions,
                get_alignments,
                get_spans,
                load_alignment_model,
                postprocess_results,
                preprocess_text,
            )
        except Exception as exc:  # pragma: no cover - import env dependent
            raise DependencyMissingError(
                "Failed to import ctc-forced-aligner dependencies."
            ) from exc

        iso_language = _LANGS_TO_ISO.get(language.lower().strip(), "eng")
        dtype = torch.float16 if device == "cuda" else torch.float32
        try:
            alignment_model, alignment_tokenizer = load_alignment_model(device, dtype=dtype)
            emissions, stride = generate_emissions(
                alignment_model,
                torch.from_numpy(audio_waveform)
                .to(alignment_model.dtype)
                .to(alignment_model.device),
                batch_size=max(1, batch_size),
            )
            del alignment_model
            if device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            tokens_starred, text_starred = preprocess_text(
                normalized_text,
                romanize=True,
                language=iso_language,
            )
            segments, scores, blank_token = get_alignments(
                emissions,
                tokens_starred,
                alignment_tokenizer,
            )
            spans = get_spans(tokens_starred, segments, blank_token)
            aligned = postprocess_results(text_starred, spans, stride, scores)
        except Exception as exc:  # pragma: no cover - runtime path
            raise NonRetryableAudioStageError(
                "CTC forced alignment failed."
            ) from exc

        words: list[WordTimestamp] = []
        for raw in aligned:
            text_value = str(raw.get("text") or raw.get("word") or "").strip()
            if not text_value:
                continue
            start_seconds = raw.get("start")
            end_seconds = raw.get("end")
            if start_seconds is None or end_seconds is None:
                continue
            start_ms = max(0, int(round(float(start_seconds) * 1000)))
            end_ms = max(start_ms, int(round(float(end_seconds) * 1000)))
            words.append(
                WordTimestamp(
                    word=text_value,
                    start_time_ms=start_ms,
                    end_time_ms=end_ms,
                )
            )
        return words

