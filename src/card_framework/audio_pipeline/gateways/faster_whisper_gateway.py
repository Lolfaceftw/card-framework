"""Faster-Whisper adapter for timed ASR segments."""

from __future__ import annotations

from pathlib import Path
import re

from card_framework.audio_pipeline.contracts import (
    SpeechTranscriber,
    TimedTextSegment,
    TranscriptionResult,
    WordTimestamp,
)
from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from card_framework.audio_pipeline.errors import DependencyMissingError, NonRetryableAudioStageError
from card_framework.audio_pipeline.runtime import ensure_command_available
from card_framework.audio_pipeline.gateways.ctc_forced_aligner_gateway import CtcForcedAlignerGateway


class FasterWhisperTranscriber(SpeechTranscriber):
    """Transcribe audio with Faster-Whisper."""

    def __init__(
        self,
        *,
        model_name: str = "large-v3",
        compute_type_cuda: str = "int8_float16",
        compute_type_cpu: str = "int8",
        beam_size: int = 5,
        vad_filter: bool = True,
        batch_size: int = 8,
        language: str | None = None,
        suppress_numerals: bool = False,
        enable_forced_alignment: bool = True,
        require_forced_alignment: bool = True,
        forced_alignment_batch_size: int = 8,
        forced_aligner: CtcForcedAlignerGateway | None = None,
    ) -> None:
        self.model_name = model_name
        self.compute_type_cuda = compute_type_cuda
        self.compute_type_cpu = compute_type_cpu
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.batch_size = batch_size
        self.language = language
        self.suppress_numerals = suppress_numerals
        self.enable_forced_alignment = enable_forced_alignment
        self.require_forced_alignment = require_forced_alignment
        self.forced_alignment_batch_size = forced_alignment_batch_size
        self.forced_aligner = forced_aligner or CtcForcedAlignerGateway()
        self._model_cache: dict[tuple[str, str], object] = {}

    def transcribe(
        self,
        audio_path: Path,
        *,
        device: str,
        progress_callback: StageProgressCallback | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio and return normalized timed segments.

        Args:
            audio_path: Input audio path.
            device: Runtime device (``cpu`` or ``cuda``).
            progress_callback: Optional callback for progress updates.

        Returns:
            Transcription result with segment-level and word-level timings.
        """
        ensure_command_available("ffmpeg")
        ensure_command_available("ffprobe")
        model = self._get_model(device=device)

        try:
            from faster_whisper import BatchedInferencePipeline, decode_audio  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - runtime dependency path
            raise DependencyMissingError(
                "faster-whisper runtime helpers are unavailable."
            ) from exc

        audio_waveform = decode_audio(str(audio_path))
        suppress_tokens = (
            self._find_numeral_symbol_tokens(getattr(model, "hf_tokenizer", None))
            if self.suppress_numerals
            else [-1]
        )

        if self.batch_size > 0:
            pipeline = BatchedInferencePipeline(model)
            segments, info = pipeline.transcribe(
                audio_waveform,
                language=self.language,
                suppress_tokens=suppress_tokens,
                batch_size=self.batch_size,
            )
        else:
            segments, info = model.transcribe(
                audio_waveform,
                language=self.language,
                suppress_tokens=suppress_tokens,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                word_timestamps=False,
            )

        normalized: list[TimedTextSegment] = []
        max_processed_audio_ms = 0
        full_text_parts: list[str] = []
        for segment in segments:
            text = str(segment.text or "").strip()
            if not text:
                continue
            start_time_ms = max(0, int(round(float(segment.start) * 1000)))
            end_time_ms = max(start_time_ms, int(round(float(segment.end) * 1000)))
            max_processed_audio_ms = max(max_processed_audio_ms, end_time_ms)
            full_text_parts.append(text)
            normalized.append(
                TimedTextSegment(
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                    text=text,
                )
            )
            if progress_callback is not None:
                try:
                    progress_callback(
                        StageProgressUpdate(processed_audio_ms=end_time_ms)
                    )
                except Exception:
                    pass

        if not normalized:
            raise NonRetryableAudioStageError(
                "Faster-Whisper returned no transcript segments."
            )

        language = str(getattr(info, "language", "") or self.language or "en")
        words = self._resolve_word_timestamps(
            audio_waveform=audio_waveform,
            transcript_text=" ".join(full_text_parts),
            language=language,
            device=device,
            fallback_segments=normalized,
        )

        if progress_callback is not None:
            try:
                progress_callback(
                    StageProgressUpdate(processed_audio_ms=max_processed_audio_ms)
                )
            except Exception:
                pass
        return TranscriptionResult(
            segments=normalized,
            word_timestamps=words,
            language=language,
        )

    def _resolve_word_timestamps(
        self,
        *,
        audio_waveform: object,
        transcript_text: str,
        language: str,
        device: str,
        fallback_segments: list[TimedTextSegment],
    ) -> list[WordTimestamp]:
        """Resolve word-level timestamps from forced alignment when enabled."""
        if not self.enable_forced_alignment:
            return self._fallback_word_timestamps(fallback_segments)

        try:
            words = self.forced_aligner.align_words(
                audio_waveform=audio_waveform,
                transcript_text=transcript_text,
                language=language,
                device=device,
                batch_size=self.forced_alignment_batch_size,
            )
        except Exception:
            if self.require_forced_alignment:
                raise
            return self._fallback_word_timestamps(fallback_segments)

        if words:
            return words
        if self.require_forced_alignment:
            raise NonRetryableAudioStageError(
                "Forced alignment completed but returned zero word timestamps."
            )
        return self._fallback_word_timestamps(fallback_segments)

    def _fallback_word_timestamps(
        self,
        segments: list[TimedTextSegment],
    ) -> list[WordTimestamp]:
        """Best-effort word timestamps when forced alignment is unavailable."""
        words: list[WordTimestamp] = []
        for segment in segments:
            raw_words = [word for word in re.split(r"\s+", segment.text.strip()) if word]
            if not raw_words:
                continue
            total_duration = max(1, segment.end_time_ms - segment.start_time_ms)
            unit = max(1, total_duration // len(raw_words))
            cursor = segment.start_time_ms
            for index, raw_word in enumerate(raw_words):
                end_time = (
                    segment.end_time_ms
                    if index == len(raw_words) - 1
                    else min(segment.end_time_ms, cursor + unit)
                )
                words.append(
                    WordTimestamp(
                        word=raw_word,
                        start_time_ms=cursor,
                        end_time_ms=end_time,
                    )
                )
                cursor = end_time
        return words

    def _get_model(self, *, device: str):
        """Lazy-load and cache whisper model per device/compute type."""
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - import environment dependent
            raise DependencyMissingError(
                "faster-whisper is not installed or failed to import."
            ) from exc

        compute_type = (
            self.compute_type_cuda if device == "cuda" else self.compute_type_cpu
        )
        cache_key = (device, compute_type)
        if cache_key in self._model_cache:
            return self._model_cache[cache_key]

        model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute_type,
        )
        self._model_cache[cache_key] = model
        return model

    def _find_numeral_symbol_tokens(self, tokenizer: object | None) -> list[int]:
        """Identify token ids containing numerals/symbols for suppression."""
        if tokenizer is None:
            return [-1]
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if not callable(get_vocab):
            return [-1]
        try:
            vocab = get_vocab()
        except Exception:
            return [-1]
        symbol_pattern = re.compile(r"[0-9$%Â£â‚¬Â¥â‚±]")
        token_ids = [
            int(token_id)
            for token, token_id in vocab.items()
            if symbol_pattern.search(str(token))
        ]
        return token_ids or [-1]

