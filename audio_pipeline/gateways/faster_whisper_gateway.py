"""Faster-Whisper adapter for timed ASR segments."""

from __future__ import annotations

from pathlib import Path

from audio_pipeline.contracts import SpeechTranscriber, TimedTextSegment
from audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from audio_pipeline.errors import DependencyMissingError, NonRetryableAudioStageError
from audio_pipeline.runtime import ensure_command_available


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
    ) -> None:
        self.model_name = model_name
        self.compute_type_cuda = compute_type_cuda
        self.compute_type_cpu = compute_type_cpu
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self._model_cache: dict[tuple[str, str], object] = {}

    def transcribe(
        self,
        audio_path: Path,
        *,
        device: str,
        progress_callback: StageProgressCallback | None = None,
    ) -> list[TimedTextSegment]:
        """
        Transcribe audio and return normalized timed segments.

        Args:
            audio_path: Input audio path.
            device: Runtime device (``cpu`` or ``cuda``).
            progress_callback: Optional callback for progress updates.

        Returns:
            List of timed text segments.
        """
        ensure_command_available("ffmpeg")
        model = self._get_model(device=device)
        segments, _ = model.transcribe(
            str(audio_path),
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            word_timestamps=False,
        )

        normalized: list[TimedTextSegment] = []
        max_processed_audio_ms = 0
        for segment in segments:
            text = str(segment.text or "").strip()
            if not text:
                continue
            start_time_ms = max(0, int(round(float(segment.start) * 1000)))
            end_time_ms = max(start_time_ms, int(round(float(segment.end) * 1000)))
            max_processed_audio_ms = max(max_processed_audio_ms, end_time_ms)
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
        if progress_callback is not None:
            try:
                progress_callback(
                    StageProgressUpdate(processed_audio_ms=max_processed_audio_ms)
                )
            except Exception:
                pass
        return normalized

    def _get_model(self, *, device: str):
        """Lazy-load and cache whisper model per device/compute type."""
        try:
            from faster_whisper import WhisperModel
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
