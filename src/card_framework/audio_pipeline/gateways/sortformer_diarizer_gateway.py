"""NeMo Sortformer adapters for end-to-end speaker diarization."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from card_framework.audio_pipeline.contracts import DiarizationTurn, SpeakerDiarizer
from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from card_framework.audio_pipeline.errors import DependencyMissingError, NonRetryableAudioStageError
from card_framework.audio_pipeline.gateways.diarization_common import (
    normalize_speaker_labels,
    prepare_diarization_audio,
)


class SortformerSpeakerDiarizer(SpeakerDiarizer):
    """Run end-to-end diarization with NeMo Sortformer checkpoints."""

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: str | None = None,
        streaming_mode: bool = False,
        batch_size: int = 1,
        chunk_len: int = 340,
        chunk_right_context: int = 40,
        fifo_len: int = 40,
        spkcache_update_period: int = 300,
        spkcache_len: int = 188,
    ) -> None:
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.streaming_mode = streaming_mode
        self.batch_size = batch_size
        self.chunk_len = chunk_len
        self.chunk_right_context = chunk_right_context
        self.fifo_len = fifo_len
        self.spkcache_update_period = spkcache_update_period
        self.spkcache_len = spkcache_len
        self._model_cache: dict[str, Any] = {}

    def diarize(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        device: str,
        progress_callback: StageProgressCallback | None = None,
    ) -> list[DiarizationTurn]:
        """Run diarization and return normalized turns."""
        if not audio_path.exists():
            raise NonRetryableAudioStageError(f"Audio file does not exist: {audio_path}")

        if progress_callback is not None:
            self._emit_progress(
                progress_callback,
                note="sortformer diarization started",
                completed_units=0,
            )

        prepared_audio_path = prepare_diarization_audio(
            audio_path=audio_path,
            output_dir=output_dir,
        )
        model = self._load_model(device=device)
        predicted_segments = self._run_inference(
            model=model,
            audio_path=prepared_audio_path,
        )
        turns = self._segments_to_turns(predicted_segments)
        if not turns:
            raise NonRetryableAudioStageError(
                "Sortformer diarization returned zero speaker turns."
            )

        if progress_callback is not None:
            self._emit_progress(
                progress_callback,
                note="sortformer diarization completed",
                completed_units=1,
            )
        return turns

    def _load_model(self, *, device: str) -> Any:
        """Lazy-load and configure the Sortformer model for one device."""
        cached = self._model_cache.get(device)
        if cached is not None:
            return cached

        try:
            from nemo.collections.asr.models import (  # type: ignore[import-not-found]
                SortformerEncLabelModel,
            )
        except Exception as exc:
            raise DependencyMissingError(
                "NeMo Sortformer runtime is unavailable."
            ) from exc

        checkpoint_path = Path(self.checkpoint_path).expanduser() if self.checkpoint_path else None
        try:
            if checkpoint_path is not None and checkpoint_path.exists():
                model = SortformerEncLabelModel.restore_from(
                    restore_path=str(checkpoint_path.resolve()),
                    map_location=device,
                    strict=False,
                )
            else:
                model = SortformerEncLabelModel.from_pretrained(self.model_name)
        except Exception as exc:
            raise NonRetryableAudioStageError(
                f"Failed to load Sortformer checkpoint '{self.model_name}'."
            ) from exc

        try:
            import torch
        except Exception as exc:
            raise DependencyMissingError(
                "PyTorch runtime is unavailable for Sortformer diarization."
            ) from exc

        target_device = torch.device("cuda" if device == "cuda" else "cpu")
        model = model.to(target_device)
        model.eval()

        if self.streaming_mode:
            self._configure_streaming(model)

        self._model_cache[device] = model
        return model

    def _configure_streaming(self, model: Any) -> None:
        """Apply streaming parameters from config when the model supports them."""
        modules = getattr(model, "sortformer_modules", None)
        if modules is None:
            raise NonRetryableAudioStageError(
                "Streaming Sortformer model is missing sortformer_modules."
            )
        modules.chunk_len = int(self.chunk_len)
        modules.chunk_right_context = int(self.chunk_right_context)
        modules.fifo_len = int(self.fifo_len)
        modules.spkcache_update_period = int(self.spkcache_update_period)
        modules.spkcache_len = int(self.spkcache_len)
        check_streaming_parameters = getattr(modules, "_check_streaming_parameters", None)
        if callable(check_streaming_parameters):
            check_streaming_parameters()

    def _run_inference(self, *, model: Any, audio_path: Path) -> Any:
        """Execute one inference call on the prepared audio path."""
        try:
            predicted_segments = model.diarize(
                audio=[str(audio_path)],
                batch_size=int(self.batch_size),
            )
        except TypeError:
            predicted_segments = model.diarize(
                audio=str(audio_path),
                batch_size=int(self.batch_size),
            )
        except Exception as exc:
            raise NonRetryableAudioStageError("Sortformer diarization failed.") from exc

        if (
            isinstance(predicted_segments, list)
            and len(predicted_segments) == 1
            and isinstance(predicted_segments[0], list)
        ):
            return predicted_segments[0]
        return predicted_segments

    def _segments_to_turns(self, segments: Any) -> list[DiarizationTurn]:
        """Normalize model outputs into repo-standard diarization turns."""
        if not isinstance(segments, list):
            raise NonRetryableAudioStageError(
                "Sortformer diarization returned an unexpected payload type."
            )

        raw_turns: list[DiarizationTurn] = []
        for segment in segments:
            parsed = self._parse_segment(segment)
            if parsed is not None:
                raw_turns.append(parsed)
        return normalize_speaker_labels(raw_turns)

    def _parse_segment(self, segment: Any) -> DiarizationTurn | None:
        """Parse one segment record from supported Sortformer output shapes."""
        if isinstance(segment, (list, tuple)) and len(segment) >= 3:
            return self._build_turn(
                start_value=segment[0],
                end_value=segment[1],
                speaker_value=segment[2],
            )
        if isinstance(segment, dict):
            if {"start", "end", "speaker"} <= set(segment):
                return self._build_turn(
                    start_value=segment["start"],
                    end_value=segment["end"],
                    speaker_value=segment["speaker"],
                )
            if {"start_time_ms", "end_time_ms", "speaker"} <= set(segment):
                return DiarizationTurn(
                    speaker=str(segment["speaker"]).strip() or "UNKNOWN",
                    start_time_ms=max(0, int(segment["start_time_ms"])),
                    end_time_ms=max(0, int(segment["end_time_ms"])),
                )
        if isinstance(segment, str):
            cleaned = segment.replace(",", " ").replace("(", " ").replace(")", " ")
            parts = [part for part in cleaned.split() if part]
            if len(parts) >= 3:
                return self._build_turn(
                    start_value=parts[0],
                    end_value=parts[1],
                    speaker_value=parts[2],
                )
        return None

    def _build_turn(
        self,
        *,
        start_value: Any,
        end_value: Any,
        speaker_value: Any,
    ) -> DiarizationTurn:
        """Build one normalized turn from mixed seconds/speaker payloads."""
        start_seconds = float(start_value)
        end_seconds = float(end_value)
        if end_seconds < start_seconds:
            end_seconds = start_seconds

        speaker_text = str(speaker_value).strip()
        speaker_match = re.search(r"(\d+)$", speaker_text)
        if speaker_match is not None:
            speaker_text = f"SPEAKER_{int(speaker_match.group(1)):02d}"
        elif not speaker_text:
            speaker_text = "SPEAKER_00"

        return DiarizationTurn(
            speaker=speaker_text,
            start_time_ms=max(0, int(round(start_seconds * 1000))),
            end_time_ms=max(0, int(round(end_seconds * 1000))),
        )

    def _emit_progress(
        self,
        progress_callback: StageProgressCallback,
        *,
        note: str,
        completed_units: int,
    ) -> None:
        """Best-effort progress emission wrapper."""
        try:
            progress_callback(
                StageProgressUpdate(
                    completed_units=completed_units,
                    total_units=1,
                    note=note,
                )
            )
        except Exception:
            return

