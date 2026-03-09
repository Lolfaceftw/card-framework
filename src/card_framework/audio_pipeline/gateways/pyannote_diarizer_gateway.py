"""pyannote.audio adapter for speaker diarization."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from card_framework.audio_pipeline.contracts import DiarizationTurn, SpeakerDiarizer
from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from card_framework.audio_pipeline.errors import DependencyMissingError, NonRetryableAudioStageError
from card_framework.audio_pipeline.gateways.diarization_common import (
    normalize_speaker_labels,
    prepare_diarization_audio,
)


class PyannoteSpeakerDiarizer(SpeakerDiarizer):
    """Run speaker diarization with a ``pyannote.audio`` pipeline."""

    def __init__(
        self,
        *,
        pipeline_name: str = "pyannote/speaker-diarization-community-1",
        auth_token: str | None = None,
        auth_token_env: str = "HUGGINGFACE_TOKEN",
        use_exclusive_diarization: bool = True,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> None:
        self.pipeline_name = pipeline_name
        self.auth_token = auth_token
        self.auth_token_env = auth_token_env
        self.use_exclusive_diarization = use_exclusive_diarization
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline_cache: dict[str, Any] = {}

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
                note="pyannote diarization started",
                completed_units=0,
            )

        prepared_audio_path = prepare_diarization_audio(
            audio_path=audio_path,
            output_dir=output_dir,
        )
        annotation = self._run_pipeline(audio_path=prepared_audio_path, device=device)
        turns = self._annotation_to_turns(annotation)
        if not turns:
            raise NonRetryableAudioStageError(
                "pyannote diarization returned zero speaker turns."
            )

        if progress_callback is not None:
            self._emit_progress(
                progress_callback,
                note="pyannote diarization completed",
                completed_units=1,
            )
        return turns

    def _run_pipeline(self, *, audio_path: Path, device: str) -> Any:
        """Execute the configured pyannote pipeline."""
        pipeline = self._pipeline_cache.get(device)
        if pipeline is None:
            try:
                from pyannote.audio import Pipeline  # type: ignore[import-not-found]
            except Exception as exc:
                raise DependencyMissingError(
                    "pyannote.audio is not installed or failed to import."
                ) from exc

            token = self._resolve_auth_token()
            try:
                if token is not None:
                    pipeline = Pipeline.from_pretrained(
                        self.pipeline_name,
                        token=token,
                    )
                else:
                    pipeline = Pipeline.from_pretrained(self.pipeline_name)
            except TypeError:
                if token is not None:
                    pipeline = Pipeline.from_pretrained(
                        self.pipeline_name,
                        use_auth_token=token,
                    )
                else:
                    pipeline = Pipeline.from_pretrained(self.pipeline_name)
            except Exception as exc:
                raise NonRetryableAudioStageError(
                    f"Failed to load pyannote pipeline '{self.pipeline_name}'."
                ) from exc

            if device == "cuda":
                try:
                    import torch
                except Exception as exc:
                    raise DependencyMissingError(
                        "PyTorch CUDA runtime is unavailable for pyannote diarization."
                    ) from exc
                pipeline.to(torch.device("cuda"))
            self._pipeline_cache[device] = pipeline

        call_kwargs: dict[str, int] = {}
        if self.min_speakers is not None:
            call_kwargs["min_speakers"] = int(self.min_speakers)
        if self.max_speakers is not None:
            call_kwargs["max_speakers"] = int(self.max_speakers)

        try:
            output = pipeline(str(audio_path), **call_kwargs)
        except Exception as exc:
            raise NonRetryableAudioStageError("pyannote diarization failed.") from exc

        if (
            self.use_exclusive_diarization
            and hasattr(output, "exclusive_speaker_diarization")
        ):
            return output.exclusive_speaker_diarization
        if hasattr(output, "speaker_diarization"):
            return output.speaker_diarization
        return output

    def _resolve_auth_token(self) -> str | None:
        """Resolve optional Hugging Face token from config or environment."""
        explicit_token = (self.auth_token or "").strip()
        if explicit_token:
            return explicit_token
        env_name = self.auth_token_env.strip()
        if env_name:
            env_token = os.getenv(env_name, "").strip()
            if env_token:
                return env_token
        return None

    def _annotation_to_turns(self, annotation: Any) -> list[DiarizationTurn]:
        """Convert pyannote annotations into normalized diarization turns."""
        itertracks = getattr(annotation, "itertracks", None)
        if not callable(itertracks):
            raise NonRetryableAudioStageError(
                "pyannote diarization output does not expose itertracks()."
            )

        raw_turns: list[DiarizationTurn] = []
        for segment, _, speaker in itertracks(yield_label=True):
            speaker_label = str(speaker).strip() or "UNKNOWN"
            start_time_ms = max(0, int(round(float(segment.start) * 1000)))
            end_time_ms = max(start_time_ms, int(round(float(segment.end) * 1000)))
            raw_turns.append(
                DiarizationTurn(
                    speaker=speaker_label,
                    start_time_ms=start_time_ms,
                    end_time_ms=end_time_ms,
                )
            )
        return normalize_speaker_labels(raw_turns)

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

