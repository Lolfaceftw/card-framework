"""Demucs adapter for vocal source separation."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

from card_framework.audio_pipeline.contracts import SourceSeparator
from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from card_framework.audio_pipeline.errors import NonRetryableAudioStageError, RetryableAudioStageError
from card_framework.audio_pipeline.runtime import ensure_command_available, ensure_module_available


class DemucsSourceSeparator(SourceSeparator):
    """
    Separate vocal stem using Demucs ``htdemucs``.

    The adapter invokes Demucs through ``python -m demucs.separate`` so behavior is
    consistent across Windows/macOS/Linux and independent of shell syntax.
    """

    def __init__(
        self,
        *,
        model_name: str = "htdemucs",
        timeout_seconds: int = 1800,
        max_retries: int = 2,
        retry_base_seconds: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds

    def separate_vocals(
        self,
        input_audio_path: Path,
        output_dir: Path,
        *,
        device: str,
        progress_callback: StageProgressCallback | None = None,
    ) -> Path:
        """
        Separate input audio and return vocals file path.

        Args:
            input_audio_path: Path to source audio file.
            output_dir: Directory where demucs outputs stems.
            device: Runtime device (``cpu`` or ``cuda``).
            progress_callback: Optional callback for progress updates.

        Returns:
            Path to generated ``vocals.wav`` file.
        """
        if not input_audio_path.exists():
            raise NonRetryableAudioStageError(
                f"Audio file does not exist: {input_audio_path}"
            )

        ensure_module_available("demucs")
        ensure_command_available("ffmpeg")
        if progress_callback is not None:
            try:
                progress_callback(
                    StageProgressUpdate(
                        completed_units=0,
                        total_units=1,
                        note="demucs separation started",
                    )
                )
            except Exception:
                pass

        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "demucs.separate",
            "--two-stems",
            "vocals",
            "-n",
            self.model_name,
            "-o",
            str(output_dir),
            str(input_audio_path),
        ]
        if device in {"cpu", "cuda"}:
            command.extend(["--device", device])

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=self.timeout_seconds,
                )
                break
            except subprocess.TimeoutExpired as exc:
                last_error = exc
                if attempt == self.max_retries:
                    raise RetryableAudioStageError(
                        "Demucs separation timed out."
                    ) from exc
                time.sleep(self.retry_base_seconds * (2 ** (attempt - 1)))
            except subprocess.CalledProcessError as exc:
                stderr = (exc.stderr or "").strip()
                raise NonRetryableAudioStageError(
                    "Demucs separation failed. "
                    f"Command: {' '.join(command)}. Stderr: {stderr[:500]}"
                ) from exc

        vocals_path = output_dir / self.model_name / input_audio_path.stem / "vocals.wav"
        if vocals_path.exists():
            if progress_callback is not None:
                try:
                    progress_callback(
                        StageProgressUpdate(
                            completed_units=1,
                            total_units=1,
                            note="demucs separation finished",
                        )
                    )
                except Exception:
                    pass
            return vocals_path

        fallback = next(output_dir.rglob("vocals.wav"), None)
        if fallback is not None and fallback.exists():
            if progress_callback is not None:
                try:
                    progress_callback(
                        StageProgressUpdate(
                            completed_units=1,
                            total_units=1,
                            note="demucs separation finished",
                        )
                    )
                except Exception:
                    pass
            return fallback

        raise NonRetryableAudioStageError(
            f"Demucs completed but vocals stem was not found in '{output_dir}'."
        ) from last_error

