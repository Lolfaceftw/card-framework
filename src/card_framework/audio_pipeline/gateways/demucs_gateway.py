"""Demucs adapter for vocal source separation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
import time

from card_framework.audio_pipeline.contracts import SourceSeparator
from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from card_framework.audio_pipeline.errors import NonRetryableAudioStageError, RetryableAudioStageError
from card_framework.audio_pipeline.runtime import (
    ensure_command_available,
    ensure_module_available,
    probe_audio_duration_ms,
)
from card_framework.shared.events import event_bus
from card_framework.shared.perf import report_loading_duration

_TEMP_OUTPUT_SUFFIX = ".tmp.wav"


@dataclass(slots=True, frozen=True)
class _SeparationWindow:
    """Describe one base/context window for long-audio Demucs runs."""

    base_start_ms: int
    base_end_ms: int
    context_start_ms: int
    context_end_ms: int

    @property
    def base_duration_ms(self) -> int:
        """Return the base window duration in milliseconds."""
        return max(0, self.base_end_ms - self.base_start_ms)


class DemucsSourceSeparator(SourceSeparator):
    """
    Separate vocal stem using Demucs ``htdemucs``.

    The adapter invokes Demucs through ``python -m demucs.separate`` so behavior is
    consistent across Windows/macOS/Linux and independent of shell syntax. For
    long recordings it can split the source into outer windows first so peak RAM
    usage scales with one window at a time instead of the entire file.
    """

    def __init__(
        self,
        *,
        model_name: str = "htdemucs",
        timeout_seconds: int = 1800,
        max_retries: int = 2,
        retry_base_seconds: float = 1.0,
        window_length_seconds: int = 600,
        window_context_seconds: int = 5,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0.")
        if max_retries <= 0:
            raise ValueError("max_retries must be > 0.")
        if retry_base_seconds <= 0:
            raise ValueError("retry_base_seconds must be > 0.")
        if window_length_seconds <= 0:
            raise ValueError("window_length_seconds must be > 0.")
        if window_context_seconds < 0:
            raise ValueError("window_context_seconds must be >= 0.")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.window_length_seconds = window_length_seconds
        self.window_context_seconds = window_context_seconds

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
        output_dir.mkdir(parents=True, exist_ok=True)
        total_duration_ms = probe_audio_duration_ms(input_audio_path)

        with report_loading_duration(
            f"Demucs source separation '{self.model_name}'",
            reporter=self._report_system_message,
        ):
            if self._should_window_separation(total_duration_ms):
                return self._separate_windowed_vocals(
                    input_audio_path=input_audio_path,
                    output_dir=output_dir,
                    device=device,
                    total_duration_ms=total_duration_ms,
                    progress_callback=progress_callback,
                )

            vocals_path = self._separate_single_pass_vocals(
                input_audio_path=input_audio_path,
                output_dir=output_dir,
                device=device,
            )
            self._publish_progress_update(
                progress_callback=progress_callback,
                completed_units=1,
                total_units=1,
                processed_audio_ms=total_duration_ms,
                note="chunk 1/1",
            )
            return vocals_path

    def _should_window_separation(self, total_duration_ms: int | None) -> bool:
        """Return whether long-audio windowing should be used for this source."""
        if total_duration_ms is None:
            return False
        return total_duration_ms > self.window_length_seconds * 1000

    def _separate_single_pass_vocals(
        self,
        *,
        input_audio_path: Path,
        output_dir: Path,
        device: str,
    ) -> Path:
        """Run one Demucs pass and return the generated vocals stem path."""
        command = self._build_demucs_command(
            input_audio_path=input_audio_path,
            output_dir=output_dir,
            device=device,
        )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                subprocess.run(
                    command,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.timeout_seconds,
                )
                return self._resolve_vocals_path(
                    output_dir=output_dir,
                    track_stem=input_audio_path.stem,
                    last_error=last_error,
                )
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

        raise NonRetryableAudioStageError(
            "Demucs separation failed before producing a vocals stem."
        ) from last_error

    def _separate_windowed_vocals(
        self,
        *,
        input_audio_path: Path,
        output_dir: Path,
        device: str,
        total_duration_ms: int | None,
        progress_callback: StageProgressCallback | None,
    ) -> Path:
        """Separate long audio in outer windows and stitch the trimmed vocals."""
        if total_duration_ms is None or total_duration_ms <= 0:
            raise NonRetryableAudioStageError(
                "Windowed Demucs separation requires a known positive audio duration."
            )
        windows = self._plan_windows(total_duration_ms=total_duration_ms)
        if not windows:
            raise NonRetryableAudioStageError(
                "Demucs window planning returned no audio windows."
            )

        temp_root = output_dir / "_windowed" / input_audio_path.stem
        self._remove_dir(temp_root)
        inputs_dir = temp_root / "inputs"
        trimmed_dir = temp_root / "trimmed"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        trimmed_dir.mkdir(parents=True, exist_ok=True)
        trimmed_paths: list[Path] = []
        try:
            for index, window in enumerate(windows, start=1):
                chunk_input_path = inputs_dir / f"{input_audio_path.stem}_chunk_{index:04d}.wav"
                chunk_output_dir = temp_root / "demucs" / f"chunk_{index:04d}"
                trimmed_output_path = trimmed_dir / f"{index:04d}.wav"

                self._extract_audio_window(
                    source_audio_path=input_audio_path,
                    start_ms=window.context_start_ms,
                    end_ms=window.context_end_ms,
                    output_path=chunk_input_path,
                )
                chunk_vocals_path = self._separate_single_pass_vocals(
                    input_audio_path=chunk_input_path,
                    output_dir=chunk_output_dir,
                    device=device,
                )
                trim_start_ms = window.base_start_ms - window.context_start_ms
                self._trim_audio_window(
                    source_audio_path=chunk_vocals_path,
                    start_ms=trim_start_ms,
                    end_ms=trim_start_ms + window.base_duration_ms,
                    output_path=trimmed_output_path,
                )
                trimmed_paths.append(trimmed_output_path)
                self._remove_file(chunk_input_path)
                self._remove_dir(chunk_output_dir)
                self._publish_progress_update(
                    progress_callback=progress_callback,
                    completed_units=index,
                    total_units=len(windows),
                    processed_audio_ms=window.base_end_ms,
                    note=f"chunk {index}/{len(windows)}",
                )

            final_vocals_path = (
                output_dir / self.model_name / input_audio_path.stem / "vocals.wav"
            )
            self._concat_audio_files(
                input_paths=trimmed_paths,
                output_path=final_vocals_path,
            )
            return final_vocals_path
        finally:
            self._remove_dir(temp_root)

    def _plan_windows(self, *, total_duration_ms: int) -> list[_SeparationWindow]:
        """Plan base/context windows for one long source audio file."""
        if total_duration_ms <= 0:
            return []
        window_length_ms = self.window_length_seconds * 1000
        context_ms = self.window_context_seconds * 1000
        windows: list[_SeparationWindow] = []
        base_start_ms = 0
        while base_start_ms < total_duration_ms:
            base_end_ms = min(total_duration_ms, base_start_ms + window_length_ms)
            windows.append(
                _SeparationWindow(
                    base_start_ms=base_start_ms,
                    base_end_ms=base_end_ms,
                    context_start_ms=max(0, base_start_ms - context_ms),
                    context_end_ms=min(total_duration_ms, base_end_ms + context_ms),
                )
            )
            base_start_ms = base_end_ms
        return windows

    def _extract_audio_window(
        self,
        *,
        source_audio_path: Path,
        start_ms: int,
        end_ms: int,
        output_path: Path,
    ) -> None:
        """Extract one source-audio window to a temporary WAV file."""
        duration_ms = max(0, end_ms - start_ms)
        if duration_ms <= 0:
            raise NonRetryableAudioStageError(
                "Demucs window extraction requires a positive duration."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            _format_ffmpeg_seconds(start_ms),
            "-t",
            _format_ffmpeg_seconds(duration_ms),
            "-i",
            str(source_audio_path),
            "-vn",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(output_path),
        ]
        self._run_ffmpeg_command(
            command=command,
            failure_prefix="Failed to extract Demucs audio window.",
        )

    def _trim_audio_window(
        self,
        *,
        source_audio_path: Path,
        start_ms: int,
        end_ms: int,
        output_path: Path,
    ) -> None:
        """Trim one Demucs chunk back to the base window duration."""
        duration_ms = max(0, end_ms - start_ms)
        if duration_ms <= 0:
            raise NonRetryableAudioStageError(
                "Demucs window trim requires a positive duration."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_audio_path),
            "-ss",
            _format_ffmpeg_seconds(start_ms),
            "-t",
            _format_ffmpeg_seconds(duration_ms),
            "-vn",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(output_path),
        ]
        self._run_ffmpeg_command(
            command=command,
            failure_prefix="Failed to trim Demucs vocals window.",
        )

    def _concat_audio_files(
        self,
        *,
        input_paths: list[Path],
        output_path: Path,
    ) -> None:
        """Concatenate trimmed window outputs into one final vocals file."""
        if not input_paths:
            raise NonRetryableAudioStageError(
                "Demucs concat requires at least one trimmed input path."
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        concat_manifest_path = output_path.parent / "concat_inputs.txt"
        temp_output_path = output_path.with_name(f"{output_path.stem}{_TEMP_OUTPUT_SUFFIX}")
        concat_manifest_path.write_text(
            "\n".join(_format_concat_manifest_line(path) for path in input_paths) + "\n",
            encoding="utf-8",
        )
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_manifest_path),
            "-vn",
            "-c:a",
            "pcm_s16le",
            "-f",
            "wav",
            str(temp_output_path),
        ]
        try:
            self._run_ffmpeg_command(
                command=command,
                failure_prefix="Failed to concatenate Demucs vocals windows.",
            )
            temp_output_path.replace(output_path)
        finally:
            self._remove_file(concat_manifest_path)
            self._remove_file(temp_output_path)

    def _run_ffmpeg_command(
        self,
        *,
        command: list[str],
        failure_prefix: str,
    ) -> None:
        """Run one FFmpeg command and normalize timeout/process errors."""
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise NonRetryableAudioStageError(
                f"{failure_prefix} Command: {' '.join(command)}."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise NonRetryableAudioStageError(
                f"{failure_prefix} Command: {' '.join(command)}. "
                f"Stderr: {(exc.stderr or '').strip()[:500]}"
            ) from exc

    def _build_demucs_command(
        self,
        *,
        input_audio_path: Path,
        output_dir: Path,
        device: str,
    ) -> list[str]:
        """Build the Demucs CLI command for one input track."""
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
        return command

    def _resolve_vocals_path(
        self,
        *,
        output_dir: Path,
        track_stem: str,
        last_error: Exception | None,
    ) -> Path:
        """Resolve the expected Demucs vocals output path or raise clearly."""
        vocals_path = output_dir / self.model_name / track_stem / "vocals.wav"
        if vocals_path.exists():
            return vocals_path

        fallback = next(output_dir.rglob("vocals.wav"), None)
        if fallback is not None and fallback.exists():
            return fallback

        raise NonRetryableAudioStageError(
            f"Demucs completed but vocals stem was not found in '{output_dir}'."
        ) from last_error

    def _publish_progress_update(
        self,
        *,
        progress_callback: StageProgressCallback | None,
        completed_units: int,
        total_units: int,
        note: str,
        processed_audio_ms: int | None = None,
    ) -> None:
        """Publish one best-effort progress callback update."""
        if progress_callback is None:
            return
        try:
            progress_callback(
                StageProgressUpdate(
                    completed_units=completed_units,
                    total_units=total_units,
                    processed_audio_ms=processed_audio_ms,
                    note=note,
                )
            )
        except Exception:
            pass

    def _report_system_message(self, message: str) -> None:
        """Publish one operator-facing loading message through the event bus."""
        event_bus.publish("system_message", message)

    def _remove_dir(self, directory_path: Path) -> None:
        """Best-effort cleanup for temporary directories."""
        try:
            if directory_path.exists():
                shutil.rmtree(directory_path)
        except OSError:
            pass

    def _remove_file(self, file_path: Path) -> None:
        """Best-effort cleanup for temporary files."""
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass


def _format_ffmpeg_seconds(milliseconds: int) -> str:
    """Convert milliseconds to a decimal seconds string for FFmpeg."""
    return f"{max(0, milliseconds) / 1000:.3f}"


def _format_concat_manifest_line(path: Path) -> str:
    """Format one concat demuxer manifest line for FFmpeg."""
    escaped_path = path.resolve().as_posix().replace("'", "'\\''")
    return f"file '{escaped_path}'"
