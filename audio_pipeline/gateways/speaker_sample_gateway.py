"""FFmpeg-backed gateway for speaker-sample audio rendering."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess

from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.runtime import ensure_command_available
from audio_pipeline.speaker_samples import AudioSlice, SpeakerSampleExporter


class FfmpegSpeakerSampleExporter(SpeakerSampleExporter):
    """Render speaker samples by trimming and concatenating source-audio slices."""

    def export(
        self,
        *,
        source_audio_path: Path,
        slices: Sequence[AudioSlice],
        output_path: Path,
        sample_rate_hz: int,
        channels: int,
    ) -> None:
        """
        Export one speaker sample as WAV using FFmpeg atrim/concat filters.

        Args:
            source_audio_path: Input audio file.
            slices: Ordered audio slices in milliseconds.
            output_path: Target output WAV path.
            sample_rate_hz: Output sample rate.
            channels: Output channel count.
        """
        if not slices:
            raise NonRetryableAudioStageError(
                "Speaker sample export requires at least one audio slice."
            )
        ensure_command_available("ffmpeg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = _build_temp_output_path(output_path)
        filter_complex = _build_filter_complex(slices)
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source_audio_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outa]",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate_hz),
            "-vn",
            "-f",
            "wav",
            str(temp_path),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            temp_path.replace(output_path)
        except subprocess.CalledProcessError as exc:
            _remove_temp_output(temp_path)
            raise NonRetryableAudioStageError(
                "Failed to export speaker sample with ffmpeg. "
                f"Command: {' '.join(command)}. "
                f"Stderr: {(exc.stderr or '').strip()[:500]}"
            ) from exc
        except Exception as exc:
            _remove_temp_output(temp_path)
            raise NonRetryableAudioStageError(
                f"Failed to persist speaker sample artifact: {output_path}"
            ) from exc


def _build_temp_output_path(output_path: Path) -> Path:
    """
    Build a temporary output path that preserves the target audio extension.

    FFmpeg infers the output muxer from the final filename suffix. Using
    `speaker.wav.tmp` causes ffmpeg to treat the output as `.tmp`, which can
    fail format inference on Windows and Linux.
    """
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.tmp")


def _remove_temp_output(temp_path: Path) -> None:
    """Best-effort cleanup for temporary output files after failed exports."""
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass


def _build_filter_complex(slices: Sequence[AudioSlice]) -> str:
    """Build FFmpeg filter graph for one or more contiguous slices."""
    if len(slices) == 1:
        time_slice = slices[0]
        return (
            "[0:a]"
            f"atrim=start={_as_seconds(time_slice.start_time_ms)}:"
            f"end={_as_seconds(time_slice.end_time_ms)},"
            "asetpts=PTS-STARTPTS[outa]"
        )

    filter_parts: list[str] = []
    for index, time_slice in enumerate(slices):
        filter_parts.append(
            "[0:a]"
            f"atrim=start={_as_seconds(time_slice.start_time_ms)}:"
            f"end={_as_seconds(time_slice.end_time_ms)},"
            f"asetpts=PTS-STARTPTS[a{index}]"
        )
    inputs = "".join(f"[a{index}]" for index in range(len(slices)))
    filter_parts.append(f"{inputs}concat=n={len(slices)}:v=0:a=1[outa]")
    return ";".join(filter_parts)


def _as_seconds(milliseconds: int) -> str:
    """Convert milliseconds to FFmpeg-compatible decimal seconds string."""
    return f"{milliseconds / 1000:.3f}"
