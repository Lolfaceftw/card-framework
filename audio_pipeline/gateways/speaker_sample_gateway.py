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

    def __init__(self, *, timeout_seconds: int = 300) -> None:
        self.timeout_seconds = timeout_seconds

    def export(
        self,
        *,
        source_audio_path: Path,
        slices: Sequence[AudioSlice],
        output_path: Path,
        sample_rate_hz: int,
        channels: int,
        edge_fade_ms: int = 0,
        audio_codec: str = "pcm_s24le",
    ) -> None:
        """
        Export one speaker sample as WAV using FFmpeg atrim/concat filters.

        Args:
            source_audio_path: Input audio file.
            slices: Ordered audio slices in milliseconds.
            output_path: Target output WAV path.
            sample_rate_hz: Output sample rate.
            channels: Output channel count.
            edge_fade_ms: Edge fade applied to each slice before concat.
            audio_codec: Explicit output PCM codec.
        """
        if not slices:
            raise NonRetryableAudioStageError(
                "Speaker sample export requires at least one audio slice."
            )
        if edge_fade_ms < 0:
            raise ValueError("edge_fade_ms must be >= 0")
        ensure_command_available("ffmpeg")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = _build_temp_output_path(output_path)
        filter_complex = _build_filter_complex(slices, edge_fade_ms=edge_fade_ms)
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
            "-c:a",
            str(audio_codec),
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
                timeout=self.timeout_seconds,
            )
            temp_path.replace(output_path)
        except subprocess.TimeoutExpired as exc:
            _remove_temp_output(temp_path)
            raise NonRetryableAudioStageError(
                "Failed to export speaker sample with ffmpeg due to timeout. "
                f"Command: {' '.join(command)}."
            ) from exc
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


def _build_filter_complex(slices: Sequence[AudioSlice], *, edge_fade_ms: int) -> str:
    """Build FFmpeg filter graph for one or more contiguous slices."""
    if len(slices) == 1:
        time_slice = slices[0]
        base_filter = (
            "[0:a]"
            f"atrim=start={_as_seconds(time_slice.start_time_ms)}:"
            f"end={_as_seconds(time_slice.end_time_ms)},"
            "asetpts=PTS-STARTPTS"
        )
        return f"{_append_edge_fades(base_filter, time_slice, edge_fade_ms=edge_fade_ms)}[outa]"

    filter_parts: list[str] = []
    for index, time_slice in enumerate(slices):
        base_filter = (
            "[0:a]"
            f"atrim=start={_as_seconds(time_slice.start_time_ms)}:"
            f"end={_as_seconds(time_slice.end_time_ms)},"
            "asetpts=PTS-STARTPTS"
        )
        slice_filter = _append_edge_fades(
            base_filter,
            time_slice,
            edge_fade_ms=edge_fade_ms,
        )
        filter_parts.append(
            f"{slice_filter}[a{index}]"
        )
    inputs = "".join(f"[a{index}]" for index in range(len(slices)))
    filter_parts.append(f"{inputs}concat=n={len(slices)}:v=0:a=1[outa]")
    return ";".join(filter_parts)


def _as_seconds(milliseconds: int) -> str:
    """Convert milliseconds to FFmpeg-compatible decimal seconds string."""
    return f"{milliseconds / 1000:.3f}"


def _append_edge_fades(
    base_filter: str,
    time_slice: AudioSlice,
    *,
    edge_fade_ms: int,
) -> str:
    """Append optional fade-in/out filters to one slice chain."""
    if edge_fade_ms <= 0:
        return base_filter
    slice_duration_ms = max(0, time_slice.duration_ms)
    if slice_duration_ms <= 0:
        return base_filter
    fade_ms = min(edge_fade_ms, max(0, slice_duration_ms // 2))
    if fade_ms <= 0:
        return base_filter
    fade_seconds = _as_seconds(fade_ms)
    fade_out_start_ms = max(0, slice_duration_ms - fade_ms)
    fade_out_start = _as_seconds(fade_out_start_ms)
    return (
        f"{base_filter},"
        f"afade=t=in:st=0:d={fade_seconds},"
        f"afade=t=out:st={fade_out_start}:d={fade_seconds}"
    )
