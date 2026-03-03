from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.gateways.speaker_sample_gateway import FfmpegSpeakerSampleExporter
from audio_pipeline.speaker_samples import AudioSlice


def test_gateway_builds_concat_command_and_writes_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exporter = FfmpegSpeakerSampleExporter()
    calls: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(command)
        temp_output_path = Path(command[-1])
        temp_output_path.write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "audio_pipeline.gateways.speaker_sample_gateway.ensure_command_available",
        lambda _: None,
    )
    monkeypatch.setattr(subprocess, "run", _fake_run)

    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"source")
    output_path = tmp_path / "speaker.wav"
    exporter.export(
        source_audio_path=source_audio,
        slices=(
            AudioSlice(start_time_ms=1_000, end_time_ms=2_000),
            AudioSlice(start_time_ms=5_000, end_time_ms=6_000),
        ),
        output_path=output_path,
        sample_rate_hz=16_000,
        channels=1,
        edge_fade_ms=20,
        audio_codec="pcm_s24le",
    )

    assert output_path.exists()
    assert calls
    command = calls[0]
    assert command[:4] == ["ffmpeg", "-y", "-i", str(source_audio)]
    assert "-filter_complex" in command
    filter_graph = command[command.index("-filter_complex") + 1]
    assert "concat=n=2" in filter_graph
    assert "atrim=start=1.000:end=2.000" in filter_graph
    assert "atrim=start=5.000:end=6.000" in filter_graph
    assert "afade=t=in:st=0:d=0.020" in filter_graph
    assert command[command.index("-f") + 1] == "wav"
    assert command[command.index("-c:a") + 1] == "pcm_s24le"
    assert Path(command[-1]).name == "speaker.tmp.wav"


def test_gateway_surfaces_ffmpeg_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exporter = FfmpegSpeakerSampleExporter()

    def _raise(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="ffmpeg failed",
        )

    monkeypatch.setattr(
        "audio_pipeline.gateways.speaker_sample_gateway.ensure_command_available",
        lambda _: None,
    )
    monkeypatch.setattr(subprocess, "run", _raise)

    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"source")

    with pytest.raises(NonRetryableAudioStageError, match="Failed to export speaker sample"):
        exporter.export(
            source_audio_path=source_audio,
            slices=(AudioSlice(start_time_ms=0, end_time_ms=1_000),),
            output_path=tmp_path / "speaker.wav",
            sample_rate_hz=16_000,
            channels=1,
        )


def test_gateway_surfaces_ffmpeg_timeout_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exporter = FfmpegSpeakerSampleExporter(timeout_seconds=1)

    def _raise(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        command = args[0]
        del kwargs
        raise subprocess.TimeoutExpired(cmd=command, timeout=1)

    monkeypatch.setattr(
        "audio_pipeline.gateways.speaker_sample_gateway.ensure_command_available",
        lambda _: None,
    )
    monkeypatch.setattr(subprocess, "run", _raise)

    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"source")

    with pytest.raises(NonRetryableAudioStageError, match="due to timeout"):
        exporter.export(
            source_audio_path=source_audio,
            slices=(AudioSlice(start_time_ms=0, end_time_ms=1_000),),
            output_path=tmp_path / "speaker.wav",
            sample_rate_hz=16_000,
            channels=1,
        )


def test_gateway_removes_stale_temp_file_on_ffmpeg_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    exporter = FfmpegSpeakerSampleExporter()

    def _raise(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="ffmpeg failed",
        )

    monkeypatch.setattr(
        "audio_pipeline.gateways.speaker_sample_gateway.ensure_command_available",
        lambda _: None,
    )
    monkeypatch.setattr(subprocess, "run", _raise)

    source_audio = tmp_path / "source.wav"
    source_audio.write_bytes(b"source")
    output_path = tmp_path / "speaker.wav"
    stale_temp_path = tmp_path / "speaker.tmp.wav"
    stale_temp_path.write_bytes(b"stale")

    with pytest.raises(NonRetryableAudioStageError, match="Failed to export speaker sample"):
        exporter.export(
            source_audio_path=source_audio,
            slices=(AudioSlice(start_time_ms=0, end_time_ms=1_000),),
            output_path=output_path,
            sample_rate_hz=16_000,
            channels=1,
        )

    assert not stale_temp_path.exists()
