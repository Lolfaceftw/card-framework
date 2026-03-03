from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.gateways.nemo_diarizer_gateway import NemoSpeakerDiarizer


def test_prepare_diarization_audio_normalizes_to_mono(monkeypatch, tmp_path: Path) -> None:
    diarizer = NemoSpeakerDiarizer()
    calls: list[list[str]] = []

    def _fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        del kwargs
        calls.append(command)
        output_path = Path(command[-1])
        output_path.write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "audio_pipeline.gateways.nemo_diarizer_gateway.ensure_command_available",
        lambda _: None,
    )
    monkeypatch.setattr(subprocess, "run", _fake_run)

    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"input")

    normalized = diarizer._prepare_diarization_audio(
        audio_path=input_audio,
        output_dir=tmp_path,
    )

    assert normalized == tmp_path / "diarization_input_mono.wav"
    assert normalized.exists()
    assert calls
    command = calls[0]
    assert command[:4] == ["ffmpeg", "-y", "-i", str(input_audio)]
    assert "-ac" in command and "1" in command
    assert "-ar" in command and "16000" in command


def test_prepare_diarization_audio_surfaces_ffmpeg_errors(
    monkeypatch,
    tmp_path: Path,
) -> None:
    diarizer = NemoSpeakerDiarizer()

    def _raise(*args, **kwargs) -> subprocess.CompletedProcess[str]:
        del args, kwargs
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=["ffmpeg"],
            stderr="ffmpeg failed",
        )

    monkeypatch.setattr(
        "audio_pipeline.gateways.nemo_diarizer_gateway.ensure_command_available",
        lambda _: None,
    )
    monkeypatch.setattr(subprocess, "run", _raise)

    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"input")

    with pytest.raises(NonRetryableAudioStageError, match="Failed to prepare mono"):
        diarizer._prepare_diarization_audio(
            audio_path=input_audio,
            output_dir=tmp_path,
        )


def test_diarize_raises_when_single_speaker_fallback_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    diarizer = NemoSpeakerDiarizer(allow_single_speaker_fallback=False)

    def _raise(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("nemo failed")

    monkeypatch.setattr(diarizer, "_run_nemo_msdd", _raise)
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"input")

    with pytest.raises(NonRetryableAudioStageError, match="NeMo diarization failed"):
        diarizer.diarize(audio_path=input_audio, output_dir=tmp_path / "out", device="cpu")


def test_diarize_can_fallback_to_single_speaker_when_enabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    diarizer = NemoSpeakerDiarizer(allow_single_speaker_fallback=True)

    def _raise(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("nemo failed")

    monkeypatch.setattr(diarizer, "_run_nemo_msdd", _raise)
    monkeypatch.setattr(diarizer, "_probe_duration_ms", lambda _: 3210)
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"input")

    turns = diarizer.diarize(audio_path=input_audio, output_dir=tmp_path / "out", device="cpu")

    assert len(turns) == 1
    assert turns[0].speaker == "SPEAKER_00"
    assert turns[0].start_time_ms == 0
    assert turns[0].end_time_ms == 3210
