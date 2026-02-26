from __future__ import annotations

from pathlib import Path

from audio_pipeline.gateways.fallback_gateways import (
    PassthroughSourceSeparator,
    SingleSpeakerDiarizer,
)


def test_passthrough_separator_returns_input_audio_path(tmp_path: Path) -> None:
    separator = PassthroughSourceSeparator()
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"audio")

    result = separator.separate_vocals(
        input_audio_path=input_audio,
        output_dir=tmp_path / "unused",
        device="cpu",
    )

    assert result == input_audio


def test_single_speaker_diarizer_returns_single_timeline() -> None:
    diarizer = SingleSpeakerDiarizer(speaker_label="SPEAKER_09", duration_ms=1234)

    turns = diarizer.diarize(
        audio_path=Path("ignored.wav"),
        output_dir=Path("ignored"),
        device="cpu",
    )

    assert len(turns) == 1
    assert turns[0].speaker == "SPEAKER_09"
    assert turns[0].start_time_ms == 0
    assert turns[0].end_time_ms == 1234
