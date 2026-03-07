from __future__ import annotations

from pathlib import Path

from audio_pipeline.gateways.pyannote_diarizer_gateway import PyannoteSpeakerDiarizer


class _FakeSegment:
    def __init__(self, *, start: float, end: float) -> None:
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, rows: list[tuple[float, float, str]]) -> None:
        self._rows = rows

    def itertracks(self, *, yield_label: bool) -> list[tuple[_FakeSegment, None, str]]:
        assert yield_label is True
        return [
            (_FakeSegment(start=start, end=end), None, speaker)
            for start, end, speaker in self._rows
        ]


class _FakeOutput:
    def __init__(self) -> None:
        self.speaker_diarization = _FakeAnnotation(
            [(0.0, 1.0, "speaker_a"), (1.0, 2.0, "speaker_b")]
        )
        self.exclusive_speaker_diarization = _FakeAnnotation(
            [(0.0, 0.8, "speaker_b"), (0.8, 2.0, "speaker_a")]
        )


class _FakePipeline:
    def __call__(self, *args, **kwargs) -> _FakeOutput:
        del args, kwargs
        return _FakeOutput()


def test_run_pipeline_prefers_exclusive_output_when_enabled(tmp_path: Path) -> None:
    diarizer = PyannoteSpeakerDiarizer(use_exclusive_diarization=True)
    diarizer._pipeline_cache["cpu"] = _FakePipeline()
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"wav")

    annotation = diarizer._run_pipeline(audio_path=input_audio, device="cpu")

    rows = annotation.itertracks(yield_label=True)
    assert rows[0][2] == "speaker_b"


def test_diarize_returns_normalized_turns(monkeypatch, tmp_path: Path) -> None:
    diarizer = PyannoteSpeakerDiarizer(use_exclusive_diarization=False)
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"wav")

    monkeypatch.setattr(
        "audio_pipeline.gateways.pyannote_diarizer_gateway.prepare_diarization_audio",
        lambda *, audio_path, output_dir: audio_path,
    )
    monkeypatch.setattr(
        diarizer,
        "_run_pipeline",
        lambda *, audio_path, device: _FakeAnnotation(
            [(0.0, 1.0, "speaker_b"), (1.0, 2.0, "speaker_a")]
        ),
    )

    turns = diarizer.diarize(
        audio_path=input_audio,
        output_dir=tmp_path / "out",
        device="cpu",
    )

    assert [turn.speaker for turn in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[0].start_time_ms == 0
    assert turns[1].end_time_ms == 2000
