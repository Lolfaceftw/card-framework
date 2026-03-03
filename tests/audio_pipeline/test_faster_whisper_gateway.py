from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.gateways.faster_whisper_gateway import FasterWhisperTranscriber


class _FakeSegment:
    def __init__(self, *, start: float, end: float, text: str) -> None:
        self.start = start
        self.end = end
        self.text = text


class _FakeInfo:
    language = "en"


class _FakeModel:
    hf_tokenizer = None

    def transcribe(self, *args, **kwargs):
        del args, kwargs
        return (
            [
                _FakeSegment(start=0.0, end=0.5, text="hello"),
                _FakeSegment(start=0.5, end=1.0, text="world"),
            ],
            _FakeInfo(),
        )


class _FakeBatchInferencePipeline:
    def __init__(self, model: object) -> None:
        del model

    def transcribe(self, *args, **kwargs):
        del args, kwargs
        return ([_FakeSegment(start=0.0, end=1.0, text="hello world")], _FakeInfo())


def _install_fake_faster_whisper(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        BatchedInferencePipeline=_FakeBatchInferencePipeline,
        decode_audio=lambda _: [0.0] * 16000,
    )
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)


def test_transcriber_returns_fallback_word_timestamps_when_alignment_disabled(monkeypatch) -> None:
    _install_fake_faster_whisper(monkeypatch)
    monkeypatch.setattr(
        "audio_pipeline.gateways.faster_whisper_gateway.ensure_command_available",
        lambda _: None,
    )

    transcriber = FasterWhisperTranscriber(
        batch_size=0,
        enable_forced_alignment=False,
        require_forced_alignment=False,
    )
    monkeypatch.setattr(transcriber, "_get_model", lambda device: _FakeModel())

    result = transcriber.transcribe(audio_path=Path("dummy.wav"), device="cpu")

    assert len(result.segments) == 2
    assert len(result.word_timestamps) >= 2
    assert result.language == "en"


def test_transcriber_raises_when_forced_alignment_required_and_fails(monkeypatch) -> None:
    _install_fake_faster_whisper(monkeypatch)
    monkeypatch.setattr(
        "audio_pipeline.gateways.faster_whisper_gateway.ensure_command_available",
        lambda _: None,
    )

    class _ExplodingAligner:
        def align_words(self, **kwargs):
            del kwargs
            raise NonRetryableAudioStageError("alignment failed")

    transcriber = FasterWhisperTranscriber(
        batch_size=0,
        enable_forced_alignment=True,
        require_forced_alignment=True,
        forced_aligner=_ExplodingAligner(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(transcriber, "_get_model", lambda device: _FakeModel())

    with pytest.raises(NonRetryableAudioStageError, match="alignment failed"):
        transcriber.transcribe(audio_path=Path("dummy.wav"), device="cpu")
