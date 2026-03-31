"""Tests for the Granite Speech ASR gateway."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from card_framework.audio_pipeline.gateways import granite_speech_gateway
from card_framework.audio_pipeline.contracts import TimedTextSegment, WordTimestamp
from card_framework.audio_pipeline.gateways.granite_speech_gateway import (
    GraniteSpeechTranscriber,
    _GraniteRuntime,
    _dedupe_chunk_text,
    _strip_prompt_echo,
)
from card_framework.shared.events import event_bus


def test_granite_transcriber_plans_30_second_chunks_with_5_second_overlap() -> None:
    """Plan 30-second chunks that advance by 25 seconds."""
    transcriber = GraniteSpeechTranscriber(
        chunk_length_seconds=30,
        chunk_overlap_seconds=5,
    )

    ranges = transcriber._plan_chunk_ranges(total_samples=70 * 16_000)

    assert ranges == [
        (0, 30 * 16_000),
        (25 * 16_000, 55 * 16_000),
        (50 * 16_000, 70 * 16_000),
    ]


def test_dedupe_chunk_text_removes_repeated_boundary_tokens() -> None:
    """Drop repeated overlap tokens before appending the next chunk text."""
    deduped = _dedupe_chunk_text(
        previous_text="Hello there, general",
        current_text="there general Kenobi arrives",
    )

    assert deduped == "Kenobi arrives"


def test_strip_prompt_echo_removes_role_tagged_prompt_echo_inside_chunk() -> None:
    """Strip Granite's chat-style USER/ASSISTANT prompt echo anywhere in the chunk."""
    cleaned = _strip_prompt_echo(
        text=(
            "can you imagine this is our four. USER: can you transcribe the speech "
            "into a written format? ASSISTANT:of pwl capital."
        ),
        prompt="<|audio|>can you transcribe the speech into a written format?",
        rendered_prompt="USER: can you transcribe the speech into a written format? ASSISTANT:",
    )

    assert cleaned == "can you imagine this is our four. of pwl capital."


def test_granite_transcribe_chunks_audio_and_reports_progress(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Chunk audio, dedupe boundary text, and emit segment progress notes."""
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    transcriber = GraniteSpeechTranscriber(enable_forced_alignment=False)

    monkeypatch.setattr(
        transcriber,
        "_get_runtime",
        lambda *, device: SimpleNamespace(tokenizer=object(), processor=object(), model=object(), device=device),
    )
    monkeypatch.setattr(
        transcriber,
        "_probe_audio_metadata",
        lambda audio_path: (16_000, 70 * 16_000),
    )
    monkeypatch.setattr(
        transcriber,
        "_load_audio_waveform",
        lambda audio_path: (_ for _ in ()).throw(
            AssertionError("full audio load should not run when forced alignment is disabled")
        ),
    )
    chunk_texts = iter(
        [
            "hello there",
            "there general kenobi",
            "kenobi final words",
        ]
    )
    monkeypatch.setattr(
        transcriber,
        "_load_prepared_audio_chunk",
        lambda **kwargs: np.zeros(
            kwargs["end_sample"] - kwargs["start_sample"],
            dtype=np.float32,
        ),
    )
    monkeypatch.setattr(
        transcriber,
        "_transcribe_chunk",
        lambda *, runtime, audio_chunk: next(chunk_texts),
    )

    updates = []
    result = transcriber.transcribe(
        audio_path,
        device="cpu",
        progress_callback=lambda update: updates.append(update),
    )

    assert [update.note for update in updates] == [
        "segment 1/3",
        "segment 2/3",
        "segment 3/3",
    ]
    assert [update.processed_audio_ms for update in updates] == [25_000, 49_000, 70_000]
    assert [segment.text for segment in result.segments] == [
        "hello there",
        "general kenobi",
        "final words",
    ]
    assert result.language == "en"
    assert len(result.word_timestamps) == 6


def test_granite_transcribe_chunk_uses_rendered_prompt_cleanup(monkeypatch) -> None:
    """Use the rendered Granite chat prompt when stripping echoed role labels."""
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(inference_mode=nullcontext),
    )

    class _FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            del messages, tokenize, add_generation_prompt
            return "USER: can you transcribe the speech into a written format? ASSISTANT:"

    class _FakeModelInputs(dict):
        def to(self, device):
            del device
            return self

    class _FakeProcessor:
        def __call__(self, prompt, audio_chunk, device, return_tensors):
            del audio_chunk, device, return_tensors
            assert (
                prompt
                == "USER: can you transcribe the speech into a written format? ASSISTANT:"
            )
            return _FakeModelInputs()

        def batch_decode(self, generated, skip_special_tokens):
            del generated, skip_special_tokens
            return [
                (
                    "can you imagine this is our four. USER: can you transcribe the "
                    "speech into a written format? ASSISTANT:of pwl capital."
                )
            ]

    generate_calls: list[dict[str, object]] = []

    class _FakeModel:
        def generate(self, **kwargs):
            generate_calls.append(kwargs)
            return [[1, 2, 3]]

    transcriber = GraniteSpeechTranscriber(enable_forced_alignment=False)
    runtime = SimpleNamespace(
        tokenizer=_FakeTokenizer(),
        processor=_FakeProcessor(),
        model=_FakeModel(),
        device=SimpleNamespace(type="cpu"),
    )

    chunk_text = transcriber._transcribe_chunk(
        runtime=runtime,
        audio_chunk=np.zeros(16_000, dtype=np.float32),
    )

    assert chunk_text == "can you imagine this is our four. of pwl capital."
    assert generate_calls[0]["max_new_tokens"] == 96
    assert generate_calls[0]["repetition_penalty"] == pytest.approx(1.25)


def test_granite_runtime_reports_loading_messages(monkeypatch) -> None:
    """Emit the shared loading-context messages around model initialization."""
    fake_torch = SimpleNamespace(
        float16="float16",
        float32="float32",
        bfloat16="bfloat16",
        cuda=SimpleNamespace(is_bf16_supported=lambda: False),
        device=lambda value: SimpleNamespace(type=value),
    )

    class _FakeProcessor:
        tokenizer = SimpleNamespace()

    class _FakeModel:
        def to(self, device):
            self.device = device
            return self

        def eval(self) -> None:
            return None

    fake_transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(
            from_pretrained=lambda model_name: _FakeProcessor()
        ),
        AutoModelForSpeechSeq2Seq=SimpleNamespace(
            from_pretrained=lambda model_name, torch_dtype, low_cpu_mem_usage: _FakeModel()
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    messages: list[str] = []

    def _capture(message: str) -> None:
        messages.append(message)

    event_bus.subscribe("system_message", _capture)
    try:
        transcriber = GraniteSpeechTranscriber()
        transcriber._get_runtime(device="cpu")
    finally:
        event_bus.unsubscribe("system_message", _capture)

    assert messages[0] == (
        "Loading Granite Speech ASR model 'ibm-granite/granite-4.0-1b-speech'..."
    )
    assert messages[1].startswith(
        "Loaded Granite Speech ASR model 'ibm-granite/granite-4.0-1b-speech' in "
    )


def test_granite_probe_audio_metadata_falls_back_to_wave(monkeypatch) -> None:
    """Use the WAV header when torchaudio lacks an `info()` helper."""
    monkeypatch.setitem(sys.modules, "torchaudio", SimpleNamespace())

    class _FakeWaveFile:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def getframerate(self) -> int:
            return 44_100

        def getnframes(self) -> int:
            return 88_200

    monkeypatch.setattr(
        granite_speech_gateway.wave,
        "open",
        lambda path, mode: _FakeWaveFile(),
    )

    transcriber = GraniteSpeechTranscriber()

    sample_rate, frame_count = transcriber._probe_audio_metadata(Path("ignored.wav"))

    assert sample_rate == 44_100
    assert frame_count == 88_200


def test_granite_release_cached_resources_clears_device_cache(monkeypatch) -> None:
    """Drop the cached runtime bundle and release device memory for one device."""
    transcriber = GraniteSpeechTranscriber()
    transcriber._runtime_cache["cuda"] = _GraniteRuntime(
        model=object(),
        processor=object(),
        tokenizer=object(),
        device="cuda",
    )
    released_devices: list[str] = []

    monkeypatch.setattr(
        granite_speech_gateway,
        "release_torch_device_memory_with_reporting",
        lambda *, device, description: released_devices.append(
            f"{device}:{description}"
        ),
    )

    transcriber.release_cached_resources(device="cuda")

    assert transcriber._runtime_cache == {}
    assert released_devices == [
        "cuda:Granite Speech ASR model 'ibm-granite/granite-4.0-1b-speech'"
    ]


def test_granite_skips_long_audio_forced_alignment_with_fallback() -> None:
    """Fall back to approximate words before CTC alignment on very long audio."""
    class _FakeAligner:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def align_words(self, **kwargs) -> list[WordTimestamp]:
            self.calls.append(kwargs)
            return []

    fake_aligner = _FakeAligner()
    transcriber = GraniteSpeechTranscriber(
        forced_alignment_max_audio_duration_seconds=60,
        forced_aligner=fake_aligner,
    )
    fallback_segments = [
        TimedTextSegment(
            start_time_ms=0,
            end_time_ms=2000,
            text="hello world",
        )
    ]
    messages: list[str] = []

    def _capture(message: str) -> None:
        messages.append(message)

    event_bus.subscribe("system_message", _capture)
    try:
        words = transcriber._resolve_word_timestamps(
            audio_path=Path("ignored.wav"),
            prepared_total_samples=61 * 16_000,
            transcript_text="hello world",
            language="en",
            device="cuda",
            fallback_segments=fallback_segments,
        )
    finally:
        event_bus.unsubscribe("system_message", _capture)

    assert fake_aligner.calls == []
    assert [word.word for word in words] == ["hello", "world"]
    assert any("Skipping CTC forced alignment" in message for message in messages)


def test_granite_releases_cuda_runtime_before_forced_alignment() -> None:
    """Unload the Granite ASR runtime before launching CUDA forced alignment."""
    class _FakeAligner:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def align_words(self, **kwargs) -> list[WordTimestamp]:
            self.calls.append(kwargs)
            return [
                WordTimestamp(
                    word="hello",
                    start_time_ms=0,
                    end_time_ms=500,
                )
            ]

    fake_aligner = _FakeAligner()
    transcriber = GraniteSpeechTranscriber(
        forced_alignment_max_audio_duration_seconds=None,
        forced_aligner=fake_aligner,
    )
    released_devices: list[str] = []
    transcriber.release_cached_resources = lambda *, device: released_devices.append(device)
    transcriber._load_audio_waveform = lambda audio_path: ("ignored", 16_000)
    transcriber._prepare_audio_waveform = (
        lambda *, waveform: np.zeros(16_000, dtype=np.float32)
    )

    words = transcriber._resolve_word_timestamps(
        audio_path=Path("ignored.wav"),
        prepared_total_samples=16_000,
        transcript_text="hello",
        language="en",
        device="cuda",
        fallback_segments=[
            TimedTextSegment(
                start_time_ms=0,
                end_time_ms=1000,
                text="hello",
            )
        ],
    )

    assert released_devices == ["cuda"]
    assert len(fake_aligner.calls) == 1
    assert [word.word for word in words] == ["hello"]
