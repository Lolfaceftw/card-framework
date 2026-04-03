"""Tests for emotion2vec+ gateway retry and error handling."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from card_framework.audio_pipeline.emotion.affect_analysis import AffectClip
from card_framework.audio_pipeline.errors import NonRetryableAudioStageError
from card_framework.audio_pipeline.gateways import (
    emotion2vec_affect_gateway as gateway_module,
)
from card_framework.audio_pipeline.gateways.emotion2vec_affect_gateway import (
    Emotion2VecPlusAffectGateway,
    _Emotion2VecRuntime,
)


def _build_clip(index: int) -> AffectClip:
    return AffectClip(
        segment_index=index,
        speaker="SPEAKER_00",
        text=f"segment {index}",
        start_time_ms=index * 1000,
        end_time_ms=index * 1000 + 1200,
        samples=np.zeros(16000, dtype=np.float32),
        sample_rate_hz=16000,
    )


def test_emotion2vec_gateway_retries_cuda_oom_with_smaller_microbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    cleanup_calls: list[str] = []
    notices: list[str] = []

    class _FakeModel:
        def generate(self, *, input, output_dir, granularity, extract_embedding):
            del output_dir, granularity, extract_embedding
            calls.append(len(input))
            if len(input) > 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 9.80 GiB.")
            return [{"labels": ["happy"], "scores": [0.9], "feats": [0.1, 0.2]}]

    gateway = Emotion2VecPlusAffectGateway(device="cuda", batch_size=4)
    monkeypatch.setattr(
        gateway,
        "_get_runtime",
        lambda *, device: _Emotion2VecRuntime(
            model=_FakeModel(), resolved_device="cuda:0"
        ),
    )
    monkeypatch.setattr(
        gateway,
        "_write_batch_inputs",
        lambda *, batch, temp_dir: [
            str(Path(temp_dir) / f"clip-{clip.segment_index}.wav") for clip in batch
        ],
    )
    monkeypatch.setattr(
        gateway_module,
        "release_torch_device_memory_with_reporting",
        lambda **kwargs: cleanup_calls.append(str(kwargs.get("device", ""))),
    )
    monkeypatch.setattr(
        gateway_module.event_bus,
        "publish",
        lambda channel, message, *args, **kwargs: notices.append(
            f"{channel}:{message}"
        ),
    )

    updates = gateway.analyze([_build_clip(index) for index in range(4)])

    assert len(updates) == 4
    assert calls == [4, 2, 1, 1, 2, 1, 1]
    assert cleanup_calls == ["cuda", "cuda", "cuda"]
    assert any(
        "retrying on CUDA with smaller micro-batches" in notice for notice in notices
    )


def test_emotion2vec_gateway_surfaces_cuda_oom_detail_for_single_clip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeModel:
        def generate(self, *, input, output_dir, granularity, extract_embedding):
            del input, output_dir, granularity, extract_embedding
            raise RuntimeError("CUDA out of memory. Tried to allocate 9.80 GiB.")

    gateway = Emotion2VecPlusAffectGateway(device="cuda", batch_size=1)
    monkeypatch.setattr(
        gateway,
        "_get_runtime",
        lambda *, device: _Emotion2VecRuntime(
            model=_FakeModel(), resolved_device="cuda:0"
        ),
    )
    monkeypatch.setattr(
        gateway,
        "_write_batch_inputs",
        lambda *, batch, temp_dir: [
            str(Path(temp_dir) / f"clip-{clip.segment_index}.wav") for clip in batch
        ],
    )

    with pytest.raises(NonRetryableAudioStageError, match="CUDA out of memory"):
        gateway.analyze([_build_clip(0)])


def test_emotion2vec_gateway_splits_long_clip_and_aggregates_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class _FakeModel:
        def generate(self, *, input, output_dir, granularity, extract_embedding):
            del output_dir, granularity, extract_embedding
            inputs = list(input)
            calls.append(inputs)
            results = []
            for item in inputs:
                if item.endswith("part0.wav"):
                    results.append(
                        {"labels": ["happy", "neutral"], "scores": [0.8, 0.2]}
                    )
                else:
                    results.append(
                        {"labels": ["happy", "neutral"], "scores": [0.4, 0.6]}
                    )
            return results

    clip = AffectClip(
        segment_index=0,
        speaker="SPEAKER_00",
        text="long segment",
        start_time_ms=0,
        end_time_ms=60000,
        samples=np.zeros(32000, dtype=np.float32),
        sample_rate_hz=1600,
    )
    gateway = Emotion2VecPlusAffectGateway(
        device="cuda",
        batch_size=4,
        max_chunk_duration_seconds=10.0,
    )
    monkeypatch.setattr(
        gateway,
        "_get_runtime",
        lambda *, device: _Emotion2VecRuntime(
            model=_FakeModel(), resolved_device="cuda:0"
        ),
    )
    monkeypatch.setattr(
        gateway,
        "_write_batch_inputs",
        lambda *, batch, temp_dir: [
            str(Path(temp_dir) / f"part{index}.wav")
            for index, _clip in enumerate(batch)
        ],
    )
    monkeypatch.setattr(
        gateway_module.event_bus, "publish", lambda *args, **kwargs: None
    )

    updates = gateway.analyze([clip])

    assert len(updates) == 1
    assert len(calls) == 1
    assert len(calls[0]) == 2
    assert updates[0].emotion_scores[0].label == "happy"
    assert updates[0].emotion_scores[0].score == pytest.approx(0.6)
