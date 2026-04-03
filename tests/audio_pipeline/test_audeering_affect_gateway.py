"""Tests for audEERING affect gateway retry and chunking."""

from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import pytest

from card_framework.audio_pipeline.emotion.affect_analysis import AffectClip
from card_framework.audio_pipeline.errors import NonRetryableAudioStageError
from card_framework.audio_pipeline.gateways import (
    audeering_affect_gateway as gateway_module,
)
from card_framework.audio_pipeline.gateways.audeering_affect_gateway import (
    AudeeringMspDimAffectGateway,
    _AudeeringInferenceUnit,
    _AudeeringRuntime,
)


def _build_clip(
    index: int, *, sample_count: int = 16000, sample_rate_hz: int = 16000
) -> AffectClip:
    return AffectClip(
        segment_index=index,
        speaker="SPEAKER_00",
        text=f"segment {index}",
        start_time_ms=index * 1000,
        end_time_ms=index * 1000 + int(round((sample_count / sample_rate_hz) * 1000.0)),
        samples=np.zeros(sample_count, dtype=np.float32),
        sample_rate_hz=sample_rate_hz,
    )


def test_audeering_gateway_retries_cuda_oom_with_smaller_microbatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    cleanup_calls: list[str] = []

    class _FakeTensor:
        def __init__(self, batch_size: int) -> None:
            self.batch_size = batch_size

        def to(self, *args, **kwargs):
            del args, kwargs
            return self

    class _FakeArray:
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return self._values

    class _FakeProcessor:
        def __call__(self, waveforms, sampling_rate, return_tensors, padding):
            del sampling_rate, return_tensors, padding
            return {"input_values": _FakeTensor(len(waveforms))}

    class _FakeModel:
        def __call__(self, model_inputs):
            calls.append(model_inputs.batch_size)
            if model_inputs.batch_size > 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 19.66 GiB.")
            pooled = _FakeArray(np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
            logits = _FakeArray(np.array([[0.4, 0.5, 0.6]], dtype=np.float32))
            return pooled, logits

    gateway = AudeeringMspDimAffectGateway(device="cuda", batch_size=4)
    monkeypatch.setattr(
        gateway_module,
        "release_torch_device_memory_with_reporting",
        lambda **kwargs: cleanup_calls.append(str(kwargs.get("device", ""))),
    )
    monkeypatch.setattr(
        gateway_module.event_bus, "publish", lambda *args, **kwargs: None
    )

    runtime = _AudeeringRuntime(
        processor=_FakeProcessor(),
        model=_FakeModel(),
        device="cuda:0",
        torch_dtype=None,
    )
    unit_batch = [
        _AudeeringInferenceUnit(
            segment_index=index, sample_count=16000, clip=_build_clip(index)
        )
        for index in range(4)
    ]

    results = gateway._run_unit_batch(
        runtime=runtime,
        unit_batch=unit_batch,
        torch_module=type("_FakeTorch", (), {"no_grad": staticmethod(nullcontext)}),
    )

    assert len(results) == 4
    assert calls == [4, 2, 1, 1, 2, 1, 1]
    assert cleanup_calls == ["cuda", "cuda", "cuda"]


def test_audeering_gateway_retries_cuda_oom_when_tensor_transfer_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    cleanup_calls: list[str] = []

    class _FakeTensor:
        def __init__(self, batch_size: int) -> None:
            self.batch_size = batch_size

        def to(self, *args, **kwargs):
            del kwargs
            target = args[0] if args else None
            calls.append(target)
            if target == "cuda:0" and self.batch_size > 1:
                raise RuntimeError("CUDA out of memory. Tried to allocate 19.66 GiB.")
            return self

    class _FakeArray:
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self) -> np.ndarray:
            return self._values

    class _FakeProcessor:
        def __call__(self, waveforms, sampling_rate, return_tensors, padding):
            del sampling_rate, return_tensors, padding
            return {"input_values": _FakeTensor(len(waveforms))}

    class _FakeModel:
        def __call__(self, model_inputs):
            del model_inputs
            pooled = _FakeArray(np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
            logits = _FakeArray(np.array([[0.4, 0.5, 0.6]], dtype=np.float32))
            return pooled, logits

    gateway = AudeeringMspDimAffectGateway(device="cuda", batch_size=4)
    monkeypatch.setattr(
        gateway_module,
        "release_torch_device_memory_with_reporting",
        lambda **kwargs: cleanup_calls.append(str(kwargs.get("device", ""))),
    )
    monkeypatch.setattr(
        gateway_module.event_bus, "publish", lambda *args, **kwargs: None
    )

    runtime = _AudeeringRuntime(
        processor=_FakeProcessor(),
        model=_FakeModel(),
        device="cuda:0",
        torch_dtype=None,
    )
    unit_batch = [
        _AudeeringInferenceUnit(
            segment_index=index, sample_count=16000, clip=_build_clip(index)
        )
        for index in range(4)
    ]

    results = gateway._run_unit_batch(
        runtime=runtime,
        unit_batch=unit_batch,
        torch_module=type("_FakeTorch", (), {"no_grad": staticmethod(nullcontext)}),
    )

    assert len(results) == 4
    assert cleanup_calls == ["cuda", "cuda", "cuda"]
    assert calls == [
        "cuda:0",
        "cuda:0",
        "cuda:0",
        "cuda:0",
        "cuda:0",
        "cuda:0",
        "cuda:0",
    ]


def test_audeering_gateway_surfaces_cuda_oom_detail_for_single_clip() -> None:
    class _FakeTensor:
        def to(self, *args, **kwargs):
            del args, kwargs
            return self

    class _FakeProcessor:
        def __call__(self, waveforms, sampling_rate, return_tensors, padding):
            del waveforms, sampling_rate, return_tensors, padding
            return {"input_values": _FakeTensor()}

    class _FakeModel:
        def __call__(self, model_inputs):
            del model_inputs
            raise RuntimeError("CUDA out of memory. Tried to allocate 19.66 GiB.")

    gateway = AudeeringMspDimAffectGateway(device="cuda", batch_size=1)
    runtime = _AudeeringRuntime(
        processor=_FakeProcessor(),
        model=_FakeModel(),
        device="cuda:0",
        torch_dtype=None,
    )

    with pytest.raises(NonRetryableAudioStageError, match="CUDA out of memory"):
        gateway._run_unit_batch(
            runtime=runtime,
            unit_batch=[
                _AudeeringInferenceUnit(
                    segment_index=0,
                    sample_count=16000,
                    clip=_build_clip(0),
                )
            ],
            torch_module=type("_FakeTorch", (), {"no_grad": staticmethod(nullcontext)}),
        )


def test_audeering_gateway_splits_long_clip_and_aggregates_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    clip = _build_clip(0, sample_count=32000, sample_rate_hz=1600)
    gateway = AudeeringMspDimAffectGateway(
        device="cuda",
        batch_size=4,
        max_chunk_duration_seconds=10.0,
    )
    monkeypatch.setattr(
        gateway,
        "_get_runtime",
        lambda *, device: _AudeeringRuntime(
            processor=object(),
            model=object(),
            device="cuda:0",
            torch_dtype=None,
        ),
    )
    monkeypatch.setattr(
        gateway,
        "_run_unit_batch",
        lambda *, runtime, unit_batch, torch_module: (
            calls.append(len(unit_batch))
            or [
                {
                    "arousal": (0.2 if index == 0 else 0.8),
                    "dominance": 0.5,
                    "valence": (0.7 if index == 0 else 0.3),
                    "embedding_dimension": 3,
                }
                for index, _unit in enumerate(unit_batch)
            ]
        ),
    )
    monkeypatch.setattr(
        gateway_module.event_bus, "publish", lambda *args, **kwargs: None
    )

    updates = gateway.analyze([clip])

    assert calls == [2]
    assert len(updates) == 1
    assert updates[0].arousal == pytest.approx(0.5)
    assert updates[0].valence == pytest.approx(0.5)
    assert updates[0].embedding_dimension == 3
