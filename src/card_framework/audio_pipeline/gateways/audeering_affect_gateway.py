"""audEERING MSP-dim adapter for segment-level affect regression."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from card_framework.audio_pipeline.emotion.affect_analysis import (
    AffectClip,
    SegmentAffectProvider,
    SegmentAffectUpdate,
)
from card_framework.audio_pipeline.errors import (
    DependencyMissingError,
    NonRetryableAudioStageError,
)
from card_framework.audio_pipeline.runtime import (
    release_torch_device_memory_with_reporting,
)
from card_framework.shared.events import event_bus
from card_framework.shared.perf import report_loading_duration


_AUDEERING_LABELS = ("arousal", "dominance", "valence")


class _RegressionHead:
    """Regression head from the official audEERING model-card example."""

    def __init__(self, config: Any) -> None:
        try:
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover - import guard
            raise DependencyMissingError(
                "audEERING MSP-dim requires the 'torch' runtime."
            ) from exc
        self._dense = nn.Linear(config.hidden_size, config.hidden_size)
        self._dropout = nn.Dropout(config.final_dropout)
        self._out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def as_module(self) -> Any:
        """Return the wrapped torch module."""
        try:
            import torch.nn as nn
        except Exception as exc:  # pragma: no cover - import guard
            raise DependencyMissingError(
                "audEERING MSP-dim requires the 'torch' runtime."
            ) from exc

        class _WrappedRegressionHead(nn.Module):
            def __init__(self, dense: Any, dropout: Any, out_proj: Any) -> None:
                super().__init__()
                self.dense = dense
                self.dropout = dropout
                self.out_proj = out_proj

            def forward(self, features, **kwargs):
                del kwargs
                hidden = self.dropout(features)
                hidden = self.dense(hidden)
                hidden = hidden.tanh()
                hidden = self.dropout(hidden)
                return self.out_proj(hidden)

        return _WrappedRegressionHead(self._dense, self._dropout, self._out_proj)


def _build_audeering_emotion_model_class() -> type[Any]:
    """Build the official audEERING `EmotionModel` class lazily."""
    try:
        from transformers.models.wav2vec2.modeling_wav2vec2 import (
            Wav2Vec2Model,
            Wav2Vec2PreTrainedModel,
        )
    except Exception as exc:
        raise DependencyMissingError(
            "audEERING MSP-dim requires the 'transformers' runtime."
        ) from exc

    try:
        import torch
    except Exception as exc:
        raise DependencyMissingError(
            "audEERING MSP-dim requires the 'torch' runtime."
        ) from exc

    class EmotionModel(Wav2Vec2PreTrainedModel):
        """Speech emotion regressor from the official model-card example."""

        def __init__(self, config):
            super().__init__(config)
            self.config = config
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = _RegressionHead(config).as_module()
            self.init_weights()

        def forward(self, input_values):
            outputs = self.wav2vec2(input_values)
            hidden_states = outputs[0]
            hidden_states = torch.mean(hidden_states, dim=1)
            logits = self.classifier(hidden_states)
            return hidden_states, logits

    return EmotionModel


@dataclass(slots=True, frozen=True)
class _AudeeringRuntime:
    """Hold one loaded audEERING runtime for a concrete device."""

    processor: Any
    model: Any
    device: Any
    torch_dtype: Any


@dataclass(slots=True, frozen=True)
class _AudeeringInferenceUnit:
    """One bounded-duration inference unit derived from a transcript clip."""

    segment_index: int
    sample_count: int
    clip: AffectClip


class AudeeringMspDimAffectGateway(SegmentAffectProvider):
    """Infer arousal, dominance, and valence for transcript segments."""

    def __init__(
        self,
        *,
        model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device: str = "cpu",
        use_fp16: bool = False,
        batch_size: int = 4,
        max_chunk_duration_seconds: float = 30.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        self.model_name = model_name
        self._device = str(device).strip().lower() or "cpu"
        self.use_fp16 = bool(use_fp16)
        self.batch_size = int(batch_size)
        self.max_chunk_duration_seconds = float(max_chunk_duration_seconds)
        if self.max_chunk_duration_seconds <= 0:
            raise ValueError("max_chunk_duration_seconds must be > 0.")
        self._runtime_cache: dict[str, _AudeeringRuntime] = {}

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier."""
        return "audeering_msp_dim"

    @property
    def device(self) -> str:
        """Return the configured device."""
        return self._device

    def analyze(self, clips: Sequence[AffectClip]) -> list[SegmentAffectUpdate]:
        """Run dimensional SER on one or more prepared clips."""
        if not clips:
            return []
        runtime = self._get_runtime(device=self._device)
        try:
            import torch
        except Exception as exc:
            raise DependencyMissingError(
                "audEERING MSP-dim requires the 'torch' runtime."
            ) from exc

        updates: list[SegmentAffectUpdate] = []
        total_batches = (len(clips) + self.batch_size - 1) // self.batch_size
        for batch_index, batch_start in enumerate(
            range(0, len(clips), self.batch_size), start=1
        ):
            batch = list(clips[batch_start : batch_start + self.batch_size])
            inference_units = self._build_inference_units(batch=batch)
            results_by_segment: dict[int, list[tuple[int, dict[str, Any]]]] = {
                clip.segment_index: [] for clip in batch
            }
            for unit_start in range(0, len(inference_units), self.batch_size):
                unit_batch = inference_units[unit_start : unit_start + self.batch_size]
                unit_results = self._run_unit_batch(
                    runtime=runtime,
                    unit_batch=unit_batch,
                    torch_module=torch,
                )
                for unit, result in zip(unit_batch, unit_results, strict=True):
                    results_by_segment[unit.segment_index].append(
                        (unit.sample_count, result)
                    )
            for clip in batch:
                aggregated = self._aggregate_results_for_clip(
                    chunk_results=results_by_segment.get(clip.segment_index, []),
                )
                updates.append(
                    SegmentAffectUpdate(
                        segment_index=clip.segment_index,
                        arousal=aggregated.get("arousal"),
                        dominance=aggregated.get("dominance"),
                        valence=aggregated.get("valence"),
                        embedding_dimension=aggregated.get("embedding_dimension"),
                        provider_details={
                            self.provider_name: {
                                "provider": self.provider_name,
                                "model_name": self.model_name,
                            }
                        },
                    )
                )
            event_bus.publish(
                "system_message",
                (
                    "Affect analysis: audEERING batch "
                    f"{batch_index}/{total_batches} completed "
                    f"for {len(batch)} clip(s)."
                ),
            )
        return updates

    def _run_unit_batch(
        self,
        *,
        runtime: _AudeeringRuntime,
        unit_batch: Sequence[_AudeeringInferenceUnit],
        torch_module: Any,
    ) -> list[dict[str, Any]]:
        """Run one bounded audEERING batch, retrying smaller on CUDA OOM."""
        waveforms = [unit.clip.samples for unit in unit_batch]
        try:
            inputs = runtime.processor(
                waveforms,
                sampling_rate=unit_batch[0].clip.sample_rate_hz,
                return_tensors="pt",
                padding=True,
            )
        except Exception as exc:
            raise NonRetryableAudioStageError(
                "audEERING MSP-dim failed while tokenizing audio clips."
            ) from exc
        input_values = inputs.get("input_values")
        if input_values is None or not hasattr(input_values, "to"):
            raise NonRetryableAudioStageError(
                "audEERING MSP-dim processor did not return input_values."
            )
        try:
            model_inputs = input_values.to(runtime.device)
            if runtime.torch_dtype is not None and hasattr(model_inputs, "to"):
                model_inputs = model_inputs.to(dtype=runtime.torch_dtype)
            with torch_module.no_grad():
                pooled_states, logits = runtime.model(model_inputs)
        except Exception as exc:
            if self._should_retry_with_smaller_cuda_batch(
                runtime=runtime,
                unit_batch=unit_batch,
                error=exc,
            ):
                event_bus.publish(
                    "system_message",
                    (
                        "Affect analysis: audEERING batch of "
                        f"{len(unit_batch)} clip(s) hit CUDA OOM; retrying on CUDA "
                        "with smaller micro-batches."
                    ),
                )
                release_torch_device_memory_with_reporting(
                    device="cuda",
                    description="audEERING MSP-dim CUDA OOM retry preparation",
                )
                midpoint = max(1, len(unit_batch) // 2)
                left_results = self._run_unit_batch(
                    runtime=runtime,
                    unit_batch=unit_batch[:midpoint],
                    torch_module=torch_module,
                )
                right_results = self._run_unit_batch(
                    runtime=runtime,
                    unit_batch=unit_batch[midpoint:],
                    torch_module=torch_module,
                )
                return [*left_results, *right_results]
            raise NonRetryableAudioStageError(
                "audEERING MSP-dim inference failed. "
                f"Reason: {_render_exception_detail(exc)}"
            ) from exc
        scores_by_batch = logits.detach().float().cpu().numpy()
        pooled_states_np = pooled_states.detach().float().cpu().numpy()
        return [
            {
                "arousal": float(values[0]) if len(values) > 0 else None,
                "dominance": float(values[1]) if len(values) > 1 else None,
                "valence": float(values[2]) if len(values) > 2 else None,
                "embedding_dimension": int(embedding.shape[-1]),
            }
            for values, embedding in zip(scores_by_batch, pooled_states_np, strict=True)
        ]

    def _build_inference_units(
        self,
        *,
        batch: Sequence[AffectClip],
    ) -> list[_AudeeringInferenceUnit]:
        """Expand long clips into bounded inference windows."""
        units: list[_AudeeringInferenceUnit] = []
        for clip in batch:
            units.extend(self._split_clip_into_units(clip=clip))
        return units

    def _split_clip_into_units(
        self, *, clip: AffectClip
    ) -> list[_AudeeringInferenceUnit]:
        """Split one clip into bounded-duration inference units."""
        sample_rate_hz = int(clip.sample_rate_hz)
        max_samples = max(
            1,
            int(round(self.max_chunk_duration_seconds * float(sample_rate_hz))),
        )
        total_samples = int(np.asarray(clip.samples).shape[-1])
        if total_samples <= max_samples:
            return [
                _AudeeringInferenceUnit(
                    segment_index=clip.segment_index,
                    sample_count=total_samples,
                    clip=clip,
                )
            ]
        units: list[_AudeeringInferenceUnit] = []
        total_chunks = (total_samples + max_samples - 1) // max_samples
        for chunk_index, start_sample in enumerate(
            range(0, total_samples, max_samples)
        ):
            end_sample = min(total_samples, start_sample + max_samples)
            chunk_samples = np.asarray(
                clip.samples[start_sample:end_sample],
                dtype=np.float32,
                copy=False,
            )
            chunk_start_ms = clip.start_time_ms + int(
                round((start_sample / float(sample_rate_hz)) * 1000.0)
            )
            chunk_end_ms = chunk_start_ms + int(
                round((len(chunk_samples) / float(sample_rate_hz)) * 1000.0)
            )
            units.append(
                _AudeeringInferenceUnit(
                    segment_index=clip.segment_index,
                    sample_count=int(len(chunk_samples)),
                    clip=AffectClip(
                        segment_index=clip.segment_index,
                        speaker=clip.speaker,
                        text=clip.text,
                        start_time_ms=chunk_start_ms,
                        end_time_ms=chunk_end_ms,
                        samples=chunk_samples,
                        sample_rate_hz=sample_rate_hz,
                    ),
                )
            )
            event_bus.publish(
                "system_message",
                (
                    "Affect analysis: split long audEERING clip for segment "
                    f"{clip.segment_index} into chunk {chunk_index + 1}/{total_chunks}."
                ),
            )
        return units

    def _aggregate_results_for_clip(
        self,
        *,
        chunk_results: Sequence[tuple[int, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Merge chunk-level dimensional results back into one segment result."""
        if not chunk_results:
            raise NonRetryableAudioStageError(
                "audEERING MSP-dim did not return any chunk results for one transcript segment."
            )
        total_weight = float(sum(max(weight, 1) for weight, _ in chunk_results))
        weighted_values = {label: 0.0 for label in _AUDEERING_LABELS}
        embedding_dimension = None
        for weight, result in chunk_results:
            embedding_dimension = (
                max(
                    int(result.get("embedding_dimension", 0) or 0),
                    int(embedding_dimension or 0),
                )
                or embedding_dimension
            )
            for label in _AUDEERING_LABELS:
                value = result.get(label)
                if value is None:
                    continue
                weighted_values[label] += max(weight, 1) * float(value)
        return {
            label: weighted_values[label] / total_weight for label in _AUDEERING_LABELS
        } | {"embedding_dimension": embedding_dimension}

    def _should_retry_with_smaller_cuda_batch(
        self,
        *,
        runtime: _AudeeringRuntime,
        unit_batch: Sequence[_AudeeringInferenceUnit],
        error: Exception,
    ) -> bool:
        """Return whether one CUDA OOM should retry with smaller micro-batches."""
        return (
            str(runtime.device).startswith("cuda")
            and len(unit_batch) > 1
            and _looks_cuda_oom(error)
        )

    def release_cached_resources(self) -> None:
        """Release loaded runtime state and CUDA allocator slack."""
        runtime = self._runtime_cache.pop(self._device, None)
        if runtime is not None:
            del runtime
        release_torch_device_memory_with_reporting(
            device=self._device,
            description=f"audEERING MSP-dim model '{self.model_name}'",
        )

    def _get_runtime(self, *, device: str) -> _AudeeringRuntime:
        """Lazy-load one runtime per configured device."""
        cached = self._runtime_cache.get(device)
        if cached is not None:
            return cached
        try:
            import torch
            from transformers import Wav2Vec2Processor
        except Exception as exc:
            raise DependencyMissingError(
                "audEERING MSP-dim requires the 'transformers' and 'torch' runtimes."
            ) from exc

        EmotionModel = _build_audeering_emotion_model_class()
        torch_dtype = (
            torch.float16 if device == "cuda" and self.use_fp16 else torch.float32
        )
        target_device = torch.device("cuda" if device == "cuda" else "cpu")
        with report_loading_duration(
            f"audEERING MSP-dim model '{self.model_name}'",
            reporter=lambda message: event_bus.publish("system_message", message),
        ):
            try:
                processor = Wav2Vec2Processor.from_pretrained(self.model_name)
                model = EmotionModel.from_pretrained(self.model_name)
                model = model.to(target_device)
                if torch_dtype == torch.float16:
                    model = model.half()
                model.eval()
            except Exception as exc:
                raise NonRetryableAudioStageError(
                    f"Failed to load audEERING MSP-dim model '{self.model_name}'."
                ) from exc
        runtime = _AudeeringRuntime(
            processor=processor,
            model=model,
            device=target_device,
            torch_dtype=torch_dtype,
        )
        self._runtime_cache[device] = runtime
        return runtime


def _looks_cuda_oom(exc: Exception) -> bool:
    """Return whether one exception looks like a CUDA out-of-memory failure."""
    lowered = f"{exc.__class__.__name__}: {exc}".strip().lower()
    return "cuda out of memory" in lowered or "outofmemoryerror" in lowered


def _render_exception_detail(exc: Exception) -> str:
    """Collapse one exception into a compact single-line detail."""
    detail = " ".join(str(exc).split())
    if detail:
        return detail[:500] + ("..." if len(detail) > 500 else "")
    return exc.__class__.__name__
