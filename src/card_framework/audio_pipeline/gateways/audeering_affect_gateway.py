"""audEERING MSP-dim adapter for segment-level affect regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from card_framework.audio_pipeline.affect_analysis import (
    AffectClip,
    SegmentAffectProvider,
    SegmentAffectUpdate,
)
from card_framework.audio_pipeline.errors import DependencyMissingError, NonRetryableAudioStageError
from card_framework.audio_pipeline.runtime import release_torch_device_memory_with_reporting
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


class AudeeringMspDimAffectGateway(SegmentAffectProvider):
    """Infer arousal, dominance, and valence for transcript segments."""

    def __init__(
        self,
        *,
        model_name: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        device: str = "cpu",
        use_fp16: bool = False,
        batch_size: int = 4,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        self.model_name = model_name
        self._device = str(device).strip().lower() or "cpu"
        self.use_fp16 = bool(use_fp16)
        self.batch_size = int(batch_size)
        self._runtime_cache: dict[str, _AudeeringRuntime] = {}

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier."""
        return "audeering_msp_dim"

    @property
    def device(self) -> str:
        """Return the configured device."""
        return self._device

    def analyze(self, clips: list[AffectClip]) -> list[SegmentAffectUpdate]:
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
        for batch_index, batch_start in enumerate(range(0, len(clips), self.batch_size), start=1):
            batch = clips[batch_start : batch_start + self.batch_size]
            waveforms = [clip.samples for clip in batch]
            try:
                inputs = runtime.processor(
                    waveforms,
                    sampling_rate=batch[0].sample_rate_hz,
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
            model_inputs = input_values.to(runtime.device)
            if runtime.torch_dtype is not None and hasattr(model_inputs, "to"):
                model_inputs = model_inputs.to(dtype=runtime.torch_dtype)
            try:
                with torch.no_grad():
                    pooled_states, logits = runtime.model(model_inputs)
            except Exception as exc:
                raise NonRetryableAudioStageError(
                    "audEERING MSP-dim inference failed."
                ) from exc
            scores_by_batch = logits.detach().float().cpu().numpy()
            pooled_states_np = pooled_states.detach().float().cpu().numpy()
            for clip, values, embedding in zip(
                batch,
                scores_by_batch,
                pooled_states_np,
                strict=True,
            ):
                scores = {
                    _AUDEERING_LABELS[index]: float(value)
                    for index, value in enumerate(values[: len(_AUDEERING_LABELS)])
                }
                updates.append(
                    SegmentAffectUpdate(
                        segment_index=clip.segment_index,
                        arousal=scores.get("arousal"),
                        dominance=scores.get("dominance"),
                        valence=scores.get("valence"),
                        embedding_dimension=int(embedding.shape[-1]),
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
        torch_dtype = torch.float16 if device == "cuda" and self.use_fp16 else torch.float32
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
