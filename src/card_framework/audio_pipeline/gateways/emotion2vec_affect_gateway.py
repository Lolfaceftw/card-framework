"""emotion2vec+ adapter for segment-level categorical affect analysis."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from collections.abc import Sequence

import numpy as np

from card_framework.audio_pipeline.emotion.affect_analysis import (
    AffectClip,
    AffectScore,
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


@dataclass(slots=True, frozen=True)
class _Emotion2VecRuntime:
    """Hold one loaded emotion2vec runtime for a concrete device."""

    model: Any
    resolved_device: str


@dataclass(slots=True, frozen=True)
class _Emotion2VecInferenceUnit:
    """One bounded-duration inference unit derived from a transcript clip."""

    segment_index: int
    sample_count: int
    clip: AffectClip


class Emotion2VecPlusAffectGateway(SegmentAffectProvider):
    """Infer categorical emotion scores and optional embeddings for clips."""

    def __init__(
        self,
        *,
        model_name: str = "emotion2vec_plus_large",
        device: str = "cpu",
        batch_size: int = 4,
        hub: str = "hf",
        granularity: str = "utterance",
        extract_embedding: bool = True,
        disable_update_check: bool = True,
        max_chunk_duration_seconds: float = 30.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        normalized_granularity = str(granularity).strip().lower()
        if normalized_granularity != "utterance":
            raise ValueError(
                "emotion2vec+ currently supports only utterance granularity in this adapter."
            )
        self.model_name = str(model_name).strip() or "emotion2vec_plus_large"
        self._device = str(device).strip().lower() or "cpu"
        self.batch_size = int(batch_size)
        self.hub = str(hub).strip() or "hf"
        self.granularity = normalized_granularity
        self.extract_embedding = bool(extract_embedding)
        self.disable_update_check = bool(disable_update_check)
        self.max_chunk_duration_seconds = float(max_chunk_duration_seconds)
        if self.max_chunk_duration_seconds <= 0:
            raise ValueError("max_chunk_duration_seconds must be > 0.")
        self._runtime_cache: dict[str, _Emotion2VecRuntime] = {}

    @property
    def provider_name(self) -> str:
        """Return the stable provider identifier."""
        return "emotion2vec_plus"

    @property
    def device(self) -> str:
        """Return the configured execution device."""
        return self._device

    def analyze(self, clips: Sequence[AffectClip]) -> list[SegmentAffectUpdate]:
        """Run categorical SER on one or more prepared clips."""
        if not clips:
            return []
        runtime = self._get_runtime(device=self._device)
        updates: list[SegmentAffectUpdate] = []
        total_batches = (len(clips) + self.batch_size - 1) // self.batch_size
        with TemporaryDirectory(prefix="card_framework_emotion2vec_") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            for batch_index, batch_start in enumerate(
                range(0, len(clips), self.batch_size),
                start=1,
            ):
                batch = list(clips[batch_start : batch_start + self.batch_size])
                inference_units = self._build_inference_units(batch=batch)
                results_by_segment: dict[int, list[tuple[int, dict[str, Any]]]] = {
                    clip.segment_index: [] for clip in batch
                }
                total_unit_batches = (
                    len(inference_units) + self.batch_size - 1
                ) // self.batch_size
                for unit_batch_index, unit_start in enumerate(
                    range(0, len(inference_units), self.batch_size),
                    start=1,
                ):
                    unit_batch = inference_units[
                        unit_start : unit_start + self.batch_size
                    ]
                    batch_inputs = self._write_batch_inputs(
                        batch=[unit.clip for unit in unit_batch],
                        temp_dir=temp_dir,
                    )
                    results = self._generate_batch(
                        runtime=runtime,
                        batch_inputs=batch_inputs,
                        output_dir=(
                            temp_dir
                            / f"batch_{batch_index:04d}_units_{unit_batch_index:04d}_of_{total_unit_batches:04d}"
                        ),
                    )
                    if len(results) != len(unit_batch):
                        raise NonRetryableAudioStageError(
                            "emotion2vec+ returned a different number of results than input clips."
                        )
                    for unit, result in zip(unit_batch, results, strict=True):
                        results_by_segment[unit.segment_index].append(
                            (unit.sample_count, result)
                        )
                for clip in batch:
                    aggregated_result = self._aggregate_results_for_clip(
                        results_by_segment.get(clip.segment_index, [])
                    )
                    updates.append(
                        self._build_update(clip=clip, result=aggregated_result)
                    )
                event_bus.publish(
                    "system_message",
                    (
                        "Affect analysis: emotion2vec+ batch "
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
            description=f"emotion2vec+ model '{self.model_name}'",
        )

    def _get_runtime(self, *, device: str) -> _Emotion2VecRuntime:
        """Lazy-load one runtime per configured device."""
        cached = self._runtime_cache.get(device)
        if cached is not None:
            return cached
        try:
            from funasr import AutoModel
        except Exception as exc:
            raise DependencyMissingError(
                "emotion2vec+ requires the 'funasr' runtime."
            ) from exc

        with report_loading_duration(
            f"emotion2vec+ model '{self.model_name}'",
            reporter=lambda message: event_bus.publish("system_message", message),
        ):
            with _suppress_vendor_output() as captured_output:
                try:
                    model = AutoModel(
                        model=self.model_name,
                        device=_resolve_funasr_device(device),
                        hub=self.hub,
                        disable_update=self.disable_update_check,
                    )
                except Exception as exc:
                    raise NonRetryableAudioStageError(
                        _build_vendor_error_message(
                            base_message=(
                                f"Failed to load emotion2vec+ model '{self.model_name}'."
                            ),
                            captured_output=captured_output.getvalue(),
                        )
                    ) from exc
        runtime = _Emotion2VecRuntime(
            model=model,
            resolved_device=_resolve_funasr_device(device),
        )
        self._runtime_cache[device] = runtime
        return runtime

    def _write_batch_inputs(
        self,
        *,
        batch: Sequence[AffectClip],
        temp_dir: Path,
    ) -> list[str]:
        """Persist one batch of normalized clips as temporary WAV files."""
        try:
            import soundfile as sf
        except Exception as exc:
            raise DependencyMissingError(
                "emotion2vec+ requires the 'soundfile' runtime."
            ) from exc

        batch_inputs: list[str] = []
        for clip in batch:
            output_path = (
                temp_dir
                / f"segment_{clip.segment_index:06d}_{clip.start_time_ms}_{clip.end_time_ms}.wav"
            )
            try:
                sf.write(
                    file=str(output_path),
                    data=np.asarray(clip.samples, dtype=np.float32),
                    samplerate=int(clip.sample_rate_hz),
                )
            except Exception as exc:
                raise NonRetryableAudioStageError(
                    f"emotion2vec+ failed while writing temp audio for segment {clip.segment_index}."
                ) from exc
            batch_inputs.append(str(output_path))
        return batch_inputs

    def _generate_batch(
        self,
        *,
        runtime: _Emotion2VecRuntime,
        batch_inputs: list[str],
        output_dir: Path,
    ) -> list[dict[str, Any]]:
        """Run one emotion2vec+ batch and normalize the result shape."""
        with _suppress_vendor_output() as captured_output:
            try:
                raw_result = runtime.model.generate(
                    input=batch_inputs,
                    output_dir=str(output_dir),
                    granularity=self.granularity,
                    extract_embedding=self.extract_embedding,
                )
            except Exception as exc:
                if self._should_retry_with_smaller_cuda_batch(
                    runtime=runtime,
                    batch_inputs=batch_inputs,
                    error=exc,
                ):
                    event_bus.publish(
                        "system_message",
                        (
                            "Affect analysis: emotion2vec+ batch of "
                            f"{len(batch_inputs)} clip(s) hit CUDA OOM; retrying on "
                            "CUDA with smaller micro-batches."
                        ),
                    )
                    release_torch_device_memory_with_reporting(
                        device="cuda",
                        description="emotion2vec+ CUDA OOM retry preparation",
                    )
                    midpoint = max(1, len(batch_inputs) // 2)
                    left_results = self._generate_batch(
                        runtime=runtime,
                        batch_inputs=batch_inputs[:midpoint],
                        output_dir=output_dir / "part_a",
                    )
                    right_results = self._generate_batch(
                        runtime=runtime,
                        batch_inputs=batch_inputs[midpoint:],
                        output_dir=output_dir / "part_b",
                    )
                    return [*left_results, *right_results]
                raise NonRetryableAudioStageError(
                    _build_vendor_error_message(
                        base_message=(
                            "emotion2vec+ inference failed. "
                            f"Reason: {_render_exception_detail(exc)}"
                        ),
                        captured_output=captured_output.getvalue(),
                    )
                ) from exc
        if isinstance(raw_result, list):
            return [result for result in raw_result if isinstance(result, dict)]
        if isinstance(raw_result, dict):
            return [raw_result]
        raise NonRetryableAudioStageError(
            "emotion2vec+ returned an unsupported result payload."
        )

    def _build_update(
        self,
        *,
        clip: AffectClip,
        result: dict[str, Any],
    ) -> SegmentAffectUpdate:
        """Convert one emotion2vec+ result row into the shared affect contract."""
        raw_labels = result.get("labels", [])
        raw_scores = result.get("scores", [])
        labels = list(raw_labels) if isinstance(raw_labels, list) else []
        scores = list(raw_scores) if isinstance(raw_scores, list) else []
        emotion_scores = tuple(
            AffectScore(
                label=normalized_label,
                score=float(score),
            )
            for label, score in zip(labels, scores, strict=False)
            if (normalized_label := _normalize_emotion2vec_label(label))
        )
        sorted_scores = tuple(
            sorted(emotion_scores, key=lambda item: item.score, reverse=True)
        )
        feats = result.get("feats")
        embedding_dimension = _resolve_embedding_dimension(feats)
        top_label = sorted_scores[0].label if sorted_scores else ""
        top_score = sorted_scores[0].score if sorted_scores else None
        return SegmentAffectUpdate(
            segment_index=clip.segment_index,
            emotion_scores=emotion_scores,
            embedding_dimension=embedding_dimension,
            provider_details={
                self.provider_name: {
                    "provider": self.provider_name,
                    "model_name": self.model_name,
                    "top_label": top_label,
                    "top_score": top_score,
                    "granularity": self.granularity,
                }
            },
        )

    def _build_inference_units(
        self,
        *,
        batch: Sequence[AffectClip],
    ) -> list[_Emotion2VecInferenceUnit]:
        """Expand long clips into bounded inference windows."""
        units: list[_Emotion2VecInferenceUnit] = []
        for clip in batch:
            units.extend(self._split_clip_into_units(clip=clip))
        return units

    def _split_clip_into_units(
        self, *, clip: AffectClip
    ) -> list[_Emotion2VecInferenceUnit]:
        """Split one clip into bounded-duration inference units."""
        sample_rate_hz = int(clip.sample_rate_hz)
        max_samples = max(
            1,
            int(round(self.max_chunk_duration_seconds * float(sample_rate_hz))),
        )
        total_samples = int(np.asarray(clip.samples).shape[-1])
        if total_samples <= max_samples:
            return [
                _Emotion2VecInferenceUnit(
                    segment_index=clip.segment_index,
                    sample_count=total_samples,
                    clip=clip,
                )
            ]
        units: list[_Emotion2VecInferenceUnit] = []
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
                _Emotion2VecInferenceUnit(
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
                    "Affect analysis: split long emotion2vec+ clip for segment "
                    f"{clip.segment_index} into chunk {chunk_index + 1}/{(total_samples + max_samples - 1) // max_samples}."
                ),
            )
        return units

    def _aggregate_results_for_clip(
        self,
        chunk_results: Sequence[tuple[int, dict[str, Any]]],
    ) -> dict[str, Any]:
        """Merge chunk-level categorical results back into one segment result."""
        if not chunk_results:
            raise NonRetryableAudioStageError(
                "emotion2vec+ did not return any chunk results for one transcript segment."
            )
        if len(chunk_results) == 1:
            return dict(chunk_results[0][1])
        weighted_scores: dict[str, float] = {}
        total_weight = float(sum(max(weight, 1) for weight, _ in chunk_results))
        first_feats = None
        for weight, result in chunk_results:
            if first_feats is None and result.get("feats") is not None:
                first_feats = result.get("feats")
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            if not isinstance(labels, list) or not isinstance(scores, list):
                continue
            for label, score in zip(labels, scores, strict=False):
                label_text = str(label).strip()
                if not label_text:
                    continue
                weighted_scores[label_text] = weighted_scores.get(label_text, 0.0) + (
                    max(weight, 1) * float(score)
                )
        sorted_scores = sorted(
            (
                (label, weighted_score / total_weight)
                for label, weighted_score in weighted_scores.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        return {
            "labels": [label for label, _score in sorted_scores],
            "scores": [score for _label, score in sorted_scores],
            "feats": first_feats,
        }

    def _should_retry_with_smaller_cuda_batch(
        self,
        *,
        runtime: _Emotion2VecRuntime,
        batch_inputs: list[str],
        error: Exception,
    ) -> bool:
        """Return whether one CUDA OOM should retry with smaller micro-batches."""
        return (
            runtime.resolved_device.startswith("cuda")
            and len(batch_inputs) > 1
            and _looks_cuda_oom(error)
        )


def _resolve_funasr_device(device: str) -> str:
    """Map repo device strings onto FunASR runtime identifiers."""
    return "cuda:0" if str(device).strip().lower() == "cuda" else "cpu"


def _resolve_embedding_dimension(value: Any) -> int | None:
    """Best-effort extraction of one embedding dimension size."""
    if value is None:
        return None
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and shape:
        return int(shape[-1])
    if isinstance(value, list) and value:
        return len(value)
    return None


def _normalize_emotion2vec_label(raw_label: Any) -> str:
    """Normalize emotion2vec label strings into lowercase English tags."""
    text = str(raw_label).strip()
    if not text:
        return ""
    english_fragment = text.split("/")[-1].strip().lower()
    if english_fragment in {"<unk>", "unk"}:
        return "unknown"
    return english_fragment


def _build_vendor_error_message(*, base_message: str, captured_output: str) -> str:
    """Attach trimmed vendor output to one raised runtime error when present."""
    detail = " ".join(captured_output.split())
    if not detail:
        return base_message
    trimmed = detail[:500]
    suffix = "..." if len(detail) > len(trimmed) else ""
    return f"{base_message} Captured vendor output: {trimmed}{suffix}"


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


class _suppress_vendor_output:
    """Capture noisy third-party stdout and stderr during model operations."""

    def __init__(self) -> None:
        self.buffer = io.StringIO()
        self._stdout = redirect_stdout(self.buffer)
        self._stderr = redirect_stderr(self.buffer)

    def __enter__(self) -> io.StringIO:
        self._stdout.__enter__()
        self._stderr.__enter__()
        return self.buffer

    def __exit__(self, exc_type, exc, exc_tb) -> None:
        self._stderr.__exit__(exc_type, exc, exc_tb)
        self._stdout.__exit__(exc_type, exc, exc_tb)
        return None
