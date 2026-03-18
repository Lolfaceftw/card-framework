"""Granite Speech adapter for timed ASR segments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

import numpy as np

from card_framework.audio_pipeline.contracts import (
    SpeechTranscriber,
    TimedTextSegment,
    TranscriptionResult,
    WordTimestamp,
)
from card_framework.audio_pipeline.eta import StageProgressCallback, StageProgressUpdate
from card_framework.audio_pipeline.errors import DependencyMissingError, NonRetryableAudioStageError
from card_framework.audio_pipeline.gateways.ctc_forced_aligner_gateway import CtcForcedAlignerGateway
from card_framework.audio_pipeline.runtime import release_torch_device_memory_with_reporting
from card_framework.shared.events import event_bus
from card_framework.shared.perf import report_loading_duration

_TARGET_SAMPLE_RATE_HZ = 16_000
_DEFAULT_PROMPT = "<|audio|>can you transcribe the speech into a written format?"
_NON_WORD_EDGE_PATTERN = re.compile(r"^\W+|\W+$")
_SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^>]+?\|>")
_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(slots=True, frozen=True)
class _GraniteRuntime:
    """Hold one cached Granite runtime bundle for a concrete device."""

    model: Any
    processor: Any
    tokenizer: Any
    device: Any


class GraniteSpeechTranscriber(SpeechTranscriber):
    """Transcribe audio with IBM Granite Speech."""

    def __init__(
        self,
        *,
        model_name: str = "ibm-granite/granite-4.0-1b-speech",
        language: str | None = "en",
        chunk_length_seconds: int = 25,
        chunk_overlap_seconds: int = 2,
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.1,
        prompt: str = _DEFAULT_PROMPT,
        enable_forced_alignment: bool = True,
        require_forced_alignment: bool = True,
        forced_alignment_batch_size: int = 8,
        forced_aligner: CtcForcedAlignerGateway | None = None,
    ) -> None:
        if chunk_length_seconds <= 0:
            raise ValueError("chunk_length_seconds must be > 0.")
        if chunk_overlap_seconds < 0:
            raise ValueError("chunk_overlap_seconds must be >= 0.")
        if chunk_overlap_seconds >= chunk_length_seconds:
            raise ValueError(
                "chunk_overlap_seconds must be smaller than chunk_length_seconds."
            )
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0.")
        if repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0.")
        if not prompt.strip():
            raise ValueError("prompt must be non-empty.")
        self.model_name = model_name
        self.language = language
        self.chunk_length_seconds = chunk_length_seconds
        self.chunk_overlap_seconds = chunk_overlap_seconds
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.prompt = prompt.strip()
        self.enable_forced_alignment = enable_forced_alignment
        self.require_forced_alignment = require_forced_alignment
        self.forced_alignment_batch_size = forced_alignment_batch_size
        self.forced_aligner = forced_aligner or CtcForcedAlignerGateway()
        self._runtime_cache: dict[str, _GraniteRuntime] = {}

    def transcribe(
        self,
        audio_path: Path,
        *,
        device: str,
        progress_callback: StageProgressCallback | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio and return normalized timed segments."""
        if not audio_path.exists():
            raise NonRetryableAudioStageError(f"Audio file does not exist: {audio_path}")

        runtime = self._get_runtime(device=device)
        waveform = self._load_audio_waveform(audio_path)
        prepared_audio = self._prepare_audio_waveform(waveform=waveform)
        chunk_ranges = self._plan_chunk_ranges(total_samples=len(prepared_audio))
        if not chunk_ranges:
            raise NonRetryableAudioStageError(
                "Granite Speech received empty audio after preprocessing."
            )

        normalized_segments: list[TimedTextSegment] = []
        full_text = ""
        total_chunks = len(chunk_ranges)
        for index, (start_sample, end_sample) in enumerate(chunk_ranges, start=1):
            chunk_audio = prepared_audio[start_sample:end_sample]
            chunk_text = self._transcribe_chunk(runtime=runtime, audio_chunk=chunk_audio)
            unique_text = _dedupe_chunk_text(
                previous_text=full_text,
                current_text=chunk_text,
            )
            start_time_ms = _samples_to_milliseconds(start_sample)
            end_time_ms = _samples_to_milliseconds(end_sample)
            if unique_text:
                normalized_segments.append(
                    TimedTextSegment(
                        start_time_ms=start_time_ms,
                        end_time_ms=end_time_ms,
                        text=unique_text,
                    )
                )
                full_text = f"{full_text} {unique_text}".strip()
            if progress_callback is not None:
                try:
                    progress_callback(
                        StageProgressUpdate(
                            completed_units=index,
                            total_units=total_chunks,
                            processed_audio_ms=end_time_ms,
                            note=f"segment {index}/{total_chunks}",
                        )
                    )
                except Exception:
                    pass

        if not normalized_segments:
            raise NonRetryableAudioStageError(
                "Granite Speech returned no transcript segments."
            )

        resolved_language = (self.language or "en").strip() or "en"
        words = self._resolve_word_timestamps(
            audio_waveform=prepared_audio,
            transcript_text=full_text,
            language=resolved_language,
            device=device,
            fallback_segments=normalized_segments,
        )
        return TranscriptionResult(
            segments=normalized_segments,
            word_timestamps=words,
            language=resolved_language,
        )

    def _get_runtime(self, *, device: str) -> _GraniteRuntime:
        """Lazy-load and cache the Granite processor/model per device."""
        cached_runtime = self._runtime_cache.get(device)
        if cached_runtime is not None:
            return cached_runtime
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except Exception as exc:
            raise DependencyMissingError(
                "Granite Speech requires the 'transformers' and 'torch' runtimes."
            ) from exc

        torch_dtype = _resolve_torch_dtype(torch_module=torch, device=device)
        target_device = torch.device("cuda" if device == "cuda" else "cpu")
        def _reporter(message: str) -> None:
            event_bus.publish("system_message", message)

        with report_loading_duration(
            f"Granite Speech ASR model '{self.model_name}'",
            reporter=_reporter,
        ):
            try:
                processor = AutoProcessor.from_pretrained(self.model_name)
                tokenizer = getattr(processor, "tokenizer", None)
                if tokenizer is None:
                    raise NonRetryableAudioStageError(
                        "Granite Speech processor does not expose a tokenizer."
                    )
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    low_cpu_mem_usage=True,
                )
                model = model.to(target_device)
                model.eval()
            except Exception as exc:
                raise NonRetryableAudioStageError(
                    f"Failed to load Granite Speech model '{self.model_name}'."
                ) from exc
        runtime = _GraniteRuntime(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=target_device,
        )
        self._runtime_cache[device] = runtime
        return runtime

    def release_cached_resources(self, *, device: str) -> None:
        """Drop cached model state for one device and release CUDA allocator slack."""
        runtime = self._runtime_cache.pop(device, None)
        if runtime is not None:
            del runtime
        release_torch_device_memory_with_reporting(
            device=device,
            description=f"Granite Speech ASR model '{self.model_name}'",
        )

    def _load_audio_waveform(self, audio_path: Path):
        """Load one audio file into a tensor waveform."""
        try:
            import torchaudio
        except Exception as exc:
            raise DependencyMissingError(
                "Granite Speech requires the 'torchaudio' runtime."
            ) from exc
        try:
            waveform, sample_rate = torchaudio.load(str(audio_path))
        except Exception as exc:
            raise NonRetryableAudioStageError(
                f"Unable to load audio for Granite Speech: {audio_path}"
            ) from exc
        if waveform.ndim != 2 or waveform.shape[-1] <= 0:
            raise NonRetryableAudioStageError(
                "Granite Speech expected a non-empty audio waveform."
            )
        return waveform, int(sample_rate)

    def _prepare_audio_waveform(self, *, waveform: Any) -> np.ndarray:
        """Convert input waveform to mono 16 kHz float audio."""
        try:
            import torch
            import torchaudio
        except Exception as exc:
            raise DependencyMissingError(
                "Granite Speech requires 'torch' and 'torchaudio' for audio preprocessing."
            ) from exc

        raw_waveform, sample_rate = waveform
        mono_waveform = raw_waveform
        if mono_waveform.shape[0] > 1:
            mono_waveform = mono_waveform.mean(dim=0, keepdim=True)
        if sample_rate != _TARGET_SAMPLE_RATE_HZ:
            mono_waveform = torchaudio.functional.resample(
                mono_waveform,
                sample_rate,
                _TARGET_SAMPLE_RATE_HZ,
            )
        flattened = mono_waveform.squeeze(0).to(torch.float32).cpu()
        return flattened.numpy()

    def _plan_chunk_ranges(self, *, total_samples: int) -> list[tuple[int, int]]:
        """Build overlapping chunk ranges for the prepared audio."""
        if total_samples <= 0:
            return []
        chunk_samples = self.chunk_length_seconds * _TARGET_SAMPLE_RATE_HZ
        step_samples = (
            self.chunk_length_seconds - self.chunk_overlap_seconds
        ) * _TARGET_SAMPLE_RATE_HZ
        if step_samples <= 0:
            raise ValueError("Chunk step must be > 0 after overlap subtraction.")
        ranges: list[tuple[int, int]] = []
        start_sample = 0
        while start_sample < total_samples:
            end_sample = min(total_samples, start_sample + chunk_samples)
            ranges.append((start_sample, end_sample))
            if end_sample >= total_samples:
                break
            start_sample += step_samples
        return ranges

    def _transcribe_chunk(
        self,
        *,
        runtime: _GraniteRuntime,
        audio_chunk: np.ndarray,
    ) -> str:
        """Run Granite generation for one preprocessed audio chunk."""
        if audio_chunk.size == 0:
            return ""
        try:
            import torch
        except Exception as exc:
            raise DependencyMissingError(
                "Granite Speech requires the 'torch' runtime."
            ) from exc

        prompt = self._build_prompt(tokenizer=runtime.tokenizer)
        try:
            model_inputs = runtime.processor(
                prompt,
                audio_chunk,
                device=runtime.device.type,
                return_tensors="pt",
            ).to(runtime.device)
        except Exception as exc:
            raise NonRetryableAudioStageError(
                "Granite Speech processor failed to prepare model inputs."
            ) from exc

        try:
            with torch.inference_mode():
                generated = runtime.model.generate(
                    **model_inputs,
                    max_new_tokens=self.max_new_tokens,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=False,
                    num_beams=1,
                )
            decoded = runtime.processor.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        except Exception as exc:
            raise NonRetryableAudioStageError(
                "Granite Speech generation failed during transcription."
            ) from exc
        if not decoded:
            return ""
        return _strip_prompt_echo(
            text=str(decoded[0]),
            prompt=self.prompt,
            rendered_prompt=prompt,
        )

    def _build_prompt(self, *, tokenizer: Any) -> str:
        """Build the chat-templated Granite transcription prompt."""
        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                templated = apply_chat_template(
                    [{"role": "user", "content": self.prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                templated = self.prompt
            else:
                if isinstance(templated, str) and templated.strip():
                    return templated
        return self.prompt

    def _resolve_word_timestamps(
        self,
        *,
        audio_waveform: np.ndarray,
        transcript_text: str,
        language: str,
        device: str,
        fallback_segments: list[TimedTextSegment],
    ) -> list[WordTimestamp]:
        """Resolve word-level timestamps from forced alignment when enabled."""
        if not self.enable_forced_alignment:
            return self._fallback_word_timestamps(fallback_segments)

        try:
            words = self.forced_aligner.align_words(
                audio_waveform=audio_waveform,
                transcript_text=transcript_text,
                language=language,
                device=device,
                batch_size=self.forced_alignment_batch_size,
            )
        except Exception:
            if self.require_forced_alignment:
                raise
            return self._fallback_word_timestamps(fallback_segments)

        if words:
            return words
        if self.require_forced_alignment:
            raise NonRetryableAudioStageError(
                "Forced alignment completed but returned zero word timestamps."
            )
        return self._fallback_word_timestamps(fallback_segments)

    def _fallback_word_timestamps(
        self,
        segments: list[TimedTextSegment],
    ) -> list[WordTimestamp]:
        """Build best-effort word timestamps from chunk segments."""
        words: list[WordTimestamp] = []
        for segment in segments:
            raw_words = [word for word in re.split(r"\s+", segment.text.strip()) if word]
            if not raw_words:
                continue
            total_duration = max(1, segment.end_time_ms - segment.start_time_ms)
            unit = max(1, total_duration // len(raw_words))
            cursor = segment.start_time_ms
            for index, raw_word in enumerate(raw_words):
                end_time = (
                    segment.end_time_ms
                    if index == len(raw_words) - 1
                    else min(segment.end_time_ms, cursor + unit)
                )
                words.append(
                    WordTimestamp(
                        word=raw_word,
                        start_time_ms=cursor,
                        end_time_ms=end_time,
                    )
                )
                cursor = end_time
        return words


def _resolve_torch_dtype(*, torch_module: Any, device: str) -> Any:
    """Resolve the runtime torch dtype for Granite model loading."""
    if device == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and callable(getattr(cuda, "is_bf16_supported", None)):
            if cuda.is_bf16_supported():
                return torch_module.bfloat16
        return torch_module.float16
    return torch_module.float32


def _samples_to_milliseconds(sample_count: int) -> int:
    """Convert 16 kHz sample counts into millisecond timestamps."""
    return max(0, int(round((sample_count / _TARGET_SAMPLE_RATE_HZ) * 1000.0)))


def _strip_prompt_echo(
    *,
    text: str,
    prompt: str,
    rendered_prompt: str | None = None,
) -> str:
    """Remove Granite prompt echoes from model output when present."""
    normalized = text.strip()
    if not normalized:
        return ""

    for candidate in _iter_prompt_echo_candidates(
        prompt=prompt,
        rendered_prompt=rendered_prompt,
    ):
        candidate_has_role_labels = any(
            role_label in candidate.casefold() for role_label in ("user:", "assistant:")
        )
        if candidate_has_role_labels:
            normalized = _remove_flexible_match(
                text=normalized,
                candidate=candidate,
                replace_all=True,
            )
            continue
        normalized = _remove_flexible_match(
            text=normalized,
            candidate=candidate,
            replace_all=False,
        )
    return _collapse_inline_whitespace(normalized)


def _iter_prompt_echo_candidates(
    *,
    prompt: str,
    rendered_prompt: str | None,
) -> list[str]:
    """Build normalized prompt-echo candidates for cleanup."""
    seen: set[str] = set()
    candidates: list[str] = []
    for raw_candidate in (rendered_prompt, prompt):
        if not raw_candidate:
            continue
        for candidate in (
            _collapse_inline_whitespace(raw_candidate),
            _collapse_inline_whitespace(_SPECIAL_TOKEN_PATTERN.sub(" ", raw_candidate)),
        ):
            if not candidate:
                continue
            dedupe_key = candidate.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            candidates.append(candidate)
    return candidates


def _remove_flexible_match(
    *,
    text: str,
    candidate: str,
    replace_all: bool,
) -> str:
    """Remove one prompt echo candidate with whitespace-tolerant matching."""
    collapsed_candidate = _collapse_inline_whitespace(candidate)
    if not collapsed_candidate:
        return text
    pattern = re.escape(collapsed_candidate).replace(r"\ ", r"\s+")
    if replace_all:
        return re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return re.sub(rf"^{pattern}", "", text, count=1, flags=re.IGNORECASE).lstrip()


def _collapse_inline_whitespace(text: str) -> str:
    """Collapse repeated whitespace inside one transcript chunk."""
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def _dedupe_chunk_text(*, previous_text: str, current_text: str) -> str:
    """Remove repeated boundary tokens between adjacent chunk transcriptions."""
    current_tokens = [token for token in current_text.strip().split() if token]
    if not current_tokens:
        return ""
    previous_tokens = [token for token in previous_text.strip().split() if token]
    if not previous_tokens:
        return " ".join(current_tokens)

    previous_normalized = [_normalize_boundary_token(token) for token in previous_tokens]
    current_normalized = [_normalize_boundary_token(token) for token in current_tokens]
    max_overlap = min(len(previous_normalized), len(current_normalized), 40)
    overlap_size = 0
    for candidate_size in range(max_overlap, 0, -1):
        if previous_normalized[-candidate_size:] == current_normalized[:candidate_size]:
            overlap_size = candidate_size
            break
    deduped_tokens = current_tokens[overlap_size:]
    return " ".join(deduped_tokens).strip()


def _normalize_boundary_token(token: str) -> str:
    """Normalize a token for chunk-boundary overlap matching."""
    normalized = _NON_WORD_EDGE_PATTERN.sub("", token.strip().lower())
    return normalized or token.strip().lower()
