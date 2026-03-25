"""Stage-4 interjector workflow for overlapping backchannels and echo agreements.

Current Literature and Gaps:
    The literature supports turn-end projection, short gaps, and short overlaps in
    natural conversation, but it does not provide a drop-in algorithm for placing
    synthetic, summary-derived listener interjections inside cloned audio. This
    module therefore implements a conservative internal heuristic that adapts those
    findings to this repository's summary-audio pipeline and records the chosen
    assumptions in code.

References:
    [1] H. Sacks, E. A. Schegloff, and G. Jefferson, "A simplest systematics for
    the organization of turn-taking for conversation," Language, vol. 50, no. 4,
    pp. 696-735, 1974, doi: 10.1353/lan.1974.0010.
    [2] J. P. de Ruiter, H. Mitterer, and N. J. Enfield, "Projecting the End of a
    Speaker's Turn: A Cognitive Cornerstone of Conversation," Language, vol. 82,
    no. 3, pp. 515-535, 2006, doi: 10.1353/lan.2006.0130.
    [3] M. Heldner and J. Edlund, "Pauses, gaps and overlaps in conversations,"
    Journal of Phonetics, vol. 38, no. 4, pp. 555-568, 2010,
    doi: 10.1016/j.wocn.2010.08.002.
    [4] S. C. Levinson and F. Torreira, "Timing in turn-taking and its implications
    for processing models of language," Frontiers in Psychology, vol. 6, 2015,
    doi: 10.3389/fpsyg.2015.00731.
    [5] N. Ward, "Using prosodic clues to decide when to produce back-channel
    utterances," in Proc. ICSLP 1996, pp. 1728-1731,
    doi: 10.21437/ICSLP.1996-439.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterator, Literal, Protocol, TypeAlias

from pydantic import BaseModel, Field

from card_framework.audio_pipeline.audio_normalization import (
    AudioNormalizationPolicy,
    normalize_audio_file,
)
from card_framework.audio_pipeline.contracts import WordTimestamp
from card_framework.audio_pipeline.errors import ArtifactWriteError, NonRetryableAudioStageError
from card_framework.audio_pipeline.gateways.ctc_forced_aligner_gateway import CtcForcedAlignerGateway
from card_framework.audio_pipeline.runtime import (
    ensure_command_available,
    probe_audio_duration_ms,
    resolve_command_path,
)
from card_framework.audio_pipeline.voice_clone_contracts import VoiceCloneProvider, VoiceCloneTurn
from card_framework.audio_pipeline.voice_clone_orchestrator import (
    load_speaker_sample_references,
    merge_audio_artifacts_to_wav,
    parse_summary_turns,
)
from card_framework.shared.llm_provider import LLMProvider
from card_framework.shared.events import event_bus


InterjectionStyle: TypeAlias = Literal["backchannel", "echo_agreement"]
AlignmentSource: TypeAlias = Literal[
    "forced_alignment",
    "anchor_window_refinement",
    "echo_phrase_match",
    "uniform_fallback",
]

_FORCED_ALIGNMENT_SAMPLE_RATE_HZ = 16000
_ALIGNMENT_DURATION_TOLERANCE_MS = 50


class InterjectionDecision(BaseModel):
    """Represent one planner decision for a host summary turn."""

    host_turn_index: int = Field(gt=0)
    should_interject: bool = False
    interjection_style: InterjectionStyle | None = None
    interjection_speaker: str | None = None
    interjection_text: str | None = None
    anchor_start_token_index: int | None = None
    anchor_end_token_index: int | None = None
    anchor_text: str | None = None


class InterjectionPlanPayload(BaseModel):
    """Represent the structured planner payload returned by the LLM."""

    decisions: list[InterjectionDecision] = Field(default_factory=list)


class InterjectionRepairPayload(BaseModel):
    """Represent one repair-only response for an interjection candidate."""

    interjection_text: str = ""


_INTERJECTION_DECISION_TOOL_NAME = "submit_interjection_decision"


@dataclass(slots=True, frozen=True)
class AlignedToken:
    """Represent one summary token aligned to a stage-3 audio artifact."""

    token: str
    start_time_ms: int
    end_time_ms: int


@dataclass(slots=True, frozen=True)
class TurnAlignmentResult:
    """Describe one turn-level timing result used for interjection placement."""

    tokens: tuple[AlignedToken, ...]
    source: AlignmentSource


@dataclass(slots=True, frozen=True)
class _OverlayTimingWindow:
    """Describe the legal overlap window for one rendered interjection."""

    overlay_start_time_ms: int
    host_turn_start_time_ms: int
    host_turn_end_time_ms: int
    next_turn_start_time_ms: int
    legal_overlap_end_time_ms: int
    available_overlap_ms: int


@dataclass(slots=True, frozen=True)
class _PreparedOverlayAudio:
    """Describe one synthesized overlay after optional onset cleanup."""

    path: Path
    duration_ms: int
    leading_silence_trim_ms: int
    tempo_adjustment_ratio: float = 1.0
    tempo_adjustment_mode: Literal["none", "phrase", "word_level"] = "none"
    word_level_retiming_steps: tuple[_WordLevelRetimingStep, ...] = ()


@dataclass(slots=True, frozen=True)
class _MatchedPhraseSpan:
    """Describe one matched phrase span inside aligned words."""

    start_word_index: int
    end_word_index: int
    start_time_ms: int
    end_time_ms: int
    similarity: float
    matched_words: tuple[WordTimestamp, ...] = ()

    @property
    def duration_ms(self) -> int:
        """Return the inclusive duration of the matched phrase span."""
        return max(0, self.end_time_ms - self.start_time_ms)


@dataclass(slots=True, frozen=True)
class _WordLevelRetimingStep:
    """Describe one word-sized echo retiming segment."""

    word: str
    source_start_ms: int
    source_end_ms: int
    source_duration_ms: int
    target_duration_ms: int
    applied_tempo_ratio: float
    adjusted_duration_ms: int


@dataclass(slots=True, frozen=True)
class InterjectionRepairAttempt:
    """Describe one rendered interjection candidate and whether it fit."""

    attempt_index: int
    text: str
    duration_ms: int
    available_overlap_ms: int
    timing_source: AlignmentSource
    timing_variant: Literal["default", "tightened"]
    leading_silence_trim_ms: int
    tempo_adjustment_ratio: float
    status: Literal["accepted", "retry"]
    tempo_adjustment_mode: Literal["none", "phrase", "word_level"] = "none"


@dataclass(slots=True, frozen=True)
class InterjectorArtifact:
    """Describe one generated overlapping interjection artifact."""

    host_turn_index: int
    interjection_style: InterjectionStyle
    speaker: str
    text: str
    anchor_text: str
    anchor_start_token_index: int
    anchor_end_token_index: int
    start_time_ms: int
    end_time_ms: int
    duration_ms: int
    host_turn_start_time_ms: int
    host_turn_end_time_ms: int
    next_turn_start_time_ms: int
    legal_overlap_end_time_ms: int
    available_overlap_ms: int
    timing_source: AlignmentSource
    timing_variant: Literal["default", "tightened"]
    leading_silence_trim_ms: int
    tempo_adjustment_ratio: float
    repair_attempt_count: int
    repair_attempts: tuple[InterjectionRepairAttempt, ...]
    output_audio_path: Path
    tempo_adjustment_mode: Literal["none", "phrase", "word_level"] = "none"
    word_level_retiming_steps: tuple[_WordLevelRetimingStep, ...] = ()


@dataclass(slots=True, frozen=True)
class RejectedInterjectionArtifact:
    """Describe one interjection that could not be repaired into a valid overlap."""

    host_turn_index: int
    interjection_style: InterjectionStyle
    speaker: str
    anchor_text: str
    anchor_start_token_index: int
    anchor_end_token_index: int
    host_turn_start_time_ms: int
    host_turn_end_time_ms: int
    next_turn_start_time_ms: int
    legal_overlap_end_time_ms: int
    available_overlap_ms: int
    timing_source: AlignmentSource
    timing_variant: Literal["default", "tightened"]
    failure_reason: str
    last_text: str
    repair_attempts: tuple[InterjectionRepairAttempt, ...]


@dataclass(slots=True, frozen=True)
class InterjectorRunResult:
    """Bundle output paths and metadata for one Stage-4 interjector run."""

    output_dir: Path
    manifest_path: Path
    generated_at_utc: str
    artifacts: tuple[InterjectorArtifact, ...]
    rejected_artifacts: tuple[RejectedInterjectionArtifact, ...]
    merged_output_audio_path: Path


@dataclass(slots=True, frozen=True)
class _EligibleTurn:
    """Represent one host turn and the next speaker eligible to interject."""

    host_turn_index: int
    host_turn: VoiceCloneTurn
    next_turn: VoiceCloneTurn


@dataclass(slots=True, frozen=True)
class _VoiceCloneBaseTurn:
    """Represent one stage-3 turn artifact positioned on the merged timeline."""

    turn_index: int
    speaker: str
    text: str
    output_audio_path: Path
    duration_ms: int
    global_start_time_ms: int


@dataclass(slots=True, frozen=True)
class _VoiceCloneManifestBundle:
    """Represent the stage-3 artifacts consumed by Stage 4."""

    base_turns: tuple[_VoiceCloneBaseTurn, ...]
    base_audio_path: Path
    speaker_samples_manifest_path: Path


@dataclass(slots=True, frozen=True)
class _SynthesizedOverlay:
    """Represent one delayed overlay track to be mixed into the base audio."""

    path: Path
    start_time_ms: int


class InterjectionPlanner(Protocol):
    """Protocol for components that choose where overlaps should be inserted."""

    def plan(self, summary_turns: Sequence[VoiceCloneTurn]) -> list[InterjectionDecision]:
        """Return validated per-turn interjection decisions."""


class WordTimingBackend(Protocol):
    """Protocol for optional short-audio word-timing backends."""

    def align_words(self, *, audio_path: Path, language: str) -> list[WordTimestamp]:
        """Return ordered word timestamps for one short audio clip."""

    def close(self) -> None:
        """Release any cached backend resources."""


@dataclass(slots=True)
class _FasterWhisperNativeWordTimingBackend:
    """Provide native Faster-Whisper word timestamps for short echo fitting."""

    model_name: str = "tiny.en"
    device: str = "cpu"
    compute_type_cpu: str = "int8"
    compute_type_cuda: str = "int8_float16"
    beam_size: int = 1
    _model_cache: dict[tuple[str, str], Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def align_words(self, *, audio_path: Path, language: str) -> list[WordTimestamp]:
        """Return native Whisper word timestamps for one short audio clip."""
        ensure_command_available("ffmpeg")
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - dependency path
            raise NonRetryableAudioStageError(
                "faster-whisper is unavailable for echo pacing fit."
            ) from exc

        compute_type = (
            self.compute_type_cuda if self.device == "cuda" else self.compute_type_cpu
        )
        cache_key = (self.device, compute_type)
        model = self._model_cache.get(cache_key)
        if model is None:
            model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=compute_type,
            )
            self._model_cache[cache_key] = model

        normalized_language = language.strip().lower() or None
        try:
            segments, _info = model.transcribe(
                str(audio_path),
                language=normalized_language,
                beam_size=max(1, self.beam_size),
                word_timestamps=True,
                vad_filter=False,
                condition_on_previous_text=False,
            )
        except Exception as exc:  # pragma: no cover - runtime path
            raise NonRetryableAudioStageError(
                f"Faster-Whisper echo timing failed for {audio_path}."
            ) from exc

        words: list[WordTimestamp] = []
        for segment in segments:
            for word in getattr(segment, "words", ()) or ():
                text_value = str(getattr(word, "word", "") or "").strip()
                start_seconds = getattr(word, "start", None)
                end_seconds = getattr(word, "end", None)
                if not text_value or start_seconds is None or end_seconds is None:
                    continue
                start_ms = max(0, int(round(float(start_seconds) * 1000)))
                end_ms = max(start_ms, int(round(float(end_seconds) * 1000)))
                words.append(
                    WordTimestamp(
                        word=text_value,
                        start_time_ms=start_ms,
                        end_time_ms=end_ms,
                    )
                )
        return words

    def close(self) -> None:
        """Release cached model instances and their device memory."""
        for cache_key, model in list(self._model_cache.items()):
            close = getattr(model, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
            del model
            self._model_cache.pop(cache_key, None)


@dataclass(slots=True)
class LLMInterjectionPlanner:
    """Use an LLM to decide whether summary turns need overlapping interjections."""

    llm: LLMProvider
    max_tokens: int = 900
    repair_max_tokens: int = 220
    max_interjection_words: int = 5
    min_host_progress_ratio: float = 0.35
    echo_min_host_progress_ratio: float = 0.05
    max_host_progress_ratio: float = 0.90
    prefer_tool_call_planning: bool = True

    def plan(self, summary_turns: Sequence[VoiceCloneTurn]) -> list[InterjectionDecision]:
        """Return one validated decision per eligible host turn."""
        from card_framework.shared.prompt_manager import PromptManager

        eligible_turns = _build_eligible_turns(summary_turns)
        if not eligible_turns:
            return []

        system_prompt = PromptManager.get_prompt(
            "interjector_system",
            max_interjection_words=self.max_interjection_words,
            min_host_progress_percent=round(self.min_host_progress_ratio * 100),
            echo_min_host_progress_percent=round(self.echo_min_host_progress_ratio * 100),
            max_host_progress_percent=round(self.max_host_progress_ratio * 100),
        )
        user_prompt = PromptManager.get_prompt(
            "interjector_user",
            eligible_turns_block=_render_eligible_turns_block(
                eligible_turns,
                min_host_progress_ratio=self.min_host_progress_ratio,
                echo_min_host_progress_ratio=self.echo_min_host_progress_ratio,
                max_host_progress_ratio=self.max_host_progress_ratio,
            ),
            max_interjection_words=self.max_interjection_words,
            min_host_progress_percent=round(self.min_host_progress_ratio * 100),
            echo_min_host_progress_percent=round(self.echo_min_host_progress_ratio * 100),
            max_host_progress_percent=round(self.max_host_progress_ratio * 100),
        )
        payload = None
        if self.prefer_tool_call_planning:
            try:
                with _temporarily_disable_llm_thinking(self.llm):
                    payload = self._plan_with_tool_call(
                        eligible_turns=eligible_turns,
                    )
            except Exception:
                payload = None

        if payload is None:
            try:
                with _temporarily_disable_llm_thinking(self.llm):
                    raw_response = self.llm.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=self.max_tokens,
                    )
            except Exception:
                return _build_default_interjection_decisions(eligible_turns)
            payload = _parse_plan_payload(raw_response)

        if payload is None:
            return _build_default_interjection_decisions(eligible_turns)
        return _validate_llm_decisions(
            payload.decisions,
            eligible_turns=eligible_turns,
            max_interjection_words=self.max_interjection_words,
            min_host_progress_ratio=self.min_host_progress_ratio,
            echo_min_host_progress_ratio=self.echo_min_host_progress_ratio,
            max_host_progress_ratio=self.max_host_progress_ratio,
        )

    def repair_interjection_text(
        self,
        *,
        host_turn_text: str,
        next_turn_text: str,
        interjection_style: InterjectionStyle,
        anchor_text: str,
        current_text: str,
        available_overlap_ms: int,
        measured_duration_ms: int,
        attempt_index: int,
    ) -> str | None:
        """Return a shorter grounded repair candidate that fits the overlap budget."""
        from card_framework.shared.prompt_manager import PromptManager

        target_word_budget = _estimate_target_word_budget(
            available_overlap_ms=available_overlap_ms,
            max_interjection_words=self.max_interjection_words,
        )
        system_prompt = (
            "You revise one overlapping listener interjection so it fits a strict "
            "audio timing budget. Keep the same conversational intent, stay "
            "grounded in the host anchor phrase, do not add new facts, and return "
            'JSON only as {"interjection_text": "..."} with no prose.'
        )
        user_prompt = PromptManager.get_prompt(
            "interjector_repair",
            host_turn_text=host_turn_text,
            next_turn_text=next_turn_text,
            interjection_style=interjection_style,
            anchor_text=anchor_text,
            current_text=current_text,
            available_overlap_ms=available_overlap_ms,
            measured_duration_ms=measured_duration_ms,
            attempt_index=attempt_index,
            max_interjection_words=self.max_interjection_words,
            target_word_budget=target_word_budget,
        )
        try:
            with _temporarily_disable_llm_thinking(self.llm):
                raw_response = self.llm.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=self.repair_max_tokens,
                )
        except Exception:
            return None

        payload = _parse_interjection_repair_payload(raw_response)
        if payload is None:
            return None
        normalized = " ".join(payload.interjection_text.split()).strip()
        if not normalized:
            return None
        if len(_split_tokens(normalized)) > self.max_interjection_words:
            return None
        return normalized

    def _plan_with_tool_call(
        self,
        *,
        eligible_turns: Sequence[_EligibleTurn],
    ) -> InterjectionPlanPayload | None:
        """Prefer one small structured tool call per eligible turn."""
        raw_decisions: list[InterjectionDecision] = []

        for item in eligible_turns:
            response_message = self.llm.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Decide whether this single host turn should receive "
                            "a short overlapping listener interjection. "
                            f"Call `{_INTERJECTION_DECISION_TOOL_NAME}` exactly "
                            "once and do not write prose. Use only the provided "
                            "next speaker, keep anchors inside the preferred token "
                            "window, do not add new facts, and keep the "
                            f"interjection to at most {self.max_interjection_words} "
                            "words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Choose the decision for this single eligible turn "
                            "and call the tool once.\n\nEligible turn:\n"
                            + _render_eligible_turns_block(
                                [item],
                                min_host_progress_ratio=self.min_host_progress_ratio,
                                echo_min_host_progress_ratio=self.echo_min_host_progress_ratio,
                                max_host_progress_ratio=self.max_host_progress_ratio,
                            )
                        ),
                    },
                ],
                tools=[_interjection_decision_tool_schema()],
                tool_choice={
                    "type": "function",
                    "function": {"name": _INTERJECTION_DECISION_TOOL_NAME},
                },
                max_tokens=min(self.max_tokens, 400),
            )
            decision = _parse_interjection_decision_from_chat_response(
                response_message=response_message,
                expected_host_turn_index=item.host_turn_index,
            )
            if decision is None:
                raw_decisions.append(
                    InterjectionDecision(
                        host_turn_index=item.host_turn_index,
                        should_interject=False,
                    )
                )
                continue
            raw_decisions.append(decision)

        return InterjectionPlanPayload(decisions=raw_decisions)


@dataclass(slots=True)
class InterjectorOrchestrator:
    """Generate and mix Stage-4 overlapping interjections into merged summary audio."""

    planner: InterjectionPlanner
    provider: VoiceCloneProvider
    output_dir: Path
    manifest_filename: str = "interjector_manifest.json"
    merged_output_filename: str = "voice_cloned_interjected.wav"
    max_interjection_words: int = 5
    max_interjections_per_host_turn: int = 1
    backchannel_reaction_latency_ms: int = 120
    echo_alignment_offset_ms: int = 0
    min_host_progress_ratio: float = 0.35
    echo_min_host_progress_ratio: float = 0.05
    max_host_progress_ratio: float = 0.90
    turn_end_guard_ms: int = 180
    mix_audio_codec: str = "pcm_s24le"
    mix_timeout_seconds: int = 300
    merged_output_normalization: AudioNormalizationPolicy | None = None
    aligner: CtcForcedAlignerGateway | None = None
    alignment_batch_size: int = 8
    timing_refinement_enabled: bool = True
    fallback_anchor_realignment_enabled: bool = True
    repair_enabled: bool = True
    repair_max_attempts: int | None = None
    trim_leading_silence_enabled: bool = True
    trim_leading_silence_threshold_db: float = -50.0
    trim_leading_silence_min_duration_ms: int = 0
    trim_leading_silence_preserve_ms: int = 5
    echo_pacing_fit_enabled: bool = True
    echo_pacing_timing_backend: str = "faster_whisper_native"
    echo_pacing_model_name: str = "tiny.en"
    echo_pacing_device: str = "cpu"
    echo_pacing_compute_type_cpu: str = "int8"
    echo_pacing_compute_type_cuda: str = "int8_float16"
    echo_pacing_beam_size: int = 1
    echo_pacing_target_host_fill_ratio: float = 0.95
    echo_pacing_min_overflow_ratio: float = 1.05
    echo_pacing_max_tempo: float = 1.75
    echo_pacing_min_phrase_match_similarity: float = 0.75
    echo_pacing_max_window_delta: int = 1
    echo_pacing_word_level_enabled: bool = True
    echo_pacing_word_level_min_segment_ms: int = 80
    echo_pacing_word_level_max_tempo: float = 1.75
    _echo_word_timing_backend: WordTimingBackend | None = field(
        init=False,
        default=None,
        repr=False,
    )
    _echo_word_timing_cache: dict[tuple[str, str], tuple[WordTimestamp, ...]] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate orchestrator configuration."""
        if not self.manifest_filename.strip():
            raise ValueError("manifest_filename must be non-empty.")
        if not self.merged_output_filename.strip():
            raise ValueError("merged_output_filename must be non-empty.")
        if Path(self.merged_output_filename).suffix.lower() != ".wav":
            raise ValueError("merged_output_filename must end with '.wav'.")
        if self.max_interjection_words <= 0:
            raise ValueError("max_interjection_words must be > 0.")
        if self.max_interjections_per_host_turn <= 0:
            raise ValueError("max_interjections_per_host_turn must be > 0.")
        if self.backchannel_reaction_latency_ms < 0:
            raise ValueError("backchannel_reaction_latency_ms must be >= 0.")
        if self.echo_alignment_offset_ms < 0:
            raise ValueError("echo_alignment_offset_ms must be >= 0.")
        if not 0.0 <= self.min_host_progress_ratio <= 1.0:
            raise ValueError("min_host_progress_ratio must be within [0.0, 1.0].")
        if not 0.0 <= self.echo_min_host_progress_ratio <= 1.0:
            raise ValueError("echo_min_host_progress_ratio must be within [0.0, 1.0].")
        if not 0.0 <= self.max_host_progress_ratio <= 1.0:
            raise ValueError("max_host_progress_ratio must be within [0.0, 1.0].")
        if self.min_host_progress_ratio > self.max_host_progress_ratio:
            raise ValueError(
                "min_host_progress_ratio must be <= max_host_progress_ratio."
            )
        if self.echo_min_host_progress_ratio > self.max_host_progress_ratio:
            raise ValueError(
                "echo_min_host_progress_ratio must be <= max_host_progress_ratio."
            )
        if self.turn_end_guard_ms < 0:
            raise ValueError("turn_end_guard_ms must be >= 0.")
        if not self.mix_audio_codec.strip():
            raise ValueError("mix_audio_codec must be non-empty.")
        if self.mix_timeout_seconds <= 0:
            raise ValueError("mix_timeout_seconds must be > 0.")
        if self.alignment_batch_size <= 0:
            raise ValueError("alignment_batch_size must be > 0.")
        if self.repair_max_attempts is not None and self.repair_max_attempts <= 0:
            raise ValueError("repair_max_attempts must be > 0 when configured.")
        if self.trim_leading_silence_min_duration_ms < 0:
            raise ValueError("trim_leading_silence_min_duration_ms must be >= 0.")
        if self.trim_leading_silence_preserve_ms < 0:
            raise ValueError("trim_leading_silence_preserve_ms must be >= 0.")
        if self.echo_pacing_beam_size <= 0:
            raise ValueError("echo_pacing_beam_size must be > 0.")
        if not 0.0 < self.echo_pacing_target_host_fill_ratio <= 1.0:
            raise ValueError(
                "echo_pacing_target_host_fill_ratio must be within (0.0, 1.0]."
            )
        if self.echo_pacing_min_overflow_ratio < 1.0:
            raise ValueError("echo_pacing_min_overflow_ratio must be >= 1.0.")
        if not 1.0 <= self.echo_pacing_max_tempo <= 2.0:
            raise ValueError("echo_pacing_max_tempo must be within [1.0, 2.0].")
        if not 0.0 <= self.echo_pacing_min_phrase_match_similarity <= 1.0:
            raise ValueError(
                "echo_pacing_min_phrase_match_similarity must be within [0.0, 1.0]."
            )
        if self.echo_pacing_max_window_delta < 0:
            raise ValueError("echo_pacing_max_window_delta must be >= 0.")
        if self.echo_pacing_word_level_min_segment_ms <= 0:
            raise ValueError("echo_pacing_word_level_min_segment_ms must be > 0.")
        if not 1.0 <= self.echo_pacing_word_level_max_tempo <= 2.0:
            raise ValueError(
                "echo_pacing_word_level_max_tempo must be within [1.0, 2.0]."
            )
        if self.aligner is None:
            self.aligner = CtcForcedAlignerGateway()
        if self.echo_pacing_fit_enabled:
            if self.echo_pacing_timing_backend != "faster_whisper_native":
                raise ValueError(
                    "Unsupported echo_pacing_timing_backend: "
                    f"{self.echo_pacing_timing_backend}"
                )
            self._echo_word_timing_backend = _FasterWhisperNativeWordTimingBackend(
                model_name=self.echo_pacing_model_name,
                device=self.echo_pacing_device,
                compute_type_cpu=self.echo_pacing_compute_type_cpu,
                compute_type_cuda=self.echo_pacing_compute_type_cuda,
                beam_size=self.echo_pacing_beam_size,
            )

    def run(
        self,
        *,
        summary_xml: str,
        voice_clone_manifest_path: Path,
        language: str = "en",
    ) -> InterjectorRunResult:
        """Generate overlaps and mix them into the stage-3 merged audio."""
        try:
            summary_turns = parse_summary_turns(summary_xml)
            manifest_bundle = _load_voice_clone_manifest_bundle(
                summary_turns=summary_turns,
                manifest_path=voice_clone_manifest_path,
            )
            sample_refs = load_speaker_sample_references(
                manifest_bundle.speaker_samples_manifest_path
            )
            decisions = self.planner.plan(summary_turns)

            self.output_dir.mkdir(parents=True, exist_ok=True)
            artifacts, overlays, rejected_artifacts = self._render_interjection_artifacts(
                decisions=decisions,
                manifest_bundle=manifest_bundle,
                sample_refs=sample_refs,
                language=language,
            )
            if not artifacts:
                fallback_decisions = _build_heuristic_interjection_decisions(
                    _build_eligible_turns(summary_turns),
                    max_interjection_words=self.max_interjection_words,
                    min_host_progress_ratio=self.min_host_progress_ratio,
                    max_host_progress_ratio=min(self.max_host_progress_ratio, 0.75),
                    max_interjections=self.max_interjections_per_host_turn,
                )
                fallback_artifacts, fallback_overlays, fallback_rejections = (
                    self._render_interjection_artifacts(
                        decisions=fallback_decisions,
                        manifest_bundle=manifest_bundle,
                        sample_refs=sample_refs,
                        language=language,
                    )
                )
                artifacts = fallback_artifacts
                overlays = fallback_overlays
                rejected_artifacts.extend(fallback_rejections)

            merged_output_audio_path = self.output_dir / self.merged_output_filename
            _mix_audio_with_overlays(
                base_audio_path=manifest_bundle.base_audio_path,
                overlays=overlays,
                output_path=merged_output_audio_path,
                audio_codec=self.mix_audio_codec,
                timeout_seconds=self.mix_timeout_seconds,
            )
            normalize_audio_file(
                audio_path=merged_output_audio_path,
                policy=self.merged_output_normalization,
            )

            generated_at_utc = _utc_now_iso()
            manifest_path = self.output_dir / self.manifest_filename
            _write_json_atomic(
                {
                    "generated_at_utc": generated_at_utc,
                    "voice_clone_manifest_path": str(voice_clone_manifest_path),
                    "speaker_samples_manifest_path": str(
                        manifest_bundle.speaker_samples_manifest_path
                    ),
                    "artifact_count": len(artifacts),
                    "rejected_artifact_count": len(rejected_artifacts),
                    "merged_output_audio_path": str(merged_output_audio_path),
                    "artifacts": [
                    {
                        "host_turn_index": artifact.host_turn_index,
                        "interjection_style": artifact.interjection_style,
                        "speaker": artifact.speaker,
                        "text": artifact.text,
                        "anchor_text": artifact.anchor_text,
                        "anchor_start_token_index": artifact.anchor_start_token_index,
                        "anchor_end_token_index": artifact.anchor_end_token_index,
                        "start_time_ms": artifact.start_time_ms,
                        "end_time_ms": artifact.end_time_ms,
                        "duration_ms": artifact.duration_ms,
                        "host_turn_start_time_ms": artifact.host_turn_start_time_ms,
                        "host_turn_end_time_ms": artifact.host_turn_end_time_ms,
                        "next_turn_start_time_ms": artifact.next_turn_start_time_ms,
                        "legal_overlap_end_time_ms": artifact.legal_overlap_end_time_ms,
                        "available_overlap_ms": artifact.available_overlap_ms,
                        "timing_source": artifact.timing_source,
                        "timing_variant": artifact.timing_variant,
                        "leading_silence_trim_ms": artifact.leading_silence_trim_ms,
                        "tempo_adjustment_ratio": artifact.tempo_adjustment_ratio,
                        "tempo_adjustment_mode": artifact.tempo_adjustment_mode,
                        "word_level_retiming_steps": [
                            {
                                "word": step.word,
                                "source_start_ms": step.source_start_ms,
                                "source_end_ms": step.source_end_ms,
                                "source_duration_ms": step.source_duration_ms,
                                "target_duration_ms": step.target_duration_ms,
                                "applied_tempo_ratio": step.applied_tempo_ratio,
                                "adjusted_duration_ms": step.adjusted_duration_ms,
                            }
                            for step in artifact.word_level_retiming_steps
                        ],
                        "repair_attempt_count": artifact.repair_attempt_count,
                        "repair_attempts": [
                            {
                                "attempt_index": attempt.attempt_index,
                                "text": attempt.text,
                                "duration_ms": attempt.duration_ms,
                                "available_overlap_ms": attempt.available_overlap_ms,
                                "timing_source": attempt.timing_source,
                                "timing_variant": attempt.timing_variant,
                                "leading_silence_trim_ms": attempt.leading_silence_trim_ms,
                                "tempo_adjustment_ratio": attempt.tempo_adjustment_ratio,
                                "tempo_adjustment_mode": attempt.tempo_adjustment_mode,
                                "status": attempt.status,
                            }
                            for attempt in artifact.repair_attempts
                        ],
                        "output_audio_path": str(artifact.output_audio_path),
                    }
                    for artifact in artifacts
                    ],
                    "rejected_artifacts": [
                    {
                        "host_turn_index": artifact.host_turn_index,
                        "interjection_style": artifact.interjection_style,
                        "speaker": artifact.speaker,
                        "anchor_text": artifact.anchor_text,
                        "anchor_start_token_index": artifact.anchor_start_token_index,
                        "anchor_end_token_index": artifact.anchor_end_token_index,
                        "host_turn_start_time_ms": artifact.host_turn_start_time_ms,
                        "host_turn_end_time_ms": artifact.host_turn_end_time_ms,
                        "next_turn_start_time_ms": artifact.next_turn_start_time_ms,
                        "legal_overlap_end_time_ms": artifact.legal_overlap_end_time_ms,
                        "available_overlap_ms": artifact.available_overlap_ms,
                        "timing_source": artifact.timing_source,
                        "timing_variant": artifact.timing_variant,
                        "failure_reason": artifact.failure_reason,
                        "last_text": artifact.last_text,
                        "repair_attempts": [
                            {
                                "attempt_index": attempt.attempt_index,
                                "text": attempt.text,
                                "duration_ms": attempt.duration_ms,
                                "available_overlap_ms": attempt.available_overlap_ms,
                                "timing_source": attempt.timing_source,
                                "timing_variant": attempt.timing_variant,
                                "leading_silence_trim_ms": attempt.leading_silence_trim_ms,
                                "tempo_adjustment_ratio": attempt.tempo_adjustment_ratio,
                                "tempo_adjustment_mode": attempt.tempo_adjustment_mode,
                                "status": attempt.status,
                            }
                            for attempt in artifact.repair_attempts
                        ],
                    }
                    for artifact in rejected_artifacts
                    ],
                },
                manifest_path,
            )
            return InterjectorRunResult(
                output_dir=self.output_dir,
                manifest_path=manifest_path,
                generated_at_utc=generated_at_utc,
                artifacts=tuple(artifacts),
                rejected_artifacts=tuple(rejected_artifacts),
                merged_output_audio_path=merged_output_audio_path,
            )
        finally:
            self._echo_word_timing_cache.clear()
            if self._echo_word_timing_backend is not None:
                try:
                    self._echo_word_timing_backend.close()
                except Exception:
                    pass

    def _render_interjection_artifacts(
        self,
        *,
        decisions: Sequence[InterjectionDecision],
        manifest_bundle: _VoiceCloneManifestBundle,
        sample_refs: dict[str, Any],
        language: str,
    ) -> tuple[
        list[InterjectorArtifact],
        list[_SynthesizedOverlay],
        list[RejectedInterjectionArtifact],
    ]:
        """Render audio artifacts for a set of interjection decisions."""
        artifacts: list[InterjectorArtifact] = []
        overlays: list[_SynthesizedOverlay] = []
        rejected_artifacts: list[RejectedInterjectionArtifact] = []
        host_turn_counts: dict[int, int] = {}
        turn_alignment_cache: dict[int, TurnAlignmentResult] = {}

        for decision in decisions:
            if not decision.should_interject:
                continue

            host_turn = manifest_bundle.base_turns[decision.host_turn_index - 1]
            if (decision.interjection_speaker or "").strip() == host_turn.speaker:
                continue
            host_turn_counts.setdefault(host_turn.turn_index, 0)
            if (
                host_turn_counts[host_turn.turn_index]
                >= self.max_interjections_per_host_turn
            ):
                continue
            alignment_result = turn_alignment_cache.get(host_turn.turn_index)
            if alignment_result is None:
                alignment_result = _align_turn_tokens(
                    audio_path=host_turn.output_audio_path,
                    text=host_turn.text,
                    language=language,
                    aligner=self.aligner,
                    batch_size=self.alignment_batch_size,
                )
                turn_alignment_cache[host_turn.turn_index] = alignment_result
            if not alignment_result.tokens:
                continue

            anchor_tokens, timing_source = self._resolve_anchor_tokens(
                decision=decision,
                host_turn=host_turn,
                alignment_result=alignment_result,
                language=language,
            )
            if not anchor_tokens:
                continue

            reference = sample_refs.get(decision.interjection_speaker or "")
            if reference is None:
                continue

            rendered_artifact, rejected_artifact = self._render_interjection_with_repairs(
                decision=decision,
                host_turn=host_turn,
                next_turn_start_time_ms=host_turn.global_start_time_ms + host_turn.duration_ms,
                timing_source=timing_source,
                anchor_tokens=anchor_tokens,
                reference_audio_path=reference.path,
                next_turn_text=self._lookup_next_turn_text(manifest_bundle, host_turn),
                language=language,
            )
            if rendered_artifact is None:
                if rejected_artifact is not None:
                    rejected_artifacts.append(rejected_artifact)
                continue

            artifact, overlay = rendered_artifact
            artifacts.append(artifact)
            host_turn_counts[host_turn.turn_index] += 1
            overlays.append(overlay)
        return artifacts, overlays, rejected_artifacts

    def _resolve_anchor_tokens(
        self,
        *,
        decision: InterjectionDecision,
        host_turn: _VoiceCloneBaseTurn,
        alignment_result: TurnAlignmentResult,
        language: str,
    ) -> tuple[tuple[AlignedToken, ...], AlignmentSource]:
        """Resolve anchor timing, refining fallback alignment around the anchor when possible."""
        anchor_tokens = alignment_result.tokens[
            decision.anchor_start_token_index : decision.anchor_end_token_index + 1
        ]
        if anchor_tokens and (
            alignment_result.source != "uniform_fallback"
            or not self.timing_refinement_enabled
            or not self.fallback_anchor_realignment_enabled
        ):
            return anchor_tokens, alignment_result.source

        refined_tokens = _refine_anchor_window_alignment(
            audio_path=host_turn.output_audio_path,
            full_text=host_turn.text,
            anchor_start_token_index=decision.anchor_start_token_index or 0,
            anchor_end_token_index=decision.anchor_end_token_index or 0,
            language=language,
            aligner=self.aligner,
            batch_size=self.alignment_batch_size,
        )
        if refined_tokens:
            return refined_tokens, "anchor_window_refinement"
        return anchor_tokens, alignment_result.source

    def _render_interjection_with_repairs(
        self,
        *,
        decision: InterjectionDecision,
        host_turn: _VoiceCloneBaseTurn,
        next_turn_start_time_ms: int,
        timing_source: AlignmentSource,
        anchor_tokens: Sequence[AlignedToken],
        reference_audio_path: Path,
        next_turn_text: str,
        language: str,
    ) -> tuple[
        tuple[InterjectorArtifact, _SynthesizedOverlay] | None,
        RejectedInterjectionArtifact | None,
    ]:
        """Render one interjection and repair it until it fits the legal overlap window."""
        speaker = decision.interjection_speaker or ""
        style = decision.interjection_style or "backchannel"
        current_text = decision.interjection_text or ""
        base_output_audio_path = self.output_dir / _build_interjection_output_filename(
            host_turn_index=host_turn.turn_index,
            speaker=speaker,
            style=style,
        )
        repair_attempts: list[InterjectionRepairAttempt] = []
        attempt_index = 1
        timing_variant: Literal["default", "tightened"] = "default"
        seen_states: set[tuple[str, str]] = set()

        while True:
            echo_phrase_span = self._resolve_echo_phrase_span(
                host_turn=host_turn,
                anchor_tokens=anchor_tokens,
                current_text=current_text,
                language=language,
                style=style,
            )
            resolved_timing_source: AlignmentSource = (
                "echo_phrase_match" if echo_phrase_span is not None else timing_source
            )
            overlay_window = self._compute_overlay_timing_window(
                decision=decision,
                host_turn=host_turn,
                anchor_tokens=anchor_tokens,
                timing_variant=timing_variant,
                echo_phrase_span=echo_phrase_span,
            )
            if overlay_window is None:
                seen_states.add((current_text, timing_variant))
                if (
                    self.repair_max_attempts is not None
                    and attempt_index >= self.repair_max_attempts
                ):
                    return (
                        None,
                        self._build_rejected_interjection_artifact(
                            decision=decision,
                            host_turn=host_turn,
                            next_turn_start_time_ms=next_turn_start_time_ms,
                            timing_source=resolved_timing_source,
                            timing_variant=timing_variant,
                            last_text=current_text,
                            available_overlap_ms=0,
                            repair_attempts=repair_attempts,
                            failure_reason="repair_attempt_limit_reached",
                        ),
                    )
                next_state = self._next_repair_state(
                    decision=decision,
                    host_turn=host_turn,
                    next_turn_text=next_turn_text,
                    current_text=current_text,
                    measured_duration_ms=None,
                    available_overlap_ms=0,
                    attempt_index=attempt_index,
                    timing_variant=timing_variant,
                    seen_states=seen_states,
                )
                if next_state is None:
                    return (
                        None,
                    self._build_rejected_interjection_artifact(
                        decision=decision,
                        host_turn=host_turn,
                        next_turn_start_time_ms=next_turn_start_time_ms,
                        timing_source=resolved_timing_source,
                        timing_variant=timing_variant,
                        last_text=current_text,
                        available_overlap_ms=0,
                        repair_attempts=repair_attempts,
                        failure_reason="no_legal_overlap_window",
                        ),
                    )
                current_text, timing_variant = next_state
                attempt_index += 1
                continue

            attempt_output_path = _build_interjection_attempt_output_path(
                base_output_audio_path=base_output_audio_path,
                attempt_index=attempt_index,
            )
            rendered_path = self.provider.synthesize(
                reference_audio_path=reference_audio_path,
                text=current_text,
                output_audio_path=attempt_output_path,
            )
            raw_duration_ms = probe_audio_duration_ms(rendered_path)
            if raw_duration_ms is None:
                raise NonRetryableAudioStageError(
                    f"Unable to determine interjection duration for {rendered_path}."
                )
            prepared_audio = _prepare_interjection_attempt_audio(
                rendered_path=rendered_path,
                original_duration_ms=raw_duration_ms,
                trim_leading_silence_enabled=self.trim_leading_silence_enabled,
                trim_leading_silence_threshold_db=self.trim_leading_silence_threshold_db,
                trim_leading_silence_min_duration_ms=self.trim_leading_silence_min_duration_ms,
                trim_leading_silence_preserve_ms=self.trim_leading_silence_preserve_ms,
                audio_codec=self.mix_audio_codec,
                timeout_seconds=self.mix_timeout_seconds,
            )
            prepared_audio = self._maybe_apply_echo_pacing_fit(
                decision=decision,
                host_turn=host_turn,
                current_text=current_text,
                prepared_audio=prepared_audio,
                echo_phrase_span=echo_phrase_span,
                language=language,
            )
            duration_ms = prepared_audio.duration_ms

            fits_window = duration_ms <= overlay_window.available_overlap_ms
            repair_attempts.append(
                InterjectionRepairAttempt(
                    attempt_index=attempt_index,
                    text=current_text,
                    duration_ms=duration_ms,
                    available_overlap_ms=overlay_window.available_overlap_ms,
                    timing_source=resolved_timing_source,
                    timing_variant=timing_variant,
                    leading_silence_trim_ms=prepared_audio.leading_silence_trim_ms,
                    tempo_adjustment_ratio=prepared_audio.tempo_adjustment_ratio,
                    tempo_adjustment_mode=prepared_audio.tempo_adjustment_mode,
                    status="accepted" if fits_window else "retry",
                )
            )
            seen_states.add((current_text, timing_variant))
            if fits_window:
                final_output_path = _finalize_interjection_attempt_output(
                    rendered_path=prepared_audio.path,
                    final_output_path=base_output_audio_path,
                )
                artifact = InterjectorArtifact(
                    host_turn_index=host_turn.turn_index,
                    interjection_style=style,
                    speaker=speaker,
                    text=current_text,
                    anchor_text=decision.anchor_text or "",
                    anchor_start_token_index=decision.anchor_start_token_index or 0,
                    anchor_end_token_index=decision.anchor_end_token_index or 0,
                    start_time_ms=overlay_window.overlay_start_time_ms,
                    end_time_ms=overlay_window.overlay_start_time_ms + duration_ms,
                    duration_ms=duration_ms,
                    host_turn_start_time_ms=overlay_window.host_turn_start_time_ms,
                    host_turn_end_time_ms=overlay_window.host_turn_end_time_ms,
                    next_turn_start_time_ms=next_turn_start_time_ms,
                    legal_overlap_end_time_ms=overlay_window.legal_overlap_end_time_ms,
                    available_overlap_ms=overlay_window.available_overlap_ms,
                    timing_source=resolved_timing_source,
                    timing_variant=timing_variant,
                    leading_silence_trim_ms=prepared_audio.leading_silence_trim_ms,
                    tempo_adjustment_ratio=prepared_audio.tempo_adjustment_ratio,
                    tempo_adjustment_mode=prepared_audio.tempo_adjustment_mode,
                    word_level_retiming_steps=prepared_audio.word_level_retiming_steps,
                    repair_attempt_count=max(0, len(repair_attempts) - 1),
                    repair_attempts=tuple(repair_attempts),
                    output_audio_path=final_output_path,
                )
                return (
                    (
                        artifact,
                        _SynthesizedOverlay(
                            path=final_output_path,
                            start_time_ms=overlay_window.overlay_start_time_ms,
                        ),
                    ),
                    None,
                )

            _remove_temp_output(prepared_audio.path)
            if (
                self.repair_max_attempts is not None
                and attempt_index >= self.repair_max_attempts
            ):
                return (
                    None,
                    self._build_rejected_interjection_artifact(
                        decision=decision,
                        host_turn=host_turn,
                        next_turn_start_time_ms=next_turn_start_time_ms,
                        timing_source=resolved_timing_source,
                        timing_variant=timing_variant,
                        last_text=current_text,
                        available_overlap_ms=overlay_window.available_overlap_ms,
                        repair_attempts=repair_attempts,
                        failure_reason="repair_attempt_limit_reached",
                    ),
                )

            next_state = self._next_repair_state(
                decision=decision,
                host_turn=host_turn,
                next_turn_text=next_turn_text,
                current_text=current_text,
                available_overlap_ms=overlay_window.available_overlap_ms,
                measured_duration_ms=duration_ms,
                attempt_index=attempt_index,
                timing_variant=timing_variant,
                seen_states=seen_states,
            )
            if next_state is None:
                return (
                    None,
                    self._build_rejected_interjection_artifact(
                        decision=decision,
                        host_turn=host_turn,
                        next_turn_start_time_ms=next_turn_start_time_ms,
                        timing_source=resolved_timing_source,
                        timing_variant=timing_variant,
                        last_text=current_text,
                        available_overlap_ms=overlay_window.available_overlap_ms,
                        repair_attempts=repair_attempts,
                        failure_reason="repair_state_exhausted",
                    ),
                )
            current_text, timing_variant = next_state
            attempt_index += 1

    def _next_repair_state(
        self,
        *,
        decision: InterjectionDecision,
        host_turn: _VoiceCloneBaseTurn,
        next_turn_text: str,
        current_text: str,
        available_overlap_ms: int,
        measured_duration_ms: int | None,
        attempt_index: int,
        timing_variant: Literal["default", "tightened"],
        seen_states: set[tuple[str, str]],
    ) -> tuple[str, Literal["default", "tightened"]] | None:
        """Return the next repair state without relying on an arbitrary numeric cap."""
        if not self.repair_enabled:
            return None
        event_bus.publish(
            "system_message",
            (
                "Repairing stage-4 interjection so it fits the overlap window: "
                f"host_turn={host_turn.turn_index}, attempt={attempt_index}, "
                f"duration_ms={measured_duration_ms}, "
                f"available_overlap_ms={available_overlap_ms}"
            ),
        )
        text_candidates: list[str] = []
        if measured_duration_ms is not None and attempt_index == 1:
            planner_repair = getattr(self.planner, "repair_interjection_text", None)
            if callable(planner_repair):
                repaired = planner_repair(
                    host_turn_text=host_turn.text,
                    next_turn_text=next_turn_text,
                    interjection_style=decision.interjection_style or "backchannel",
                    anchor_text=decision.anchor_text or "",
                    current_text=current_text,
                    available_overlap_ms=available_overlap_ms,
                    measured_duration_ms=measured_duration_ms,
                    attempt_index=attempt_index,
                )
                if repaired:
                    text_candidates.append(repaired)

        text_candidates.extend(
            candidate
            for candidate in (
                _deterministically_shorten_interjection_text(current_text),
                "Mm.",
                "Yeah.",
                "Right.",
            )
            if candidate
        )

        for candidate in text_candidates:
            normalized = " ".join(candidate.split()).strip()
            if not normalized:
                continue
            if (normalized, timing_variant) in seen_states:
                continue
            return normalized, timing_variant

        if timing_variant == "default":
            tightened_variant: Literal["default", "tightened"] = "tightened"
            for candidate in [current_text, *text_candidates]:
                normalized = " ".join(candidate.split()).strip()
                if not normalized:
                    continue
                if (normalized, tightened_variant) in seen_states:
                    continue
                return normalized, tightened_variant
        return None

    def _build_rejected_interjection_artifact(
        self,
        *,
        decision: InterjectionDecision,
        host_turn: _VoiceCloneBaseTurn,
        next_turn_start_time_ms: int,
        timing_source: AlignmentSource,
        timing_variant: Literal["default", "tightened"],
        last_text: str,
        available_overlap_ms: int,
        repair_attempts: Sequence[InterjectionRepairAttempt],
        failure_reason: str,
    ) -> RejectedInterjectionArtifact:
        """Build one rejected interjection record for the stage-4 manifest."""
        host_turn_end_time_ms = host_turn.global_start_time_ms + host_turn.duration_ms
        return RejectedInterjectionArtifact(
            host_turn_index=host_turn.turn_index,
            interjection_style=decision.interjection_style or "backchannel",
            speaker=decision.interjection_speaker or "",
            anchor_text=decision.anchor_text or "",
            anchor_start_token_index=decision.anchor_start_token_index or 0,
            anchor_end_token_index=decision.anchor_end_token_index or 0,
            host_turn_start_time_ms=host_turn.global_start_time_ms,
            host_turn_end_time_ms=host_turn_end_time_ms,
            next_turn_start_time_ms=next_turn_start_time_ms,
            legal_overlap_end_time_ms=max(
                host_turn.global_start_time_ms,
                host_turn_end_time_ms - self.turn_end_guard_ms,
            ),
            available_overlap_ms=available_overlap_ms,
            timing_source=timing_source,
            timing_variant=timing_variant,
            failure_reason=failure_reason,
            last_text=last_text,
            repair_attempts=tuple(repair_attempts),
        )

    def _resolve_echo_phrase_span(
        self,
        *,
        host_turn: _VoiceCloneBaseTurn,
        anchor_tokens: Sequence[AlignedToken],
        current_text: str,
        language: str,
        style: InterjectionStyle,
    ) -> _MatchedPhraseSpan | None:
        """Return a matched host-phrase span for echo timing when available."""
        if (
            not self.echo_pacing_fit_enabled
            or style != "echo_agreement"
            or self._echo_word_timing_backend is None
        ):
            return None
        normalized_tokens = _normalized_text_tokens(current_text)
        if len(normalized_tokens) < 2:
            return None
        try:
            host_words = self._get_echo_word_timings(
                audio_path=host_turn.output_audio_path,
                language=language,
            )
        except Exception:
            return None
        preferred_center_ms: int | None = None
        if anchor_tokens:
            preferred_center_ms = int(
                round(
                    (anchor_tokens[0].start_time_ms + anchor_tokens[-1].end_time_ms)
                    / 2.0
                )
            )
        return _find_best_matching_phrase_span(
            aligned_words=host_words,
            target_text=current_text,
            min_similarity=self.echo_pacing_min_phrase_match_similarity,
            max_window_delta=self.echo_pacing_max_window_delta,
            preferred_center_ms=preferred_center_ms,
        )

    def _get_echo_word_timings(
        self,
        *,
        audio_path: Path,
        language: str,
    ) -> tuple[WordTimestamp, ...]:
        """Return cached native word timings for one short audio artifact."""
        if self._echo_word_timing_backend is None:
            return ()
        cache_key = (str(audio_path.resolve()), language.strip().lower())
        cached = self._echo_word_timing_cache.get(cache_key)
        if cached is not None:
            return cached
        words = tuple(
            self._echo_word_timing_backend.align_words(
                audio_path=audio_path,
                language=language,
            )
        )
        self._echo_word_timing_cache[cache_key] = words
        return words

    def _maybe_apply_echo_pacing_fit(
        self,
        *,
        decision: InterjectionDecision,
        host_turn: _VoiceCloneBaseTurn,
        current_text: str,
        prepared_audio: _PreparedOverlayAudio,
        echo_phrase_span: _MatchedPhraseSpan | None,
        language: str,
    ) -> _PreparedOverlayAudio:
        """Time-compress one echo overlay when its internal pacing is too slow."""
        if (
            not self.echo_pacing_fit_enabled
            or decision.interjection_style != "echo_agreement"
            or echo_phrase_span is None
            or echo_phrase_span.duration_ms <= 0
            or self._echo_word_timing_backend is None
        ):
            return prepared_audio
        normalized_tokens = _normalized_text_tokens(current_text)
        if len(normalized_tokens) < 2:
            return prepared_audio
        try:
            overlay_words = self._get_echo_word_timings(
                audio_path=prepared_audio.path,
                language=language,
            )
        except Exception:
            return prepared_audio
        overlay_phrase_span = _find_best_matching_phrase_span(
            aligned_words=overlay_words,
            target_text=current_text,
            min_similarity=self.echo_pacing_min_phrase_match_similarity,
            max_window_delta=self.echo_pacing_max_window_delta,
        )
        if overlay_phrase_span is None or overlay_phrase_span.duration_ms <= 0:
            return prepared_audio

        target_phrase_duration_ms = max(
            1,
            int(
                round(
                    echo_phrase_span.duration_ms * self.echo_pacing_target_host_fill_ratio
                )
            ),
        )
        overflow_ratio = overlay_phrase_span.duration_ms / float(target_phrase_duration_ms)
        if overflow_ratio < self.echo_pacing_min_overflow_ratio:
            return prepared_audio

        if self.echo_pacing_word_level_enabled:
            expected_word_count = len(normalized_tokens)
            if (
                len(echo_phrase_span.matched_words) == expected_word_count
                and len(overlay_phrase_span.matched_words) == expected_word_count
            ):
                word_level_plan = _build_word_level_echo_retiming_plan(
                    host_phrase_span=echo_phrase_span,
                    overlay_phrase_span=overlay_phrase_span,
                    target_host_fill_ratio=self.echo_pacing_target_host_fill_ratio,
                    min_overflow_ratio=self.echo_pacing_min_overflow_ratio,
                    max_tempo=self.echo_pacing_word_level_max_tempo,
                    min_segment_ms=self.echo_pacing_word_level_min_segment_ms,
                )
                if word_level_plan:
                    adjusted_duration_ms, retimed_steps = _apply_word_level_echo_retiming_in_place(
                        audio_path=prepared_audio.path,
                        steps=word_level_plan,
                        audio_codec=self.mix_audio_codec,
                        timeout_seconds=self.mix_timeout_seconds,
                    )
                    if adjusted_duration_ms is not None and adjusted_duration_ms > 0:
                        return _PreparedOverlayAudio(
                            path=prepared_audio.path,
                            duration_ms=adjusted_duration_ms,
                            leading_silence_trim_ms=prepared_audio.leading_silence_trim_ms,
                            tempo_adjustment_ratio=max(
                                step.applied_tempo_ratio for step in retimed_steps
                            ),
                            tempo_adjustment_mode="word_level",
                            word_level_retiming_steps=retimed_steps,
                        )

        tempo_adjustment_ratio = min(
            self.echo_pacing_max_tempo,
            max(1.0, overflow_ratio),
        )
        adjusted_duration_ms = _apply_audio_tempo_in_place(
            audio_path=prepared_audio.path,
            tempo_ratio=tempo_adjustment_ratio,
            audio_codec=self.mix_audio_codec,
            timeout_seconds=self.mix_timeout_seconds,
        )
        if adjusted_duration_ms is None or adjusted_duration_ms <= 0:
            return prepared_audio
        return _PreparedOverlayAudio(
            path=prepared_audio.path,
            duration_ms=adjusted_duration_ms,
            leading_silence_trim_ms=prepared_audio.leading_silence_trim_ms,
            tempo_adjustment_ratio=tempo_adjustment_ratio,
            tempo_adjustment_mode="phrase",
        )

    def _lookup_next_turn_text(
        self,
        manifest_bundle: _VoiceCloneManifestBundle,
        host_turn: _VoiceCloneBaseTurn,
    ) -> str:
        """Return the next full summary turn text after the host turn."""
        next_turn_offset = host_turn.turn_index
        if next_turn_offset < len(manifest_bundle.base_turns):
            return manifest_bundle.base_turns[next_turn_offset].text
        return ""

    def _compute_overlay_timing_window(
        self,
        *,
        decision: InterjectionDecision,
        host_turn: _VoiceCloneBaseTurn,
        anchor_tokens: Sequence[AlignedToken],
        timing_variant: Literal["default", "tightened"] = "default",
        echo_phrase_span: _MatchedPhraseSpan | None = None,
    ) -> _OverlayTimingWindow | None:
        """Return the legal overlap timing window for a validated decision."""
        if host_turn.duration_ms <= 0:
            return None

        if (
            echo_phrase_span is not None
            and (decision.interjection_style or "backchannel") == "echo_agreement"
        ):
            anchor_start_ms = host_turn.global_start_time_ms + echo_phrase_span.start_time_ms
            anchor_end_ms = host_turn.global_start_time_ms + echo_phrase_span.end_time_ms
        else:
            anchor_start_ms = host_turn.global_start_time_ms + anchor_tokens[0].start_time_ms
            anchor_end_ms = host_turn.global_start_time_ms + anchor_tokens[-1].end_time_ms
        if timing_variant == "tightened":
            overlay_start_time_ms = anchor_start_ms
        elif (decision.interjection_style or "backchannel") == "echo_agreement":
            # IEEE citation: Adapted overlap-start heuristic from turn-projection and
            # backchannel timing literature [2], [4], [5].
            overlay_start_time_ms = anchor_start_ms + self.echo_alignment_offset_ms
        else:
            # IEEE citation: Short post-anchor reaction timing is grounded in rapid
            # turn-taking findings and backchannel cue work [3], [4], [5].
            overlay_start_time_ms = anchor_end_ms + self.backchannel_reaction_latency_ms

        progress_ratio = (
            overlay_start_time_ms - host_turn.global_start_time_ms
        ) / float(host_turn.duration_ms)
        min_progress_ratio = (
            self.echo_min_host_progress_ratio
            if (decision.interjection_style or "backchannel") == "echo_agreement"
            else self.min_host_progress_ratio
        )
        if progress_ratio < min_progress_ratio:
            return None
        if progress_ratio > self.max_host_progress_ratio:
            return None

        host_turn_end_time_ms = host_turn.global_start_time_ms + host_turn.duration_ms
        legal_overlap_end_time_ms = max(
            host_turn.global_start_time_ms,
            host_turn_end_time_ms - self.turn_end_guard_ms,
        )
        available_overlap_ms = legal_overlap_end_time_ms - overlay_start_time_ms
        if available_overlap_ms <= 0:
            return None
        return _OverlayTimingWindow(
            overlay_start_time_ms=overlay_start_time_ms,
            host_turn_start_time_ms=host_turn.global_start_time_ms,
            host_turn_end_time_ms=host_turn_end_time_ms,
            next_turn_start_time_ms=host_turn_end_time_ms,
            legal_overlap_end_time_ms=legal_overlap_end_time_ms,
            available_overlap_ms=available_overlap_ms,
        )


@contextmanager
def _temporarily_disable_llm_thinking(llm: LLMProvider) -> Iterator[None]:
    """Temporarily disable visible thinking on wrapped providers when supported."""
    toggled_providers: list[tuple[Any, bool]] = []
    current_provider: Any = llm
    seen_provider_ids: set[int] = set()

    while current_provider is not None and id(current_provider) not in seen_provider_ids:
        seen_provider_ids.add(id(current_provider))
        if hasattr(current_provider, "enable_thinking"):
            original_value = bool(getattr(current_provider, "enable_thinking"))
            if original_value:
                setattr(current_provider, "enable_thinking", False)
                toggled_providers.append((current_provider, original_value))
        current_provider = getattr(current_provider, "inner_provider", None)

    try:
        yield
    finally:
        for provider, original_value in reversed(toggled_providers):
            setattr(provider, "enable_thinking", original_value)


def _build_eligible_turns(summary_turns: Sequence[VoiceCloneTurn]) -> list[_EligibleTurn]:
    """Return host turns that can receive an overlap from the next speaker."""
    eligible: list[_EligibleTurn] = []
    for index, host_turn in enumerate(summary_turns[:-1], start=1):
        next_turn = summary_turns[index]
        if next_turn.speaker == host_turn.speaker:
            continue
        eligible.append(
            _EligibleTurn(
                host_turn_index=index,
                host_turn=host_turn,
                next_turn=next_turn,
            )
        )
    return eligible


def _render_eligible_turns_block(
    eligible_turns: Sequence[_EligibleTurn],
    *,
    min_host_progress_ratio: float,
    echo_min_host_progress_ratio: float,
    max_host_progress_ratio: float,
) -> str:
    """Render prompt-friendly eligible turn descriptions for the LLM planner."""
    lines: list[str] = []
    for item in eligible_turns:
        host_tokens = _split_tokens(item.host_turn.text)
        backchannel_anchor_window = _compute_anchor_window_indices(
            host_token_count=len(host_tokens),
            min_host_progress_ratio=min_host_progress_ratio,
            max_host_progress_ratio=max_host_progress_ratio,
        )
        echo_anchor_window = _compute_anchor_window_indices(
            host_token_count=len(host_tokens),
            min_host_progress_ratio=echo_min_host_progress_ratio,
            max_host_progress_ratio=max_host_progress_ratio,
        )
        backchannel_window_text = (
            "none"
            if backchannel_anchor_window is None
            else f"{backchannel_anchor_window[0]}..{backchannel_anchor_window[1]}"
        )
        echo_window_text = (
            "none"
            if echo_anchor_window is None
            else f"{echo_anchor_window[0]}..{echo_anchor_window[1]}"
        )
        anchor_window_text = (
            "Preferred backchannel anchor token window "
            f"(approx {round(min_host_progress_ratio * 100)}%-"
            f"{round(max_host_progress_ratio * 100)}% through host turn): "
            f"{backchannel_window_text}. "
            "Preferred echo anchor token window "
            f"(approx {round(echo_min_host_progress_ratio * 100)}%-"
            f"{round(max_host_progress_ratio * 100)}% through host turn): "
            f"{echo_window_text}. "
            "If a style's window is `none`, do not use that style on this turn. "
            "For backchannel, keep `anchor_end_token_index` inside the backchannel window. "
            "For echo_agreement, keep `anchor_start_token_index` inside the echo window."
        )
        lines.extend(
            [
                f"Host turn index: {item.host_turn_index}",
                f"Host speaker: {item.host_turn.speaker}",
                f"Next speaker: {item.next_turn.speaker}",
                f"Host token count: {len(host_tokens)}",
                anchor_window_text,
                (
                    "Host tokens (0-based whitespace split): "
                    f"{json.dumps(host_tokens, ensure_ascii=False)}"
                ),
                f"Host text: {item.host_turn.text}",
                f"Next full turn text: {item.next_turn.text}",
                "",
            ]
        )
    return "\n".join(lines).strip()


def _build_default_interjection_decisions(
    eligible_turns: Sequence[_EligibleTurn],
) -> list[InterjectionDecision]:
    """Return all-false decisions when planning fails."""
    return [
        InterjectionDecision(
            host_turn_index=item.host_turn_index,
            should_interject=False,
        )
        for item in eligible_turns
    ]


def _build_heuristic_interjection_decisions(
    eligible_turns: Sequence[_EligibleTurn],
    *,
    max_interjection_words: int,
    min_host_progress_ratio: float,
    max_host_progress_ratio: float,
    max_interjections: int,
) -> list[InterjectionDecision]:
    """Return deterministic fallback decisions when the LLM yields no usable plan."""
    decisions: list[InterjectionDecision] = []
    emitted_interjections = 0

    for item in eligible_turns:
        if emitted_interjections >= max_interjections:
            decisions.append(
                InterjectionDecision(
                    host_turn_index=item.host_turn_index,
                    should_interject=False,
                )
            )
            continue

        host_tokens = _split_tokens(item.host_turn.text)
        anchor_window = _compute_anchor_window_indices(
            host_token_count=len(host_tokens),
            min_host_progress_ratio=min_host_progress_ratio,
            max_host_progress_ratio=max_host_progress_ratio,
        )
        interjection_text = _choose_heuristic_interjection_text(
            item.next_turn.text,
            max_interjection_words=max_interjection_words,
        )
        if anchor_window is None or not interjection_text:
            decisions.append(
                InterjectionDecision(
                    host_turn_index=item.host_turn_index,
                    should_interject=False,
                )
            )
            continue

        anchor_end_index = min(
            max(anchor_window[0], math.floor(len(host_tokens) * 0.58) - 1),
            anchor_window[1],
        )
        anchor_end_index = max(anchor_window[0], anchor_end_index)
        anchor_text = host_tokens[anchor_end_index].strip()
        decisions.append(
            InterjectionDecision(
                host_turn_index=item.host_turn_index,
                should_interject=True,
                interjection_style="backchannel",
                interjection_speaker=item.next_turn.speaker,
                interjection_text=interjection_text,
                anchor_start_token_index=anchor_end_index,
                anchor_end_token_index=anchor_end_index,
                anchor_text=anchor_text,
            )
        )
        emitted_interjections += 1

    return decisions


def _choose_heuristic_interjection_text(
    next_turn_text: str,
    *,
    max_interjection_words: int,
) -> str:
    """Choose a short grounded backchannel from the next speaker turn."""
    normalized_tokens = [
        _normalize_token(token) for token in _split_tokens(next_turn_text)[:4]
    ]
    canonical_by_token = {
        "yeah": "Yeah.",
        "yep": "Yeah.",
        "yes": "Yeah.",
        "right": "Right.",
        "totally": "Totally.",
        "exactly": "Exactly.",
        "absolutely": "Absolutely.",
    }
    for token in normalized_tokens:
        if token in canonical_by_token:
            candidate = canonical_by_token[token]
            if len(_split_tokens(candidate)) <= max_interjection_words:
                return candidate
    return "Yeah." if max_interjection_words >= 1 else ""


def _interjection_decision_tool_schema() -> dict[str, Any]:
    """Return the structured tool contract for one eligible turn decision."""
    return {
        "type": "function",
        "function": {
            "name": _INTERJECTION_DECISION_TOOL_NAME,
            "description": "Return the interjection decision for this single host turn.",
            "parameters": {
                "type": "object",
                "properties": {
                    "host_turn_index": {"type": "integer"},
                    "should_interject": {"type": "boolean"},
                    "interjection_style": {
                        "type": "string",
                        "enum": ["backchannel", "echo_agreement"],
                    },
                    "interjection_speaker": {"type": "string"},
                    "interjection_text": {"type": "string"},
                    "anchor_start_token_index": {"type": "integer"},
                    "anchor_end_token_index": {"type": "integer"},
                    "anchor_text": {"type": "string"},
                },
                "required": ["host_turn_index", "should_interject"],
            },
        },
    }


def _parse_interjection_decision_from_chat_response(
    *,
    response_message: Any,
    expected_host_turn_index: int,
) -> InterjectionDecision | None:
    """Extract one structured interjection decision from an assistant tool call."""
    raw_tool_calls = getattr(response_message, "tool_calls", None)
    if not raw_tool_calls:
        return None

    for tool_call in raw_tool_calls:
        function = getattr(tool_call, "function", None)
        if function is None:
            continue
        if str(getattr(function, "name", "")).strip() != _INTERJECTION_DECISION_TOOL_NAME:
            continue
        raw_arguments = str(getattr(function, "arguments", "") or "")
        if not raw_arguments.strip():
            continue
        try:
            decision = InterjectionDecision.model_validate_json(raw_arguments)
        except Exception:
            return None
        if decision.host_turn_index != expected_host_turn_index:
            return None
        return decision
    return None


def _parse_plan_payload(raw_response: str) -> InterjectionPlanPayload | None:
    """Extract and validate a planner payload from model output text."""
    candidate = _extract_json_candidate(raw_response)
    if candidate is None:
        return None
    try:
        return InterjectionPlanPayload.model_validate_json(candidate)
    except Exception:
        return _parse_partial_plan_payload(candidate)


def _extract_json_candidate(raw_response: str) -> str | None:
    """Return the most likely JSON object embedded in model output."""
    stripped = raw_response.strip()
    if not stripped:
        return None

    candidates = [stripped]
    fenced_match = re.search(r"```json\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())
    start_index = stripped.find("{")
    end_index = stripped.rfind("}")
    if start_index != -1 and end_index > start_index:
        candidates.append(stripped[start_index : end_index + 1])
    return next((candidate for candidate in candidates if candidate.strip()), None)


def _parse_partial_plan_payload(candidate: str) -> InterjectionPlanPayload | None:
    """Recover complete decisions from a truncated planner payload when possible."""
    decisions_array = _extract_decisions_array_text(candidate)
    if decisions_array is None:
        return None
    decoder = json.JSONDecoder()
    parsed_decisions: list[dict[str, Any]] = []
    cursor = 0
    while cursor < len(decisions_array):
        while cursor < len(decisions_array) and decisions_array[cursor] in {
            " ",
            "\t",
            "\r",
            "\n",
            ",",
        }:
            cursor += 1
        if cursor >= len(decisions_array):
            break
        if decisions_array[cursor] == "]":
            break
        try:
            parsed_value, next_cursor = decoder.raw_decode(decisions_array, cursor)
        except json.JSONDecodeError:
            break
        if isinstance(parsed_value, dict):
            parsed_decisions.append(parsed_value)
        cursor = next_cursor
    if not parsed_decisions:
        return None
    try:
        return InterjectionPlanPayload.model_validate({"decisions": parsed_decisions})
    except Exception:
        return None


def _extract_decisions_array_text(candidate: str) -> str | None:
    """Return the raw decisions-array slice from complete or truncated JSON text."""
    match = re.search(r'"decisions"\s*:\s*\[', candidate)
    if match is None:
        return None
    return candidate[match.end() :]


def _parse_interjection_repair_payload(raw_response: str) -> InterjectionRepairPayload | None:
    """Parse one repair-only JSON payload from an LLM response."""
    candidate = raw_response.strip()
    if not candidate:
        return None
    try:
        return InterjectionRepairPayload.model_validate_json(candidate)
    except Exception:
        pass

    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if match is None:
        return None
    try:
        return InterjectionRepairPayload.model_validate_json(match.group(0))
    except Exception:
        return None


def _validate_llm_decisions(
    raw_decisions: Sequence[InterjectionDecision],
    *,
    eligible_turns: Sequence[_EligibleTurn],
    max_interjection_words: int,
    min_host_progress_ratio: float = 0.35,
    echo_min_host_progress_ratio: float = 0.05,
    max_host_progress_ratio: float = 0.90,
) -> list[InterjectionDecision]:
    """Validate planner output against deterministic pipeline constraints."""
    eligible_by_index = {item.host_turn_index: item for item in eligible_turns}
    decisions_by_host_index: dict[int, InterjectionDecision] = {}

    for raw_decision in raw_decisions:
        eligible = eligible_by_index.get(raw_decision.host_turn_index)
        if eligible is None or raw_decision.host_turn_index in decisions_by_host_index:
            continue
        if not raw_decision.should_interject:
            decisions_by_host_index[raw_decision.host_turn_index] = InterjectionDecision(
                host_turn_index=raw_decision.host_turn_index,
                should_interject=False,
            )
            continue

        host_tokens = _split_tokens(eligible.host_turn.text)
        if not host_tokens:
            continue
        start_index = raw_decision.anchor_start_token_index
        end_index = raw_decision.anchor_end_token_index
        if start_index is None or end_index is None:
            continue
        if start_index < 0 or end_index < start_index or end_index >= len(host_tokens):
            continue
        style = raw_decision.interjection_style
        if style not in {"backchannel", "echo_agreement"}:
            continue
        if style == "echo_agreement":
            trimmed_anchor_span = _trim_echo_agreement_anchor_indices(
                host_tokens,
                start_index=start_index,
                end_index=end_index,
            )
            if trimmed_anchor_span is None:
                continue
            start_index, end_index = trimmed_anchor_span
        anchor_progress_ratio = _estimate_anchor_progress_ratio(
            host_token_count=len(host_tokens),
            anchor_start_token_index=start_index,
            anchor_end_token_index=end_index,
            style=style,
        )
        min_progress_ratio = (
            echo_min_host_progress_ratio if style == "echo_agreement" else min_host_progress_ratio
        )
        if anchor_progress_ratio < min_progress_ratio:
            continue
        if anchor_progress_ratio > max_host_progress_ratio:
            continue

        anchor_text = " ".join(host_tokens[start_index : end_index + 1]).strip()
        interjection_speaker = (
            raw_decision.interjection_speaker or eligible.next_turn.speaker
        ).strip()
        if interjection_speaker != eligible.next_turn.speaker:
            continue

        interjection_text = " ".join(
            str(raw_decision.interjection_text or "").split()
        ).strip()
        if style == "echo_agreement":
            interjection_text = _sanitize_echo_agreement_text(interjection_text)
        if style == "echo_agreement" and not interjection_text:
            interjection_text = anchor_text
        elif style == "echo_agreement" and not _texts_share_anchor_tokens(
            interjection_text,
            anchor_text,
        ):
            continue
        if not interjection_text:
            continue
        if len(_split_tokens(interjection_text)) > max_interjection_words:
            continue
        if style == "echo_agreement" and len(_split_tokens(anchor_text)) > max_interjection_words:
            continue

        decisions_by_host_index[raw_decision.host_turn_index] = InterjectionDecision(
            host_turn_index=raw_decision.host_turn_index,
            should_interject=True,
            interjection_style=style,
            interjection_speaker=interjection_speaker,
            interjection_text=interjection_text,
            anchor_start_token_index=start_index,
            anchor_end_token_index=end_index,
            anchor_text=anchor_text,
        )

    return [
        decisions_by_host_index.get(
            item.host_turn_index,
            InterjectionDecision(
                host_turn_index=item.host_turn_index,
                should_interject=False,
            ),
        )
        for item in eligible_turns
    ]


def _compute_anchor_window_indices(
    *,
    host_token_count: int,
    min_host_progress_ratio: float,
    max_host_progress_ratio: float,
) -> tuple[int, int] | None:
    """Approximate the valid host-token window for stage-4 anchors."""
    if host_token_count <= 0:
        return None
    min_index = max(0, math.ceil(host_token_count * min_host_progress_ratio) - 1)
    max_index = min(
        host_token_count - 1,
        math.floor(host_token_count * max_host_progress_ratio) - 1,
    )
    if max_index < min_index:
        return None
    return min_index, max_index


def _estimate_anchor_progress_ratio(
    *,
    host_token_count: int,
    anchor_start_token_index: int,
    anchor_end_token_index: int,
    style: InterjectionStyle,
) -> float:
    """Approximate anchor placement as a fraction of the host turn."""
    if style == "echo_agreement":
        anchor_token_position = anchor_start_token_index + 1
    else:
        anchor_token_position = anchor_end_token_index + 1
    return anchor_token_position / float(host_token_count)


def _texts_share_anchor_tokens(interjection_text: str, anchor_text: str) -> bool:
    """Return whether two short phrases share at least one normalized token."""
    interjection_tokens = {
        _normalize_token(token)
        for token in _split_tokens(interjection_text)
        if _normalize_token(token)
    }
    anchor_tokens = {
        _normalize_token(token)
        for token in _split_tokens(anchor_text)
        if _normalize_token(token)
    }
    return bool(interjection_tokens & anchor_tokens)


def _sanitize_echo_agreement_text(text: str) -> str:
    """Trim trailing text that spills past a terminal punctuation boundary."""
    tokens = _split_tokens(text)
    if not tokens:
        return ""
    last_terminal_index: int | None = None
    for index, token in enumerate(tokens[:-1]):
        if _token_ends_with_terminal_punctuation(token):
            last_terminal_index = index
    if last_terminal_index is None:
        return " ".join(tokens).strip()
    return " ".join(tokens[: last_terminal_index + 1]).strip()


def _trim_echo_agreement_anchor_indices(
    host_tokens: Sequence[str],
    *,
    start_index: int,
    end_index: int,
) -> tuple[int, int] | None:
    """Trim an echo anchor span back to the last sentence-ending token when present."""
    last_terminal_index: int | None = None
    for index in range(start_index, end_index):
        if _token_ends_with_terminal_punctuation(host_tokens[index]):
            last_terminal_index = index
    if last_terminal_index is None:
        return start_index, end_index
    return start_index, last_terminal_index


def _estimate_target_word_budget(
    *,
    available_overlap_ms: int,
    max_interjection_words: int,
) -> int:
    """Estimate a practical word budget from the available overlap window."""
    estimated = max(1, available_overlap_ms // 320)
    return max(1, min(max_interjection_words, int(estimated)))


def _deterministically_shorten_interjection_text(text: str) -> str:
    """Return a shorter fallback candidate when the repair LLM cannot help."""
    tokens = _split_tokens(text)
    if not tokens:
        return "Mm."
    if len(tokens) == 1:
        return "Mm."
    shortened = _sanitize_echo_agreement_text(" ".join(tokens[:-1]).strip())
    return shortened or "Mm."


def _split_tokens(text: str) -> list[str]:
    """Return stable whitespace-tokenized text used by the LLM contract."""
    return [token for token in re.split(r"\s+", text.strip()) if token]


def _normalize_token(token: str) -> str:
    """Normalize token text for tolerant equality checks."""
    normalized = re.sub(r"^[^\w]+|[^\w]+$", "", token, flags=re.UNICODE)
    return normalized.casefold()


def _token_ends_with_terminal_punctuation(token: str) -> bool:
    """Return whether a token ends with sentence-final punctuation."""
    return bool(re.search(r"[.!?](?:[\"')\]]*)$", token))


def _normalized_text_tokens(text: str) -> list[str]:
    """Return normalized tokens with punctuation stripped for fuzzy phrase matching."""
    return [
        normalized
        for token in _split_tokens(text)
        if (normalized := _normalize_token(token))
    ]


def _find_best_matching_phrase_span(
    *,
    aligned_words: Sequence[WordTimestamp],
    target_text: str,
    min_similarity: float,
    max_window_delta: int,
    preferred_center_ms: int | None = None,
) -> _MatchedPhraseSpan | None:
    """Return the best-matching sequential word span for a short echo phrase."""
    target_tokens = _normalized_text_tokens(target_text)
    if not target_tokens:
        return None

    normalized_words = [
        (_normalize_token(word.word), word)
        for word in aligned_words
        if _normalize_token(word.word)
    ]
    if not normalized_words:
        return None

    target_phrase = " ".join(target_tokens)
    min_window_size = max(1, len(target_tokens) - max_window_delta)
    max_window_size = min(
        len(normalized_words),
        len(target_tokens) + max_window_delta,
    )
    best_span: _MatchedPhraseSpan | None = None
    best_distance_ms: int | None = None
    for window_size in range(min_window_size, max_window_size + 1):
        for start_index in range(0, len(normalized_words) - window_size + 1):
            window = normalized_words[start_index : start_index + window_size]
            window_tokens = [item[0] for item in window]
            window_words = tuple(item[1] for item in window)
            window_phrase = " ".join(window_tokens)
            phrase_similarity = SequenceMatcher(
                None,
                target_phrase,
                window_phrase,
            ).ratio()
            boundary_similarity = (
                SequenceMatcher(None, target_tokens[0], window_tokens[0]).ratio()
                + SequenceMatcher(None, target_tokens[-1], window_tokens[-1]).ratio()
            ) / 2.0
            score = (phrase_similarity * 0.7) + (boundary_similarity * 0.3)
            if score < min_similarity:
                continue
            candidate = _MatchedPhraseSpan(
                start_word_index=start_index,
                end_word_index=start_index + window_size - 1,
                start_time_ms=window_words[0].start_time_ms,
                end_time_ms=window_words[-1].end_time_ms,
                similarity=score,
                matched_words=window_words,
            )
            candidate_center_ms = int(
                round((candidate.start_time_ms + candidate.end_time_ms) / 2.0)
            )
            candidate_distance_ms = (
                abs(candidate_center_ms - preferred_center_ms)
                if preferred_center_ms is not None
                else None
            )
            if best_span is None:
                best_span = candidate
                best_distance_ms = candidate_distance_ms
                continue
            if candidate.similarity > best_span.similarity:
                best_span = candidate
                best_distance_ms = candidate_distance_ms
                continue
            if (
                preferred_center_ms is not None
                and abs(candidate.similarity - best_span.similarity) <= 0.01
                and candidate_distance_ms is not None
                and (
                    best_distance_ms is None
                    or candidate_distance_ms < best_distance_ms
                )
            ):
                best_span = candidate
                best_distance_ms = candidate_distance_ms
    return best_span


def _build_word_level_echo_retiming_plan(
    *,
    host_phrase_span: _MatchedPhraseSpan,
    overlay_phrase_span: _MatchedPhraseSpan,
    target_host_fill_ratio: float,
    min_overflow_ratio: float,
    max_tempo: float,
    min_segment_ms: int,
) -> tuple[_WordLevelRetimingStep, ...]:
    """Build a per-word retiming plan for one short echo phrase."""
    host_words = tuple(host_phrase_span.matched_words)
    overlay_words = tuple(overlay_phrase_span.matched_words)
    if len(host_words) < 2 or len(host_words) != len(overlay_words):
        return ()

    steps: list[_WordLevelRetimingStep] = []
    speedup_applied = False
    for index, overlay_word in enumerate(overlay_words):
        source_start_ms = overlay_word.start_time_ms
        if index + 1 < len(overlay_words):
            source_end_ms = overlay_words[index + 1].start_time_ms
            host_segment_duration_ms = (
                host_words[index + 1].start_time_ms - host_words[index].start_time_ms
            )
        else:
            source_end_ms = overlay_phrase_span.end_time_ms
            host_segment_duration_ms = (
                host_phrase_span.end_time_ms - host_words[index].start_time_ms
            )
        source_duration_ms = max(0, source_end_ms - source_start_ms)
        target_duration_ms = max(
            1,
            int(round(host_segment_duration_ms * target_host_fill_ratio)),
        )
        if source_duration_ms < min_segment_ms or target_duration_ms < min_segment_ms:
            return ()
        overflow_ratio = source_duration_ms / float(target_duration_ms)
        applied_tempo_ratio = 1.0
        if overflow_ratio >= min_overflow_ratio:
            applied_tempo_ratio = min(max_tempo, max(1.0, overflow_ratio))
            speedup_applied = speedup_applied or applied_tempo_ratio > 1.0
        adjusted_duration_ms = max(
            1,
            int(round(source_duration_ms / applied_tempo_ratio)),
        )
        steps.append(
            _WordLevelRetimingStep(
                word=overlay_word.word,
                source_start_ms=source_start_ms,
                source_end_ms=source_end_ms,
                source_duration_ms=source_duration_ms,
                target_duration_ms=target_duration_ms,
                applied_tempo_ratio=applied_tempo_ratio,
                adjusted_duration_ms=adjusted_duration_ms,
            )
        )
    if not speedup_applied:
        return ()
    return tuple(steps)


def _align_turn_tokens(
    *,
    audio_path: Path,
    text: str,
    language: str,
    aligner: CtcForcedAlignerGateway | None,
    batch_size: int,
) -> TurnAlignmentResult:
    """Align host-turn tokens to the synthesized stage-3 audio artifact."""
    tokens = _split_tokens(text)
    if not tokens:
        return TurnAlignmentResult(tokens=(), source="uniform_fallback")

    duration_ms = probe_audio_duration_ms(audio_path)
    if duration_ms is None:
        raise NonRetryableAudioStageError(
            f"Unable to determine duration for voice-clone artifact {audio_path}."
        )

    if aligner is None:
        return TurnAlignmentResult(
            tokens=tuple(_fallback_aligned_tokens(tokens=tokens, duration_ms=duration_ms)),
            source="uniform_fallback",
        )

    try:
        waveform = _decode_audio_waveform(audio_path)
        aligned_words = aligner.align_words(
            audio_waveform=waveform,
            transcript_text=text,
            language=language,
            device="cpu",
            batch_size=batch_size,
        )
    except Exception:
        return TurnAlignmentResult(
            tokens=tuple(_fallback_aligned_tokens(tokens=tokens, duration_ms=duration_ms)),
            source="uniform_fallback",
        )

    if len(aligned_words) != len(tokens):
        return TurnAlignmentResult(
            tokens=tuple(_fallback_aligned_tokens(tokens=tokens, duration_ms=duration_ms)),
            source="uniform_fallback",
        )
    if _aligned_words_exceed_audio_duration(
        aligned_words=aligned_words,
        duration_ms=duration_ms,
    ):
        return TurnAlignmentResult(
            tokens=tuple(_fallback_aligned_tokens(tokens=tokens, duration_ms=duration_ms)),
            source="uniform_fallback",
        )

    matches = sum(
        1
        for token, aligned_word in zip(tokens, aligned_words, strict=True)
        if _normalize_token(token) == _normalize_token(aligned_word.word)
    )
    if matches < max(1, len(tokens) // 2):
        return TurnAlignmentResult(
            tokens=tuple(_fallback_aligned_tokens(tokens=tokens, duration_ms=duration_ms)),
            source="uniform_fallback",
        )
    return TurnAlignmentResult(
        tokens=tuple(
            AlignedToken(
                token=token,
                start_time_ms=aligned_word.start_time_ms,
                end_time_ms=aligned_word.end_time_ms,
            )
            for token, aligned_word in zip(tokens, aligned_words, strict=True)
        ),
        source="forced_alignment",
    )


def _refine_anchor_window_alignment(
    *,
    audio_path: Path,
    full_text: str,
    anchor_start_token_index: int,
    anchor_end_token_index: int,
    language: str,
    aligner: CtcForcedAlignerGateway | None,
    batch_size: int,
) -> tuple[AlignedToken, ...]:
    """Retry alignment on a small anchor-centered transcript window when full-turn alignment falls back."""
    if aligner is None:
        return ()

    full_tokens = _split_tokens(full_text)
    if not full_tokens:
        return ()
    window_start_index = max(0, anchor_start_token_index - 1)
    window_end_index = min(len(full_tokens) - 1, anchor_end_token_index + 1)
    if window_end_index < window_start_index:
        return ()

    window_tokens = full_tokens[window_start_index : window_end_index + 1]
    if not window_tokens:
        return ()

    duration_ms = probe_audio_duration_ms(audio_path)
    if duration_ms is None:
        raise NonRetryableAudioStageError(
            f"Unable to determine duration for voice-clone artifact {audio_path}."
        )

    try:
        waveform = _decode_audio_waveform(audio_path)
        aligned_words = aligner.align_words(
            audio_waveform=waveform,
            transcript_text=" ".join(window_tokens),
            language=language,
            device="cpu",
            batch_size=batch_size,
        )
    except Exception:
        return ()

    if len(aligned_words) != len(window_tokens):
        return ()
    if _aligned_words_exceed_audio_duration(
        aligned_words=aligned_words,
        duration_ms=duration_ms,
    ):
        return ()
    matches = sum(
        1
        for token, aligned_word in zip(window_tokens, aligned_words, strict=True)
        if _normalize_token(token) == _normalize_token(aligned_word.word)
    )
    if matches < max(1, len(window_tokens) // 2):
        return ()

    local_anchor_start = anchor_start_token_index - window_start_index
    local_anchor_end = anchor_end_token_index - window_start_index
    if local_anchor_start < 0 or local_anchor_end >= len(aligned_words):
        return ()

    return tuple(
        AlignedToken(
            token=window_tokens[index],
            start_time_ms=aligned_words[index].start_time_ms,
            end_time_ms=aligned_words[index].end_time_ms,
        )
        for index in range(local_anchor_start, local_anchor_end + 1)
    )


def _decode_audio_waveform(audio_path: Path) -> Any:
    """Decode one WAV artifact into a mono float32 waveform for forced alignment."""
    try:
        import numpy as np
    except ImportError as exc:
        raise NonRetryableAudioStageError(
            "NumPy is required to decode voice-clone artifacts for forced alignment."
        ) from exc
    ensure_command_available("ffmpeg")
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(_FORCED_ALIGNMENT_SAMPLE_RATE_HZ),
        "-",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        raise NonRetryableAudioStageError(
            "Failed to decode voice-clone artifact for forced alignment. "
            f"Command: {' '.join(command)}. "
            f"Stderr: {(exc.stderr or b'').decode(errors='ignore').strip()[:500]}"
        ) from exc
    waveform = np.frombuffer(completed.stdout, dtype=np.float32)
    if waveform.size == 0:
        raise NonRetryableAudioStageError(
            f"Decoded waveform is empty for voice-clone artifact {audio_path}."
        )
    return waveform


def _aligned_words_exceed_audio_duration(
    *,
    aligned_words: Sequence[Any],
    duration_ms: int,
) -> bool:
    """Return whether any aligned token extends beyond the probed audio duration."""
    max_end_time_ms = max(
        (
            max(
                int(getattr(aligned_word, "start_time_ms", 0) or 0),
                int(getattr(aligned_word, "end_time_ms", 0) or 0),
            )
            for aligned_word in aligned_words
        ),
        default=0,
    )
    return max_end_time_ms > duration_ms + _ALIGNMENT_DURATION_TOLERANCE_MS


def _fallback_aligned_tokens(
    *,
    tokens: Sequence[str],
    duration_ms: int,
) -> list[AlignedToken]:
    """Build deterministic token timings when forced alignment is unavailable."""
    if not tokens:
        return []
    step_ms = max(1, duration_ms // len(tokens))
    cursor_ms = 0
    aligned: list[AlignedToken] = []
    for index, token in enumerate(tokens):
        end_time_ms = duration_ms if index == len(tokens) - 1 else min(
            duration_ms,
            cursor_ms + step_ms,
        )
        aligned.append(
            AlignedToken(
                token=token,
                start_time_ms=cursor_ms,
                end_time_ms=end_time_ms,
            )
        )
        cursor_ms = end_time_ms
    return aligned


def _load_voice_clone_manifest_bundle(
    *,
    summary_turns: Sequence[VoiceCloneTurn],
    manifest_path: Path,
) -> _VoiceCloneManifestBundle:
    """Load stage-3 artifacts and resolve their timeline positions."""
    if not manifest_path.exists():
        raise NonRetryableAudioStageError(
            f"Voice-clone manifest does not exist: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise NonRetryableAudioStageError(
            f"Voice-clone manifest is not valid JSON: {manifest_path}"
        ) from exc

    raw_artifacts = payload.get("artifacts", [])
    if not isinstance(raw_artifacts, list):
        raise NonRetryableAudioStageError(
            "Voice-clone manifest must include an 'artifacts' list."
        )
    if len(raw_artifacts) != len(summary_turns):
        raise NonRetryableAudioStageError(
            "Voice-clone manifest artifact count does not match summary turn count."
        )

    raw_speaker_samples_manifest_path = str(
        payload.get("speaker_samples_manifest_path", "")
    ).strip()
    if not raw_speaker_samples_manifest_path:
        raise NonRetryableAudioStageError(
            "Voice-clone manifest must include speaker_samples_manifest_path."
        )
    speaker_samples_manifest_path = _resolve_manifest_path(
        manifest_path=manifest_path,
        path_value=raw_speaker_samples_manifest_path,
    )

    cumulative_start_ms = 0
    base_turns: list[_VoiceCloneBaseTurn] = []
    for summary_turn, raw_artifact in zip(summary_turns, raw_artifacts, strict=True):
        if not isinstance(raw_artifact, dict):
            raise NonRetryableAudioStageError(
                "Voice-clone manifest contains a non-object artifact entry."
            )
        raw_output_audio_path = str(raw_artifact.get("output_audio_path", "")).strip()
        if not raw_output_audio_path:
            raise NonRetryableAudioStageError(
                "Voice-clone manifest artifact is missing output_audio_path."
            )
        output_audio_path = _resolve_manifest_path(
            manifest_path=manifest_path,
            path_value=raw_output_audio_path,
        )
        if not output_audio_path.exists():
            raise NonRetryableAudioStageError(
                f"Voice-clone artifact does not exist: {output_audio_path}"
            )
        duration_ms = probe_audio_duration_ms(output_audio_path)
        if duration_ms is None:
            raise NonRetryableAudioStageError(
                f"Unable to determine duration for voice-clone artifact {output_audio_path}."
            )
        turn_index = int(raw_artifact.get("turn_index", len(base_turns) + 1))
        base_turns.append(
            _VoiceCloneBaseTurn(
                turn_index=turn_index,
                speaker=summary_turn.speaker,
                text=summary_turn.text,
                output_audio_path=output_audio_path,
                duration_ms=duration_ms,
                global_start_time_ms=cumulative_start_ms,
            )
        )
        cumulative_start_ms += duration_ms

    raw_base_audio_path = str(payload.get("merged_output_audio_path", "")).strip()
    if raw_base_audio_path:
        candidate = _resolve_manifest_path(
            manifest_path=manifest_path,
            path_value=raw_base_audio_path,
        )
        if candidate.exists():
            base_audio_path = candidate
        else:
            base_audio_path = _build_base_audio_from_turns(
                manifest_path=manifest_path,
                artifact_paths=[turn.output_audio_path for turn in base_turns],
            )
    else:
        base_audio_path = _build_base_audio_from_turns(
            manifest_path=manifest_path,
            artifact_paths=[turn.output_audio_path for turn in base_turns],
        )

    return _VoiceCloneManifestBundle(
        base_turns=tuple(base_turns),
        base_audio_path=base_audio_path,
        speaker_samples_manifest_path=speaker_samples_manifest_path,
    )


def _build_base_audio_from_turns(
    *,
    manifest_path: Path,
    artifact_paths: Sequence[Path],
) -> Path:
    """Create a sequential base mix when stage-3 merged audio is unavailable."""
    output_path = manifest_path.parent / "voice_cloned_base.wav"
    merge_audio_artifacts_to_wav(
        artifact_paths=artifact_paths,
        output_path=output_path,
        audio_codec="pcm_s24le",
        timeout_seconds=300,
    )
    return output_path


def _resolve_manifest_path(*, manifest_path: Path, path_value: str) -> Path:
    """Resolve one path listed inside a stage manifest."""
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (manifest_path.parent / candidate).resolve()


def _build_interjection_output_filename(
    *,
    host_turn_index: int,
    speaker: str,
    style: InterjectionStyle,
) -> str:
    """Return a deterministic filename for one generated overlap artifact."""
    sanitized_speaker = re.sub(r"[^A-Za-z0-9_.-]+", "_", speaker.strip()).strip("._")
    if not sanitized_speaker:
        sanitized_speaker = "speaker"
    return f"{host_turn_index:03d}_{style}_{sanitized_speaker}.wav"


def _build_interjection_attempt_output_path(
    *,
    base_output_audio_path: Path,
    attempt_index: int,
) -> Path:
    """Return the temporary synth path for one interjection repair attempt."""
    stem = base_output_audio_path.stem
    suffix = base_output_audio_path.suffix or ".wav"
    return base_output_audio_path.with_name(f"{stem}.attempt_{attempt_index:03d}{suffix}")


def _prepare_interjection_attempt_audio(
    *,
    rendered_path: Path,
    original_duration_ms: int,
    trim_leading_silence_enabled: bool,
    trim_leading_silence_threshold_db: float,
    trim_leading_silence_min_duration_ms: int,
    trim_leading_silence_preserve_ms: int,
    audio_codec: str,
    timeout_seconds: int,
) -> _PreparedOverlayAudio:
    """Return one rendered overlay after optional leading-silence trimming."""
    if (
        not trim_leading_silence_enabled
        or original_duration_ms <= 0
    ):
        return _PreparedOverlayAudio(
            path=rendered_path,
            duration_ms=original_duration_ms,
            leading_silence_trim_ms=0,
        )

    trimmed_duration_ms = _trim_leading_silence_in_place(
        audio_path=rendered_path,
        original_duration_ms=original_duration_ms,
        threshold_db=trim_leading_silence_threshold_db,
        min_duration_ms=trim_leading_silence_min_duration_ms,
        preserve_ms=trim_leading_silence_preserve_ms,
        audio_codec=audio_codec,
        timeout_seconds=timeout_seconds,
    )
    if trimmed_duration_ms is None or trimmed_duration_ms <= 0:
        return _PreparedOverlayAudio(
            path=rendered_path,
            duration_ms=original_duration_ms,
            leading_silence_trim_ms=0,
        )
    leading_silence_trim_ms = max(0, original_duration_ms - trimmed_duration_ms)
    return _PreparedOverlayAudio(
        path=rendered_path,
        duration_ms=trimmed_duration_ms,
        leading_silence_trim_ms=leading_silence_trim_ms,
    )


def _trim_leading_silence_in_place(
    *,
    audio_path: Path,
    original_duration_ms: int,
    threshold_db: float,
    min_duration_ms: int,
    preserve_ms: int,
    audio_codec: str,
    timeout_seconds: int,
) -> int | None:
    """Trim leading silence from one rendered overlay in place when ffmpeg can do so safely."""
    temp_output_path = _build_temp_output_path(audio_path)
    ffmpeg_executable = resolve_command_path("ffmpeg")
    filter_expression = (
        "silenceremove="
        "start_periods=1:"
        f"start_duration={max(0, min_duration_ms) / 1000.0:.3f}:"
        f"start_threshold={threshold_db:.2f}dB:"
        f"start_silence={max(0, preserve_ms) / 1000.0:.3f}"
    )
    command = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(audio_path),
        "-af",
        filter_expression,
        "-c:a",
        audio_codec,
        "-vn",
        "-f",
        "wav",
        str(temp_output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        _remove_temp_output(temp_output_path)
        return None
    if completed.returncode != 0:
        _remove_temp_output(temp_output_path)
        return None
    trimmed_duration_ms = probe_audio_duration_ms(temp_output_path)
    if (
        trimmed_duration_ms is None
        or trimmed_duration_ms <= 0
        or trimmed_duration_ms >= original_duration_ms
    ):
        _remove_temp_output(temp_output_path)
        return None
    try:
        temp_output_path.replace(audio_path)
    except OSError:
        _remove_temp_output(temp_output_path)
        return None
    return trimmed_duration_ms


def _apply_audio_tempo_in_place(
    *,
    audio_path: Path,
    tempo_ratio: float,
    audio_codec: str,
    timeout_seconds: int,
) -> int | None:
    """Speed up one overlay in place so an echo phrase can land on time."""
    if tempo_ratio <= 1.0:
        return probe_audio_duration_ms(audio_path)
    temp_output_path = _build_temp_output_path(audio_path)
    ffmpeg_executable = resolve_command_path("ffmpeg")
    command = [
        ffmpeg_executable,
        "-y",
        "-i",
        str(audio_path),
        "-af",
        _build_atempo_filter_chain(tempo_ratio),
        "-c:a",
        audio_codec,
        "-vn",
        "-f",
        "wav",
        str(temp_output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        _remove_temp_output(temp_output_path)
        return None
    if completed.returncode != 0:
        _remove_temp_output(temp_output_path)
        return None
    adjusted_duration_ms = probe_audio_duration_ms(temp_output_path)
    if adjusted_duration_ms is None or adjusted_duration_ms <= 0:
        _remove_temp_output(temp_output_path)
        return None
    try:
        temp_output_path.replace(audio_path)
    except OSError:
        _remove_temp_output(temp_output_path)
        return None
    return adjusted_duration_ms


def _apply_word_level_echo_retiming_in_place(
    *,
    audio_path: Path,
    steps: Sequence[_WordLevelRetimingStep],
    audio_codec: str,
    timeout_seconds: int,
) -> tuple[int | None, tuple[_WordLevelRetimingStep, ...]]:
    """Retiming one echo clip by processing matched word segments separately."""
    if not steps:
        return None, ()

    ffmpeg_executable = resolve_command_path("ffmpeg")
    temp_dir = audio_path.parent / f"{audio_path.stem}_wordfit"
    temp_dir.mkdir(parents=True, exist_ok=True)
    rendered_steps: list[_WordLevelRetimingStep] = []
    segment_paths: list[Path] = []
    try:
        for index, step in enumerate(steps, start=1):
            segment_path = temp_dir / f"segment_{index:02d}.wav"
            filters = [
                (
                    f"atrim=start={step.source_start_ms / 1000.0:.5f}:"
                    f"end={step.source_end_ms / 1000.0:.5f}"
                ),
                "asetpts=PTS-STARTPTS",
            ]
            if step.applied_tempo_ratio != 1.0:
                filters.append(_build_atempo_filter_chain(step.applied_tempo_ratio))
            command = [
                ffmpeg_executable,
                "-y",
                "-i",
                str(audio_path),
                "-af",
                ",".join(filters),
                "-c:a",
                audio_codec,
                "-vn",
                "-f",
                "wav",
                str(segment_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired):
                return None, ()
            if completed.returncode != 0:
                return None, ()
            adjusted_duration_ms = probe_audio_duration_ms(segment_path)
            if adjusted_duration_ms is None or adjusted_duration_ms <= 0:
                return None, ()
            segment_paths.append(segment_path)
            rendered_steps.append(
                _WordLevelRetimingStep(
                    word=step.word,
                    source_start_ms=step.source_start_ms,
                    source_end_ms=step.source_end_ms,
                    source_duration_ms=step.source_duration_ms,
                    target_duration_ms=step.target_duration_ms,
                    applied_tempo_ratio=step.applied_tempo_ratio,
                    adjusted_duration_ms=adjusted_duration_ms,
                )
            )

        temp_output_path = _build_temp_output_path(audio_path)
        merge_audio_artifacts_to_wav(
            artifact_paths=segment_paths,
            output_path=temp_output_path,
            audio_codec=audio_codec,
            timeout_seconds=timeout_seconds,
        )
        adjusted_duration_ms = probe_audio_duration_ms(temp_output_path)
        if adjusted_duration_ms is None or adjusted_duration_ms <= 0:
            _remove_temp_output(temp_output_path)
            return None, ()
        try:
            temp_output_path.replace(audio_path)
        except OSError:
            _remove_temp_output(temp_output_path)
            return None, ()
        return adjusted_duration_ms, tuple(rendered_steps)
    finally:
        for segment_path in segment_paths:
            _remove_temp_output(segment_path)
        shutil.rmtree(temp_dir, ignore_errors=True)


def _build_atempo_filter_chain(tempo_ratio: float) -> str:
    """Return an ffmpeg `atempo` filter chain for one tempo ratio."""
    if tempo_ratio <= 0.0:
        raise ValueError("tempo_ratio must be > 0.")
    remaining_ratio = tempo_ratio
    filters: list[str] = []
    while remaining_ratio > 2.0 + 1e-6:
        filters.append("atempo=2.00000")
        remaining_ratio /= 2.0
    while remaining_ratio < 0.5 - 1e-6:
        filters.append("atempo=0.50000")
        remaining_ratio /= 0.5
    filters.append(f"atempo={remaining_ratio:.5f}")
    return ",".join(filters)


def _finalize_interjection_attempt_output(
    *,
    rendered_path: Path,
    final_output_path: Path,
) -> Path:
    """Move the accepted attempt output into the canonical interjection artifact path."""
    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    if rendered_path == final_output_path:
        return final_output_path
    if final_output_path.exists():
        final_output_path.unlink()
    rendered_path.replace(final_output_path)
    return final_output_path


def _mix_audio_with_overlays(
    *,
    base_audio_path: Path,
    overlays: Sequence[_SynthesizedOverlay],
    output_path: Path,
    audio_codec: str,
    timeout_seconds: int,
) -> None:
    """Mix delayed overlay tracks onto the sequential base audio output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = _build_temp_output_path(output_path)

    if not overlays:
        try:
            shutil.copyfile(base_audio_path, temp_output_path)
            temp_output_path.replace(output_path)
            return
        except OSError as exc:
            _remove_temp_output(temp_output_path)
            raise NonRetryableAudioStageError(
                f"Failed to persist interjector output to {output_path}."
            ) from exc

    ensure_command_available("ffmpeg")
    filter_parts: list[str] = []
    mix_inputs = ["[0:a]"]
    command: list[str] = ["ffmpeg", "-y", "-i", str(base_audio_path)]
    for index, overlay in enumerate(overlays, start=1):
        command.extend(["-i", str(overlay.path)])
        delay_ms = max(0, overlay.start_time_ms)
        filter_parts.append(f"[{index}:a]adelay={delay_ms}|{delay_ms}[ov{index}]")
        mix_inputs.append(f"[ov{index}]")
    filter_parts.append(
        "".join(mix_inputs)
        + f"amix=inputs={len(mix_inputs)}:duration=longest:normalize=0:dropout_transition=0[outa]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filter_parts),
            "-map",
            "[outa]",
            "-c:a",
            audio_codec,
            "-vn",
            "-f",
            "wav",
            str(temp_output_path),
        ]
    )

    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
        temp_output_path.replace(output_path)
    except subprocess.TimeoutExpired as exc:
        _remove_temp_output(temp_output_path)
        raise NonRetryableAudioStageError(
            "Failed to mix interjection overlays due to timeout. "
            f"Command: {' '.join(command)}."
        ) from exc
    except subprocess.CalledProcessError as exc:
        _remove_temp_output(temp_output_path)
        raise NonRetryableAudioStageError(
            "Failed to mix interjection overlays with ffmpeg. "
            f"Command: {' '.join(command)}. "
            f"Stderr: {(exc.stderr or '').strip()[:500]}"
        ) from exc
    except Exception as exc:
        _remove_temp_output(temp_output_path)
        raise NonRetryableAudioStageError(
            f"Failed to persist interjector output to {output_path}."
        ) from exc


def _build_temp_output_path(output_path: Path) -> Path:
    """Build a temporary output path that preserves the final extension."""
    if output_path.suffix:
        return output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")
    return output_path.with_name(f"{output_path.name}.tmp")


def _remove_temp_output(temp_path: Path) -> None:
    """Remove stale temporary output files on a best-effort basis."""
    try:
        if temp_path.exists():
            temp_path.unlink()
    except OSError:
        pass


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_json_atomic(payload: dict[str, Any], output_path: Path) -> None:
    """Persist JSON payload atomically via a temporary sibling path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    try:
        temp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(output_path)
    except Exception as exc:
        raise ArtifactWriteError(
            f"Failed to write interjector manifest to '{output_path}'."
        ) from exc

