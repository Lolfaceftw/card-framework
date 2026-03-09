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
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Literal, Protocol, TypeAlias

from pydantic import BaseModel, Field

from card_framework.audio_pipeline.errors import ArtifactWriteError, NonRetryableAudioStageError
from card_framework.audio_pipeline.gateways.ctc_forced_aligner_gateway import CtcForcedAlignerGateway
from card_framework.audio_pipeline.runtime import ensure_command_available, probe_audio_duration_ms
from card_framework.audio_pipeline.voice_clone_contracts import VoiceCloneProvider, VoiceCloneTurn
from card_framework.audio_pipeline.voice_clone_orchestrator import (
    load_speaker_sample_references,
    merge_audio_artifacts_to_wav,
    parse_summary_turns,
)
from card_framework.shared.llm_provider import LLMProvider


InterjectionStyle: TypeAlias = Literal["backchannel", "echo_agreement"]


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


@dataclass(slots=True, frozen=True)
class AlignedToken:
    """Represent one summary token aligned to a stage-3 audio artifact."""

    token: str
    start_time_ms: int
    end_time_ms: int


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
    duration_ms: int
    output_audio_path: Path


@dataclass(slots=True, frozen=True)
class InterjectorRunResult:
    """Bundle output paths and metadata for one Stage-4 interjector run."""

    output_dir: Path
    manifest_path: Path
    generated_at_utc: str
    artifacts: tuple[InterjectorArtifact, ...]
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


@dataclass(slots=True)
class LLMInterjectionPlanner:
    """Use an LLM to decide whether summary turns need overlapping interjections."""

    llm: LLMProvider
    max_tokens: int = 900
    max_interjection_words: int = 5
    min_host_progress_ratio: float = 0.35
    max_host_progress_ratio: float = 0.90

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
            max_host_progress_percent=round(self.max_host_progress_ratio * 100),
        )
        user_prompt = PromptManager.get_prompt(
            "interjector_user",
            eligible_turns_block=_render_eligible_turns_block(
                eligible_turns,
                min_host_progress_ratio=self.min_host_progress_ratio,
                max_host_progress_ratio=self.max_host_progress_ratio,
            ),
            max_interjection_words=self.max_interjection_words,
            min_host_progress_percent=round(self.min_host_progress_ratio * 100),
            max_host_progress_percent=round(self.max_host_progress_ratio * 100),
        )
        try:
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
            max_host_progress_ratio=self.max_host_progress_ratio,
        )


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
    echo_alignment_offset_ms: int = 20
    min_host_progress_ratio: float = 0.35
    max_host_progress_ratio: float = 0.90
    min_available_overlap_ms: int = 120
    turn_end_guard_ms: int = 180
    mix_audio_codec: str = "pcm_s24le"
    mix_timeout_seconds: int = 300
    aligner: CtcForcedAlignerGateway | None = None
    alignment_batch_size: int = 8

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
        if not 0.0 <= self.max_host_progress_ratio <= 1.0:
            raise ValueError("max_host_progress_ratio must be within [0.0, 1.0].")
        if self.min_host_progress_ratio > self.max_host_progress_ratio:
            raise ValueError(
                "min_host_progress_ratio must be <= max_host_progress_ratio."
            )
        if self.min_available_overlap_ms <= 0:
            raise ValueError("min_available_overlap_ms must be > 0.")
        if self.turn_end_guard_ms < 0:
            raise ValueError("turn_end_guard_ms must be >= 0.")
        if not self.mix_audio_codec.strip():
            raise ValueError("mix_audio_codec must be non-empty.")
        if self.mix_timeout_seconds <= 0:
            raise ValueError("mix_timeout_seconds must be > 0.")
        if self.alignment_batch_size <= 0:
            raise ValueError("alignment_batch_size must be > 0.")
        if self.aligner is None:
            self.aligner = CtcForcedAlignerGateway()

    def run(
        self,
        *,
        summary_xml: str,
        voice_clone_manifest_path: Path,
        language: str = "en",
    ) -> InterjectorRunResult:
        """Generate overlaps and mix them into the stage-3 merged audio."""
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
        artifacts: list[InterjectorArtifact] = []
        overlays: list[_SynthesizedOverlay] = []
        host_turn_counts: dict[int, int] = {}
        turn_alignment_cache: dict[int, tuple[AlignedToken, ...]] = {}

        for decision in decisions:
            if not decision.should_interject:
                continue

            host_turn = manifest_bundle.base_turns[decision.host_turn_index - 1]
            host_turn_counts.setdefault(host_turn.turn_index, 0)
            if (
                host_turn_counts[host_turn.turn_index]
                >= self.max_interjections_per_host_turn
            ):
                continue
            aligned_tokens = turn_alignment_cache.get(host_turn.turn_index)
            if aligned_tokens is None:
                aligned_tokens = tuple(
                    _align_turn_tokens(
                        audio_path=host_turn.output_audio_path,
                        text=host_turn.text,
                        language=language,
                        aligner=self.aligner,
                        batch_size=self.alignment_batch_size,
                    )
                )
                turn_alignment_cache[host_turn.turn_index] = aligned_tokens
            if not aligned_tokens:
                continue

            anchor_tokens = aligned_tokens[
                decision.anchor_start_token_index : decision.anchor_end_token_index + 1
            ]
            if not anchor_tokens:
                continue

            overlay_start_time_ms = self._compute_overlay_start_time_ms(
                decision=decision,
                host_turn=host_turn,
                anchor_tokens=anchor_tokens,
            )
            if overlay_start_time_ms is None:
                continue

            reference = sample_refs.get(decision.interjection_speaker or "")
            if reference is None:
                continue

            output_audio_path = self.output_dir / _build_interjection_output_filename(
                host_turn_index=host_turn.turn_index,
                speaker=decision.interjection_speaker or reference.speaker,
                style=decision.interjection_style or "backchannel",
            )
            rendered_path = self.provider.synthesize(
                reference_audio_path=reference.path,
                text=decision.interjection_text or "",
                output_audio_path=output_audio_path,
            )
            duration_ms = probe_audio_duration_ms(rendered_path)
            if duration_ms is None:
                raise NonRetryableAudioStageError(
                    f"Unable to determine interjection duration for {rendered_path}."
                )

            artifact = InterjectorArtifact(
                host_turn_index=host_turn.turn_index,
                interjection_style=decision.interjection_style or "backchannel",
                speaker=decision.interjection_speaker or reference.speaker,
                text=decision.interjection_text or "",
                anchor_text=decision.anchor_text or "",
                anchor_start_token_index=decision.anchor_start_token_index or 0,
                anchor_end_token_index=decision.anchor_end_token_index or 0,
                start_time_ms=overlay_start_time_ms,
                duration_ms=duration_ms,
                output_audio_path=rendered_path,
            )
            artifacts.append(artifact)
            host_turn_counts[host_turn.turn_index] += 1
            overlays.append(
                _SynthesizedOverlay(
                    path=rendered_path,
                    start_time_ms=overlay_start_time_ms,
                )
            )

        merged_output_audio_path = self.output_dir / self.merged_output_filename
        _mix_audio_with_overlays(
            base_audio_path=manifest_bundle.base_audio_path,
            overlays=overlays,
            output_path=merged_output_audio_path,
            audio_codec=self.mix_audio_codec,
            timeout_seconds=self.mix_timeout_seconds,
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
                        "duration_ms": artifact.duration_ms,
                        "output_audio_path": str(artifact.output_audio_path),
                    }
                    for artifact in artifacts
                ],
            },
            manifest_path,
        )
        return InterjectorRunResult(
            output_dir=self.output_dir,
            manifest_path=manifest_path,
            generated_at_utc=generated_at_utc,
            artifacts=tuple(artifacts),
            merged_output_audio_path=merged_output_audio_path,
        )

    def _compute_overlay_start_time_ms(
        self,
        *,
        decision: InterjectionDecision,
        host_turn: _VoiceCloneBaseTurn,
        anchor_tokens: Sequence[AlignedToken],
    ) -> int | None:
        """Return global overlay onset for a validated decision or ``None``."""
        if host_turn.duration_ms <= 0:
            return None

        anchor_start_ms = host_turn.global_start_time_ms + anchor_tokens[0].start_time_ms
        anchor_end_ms = host_turn.global_start_time_ms + anchor_tokens[-1].end_time_ms
        if (decision.interjection_style or "backchannel") == "echo_agreement":
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
        if progress_ratio < self.min_host_progress_ratio:
            return None
        if progress_ratio > self.max_host_progress_ratio:
            return None

        available_overlap_ms = (
            host_turn.global_start_time_ms
            + host_turn.duration_ms
            - self.turn_end_guard_ms
            - overlay_start_time_ms
        )
        if available_overlap_ms < self.min_available_overlap_ms:
            return None
        return overlay_start_time_ms


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
    max_host_progress_ratio: float,
) -> str:
    """Render prompt-friendly eligible turn descriptions for the LLM planner."""
    lines: list[str] = []
    for item in eligible_turns:
        host_tokens = _split_tokens(item.host_turn.text)
        anchor_window = _compute_anchor_window_indices(
            host_token_count=len(host_tokens),
            min_host_progress_ratio=min_host_progress_ratio,
            max_host_progress_ratio=max_host_progress_ratio,
        )
        anchor_window_text = (
            (
                "Preferred anchor token window "
                f"(approx {round(min_host_progress_ratio * 100)}%-"
                f"{round(max_host_progress_ratio * 100)}% through host turn): "
                f"{anchor_window[0]}..{anchor_window[1]}. "
                "For backchannel, keep `anchor_end_token_index` inside this window. "
                "For echo_agreement, keep `anchor_start_token_index` inside this "
                "window."
            )
            if anchor_window is not None
            else "Preferred anchor token window: none. Return should_interject=false."
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


def _validate_llm_decisions(
    raw_decisions: Sequence[InterjectionDecision],
    *,
    eligible_turns: Sequence[_EligibleTurn],
    max_interjection_words: int,
    min_host_progress_ratio: float = 0.35,
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
        anchor_progress_ratio = _estimate_anchor_progress_ratio(
            host_token_count=len(host_tokens),
            anchor_start_token_index=start_index,
            anchor_end_token_index=end_index,
            style=style,
        )
        if anchor_progress_ratio < min_host_progress_ratio:
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
        if style == "echo_agreement" and (
            not interjection_text
            or not _texts_share_anchor_tokens(interjection_text, anchor_text)
        ):
            interjection_text = anchor_text
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


def _split_tokens(text: str) -> list[str]:
    """Return stable whitespace-tokenized text used by the LLM contract."""
    return [token for token in re.split(r"\s+", text.strip()) if token]


def _normalize_token(token: str) -> str:
    """Normalize token text for tolerant equality checks."""
    normalized = re.sub(r"^[^\w]+|[^\w]+$", "", token, flags=re.UNICODE)
    return normalized.casefold()


def _align_turn_tokens(
    *,
    audio_path: Path,
    text: str,
    language: str,
    aligner: CtcForcedAlignerGateway | None,
    batch_size: int,
) -> list[AlignedToken]:
    """Align host-turn tokens to the synthesized stage-3 audio artifact."""
    tokens = _split_tokens(text)
    if not tokens:
        return []

    duration_ms = probe_audio_duration_ms(audio_path)
    if duration_ms is None:
        raise NonRetryableAudioStageError(
            f"Unable to determine duration for voice-clone artifact {audio_path}."
        )

    if aligner is None:
        return _fallback_aligned_tokens(tokens=tokens, duration_ms=duration_ms)

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
        return _fallback_aligned_tokens(tokens=tokens, duration_ms=duration_ms)

    if len(aligned_words) != len(tokens):
        return _fallback_aligned_tokens(tokens=tokens, duration_ms=duration_ms)

    matches = sum(
        1
        for token, aligned_word in zip(tokens, aligned_words, strict=True)
        if _normalize_token(token) == _normalize_token(aligned_word.word)
    )
    if matches < max(1, len(tokens) // 2):
        return _fallback_aligned_tokens(tokens=tokens, duration_ms=duration_ms)

    return [
        AlignedToken(
            token=token,
            start_time_ms=aligned_word.start_time_ms,
            end_time_ms=aligned_word.end_time_ms,
        )
        for token, aligned_word in zip(tokens, aligned_words, strict=True)
    ]


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

