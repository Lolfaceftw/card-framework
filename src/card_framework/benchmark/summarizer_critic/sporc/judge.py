"""SPoRC-specific G-Eval-style judge helpers.

Reference:
    [1] Y. Liu, D. Iter, Y. Xu, S. Wang, R. Xu, and C. Zhu,
    "G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment,"
    arXiv:2303.16634, 2023.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Protocol

from card_framework.benchmark.reference_free.contracts import (
    clamp_score,
    normalize_numeric,
)

DIMENSION_NAMES = (
    "Factualness",
    "Naturalness",
    "Speaker Grammar Similarity",
)


class JudgeLLMProtocol(Protocol):
    """Protocol for judge-capable LLM providers."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text output for the provided prompts."""
        raise NotImplementedError


@dataclass(slots=True, frozen=True)
class SPoRCJudgeConfig:
    """Configuration for SPoRC judge scoring."""

    max_tokens: int = 900


@dataclass(slots=True, frozen=True)
class SPoRCJudgeResult:
    """Outcome of one SPoRC judge evaluation."""

    status: str
    scores: dict[str, float] | None
    error_message: str | None = None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    fence_match = re.search(r"```json\s*(\{.*?\})\s*```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    return payload


class SPoRCJudge:
    """Run SPoRC-specific G-Eval-style scoring."""

    def __init__(
        self,
        *,
        judge_llm: JudgeLLMProtocol,
        config: SPoRCJudgeConfig | None = None,
    ) -> None:
        self._judge_llm = judge_llm
        self._config = config or SPoRCJudgeConfig()

    def _build_prompts(
        self,
        *,
        source_text: str,
        summary_xml: str,
        transcript_per_speaker_messages: dict[str, list[str]],
        summary_per_speaker_messages: dict[str, list[str]],
        speaker_alias_map: dict[str, str],
        speaker_notes: list[str],
    ) -> tuple[str, str]:
        # G-Eval-style form filling uses an explicit rubric and fixed JSON output.
        # See [1] in the module docstring.
        system_prompt = (
            "You are a strict podcast summarization evaluator using a G-Eval-style "
            "form-filling rubric. Return JSON only. Score each dimension from 0.0 to "
            "1.0, where 1.0 is best. Use the transcript, summary XML, and per-speaker "
            "message lists to judge content faithfulness, flow, and speaker-specific "
            "style preservation."
        )
        user_prompt = (
            "Evaluate the candidate summary with exactly these dimensions:\n"
            "- Factualness: every claim must be supported by the transcript, major "
            "topics should not be skipped without good compression reason, and avoid "
            "unnecessary information loss.\n"
            "- Naturalness: the summary should read like a coherent spoken podcast "
            "segment, with chronological flow, smooth transitions, and no awkward "
            "phrases or abrupt jumps.\n"
            "- Speaker Grammar Similarity: the summary should preserve each speaker's "
            "noticeable grammar, cadence, filler words, fragments, and personality when "
            "appropriate under compression, while still keeping speaker-specific content "
            "with the right speaker.\n\n"
            "Guide questions:\n"
            "1. Factualness\n"
            "   - Is every factual statement grounded in the transcript?\n"
            "   - Did the summary omit any major topic, segment, or claim that should "
            "remain for the requested duration?\n"
            "   - How much important information was lost beyond what compression "
            "requires?\n"
            "2. Naturalness\n"
            "   - Does the flow feel natural, chronological, and podcast-like?\n"
            "   - Are there abrupt jumps, stitched-together sentences, or awkward "
            "phrases?\n"
            "3. Speaker Grammar Similarity\n"
            "   - Does each speaker still sound like themselves rather than a generic "
            "summarizer voice?\n"
            "   - Are filler words such as 'um', 'uh', and conversational fragments "
            "preserved when they are important to personality and tone?\n"
            "   - Is speaker-specific content still associated with the correct "
            "speaker?\n\n"
            "Scoring anchors:\n"
            "- 1.00: Excellent; essentially no meaningful issue.\n"
            "- 0.75: Strong; minor issues only.\n"
            "- 0.50: Mixed; noticeable weaknesses but still usable.\n"
            "- 0.25: Weak; major problems or substantial loss.\n"
            "- 0.00: Fails badly.\n\n"
            "Output JSON only with this exact schema:\n"
            "{\n"
            '  "Factualness": <float>,\n'
            '  "Naturalness": <float>,\n'
            '  "Speaker Grammar Similarity": <float>,\n'
            '  "confidence": <float>,\n'
            '  "rationale": "short explanation"\n'
            "}\n\n"
            "Source transcript:\n"
            f"{source_text}\n\n"
            "Candidate summary XML:\n"
            f"{summary_xml}\n\n"
            "Transcript per-speaker message lists:\n"
            f"{json.dumps(transcript_per_speaker_messages, indent=2, sort_keys=True)}\n\n"
            "Summary per-speaker message lists:\n"
            f"{json.dumps(summary_per_speaker_messages, indent=2, sort_keys=True)}\n\n"
            "Speaker alias mapping:\n"
            f"{json.dumps(speaker_alias_map, indent=2, sort_keys=True)}\n\n"
            "Speaker notes:\n"
            f"{json.dumps(speaker_notes, indent=2)}"
        )
        return system_prompt, user_prompt

    def evaluate(
        self,
        *,
        source_text: str,
        summary_xml: str,
        transcript_per_speaker_messages: dict[str, list[str]],
        summary_per_speaker_messages: dict[str, list[str]],
        speaker_alias_map: dict[str, str],
        speaker_notes: list[str],
    ) -> SPoRCJudgeResult:
        """Evaluate one SPoRC sample using the configured judge model."""
        if not source_text.strip() or not summary_xml.strip():
            return SPoRCJudgeResult(
                status="error",
                scores=None,
                error_message="Source transcript and summary must both be non-empty.",
            )
        system_prompt, user_prompt = self._build_prompts(
            source_text=source_text,
            summary_xml=summary_xml,
            transcript_per_speaker_messages=transcript_per_speaker_messages,
            summary_per_speaker_messages=summary_per_speaker_messages,
            speaker_alias_map=speaker_alias_map,
            speaker_notes=speaker_notes,
        )
        response = self._judge_llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self._config.max_tokens,
        )
        payload = _extract_json_object(response)
        if payload is None:
            return SPoRCJudgeResult(
                status="error",
                scores=None,
                error_message="Judge response did not contain valid JSON.",
            )
        scores: dict[str, float] = {}
        for dimension_name in DIMENSION_NAMES:
            numeric = normalize_numeric(payload.get(dimension_name))
            if numeric is None:
                return SPoRCJudgeResult(
                    status="error",
                    scores=None,
                    error_message=(
                        "Judge response was missing a valid numeric score for "
                        f"'{dimension_name}'."
                    ),
                )
            scores[dimension_name] = clamp_score(float(numeric))
        return SPoRCJudgeResult(status="ok", scores=scores)
