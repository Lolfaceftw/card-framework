"""Reusable multi-speaker summary judge helpers."""

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
DIMENSION_ITEMS = {
    "Factualness": ("F1", "F2", "F3"),
    "Naturalness": ("N1", "N2", "N3"),
    "Speaker Grammar Similarity": ("S1", "S2", "S3"),
}
ITEM_NAMES = tuple(
    item_name
    for dimension_name in DIMENSION_NAMES
    for item_name in DIMENSION_ITEMS[dimension_name]
)
_SYSTEM_TEMPLATE_KEY = "conversation_summary_judge_system"
_USER_TEMPLATE_KEY = "conversation_summary_judge_user"


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
class ConversationSummaryJudgeConfig:
    """Configuration for reusable conversation-summary judge scoring."""

    max_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class ConversationSummaryJudgeResult:
    """Outcome of one reusable conversation-summary judge evaluation."""

    status: str
    scores: dict[str, float] | None
    item_scores: dict[str, float] | None = None
    confidence: float | None = None
    rationale: str = ""
    error_message: str | None = None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from a raw judge response."""
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


def _normalize_unit_score(value: Any) -> float | None:
    """Parse one normalized 0-to-1 judge score without silently clamping."""
    numeric = normalize_numeric(value)
    if numeric is None or numeric < 0.0 or numeric > 1.0:
        return None
    return clamp_score(float(numeric))


class ConversationSummaryJudge:
    """Run reusable G-Eval-style scoring for multi-speaker summaries."""

    def __init__(
        self,
        *,
        judge_llm: JudgeLLMProtocol,
        config: ConversationSummaryJudgeConfig | None = None,
    ) -> None:
        self._judge_llm = judge_llm
        self._config = config or ConversationSummaryJudgeConfig()

    def _build_prompts(
        self,
        *,
        source_text: str,
        summary_xml: str,
        transcript_per_speaker_messages: dict[str, list[str]],
        summary_per_speaker_messages: dict[str, list[str]],
        speaker_alias_map: dict[str, str],
        speaker_notes: list[str],
        target_duration_label: str = "",
        audio_context_note: str = "",
        scoring_profile_note: str = "",
    ) -> tuple[str, str]:
        """Render the shared conversation-summary judge prompt templates."""
        from card_framework.shared.prompt_manager import PromptManager

        prompt_context = {
            "dimension_names": list(DIMENSION_NAMES),
            "source_text": source_text,
            "summary_xml": summary_xml,
            "transcript_per_speaker_messages_json": json.dumps(
                transcript_per_speaker_messages,
                indent=2,
                sort_keys=True,
            ),
            "summary_per_speaker_messages_json": json.dumps(
                summary_per_speaker_messages,
                indent=2,
                sort_keys=True,
            ),
            "speaker_alias_map_json": json.dumps(
                speaker_alias_map,
                indent=2,
                sort_keys=True,
            ),
            "speaker_notes_json": json.dumps(speaker_notes, indent=2),
            "target_duration_label": str(target_duration_label).strip(),
            "audio_context_note": str(audio_context_note).strip(),
            "scoring_profile_note": str(scoring_profile_note).strip(),
        }
        system_prompt = PromptManager.get_prompt(
            _SYSTEM_TEMPLATE_KEY,
            dimension_names=prompt_context["dimension_names"],
        )
        user_prompt = PromptManager.get_prompt(
            _USER_TEMPLATE_KEY,
            **prompt_context,
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
        target_duration_label: str = "",
        audio_context_note: str = "",
        scoring_profile_note: str = "",
    ) -> ConversationSummaryJudgeResult:
        """Evaluate one summary using the configured judge model."""
        if not source_text.strip() or not summary_xml.strip():
            return ConversationSummaryJudgeResult(
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
            target_duration_label=target_duration_label,
            audio_context_note=audio_context_note,
            scoring_profile_note=scoring_profile_note,
        )
        response = self._judge_llm.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self._config.max_tokens,
        )
        payload = _extract_json_object(response)
        if payload is None:
            return ConversationSummaryJudgeResult(
                status="error",
                scores=None,
                error_message="Judge response did not contain valid JSON.",
            )
        item_scores: dict[str, float] = {}
        raw_item_scores = payload.get("item_scores", {})
        if isinstance(raw_item_scores, dict):
            for item_name in ITEM_NAMES:
                numeric = _normalize_unit_score(raw_item_scores.get(item_name))
                if numeric is not None:
                    item_scores[item_name] = numeric

        scores: dict[str, float] = {}
        for dimension_name in DIMENSION_NAMES:
            if all(item_name in item_scores for item_name in DIMENSION_ITEMS[dimension_name]):
                numeric = sum(
                    item_scores[item_name] for item_name in DIMENSION_ITEMS[dimension_name]
                ) / len(DIMENSION_ITEMS[dimension_name])
                numeric = clamp_score(numeric)
            else:
                numeric = _normalize_unit_score(payload.get(dimension_name))
            if numeric is None:
                return ConversationSummaryJudgeResult(
                    status="error",
                    scores=None,
                    error_message=(
                        "Judge response was missing a valid numeric score for "
                        f"'{dimension_name}'."
                    ),
                )
            scores[dimension_name] = numeric
        confidence = normalize_numeric(payload.get("confidence"))
        return ConversationSummaryJudgeResult(
            status="ok",
            scores=scores,
            item_scores=item_scores or None,
            confidence=clamp_score(float(confidence)) if confidence is not None else None,
            rationale=str(payload.get("rationale", "")).strip(),
        )


__all__ = [
    "ConversationSummaryJudge",
    "ConversationSummaryJudgeConfig",
    "ConversationSummaryJudgeResult",
    "DIMENSION_ITEMS",
    "DIMENSION_NAMES",
    "ITEM_NAMES",
    "JudgeLLMProtocol",
]
