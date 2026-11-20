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

    max_tokens: int = 900


@dataclass(slots=True, frozen=True)
class ConversationSummaryJudgeResult:
    """Outcome of one reusable conversation-summary judge evaluation."""

    status: str
    scores: dict[str, float] | None
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
        scores: dict[str, float] = {}
        for dimension_name in DIMENSION_NAMES:
            numeric = normalize_numeric(payload.get(dimension_name))
            if numeric is None:
                return ConversationSummaryJudgeResult(
                    status="error",
                    scores=None,
                    error_message=(
                        "Judge response was missing a valid numeric score for "
                        f"'{dimension_name}'."
                    ),
                )
            scores[dimension_name] = clamp_score(float(numeric))
        return ConversationSummaryJudgeResult(status="ok", scores=scores)


__all__ = [
    "ConversationSummaryJudge",
    "ConversationSummaryJudgeConfig",
    "ConversationSummaryJudgeResult",
    "DIMENSION_NAMES",
    "JudgeLLMProtocol",
]
