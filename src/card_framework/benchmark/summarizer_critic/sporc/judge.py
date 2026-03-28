"""Backward-compatible SPoRC aliases for the shared conversation-summary judge."""

from card_framework.benchmark.reference_free.conversation_judge import (
    DIMENSION_NAMES,
    ConversationSummaryJudge as SPoRCJudge,
    ConversationSummaryJudgeConfig as SPoRCJudgeConfig,
    ConversationSummaryJudgeResult as SPoRCJudgeResult,
    JudgeLLMProtocol,
)

__all__ = [
    "DIMENSION_NAMES",
    "JudgeLLMProtocol",
    "SPoRCJudge",
    "SPoRCJudgeConfig",
    "SPoRCJudgeResult",
]
