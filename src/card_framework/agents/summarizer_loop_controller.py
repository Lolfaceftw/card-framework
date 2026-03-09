from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from card_framework.shared.events import EventBus


class SummarizerToolRegistry(Protocol):
    """Interface required for summarizer tool-loop execution."""

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible tool schema definitions."""


RunAgentLoopCallable = Callable[
    [list[dict[str, Any]], list[dict[str, Any]], int, dict[str, Any]],
    Awaitable[dict[str, Any] | None],
]

_DRAFT_REVIEW_CHAT_MAX_TOKENS = 256


@dataclass(slots=True)
class SummarizerLoopController:
    """Control summarizer tool-loop setup and execution orchestration."""

    run_agent_loop: RunAgentLoopCallable
    max_tool_turns: int
    is_embedding_enabled: bool
    event_bus: EventBus

    def build_context_data(
        self,
        *,
        tool_registry: SummarizerToolRegistry,
        target_seconds: int,
        duration_tolerance_ratio: float,
        loop_guardrails: dict[str, Any],
    ) -> dict[str, Any]:
        """Build per-run loop state consumed by BaseA2AExecutor and tool dispatch."""
        return {
            "tool_registry": tool_registry,
            "target_seconds": target_seconds,
            "duration_tolerance_ratio": duration_tolerance_ratio,
            "max_tool_calls_per_turn": 1,
            "signature_dedupe_window_turns": 1,
            "replay_dedupe_tools": {
                "add_speaker_message",
                "edit_message",
                "remove_message",
                "finalize_draft",
            },
            "enable_extended_text_tool_parser": loop_guardrails[
                "enable_extended_text_tool_parser"
            ],
            "enable_stall_guidance": loop_guardrails["enable_stall_guidance"],
            "enable_noop_edit_detection": loop_guardrails["enable_noop_edit_detection"],
            "stall_guidance_threshold_turns": loop_guardrails[
                "stall_guidance_threshold_turns"
            ],
            "stall_guidance_cooldown_turns": loop_guardrails[
                "stall_guidance_cooldown_turns"
            ],
            "stagnation_turns": 0,
            "last_stall_guidance_turn": -10_000,
            "loop_turn_index": 0,
            "last_mutation_signature": None,
            "last_total_word_count": None,
            "last_total_estimated_seconds": None,
            "recent_line_edit_fingerprints": [],
            "chat_max_tokens": None,
            "default_chat_max_tokens": None,
            "draft_ready_for_review": False,
            "break_on_no_tool_call_when_draft_ready": True,
            "draft_review_chat_max_tokens": _DRAFT_REVIEW_CHAT_MAX_TOKENS,
            "loop_guardrail_provider": loop_guardrails["provider_name"],
            "loop_guardrail_model": loop_guardrails["model_id"],
            "enable_staged_discovery": self.is_embedding_enabled
            and loop_guardrails["enable_staged_discovery"],
            "required_discovery_queries": loop_guardrails["required_discovery_queries"],
            "max_discovery_queries": loop_guardrails["max_discovery_queries"],
            "require_unique_discovery_queries": loop_guardrails[
                "require_unique_discovery_queries"
            ],
            "discovery_queries_completed": 0,
            "discovery_query_history": [],
        }

    async def run(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_registry: SummarizerToolRegistry,
        target_seconds: int,
        duration_tolerance_ratio: float,
        loop_guardrails: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the summarizer loop and return final loop context data."""
        context_data = self.build_context_data(
            tool_registry=tool_registry,
            target_seconds=target_seconds,
            duration_tolerance_ratio=duration_tolerance_ratio,
            loop_guardrails=loop_guardrails,
        )
        if context_data["enable_staged_discovery"]:
            max_discovery_queries = int(context_data["max_discovery_queries"])
            max_info = (
                f", max {max_discovery_queries} total query_transcript calls"
                if max_discovery_queries > 0
                else ""
            )
            self.event_bus.publish(
                "system_message",
                (
                    "Staged discovery enabled: require "
                    f"{context_data['required_discovery_queries']} successful "
                    f"query_transcript calls before mutation/finalize{max_info}."
                ),
            )

        await self.run_agent_loop(
            messages,
            tool_registry.get_tool_schemas(),
            self.max_tool_turns,
            context_data,
        )
        return context_data

