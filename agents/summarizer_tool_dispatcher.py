"""Tool-dispatch helpers for the duration-aware summarizer loop."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from agents.tool_call_utils import build_tool_signature
from events import EventBus

_DEFAULT_DRAFT_REVIEW_CHAT_MAX_TOKENS = 256
_DRAFT_REVIEW_PROMPT = (
    "[DRAFT_REVIEW]\n"
    "The draft is already within budget and has been auto-saved. Your next response "
    "must contain exactly one tool call: call `finalize_draft()` if the draft is ready, "
    "otherwise call one mutating tool to improve it. Do not narrate your review in "
    "plain text."
)


def coerce_positive_int(raw_value: Any, *, default: int, minimum: int = 0) -> int:
    """Parse a positive integer with fallback for malformed runtime values."""
    if isinstance(raw_value, bool):
        return default
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= minimum else default


def normalize_query_text(raw_query: Any) -> str:
    """Normalize retrieval queries for staged-discovery dedupe/accounting."""
    if raw_query is None:
        return ""
    return " ".join(str(raw_query).strip().lower().split())


def build_mutation_fingerprint(name: str, arguments: dict[str, Any]) -> str:
    """Build a lightweight fingerprint used for oscillation detection."""
    if name == "edit_message":
        line = arguments.get("line", "unknown")
        content = str(arguments.get("new_content", ""))
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        emo_preset = str(arguments.get("emo_preset", ""))
        return f"edit:{line}:{emo_preset}:{digest}"
    if name == "remove_message":
        line = arguments.get("line", "unknown")
        return f"remove:{line}"
    if name == "add_speaker_message":
        speaker_id = str(arguments.get("speaker_id", "UNKNOWN"))
        content = str(arguments.get("content", ""))
        emo_preset = str(arguments.get("emo_preset", ""))
        digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
        return f"add:{speaker_id}:{emo_preset}:{digest}"
    return name


def is_oscillating(fingerprints: list[str]) -> bool:
    """Detect short ABAB mutation oscillation patterns."""
    if len(fingerprints) < 4:
        return False
    a, b, c, d = fingerprints[-4:]
    return a == c and b == d and a != b


def build_stall_guidance(
    *,
    target_seconds: int,
    duration_tolerance_ratio: float,
    total_seconds: float | None,
    stagnation_turns: int,
) -> str:
    """Build targeted guidance after repeated no-progress turns."""
    tolerance_seconds = target_seconds * duration_tolerance_ratio
    min_seconds = target_seconds - tolerance_seconds
    max_seconds = target_seconds + tolerance_seconds
    base = (
        "You are repeating edits without meaningful progress "
        f"for {stagnation_turns} turns. "
    )
    if total_seconds is None:
        return (
            base
            + "Make one materially different edit and wait for estimate_duration before the next tool call."
        )

    if total_seconds > max_seconds:
        return (
            base
            + f"Current estimate is {total_seconds:.2f}s, above the {max_seconds:.2f}s max. "
            + "Remove a line with remove_message(line) or materially shorten one line with "
            + "edit_message(line, shorter_content). Do not re-issue unchanged edits."
        )
    if total_seconds < min_seconds:
        return (
            base
            + f"Current estimate is {total_seconds:.2f}s, below the {min_seconds:.2f}s min. "
            + "Add a new line with add_speaker_message(...) or materially expand one line with "
            + "edit_message(line, expanded_content). Do not re-issue unchanged edits."
        )
    return (
        base
        + f"Current estimate is {total_seconds:.2f}s, within {min_seconds:.2f}-{max_seconds:.2f}s. "
        + "If no substantive improvements remain, call finalize_draft(). "
        + "Otherwise make one materially different edit only."
    )


@dataclass(slots=True)
class SummarizerToolDispatcher:
    """Dispatch summarizer tool calls and append tool results to chat history."""

    agent_name: str
    event_bus: EventBus

    async def process_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        context_data: dict[str, Any],
    ) -> tuple[bool, dict | None]:
        """Execute summarizer tool calls for one assistant turn."""
        tool_registry = context_data["tool_registry"]
        target_seconds = int(context_data["target_seconds"])
        duration_tolerance_ratio = float(context_data["duration_tolerance_ratio"])
        finalized = False
        mutating_tools = {"add_speaker_message", "edit_message", "remove_message"}
        mutating_executed_this_turn = False

        context_data["loop_turn_index"] = (
            int(context_data.get("loop_turn_index", 0)) + 1
        )
        loop_turn_index = int(context_data["loop_turn_index"])

        enable_stall_guidance = bool(context_data.get("enable_stall_guidance", False))
        enable_noop_edit_detection = bool(
            context_data.get("enable_noop_edit_detection", False)
        )
        stall_threshold = coerce_positive_int(
            context_data.get("stall_guidance_threshold_turns"), default=3, minimum=1
        )
        guidance_cooldown_turns = coerce_positive_int(
            context_data.get("stall_guidance_cooldown_turns"), default=2, minimum=0
        )

        recent_fingerprints_raw = context_data.get("recent_line_edit_fingerprints", [])
        recent_fingerprints = (
            list(recent_fingerprints_raw)
            if isinstance(recent_fingerprints_raw, list)
            else []
        )

        turn_mutation_attempted = False
        turn_progressed = False
        stagnation_reason: str | None = None
        turn_total_seconds: float | None = None
        turn_signature: str | None = None
        staged_discovery_enabled = bool(
            context_data.get("enable_staged_discovery", False)
        )
        required_discovery_queries = coerce_positive_int(
            context_data.get("required_discovery_queries"),
            default=0,
            minimum=0,
        )
        max_discovery_queries = coerce_positive_int(
            context_data.get("max_discovery_queries"),
            default=0,
            minimum=0,
        )
        require_unique_discovery_queries = bool(
            context_data.get("require_unique_discovery_queries", False)
        )
        discovery_queries_completed = coerce_positive_int(
            context_data.get("discovery_queries_completed"),
            default=0,
            minimum=0,
        )
        raw_discovery_history = context_data.get("discovery_query_history", [])
        discovery_query_history = (
            list(raw_discovery_history)
            if isinstance(raw_discovery_history, list)
            else []
        )
        discovery_query_set = {
            entry
            for entry in discovery_query_history
            if isinstance(entry, str) and entry
        }
        staged_blocked_tools = mutating_tools | {"finalize_draft"}
        registry_mutation_tools = {
            "add_speaker_message",
            "edit_message",
            "remove_message",
        }

        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]

            if (
                staged_discovery_enabled
                and name in staged_blocked_tools
                and discovery_queries_completed < required_discovery_queries
            ):
                remaining_queries = (
                    required_discovery_queries - discovery_queries_completed
                )
                skip_result = {
                    "status": "skipped",
                    "reason": "discovery_phase_incomplete",
                    "required_discovery_queries": required_discovery_queries,
                    "completed_discovery_queries": discovery_queries_completed,
                    "remaining_discovery_queries": remaining_queries,
                    "message": (
                        "Complete staged discovery first. Call query_transcript with "
                        f"{remaining_queries} more focused query"
                        f"{'' if remaining_queries == 1 else 'ies'} before mutating "
                        "the draft or finalizing."
                    ),
                }
                self._append_tool_result(
                    messages=messages,
                    tool_call_id=tc.get("id", "unknown"),
                    name=name,
                    result=skip_result,
                )
                continue

            normalized_query = ""
            if staged_discovery_enabled and name == "query_transcript":
                normalized_query = normalize_query_text(args.get("query", ""))
                if not normalized_query:
                    skip_result = {
                        "status": "skipped",
                        "reason": "empty_query",
                        "message": (
                            "query_transcript requires a non-empty query string during "
                            "staged discovery."
                        ),
                    }
                    self._append_tool_result(
                        messages=messages,
                        tool_call_id=tc.get("id", "unknown"),
                        name=name,
                        result=skip_result,
                    )
                    continue

                if (
                    max_discovery_queries > 0
                    and discovery_queries_completed >= max_discovery_queries
                ):
                    skip_result = {
                        "status": "skipped",
                        "reason": "discovery_query_budget_exhausted",
                        "max_discovery_queries": max_discovery_queries,
                        "completed_discovery_queries": discovery_queries_completed,
                        "message": (
                            "Discovery query budget has been exhausted. Continue with "
                            "drafting or refinement."
                        ),
                    }
                    self._append_tool_result(
                        messages=messages,
                        tool_call_id=tc.get("id", "unknown"),
                        name=name,
                        result=skip_result,
                    )
                    continue

                if (
                    require_unique_discovery_queries
                    and normalized_query in discovery_query_set
                ):
                    skip_result = {
                        "status": "skipped",
                        "reason": "duplicate_discovery_query",
                        "message": (
                            "Use a different query_transcript string that covers a "
                            "new facet of the transcript."
                        ),
                    }
                    self._append_tool_result(
                        messages=messages,
                        tool_call_id=tc.get("id", "unknown"),
                        name=name,
                        result=skip_result,
                    )
                    continue

            if name in mutating_tools and mutating_executed_this_turn:
                skip_result = {
                    "status": "skipped",
                    "reason": "single_mutating_call_per_turn",
                    "message": (
                        "Only one mutating tool call is executed per assistant turn. "
                        "Issue the next mutation in a new turn after reading estimate_duration."
                    ),
                }
                self._append_tool_result(
                    messages=messages,
                    tool_call_id=tc.get("id", "unknown"),
                    name=name,
                    result=skip_result,
                )
                continue

            if name in mutating_tools:
                mutating_executed_this_turn = True
                turn_mutation_attempted = True
                turn_signature = build_tool_signature(name, args)

            result = await tool_registry.dispatch(name, args)
            if result is None:
                result = {
                    "error": f"Unknown tool: {name}",
                    "error_code": "unknown_tool",
                }

            self._append_tool_result(
                messages=messages,
                tool_call_id=tc.get("id", "unknown"),
                name=name,
                result=result,
            )

            if (
                staged_discovery_enabled
                and name == "query_transcript"
                and "error" not in result
            ):
                if normalized_query:
                    if normalized_query not in discovery_query_set:
                        discovery_query_history.append(normalized_query)
                        discovery_query_set.add(normalized_query)
                    discovery_queries_completed += 1
                    context_data["discovery_queries_completed"] = (
                        discovery_queries_completed
                    )
                    context_data["discovery_query_history"] = discovery_query_history
                    if required_discovery_queries > 0:
                        remaining_queries = max(
                            required_discovery_queries - discovery_queries_completed, 0
                        )
                        if remaining_queries == 0:
                            self.event_bus.publish(
                                "status_message",
                                message=(
                                    "Staged discovery complete. Draft mutations are now allowed."
                                ),
                            )
                        else:
                            self.event_bus.publish(
                                "status_message",
                                message=(
                                    "Staged discovery progress: "
                                    f"{discovery_queries_completed}/"
                                    f"{required_discovery_queries} required "
                                    "query_transcript calls completed."
                                ),
                            )

            if name in registry_mutation_tools and "error" not in result:
                duration_result = await tool_registry.dispatch("estimate_duration", {})
                if duration_result is None:
                    duration_result = {
                        "total_estimated_seconds": 0.0,
                        "budget": {"in_budget": False},
                    }
                auto_duration_id = f"auto_duration_{tc.get('id', 'unknown')}"
                self._append_tool_result(
                    messages=messages,
                    tool_call_id=auto_duration_id,
                    name="estimate_duration",
                    result=duration_result,
                    event_name="estimate_duration (auto)",
                    event_result=(
                        f"Total: {float(duration_result.get('total_estimated_seconds', 0.0)):.3f}s"
                    ),
                )
                turn_total_seconds = float(
                    duration_result.get("total_estimated_seconds", 0.0)
                )
                context_data["last_total_estimated_seconds"] = turn_total_seconds

                count_result = await tool_registry.dispatch("count_words", {})
                if count_result is None:
                    count_result = {"total_word_count": 0}
                auto_count_id = f"auto_count_{tc.get('id', 'unknown')}"
                self._append_tool_result(
                    messages=messages,
                    tool_call_id=auto_count_id,
                    name="count_words",
                    result=count_result,
                    event_name="count_words (auto)",
                    event_result=f"Total: {int(count_result.get('total_word_count', 0))} words",
                )
                context_data["last_total_word_count"] = int(
                    count_result.get("total_word_count", 0)
                )

                budget = duration_result.get("budget", {})
                if bool(budget.get("in_budget", False)):
                    context_data["draft_ready_for_review"] = True
                    review_chat_max_tokens = coerce_positive_int(
                        context_data.get("draft_review_chat_max_tokens"),
                        default=_DEFAULT_DRAFT_REVIEW_CHAT_MAX_TOKENS,
                        minimum=32,
                    )
                    context_data["chat_max_tokens"] = review_chat_max_tokens
                    self.event_bus.publish(
                        "status_message",
                        message=(
                            f"Duration in range: {turn_total_seconds:.2f}s "
                            f"(target {float(budget.get('min_seconds', 0.0)):.2f}-"
                            f"{float(budget.get('max_seconds', 0.0)):.2f}s) - "
                            "waiting for LLM to call finalize_draft"
                        ),
                    )

                    save_result = await tool_registry.dispatch("save_draft", {})
                    if save_result is not None:
                        save_id = f"auto_save_{tc.get('id', 'unknown')}"
                        self._append_tool_result(
                            messages=messages,
                            tool_call_id=save_id,
                            name="save_draft",
                            result=save_result,
                            event_name="save_draft (auto)",
                            event_result=(
                                f"Saved {int(save_result.get('total_messages', 0))} messages"
                            ),
                        )
                    messages.append(
                        {
                            "role": "user",
                            "content": _DRAFT_REVIEW_PROMPT,
                        }
                    )
                else:
                    context_data["draft_ready_for_review"] = False
                    default_chat_max_tokens = context_data.get(
                        "default_chat_max_tokens"
                    )
                    if (
                        isinstance(default_chat_max_tokens, int)
                        and default_chat_max_tokens > 0
                    ):
                        context_data["chat_max_tokens"] = default_chat_max_tokens
                    else:
                        context_data["chat_max_tokens"] = None

            if name in registry_mutation_tools:
                if "error" in result:
                    stagnation_reason = str(
                        result.get("error_code") or result.get("error") or "tool_error"
                    )
                else:
                    turn_progressed = True

                if (
                    name == "edit_message"
                    and enable_noop_edit_detection
                    and result.get("changed") is False
                ):
                    turn_progressed = False
                    stagnation_reason = "noop_edit"

                if turn_signature:
                    last_signature = context_data.get("last_mutation_signature")
                    if (
                        isinstance(last_signature, str)
                        and last_signature == turn_signature
                    ):
                        if not turn_progressed:
                            stagnation_reason = (
                                stagnation_reason or "repeated_signature"
                            )

                fingerprint = build_mutation_fingerprint(name, args)
                recent_fingerprints.append(fingerprint)
                if len(recent_fingerprints) > 6:
                    recent_fingerprints = recent_fingerprints[-6:]
                if is_oscillating(recent_fingerprints):
                    turn_progressed = False
                    stagnation_reason = "oscillation"

            if name == "finalize_draft":
                finalized = True
                draft_xml = str(result.get("draft_xml", "")).strip()
                if draft_xml:
                    self.event_bus.publish(
                        "agent_message",
                        self.agent_name,
                        f"Finalized Draft:\n```xml\n{draft_xml}\n```",
                    )
                self.event_bus.publish(
                    "status_message",
                    message="LLM called finalize_draft - submitting to Critic",
                )

        context_data["recent_line_edit_fingerprints"] = recent_fingerprints
        context_data["discovery_queries_completed"] = discovery_queries_completed
        context_data["discovery_query_history"] = discovery_query_history
        if turn_signature:
            context_data["last_mutation_signature"] = turn_signature

        if turn_mutation_attempted:
            stagnation_turns = int(context_data.get("stagnation_turns", 0))
            if turn_progressed:
                stagnation_turns = 0
            else:
                stagnation_turns += 1
            context_data["stagnation_turns"] = stagnation_turns

            if enable_stall_guidance and stagnation_turns >= stall_threshold:
                last_guidance_turn = int(
                    context_data.get("last_stall_guidance_turn", -10_000)
                )
                if (loop_turn_index - last_guidance_turn) >= (
                    guidance_cooldown_turns + 1
                ):
                    if turn_total_seconds is None:
                        last_total = context_data.get("last_total_estimated_seconds")
                        turn_total_seconds = (
                            float(last_total)
                            if isinstance(last_total, (int, float))
                            else None
                        )
                    guidance = build_stall_guidance(
                        target_seconds=target_seconds,
                        duration_tolerance_ratio=duration_tolerance_ratio,
                        total_seconds=turn_total_seconds,
                        stagnation_turns=stagnation_turns,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[STALL_GUIDANCE]\n{guidance}",
                        }
                    )
                    context_data["last_stall_guidance_turn"] = loop_turn_index
                    self.event_bus.publish(
                        "status_message",
                        message=(
                            "Stall guardrail triggered: repetitive/no-op edits detected. "
                            "Injected corrective guidance."
                        ),
                        model_id=context_data.get("loop_guardrail_model"),
                        provider=context_data.get("loop_guardrail_provider"),
                        stagnation_turns=stagnation_turns,
                        stall_reason=stagnation_reason or "unknown",
                    )

        if finalized:
            return True, None
        return False, None

    def _append_tool_result(
        self,
        *,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        name: str,
        result: dict[str, Any],
        event_name: str | None = None,
        event_result: str | None = None,
    ) -> None:
        """Append a tool result to the message log and event bus."""
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": name,
                "content": json.dumps(result),
            }
        )
        self.event_bus.publish(
            "tool_result",
            tool_name=event_name or name,
            result=event_result or json.dumps(result, indent=2),
        )
