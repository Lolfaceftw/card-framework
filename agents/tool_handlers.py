"""Tool handler registry and duration-aware summarizer tools."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from agents.client import AgentTaskClient, get_default_agent_client
from agents.message_registry import MessageRegistry
from audio_pipeline.calibration import VoiceCloneCalibration
from summary_xml import DEFAULT_EMO_PRESET, SummaryTurn, serialize_summary_turns


class ToolHandler(ABC):
    """Define the interface used by all summarizer tool handlers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the tool name exposed to the LLM."""

    @property
    @abstractmethod
    def schema(self) -> dict[str, Any]:
        """Return an OpenAI-compatible function schema."""

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute the tool and return a JSON-serializable payload."""


class ToolRegistry:
    """Register and dispatch summarizer tool handlers by name."""

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, handler: ToolHandler) -> None:
        """Register one handler instance."""
        self._handlers[handler.name] = handler

    def get(self, name: str) -> ToolHandler | None:
        """Return one handler by name when registered."""
        return self._handlers.get(name)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return all registered schemas in LLM API format."""
        return [
            {"type": "function", "function": handler.schema}
            for handler in self._handlers.values()
        ]

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one tool call by name."""
        handler = self._handlers.get(name)
        if handler is None:
            return None
        return await handler.execute(arguments)


class BudgetContext:
    """Compute duration-budget statistics from the current registry state."""

    def __init__(
        self,
        registry: MessageRegistry,
        calibration: VoiceCloneCalibration,
        target_seconds: int,
        duration_tolerance_ratio: float,
    ) -> None:
        self._registry = registry
        self._calibration = calibration
        self.target_seconds = target_seconds
        self.duration_tolerance_ratio = duration_tolerance_ratio

    def compute_stats(self) -> dict[str, Any]:
        """Return duration-budget guidance derived from the current registry."""
        counts = self._registry.get_counts()
        duration = self._registry.get_duration_breakdown(self._calibration)
        total_seconds = float(duration["total_estimated_seconds"])
        num_lines = len(self._registry)
        avg_seconds_per_line = round(total_seconds / num_lines, 3) if num_lines > 0 else 0.0
        tolerance_seconds = round(
            self.target_seconds * self.duration_tolerance_ratio,
            3,
        )
        min_seconds = round(self.target_seconds - tolerance_seconds, 3)
        max_seconds = round(self.target_seconds + tolerance_seconds, 3)
        delta_target_seconds = round(total_seconds - self.target_seconds, 3)
        if avg_seconds_per_line > 0:
            estimated_lines_to_change = round(
                abs(delta_target_seconds) / avg_seconds_per_line,
                1,
            )
        else:
            estimated_lines_to_change = 0.0
        if total_seconds > max_seconds:
            recommended_action = "remove"
        elif total_seconds < min_seconds:
            recommended_action = "add"
        else:
            recommended_action = "none"
        return {
            "budget": {
                "target_seconds": self.target_seconds,
                "duration_tolerance_ratio": self.duration_tolerance_ratio,
                "tolerance_seconds": tolerance_seconds,
                "min_seconds": min_seconds,
                "max_seconds": max_seconds,
                "total_estimated_seconds": round(total_seconds, 3),
                "total_words": counts["total_word_count"],
                "delta_to_lower_bound": round(total_seconds - min_seconds, 3),
                "delta_to_upper_bound": round(total_seconds - max_seconds, 3),
                "delta_to_target_seconds": delta_target_seconds,
                "in_budget": min_seconds <= total_seconds <= max_seconds,
                "avg_seconds_per_line": avg_seconds_per_line,
                "estimated_lines_to_change": estimated_lines_to_change,
                "recommended_action": recommended_action,
            }
        }


class AddSpeakerMessageHandler(ToolHandler):
    """Append one speaker message to the registry."""

    def __init__(self, registry: MessageRegistry, budget: BudgetContext) -> None:
        self._registry = registry
        self._budget = budget

    @property
    def name(self) -> str:
        return "add_speaker_message"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "add_speaker_message",
            "description": (
                "Adds a single speaker message to the summary. Call this once per "
                "speaker turn as you build the summary incrementally. After this call, "
                "estimate_duration and count_words will be automatically invoked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "speaker_id": {
                        "type": "string",
                        "description": "The speaker ID (for example SPEAKER_00).",
                    },
                    "content": {
                        "type": "string",
                        "description": "The summarized content for this speaker turn.",
                    },
                    "emo_preset": {
                        "type": "string",
                        "description": "Preset emotion/style label for this speaker turn.",
                    },
                },
                "required": ["speaker_id", "content", "emo_preset"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self._registry.add(
            str(arguments["speaker_id"]),
            str(arguments["content"]),
            str(arguments["emo_preset"]),
        )
        return {**result, **self._budget.compute_stats()}


class EstimateDurationHandler(ToolHandler):
    """Return the calibrated duration estimate breakdown."""

    def __init__(
        self,
        registry: MessageRegistry,
        budget: BudgetContext,
        calibration: VoiceCloneCalibration,
    ) -> None:
        self._registry = registry
        self._budget = budget
        self._calibration = calibration

    @property
    def name(self) -> str:
        return "estimate_duration"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "estimate_duration",
            "description": (
                "Returns the current estimated spoken-duration breakdown using "
                "calibrated per-speaker WPM and the chosen emo preset for each line."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        duration = self._registry.get_duration_breakdown(self._calibration)
        return {**duration, **self._budget.compute_stats()}


class CountWordsHandler(ToolHandler):
    """Return the word-count breakdown while keeping duration budget stats attached."""

    def __init__(self, registry: MessageRegistry, budget: BudgetContext) -> None:
        self._registry = registry
        self._budget = budget

    @property
    def name(self) -> str:
        return "count_words"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "count_words",
            "description": (
                "Returns the current word-count breakdown: total count, per-message "
                "counts with line numbers, and per-speaker totals. No arguments needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        counts = self._registry.get_counts()
        return {**counts, **self._budget.compute_stats()}


class EditMessageHandler(ToolHandler):
    """Replace the content or preset of an existing message."""

    def __init__(self, registry: MessageRegistry, budget: BudgetContext) -> None:
        self._registry = registry
        self._budget = budget

    @property
    def name(self) -> str:
        return "edit_message"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "edit_message",
            "description": (
                "Replaces the content of a message at a given line number. Use this to "
                "trim, expand, or retone a specific speaker turn to stay within budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "line": {
                        "type": "integer",
                        "description": "The 1-indexed line number of the message to edit.",
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The replacement content.",
                    },
                    "emo_preset": {
                        "type": "string",
                        "description": (
                            "Optional replacement emo preset for this line. If omitted, "
                            "the existing preset is preserved."
                        ),
                    },
                },
                "required": ["line", "new_content"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_line = arguments.get("line")
        try:
            line = int(raw_line)
        except (TypeError, ValueError):
            return {
                "error": "Argument 'line' must be an integer line number.",
                "error_code": "invalid_line_argument",
                "line": raw_line,
            }

        new_content = arguments.get("new_content")
        if not isinstance(new_content, str):
            return {
                "error": "Argument 'new_content' must be a string.",
                "error_code": "invalid_new_content_argument",
                "line": line,
            }
        emo_preset = arguments.get("emo_preset")
        if emo_preset is not None and not isinstance(emo_preset, str):
            return {
                "error": "Argument 'emo_preset' must be a string when provided.",
                "error_code": "invalid_emo_preset_argument",
                "line": line,
            }

        result = self._registry.edit(line, new_content, emo_preset=emo_preset)
        if "error" not in result:
            result.update(self._budget.compute_stats())
        return result


class RemoveMessageHandler(ToolHandler):
    """Remove one message by line number."""

    def __init__(self, registry: MessageRegistry, budget: BudgetContext) -> None:
        self._registry = registry
        self._budget = budget

    @property
    def name(self) -> str:
        return "remove_message"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "remove_message",
            "description": (
                "Removes the message at a given line number. Remaining messages are "
                "re-indexed. Use this to drop turns when over budget."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "line": {
                        "type": "integer",
                        "description": "The 1-indexed line number of the message to remove.",
                    },
                },
                "required": ["line"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        raw_line = arguments.get("line")
        try:
            line = int(raw_line)
        except (TypeError, ValueError):
            return {
                "error": "Argument 'line' must be an integer line number.",
                "error_code": "invalid_line_argument",
                "line": raw_line,
            }

        result = self._registry.remove(line)
        if "error" not in result:
            result.update(self._budget.compute_stats())
        return result


class QueryTranscriptHandler(ToolHandler):
    """Query the retrieval agent for transcript segments."""

    def __init__(
        self,
        retrieval_port: int,
        agent_client: AgentTaskClient | None = None,
    ) -> None:
        self._retrieval_port = retrieval_port
        self._agent_client = (
            agent_client if agent_client is not None else get_default_agent_client()
        )

    @property
    def name(self) -> str:
        return "query_transcript"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "query_transcript",
            "description": (
                "Searches the original transcript for segments matching your query. Use "
                "this when you need to recall what a speaker said about a specific "
                "topic, verify facts, or find additional details to include."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A natural-language query describing what you want to find."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of matching segments to return (default 5).",
                    },
                },
                "required": ["query"],
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        from agents.dtos import RetrieveTaskRequest

        query = str(arguments["query"])
        top_k = arguments.get("top_k", 5)
        retrieve_task = RetrieveTaskRequest(action="retrieve", query=query, top_k=top_k)
        raw_response = await self._agent_client.send_task(
            self._retrieval_port, retrieve_task, timeout=30.0
        )
        try:
            return json.loads(raw_response)
        except json.JSONDecodeError:
            return {
                "error": "Failed to parse retrieval response",
                "raw": raw_response[:500],
            }


class SaveDraftHandler(ToolHandler):
    """Snapshot the current registry state as a saved draft."""

    def __init__(self, registry: MessageRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "save_draft"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "save_draft",
            "description": (
                "Saves the current draft from the message registry. Called "
                "automatically when budget is met. Returns the full draft with line "
                "numbers for reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        snapshot = self._registry.snapshot()
        return {
            "status": "draft_saved",
            "messages": snapshot,
            "total_messages": len(snapshot),
            "draft_xml": _snapshot_to_xml(snapshot),
        }


class FinalizeDraftHandler(ToolHandler):
    """Signal that the draft is ready for critic evaluation."""

    def __init__(self, registry: MessageRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "finalize_draft"

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "name": "finalize_draft",
            "description": (
                "Call this only when you have reviewed your draft, confirmed it meets "
                "the duration target, reads naturally, and covers the transcript "
                "comprehensively."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        counts = self._registry.get_counts()
        snapshot = self._registry.snapshot()
        return {
            "status": "finalized",
            "total_word_count": counts["total_word_count"],
            "total_messages": len(snapshot),
            "messages": snapshot,
            "draft_xml": _snapshot_to_xml(snapshot),
        }


def _snapshot_to_xml(snapshot: list[dict[str, Any]]) -> str:
    """Serialize one registry snapshot into speaker-tagged XML."""
    return serialize_summary_turns(
        [
            SummaryTurn(
                speaker=str(message["speaker_id"]),
                text=str(message["content"]),
                emo_preset=str(message.get("emo_preset", DEFAULT_EMO_PRESET)),
            )
            for message in snapshot
        ]
    )


def build_summarizer_tools(
    registry: MessageRegistry,
    retrieval_port: int,
    calibration: VoiceCloneCalibration,
    target_seconds: int,
    duration_tolerance_ratio: float,
    is_embedding_enabled: bool = True,
    agent_client: AgentTaskClient | None = None,
) -> ToolRegistry:
    """Build the full summarizer tool registry."""
    budget = BudgetContext(
        registry,
        calibration,
        target_seconds,
        duration_tolerance_ratio,
    )
    tool_registry = ToolRegistry()
    tool_registry.register(AddSpeakerMessageHandler(registry, budget))
    tool_registry.register(EstimateDurationHandler(registry, budget, calibration))
    tool_registry.register(CountWordsHandler(registry, budget))
    tool_registry.register(EditMessageHandler(registry, budget))
    tool_registry.register(RemoveMessageHandler(registry, budget))
    if is_embedding_enabled:
        tool_registry.register(
            QueryTranscriptHandler(retrieval_port, agent_client=agent_client)
        )
    tool_registry.register(SaveDraftHandler(registry))
    tool_registry.register(FinalizeDraftHandler(registry))
    return tool_registry


def build_revise_tools(
    registry: MessageRegistry,
    retrieval_port: int,
    calibration: VoiceCloneCalibration,
    target_seconds: int,
    duration_tolerance_ratio: float,
    is_embedding_enabled: bool = True,
    agent_client: AgentTaskClient | None = None,
) -> ToolRegistry:
    """Build revise-mode tools without add_speaker_message."""
    budget = BudgetContext(
        registry,
        calibration,
        target_seconds,
        duration_tolerance_ratio,
    )
    tool_registry = ToolRegistry()
    tool_registry.register(EstimateDurationHandler(registry, budget, calibration))
    tool_registry.register(CountWordsHandler(registry, budget))
    tool_registry.register(EditMessageHandler(registry, budget))
    tool_registry.register(RemoveMessageHandler(registry, budget))
    if is_embedding_enabled:
        tool_registry.register(
            QueryTranscriptHandler(retrieval_port, agent_client=agent_client)
        )
    tool_registry.register(SaveDraftHandler(registry))
    tool_registry.register(FinalizeDraftHandler(registry))
    return tool_registry
