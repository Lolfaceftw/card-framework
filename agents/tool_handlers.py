"""
Tool Handler Abstraction
========================
Command pattern for tool dispatch. Each tool is a self-contained handler
implementing a common interface, making it trivial to register new tools
without modifying the agent loop.

All handlers use async execute() so tools that need I/O (e.g. querying
the retrieval agent) work seamlessly alongside synchronous ones.
"""

import json
from abc import ABC, abstractmethod

from agents.client import AgentTaskClient, get_default_agent_client
from agents.message_registry import MessageRegistry


class ToolHandler(ABC):
    """Abstract base for all tool handlers (Command pattern)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name as the LLM sees it."""
        ...

    @property
    @abstractmethod
    def schema(self) -> dict:
        """OpenAI-compatible function tool definition."""
        ...

    @abstractmethod
    async def execute(self, arguments: dict) -> dict:
        """Execute the tool and return JSON-serialisable result."""
        ...


class ToolRegistry:
    """
    Registry that holds tool handlers and dispatches by name.
    Follows the Registry + Strategy patterns for open/closed extensibility.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, handler: ToolHandler) -> None:
        self._handlers[handler.name] = handler

    def get(self, name: str) -> ToolHandler | None:
        return self._handlers.get(name)

    def get_tool_schemas(self) -> list[dict]:
        """Return all tool schemas in the format expected by the LLM API."""
        return [
            {"type": "function", "function": h.schema} for h in self._handlers.values()
        ]

    async def dispatch(self, name: str, arguments: dict) -> dict | None:
        """Dispatch a tool call by name. Returns None if tool not found."""
        handler = self._handlers.get(name)
        if handler is None:
            return None
        return await handler.execute(arguments)


# ── Budget Context (Strategy pattern) ─────────────────────────────────────


class BudgetContext:
    """
    Computes budget statistics from the current registry state.
    Injected into tool handlers so every response includes actionable
    guidance for the LLM (deltas, avg words/line, estimated lines to change).
    """

    def __init__(self, registry: MessageRegistry, min_words: int, max_words: int):
        self._registry = registry
        self.min_words = min_words
        self.max_words = max_words

    def compute_stats(self) -> dict:
        """Return budget stats based on current registry contents."""
        counts = self._registry.get_counts()
        total = counts["total_word_count"]
        num_lines = len(self._registry)
        avg_words_per_line = round(total / num_lines, 1) if num_lines > 0 else 0

        delta_lower = total - self.min_words  # positive = above lower bound
        delta_upper = total - self.max_words  # negative = below upper bound

        # Estimate lines to add/remove to reach target midpoint
        target_mid = (self.min_words + self.max_words) // 2
        delta_mid = total - target_mid

        if avg_words_per_line > 0:
            est_lines_to_change = round(abs(delta_mid) / avg_words_per_line, 1)
        else:
            est_lines_to_change = 0

        if delta_mid > 0:
            action = "remove"
        elif delta_mid < 0:
            action = "add"
        else:
            action = "none"

        in_budget = self.min_words <= total <= self.max_words

        return {
            "budget": {
                "min_words": self.min_words,
                "max_words": self.max_words,
                "total_words": total,
                "delta_to_lower_bound": delta_lower,
                "delta_to_upper_bound": delta_upper,
                "in_budget": in_budget,
                "avg_words_per_line": avg_words_per_line,
                "estimated_lines_to_change": est_lines_to_change,
                "recommended_action": action,
            }
        }


# ── Concrete Tool Handlers ────────────────────────────────────────────────


class AddSpeakerMessageHandler(ToolHandler):
    """Appends a single speaker message to the registry."""

    def __init__(self, registry: MessageRegistry, budget: BudgetContext) -> None:
        self._registry = registry
        self._budget = budget

    @property
    def name(self) -> str:
        return "add_speaker_message"

    @property
    def schema(self) -> dict:
        return {
            "name": "add_speaker_message",
            "description": (
                "Adds a single speaker message to the summary. "
                "Call this once per speaker turn as you build the summary incrementally. "
                "After this call, count_words will be automatically invoked."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "speaker_id": {
                        "type": "string",
                        "description": "The speaker ID (e.g. SPEAKER_00).",
                    },
                    "content": {
                        "type": "string",
                        "description": "The summarised content for this speaker turn.",
                    },
                },
                "required": ["speaker_id", "content"],
            },
        }

    async def execute(self, arguments: dict) -> dict:
        result = self._registry.add(arguments["speaker_id"], arguments["content"])
        return {**result, **self._budget.compute_stats()}


class CountWordsHandler(ToolHandler):
    """Returns full word-count breakdown from the registry."""

    def __init__(self, registry: MessageRegistry, budget: BudgetContext) -> None:
        self._registry = registry
        self._budget = budget

    @property
    def name(self) -> str:
        return "count_words"

    @property
    def schema(self) -> dict:
        return {
            "name": "count_words",
            "description": (
                "Returns the current word-count breakdown: total count, "
                "per-message counts with line numbers, and per-speaker totals. "
                "No arguments needed — reads from the message registry."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    async def execute(self, arguments: dict) -> dict:
        counts = self._registry.get_counts()
        return {**counts, **self._budget.compute_stats()}


class EditMessageHandler(ToolHandler):
    """Replaces the content of a specific message by line number."""

    def __init__(self, registry: MessageRegistry, budget: BudgetContext) -> None:
        self._registry = registry
        self._budget = budget

    @property
    def name(self) -> str:
        return "edit_message"

    @property
    def schema(self) -> dict:
        return {
            "name": "edit_message",
            "description": (
                "Replaces the content of a message at a given line number. "
                "Use this to trim or expand a specific speaker turn to stay within budget."
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
                },
                "required": ["line", "new_content"],
            },
        }

    async def execute(self, arguments: dict) -> dict:
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

        result = self._registry.edit(line, new_content)
        if "error" not in result:
            result.update(self._budget.compute_stats())
        return result


class RemoveMessageHandler(ToolHandler):
    """Removes a message by line number."""

    def __init__(self, registry: MessageRegistry, budget: BudgetContext) -> None:
        self._registry = registry
        self._budget = budget

    @property
    def name(self) -> str:
        return "remove_message"

    @property
    def schema(self) -> dict:
        return {
            "name": "remove_message",
            "description": (
                "Removes the message at a given line number. "
                "Remaining messages are re-indexed. Use to drop turns when over budget."
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

    async def execute(self, arguments: dict) -> dict:
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
    """Queries the Info Retrieval agent for transcript segments matching a semantic query."""

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
    def schema(self) -> dict:
        return {
            "name": "query_transcript",
            "description": (
                "Searches the original transcript for segments matching your query. "
                "Use this when you need to recall what a speaker said about a specific topic, "
                "verify facts, or find additional details to include in your summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A natural-language query describing what you want to find "
                            "(e.g. 'what did they say about server architecture')."
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

    async def execute(self, arguments: dict) -> dict:
        from agents.dtos import RetrieveTaskRequest

        query = arguments["query"]
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
    """Snapshots the current registry state as a saved draft."""

    def __init__(self, registry: MessageRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "save_draft"

    @property
    def schema(self) -> dict:
        return {
            "name": "save_draft",
            "description": (
                "Saves the current draft from the message registry. "
                "Called automatically when budget is met. "
                "Returns the full draft with line numbers for reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    async def execute(self, arguments: dict) -> dict:
        snapshot = self._registry.snapshot()
        return {
            "status": "draft_saved",
            "messages": snapshot,
            "total_messages": len(snapshot),
            "draft_xml": _snapshot_to_xml(snapshot),
        }


class FinalizeDraftHandler(ToolHandler):
    """Signals that the LLM is satisfied with the draft and ready for the critic."""

    def __init__(self, registry: MessageRegistry) -> None:
        self._registry = registry

    @property
    def name(self) -> str:
        return "finalize_draft"

    @property
    def schema(self) -> dict:
        return {
            "name": "finalize_draft",
            "description": (
                "Call this ONLY when you have reviewed your draft, confirmed it meets "
                "the word budget, reads naturally, and covers the transcript comprehensively. "
                "This submits the draft to the Critic for evaluation. "
                "Do NOT call this until you are fully satisfied with the draft."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        }

    async def execute(self, arguments: dict) -> dict:
        counts = self._registry.get_counts()
        snapshot = self._registry.snapshot()
        return {
            "status": "finalized",
            "total_word_count": counts["total_word_count"],
            "total_messages": len(snapshot),
            "messages": snapshot,
            "draft_xml": _snapshot_to_xml(snapshot),
        }


def _snapshot_to_xml(snapshot: list[dict]) -> str:
    """Serialize a registry snapshot into speaker-tagged XML."""
    return "\n".join(
        f"<{message['speaker_id']}>{message['content']}</{message['speaker_id']}>"
        for message in snapshot
    )


def build_summarizer_tools(
    registry: MessageRegistry,
    retrieval_port: int,
    min_words: int,
    max_words: int,
    is_embedding_enabled: bool = True,
    agent_client: AgentTaskClient | None = None,
) -> ToolRegistry:
    """Factory: wires up all summariser tool handlers against *registry*."""
    budget = BudgetContext(registry, min_words, max_words)
    tool_registry = ToolRegistry()
    tool_registry.register(AddSpeakerMessageHandler(registry, budget))
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
    min_words: int,
    max_words: int,
    is_embedding_enabled: bool = True,
    agent_client: AgentTaskClient | None = None,
) -> ToolRegistry:
    """Factory: revise-mode tools — edit/remove only, no add_speaker_message."""
    budget = BudgetContext(registry, min_words, max_words)
    tool_registry = ToolRegistry()
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

