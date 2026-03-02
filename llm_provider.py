"""
Abstract Base Classes for LLM and Embedding Providers
=====================================================
Strategy interfaces that allow swapping LLM and embedding backends
without changing any agent or retrieval logic.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import numpy as np


MessageRole = Literal["system", "user", "assistant", "tool", "developer", "model"]
VALID_MESSAGE_ROLES: set[str] = {"system", "user", "assistant", "tool", "developer", "model"}


@dataclass(slots=True, frozen=True)
class ToolCallDefinition:
    """Typed tool-call record used in assistant history messages."""

    id: str
    name: str
    arguments: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object], index: int) -> "ToolCallDefinition":
        """Parse a tool-call payload from OpenAI-compatible message history."""
        function_payload = payload.get("function")
        function = function_payload if isinstance(function_payload, Mapping) else {}
        raw_arguments = function.get("arguments", "{}")
        return cls(
            id=str(payload.get("id") or f"call_{index}"),
            name=str(function.get("name") or "unknown_tool"),
            arguments=str(raw_arguments),
        )

    def to_openai_dict(self) -> dict[str, object]:
        """Serialize this tool call into an OpenAI-compatible dictionary."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass(slots=True, frozen=True)
class Message:
    """Typed chat message value object used at LLM boundaries."""

    role: MessageRole
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: tuple[ToolCallDefinition, ...] | None = None
    reasoning_content: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "Message":
        """Build a typed message from an OpenAI-compatible mapping."""
        role_value = str(payload.get("role") or "user").strip().lower()
        normalized_role = role_value if role_value in VALID_MESSAGE_ROLES else "user"
        raw_content = payload.get("content")
        content_value = "" if raw_content is None else str(raw_content)

        parsed_tool_calls: tuple[ToolCallDefinition, ...] | None = None
        raw_tool_calls = payload.get("tool_calls")
        if isinstance(raw_tool_calls, Sequence) and not isinstance(
            raw_tool_calls, (str, bytes)
        ):
            tool_call_values = [
                ToolCallDefinition.from_mapping(raw_tool_call, index)
                for index, raw_tool_call in enumerate(raw_tool_calls)
                if isinstance(raw_tool_call, Mapping)
            ]
            if tool_call_values:
                parsed_tool_calls = tuple(tool_call_values)

        raw_reasoning = payload.get("reasoning_content")
        reasoning_content = None if raw_reasoning is None else str(raw_reasoning)

        raw_name = payload.get("name")
        name = None if raw_name is None else str(raw_name)

        raw_tool_call_id = payload.get("tool_call_id")
        tool_call_id = None if raw_tool_call_id is None else str(raw_tool_call_id)

        return cls(
            role=cast(MessageRole, normalized_role),
            content=content_value,
            name=name,
            tool_call_id=tool_call_id,
            tool_calls=parsed_tool_calls,
            reasoning_content=reasoning_content,
        )

    def to_openai_dict(self) -> dict[str, object]:
        """Serialize this message into an OpenAI-compatible payload."""
        payload: dict[str, object] = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        if self.tool_calls is not None:
            payload["tool_calls"] = [tc.to_openai_dict() for tc in self.tool_calls]
        if self.reasoning_content is not None:
            payload["reasoning_content"] = self.reasoning_content
        return payload


@dataclass(slots=True, frozen=True)
class ToolDefinition:
    """Typed tool definition used for function-calling capable providers."""

    name: str
    description: str
    parameters: Mapping[str, object]
    strict: bool | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> "ToolDefinition":
        """Parse a tool definition from either flattened or OpenAI-shaped mappings."""
        function_payload = payload.get("function")
        function = function_payload if isinstance(function_payload, Mapping) else payload
        strict_value = function.get("strict")
        strict = strict_value if isinstance(strict_value, bool) else None
        raw_parameters = function.get("parameters")
        parameters = raw_parameters if isinstance(raw_parameters, Mapping) else {}
        return cls(
            name=str(function.get("name") or "unnamed_tool"),
            description=str(function.get("description") or ""),
            parameters=parameters,
            strict=strict,
        )

    def to_openai_dict(self) -> dict[str, object]:
        """Serialize this tool definition into an OpenAI-compatible payload."""
        function_payload: dict[str, object] = {
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
        }
        if self.strict is not None:
            function_payload["strict"] = self.strict
        return {"type": "function", "function": function_payload}


MessageInput = Message | Mapping[str, object]
ToolInput = ToolDefinition | Mapping[str, object]
ToolChoice = str | Mapping[str, object]


def normalize_messages(messages: Sequence[MessageInput]) -> list[dict[str, object]]:
    """Normalize typed or mapping-based messages into OpenAI-compatible dicts."""
    normalized: list[dict[str, object]] = []
    for message in messages:
        if isinstance(message, Message):
            normalized.append(message.to_openai_dict())
            continue
        normalized.append(Message.from_mapping(message).to_openai_dict())
    return normalized


def normalize_tools(
    tools: Sequence[ToolInput] | None,
) -> list[dict[str, object]] | None:
    """Normalize typed or mapping-based tools into OpenAI-compatible dicts."""
    if tools is None:
        return None
    normalized: list[dict[str, object]] = []
    for tool in tools:
        if isinstance(tool, ToolDefinition):
            normalized.append(tool.to_openai_dict())
            continue
        normalized.append(ToolDefinition.from_mapping(tool).to_openai_dict())
    return normalized


def infer_agent_name(messages: Sequence[MessageInput], default: str = "Agent") -> str:
    """Infer display name from the latest system message in conversation history."""
    for message in reversed(messages):
        typed_message = (
            message if isinstance(message, Message) else Message.from_mapping(message)
        )
        if typed_message.role != "system":
            continue
        content = typed_message.content
        if "Summarizer" in content:
            return "Summarizer"
        if "Critic" in content:
            return "Critic"
        break
    return default


class LLMResponseCallback(Protocol):
    """Callback protocol for streaming token/thought updates from providers."""

    def on_start(self, agent_name: str) -> None:
        """Signal the beginning of a streamed response."""

    def on_thought_token(self, token: str) -> None:
        """Handle one reasoning/thought token chunk."""

    def on_content_token(self, token: str) -> None:
        """Handle one final-content token chunk."""

    def on_complete(self) -> None:
        """Signal that a streamed response has completed."""


class NullLLMResponseCallback:
    """No-op callback used when providers are not wired to a UI sink."""

    def on_start(self, agent_name: str) -> None:
        del agent_name

    def on_thought_token(self, token: str) -> None:
        del token

    def on_content_token(self, token: str) -> None:
        del token

    def on_complete(self) -> None:
        return None


class LLMProvider(ABC):
    """Strategy interface for text-generation LLMs."""

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text completion and return the full response string."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
    ):
        """
        Chat completion with support for tools.
        Returns the full response object or a structured representation including tool calls.
        """
        ...


class EmbeddingProvider(ABC):
    """Strategy interface for text-embedding models."""

    @abstractmethod
    def encode(
        self,
        texts: list[str],
        *,
        normalize: bool = True,
        show_progress: bool = False,
        prompt_name: str | None = None,
    ) -> np.ndarray:
        """Encode a list of texts into an (N, D) embedding matrix."""
        ...
