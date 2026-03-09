"""Use the DeepSeek chat API through the shared LLM provider interface."""

from __future__ import annotations

import json
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from openai import OpenAI

from card_framework.agents.dtos import AssistantMessage, Function, ToolCall
from card_framework.shared.events import event_bus
from card_framework.shared.llm_provider import (
    LLMProvider,
    LLMResponseCallback,
    MessageInput,
    NullLLMResponseCallback,
    ToolChoice,
    ToolInput,
    infer_agent_name,
    normalize_messages,
    normalize_tools,
)


class DeepSeekProvider(LLMProvider):
    """Call DeepSeek chat completions with tool and streaming support."""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        response_callback: LLMResponseCallback | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        self._response_callback: LLMResponseCallback = (
            response_callback or NullLLMResponseCallback()
        )

        event_bus.publish(
            "system_message",
            f"Connected to DeepSeek -> model={self.model}, url={self.base_url}",
        )

    def set_response_callback(self, response_callback: LLMResponseCallback) -> None:
        """Replace the streaming callback sink at runtime."""
        self._response_callback = response_callback

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate one streamed non-tool response."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**create_kwargs)

        full_content = ""
        self._response_callback.on_start("Agent")
        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    self._response_callback.on_thought_token(delta.reasoning_content)

                if delta.content:
                    self._response_callback.on_content_token(delta.content)
                    full_content += delta.content
        finally:
            self._response_callback.on_complete()

        return full_content.strip()

    @staticmethod
    def _copy_message_without_reasoning(msg: dict[str, Any]) -> dict[str, Any]:
        """Return a defensive copy of a message without stale reasoning content."""
        msg_clean = {
            key: deepcopy(value)
            for key, value in msg.items()
            if key != "reasoning_content" and value is not None
        }
        if msg.get("role") == "assistant" and "reasoning_content" in msg:
            msg_clean["reasoning_content"] = msg["reasoning_content"]
        return msg_clean

    @staticmethod
    def _tool_call_dedupe_key(tool_call: dict[str, Any]) -> str:
        """Build a stable dedupe key for tool calls in normalized history."""
        function = tool_call.get("function", {})
        key_payload = {
            "id": tool_call.get("id"),
            "name": function.get("name"),
            "arguments": function.get("arguments"),
        }
        return json.dumps(key_payload, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _merge_tool_calls(
        cls,
        existing_calls: list[dict[str, Any]] | None,
        incoming_calls: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        """Merge and deduplicate tool calls without mutating source lists."""
        merged_calls: list[dict[str, Any]] = []
        seen: set[str] = set()

        for source_calls in (existing_calls or [], incoming_calls or []):
            for tool_call in source_calls:
                copied = deepcopy(tool_call)
                dedupe_key = cls._tool_call_dedupe_key(copied)
                if dedupe_key in seen:
                    continue
                seen.add(dedupe_key)
                merged_calls.append(copied)

        return merged_calls

    def _normalize_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Normalize message history to satisfy DeepSeek tool-call constraints."""
        normalized: list[dict[str, Any]] = []
        valid_tool_call_ids: set[str] = set()

        for msg in messages:
            msg_clean = self._copy_message_without_reasoning(msg)

            if msg_clean.get("role") == "assistant" and msg_clean.get("tool_calls"):
                for tool_call in msg_clean["tool_calls"]:
                    valid_tool_call_ids.add(tool_call["id"])

            if msg_clean.get("role") == "tool":
                tool_call_id = msg_clean.get("tool_call_id")
                if tool_call_id not in valid_tool_call_ids:
                    automated_content = msg_clean.get("content", "")
                    tool_name = msg_clean.get("name", "unknown_tool")
                    msg_clean = {
                        "role": "user",
                        "content": (
                            f"[Automated System Tool Output: {tool_name}]\n"
                            f"{automated_content}"
                        ),
                    }

            if not normalized:
                normalized.append(msg_clean)
                continue

            last_msg = normalized[-1]
            if msg_clean["role"] == last_msg["role"] and msg_clean["role"] in {
                "assistant",
                "user",
            }:
                if msg_clean.get("content"):
                    if last_msg.get("content"):
                        last_msg["content"] += "\n" + msg_clean["content"]
                    else:
                        last_msg["content"] = msg_clean["content"]

                if msg_clean.get("tool_calls"):
                    merged_calls = self._merge_tool_calls(
                        existing_calls=last_msg.get("tool_calls"),
                        incoming_calls=msg_clean["tool_calls"],
                    )
                    last_msg["tool_calls"] = merged_calls
                    for tool_call in merged_calls:
                        valid_tool_call_ids.add(tool_call["id"])

                if "reasoning_content" in msg_clean:
                    if last_msg.get("reasoning_content"):
                        last_msg["reasoning_content"] += (
                            "\n" + msg_clean["reasoning_content"]
                        )
                    else:
                        last_msg["reasoning_content"] = msg_clean["reasoning_content"]
                continue

            normalized.append(msg_clean)

        final_normalized: list[dict[str, Any]] = []
        for index, message in enumerate(normalized):
            if message.get("role") == "assistant" and message.get("tool_calls"):
                is_followed_by_tool = (
                    index + 1 < len(normalized)
                    and normalized[index + 1].get("role") == "tool"
                )
                if not is_followed_by_tool:
                    message = message.copy()
                    message.pop("tool_calls")
                    if not message.get("content"):
                        message["content"] = "[Tool call skipped or invalid]"

                if message.get("role") == "assistant" and message.get("tool_calls"):
                    if (
                        "reasoning_content" not in message
                        or message["reasoning_content"] is None
                    ):
                        message["reasoning_content"] = ""

            final_normalized.append(message)

        return final_normalized

    def _process_stream(
        self,
        response_stream: Any,
        agent_name: str,
    ) -> AssistantMessage:
        """Stream reasoning/content through the configured callback and collect tools."""
        full_content = ""
        full_thought = ""
        tool_calls_data: list[dict[str, Any]] = []

        self._response_callback.on_start(agent_name)
        try:
            for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    self._response_callback.on_thought_token(delta.reasoning_content)
                    full_thought += delta.reasoning_content

                if hasattr(delta, "content") and delta.content:
                    self._response_callback.on_content_token(delta.content)
                    full_content += delta.content

                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        while len(tool_calls_data) <= tc_delta.index:
                            tool_calls_data.append(
                                {
                                    "id": tc_delta.id or "call_unknown",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )

                        tool_call = tool_calls_data[tc_delta.index]
                        if tc_delta.id:
                            tool_call["id"] = tc_delta.id
                        if tc_delta.function.name:
                            tool_call["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_call["function"]["arguments"] += (
                                tc_delta.function.arguments
                            )
        finally:
            self._response_callback.on_complete()

        final_tool_calls = [
            ToolCall(
                id=tool_data["id"],
                function=Function(
                    name=tool_data["function"]["name"],
                    arguments=tool_data["function"]["arguments"],
                ),
            )
            for tool_data in tool_calls_data
        ]

        return AssistantMessage(
            content=full_content,
            tool_calls=final_tool_calls if final_tool_calls else None,
            reasoning_content=full_thought if full_thought else None,
        )

    def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
    ) -> AssistantMessage:
        """Create one streamed chat completion with optional tool support."""
        normalized_messages = self._normalize_messages(normalize_messages(messages))
        normalized_tools = normalize_tools(tools)

        create_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": normalized_messages,
            "stream": True,
        }
        if normalized_tools:
            create_kwargs["tools"] = normalized_tools
            if tool_choice:
                create_kwargs["tool_choice"] = tool_choice

        if max_tokens:
            create_kwargs["max_tokens"] = max_tokens

        response_stream = self._client.chat.completions.create(**create_kwargs)
        agent_name = infer_agent_name(messages)
        return self._process_stream(response_stream, agent_name)

