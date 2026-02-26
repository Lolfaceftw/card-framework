"""
DeepSeek LLM Provider
======================
Uses the official OpenAI Python client pointing to DeepSeek API.
"""

import json
import sys
from copy import deepcopy
from typing import Any

from openai import OpenAI

from events import event_bus
from llm_provider import LLMProvider
from ui import ui


from agents.dtos import AssistantMessage, Function, ToolCall


class DeepSeekProvider(LLMProvider):
    """
    Concrete LLM strategy for DeepSeek API.

    Args:
        api_key: DeepSeek API key.
        model: Model name to use (e.g. 'deepseek-chat', 'deepseek-reasoner').
        base_url: The base URL of the DeepSeek API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        event_bus.publish(
            "system_message",
            f"Connected to DeepSeek → model={self.model}, url={self.base_url}",
        )

    # ── LLMProvider interface ─────────────────────────────────────────────

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        create_kwargs: dict = dict(
            model=self.model,
            messages=messages,
            stream=True,
        )
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**create_kwargs)

        full_content = ""
        full_reasoning = ""

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # Handle DeepSeek reasoning_content (Chain of Thought)
            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                # We print reasoning to stderr or with a prefix to distinguish it from the final answer
                sys.stdout.write(f"\033[90m{delta.reasoning_content}\033[0m")
                sys.stdout.flush()
                full_reasoning += delta.reasoning_content

            if delta.content:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                full_content += delta.content

        sys.stdout.write("\n")
        sys.stdout.flush()
        return full_content.strip()

    @staticmethod
    def _copy_message_without_reasoning(msg: dict[str, Any]) -> dict[str, Any]:
        """Return a defensive copy of a message without stale reasoning content."""
        msg_clean = {
            k: deepcopy(v)
            for k, v in msg.items()
            if k != "reasoning_content" and v is not None
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

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        """
        Cleans and normalizes message history for DeepSeek's strict validation.
        - Enforces alternating roles (merges consecutive assistant/user messages).
        - Corrects tool call sequences to ensure no orphaned tool responses.
        - Converts unsolicited automated tool messages into user role descriptions.
        """
        normalized = []
        valid_tool_call_ids = set()

        for msg in messages:
            # Drop reasoning_content to save context space/bandwidth as per DeepSeek docs
            msg_clean = self._copy_message_without_reasoning(msg)

            # Track valid tool calls so we can detect orphans
            if msg_clean.get("role") == "assistant" and msg_clean.get("tool_calls"):
                for tc in msg_clean["tool_calls"]:
                    valid_tool_call_ids.add(tc["id"])

            # DeepSeek strict validation: tool messages MUST have a preceding tool_call.
            # A2A sometimes auto-injects tool messages (e.g. auto count_words).
            if msg_clean.get("role") == "tool":
                tc_id = msg_clean.get("tool_call_id")
                if tc_id not in valid_tool_call_ids:
                    automated_content = msg_clean.get("content", "")
                    tool_name = msg_clean.get("name", "unknown_tool")
                    msg_clean = {
                        "role": "user",
                        "content": f"[Automated System Tool Output: {tool_name}]\n{automated_content}",
                    }

            if not normalized:
                normalized.append(msg_clean)
                continue

            last_msg = normalized[-1]

            # Rule 1: No consecutive messages of same role.
            if msg_clean["role"] == last_msg["role"] and msg_clean["role"] in [
                "assistant",
                "user",
            ]:
                if msg_clean.get("content"):
                    if last_msg.get("content"):
                        last_msg["content"] += "\n" + msg_clean["content"]
                    else:
                        last_msg["content"] = msg_clean["content"]

                if msg_clean.get("tool_calls"):
                    # CRITICAL FIX for combined tool calls:
                    # If we merge two assistant messages that both have tool_calls,
                    # we must combine them into one array so the subsequent tool responses
                    # aren't "split" across turns.
                    merged_calls = self._merge_tool_calls(
                        existing_calls=last_msg.get("tool_calls"),
                        incoming_calls=msg_clean["tool_calls"],
                    )
                    last_msg["tool_calls"] = merged_calls

                    for tc in merged_calls:
                        valid_tool_call_ids.add(tc["id"])

                if "reasoning_content" in msg_clean:
                    if last_msg.get("reasoning_content"):
                        last_msg["reasoning_content"] += (
                            "\n" + msg_clean["reasoning_content"]
                        )
                    else:
                        last_msg["reasoning_content"] = msg_clean["reasoning_content"]
            else:
                normalized.append(msg_clean)

        # Final pass: DeepSeek REQUIRES:
        # 1. 'reasoning_content' in assistant messages that have tool_calls.
        # 2. Every assistant message with 'tool_calls' MUST be followed by tool messages.
        # If we have a dangling tool call (e.g. hallucinated or skipped by agent loop), we prune it.
        final_normalized = []
        for i, m in enumerate(normalized):
            if m.get("role") == "assistant" and m.get("tool_calls"):
                is_followed_by_tool = (
                    i + 1 < len(normalized) and normalized[i + 1].get("role") == "tool"
                )
                if not is_followed_by_tool:
                    # Healing: Prune tool_calls if no response follows to satisfy API constraints
                    m = m.copy()
                    m.pop("tool_calls")
                    if not m.get("content"):
                        m["content"] = "[Tool call skipped or invalid]"

                # Re-check in case we didn't prune: must have reasoning_content
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    if "reasoning_content" not in m or m["reasoning_content"] is None:
                        m["reasoning_content"] = ""

            final_normalized.append(m)

        return final_normalized

    def _process_stream_and_ui(
        self, response_stream, agent_name: str
    ) -> AssistantMessage:
        """Handles streaming output for Chain of Thought and Content to the UI."""
        full_content = ""
        full_thought = ""
        tool_calls_data = []

        with ui.live_agent_message(agent_name) as live_msg:
            for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Handle Chain of Thought
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    live_msg.update_thought(delta.reasoning_content)
                    full_thought += delta.reasoning_content

                # Handle Final Answer Content
                if hasattr(delta, "content") and delta.content:
                    live_msg.update_content(delta.content)
                    full_content += delta.content

                # Handle Incremental Tool Calls Build
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

                        tc = tool_calls_data[tc_delta.index]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

        # Convert raw tool data to DTOs
        final_tool_calls = []
        for td in tool_calls_data:
            final_tool_calls.append(
                ToolCall(
                    id=td["id"],
                    function=Function(
                        name=td["function"]["name"],
                        arguments=td["function"]["arguments"],
                    ),
                )
            )

        return AssistantMessage(
            content=full_content,
            tool_calls=final_tool_calls if final_tool_calls else None,
            reasoning_content=full_thought if full_thought else None,
        )

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
    ) -> AssistantMessage:
        """
        Clean, scalable implementation of chat completion with support for tools.
        """
        normalized_messages = self._normalize_messages(messages)

        create_kwargs: dict = dict(
            model=self.model,
            messages=normalized_messages,
            stream=True,
        )
        if tools:
            create_kwargs["tools"] = tools
            if tool_choice:
                create_kwargs["tool_choice"] = tool_choice

        if max_tokens:
            create_kwargs["max_tokens"] = max_tokens

        # Identify agent name for UI
        agent_name = "Agent"
        for msg in reversed(messages):
            if msg.get("role") == "system":
                content = str(msg.get("content", ""))
                if "Summarizer" in content:
                    agent_name = "Summarizer"
                elif "Critic" in content:
                    agent_name = "Critic"
                break

        response_stream = self._client.chat.completions.create(**create_kwargs)
        return self._process_stream_and_ui(response_stream, agent_name)
