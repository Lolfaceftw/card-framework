"""Use the DeepSeek chat API through the shared LLM provider interface."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
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
    infer_agent_name_from_text,
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
        request_timeout_seconds: float = 30.0,
        default_max_tokens: int | None = None,
        thinking_mode: str = "",
        reasoning_effort: str = "",
        extra_body: dict[str, Any] | None = None,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        stream: bool = True,
        response_callback: LLMResponseCallback | None = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.default_max_tokens = (
            int(default_max_tokens)
            if default_max_tokens is not None and int(default_max_tokens) > 0
            else None
        )
        self.thinking_mode = str(thinking_mode or "").strip()
        self.reasoning_effort = str(reasoning_effort or "").strip()
        self.extra_body = self._coerce_jsonable(extra_body) or {}
        self.response_format = self._coerce_jsonable(response_format) or {}
        self.temperature = float(temperature) if temperature is not None else None
        self.stream = bool(stream)
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout_seconds,
        )
        self._response_callback: LLMResponseCallback = (
            response_callback or NullLLMResponseCallback()
        )
        self.last_response_details: dict[str, Any] = {}

        event_bus.publish(
            "system_message",
            f"Connected to DeepSeek -> model={self.model}, url={self.base_url}",
        )

    def set_response_callback(self, response_callback: LLMResponseCallback) -> None:
        """Replace the streaming callback sink at runtime."""
        self._response_callback = response_callback

    @classmethod
    def _coerce_jsonable(cls, value: Any) -> Any:
        """Convert nested SDK/config containers into JSON-serializable builtins."""
        if hasattr(value, "model_dump"):
            return cls._coerce_jsonable(value.model_dump())
        if isinstance(value, Mapping):
            return {
                str(key): cls._coerce_jsonable(raw_value)
                for key, raw_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._coerce_jsonable(item) for item in value]
        return value

    @classmethod
    def _merge_dicts(
        cls,
        base: dict[str, Any],
        overlay: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge nested request dictionaries without mutating inputs."""
        merged = cls._coerce_jsonable(base)
        for key, value in cls._coerce_jsonable(overlay).items():
            existing = merged.get(key)
            if isinstance(existing, Mapping) and isinstance(value, Mapping):
                merged[key] = cls._merge_dicts(dict(existing), dict(value))
                continue
            merged[key] = value
        return merged

    def _resolve_max_tokens(self, max_tokens: int | None) -> int | None:
        """Return the call-level output budget or the configured default."""
        if max_tokens is not None:
            return int(max_tokens)
        return self.default_max_tokens

    def _apply_common_request_options(
        self,
        create_kwargs: dict[str, Any],
        *,
        max_tokens: int | None,
        include_response_format: bool,
    ) -> dict[str, Any]:
        """Attach optional DeepSeek request controls to one payload."""
        effective_max_tokens = self._resolve_max_tokens(max_tokens)
        if effective_max_tokens is not None:
            create_kwargs["max_tokens"] = effective_max_tokens
        if self.temperature is not None:
            create_kwargs["temperature"] = self.temperature
        if include_response_format and self.response_format:
            create_kwargs["response_format"] = self.response_format

        extra_body = dict(self.extra_body)
        if self.thinking_mode:
            extra_body = self._merge_dicts(
                extra_body,
                {"thinking": {"type": self.thinking_mode}},
            )
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        if self.thinking_mode == "enabled" and self.reasoning_effort:
            create_kwargs["reasoning_effort"] = self.reasoning_effort
        return create_kwargs

    def _record_last_response(
        self,
        *,
        request: dict[str, Any],
        content: str,
        reasoning_content: str,
        tool_calls: list[ToolCall] | None = None,
        raw_response: Any | None = None,
    ) -> None:
        """Persist a lightweight request/response summary for artifact wrappers."""
        response_payload: dict[str, Any] = {
            "content": content,
            "reasoning_content": reasoning_content or None,
        }
        if tool_calls is not None:
            response_payload["tool_calls"] = [
                tool_call.model_dump() for tool_call in tool_calls
            ]
        if raw_response is not None:
            response_payload["raw_response"] = self._coerce_jsonable(raw_response)
        self.last_response_details = {
            "provider": "deepseek",
            "request": self._coerce_jsonable(request),
            "response": response_payload,
        }

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
            "stream": self.stream,
        }
        create_kwargs = self._apply_common_request_options(
            create_kwargs,
            max_tokens=max_tokens,
            include_response_format=True,
        )

        if not self.stream:
            response = self._client.chat.completions.create(**create_kwargs)
            message = response.choices[0].message
            content = str(getattr(message, "content", "") or "")
            reasoning_content = str(getattr(message, "reasoning_content", "") or "")
            agent_name = infer_agent_name_from_text(system_prompt)
            self._response_callback.on_start(agent_name)
            try:
                if reasoning_content:
                    self._response_callback.on_thought_token(reasoning_content)
                if content:
                    self._response_callback.on_content_token(content)
            finally:
                self._response_callback.on_complete()
            self._record_last_response(
                request=create_kwargs,
                content=content,
                reasoning_content=reasoning_content,
                raw_response=response,
            )
            return content.strip()

        response = self._client.chat.completions.create(**create_kwargs)
        full_content = ""
        full_reasoning = ""
        self._response_callback.on_start(infer_agent_name_from_text(system_prompt))
        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    self._response_callback.on_thought_token(delta.reasoning_content)
                    full_reasoning += delta.reasoning_content

                if delta.content:
                    self._response_callback.on_content_token(delta.content)
                    full_content += delta.content
        finally:
            self._response_callback.on_complete()

        self._record_last_response(
            request=create_kwargs,
            content=full_content,
            reasoning_content=full_reasoning,
        )
        return full_content.strip()

    @staticmethod
    def _copy_message_without_reasoning(msg: dict[str, Any]) -> dict[str, Any]:
        """Return a defensive copy of a message without stale reasoning content."""
        return {
            key: deepcopy(value)
            for key, value in msg.items()
            if key != "reasoning_content" and value is not None
        }

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

    def _normalize_messages(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
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
                    last_tool_calls = last_msg.get("tool_calls")
                    merged_calls = self._merge_tool_calls(
                        existing_calls=(
                            last_tool_calls
                            if isinstance(last_tool_calls, list)
                            else None
                        ),
                        incoming_calls=(
                            msg_clean["tool_calls"]
                            if isinstance(msg_clean["tool_calls"], list)
                            else None
                        ),
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

    def _assistant_message_from_response(self, response: Any) -> AssistantMessage:
        """Convert one non-streaming DeepSeek response into an assistant message."""
        message = response.choices[0].message
        raw_tool_calls = list(getattr(message, "tool_calls", None) or [])
        final_tool_calls = [
            ToolCall(
                id=str(getattr(tool_call, "id", None) or f"call_{index}"),
                function=Function(
                    name=str(getattr(tool_call.function, "name", "")),
                    arguments=str(getattr(tool_call.function, "arguments", "")),
                ),
            )
            for index, tool_call in enumerate(raw_tool_calls)
        ]
        content = str(getattr(message, "content", "") or "")
        reasoning_content = str(getattr(message, "reasoning_content", "") or "")
        return AssistantMessage(
            content=content,
            tool_calls=final_tool_calls if final_tool_calls else None,
            reasoning_content=reasoning_content if reasoning_content else None,
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
            "stream": self.stream,
        }
        if normalized_tools:
            create_kwargs["tools"] = normalized_tools
            if tool_choice:
                create_kwargs["tool_choice"] = tool_choice

        create_kwargs = self._apply_common_request_options(
            create_kwargs,
            max_tokens=max_tokens,
            include_response_format=not bool(normalized_tools),
        )

        agent_name = infer_agent_name(messages)
        if not self.stream:
            response = self._client.chat.completions.create(**create_kwargs)
            assistant_message = self._assistant_message_from_response(response)
            self._response_callback.on_start(agent_name)
            try:
                if assistant_message.reasoning_content:
                    self._response_callback.on_thought_token(
                        assistant_message.reasoning_content
                    )
                if assistant_message.content:
                    self._response_callback.on_content_token(assistant_message.content)
            finally:
                self._response_callback.on_complete()
            self._record_last_response(
                request=create_kwargs,
                content=assistant_message.content,
                reasoning_content=assistant_message.reasoning_content or "",
                tool_calls=assistant_message.tool_calls,
                raw_response=response,
            )
            return assistant_message

        response_stream = self._client.chat.completions.create(**create_kwargs)
        assistant_message = self._process_stream(response_stream, agent_name)
        self._record_last_response(
            request=create_kwargs,
            content=assistant_message.content,
            reasoning_content=assistant_message.reasoning_content or "",
            tool_calls=assistant_message.tool_calls,
        )
        return assistant_message
