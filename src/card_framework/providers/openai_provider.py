"""Use the OpenAI Responses API through the shared LLM provider interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    infer_agent_name_from_text,
    normalize_messages,
    normalize_tools,
)


def _coerce_jsonable(value: Any) -> Any:
    """Convert nested SDK values into JSON-serializable builtins."""
    if hasattr(value, "model_dump"):
        return _coerce_jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {
            str(key): _coerce_jsonable(raw_value) for key, raw_value in value.items()
        }
    if isinstance(value, list | tuple):
        return [_coerce_jsonable(item) for item in value]
    return value


class OpenAIProvider(LLMProvider):
    """Call OpenAI reasoning models through the Responses API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.4-mini",
        *,
        enable_thinking: bool = False,
        reasoning_effort: str | None = None,
        request_timeout_seconds: float = 60.0,
        response_callback: LLMResponseCallback | None = None,
        base_url: str | None = None,
    ) -> None:
        """Initialize the OpenAI provider."""
        if not str(api_key).strip():
            raise ValueError("OpenAIProvider requires a non-empty API key.")

        self.api_key = str(api_key).strip()
        self.model = str(model).strip() or "gpt-5.4-mini"
        self.enable_thinking = bool(enable_thinking)
        self.reasoning_effort = (
            str(reasoning_effort).strip().lower() if reasoning_effort else None
        )
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.base_url = str(base_url).strip() if base_url else ""
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.request_timeout_seconds,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        self._client = OpenAI(**client_kwargs)
        self._response_callback: LLMResponseCallback = (
            response_callback or NullLLMResponseCallback()
        )
        self.last_response_details: dict[str, Any] = {}

        event_bus.publish(
            "system_message",
            f"Connected to OpenAI -> model={self.model}",
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
        """Generate one non-tool response."""
        input_items: list[dict[str, Any]] = []
        if system_prompt.strip():
            input_items.append({"role": "system", "content": system_prompt})
        input_items.append(
            {"role": "user", "content": user_prompt if user_prompt else " "}
        )
        request_kwargs = self._build_request_kwargs(
            input_items=input_items,
            tools=None,
            tool_choice=None,
            max_tokens=max_tokens,
        )
        response = self._client.responses.create(**request_kwargs)
        assistant_message = self._assistant_message_from_response(response)
        self._emit_assistant_message(
            agent_name=infer_agent_name_from_text(system_prompt),
            assistant_message=assistant_message,
        )
        self.last_response_details = {
            "provider": "openai",
            "request": _coerce_jsonable(request_kwargs),
            "response": _coerce_jsonable(response),
        }
        return assistant_message.content.strip()

    def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
    ) -> AssistantMessage:
        """Create one Responses API request with optional function tools."""
        normalized_messages = normalize_messages(messages)
        input_items = self._normalize_messages(normalized_messages)
        normalized_tools = self._normalize_tools(tools)
        normalized_tool_choice = self._normalize_tool_choice(tool_choice)
        request_kwargs = self._build_request_kwargs(
            input_items=input_items,
            tools=normalized_tools,
            tool_choice=normalized_tool_choice,
            max_tokens=max_tokens,
        )
        response = self._client.responses.create(**request_kwargs)
        assistant_message = self._assistant_message_from_response(response)
        self._emit_assistant_message(
            agent_name=infer_agent_name(messages),
            assistant_message=assistant_message,
        )
        self.last_response_details = {
            "provider": "openai",
            "request": _coerce_jsonable(request_kwargs),
            "response": _coerce_jsonable(response),
        }
        return assistant_message

    def _build_request_kwargs(
        self,
        *,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Build one OpenAI Responses API payload."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": input_items,
            "store": False,
        }
        reasoning = self._build_reasoning_config()
        if reasoning is not None:
            request_kwargs["reasoning"] = reasoning
        if max_tokens is not None:
            request_kwargs["max_output_tokens"] = max_tokens
        if tools:
            request_kwargs["tools"] = tools
        if tool_choice is not None:
            request_kwargs["tool_choice"] = tool_choice
        return request_kwargs

    def _build_reasoning_config(self) -> dict[str, str] | None:
        """Resolve one benchmark-safe reasoning config."""
        effort = self.reasoning_effort
        if not effort:
            effort = "low" if self.enable_thinking else "none"
        return {"effort": effort}

    def _normalize_messages(
        self,
        normalized_messages: list[dict[str, object]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style history into Responses API input items."""
        input_items: list[dict[str, Any]] = []

        for message in normalized_messages:
            role = str(message.get("role") or "user").strip().lower()
            content = str(message.get("content") or "")
            if role == "developer":
                role = "system"

            if role == "tool":
                tool_call_id = str(message.get("tool_call_id") or "").strip()
                if tool_call_id:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": tool_call_id,
                            "output": content if content else " ",
                        }
                    )
                elif content:
                    input_items.append(
                        {
                            "role": "user",
                            "content": f"[Tool output]\n{content}",
                        }
                    )
                continue

            if role in {"assistant", "model"}:
                if content:
                    input_items.append({"role": "assistant", "content": content})
                raw_tool_calls = message.get("tool_calls")
                if isinstance(raw_tool_calls, Sequence) and not isinstance(
                    raw_tool_calls,
                    str | bytes,
                ):
                    for index, raw_tool_call in enumerate(raw_tool_calls):
                        if not isinstance(raw_tool_call, Mapping):
                            continue
                        function_payload = raw_tool_call.get("function", {})
                        function = (
                            function_payload
                            if isinstance(function_payload, Mapping)
                            else {}
                        )
                        name = str(function.get("name") or "").strip()
                        if not name:
                            continue
                        call_id = str(
                            raw_tool_call.get("id") or f"call_{index}"
                        ).strip()
                        arguments = str(function.get("arguments") or "{}")
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": call_id,
                                "name": name,
                                "arguments": arguments,
                            }
                        )
                continue

            input_items.append(
                {
                    "role": role,
                    "content": content if content else " ",
                }
            )

        return input_items

    def _normalize_tools(
        self,
        tools: Sequence[ToolInput] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert shared OpenAI-style function tools into Responses API tools."""
        normalized_tools = normalize_tools(tools)
        if not normalized_tools:
            return None
        response_tools: list[dict[str, Any]] = []
        for tool in normalized_tools:
            function_payload = tool.get("function", {})
            function = function_payload if isinstance(function_payload, Mapping) else {}
            response_tool: dict[str, Any] = {
                "type": "function",
                "name": str(function.get("name") or "unnamed_tool"),
                "description": str(function.get("description") or ""),
                "parameters": _coerce_jsonable(function.get("parameters") or {}),
            }
            strict = function.get("strict")
            if isinstance(strict, bool):
                response_tool["strict"] = strict
            response_tools.append(response_tool)
        return response_tools

    def _normalize_tool_choice(
        self,
        tool_choice: ToolChoice | None,
    ) -> str | dict[str, str] | None:
        """Map the shared tool-choice shape into Responses API tool-choice values."""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            normalized = tool_choice.strip().lower()
            if normalized in {"auto", "required", "none"}:
                return normalized
            return None
        if not isinstance(tool_choice, Mapping):
            return None
        type_name = str(tool_choice.get("type") or "").strip().lower()
        if type_name in {"auto", "required", "none"}:
            return type_name
        if type_name == "function":
            function_payload = tool_choice.get("function", {})
            function = function_payload if isinstance(function_payload, Mapping) else {}
            name = str(function.get("name") or "").strip()
            if name:
                return {"type": "function", "name": name}
        return None

    def _assistant_message_from_response(self, response: Any) -> AssistantMessage:
        """Convert one Responses API payload into a shared assistant message."""
        tool_calls: list[ToolCall] = []
        reasoning_parts: list[str] = []
        content_parts: list[str] = []

        for index, item in enumerate(getattr(response, "output", []) or []):
            item_type = str(getattr(item, "type", "") or "")
            if item_type == "function_call":
                call_id = str(
                    getattr(item, "call_id", "")
                    or getattr(item, "id", "")
                    or f"call_{index}"
                ).strip()
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        function=Function(
                            name=str(getattr(item, "name", "") or "tool"),
                            arguments=str(getattr(item, "arguments", "") or "{}"),
                        ),
                    )
                )
                continue
            if item_type == "reasoning":
                for summary in getattr(item, "summary", []) or []:
                    text = str(getattr(summary, "text", "") or "")
                    if text:
                        reasoning_parts.append(text)
                continue
            if item_type != "message":
                continue
            for part in getattr(item, "content", []) or []:
                part_text = str(getattr(part, "text", "") or "")
                if not part_text:
                    continue
                if bool(getattr(part, "thought", False)):
                    reasoning_parts.append(part_text)
                    continue
                content_parts.append(part_text)

        content = str(getattr(response, "output_text", "") or "")
        if not content:
            content = "".join(content_parts)

        reasoning_content = "\n".join(part for part in reasoning_parts if part).strip()
        return AssistantMessage(
            content=content,
            tool_calls=tool_calls or None,
            reasoning_content=reasoning_content or None,
        )

    def _emit_assistant_message(
        self,
        *,
        agent_name: str,
        assistant_message: AssistantMessage,
    ) -> None:
        """Mirror one completed assistant message through the response callback."""
        emitted_tool_preview = False

        self._response_callback.on_start(agent_name)
        try:
            if assistant_message.reasoning_content:
                self._response_callback.on_thought_token(
                    assistant_message.reasoning_content
                )
            if assistant_message.content:
                self._response_callback.on_content_token(assistant_message.content)
            for tool_call in assistant_message.tool_calls or []:
                prefix = (
                    f"\nTOOL {tool_call.function.name}: "
                    if assistant_message.content or emitted_tool_preview
                    else f"TOOL {tool_call.function.name}: "
                )
                self._response_callback.on_content_token(prefix)
                self._response_callback.on_content_token(tool_call.function.arguments)
                emitted_tool_preview = True
        finally:
            self._response_callback.on_complete()
