"""
vLLM / OpenAI-Compatible LLM Provider
======================================
Wraps any OpenAI-compatible endpoint (vLLM, ollama, LiteLLM, etc.)
behind the LLMProvider strategy interface.
"""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any

import requests
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
from card_framework.shared.logger_utils import logger


class VLLMProvider(LLMProvider):
    """
    Concrete LLM strategy for OpenAI-compatible APIs (vLLM, etc.).

    Args:
        base_url: The base URL of the API server (for example ``http://host:8000/v1``).
        api_key: API key (use ``"EMPTY"`` for keyless vLLM servers).
        enable_thinking: Whether to request reasoning/thinking chunks from vLLM.
        fallback_to_reasoning_content: Use ``reasoning_content`` when final content is empty.
        fallback_requires_structured_reasoning: Only fallback when reasoning looks structured.
        thinking_extra_body: Optional OpenAI ``extra_body`` payload for reasoning mode.
        request_timeout_seconds: Request timeout applied to model discovery and chat calls.
        response_callback: Optional callback sink for streamed token updates.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        enable_thinking: bool = True,
        fallback_to_reasoning_content: bool = True,
        fallback_requires_structured_reasoning: bool = True,
        thinking_extra_body: dict[str, Any] | None = None,
        request_timeout_seconds: float = 30.0,
        response_callback: LLMResponseCallback | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.fallback_to_reasoning_content = bool(fallback_to_reasoning_content)
        self.fallback_requires_structured_reasoning = bool(
            fallback_requires_structured_reasoning
        )
        self._did_log_reasoning_fallback_warning = False
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.thinking_extra_body = thinking_extra_body or {
            "chat_template_kwargs": {"enable_thinking": True}
        }
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout_seconds,
        )
        self._response_callback: LLMResponseCallback = (
            response_callback or NullLLMResponseCallback()
        )

        # Resolve model id from the server on startup.
        self.model_id = self._fetch_model_id()
        event_bus.publish(
            "system_message",
            f"Connected to VLLM -> model={self.model_id}, url={self.base_url}",
        )

    def set_response_callback(self, response_callback: LLMResponseCallback) -> None:
        """Replace the streaming callback sink at runtime."""
        self._response_callback = response_callback

    def _fetch_model_id(self) -> str:
        """Return the first available model id from the configured endpoint."""
        resp = requests.get(
            f"{self.base_url}/models",
            timeout=self.request_timeout_seconds,
        )
        resp.raise_for_status()
        return str(resp.json()["data"][0]["id"])

    def _maybe_enable_thinking(self, create_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Attach reasoning-mode request params when enabled."""
        if self.enable_thinking:
            create_kwargs["extra_body"] = self.thinking_extra_body
        return create_kwargs

    def _resolve_streamed_content(
        self,
        *,
        content: str,
        reasoning: str,
        operation: str,
    ) -> str:
        """
        Resolve final streamed text, with optional reasoning fallback.

        Some vLLM reasoning models emit only ``reasoning_content`` when thinking is
        enabled. Falling back prevents empty responses that can trigger retry loops.
        """
        normalized_content = content.strip()
        if normalized_content:
            return normalized_content
        if not (self.enable_thinking and self.fallback_to_reasoning_content):
            return ""

        fallback_content = reasoning.strip()
        if not fallback_content:
            return ""
        if (
            self.fallback_requires_structured_reasoning
            and not self._looks_like_structured_output(fallback_content)
        ):
            return ""

        if not self._did_log_reasoning_fallback_warning:
            self._did_log_reasoning_fallback_warning = True
            logger.warning(
                "[VLLMProvider] Empty content in %s; falling back to reasoning_content (chars=%s).",
                operation,
                len(fallback_content),
            )
        return fallback_content

    @staticmethod
    def _looks_like_structured_output(candidate: str) -> bool:
        """Return whether a text payload looks like JSON/XML/fenced structured data."""
        text = candidate.strip()
        if not text:
            return False
        if "```" in text:
            return True
        if "{" in text and "}" in text:
            return True
        if "[" in text and "]" in text:
            return True
        if re.search(r"</?[A-Za-z][^>]*>", text):
            return True
        return False

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

        create_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        create_kwargs = self._maybe_enable_thinking(create_kwargs)

        response = self._client.chat.completions.create(**create_kwargs)

        full_content = ""
        full_thought = ""
        self._response_callback.on_start("Agent")
        try:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    self._response_callback.on_thought_token(delta.reasoning_content)
                    full_thought += delta.reasoning_content

                if hasattr(delta, "content") and delta.content:
                    self._response_callback.on_content_token(delta.content)
                    full_content += delta.content
        finally:
            self._response_callback.on_complete()

        return self._resolve_streamed_content(
            content=full_content,
            reasoning=full_thought,
            operation="generate",
        )

    def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
    ) -> AssistantMessage:
        """Stream chat completion with tool-call aggregation support."""
        normalized_messages = normalize_messages(messages)
        normalized_tools = normalize_tools(tools)

        create_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": normalized_messages,
            "stream": True,
        }
        if normalized_tools is not None:
            create_kwargs["tools"] = normalized_tools
        if tool_choice is not None:
            create_kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        create_kwargs = self._maybe_enable_thinking(create_kwargs)

        response_stream = self._client.chat.completions.create(**create_kwargs)

        full_content = ""
        full_thought = ""
        tool_calls_data: list[dict[str, Any]] = []

        agent_name = infer_agent_name(messages)
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
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )

                        tool_call = tool_calls_data[tc_delta.index]
                        if tc_delta.id:
                            tool_call["id"] = tc_delta.id
                        if tc_delta.function:
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
                id=str(tc.get("id") or f"call_{index}"),
                function=Function(
                    name=str(tc["function"]["name"]),
                    arguments=str(tc["function"]["arguments"]),
                ),
            )
            for index, tc in enumerate(tool_calls_data)
        ]
        resolved_content = full_content
        if not final_tool_calls:
            resolved_content = self._resolve_streamed_content(
                content=full_content,
                reasoning=full_thought,
                operation="chat",
            )

        return AssistantMessage(
            content=resolved_content,
            tool_calls=final_tool_calls if final_tool_calls else None,
            reasoning_content=full_thought.strip() if full_thought.strip() else None,
        )

