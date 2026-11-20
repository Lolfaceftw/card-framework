"""
GLM LLM Provider via ZAI SDK
============================
Uses the official zai-sdk.
"""

from collections.abc import Sequence

from zai import ZaiClient

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


class GLMProvider(LLMProvider):
    """
    Concrete LLM strategy for GLM API (via ZAI SDK).

    Args:
        api_key: ZAI API key.
        model: Model name to use (e.g., 'glm-4.6').
    """

    def __init__(
        self,
        api_key: str,
        model: str = "glm-4.6",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = ZaiClient(api_key=self.api_key)
        self._response_callback: LLMResponseCallback = NullLLMResponseCallback()

        event_bus.publish(
            "system_message", f"Connected to GLM (ZAI) -> model={self.model}"
        )

    def set_response_callback(self, response_callback: LLMResponseCallback) -> None:
        """Replace the streaming callback sink at runtime."""
        self._response_callback = response_callback

    # â”€â”€ LLMProvider interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "thinking": {"type": "enabled"},
        }
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response_stream = self._client.chat.completions.create(**create_kwargs)

        full_content = ""
        full_reasoning = ""
        self._response_callback.on_start(infer_agent_name_from_text(system_prompt))

        try:
            for chunk in response_stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta

                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    self._response_callback.on_thought_token(delta.reasoning_content)
                    full_reasoning += delta.reasoning_content

                if hasattr(delta, "content") and delta.content:
                    self._response_callback.on_content_token(delta.content)
                    full_content += delta.content
        finally:
            self._response_callback.on_complete()

        return full_content.strip()

    def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
    ):
        """
        Chat completion mapping OpenAI-compatible message lists to GLM.
        Now supports streaming for real-time UI updates.
        """
        normalized_messages = normalize_messages(messages)
        normalized_tools = normalize_tools(tools)
        create_kwargs = {
            "model": self.model,
            "messages": normalized_messages,
            "stream": True,  # Enable streaming
            "thinking": {"type": "enabled"},
        }
        if normalized_tools:
            # Note: GLM-4.6 might have specific ways to handle tools with thinking.
            # We'll pass them through and see.
            create_kwargs["tools"] = normalized_tools
        if tool_choice:
            create_kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response_stream = self._client.chat.completions.create(**create_kwargs)

        full_content = ""
        full_thought = ""
        tool_calls = []

        # Find agent name from messages or context
        agent_name = infer_agent_name(messages)

        self._response_callback.on_start(agent_name)
        try:
            for chunk in response_stream:
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta

                # Handle Thinking/Reasoning
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    self._response_callback.on_thought_token(delta.reasoning_content)
                    full_thought += delta.reasoning_content

                # Handle Content
                if hasattr(delta, "content") and delta.content:
                    self._response_callback.on_content_token(delta.content)
                    full_content += delta.content

                # Handle Tool Calls (GLM might stream these differently, but we'll try to collect them)
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        if len(tool_calls) <= tc_delta.index:
                            tool_calls.append(
                                {
                                    "id": tc_delta.id,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )

                        tc = tool_calls[tc_delta.index]
                        function_delta = getattr(tc_delta, "function", None)
                        if function_delta is not None and function_delta.name:
                            tc["function"]["name"] += function_delta.name
                        if function_delta is not None and function_delta.arguments:
                            tc["function"]["arguments"] += function_delta.arguments
        finally:
            self._response_callback.on_complete()

        # Mock an OpenAI response compatible message to keep internal A2A logic happy
        class Function:
            def __init__(self, name, arguments):
                self.name = name
                self.arguments = arguments

        class ToolCall:
            def __init__(self, id, name, arguments):
                self.id = id
                self.type = "function"
                self.function = Function(name, arguments)

            def model_dump(self):
                return {
                    "id": self.id,
                    "type": self.type,
                    "function": {
                        "name": self.function.name,
                        "arguments": self.function.arguments,
                    },
                }

        class AssistantMessage:
            def __init__(self, content, tool_calls_data, reasoning=None):
                self.content = content
                self.role = "assistant"
                self.reasoning = reasoning
                self.tool_calls = []
                if tool_calls_data:
                    for tc in tool_calls_data:
                        self.tool_calls.append(
                            ToolCall(
                                tc["id"],
                                tc["function"]["name"],
                                tc["function"]["arguments"],
                            )
                        )
                else:
                    self.tool_calls = None

            def model_dump(self):
                d = {"role": self.role, "content": self.content}
                if self.tool_calls:
                    d["tool_calls"] = [tc.model_dump() for tc in self.tool_calls]
                if self.reasoning:
                    d["reasoning"] = self.reasoning
                return d

        return AssistantMessage(full_content, tool_calls, full_thought)
