"""
vLLM / OpenAI-Compatible LLM Provider
======================================
Wraps any OpenAI-compatible endpoint (vLLM, ollama, LiteLLM, etc.)
behind the LLMProvider strategy interface.
"""

from typing import Any

import requests
from openai import OpenAI

from llm_provider import LLMProvider
from events import event_bus
from ui import ui


class VLLMProvider(LLMProvider):
    """
    Concrete LLM strategy for OpenAI-compatible APIs (vLLM, etc.).

    Args:
        base_url: The base URL of the API server (e.g. ``http://host:8000/v1``).
        api_key:  API key (use ``"EMPTY"`` for keyless vLLM servers).
        enable_thinking: Whether to request reasoning/thinking chunks from vLLM.
        thinking_extra_body: Optional OpenAI `extra_body` payload for reasoning mode.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        enable_thinking: bool = True,
        thinking_extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.enable_thinking = enable_thinking
        self.thinking_extra_body = thinking_extra_body or {
            "chat_template_kwargs": {"enable_thinking": True}
        }
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # Resolve model id from the server on startup
        self.model_id = self._fetch_model_id()
        event_bus.publish(
            "system_message",
            f"Connected to VLLM → model={self.model_id}, url={self.base_url}",
        )

    # ── internal helpers ──────────────────────────────────────────────────

    def _fetch_model_id(self) -> str:
        resp = requests.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]

    def _maybe_enable_thinking(
        self,
        create_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """Attach reasoning-mode request params when enabled."""
        if self.enable_thinking:
            create_kwargs["extra_body"] = self.thinking_extra_body
        return create_kwargs

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

        create_kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        create_kwargs = self._maybe_enable_thinking(create_kwargs)

        response = self._client.chat.completions.create(**create_kwargs)

        full_content = ""
        full_thought = ""

        with ui.live_agent_message("Agent") as live_msg:
            for chunk in response:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Handle Reasoning/Thinking
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    live_msg.update_thought(delta.reasoning_content)
                    full_thought += delta.reasoning_content

                # Handle Content
                if hasattr(delta, "content") and delta.content:
                    live_msg.update_content(delta.content)
                    full_content += delta.content

        return full_content.strip()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
    ):
        """
        Chat completion with streaming support and tool call aggregation.
        """
        create_kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=True,
        )
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        create_kwargs = self._maybe_enable_thinking(create_kwargs)

        response_stream = self._client.chat.completions.create(**create_kwargs)

        full_content = ""
        full_thought = ""
        tool_calls_data = []

        # Identify agent name for UI
        agent_name = "Agent"
        for msg in reversed(messages):
            if msg["role"] == "system":
                if "Summarizer" in msg["content"]:
                    agent_name = "Summarizer"
                elif "Critic" in msg["content"]:
                    agent_name = "Critic"
                break

        with ui.live_agent_message(agent_name) as live_msg:
            for chunk in response_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Handle Reasoning/Thinking (Qwen3 specific field)
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    live_msg.update_thought(delta.reasoning_content)
                    full_thought += delta.reasoning_content

                # Handle Content
                if hasattr(delta, "content") and delta.content:
                    live_msg.update_content(delta.content)
                    full_content += delta.content

                # Handle Tool Calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        # Ensure we have a placeholder for this tool call index
                        while len(tool_calls_data) <= tc_delta.index:
                            tool_calls_data.append(
                                {
                                    "id": None,
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )

                        tc = tool_calls_data[tc_delta.index]
                        if tc_delta.id:
                            tc["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                tc["function"]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tc["function"]["arguments"] += (
                                    tc_delta.function.arguments
                                )

        # Reconstruct OpenAI-like response object for downstream logic
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
            def __init__(
                self, content: str, tool_calls_info: list[dict], reasoning_content=None
            ):
                self.content = content
                self.role = "assistant"
                self.reasoning_content = reasoning_content
                self.tool_calls = []
                if tool_calls_info:
                    for tc in tool_calls_info:
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
                if self.reasoning_content:
                    d["reasoning_content"] = self.reasoning_content
                return d

        return AssistantMessage(full_content, tool_calls_data, full_thought)
