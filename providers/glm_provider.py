"""
GLM LLM Provider via ZAI SDK
============================
Uses the official zai-sdk.
"""

import sys

from zai import ZaiClient

from llm_provider import LLMProvider
from ui import ui


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

        ui.print_system(f"Connected to GLM (ZAI) -> model={self.model}")

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

        # Stream chunks to console
        # According to ZAI SDK: chunk.choices[0].delta.reasoning_content and chunk.choices[0].delta.content
        for chunk in response_stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                # Print reasoning in gray
                sys.stdout.write(f"\033[90m{delta.reasoning_content}\033[0m")
                sys.stdout.flush()
                full_reasoning += delta.reasoning_content

            if hasattr(delta, "content") and delta.content:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                full_content += delta.content

        sys.stdout.write("\n")
        sys.stdout.flush()
        return full_content.strip()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
    ):
        """
        Chat completion mapping OpenAI-compatible message lists to GLM.
        Now supports streaming for real-time UI updates.
        """
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "stream": True,  # Enable streaming
            "thinking": {"type": "enabled"},
        }
        if tools:
            # Note: GLM-4.6 might have specific ways to handle tools with thinking.
            # We'll pass them through and see.
            create_kwargs["tools"] = tools
        if tool_choice:
            create_kwargs["tool_choice"] = tool_choice
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response_stream = self._client.chat.completions.create(**create_kwargs)

        full_content = ""
        full_thought = ""
        tool_calls = []

        # Find agent name from messages or context
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

                # Handle Thinking/Reasoning
                if hasattr(delta, "reasoning_content") and delta.reasoning_content:
                    live_msg.update_thought(delta.reasoning_content)
                    full_thought += delta.reasoning_content

                # Handle Content
                if hasattr(delta, "content") and delta.content:
                    live_msg.update_content(delta.content)
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
                        if tc_delta.function.name:
                            tc["function"]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc["function"]["arguments"] += tc_delta.function.arguments

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
