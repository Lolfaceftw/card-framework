"""
DeepSeek LLM Provider
======================
Uses the official OpenAI Python client pointing to DeepSeek API.
"""

import sys

from openai import OpenAI

from llm_provider import LLMProvider
from events import event_bus


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

        event_bus.publish("system_message", 
            f"Connected to DeepSeek → model={self.model}, url={self.base_url}"
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

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
    ):
        """
        Chat completion with support for tools and latest DeepSeek features.
        Note: deepseek-reasoner does not currently support tools.
        """
        create_kwargs: dict = dict(
            model=self.model,
            messages=messages,
            stream=False,
        )
        if tools is not None and "reasoner" not in self.model:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = tool_choice

        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**create_kwargs)
        return response.choices[0].message
