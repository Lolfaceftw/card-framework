"""
Nanbeige LLM Provider
=====================
Uses the OpenAI Python client pointing to the Hugging Face Serverless Inference API,
specifically tailored for Nanbeige Thinking models.
"""

import sys
from collections.abc import Sequence

from openai import OpenAI

from card_framework.shared.llm_provider import (
    LLMProvider,
    MessageInput,
    ToolChoice,
    ToolInput,
    normalize_messages,
    normalize_tools,
)
from card_framework.shared.events import event_bus


class NanbeigeProvider(LLMProvider):
    """
    Concrete LLM strategy for Nanbeige API.

    Args:
        api_key: Hugging Face API token.
        model: Model name to use (e.g. 'Nanbeige/Nanbeige4-3B-Thinking-2511').
        base_url: The base URL of the Hugging Face API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "Nanbeige/Nanbeige4-3B-Thinking-2511",
        base_url: str = "https://router.huggingface.co/hf-inference/v1/",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        event_bus.publish("system_message", 
            f"Connected to Nanbeige â†’ model={self.model}, url={self.base_url}"
        )

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

        create_kwargs: dict = dict(
            model=self.model,
            messages=messages,
            stream=True,
        )
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**create_kwargs)

        full_content = ""

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            if delta.content:
                sys.stdout.write(delta.content)
                sys.stdout.flush()
                full_content += delta.content

        sys.stdout.write("\n")
        sys.stdout.flush()
        return full_content.strip()

    def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
    ):
        """
        Chat completion with support for tools via Hugging Face Inference API.
        """
        normalized_messages = normalize_messages(messages)
        normalized_tools = normalize_tools(tools)
        create_kwargs: dict = dict(
            model=self.model,
            messages=normalized_messages,
            stream=False,
        )
        if normalized_tools is not None:
            create_kwargs["tools"] = normalized_tools
            create_kwargs["tool_choice"] = tool_choice

        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**create_kwargs)
        return response.choices[0].message

