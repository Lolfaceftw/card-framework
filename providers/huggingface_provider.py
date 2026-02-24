"""
Hugging Face LLM Provider
==========================
Uses the official OpenAI Python client pointing to the Hugging Face Serverless Inference API.
"""

import sys

from openai import OpenAI

from llm_provider import LLMProvider
from events import event_bus


class HuggingfaceProvider(LLMProvider):
    """
    Concrete LLM strategy for Hugging Face Inference API.

    Args:
        api_key: Hugging Face API token (hf_...).
        model: Model name to use (e.g. 'Team-ACE/ToolACE-8B').
        base_url: The base URL of the Hugging Face API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "Team-ACE/ToolACE-8B",
        base_url: str = "https://router.huggingface.co/hf-inference/v1/",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        event_bus.publish("system_message", 
            f"Connected to Hugging Face → model={self.model}, url={self.base_url}"
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
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
    ):
        """
        Chat completion with support for tools via Hugging Face Inference API.
        """
        create_kwargs: dict = dict(
            model=self.model,
            messages=messages,
            stream=False,
        )
        if tools is not None:
            create_kwargs["tools"] = tools
            create_kwargs["tool_choice"] = tool_choice

        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**create_kwargs)
        return response.choices[0].message
