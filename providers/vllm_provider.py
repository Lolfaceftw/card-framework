"""
vLLM / OpenAI-Compatible LLM Provider
======================================
Wraps any OpenAI-compatible endpoint (vLLM, ollama, LiteLLM, etc.)
behind the LLMProvider strategy interface.
"""

import sys

import requests
from openai import OpenAI

from llm_provider import LLMProvider
from ui import ui


class VLLMProvider(LLMProvider):
    """
    Concrete LLM strategy for OpenAI-compatible APIs (vLLM, etc.).

    Args:
        base_url: The base URL of the API server (e.g. ``http://host:8000/v1``).
        api_key:  API key (use ``"EMPTY"`` for keyless vLLM servers).
    """

    def __init__(self, base_url: str, api_key: str = "EMPTY") -> None:
        self.base_url = base_url
        self.api_key = api_key
        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

        # Resolve model id from the server on startup
        self.model_id = self._fetch_model_id()
        ui.print_system(
            f"Connected to VLLM → model={self.model_id}, url={self.base_url}"
        )

    # ── internal helpers ──────────────────────────────────────────────────

    def _fetch_model_id(self) -> str:
        resp = requests.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]

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
            model=self.model_id,
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**create_kwargs)

        full_content = ""
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content is not None:
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
        create_kwargs: dict = dict(
            model=self.model_id,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            stream=False,
        )
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens

        response = self._client.chat.completions.create(**create_kwargs)
        return response.choices[0].message
