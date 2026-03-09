"""
Google Gen AI LLM Provider
==========================
Uses the official google-genai SDK.
"""

import sys
from collections.abc import Sequence

from google import genai
from google.genai import types

from card_framework.shared.llm_provider import LLMProvider, MessageInput, ToolChoice, ToolInput, normalize_messages
from card_framework.shared.events import event_bus


class GoogleGenAIProvider(LLMProvider):
    """
    Concrete LLM strategy for Google Gen AI API (Gemini).

    Args:
        api_key: Gemini API key.
        model: Model name to use (e.g., 'gemini-2.5-pro').
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro",
    ) -> None:
        self.api_key = api_key
        self.model = model
        self._client = genai.Client(api_key=self.api_key)

        event_bus.publish("system_message", f"Connected to Google Gen AI -> model={self.model}")

    # â”€â”€ LLMProvider interface â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        contents = [
            system_prompt,
            user_prompt,
        ]

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(include_thoughts=True)
        )
        if max_tokens is not None:
            config.max_output_tokens = max_tokens

        response_stream = self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        )

        full_content = ""
        full_reasoning = ""

        # Stream chunks to console
        # The schema of response chunks includes choices/candidates.
        for chunk in response_stream:
            # According to SDK: chunk.candidates[0].content.parts[0].thought
            if chunk.candidates:
                for candidate in chunk.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.thought:
                                # Print reasoning to stderr or with a distinct color prefix
                                sys.stdout.write(f"\\033[90m{part.text}\\033[0m")
                                sys.stdout.flush()
                                full_reasoning += part.text
                            elif part.text:
                                sys.stdout.write(part.text)
                                sys.stdout.flush()
                                full_content += part.text

        sys.stdout.write("\\n")
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
        Chat completion mapping OpenAI-compatible message lists to Google Gen AI.
        """
        del tools, tool_choice
        normalized_messages = normalize_messages(messages)
        contents = []
        system_instructions = []

        for msg in normalized_messages:
            role = msg.get("role", "user")
            content = msg.get("content") or " "

            if role == "system":
                system_instructions.append(content)
            elif role in ("user", "assistant", "model"):
                genai_role = "model" if role == "assistant" else role
                contents.append(
                    types.Content(
                        role=genai_role, parts=[types.Part.from_text(text=content)]
                    )
                )

        config_kwargs = {}
        if system_instructions:
            config_kwargs["system_instruction"] = "\n".join(system_instructions)
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        config = types.GenerateContentConfig(**config_kwargs) if config_kwargs else None

        # Call the appropriate client method for generation (not strictly streaming here)
        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        # Mock an OpenAI response compatible message to keep internal A2A logic happy
        class AssistantMessage:
            def __init__(self, content):
                self.content = content
                self.role = "assistant"
                self.tool_calls = None

            def model_dump(self):
                return {"role": self.role, "content": self.content}

        return AssistantMessage(response.text)

