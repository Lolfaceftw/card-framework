"""Google Gen AI provider with streaming thought and manual tool support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from google import genai
from google.genai import types

from card_framework.agents.dtos import AssistantMessage, Function, ToolCall
from card_framework.shared.events import event_bus
from card_framework.shared.llm_provider import (
    LLMProvider,
    LLMResponseCallback,
    MessageInput,
    NullLLMResponseCallback,
    StreamingResponseAbort,
    ToolChoice,
    ToolInput,
    infer_agent_name,
    infer_agent_name_from_text,
    normalize_messages,
    normalize_tools,
)
from card_framework.shared.logger_utils import logger


def _coerce_jsonable(value: Any) -> Any:
    """Convert SDK payloads into JSON-serializable builtins."""
    if hasattr(value, "model_dump"):
        return _coerce_jsonable(value.model_dump())
    if isinstance(value, Mapping):
        return {
            str(key): _coerce_jsonable(raw_value)
            for key, raw_value in value.items()
        }
    if isinstance(value, list | tuple):
        return [_coerce_jsonable(item) for item in value]
    return value


def _compact_json(value: Any) -> str:
    """Return a deterministic compact JSON string."""
    return json.dumps(
        _coerce_jsonable(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _parse_json_object(raw_value: str) -> dict[str, Any]:
    """Parse one JSON string into a dictionary-shaped payload."""
    stripped = str(raw_value).strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {"_raw": raw_value}
    if isinstance(parsed, dict):
        return _coerce_jsonable(parsed)
    return {"value": _coerce_jsonable(parsed)}


class GoogleGenAIProvider(LLMProvider):
    """Call Gemini through the official google-genai SDK."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-pro",
        *,
        enable_thinking: bool = True,
        request_timeout_seconds: float = 60.0,
        response_callback: LLMResponseCallback | None = None,
    ) -> None:
        """Initialize the Gemini provider."""
        if not str(api_key).strip():
            raise ValueError("GoogleGenAIProvider requires a non-empty API key.")

        self.api_key = str(api_key).strip()
        self.model = str(model).strip() or "gemini-2.5-pro"
        self.enable_thinking = bool(enable_thinking)
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self._client = genai.Client(api_key=self.api_key)
        self._response_callback: LLMResponseCallback = (
            response_callback or NullLLMResponseCallback()
        )
        self.last_response_details: dict[str, Any] = {}
        self._assistant_contents_by_tool_key: dict[tuple[str, ...], types.Content] = {}

        event_bus.publish(
            "system_message",
            f"Connected to Google Gen AI -> model={self.model}",
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
        """Generate one streamed non-tool response."""
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_prompt if user_prompt else " ")],
            )
        ]
        config = self._build_generate_config(
            system_instruction=system_prompt,
            max_tokens=max_tokens,
        )

        try:
            response_stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
            full_content = ""
            full_reasoning = ""
            self._response_callback.on_start(infer_agent_name_from_text(system_prompt))
            try:
                for chunk in response_stream:
                    chunk_content, chunk_reasoning, _ = self._extract_chunk_payload(
                        chunk=chunk
                    )
                    if chunk_reasoning:
                        self._response_callback.on_thought_token(chunk_reasoning)
                        full_reasoning += chunk_reasoning
                    if chunk_content:
                        self._response_callback.on_content_token(chunk_content)
                        full_content += chunk_content
            finally:
                self._response_callback.on_complete()

            self.last_response_details = {
                "provider": "google_genai",
                "request": {
                    "model": self.model,
                    "contents": [_coerce_jsonable(content) for content in contents],
                    "config": _coerce_jsonable(config),
                },
                "response": {
                    "content": full_content,
                    "reasoning_content": full_reasoning or None,
                },
            }
            return full_content.strip()
        except StreamingResponseAbort:
            raise
        except Exception as exc:
            logger.warning(
                "[GoogleGenAIProvider] Streaming generate failed; retrying non-streaming: %s",
                exc,
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            assistant_message = self._assistant_message_from_response(response)
            self.last_response_details = {
                "provider": "google_genai",
                "request": {
                    "model": self.model,
                    "contents": [_coerce_jsonable(content) for content in contents],
                    "config": _coerce_jsonable(config),
                },
                "response": {
                    "content": assistant_message.content,
                    "reasoning_content": assistant_message.reasoning_content,
                },
            }
            return assistant_message.content.strip()

    def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
    ) -> AssistantMessage:
        """Generate one Gemini chat response with manual tool-calling support."""
        normalized_messages = normalize_messages(messages)
        self._reset_tool_call_cache_if_new_conversation(normalized_messages)
        contents, system_instruction = self._normalize_messages(normalized_messages)
        normalized_tools = self._normalize_tools(tools)
        config = self._build_generate_config(
            system_instruction=system_instruction,
            max_tokens=max_tokens,
            tools=normalized_tools,
            tool_choice=tool_choice,
        )
        agent_name = infer_agent_name(messages)
        use_non_streaming = self._requires_exact_tool_replay(
            normalized_messages=normalized_messages,
            normalized_tools=normalized_tools,
        )

        if use_non_streaming:
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            assistant_message = self._assistant_message_from_response(response)
            self._cache_assistant_tool_content(
                response=response,
                assistant_message=assistant_message,
            )
            self._emit_assistant_message(
                agent_name=agent_name,
                assistant_message=assistant_message,
            )
            self.last_response_details = {
                "provider": "google_genai",
                "request": {
                    "model": self.model,
                    "contents": [_coerce_jsonable(content) for content in contents],
                    "config": _coerce_jsonable(config),
                },
                "response": {
                    **assistant_message.model_dump(),
                    "raw_response": _coerce_jsonable(response),
                },
            }
            return assistant_message

        try:
            response_stream = self._client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )
            assistant_message = self._stream_chat_response(
                response_stream=response_stream,
                agent_name=agent_name,
            )
            self.last_response_details = {
                "provider": "google_genai",
                "request": {
                    "model": self.model,
                    "contents": [_coerce_jsonable(content) for content in contents],
                    "config": _coerce_jsonable(config),
                },
                "response": assistant_message.model_dump(),
            }
            return assistant_message
        except StreamingResponseAbort:
            raise
        except Exception as exc:
            logger.warning(
                "[GoogleGenAIProvider] Streaming chat failed; retrying non-streaming: %s",
                exc,
            )
            response = self._client.models.generate_content(
                model=self.model,
                contents=contents,
                config=config,
            )
            assistant_message = self._assistant_message_from_response(response)
            self._emit_assistant_message(
                agent_name=agent_name,
                assistant_message=assistant_message,
            )
            self.last_response_details = {
                "provider": "google_genai",
                "request": {
                    "model": self.model,
                    "contents": [_coerce_jsonable(content) for content in contents],
                    "config": _coerce_jsonable(config),
                },
                "response": {
                    **assistant_message.model_dump(),
                    "raw_response": _coerce_jsonable(response),
                },
            }
            return assistant_message

    def _build_generate_config(
        self,
        *,
        system_instruction: str,
        max_tokens: int | None,
        tools: list[types.Tool] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> types.GenerateContentConfig:
        """Build a Gemini GenerateContentConfig payload."""
        config_kwargs: dict[str, Any] = {}
        if system_instruction.strip():
            config_kwargs["system_instruction"] = system_instruction.strip()
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens
        if self.enable_thinking:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                include_thoughts=True
            )
        if tools:
            config_kwargs["tools"] = tools
            config_kwargs["automatic_function_calling"] = (
                types.AutomaticFunctionCallingConfig(disable=True)
            )
            tool_config = self._normalize_tool_choice(tool_choice)
            if tool_config is not None:
                config_kwargs["tool_config"] = tool_config
        return types.GenerateContentConfig(**config_kwargs)

    def _normalize_messages(
        self,
        normalized_messages: list[dict[str, object]],
    ) -> tuple[list[types.Content], str]:
        """Convert OpenAI-style messages into Gemini contents plus system text."""
        system_parts: list[str] = []
        contents: list[types.Content] = []

        for message in normalized_messages:
            role = str(message.get("role") or "user").strip().lower()
            if role in {"system", "developer"}:
                content = str(message.get("content") or "").strip()
                if content:
                    system_parts.append(content)
                continue

            if role == "tool":
                contents.append(self._normalize_tool_message(message))
                continue

            if role in {"assistant", "model"}:
                contents.append(self._normalize_assistant_message(message))
                continue

            contents.append(self._normalize_user_message(message))

        return contents, "\n\n".join(system_parts).strip()

    def _normalize_user_message(self, message: Mapping[str, object]) -> types.Content:
        """Convert one user-style message into Gemini content."""
        content = str(message.get("content") or "")
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=content if content else " ")],
        )

    def _normalize_assistant_message(
        self,
        message: Mapping[str, object],
    ) -> types.Content:
        """Convert one assistant-style message into Gemini model content."""
        parts: list[types.Part] = []
        content = str(message.get("content") or "")
        raw_tool_calls = message.get("tool_calls")
        cached_content = self._cached_assistant_content_for_message(
            content=content,
            raw_tool_calls=raw_tool_calls,
        )
        if cached_content is not None:
            return cached_content

        if content:
            parts.append(types.Part.from_text(text=content))

        if isinstance(raw_tool_calls, Sequence) and not isinstance(
            raw_tool_calls,
            str | bytes,
        ):
            for index, tool_call in enumerate(raw_tool_calls):
                if not isinstance(tool_call, Mapping):
                    continue
                function_payload = tool_call.get("function", {})
                function = (
                    function_payload if isinstance(function_payload, Mapping) else {}
                )
                name = str(function.get("name") or "").strip()
                if not name:
                    continue
                arguments = self._parse_tool_call_arguments(
                    function.get("arguments")
                )
                tool_id = str(tool_call.get("id") or f"call_{index}").strip()
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id=tool_id,
                            name=name,
                            args=arguments,
                        )
                    )
                )

        if not parts:
            parts.append(types.Part.from_text(text=" "))
        return types.Content(role="model", parts=parts)

    def _normalize_tool_message(self, message: Mapping[str, object]) -> types.Content:
        """Convert one tool-result message into Gemini function-response content."""
        tool_name = str(message.get("name") or "").strip()
        tool_call_id = str(message.get("tool_call_id") or "").strip()
        if not tool_name:
            tool_name = tool_call_id or "tool"
        raw_content = str(message.get("content") or "")
        response_payload = _parse_json_object(raw_content)
        return types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=tool_call_id or None,
                        name=tool_name,
                        response=response_payload,
                    )
                )
            ],
        )

    def _normalize_tools(
        self,
        tools: Sequence[ToolInput] | None,
    ) -> list[types.Tool] | None:
        """Convert shared OpenAI-style tool schemas into Gemini tools."""
        normalized_tools = normalize_tools(tools)
        if not normalized_tools:
            return None

        function_declarations: list[types.FunctionDeclaration] = []
        for tool in normalized_tools:
            function_payload = tool.get("function", {})
            function = function_payload if isinstance(function_payload, Mapping) else {}
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            parameters = _coerce_jsonable(function.get("parameters") or {})
            function_declarations.append(
                types.FunctionDeclaration(
                    name=name,
                    description=str(function.get("description") or ""),
                    parameters_json_schema=parameters,
                )
            )

        if not function_declarations:
            return None
        return [types.Tool(function_declarations=function_declarations)]

    def _normalize_tool_choice(
        self,
        tool_choice: ToolChoice | None,
    ) -> types.ToolConfig | None:
        """Map shared tool-choice values onto Gemini tool-calling config."""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            normalized = tool_choice.strip().lower()
            if normalized == "auto":
                return types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                )
            if normalized == "required":
                return types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="ANY")
                )
            return None
        if not isinstance(tool_choice, Mapping):
            return None

        type_name = str(tool_choice.get("type") or "").strip().lower()
        if type_name == "auto":
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO")
            )
        if type_name == "function":
            function_payload = tool_choice.get("function", {})
            function = (
                function_payload if isinstance(function_payload, Mapping) else {}
            )
            function_name = str(function.get("name") or "").strip()
            if not function_name:
                return None
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=[function_name],
                )
            )
        if type_name in {"required", "any"}:
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            )
        if type_name == "none":
            return types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="NONE")
            )
        return None

    def _requires_exact_tool_replay(
        self,
        *,
        normalized_messages: Sequence[Mapping[str, object]],
        normalized_tools: list[types.Tool] | None,
    ) -> bool:
        """Use non-streaming when Gemini tool-call history must be replayed exactly."""
        if normalized_tools:
            return True
        return any(
            self._message_contains_tool_history(message)
            for message in normalized_messages
        )

    def _reset_tool_call_cache_if_new_conversation(
        self,
        normalized_messages: Sequence[Mapping[str, object]],
    ) -> None:
        """Avoid leaking cached Gemini tool signatures across unrelated chats."""
        if any(
            self._message_contains_tool_history(message)
            for message in normalized_messages
        ):
            return
        self._assistant_contents_by_tool_key.clear()

    def _message_contains_tool_history(
        self,
        message: Mapping[str, object],
    ) -> bool:
        """Detect assistant tool calls or tool responses in normalized history."""
        role = str(message.get("role") or "").strip().lower()
        if role == "tool":
            return True
        raw_tool_calls = message.get("tool_calls")
        return role in {"assistant", "model"} and isinstance(
            raw_tool_calls,
            Sequence,
        ) and not isinstance(raw_tool_calls, str | bytes)

    def _cached_assistant_content_for_message(
        self,
        *,
        content: str,
        raw_tool_calls: object,
    ) -> types.Content | None:
        """Return cached Gemini assistant content with preserved thought signatures."""
        cache_key = self._assistant_tool_cache_key(
            content=content,
            raw_tool_calls=raw_tool_calls,
        )
        if cache_key is None:
            return None
        cached_content = self._assistant_contents_by_tool_key.get(cache_key)
        if cached_content is None:
            return None
        return types.Content.model_validate(cached_content.model_dump())

    def _assistant_tool_cache_key(
        self,
        *,
        content: str,
        raw_tool_calls: object,
    ) -> tuple[str, ...] | None:
        """Build a stable cache key from one assistant tool-call turn."""
        if not isinstance(raw_tool_calls, Sequence) or isinstance(
            raw_tool_calls,
            str | bytes,
        ):
            return None

        tool_signatures: list[str] = [content]
        for index, tool_call in enumerate(raw_tool_calls):
            if not isinstance(tool_call, Mapping):
                continue
            function_payload = tool_call.get("function", {})
            function = (
                function_payload if isinstance(function_payload, Mapping) else {}
            )
            name = str(function.get("name") or "").strip()
            if not name:
                continue
            tool_id = str(tool_call.get("id") or f"call_{index}").strip()
            arguments = self._compact_tool_call_arguments(
                function.get("arguments")
            )
            tool_signatures.append(f"{tool_id}|{name}|{arguments}")

        if len(tool_signatures) == 1:
            return None
        return tuple(tool_signatures)

    def _parse_tool_call_arguments(self, raw_value: object) -> dict[str, Any]:
        """Normalize tool-call arguments into the dict shape Gemini expects."""
        if raw_value is None:
            return {}
        if isinstance(raw_value, Mapping):
            return _coerce_jsonable(raw_value)
        return _parse_json_object(str(raw_value))

    def _compact_tool_call_arguments(self, raw_value: object) -> str:
        """Serialize tool-call arguments into a stable compact JSON string."""
        return _compact_json(self._parse_tool_call_arguments(raw_value))

    def _cache_assistant_tool_content(
        self,
        *,
        response: Any,
        assistant_message: AssistantMessage,
    ) -> None:
        """Remember the exact Gemini model content for later tool-turn replay."""
        if not assistant_message.tool_calls:
            return

        response_content = self._primary_response_content(response)
        if response_content is None:
            return

        cache_key = self._assistant_tool_cache_key(
            content=assistant_message.content,
            raw_tool_calls=[tool_call.model_dump() for tool_call in assistant_message.tool_calls],
        )
        if cache_key is None:
            return
        self._assistant_contents_by_tool_key[cache_key] = response_content

    def _primary_response_content(self, response: Any) -> types.Content | None:
        """Extract the first response content block as a standalone model object."""
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None
        first_candidate = candidates[0]
        content = getattr(first_candidate, "content", None)
        if content is None:
            return None
        return types.Content.model_validate(content.model_dump())

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
                self._response_callback.on_content_token(
                    tool_call.function.arguments
                )
                emitted_tool_preview = True
        finally:
            self._response_callback.on_complete()

    def _stream_chat_response(
        self,
        *,
        response_stream: Any,
        agent_name: str,
    ) -> AssistantMessage:
        """Consume a streamed Gemini chat response."""
        full_content = ""
        full_reasoning = ""
        tool_calls: list[ToolCall] = []
        seen_tool_signatures: set[tuple[str, str, str]] = set()
        emitted_tool_preview = False

        self._response_callback.on_start(agent_name)
        try:
            for chunk in response_stream:
                chunk_content, chunk_reasoning, chunk_tool_calls = (
                    self._extract_chunk_payload(chunk=chunk)
                )
                if chunk_reasoning:
                    self._response_callback.on_thought_token(chunk_reasoning)
                    full_reasoning += chunk_reasoning
                if chunk_content:
                    self._response_callback.on_content_token(chunk_content)
                    full_content += chunk_content
                for tool_call in chunk_tool_calls:
                    signature = (
                        tool_call.id,
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    if signature in seen_tool_signatures:
                        continue
                    seen_tool_signatures.add(signature)
                    prefix = (
                        f"\nTOOL {tool_call.function.name}: "
                        if full_content or emitted_tool_preview
                        else f"TOOL {tool_call.function.name}: "
                    )
                    self._response_callback.on_content_token(prefix)
                    self._response_callback.on_content_token(
                        tool_call.function.arguments
                    )
                    emitted_tool_preview = True
                    tool_calls.append(tool_call)
        finally:
            self._response_callback.on_complete()

        return AssistantMessage(
            content=full_content,
            tool_calls=tool_calls or None,
            reasoning_content=full_reasoning if full_reasoning else None,
        )

    def _assistant_message_from_response(self, response: Any) -> AssistantMessage:
        """Convert one non-stream Gemini response into an assistant message."""
        full_content = str(getattr(response, "text", "") or "")
        full_reasoning = ""
        tool_calls: list[ToolCall] = []
        seen_tool_signatures: set[tuple[str, str, str]] = set()

        candidates = getattr(response, "candidates", None)
        if candidates:
            for candidate in candidates:
                content = getattr(candidate, "content", None)
                parts = getattr(content, "parts", None)
                if not parts:
                    continue
                for part in parts:
                    if getattr(part, "thought", False) and getattr(part, "text", None):
                        full_reasoning += str(part.text)
                    function_call = getattr(part, "function_call", None)
                    if function_call is None:
                        continue
                    tool_call = self._tool_call_from_function_call(
                        function_call=function_call,
                        default_id=f"call_{len(tool_calls)}",
                    )
                    signature = (
                        tool_call.id,
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    if signature in seen_tool_signatures:
                        continue
                    seen_tool_signatures.add(signature)
                    tool_calls.append(tool_call)

        return AssistantMessage(
            content=full_content,
            tool_calls=tool_calls or None,
            reasoning_content=full_reasoning if full_reasoning else None,
        )

    def _extract_chunk_payload(
        self,
        *,
        chunk: Any,
    ) -> tuple[str, str, list[ToolCall]]:
        """Extract text, thought text, and tool calls from one Gemini stream chunk."""
        content_tokens: list[str] = []
        reasoning_tokens: list[str] = []
        tool_calls: list[ToolCall] = []

        candidates = getattr(chunk, "candidates", None)
        if not candidates:
            return "", "", []

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                text = getattr(part, "text", None)
                if getattr(part, "thought", False) and text:
                    reasoning_tokens.append(str(text))
                    continue
                if text:
                    content_tokens.append(str(text))
                    continue
                function_call = getattr(part, "function_call", None)
                if function_call is not None:
                    tool_calls.append(
                        self._tool_call_from_function_call(
                            function_call=function_call,
                            default_id=f"call_{len(tool_calls)}",
                        )
                    )

        return (
            "".join(content_tokens),
            "".join(reasoning_tokens),
            tool_calls,
        )

    def _tool_call_from_function_call(
        self,
        *,
        function_call: Any,
        default_id: str,
    ) -> ToolCall:
        """Convert one Gemini function call payload into the shared DTO."""
        raw_args = _coerce_jsonable(getattr(function_call, "args", None) or {})
        return ToolCall(
            id=str(getattr(function_call, "id", None) or default_id),
            function=Function(
                name=str(getattr(function_call, "name", None) or "unnamed_tool"),
                arguments=_compact_json(raw_args),
            ),
        )
