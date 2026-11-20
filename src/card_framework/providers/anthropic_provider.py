"""Use Anthropic Messages API through the shared LLM provider interface."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

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
    normalize_messages,
    normalize_tools,
)
from card_framework.shared.logger_utils import logger

try:
    import anthropic
except ImportError:  # pragma: no cover - optional dependency
    anthropic = None

_DEFAULT_MAX_TOKENS = 16_000
_DEFAULT_THINKING = {"type": "adaptive", "display": "summarized"}
_DEFAULT_OUTPUT_CONFIG = {"effort": "medium"}
_DEFAULT_TEMPERATURE: float | None = None


def _field(payload: object, name: str, default: Any = None) -> Any:
    """Return one field from dict-like or object-like SDK payloads."""
    if isinstance(payload, Mapping):
        return payload.get(name, default)
    if hasattr(payload, name):
        return getattr(payload, name)
    model_extra = getattr(payload, "model_extra", None)
    if isinstance(model_extra, Mapping):
        return model_extra.get(name, default)
    return default


def _coerce_jsonable(value: Any) -> Any:
    """Convert nested SDK values into JSON-serializable builtins."""
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


def _serialize_response_usage(value: object) -> dict[str, Any] | None:
    """Return a stable usage mapping when one exists."""
    usage = _coerce_jsonable(value)
    return usage if isinstance(usage, dict) and usage else None


def _extract_block_text(block: object) -> str:
    """Return human-readable text from one Anthropic content block."""
    block_type = str(_field(block, "type", "") or "")
    if block_type == "thinking":
        return str(_field(block, "thinking", "") or _field(block, "text", "") or "")
    return str(_field(block, "text", "") or "")


def _parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    """Parse one JSON arguments string into a tool input mapping."""
    stripped = raw_arguments.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return {"_raw_arguments": raw_arguments}
    if isinstance(parsed, dict):
        return parsed
    return {"value": parsed}


class AnthropicProvider(LLMProvider):
    """Call Anthropic chat completions with streaming thought and tool support."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-opus-4-6",
        *,
        enable_thinking: bool = True,
        thinking: Mapping[str, Any] | None = None,
        output_config: Mapping[str, Any] | None = None,
        temperature: float | None = _DEFAULT_TEMPERATURE,
        default_max_tokens: int = _DEFAULT_MAX_TOKENS,
        request_timeout_seconds: float = 60.0,
        response_callback: LLMResponseCallback | None = None,
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model: Anthropic model id.
            enable_thinking: Whether to request visible summarized thinking.
            thinking: Optional Anthropic thinking config override.
            output_config: Optional Anthropic output config override.
            temperature: Optional sampling temperature applied to Anthropic requests.
            default_max_tokens: Fallback max tokens when call sites omit one.
            request_timeout_seconds: Request timeout applied to API calls.
            response_callback: Optional streaming callback sink.

        Raises:
            RuntimeError: Anthropic SDK is not installed.
            ValueError: API key is empty.
        """
        if anthropic is None:
            raise RuntimeError(
                "Anthropic support requires the `anthropic` package. "
                "Install dependencies with `uv sync --dev`."
            )
        if not str(api_key).strip():
            raise ValueError("AnthropicProvider requires a non-empty API key.")

        self.api_key = str(api_key).strip()
        self.model = str(model).strip() or "claude-opus-4-6"
        self.enable_thinking = bool(enable_thinking)
        self.temperature = None if temperature is None else float(temperature)
        self.default_max_tokens = max(1, int(default_max_tokens))
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.thinking = _coerce_jsonable(thinking) or dict(_DEFAULT_THINKING)
        self.output_config = (
            _coerce_jsonable(output_config) or dict(_DEFAULT_OUTPUT_CONFIG)
        )
        self._client = anthropic.Anthropic(
            api_key=self.api_key,
            timeout=self.request_timeout_seconds,
        )
        self._response_callback: LLMResponseCallback = (
            response_callback or NullLLMResponseCallback()
        )
        self.last_response_details: dict[str, Any] = {}

        event_bus.publish(
            "system_message",
            f"Connected to Anthropic -> model={self.model}",
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
        response = self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        )
        return response.content.strip()

    def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
    ) -> AssistantMessage:
        """Create one streamed Anthropic message with optional client tools."""
        normalized_messages = normalize_messages(messages)
        system_prompt, anthropic_messages = self._normalize_messages(normalized_messages)
        anthropic_tools = self._normalize_tools(tools)
        request_kwargs = self._build_request_kwargs(
            system_prompt=system_prompt,
            messages=anthropic_messages,
            tools=anthropic_tools,
            tool_choice=tool_choice,
            max_tokens=max_tokens,
        )
        agent_name = infer_agent_name(messages)

        try:
            response_stream = self._client.messages.create(**request_kwargs, stream=True)
            return self._stream_response(
                response_stream=response_stream,
                agent_name=agent_name,
                request_kwargs=request_kwargs,
            )
        except StreamingResponseAbort:
            raise
        except Exception as exc:
            logger.warning(
                "[AnthropicProvider] Streaming chat failed; retrying non-streaming: %s",
                exc,
            )
            return self._chat_non_streaming(
                request_kwargs=request_kwargs,
                agent_name=agent_name,
            )

    def _build_request_kwargs(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: ToolChoice | None,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """Build one Anthropic Messages API request payload."""
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
        }
        if self.temperature is not None:
            temperature = self._sanitize_temperature_for_thinking(self.temperature)
            if temperature is not None:
                request_kwargs["temperature"] = temperature
        if system_prompt.strip():
            request_kwargs["system"] = system_prompt
        if tools:
            request_kwargs["tools"] = tools
            normalized_tool_choice = self._normalize_tool_choice(tool_choice)
            if normalized_tool_choice is not None:
                normalized_tool_choice = self._sanitize_tool_choice_for_thinking(
                    normalized_tool_choice
                )
                request_kwargs["tool_choice"] = normalized_tool_choice
        if self.enable_thinking:
            request_kwargs["thinking"] = dict(self.thinking)
            if self.output_config:
                request_kwargs["output_config"] = dict(self.output_config)
        return request_kwargs

    def _normalize_messages(
        self,
        normalized_messages: list[dict[str, object]],
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert OpenAI-style messages into Anthropic's message schema."""
        system_parts: list[str] = []
        anthropic_messages: list[dict[str, Any]] = []

        for message in normalized_messages:
            role = str(message.get("role") or "user").strip().lower()
            if role == "system":
                content = str(message.get("content") or "").strip()
                if content:
                    system_parts.append(content)
                continue

            if role == "tool":
                tool_message = self._normalize_tool_result_message(message)
                self._append_message(anthropic_messages, tool_message)
                continue

            if role in {"assistant", "model"}:
                assistant_message = self._normalize_assistant_message(message)
                self._append_message(anthropic_messages, assistant_message)
                continue

            user_message = self._normalize_user_message(message)
            self._append_message(anthropic_messages, user_message)

        return "\n\n".join(system_parts).strip(), anthropic_messages

    def _normalize_user_message(self, message: Mapping[str, object]) -> dict[str, Any]:
        """Convert one user-style message into Anthropic block content."""
        content = str(message.get("content") or "")
        return {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": content if content else " ",
                }
            ],
        }

    def _normalize_assistant_message(
        self,
        message: Mapping[str, object],
    ) -> dict[str, Any]:
        """Convert one assistant-style message into Anthropic block content."""
        content_blocks: list[dict[str, Any]] = []
        content = str(message.get("content") or "")
        if content:
            content_blocks.append({"type": "text", "text": content})

        raw_tool_calls = message.get("tool_calls")
        if isinstance(raw_tool_calls, Sequence) and not isinstance(
            raw_tool_calls,
            str | bytes,
        ):
            for tool_call in raw_tool_calls:
                if not isinstance(tool_call, Mapping):
                    continue
                function_payload = _field(tool_call, "function", {})
                function = (
                    function_payload if isinstance(function_payload, Mapping) else {}
                )
                name = str(_field(function, "name", "") or "").strip()
                arguments = str(_field(function, "arguments", "") or "")
                if not name:
                    continue
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(_field(tool_call, "id", "") or "").strip()
                        or name,
                        "name": name,
                        "input": _parse_tool_arguments(arguments),
                    }
                )

        if not content_blocks:
            content_blocks.append({"type": "text", "text": " "})
        return {"role": "assistant", "content": content_blocks}

    def _normalize_tool_result_message(
        self,
        message: Mapping[str, object],
    ) -> dict[str, Any]:
        """Convert one tool-result message into Anthropic user blocks."""
        tool_use_id = str(message.get("tool_call_id") or "").strip()
        content = str(message.get("content") or "")
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content if content else " ",
        }
        return {"role": "user", "content": [block]}

    def _append_message(
        self,
        messages: list[dict[str, Any]],
        message: dict[str, Any],
    ) -> None:
        """Merge consecutive same-role Anthropic messages."""
        if not messages or str(messages[-1].get("role")) != str(message.get("role")):
            messages.append(message)
            return
        prior_content = messages[-1].get("content")
        current_content = message.get("content")
        prior_blocks = prior_content if isinstance(prior_content, list) else []
        current_blocks = current_content if isinstance(current_content, list) else []
        messages[-1]["content"] = [*prior_blocks, *current_blocks]

    def _normalize_tools(
        self,
        tools: Sequence[ToolInput] | None,
    ) -> list[dict[str, Any]] | None:
        """Convert OpenAI-style tool schemas into Anthropic tools."""
        normalized_tools = normalize_tools(tools)
        if not normalized_tools:
            return None
        anthropic_tools: list[dict[str, Any]] = []
        for tool in normalized_tools:
            function_payload = tool.get("function", {})
            function = function_payload if isinstance(function_payload, Mapping) else {}
            tool_payload: dict[str, Any] = {
                "name": str(function.get("name") or "unnamed_tool"),
                "description": str(function.get("description") or ""),
                "input_schema": _coerce_jsonable(function.get("parameters") or {}),
            }
            strict = function.get("strict")
            if isinstance(strict, bool):
                tool_payload["strict"] = strict
            anthropic_tools.append(tool_payload)
        return anthropic_tools

    def _normalize_tool_choice(self, tool_choice: ToolChoice | None) -> dict[str, Any] | None:
        """Map the shared tool-choice shape into Anthropic's tool-choice schema."""
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            normalized = tool_choice.strip().lower()
            if normalized == "auto":
                return {"type": "auto"}
            if normalized == "required":
                return {"type": "any"}
            return None
        if not isinstance(tool_choice, Mapping):
            return None
        type_name = str(tool_choice.get("type") or "").strip().lower()
        if type_name in {"auto", "any"}:
            return {"type": type_name}
        if type_name == "function":
            function_payload = tool_choice.get("function", {})
            function = (
                function_payload if isinstance(function_payload, Mapping) else {}
            )
            name = str(function.get("name") or "").strip()
            if name:
                return {"type": "tool", "name": name}
        if type_name == "tool":
            name = str(tool_choice.get("name") or "").strip()
            if name:
                return {"type": "tool", "name": name}
        return None

    def _sanitize_tool_choice_for_thinking(
        self,
        tool_choice: dict[str, Any],
    ) -> dict[str, Any]:
        """Downgrade forced tool choice when Anthropic thinking is enabled."""
        if not self.enable_thinking:
            return tool_choice
        tool_choice_type = str(tool_choice.get("type") or "").strip().lower()
        if tool_choice_type not in {"any", "tool"}:
            return tool_choice
        logger.warning(
            "[AnthropicProvider] Anthropic thinking does not allow forced tool choice; using tool_choice=auto instead."
        )
        return {"type": "auto"}

    def _sanitize_temperature_for_thinking(
        self,
        temperature: float,
    ) -> float | None:
        """Drop incompatible temperatures when Anthropic thinking is enabled."""
        if not self.enable_thinking:
            return temperature
        if temperature == 1.0:
            return temperature
        logger.warning(
            "[AnthropicProvider] Anthropic thinking only accepts temperature=1; omitting explicit temperature."
        )
        return None

    def _stream_response(
        self,
        *,
        response_stream: Any,
        agent_name: str,
        request_kwargs: Mapping[str, Any],
    ) -> AssistantMessage:
        """Consume one Anthropic streaming response into an assistant message."""
        response_id = ""
        response_role = "assistant"
        stop_reason = ""
        usage: dict[str, Any] | None = None
        full_content = ""
        full_reasoning = ""
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        started_tool_streams: set[int] = set()
        emitted_tool_stream = False

        self._response_callback.on_start(agent_name)
        try:
            for event in response_stream:
                event_type = str(_field(event, "type", "") or "")
                if event_type == "message_start":
                    message = _field(event, "message")
                    response_id = str(_field(message, "id", "") or "")
                    response_role = str(_field(message, "role", "assistant") or "assistant")
                    stop_reason = str(_field(message, "stop_reason", "") or "")
                    usage = _serialize_response_usage(_field(message, "usage"))
                    continue

                if event_type == "content_block_start":
                    index = int(_field(event, "index", 0) or 0)
                    block = _field(event, "content_block") or _field(event, "block")
                    block_type = str(_field(block, "type", "") or "")
                    if block_type == "tool_use":
                        input_payload = _coerce_jsonable(_field(block, "input") or {})
                        tool_calls_by_index[index] = {
                            "id": str(_field(block, "id", "") or ""),
                            "name": str(_field(block, "name", "") or ""),
                            "arguments": (
                                json.dumps(input_payload)
                                if input_payload
                                else ""
                            ),
                        }
                    continue

                if event_type == "content_block_delta":
                    index = int(_field(event, "index", 0) or 0)
                    delta = _field(event, "delta")
                    delta_type = str(_field(delta, "type", "") or "")
                    if delta_type == "thinking_delta":
                        token = str(
                            _field(delta, "thinking", "")
                            or _field(delta, "text", "")
                            or ""
                        )
                        if token:
                            self._response_callback.on_thought_token(token)
                            full_reasoning += token
                        continue
                    if delta_type == "text_delta":
                        token = str(_field(delta, "text", "") or "")
                        if token:
                            self._response_callback.on_content_token(token)
                            full_content += token
                        continue
                    if delta_type == "input_json_delta":
                        partial_json = str(_field(delta, "partial_json", "") or "")
                        if partial_json:
                            tool_calls_by_index.setdefault(
                                index,
                                {"id": "", "name": "", "arguments": ""},
                            )
                            tool_calls_by_index[index]["arguments"] += partial_json
                            if index not in started_tool_streams:
                                tool_name = str(
                                    tool_calls_by_index[index].get("name") or "tool"
                                ).strip() or "tool"
                                prefix = (
                                    f"\nTOOL {tool_name}: "
                                    if full_content or emitted_tool_stream
                                    else f"TOOL {tool_name}: "
                                )
                                self._response_callback.on_content_token(prefix)
                                started_tool_streams.add(index)
                                emitted_tool_stream = True
                            self._response_callback.on_content_token(partial_json)
                        continue
                    continue

                if event_type == "content_block_stop":
                    index = int(_field(event, "index", 0) or 0)
                    if index in started_tool_streams:
                        self._response_callback.on_content_token("\n")
                    continue

                if event_type == "message_delta":
                    delta = _field(event, "delta")
                    stop_reason = str(_field(delta, "stop_reason", "") or stop_reason)
                    delta_usage = _serialize_response_usage(_field(event, "usage"))
                    if delta_usage:
                        usage = delta_usage
                    continue
        finally:
            self._response_callback.on_complete()

        assistant_message = AssistantMessage(
            content=full_content,
            tool_calls=self._build_tool_calls(tool_calls_by_index) or None,
            reasoning_content=full_reasoning if full_reasoning else None,
        )
        self.last_response_details = {
            "provider": "anthropic",
            "request": _coerce_jsonable(request_kwargs),
            "response": {
                "id": response_id,
                "role": response_role,
                "model": self.model,
                "stop_reason": stop_reason,
                "usage": usage,
                "content": assistant_message.content,
                "reasoning_content": assistant_message.reasoning_content,
                "tool_calls": assistant_message.model_dump().get("tool_calls"),
            },
        }
        return assistant_message

    def _chat_non_streaming(
        self,
        *,
        request_kwargs: Mapping[str, Any],
        agent_name: str,
    ) -> AssistantMessage:
        """Retry one Anthropic request without streaming."""
        del agent_name
        response = self._client.messages.create(**dict(request_kwargs))
        assistant_message = self._assistant_message_from_response(response)
        self.last_response_details = {
            "provider": "anthropic",
            "request": _coerce_jsonable(request_kwargs),
            "response": {
                "id": str(_field(response, "id", "") or ""),
                "role": str(_field(response, "role", "assistant") or "assistant"),
                "model": str(_field(response, "model", self.model) or self.model),
                "stop_reason": str(_field(response, "stop_reason", "") or ""),
                "usage": _serialize_response_usage(_field(response, "usage")),
                "content": assistant_message.content,
                "reasoning_content": assistant_message.reasoning_content,
                "tool_calls": assistant_message.model_dump().get("tool_calls"),
                "raw_content_blocks": _coerce_jsonable(_field(response, "content", [])),
            },
        }
        return assistant_message

    def _assistant_message_from_response(self, response: Any) -> AssistantMessage:
        """Convert one non-stream Anthropic response into an assistant message."""
        full_content = ""
        full_reasoning = ""
        tool_calls: list[ToolCall] = []

        raw_blocks = _field(response, "content", [])
        if not isinstance(raw_blocks, Sequence):
            raw_blocks = []

        for index, block in enumerate(raw_blocks):
            block_type = str(_field(block, "type", "") or "")
            if block_type == "thinking":
                full_reasoning += _extract_block_text(block)
                continue
            if block_type == "text":
                full_content += _extract_block_text(block)
                continue
            if block_type == "tool_use":
                input_payload = _coerce_jsonable(_field(block, "input") or {})
                tool_calls.append(
                    ToolCall(
                        id=str(_field(block, "id", "") or f"call_{index}"),
                        function=Function(
                            name=str(_field(block, "name", "") or "unnamed_tool"),
                            arguments=json.dumps(input_payload),
                        ),
                    )
                )

        return AssistantMessage(
            content=full_content,
            tool_calls=tool_calls or None,
            reasoning_content=full_reasoning if full_reasoning else None,
        )

    def _build_tool_calls(
        self,
        tool_calls_by_index: Mapping[int, Mapping[str, Any]],
    ) -> list[ToolCall]:
        """Convert accumulated streamed tool-use state into DTOs."""
        tool_calls: list[ToolCall] = []
        for index in sorted(tool_calls_by_index):
            raw_tool_call = tool_calls_by_index[index]
            tool_calls.append(
                ToolCall(
                    id=str(raw_tool_call.get("id") or f"call_{index}"),
                    function=Function(
                        name=str(raw_tool_call.get("name") or "unnamed_tool"),
                        arguments=str(raw_tool_call.get("arguments") or "{}"),
                    ),
                )
            )
        return tool_calls
