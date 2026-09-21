"""
vLLM / OpenAI-Compatible LLM Provider
======================================
Wraps any OpenAI-compatible endpoint (vLLM, ollama, LiteLLM, etc.)
behind the LLMProvider strategy interface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import time
from typing import Any

from openai import OpenAI

from card_framework.agents.dtos import AssistantMessage, Function, ToolCall
from card_framework.shared.events import event_bus
from card_framework.shared.llm_provider import (
    LLMProvider,
    LLMResponseCallback,
    MessageInput,
    NullLLMResponseCallback,
    ToolChoice,
    ToolInput,
    infer_agent_name,
    normalize_messages,
    normalize_tools,
)


class VLLMProvider(LLMProvider):
    """
    Concrete LLM strategy for OpenAI-compatible APIs (vLLM, etc.).

    Args:
        base_url: The base URL of the API server (for example ``http://host:8000/v1``).
        api_key: API key (use ``"EMPTY"`` for keyless vLLM servers).
        model: Optional explicit model id to pin for reproducible benchmarks.
        enable_thinking: Whether to request reasoning/thinking chunks from vLLM.
        default_max_tokens: Optional output-token budget used when the caller
            does not pass ``max_tokens``. Leave unset to preserve endpoint defaults.
        thinking_extra_body: Optional OpenAI ``extra_body`` payload for reasoning mode.
        request_timeout_seconds: Request timeout applied to model discovery and chat calls.
        response_callback: Optional callback sink for streamed token updates.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str | None = None,
        enable_thinking: bool = True,
        extra_body: dict[str, Any] | None = None,
        default_max_tokens: int | None = None,
        thinking_extra_body: dict[str, Any] | None = None,
        request_timeout_seconds: float = 30.0,
        response_callback: LLMResponseCallback | None = None,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.requested_model = str(model).strip() if model else ""
        self.enable_thinking = enable_thinking
        self.extra_body = self._coerce_jsonable(extra_body) or {}
        self.default_max_tokens = (
            int(default_max_tokens)
            if default_max_tokens is not None and int(default_max_tokens) > 0
            else None
        )
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.thinking_extra_body = self._coerce_jsonable(thinking_extra_body) or {
            "chat_template_kwargs": {"enable_thinking": True}
        }
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.request_timeout_seconds,
            max_retries=0,
        )
        self.last_request: dict[str, Any] | None = None
        self.last_response: dict[str, Any] | None = None
        self._response_callback: LLMResponseCallback = (
            response_callback or NullLLMResponseCallback()
        )

        # Resolve model id from the server on startup.
        self.model_id = self._fetch_model_id(self.requested_model or None)
        event_bus.publish(
            "system_message",
            f"Connected to authenticated vLLM model={self.model_id}",
        )

    @classmethod
    def _coerce_jsonable(cls, value: Any) -> Any:
        """Convert nested config containers into JSON-serializable builtins."""
        if isinstance(value, Mapping):
            return {
                str(key): cls._coerce_jsonable(raw_value)
                for key, raw_value in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [cls._coerce_jsonable(item) for item in value]
        return value

    def set_response_callback(self, response_callback: LLMResponseCallback) -> None:
        """Replace the streaming callback sink at runtime."""
        self._response_callback = response_callback

    def _fetch_model_id(self, requested_model: str | None = None) -> str:
        """Discover models using the same authenticated client as chat requests.

        Args:
            requested_model: Explicit model identifier to validate, or None to
                select the sole advertised model.

        Returns:
            The validated server model identifier.

        Raises:
            ValueError: The server advertises no models, an ambiguous selection,
                or a set that does not include the requested model.
            openai.APIError: Authentication, connectivity, or discovery fails.
        """
        models = list(self._client.models.list().data)
        available_model_ids = [model.id for model in models]
        if not available_model_ids:
            raise ValueError("The vLLM endpoint advertised no models.")
        if requested_model is None:
            if len(available_model_ids) != 1:
                raise ValueError("Multiple models advertised; set an explicit model.")
            requested_model = available_model_ids[0]
        normalized_requested_model = requested_model.strip()
        if normalized_requested_model in available_model_ids:
            selected = models[available_model_ids.index(normalized_requested_model)]
            self.server_model_metadata = selected.model_dump()
            return normalized_requested_model
        available_models_display = ", ".join(available_model_ids)
        raise ValueError(
            "Configured vLLM model "
            f"'{normalized_requested_model}' is not advertised by the vLLM endpoint. "
            f"Available models: {available_models_display}."
        )

    def _maybe_enable_thinking(self, create_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Attach configured OpenAI-compatible extra-body payloads."""
        extra_body: dict[str, Any] = dict(self.extra_body)
        if self.enable_thinking:
            extra_body = self._merge_extra_body(extra_body, self.thinking_extra_body)
        if extra_body.get("truncate_prompt_tokens") is not None:
            raise ValueError("Prompt truncation is disabled; remove truncate_prompt_tokens.")
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        return create_kwargs

    @classmethod
    def _merge_extra_body(
        cls,
        base: dict[str, Any],
        overlay: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge nested extra-body dictionaries without mutating inputs."""
        merged = cls._coerce_jsonable(base)
        for key, value in cls._coerce_jsonable(overlay).items():
            existing = merged.get(key)
            if isinstance(existing, Mapping) and isinstance(value, Mapping):
                merged[key] = cls._merge_extra_body(dict(existing), dict(value))
                continue
            merged[key] = value
        return merged

    def _resolve_max_tokens(self, max_tokens: int | None) -> int | None:
        """Return the call-level output budget or the configured default."""
        if max_tokens is not None:
            return int(max_tokens)
        return self.default_max_tokens

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        """Generate complete text without accepting length-limited responses.

        Args:
            system_prompt: Full system instruction, passed without clipping.
            user_prompt: Full user context, passed without clipping.
            max_tokens: Explicit output cap, or None to omit a client cap.

        Returns:
            The model's final text, after a valid terminal finish reason.

        Raises:
            ValueError: The response is truncated or ends without a finish reason.
            openai.APIError: The single provider request fails.
        """
        return self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
        ).content

    def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        tool_choice: ToolChoice | None = None,
        max_tokens: int | None = None,
    ) -> AssistantMessage:
        """Collect one complete response and preserve full request/response records.

        The transport streams to support long requests, but the retained response
        is a complete exchange, not token fragments. No retry, alternate transport,
        or prompt reduction follows a failed request. Partial output remains in
        ``last_response`` for diagnosis and cannot be dispatched as a tool call.

        Args:
            messages: Complete conversation supplied by the caller.
            tools: Tool schemas available for this call, or None.
            tool_choice: Explicit tool-selection instruction, or None.
            max_tokens: Explicit output limit, or None to omit a client cap.

        Returns:
            An AssistantMessage containing completed text, server-returned
            reasoning, and complete tool calls.

        Raises:
            ValueError: The server reports an incomplete finish, omits a terminal
                finish reason, or returns an incomplete tool-call identifier.
            openai.APIError: Authentication, context size, or transport fails.
        """
        self.last_request = None
        self.last_response = None
        normalized_messages = normalize_messages(messages)
        normalized_tools = normalize_tools(tools)

        create_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": normalized_messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if normalized_tools is not None:
            create_kwargs["tools"] = normalized_tools
        if tool_choice is not None:
            create_kwargs["tool_choice"] = tool_choice
        effective_max_tokens = self._resolve_max_tokens(max_tokens)
        if effective_max_tokens is not None:
            create_kwargs["max_tokens"] = effective_max_tokens
        create_kwargs = self._maybe_enable_thinking(create_kwargs)
        self.last_request = create_kwargs
        self.last_response = None
        started = time.perf_counter()
        full_content = ""
        full_thought = ""
        reasoning_fields: set[str] = set()
        tool_calls_data: list[dict[str, Any]] = []
        finish_reason = None
        usage = None
        response_model = None
        self._response_callback.on_start(infer_agent_name(messages))
        try:
            with self._client.chat.completions.create(**create_kwargs) as response_stream:
                for chunk in response_stream:
                    response_model = chunk.model
                    if chunk.usage is not None:
                        usage = chunk.usage.model_dump()
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    if choice.finish_reason is not None:
                        finish_reason = choice.finish_reason
                    delta = choice.delta
                    reasoning_content = getattr(delta, "reasoning_content", None)
                    reasoning = getattr(delta, "reasoning", None)
                    if reasoning_content and reasoning and reasoning_content != reasoning:
                        raise ValueError(
                            "The vLLM response supplied conflicting reasoning and "
                            "reasoning_content fields."
                        )
                    thought_delta = reasoning_content or reasoning
                    if thought_delta:
                        reasoning_fields.add(
                            "reasoning_content" if reasoning_content else "reasoning"
                        )
                        self._response_callback.on_thought_token(thought_delta)
                        full_thought += thought_delta

                    if hasattr(delta, "content") and delta.content:
                        self._response_callback.on_content_token(delta.content)
                        full_content += delta.content

                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tc_delta in delta.tool_calls:
                            while len(tool_calls_data) <= tc_delta.index:
                                tool_calls_data.append(
                                    {
                                        "id": None,
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                        "preview_started": False,
                                    }
                                )

                            tool_call = tool_calls_data[tc_delta.index]
                            if tc_delta.id:
                                tool_call["id"] = tc_delta.id
                            if tc_delta.function:
                                if tc_delta.function.name:
                                    tool_call["function"]["name"] += (
                                        tc_delta.function.name
                                    )
                                if tc_delta.function.arguments:
                                    if not bool(tool_call.get("preview_started")):
                                        tool_name = (
                                            str(tool_call["function"]["name"]).strip()
                                            or "tool"
                                        )
                                        self._response_callback.on_content_token(
                                            f"\nTOOL {tool_name}: "
                                        )
                                        tool_call["preview_started"] = True
                                    self._response_callback.on_content_token(
                                        tc_delta.function.arguments
                                    )
                                    tool_call["function"]["arguments"] += (
                                        tc_delta.function.arguments
                                    )
        finally:
            self.last_response = {
                "model": response_model,
                "finish_reason": finish_reason,
                "usage": usage,
                "content": full_content,
                "reasoning_content": full_thought,
                "reasoning_fields": sorted(reasoning_fields),
                "tool_calls": tool_calls_data,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }
            self._response_callback.on_complete()

        if finish_reason not in {"stop", "tool_calls"}:
            raise ValueError(
                f"Incomplete vLLM response: finish_reason={finish_reason!r}; "
                "the request will not be retried or its context shortened."
            )
        if any(not tc["id"] or not tc["function"]["name"] for tc in tool_calls_data):
            raise ValueError("Incomplete tool call in the vLLM response.")
        for call in tool_calls_data:
            arguments = json.loads(call["function"]["arguments"])
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a complete JSON object.")
        final_tool_calls = [
            ToolCall(
                id=tc["id"],
                function=Function(**tc["function"]),
            )
            for tc in tool_calls_data
        ]
        return AssistantMessage(
            content=full_content,
            tool_calls=final_tool_calls if final_tool_calls else None,
            reasoning_content=full_thought if full_thought else None,
        )
