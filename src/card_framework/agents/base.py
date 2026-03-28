import asyncio
import ast
import hashlib
import json
from abc import ABC, abstractmethod
from contextlib import contextmanager
from difflib import SequenceMatcher
import threading
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import Task, TaskState, TaskStatus
from a2a.utils import new_agent_text_message

from card_framework.agents.tool_call_utils import build_tool_signature
from card_framework.agents.prompt_compaction import (
    ContextBudgetManager,
    PromptCompactionRequest,
    PromptCompactionResult,
    apply_compaction_to_messages,
    is_context_window_overflow_error,
)
from card_framework.shared.logger_utils import logger
from card_framework.shared.llm_provider import (
    LLMProvider,
    LLMResponseCallback,
    Message,
    NullLLMResponseCallback,
    StreamingResponseAbort,
    ToolDefinition,
)
from card_framework.shared.repetition_guard import (
    count_words,
    extract_repeated_response_span,
    normalize_repetition_text,
)


class BaseA2AExecutor(AgentExecutor, ABC):
    """
    Base class for A2A executors to standardize common logic.
    """

    _provider_callback_locks: dict[int, threading.Lock] = {}
    _provider_callback_locks_guard = threading.Lock()

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    async def _emit_working_task_event(
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Publish an early working-task event so non-blocking A2A calls can poll."""
        if (
            context.task_id is None
            or context.context_id is None
            or context.message is None
        ):
            return

        existing_task = context.current_task
        if existing_task is None:
            task_event = Task(
                id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.working),
                history=[context.message],
            )
        else:
            task_event = existing_task.model_copy(
                update={"status": TaskStatus(state=TaskState.working)}
            )
        context.current_task = task_event
        await event_queue.enqueue_event(task_event)

    @staticmethod
    def _summarize_json_text(raw_text: str, *, max_preview_chars: int = 160) -> str:
        """Return a concise one-line summary for JSON-like payload logs."""
        stripped = raw_text.strip()
        if not stripped:
            return "<empty>"
        if stripped[0] in "{[":
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                return f"dict keys={list(payload.keys())}"
            if isinstance(payload, list):
                return f"list len={len(payload)}"
        compact = " ".join(raw_text.split())
        if len(compact) <= max_preview_chars:
            return compact
        return f"{compact[: max_preview_chars - 3]}..."

    @staticmethod
    def _is_synthetic_tool_call_id(tool_call_id: str) -> bool:
        """Return True for parser-generated fallback IDs that can repeat across turns."""
        return (
            tool_call_id.startswith("xml_fallback_")
            or tool_call_id.startswith("text_fallback_")
            or tool_call_id.startswith("sanitized_")
        )

    @staticmethod
    def _resolve_signature_dedupe_window(runtime_context: dict[str, Any]) -> int:
        """Return the configured signature dedupe window in turns (0 disables)."""
        raw_value = runtime_context.get("signature_dedupe_window_turns", 0)
        if isinstance(raw_value, int) and raw_value >= 0:
            return raw_value
        return 0

    @staticmethod
    def _normalize_tool_arguments(raw_arguments: Any) -> str:
        """
        Normalize tool call arguments into a valid JSON object string.

        Models occasionally emit python-literal fragments (single quotes) or
        truncated payloads (e.g. "{"). We coerce unsupported forms to "{}"
        so malformed history cannot break subsequent provider requests.
        """
        if raw_arguments is None:
            return "{}"

        parsed: Any
        if isinstance(raw_arguments, str):
            candidate = raw_arguments.strip()
            if not candidate:
                return "{}"
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(candidate)
                except (SyntaxError, ValueError):
                    return "{}"
        else:
            parsed = raw_arguments

        if not isinstance(parsed, dict):
            return "{}"

        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def _sanitize_assistant_tool_calls(cls, msg_dict: dict[str, Any]) -> dict[str, Any]:
        """Return a defensively sanitized assistant message for history replay."""
        if msg_dict.get("role") != "assistant":
            return msg_dict

        raw_tool_calls = msg_dict.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            return msg_dict

        sanitized_tool_calls: list[dict[str, Any]] = []
        for index, raw_call in enumerate(raw_tool_calls):
            if not isinstance(raw_call, dict):
                continue

            raw_function = raw_call.get("function")
            function = dict(raw_function) if isinstance(raw_function, dict) else {}
            name = function.get("name")
            normalized_name = str(name).strip() if name is not None else ""
            if not normalized_name:
                normalized_name = "unknown_tool"

            normalized_arguments = cls._normalize_tool_arguments(
                function.get("arguments")
            )

            raw_id = raw_call.get("id")
            if isinstance(raw_id, str) and raw_id.strip():
                tool_call_id = raw_id
            else:
                digest_input = (
                    f"{normalized_name}|{normalized_arguments}|{index}"
                ).encode("utf-8")
                digest = hashlib.sha1(digest_input).hexdigest()[:10]
                tool_call_id = f"sanitized_{index}_{digest}"

            sanitized_tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": normalized_name,
                        "arguments": normalized_arguments,
                    },
                }
            )

        msg_dict["tool_calls"] = sanitized_tool_calls if sanitized_tool_calls else None
        return msg_dict

    @classmethod
    def _normalize_reasoning_loop_text(cls, raw_text: str) -> str:
        """Normalize assistant text for repetition and similarity checks."""
        return normalize_repetition_text(raw_text)

    @staticmethod
    def _count_words(raw_text: str) -> int:
        """Return a loose word count for repetition guard thresholds."""
        return count_words(raw_text)

    @classmethod
    def _extract_repeated_response_span(
        cls,
        raw_text: str,
        *,
        min_words: int,
        repeat_threshold: int,
    ) -> tuple[str, int] | None:
        """Return a repeated sentence-like span when one dominates a response."""
        return extract_repeated_response_span(
            raw_text,
            min_words=min_words,
            repeat_threshold=repeat_threshold,
        )

    @staticmethod
    def _coerce_reasoning_loop_guard_config(
        raw_guard: Any,
    ) -> dict[str, Any]:
        """Normalize the opt-in no-tool repetition guard configuration."""
        guard_payload = dict(raw_guard) if isinstance(raw_guard, dict) else {}

        def _bool(name: str, default: bool) -> bool:
            raw_value = guard_payload.get(name, default)
            return raw_value if isinstance(raw_value, bool) else default

        def _int(name: str, default: int, minimum: int) -> int:
            raw_value = guard_payload.get(name, default)
            try:
                parsed = int(raw_value)
            except (TypeError, ValueError):
                return default
            return max(minimum, parsed)

        def _ratio(name: str, default: float) -> float:
            raw_value = guard_payload.get(name, default)
            try:
                parsed = float(raw_value)
            except (TypeError, ValueError):
                return default
            return max(0.0, min(1.0, parsed))

        return {
            "enabled": _bool("enabled", False),
            "max_consecutive_no_tool_turns": _int(
                "max_consecutive_no_tool_turns",
                default=2,
                minimum=1,
            ),
            "single_response_repeat_threshold": _int(
                "single_response_repeat_threshold",
                default=3,
                minimum=2,
            ),
            "min_repeated_sentence_words": _int(
                "min_repeated_sentence_words",
                default=8,
                minimum=1,
            ),
            "cross_turn_similarity_threshold": _ratio(
                "cross_turn_similarity_threshold",
                default=0.96,
            ),
        }

    @staticmethod
    def _reasoning_loop_guard_state(
        runtime_context: dict[str, Any],
    ) -> dict[str, Any]:
        """Return mutable runtime state for the reasoning-loop guard."""
        raw_state = runtime_context.get("reasoning_loop_guard_state")
        if isinstance(raw_state, dict):
            return raw_state
        state = {
            "consecutive_no_tool_turns": 0,
            "last_no_tool_response_normalized": "",
            "last_no_tool_response_word_count": 0,
        }
        runtime_context["reasoning_loop_guard_state"] = state
        return state

    @classmethod
    def _build_reasoning_loop_guard_stop_reason(
        cls,
        *,
        runtime_context: dict[str, Any],
        guard_config: dict[str, Any],
        assistant_text: str,
    ) -> str | None:
        """Return a stop reason when repetitive no-tool reasoning is detected."""
        if not guard_config["enabled"]:
            return None

        normalized_text = cls._normalize_reasoning_loop_text(assistant_text)
        state = cls._reasoning_loop_guard_state(runtime_context)
        consecutive_no_tool_turns = int(state.get("consecutive_no_tool_turns", 0)) + 1
        state["consecutive_no_tool_turns"] = consecutive_no_tool_turns

        repeated_span = cls._extract_repeated_response_span(
            assistant_text,
            min_words=int(guard_config["min_repeated_sentence_words"]),
            repeat_threshold=int(guard_config["single_response_repeat_threshold"]),
        )
        if repeated_span is not None:
            preview, repeat_count = repeated_span
            return (
                f"single-response repetition detected ({repeat_count} repeats): "
                f"{cls._summarize_json_text(preview)}"
            )

        current_word_count = cls._count_words(normalized_text)
        previous_text = str(state.get("last_no_tool_response_normalized", ""))
        previous_word_count = int(state.get("last_no_tool_response_word_count", 0))
        if (
            previous_text
            and normalized_text
            and current_word_count >= int(guard_config["min_repeated_sentence_words"])
            and previous_word_count >= int(guard_config["min_repeated_sentence_words"])
            and consecutive_no_tool_turns
            >= int(guard_config["max_consecutive_no_tool_turns"])
        ):
            similarity = SequenceMatcher(
                a=previous_text,
                b=normalized_text,
            ).ratio()
            if similarity >= float(guard_config["cross_turn_similarity_threshold"]):
                return (
                    "cross-turn repetition detected "
                    f"(similarity={similarity:.2f}, turns={consecutive_no_tool_turns})"
                )

        state["last_no_tool_response_normalized"] = normalized_text
        state["last_no_tool_response_word_count"] = current_word_count
        return None

    @staticmethod
    def _reset_reasoning_loop_guard_state(runtime_context: dict[str, Any]) -> None:
        """Clear no-tool repetition history after a successful tool-call turn."""
        state = BaseA2AExecutor._reasoning_loop_guard_state(runtime_context)
        state["consecutive_no_tool_turns"] = 0
        state["last_no_tool_response_normalized"] = ""
        state["last_no_tool_response_word_count"] = 0

    @classmethod
    def _resolve_callback_target(
        cls,
        provider: LLMProvider,
    ) -> tuple[LLMProvider, LLMResponseCallback] | None:
        """Find the deepest provider that supports callback injection."""
        current: object = provider
        seen: set[int] = set()
        while isinstance(current, LLMProvider) and id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, "set_response_callback"):
                existing_callback = (
                    current._response_callback
                    if hasattr(current, "_response_callback")
                    else None
                )
                if existing_callback is None:
                    existing_callback = NullLLMResponseCallback()
                return current, existing_callback
            current = (
                current.inner_provider if hasattr(current, "inner_provider") else None
            )
        return None

    @classmethod
    def _resolve_callback_lock(cls, provider: LLMProvider) -> threading.Lock:
        """Return a stable lock for temporary callback swaps on one provider."""
        provider_id = id(provider)
        with cls._provider_callback_locks_guard:
            callback_lock = cls._provider_callback_locks.get(provider_id)
            if callback_lock is None:
                callback_lock = threading.Lock()
                cls._provider_callback_locks[provider_id] = callback_lock
            return callback_lock

    @classmethod
    @contextmanager
    def _temporary_response_callback(
        cls,
        provider: LLMProvider,
        callback: LLMResponseCallback | None,
    ):
        """Temporarily replace the provider stream callback for one chat call."""
        if callback is None:
            yield
            return

        callback_target = cls._resolve_callback_target(provider)
        if callback_target is None:
            yield
            return

        target_provider, existing_callback = callback_target
        callback_lock = cls._resolve_callback_lock(target_provider)
        with callback_lock:
            target_provider.set_response_callback(callback)
            try:
                yield
            finally:
                target_provider.set_response_callback(existing_callback)

    def _build_runtime_stream_callback(
        self,
        runtime_context: dict[str, Any],
    ) -> LLMResponseCallback | None:
        """Build an optional per-turn callback wrapper from runtime context."""
        callback_factory = runtime_context.get("stream_response_callback_factory")
        if not callable(callback_factory):
            return None
        callback_target = self._resolve_callback_target(self.llm)
        existing_callback: LLMResponseCallback
        if callback_target is None:
            existing_callback = NullLLMResponseCallback()
        else:
            _provider, existing_callback = callback_target
        built_callback = callback_factory(existing_callback)
        return built_callback if built_callback is not None else None

    async def _chat_with_runtime_stream_callback(
        self,
        *,
        chat_kwargs: dict[str, Any],
        runtime_context: dict[str, Any],
    ) -> Any:
        """Execute one chat call with an optional temporary stream callback."""
        runtime_callback = self._build_runtime_stream_callback(runtime_context)
        if runtime_callback is None:
            return await asyncio.to_thread(self.llm.chat, **chat_kwargs)

        def _chat_with_callback() -> Any:
            with self._temporary_response_callback(self.llm, runtime_callback):
                return self.llm.chat(**chat_kwargs)

        return await asyncio.to_thread(_chat_with_callback)

    def _handle_streaming_response_abort(
        self,
        *,
        abort_error: StreamingResponseAbort,
        messages: list[dict[str, Any]],
        runtime_context: dict[str, Any],
    ) -> bool:
        """Handle one controlled stream abort and optionally retry the turn."""
        from card_framework.shared.events import event_bus

        retry_count = int(runtime_context.get("stream_abort_retry_count", 0)) + 1
        runtime_context["stream_abort_retry_count"] = retry_count
        max_retry_attempts = int(runtime_context.get("max_stream_abort_retries", 0))
        retry_prompt_builder = runtime_context.get("stream_abort_retry_prompt_builder")

        if callable(retry_prompt_builder) and retry_count <= max_retry_attempts:
            retry_prompt = retry_prompt_builder(
                abort_error=abort_error,
                attempt=retry_count,
                max_attempts=max_retry_attempts,
            )
            if isinstance(retry_prompt, str) and retry_prompt.strip():
                logger.warning(
                    "[%s] Streaming guard aborted response; retrying turn %s/%s: %s",
                    self.name,
                    retry_count,
                    max_retry_attempts,
                    abort_error.reason,
                )
                event_bus.publish(
                    "status_message",
                    message=(
                        f"{self.name} stream guard interrupted a repetitive response "
                        f"and is retrying turn {retry_count}/{max_retry_attempts}."
                    ),
                )
                runtime_context["no_tool_call_count"] = 0
                self._reset_reasoning_loop_guard_state(runtime_context)
                messages.append({"role": "user", "content": retry_prompt})
                return True

        runtime_context["loop_stop_category"] = "reasoning_loop_guard"
        runtime_context["loop_stop_reason"] = abort_error.reason
        logger.warning(
            "[%s] Streaming guard aborted response without retry: %s",
            self.name,
            abort_error.reason,
        )
        event_bus.publish(
            "status_message",
            message=(
                f"{self.name} stream guard stopped a repetitive response: "
                f"{abort_error.reason}"
            ),
        )
        return False

    def _maybe_compact_prompt_messages(
        self,
        *,
        messages: list[dict[str, Any]],
        runtime_context: dict[str, Any],
        force: bool,
        turn_index: int,
    ) -> bool:
        """Compact message history when the configured budget policy requires it."""
        budget_manager = runtime_context.get("context_budget_manager")
        if not isinstance(budget_manager, ContextBudgetManager):
            return False
        profile_name = str(
            runtime_context.get("prompt_compaction_profile_name", "")
        ).strip()
        strategy = runtime_context.get("prompt_compaction_strategy")
        request_builder = runtime_context.get("prompt_compaction_request_builder")
        if not profile_name or not callable(request_builder) or not hasattr(
            strategy, "compact"
        ):
            return False

        snapshot = budget_manager.snapshot_for_messages(
            profile_name=profile_name,
            messages=messages,
        )
        if not force and not snapshot.should_compact:
            return False

        try:
            request = request_builder(
                messages=tuple(messages),
                forced=force,
                turn_index=turn_index,
                snapshot=snapshot,
            )
        except Exception as exc:
            logger.warning(
                "[%s] Prompt compaction request build failed: %s",
                self.name,
                exc,
            )
            return False
        if not isinstance(request, PromptCompactionRequest):
            return False

        try:
            compaction = strategy.compact(request=request)
        except Exception as exc:
            logger.warning("[%s] Prompt compaction failed: %s", self.name, exc)
            return False
        if not isinstance(compaction, PromptCompactionResult):
            return False

        keep_latest_iterations = runtime_context.get(
            "prompt_compaction_keep_latest_iterations",
            budget_manager.config.keep_latest_iterations,
        )
        try:
            keep_latest_iterations_int = max(0, int(keep_latest_iterations))
        except (TypeError, ValueError):
            keep_latest_iterations_int = budget_manager.config.keep_latest_iterations

        compacted_messages = apply_compaction_to_messages(
            messages=messages,
            compaction=compaction,
            keep_latest_iterations=keep_latest_iterations_int,
        )
        if not compacted_messages:
            return False
        messages[:] = compacted_messages
        runtime_context["prompt_compaction_count"] = (
            int(runtime_context.get("prompt_compaction_count", 0)) + 1
        )
        runtime_context["prompt_compaction_last_reason"] = compaction.compression_reason
        logger.info(
            "[%s] Prompt compaction applied at turn=%s force=%s reason=%s before=%s after=%s",
            self.name,
            turn_index,
            force,
            compaction.compression_reason,
            compaction.estimated_tokens_before,
            compaction.estimated_tokens_after,
        )
        return True

    def _handle_context_window_overflow(
        self,
        *,
        error: Exception,
        messages: list[dict[str, Any]],
        runtime_context: dict[str, Any],
        turn_index: int,
    ) -> bool:
        """Compact and retry once when a provider rejects the prompt for size."""
        budget_manager = runtime_context.get("context_budget_manager")
        if not isinstance(budget_manager, ContextBudgetManager):
            return False
        if not budget_manager.config.retry_on_overflow:
            return False
        if not is_context_window_overflow_error(error):
            return False

        retry_count = int(runtime_context.get("prompt_compaction_overflow_retries", 0))
        if retry_count >= budget_manager.config.max_overflow_retries:
            return False
        runtime_context["prompt_compaction_overflow_retries"] = retry_count + 1
        compacted = self._maybe_compact_prompt_messages(
            messages=messages,
            runtime_context=runtime_context,
            force=True,
            turn_index=turn_index,
        )
        if not compacted:
            return False
        logger.warning(
            "[%s] Context-window overflow at turn=%s; compacted prompt and retrying (%s/%s).",
            self.name,
            turn_index,
            retry_count + 1,
            budget_manager.config.max_overflow_retries,
        )
        return True

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        raw_input = context.get_user_input()
        logger.info(
            f"[{self.name}] Received task: {self._summarize_json_text(str(raw_input))}"
        )
        logger.debug(f"[{self.name}] Received task payload: {raw_input}")

        if not raw_input:
            raise ValueError(f"No task provided to {self.name}")

        try:
            task_data = json.loads(raw_input)
        except json.JSONDecodeError:
            raise ValueError(f"{self.name} received non-JSON task: {raw_input[:200]}")

        try:
            await self._emit_working_task_event(context, event_queue)
            await asyncio.sleep(0)
            await self.handle_task(task_data, context, event_queue)
        except Exception as e:
            logger.exception("[%s] Error handling task: %s", self.name, e)
            await self.send_failure(str(e), context, event_queue)

    @abstractmethod
    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Subclasses implement this to handle the parsed JSON task."""
        pass

    async def send_response(
        self,
        result: str,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Helper to send a text response back."""
        logger.info(f"[{self.name}] Result: {self._summarize_json_text(result)}")
        logger.debug(f"[{self.name}] Result payload: {result}")
        task_id = getattr(context, "task_id", None)
        context_id = getattr(context, "context_id", None)
        response_message = new_agent_text_message(
            result,
            context_id=context_id,
            task_id=task_id,
        )

        if (
            isinstance(task_id, str)
            and task_id
            and isinstance(context_id, str)
            and context_id
        ):
            from a2a.server.tasks import TaskUpdater

            task_updater = TaskUpdater(
                event_queue=event_queue,
                task_id=task_id,
                context_id=context_id,
            )
            await task_updater.complete(message=response_message)
            return

        await event_queue.enqueue_event(response_message)

    async def send_failure(
        self,
        error_text: str,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Publish one terminal failed-task update with a human-readable error."""
        failure_message = new_agent_text_message(
            error_text,
            context_id=getattr(context, "context_id", None),
            task_id=getattr(context, "task_id", None),
        )
        task_id = getattr(context, "task_id", None)
        context_id = getattr(context, "context_id", None)

        if (
            isinstance(task_id, str)
            and task_id
            and isinstance(context_id, str)
            and context_id
        ):
            from a2a.server.tasks import TaskUpdater

            task_updater = TaskUpdater(
                event_queue=event_queue,
                task_id=task_id,
                context_id=context_id,
            )
            await task_updater.failed(message=failure_message)
            return

        await event_queue.enqueue_event(failure_message)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        pass

    async def process_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
        messages: list[dict[str, Any]],
        context_data: dict[str, Any],
    ) -> tuple[bool, dict | None]:
        """Subclasses should implement to dispatch tools and append results to messages. Returns (should_break, final_result)"""
        return False, None

    async def run_agent_loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_turns: int,
        context_data: dict[str, Any] | None = None,
    ) -> dict | None:
        from card_framework.agents.parsers import get_default_parser_with_options
        from card_framework.shared.events import event_bus

        runtime_context = context_data if context_data is not None else {}
        replay_dedupe_tools = set(runtime_context.get("replay_dedupe_tools", []))
        seen_tool_call_ids = set(runtime_context.get("seen_tool_call_ids", []))
        signature_window_turns = self._resolve_signature_dedupe_window(runtime_context)
        seen_tool_signature_turns_raw = runtime_context.get(
            "seen_tool_signature_turns", {}
        )
        seen_tool_signature_turns: dict[str, int] = (
            dict(seen_tool_signature_turns_raw)
            if isinstance(seen_tool_signature_turns_raw, dict)
            else {}
        )
        max_tool_calls_per_turn_raw = runtime_context.get("max_tool_calls_per_turn")
        max_tool_calls_per_turn: int | None = None
        if (
            isinstance(max_tool_calls_per_turn_raw, int)
            and max_tool_calls_per_turn_raw > 0
        ):
            max_tool_calls_per_turn = max_tool_calls_per_turn_raw

        runtime_context["seen_tool_call_ids"] = seen_tool_call_ids
        runtime_context["seen_tool_signature_turns"] = seen_tool_signature_turns
        runtime_context.setdefault("tool_replay_window", "run")
        runtime_context["no_tool_call_count"] = 0
        runtime_context.pop("loop_stop_category", None)
        runtime_context.pop("loop_stop_reason", None)
        enable_extended_text_tool_parser = bool(
            runtime_context.get("enable_extended_text_tool_parser", False)
        )
        reasoning_loop_guard = self._coerce_reasoning_loop_guard_config(
            runtime_context.get("reasoning_loop_guard")
        )
        runtime_context["reasoning_loop_guard"] = reasoning_loop_guard

        total_parsed_calls = 0
        total_executed_calls = 0
        total_skipped_calls = 0

        parser = get_default_parser_with_options(
            enable_extended_text_fallback=enable_extended_text_tool_parser
        )

        for turn in range(max_turns):
            turn_index = turn + 1
            tool_choice_raw = runtime_context.get("tool_choice")
            tool_choice: str | dict[str, Any] | None
            if isinstance(tool_choice_raw, str):
                tool_choice = tool_choice_raw
            elif isinstance(tool_choice_raw, dict):
                tool_choice = tool_choice_raw
            elif tool_choice_raw is None:
                tool_choice = None
            else:
                logger.warning(
                    "[%s] Ignoring invalid tool_choice type: %s",
                    self.name,
                    type(tool_choice_raw).__name__,
                )
                tool_choice = None

            chat_max_tokens_raw = runtime_context.get("chat_max_tokens")
            chat_max_tokens: int | None = None
            if isinstance(chat_max_tokens_raw, int) and chat_max_tokens_raw > 0:
                chat_max_tokens = chat_max_tokens_raw

            no_tool_call_patience_raw = runtime_context.get("no_tool_call_patience")
            no_tool_call_patience = (
                no_tool_call_patience_raw
                if isinstance(no_tool_call_patience_raw, int)
                and no_tool_call_patience_raw > 0
                else None
            )

            typed_messages = [Message.from_mapping(message) for message in messages]
            typed_tools = [ToolDefinition.from_mapping(tool) for tool in tools]
            chat_kwargs: dict[str, Any] = {
                "messages": typed_messages,
                "tools": typed_tools,
            }
            if tool_choice is not None:
                chat_kwargs["tool_choice"] = tool_choice
            if chat_max_tokens is not None:
                chat_kwargs["max_tokens"] = chat_max_tokens
            response_message = None
            while True:
                self._maybe_compact_prompt_messages(
                    messages=messages,
                    runtime_context=runtime_context,
                    force=False,
                    turn_index=turn_index,
                )
                typed_messages = [Message.from_mapping(message) for message in messages]
                chat_kwargs["messages"] = typed_messages
                try:
                    response_message = await self._chat_with_runtime_stream_callback(
                        chat_kwargs=chat_kwargs,
                        runtime_context=runtime_context,
                    )
                except StreamingResponseAbort as abort_error:
                    if self._handle_streaming_response_abort(
                        abort_error=abort_error,
                        messages=messages,
                        runtime_context=runtime_context,
                    ):
                        continue
                    break
                except Exception as exc:
                    if self._handle_context_window_overflow(
                        error=exc,
                        messages=messages,
                        runtime_context=runtime_context,
                        turn_index=turn_index,
                    ):
                        continue
                    raise
                runtime_context["stream_abort_retry_count"] = 0
                runtime_context["prompt_compaction_overflow_retries"] = 0
                break
            if response_message is None:
                break
            reasoning_content = ""
            if hasattr(response_message, "model_dump"):
                raw_response_payload = response_message.model_dump()
                raw_reasoning = raw_response_payload.get("reasoning_content")
                reasoning_content = (
                    str(raw_reasoning).strip() if raw_reasoning is not None else ""
                )
                msg_dict = raw_response_payload
                msg_dict = {
                    k: v
                    for k, v in msg_dict.items()
                    if k in ("role", "content", "tool_calls")
                }
            else:
                raw_reasoning = getattr(response_message, "reasoning_content", None)
                if raw_reasoning is None:
                    raw_reasoning = getattr(response_message, "reasoning", None)
                reasoning_content = (
                    str(raw_reasoning).strip() if raw_reasoning is not None else ""
                )
                msg_dict = {
                    "role": response_message.role,
                    "content": response_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (response_message.tool_calls or [])
                    ]
                    if hasattr(response_message, "tool_calls")
                    and response_message.tool_calls
                    else None,
                }
            msg_dict = self._sanitize_assistant_tool_calls(msg_dict)
            messages.append(msg_dict)

            tool_calls = parser.parse(msg_dict)
            total_parsed_calls += len(tool_calls)
            if not tool_calls:
                if bool(
                    runtime_context.get("break_on_no_tool_call_when_draft_ready", False)
                ) and bool(runtime_context.get("draft_ready_for_review", False)):
                    logger.info(
                        "[%s] Draft review response emitted no tool call after auto-save; "
                        "submitting current draft.",
                        self.name,
                    )
                    event_bus.publish(
                        "status_message",
                        message=(
                            "Draft already in budget and auto-saved; no tool call "
                            "arrived during review, submitting the current draft."
                        ),
                    )
                    break
                assistant_text = str(msg_dict.get("content") or "").strip()
                if reasoning_content and reasoning_content != assistant_text:
                    assistant_text = (
                        f"{assistant_text}\n{reasoning_content}".strip()
                        if assistant_text
                        else reasoning_content
                    )
                guard_stop_reason = self._build_reasoning_loop_guard_stop_reason(
                    runtime_context=runtime_context,
                    guard_config=reasoning_loop_guard,
                    assistant_text=assistant_text,
                )
                if guard_stop_reason is not None:
                    runtime_context["loop_stop_category"] = "reasoning_loop_guard"
                    runtime_context["loop_stop_reason"] = guard_stop_reason
                    logger.warning(
                        "[%s] Reasoning-loop guard stopped no-tool response: %s",
                        self.name,
                        guard_stop_reason,
                    )
                    event_bus.publish(
                        "status_message",
                        message=(
                            f"{self.name} reasoning-loop guard stopped a "
                            f"repetitive no-tool response: {guard_stop_reason}"
                        ),
                    )
                    break
                if no_tool_call_patience is not None:
                    no_tool_call_count = (
                        int(runtime_context.get("no_tool_call_count", 0)) + 1
                    )
                    runtime_context["no_tool_call_count"] = no_tool_call_count
                    if no_tool_call_count >= no_tool_call_patience:
                        logger.warning(
                            "[%s] No tool calls emitted for %s consecutive turns; stopping loop.",
                            self.name,
                            no_tool_call_count,
                        )
                        break
                continue
            runtime_context["no_tool_call_count"] = 0
            self._reset_reasoning_loop_guard_state(runtime_context)

            executable_tool_calls: list[dict[str, Any]] = []
            executable_call_metadata: list[dict[str, Any]] = []
            for tc in tool_calls:
                tool_name = str(tc.get("name") or "unknown_tool")
                tool_id = str(tc.get("id") or "unknown")
                signature = build_tool_signature(tool_name, tc.get("arguments", {}))
                is_synthetic_id = self._is_synthetic_tool_call_id(tool_id)
                enforce_signature_dedupe = (
                    not replay_dedupe_tools or tool_name in replay_dedupe_tools
                )

                duplicate_reason: str | None = None
                if not is_synthetic_id and tool_id in seen_tool_call_ids:
                    duplicate_reason = "tool_call_id"
                elif enforce_signature_dedupe and signature_window_turns > 0:
                    last_seen_turn = seen_tool_signature_turns.get(signature)
                    if (
                        isinstance(last_seen_turn, int)
                        and (turn - last_seen_turn) <= signature_window_turns
                    ):
                        duplicate_reason = "signature"

                if duplicate_reason is not None:
                    skip_payload: dict[str, Any] = {
                        "status": "skipped_duplicate",
                        "reason": duplicate_reason,
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": json.dumps(skip_payload),
                        }
                    )
                    total_skipped_calls += 1
                    logger.warning(
                        f"[{self.name}] Skipped duplicate tool call id={tool_id} name={tool_name} reason={duplicate_reason}"
                    )
                    event_bus.publish(
                        "tool_result",
                        tool_name=f"{tool_name} (skipped duplicate)",
                        result=json.dumps(skip_payload, indent=2),
                    )
                    continue

                executable_tool_calls.append(tc)
                executable_call_metadata.append(
                    {
                        "tool_id": tool_id,
                        "signature": signature,
                        "is_synthetic_id": is_synthetic_id,
                        "enforce_signature_dedupe": enforce_signature_dedupe,
                    }
                )

            if not executable_tool_calls:
                continue

            if (
                max_tool_calls_per_turn is not None
                and len(executable_tool_calls) > max_tool_calls_per_turn
            ):
                overflow_tool_calls = executable_tool_calls[max_tool_calls_per_turn:]
                executable_tool_calls = executable_tool_calls[:max_tool_calls_per_turn]
                executable_call_metadata = executable_call_metadata[
                    :max_tool_calls_per_turn
                ]
                for tc in overflow_tool_calls:
                    tool_name = str(tc.get("name") or "unknown_tool")
                    tool_id = str(tc.get("id") or "unknown")
                    skip_payload_cap: dict[str, Any] = {
                        "status": "skipped",
                        "reason": "max_tool_calls_per_turn",
                        "max_tool_calls_per_turn": max_tool_calls_per_turn,
                        "tool_name": tool_name,
                        "tool_call_id": tool_id,
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": json.dumps(skip_payload_cap),
                        }
                    )
                    total_skipped_calls += 1
                    logger.warning(
                        f"[{self.name}] Skipped tool call id={tool_id} name={tool_name} due to max_tool_calls_per_turn={max_tool_calls_per_turn}"
                    )
                    event_bus.publish(
                        "tool_result",
                        tool_name=f"{tool_name} (skipped turn cap)",
                        result=json.dumps(skip_payload_cap, indent=2),
                    )

            for meta in executable_call_metadata:
                tool_id = str(meta["tool_id"])
                if not bool(meta["is_synthetic_id"]):
                    seen_tool_call_ids.add(tool_id)
                if (
                    bool(meta["enforce_signature_dedupe"])
                    and signature_window_turns > 0
                ):
                    signature = str(meta["signature"])
                    seen_tool_signature_turns[signature] = turn

            total_executed_calls += len(executable_tool_calls)
            for tc in executable_tool_calls:
                event_bus.publish(
                    "tool_invocation", tool_name=tc["name"], arguments=tc["arguments"]
                )

            should_break, final_result = await self.process_tool_calls(
                executable_tool_calls, messages, runtime_context
            )

            # Ensure all tool calls were answered to prevent API validation errors (e.g. DeepSeek 400)
            answered_ids = {
                m["tool_call_id"] for m in messages if m.get("role") == "tool"
            }
            for tc in executable_tool_calls:
                if tc["id"] not in answered_ids:
                    logger.warning(
                        f"[{self.name}] Tool call {tc['id']} ('{tc['name']}') was not handled. Injecting fallback."
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": tc["name"],
                            "content": f"Error: Tool '{tc['name']}' not recognized or failed to execute.",
                        }
                    )

            if should_break:
                logger.info(
                    f"[{self.name}] Tool loop summary parsed={total_parsed_calls} executed={total_executed_calls} skipped_duplicates={total_skipped_calls}"
                )
                return final_result

        logger.info(
            f"[{self.name}] Tool loop summary parsed={total_parsed_calls} executed={total_executed_calls} skipped_duplicates={total_skipped_calls}"
        )
        return None
