import json
from abc import ABC, abstractmethod
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from agents.tool_call_utils import build_tool_signature
from logger_utils import logger


class BaseA2AExecutor(AgentExecutor, ABC):
    """
    Base class for A2A executors to standardize common logic.
    """

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def _is_synthetic_tool_call_id(tool_call_id: str) -> bool:
        """Return True for parser-generated fallback IDs that can repeat across turns."""
        return tool_call_id.startswith("xml_fallback_") or tool_call_id.startswith(
            "text_fallback_"
        )

    @staticmethod
    def _resolve_signature_dedupe_window(runtime_context: dict[str, Any]) -> int:
        """Return the configured signature dedupe window in turns (0 disables)."""
        raw_value = runtime_context.get("signature_dedupe_window_turns", 0)
        if isinstance(raw_value, int) and raw_value >= 0:
            return raw_value
        return 0

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        raw_input = context.get_user_input()
        logger.info(f"[{self.name}] Received task: {raw_input}")

        if not raw_input:
            raise ValueError(f"No task provided to {self.name}")

        try:
            task_data = json.loads(raw_input)
        except json.JSONDecodeError:
            raise ValueError(f"{self.name} received non-JSON task: {raw_input[:200]}")

        try:
            await self.handle_task(task_data, context, event_queue)
        except Exception as e:
            logger.error(f"[{self.name}] Error handling task: {e}")
            raise

    @abstractmethod
    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Subclasses implement this to handle the parsed JSON task."""
        pass

    async def send_response(self, result: str, event_queue: EventQueue) -> None:
        """Helper to send a text response back."""
        logger.info(f"[{self.name}] Result: {result}")
        await event_queue.enqueue_event(new_agent_text_message(result))

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
        from agents.parsers import get_default_parser_with_options
        from events import event_bus

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
        if isinstance(max_tool_calls_per_turn_raw, int) and max_tool_calls_per_turn_raw > 0:
            max_tool_calls_per_turn = max_tool_calls_per_turn_raw

        runtime_context["seen_tool_call_ids"] = seen_tool_call_ids
        runtime_context["seen_tool_signature_turns"] = seen_tool_signature_turns
        runtime_context.setdefault("tool_replay_window", "run")
        enable_extended_text_tool_parser = bool(
            runtime_context.get("enable_extended_text_tool_parser", False)
        )

        total_parsed_calls = 0
        total_executed_calls = 0
        total_skipped_calls = 0

        parser = get_default_parser_with_options(
            enable_extended_text_fallback=enable_extended_text_tool_parser
        )

        for turn in range(max_turns):
            response_message = self.llm.chat(messages=messages, tools=tools)
            if hasattr(response_message, "model_dump"):
                msg_dict = response_message.model_dump()
                msg_dict = {
                    k: v
                    for k, v in msg_dict.items()
                    if k in ("role", "content", "tool_calls")
                }
            else:
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
            messages.append(msg_dict)

            tool_calls = parser.parse(msg_dict)
            total_parsed_calls += len(tool_calls)
            if not tool_calls:
                continue

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
                    skip_payload = {
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
                    skip_payload = {
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
                            "content": json.dumps(skip_payload),
                        }
                    )
                    total_skipped_calls += 1
                    logger.warning(
                        f"[{self.name}] Skipped tool call id={tool_id} name={tool_name} due to max_tool_calls_per_turn={max_tool_calls_per_turn}"
                    )
                    event_bus.publish(
                        "tool_result",
                        tool_name=f"{tool_name} (skipped turn cap)",
                        result=json.dumps(skip_payload, indent=2),
                    )

            for meta in executable_call_metadata:
                tool_id = str(meta["tool_id"])
                if not bool(meta["is_synthetic_id"]):
                    seen_tool_call_ids.add(tool_id)
                if bool(meta["enforce_signature_dedupe"]) and signature_window_turns > 0:
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
