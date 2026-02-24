import json
from abc import ABC, abstractmethod

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from logger_utils import logger


class BaseA2AExecutor(AgentExecutor, ABC):
    """
    Base class for A2A executors to standardize common logic.
    """

    def __init__(self, name: str):
        self.name = name

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
        self, tool_calls: list[dict], messages: list, context_data: dict
    ) -> tuple[bool, dict | None]:
        """Subclasses should implement to dispatch tools and append results to messages. Returns (should_break, final_result)"""
        return False, None

    async def run_agent_loop(
        self,
        messages: list,
        tools: list,
        max_turns: int,
        context_data: dict | None = None,
    ) -> dict | None:
        from agents.parsers import get_default_parser
        from events import event_bus

        parser = get_default_parser()

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
            if not tool_calls:
                continue

            for tc in tool_calls:
                event_bus.publish(
                    "tool_invocation", tool_name=tc["name"], arguments=tc["arguments"]
                )

            should_break, final_result = await self.process_tool_calls(
                tool_calls, messages, context_data or {}
            )
            if should_break:
                return final_result

        return None
