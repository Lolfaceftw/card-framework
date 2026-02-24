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
