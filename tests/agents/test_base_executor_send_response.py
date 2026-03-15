"""Regression tests for A2A response completion semantics."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from a2a.types import Message, TaskState, TaskStatusUpdateEvent

from card_framework.agents.base import BaseA2AExecutor


class _RecordingQueue:
    def __init__(self) -> None:
        self.events: list[object] = []

    async def enqueue_event(self, event: object) -> None:
        self.events.append(event)


class _TestExecutor(BaseA2AExecutor):
    def __init__(self) -> None:
        super().__init__("Test")

    async def handle_task(self, task_data, context, event_queue) -> None:  # noqa: ANN001
        del task_data, context, event_queue


def test_send_response_completes_a2a_tasks_with_message() -> None:
    executor = _TestExecutor()
    queue = _RecordingQueue()
    context = SimpleNamespace(task_id="task-1", context_id="ctx-1")

    asyncio.run(executor.send_response("hello world", context, queue))

    assert len(queue.events) == 1
    event = queue.events[0]
    assert isinstance(event, TaskStatusUpdateEvent)
    assert event.final is True
    assert event.status.state == TaskState.completed
    assert event.status.message is not None
    assert event.status.message.parts[0].root.text == "hello world"


def test_send_response_falls_back_to_plain_message_without_task_context() -> None:
    executor = _TestExecutor()
    queue = _RecordingQueue()
    context = SimpleNamespace(task_id=None, context_id=None)

    asyncio.run(executor.send_response("hello world", context, queue))

    assert len(queue.events) == 1
    event = queue.events[0]
    assert isinstance(event, Message)
    assert event.parts[0].root.text == "hello world"
