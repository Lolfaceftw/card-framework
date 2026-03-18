import json
import re

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from card_framework.agents.base import BaseA2AExecutor
from card_framework.retrieval.embeddings import (
    NamespacedTranscriptIndexRegistry,
    TranscriptIndex,
)
from card_framework.shared.events import event_bus


class InfoRetrievalExecutor(BaseA2AExecutor):
    """
    A2A executor that indexes transcript segments and retrieves relevant ones
    via embedding-based MMR selection.
    """

    def __init__(
        self,
        transcript_index: TranscriptIndex | NamespacedTranscriptIndexRegistry,
    ):
        super().__init__("InfoRetrieval")
        self.index = transcript_index

    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        from card_framework.agents.dtos import (
            ClearTaskRequest,
            ClearTaskResponse,
            IndexTaskRequest,
            IndexTaskResponse,
            RetrieveTaskRequest,
            RetrieveTaskResponse,
        )

        action = task_data.get("action", "")

        if action == "index":
            req = IndexTaskRequest.model_validate(task_data)
            if isinstance(self.index, NamespacedTranscriptIndexRegistry):
                count = self.index.index_segments(
                    [s.model_dump() for s in req.segments],
                    namespace=req.namespace,
                )
            else:
                count = self.index.index_segments([s.model_dump() for s in req.segments])
            res = IndexTaskResponse(status="ok", count=count)
            result = res.model_dump_json()
            event_bus.publish("system_message", message=f"Indexed {count} segments")

        elif action == "retrieve":
            req = RetrieveTaskRequest.model_validate(task_data)
            if isinstance(self.index, NamespacedTranscriptIndexRegistry):
                selected = self.index.retrieve_mmr(
                    query=req.query,
                    namespace=req.namespace,
                    top_k=req.top_k,
                    lambda_param=req.lambda_param,
                )
            else:
                selected = self.index.retrieve_mmr(
                    query=req.query,
                    top_k=req.top_k,
                lambda_param=req.lambda_param,
            )

            total_words = sum(
                len(re.sub(r"<[^>]+>", "", seg.get("text", "")).split())
                for seg in selected
            )
            res = RetrieveTaskResponse(segments=selected, total_words=total_words)
            result = res.model_dump_json()

            event_bus.publish(
                "system_message",
                message=f"Retrieved {len(selected)} segments "
                f"({total_words} words) for query: {req.query[:60]}...",
            )
            event_bus.publish(
                "retrieval_stats",
                source="retrieval_agent",
                query=req.query,
                num_segments=len(selected),
                total_words=total_words,
                top_k=req.top_k,
                lambda_param=req.lambda_param,
            )

        elif action == "clear":
            req = ClearTaskRequest.model_validate(task_data)
            if isinstance(self.index, NamespacedTranscriptIndexRegistry):
                self.index.clear_namespace(req.namespace)
            else:
                self.index.clear()
            res = ClearTaskResponse(status="ok")
            result = res.model_dump_json()
            event_bus.publish(
                "system_message",
                message=f"Cleared retrieval namespace {req.namespace!r}",
            )

        else:
            result = json.dumps(
                {
                    "error": (
                        f"Unknown action: {action}. "
                        "Use 'index', 'retrieve', or 'clear'."
                    )
                }
            )
            event_bus.publish("error_message", message=f"Unknown action: {action}")

        await self.send_response(result, context, event_queue)
