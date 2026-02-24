import json
import re

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from embeddings import TranscriptIndex
from ui import ui


class InfoRetrievalExecutor(BaseA2AExecutor):
    """
    A2A executor that indexes transcript segments and retrieves relevant ones
    via embedding-based MMR selection.
    """

    def __init__(self, transcript_index: TranscriptIndex):
        super().__init__("InfoRetrieval")
        self.index = transcript_index

    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        action = task_data.get("action", "")

        if action == "index":
            segments = task_data.get("segments", [])
            count = self.index.index_segments(segments)
            result = json.dumps({"status": "ok", "count": count})
            ui.print_system(f"Indexed {count} segments")

        elif action == "retrieve":
            query = task_data.get("query", "main topics and key arguments")
            top_k = task_data.get("top_k", 15)
            lambda_param = task_data.get("lambda", 0.7)

            selected = self.index.retrieve_mmr(
                query=query, top_k=top_k, lambda_param=lambda_param
            )

            total_words = sum(
                len(re.sub(r"<[^>]+>", "", seg.get("text", "")).split())
                for seg in selected
            )
            result = json.dumps(
                {
                    "segments": selected,
                    "total_words": total_words,
                }
            )
            ui.print_system(
                f"Retrieved {len(selected)} segments "
                f"({total_words} words) for query: {query[:60]}..."
            )

        else:
            result = json.dumps(
                {"error": f"Unknown action: {action}. Use 'index' or 'retrieve'."}
            )
            ui.print_error(f"Unknown action: {action}")

        await self.send_response(result, event_queue)
