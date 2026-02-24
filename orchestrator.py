import json

from agents.client import agent_client
from agents.dtos import (
    CriticTaskRequest,
    CriticTaskResponse,
    IndexTaskRequest,
    SummarizerTaskRequest,
)
from events import event_bus


class Orchestrator:
    """
    Drives the Summarizer ↔ Critic loop.
    """

    _DEFAULT_TIMEOUTS = {
        "summarizer": 180,
        "critic": 180,
        "retrieval": 60,
        "index": 300,
    }

    def __init__(
        self,
        retrieval_port: int,
        summarizer_port: int,
        critic_port: int,
        timeouts: dict | None = None,
    ):
        self.retrieval_port = retrieval_port
        self.summarizer_port = summarizer_port
        self.critic_port = critic_port
        self.timeouts = {**self._DEFAULT_TIMEOUTS, **(timeouts or {})}

    async def index_transcript(self, transcript_json: dict) -> int:
        segments = transcript_json.get("segments", [])
        index_task = IndexTaskRequest(action="index", segments=segments)
        event_bus.publish(
            "system_message",
            message=f"Indexing {len(segments)} segments in Retrieval Agent...",
        )
        index_response_raw = await agent_client.send_task(
            self.retrieval_port, index_task, timeout=self.timeouts["index"]
        )
        try:
            index_result = json.loads(index_response_raw)
            count = index_result.get("count", 0)
            event_bus.publish("status_message", f"Indexed {count} segments")
            return count
        except json.JSONDecodeError:
            event_bus.publish(
                "error_message", f"Index response: {index_response_raw[:200]}"
            )
            return 0

    async def run_loop(
        self,
        min_words: int,
        max_words: int,
        max_iterations: int,
        full_transcript_text: str = "",
    ) -> str | None:
        feedback = ""
        draft = ""
        for iteration in range(1, max_iterations + 1):
            event_bus.publish(
                "system_message", f"--- Iteration {iteration}/{max_iterations} ---"
            )

            # ── Summarizer ──
            summarizer_task = SummarizerTaskRequest(
                min_words=min_words,
                max_words=max_words,
                retrieval_port=self.retrieval_port,
                feedback=feedback,
                previous_draft=draft if feedback else "",
                full_transcript=full_transcript_text,
            )
            draft = await agent_client.send_task(
                self.summarizer_port,
                summarizer_task,
                timeout=self.timeouts["summarizer"],
            )

            # ── Critic ──
            critic_task = CriticTaskRequest(
                draft=draft,
                min_words=min_words,
                max_words=max_words,
                full_transcript=full_transcript_text,
            )
            critic_response_raw = await agent_client.send_task(
                self.critic_port, critic_task, timeout=self.timeouts["critic"]
            )

            try:
                critic_verdict = CriticTaskResponse.model_validate_json(
                    critic_response_raw
                )
            except Exception:
                event_bus.publish(
                    "error_message", "Could not parse critic response, retrying..."
                )
                feedback = "Previous attempt could not be evaluated. Please try again."
                continue

            status = critic_verdict.status
            word_count = critic_verdict.word_count
            critic_feedback = critic_verdict.feedback

            if status == "pass":
                event_bus.publish(
                    "status_message",
                    message=f"✅ CONVERGENCE at iteration {iteration} (Word count: {word_count})",
                )
                return draft

            feedback = critic_feedback
            event_bus.publish(
                "system_message",
                f"Iteration {iteration} — not converged ({word_count} words).",
            )
            event_bus.publish("agent_message", "Critic Feedback", critic_feedback)

        event_bus.publish(
            "error_message", f"Max iterations ({max_iterations}) reached."
        )
        return None
