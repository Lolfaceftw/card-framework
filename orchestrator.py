import json

from agents.client import a2a_send_task
from ui import ui


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
        index_task = json.dumps({"action": "index", "segments": segments})
        ui.print_system(f"Indexing {len(segments)} segments in Retrieval Agent...")
        index_response_raw = await a2a_send_task(
            self.retrieval_port, index_task, timeout=self.timeouts["index"]
        )
        try:
            index_result = json.loads(index_response_raw)
            count = index_result.get("count", 0)
            ui.print_status(f"Indexed {count} segments")
            return count
        except json.JSONDecodeError:
            ui.print_error(f"Index response: {index_response_raw[:200]}")
            return 0

    async def run_loop(
        self, min_words: int, max_words: int, max_iterations: int
    ) -> str | None:
        feedback = ""
        draft = ""
        for iteration in range(1, max_iterations + 1):
            ui.print_system(f"--- Iteration {iteration}/{max_iterations} ---")

            # ── Summarizer ──
            summarizer_task = json.dumps(
                {
                    "min_words": min_words,
                    "max_words": max_words,
                    "retrieval_port": self.retrieval_port,
                    "feedback": feedback,
                    "previous_draft": draft if feedback else "",
                }
            )
            draft = await a2a_send_task(
                self.summarizer_port,
                summarizer_task,
                timeout=self.timeouts["summarizer"],
            )

            # ── Critic ──
            critic_task = json.dumps(
                {
                    "draft": draft,
                    "min_words": min_words,
                    "max_words": max_words,
                }
            )
            critic_response_raw = await a2a_send_task(
                self.critic_port, critic_task, timeout=self.timeouts["critic"]
            )

            try:
                critic_verdict = json.loads(critic_response_raw)
            except json.JSONDecodeError:
                ui.print_error("Could not parse critic response, retrying...")
                feedback = "Previous attempt could not be evaluated. Please try again."
                continue

            status = critic_verdict.get("status", "fail")
            word_count = critic_verdict.get("word_count", 0)
            critic_feedback = critic_verdict.get("feedback", "")

            if status == "pass":
                ui.print_status(
                    f"✅ CONVERGENCE at iteration {iteration} (Word count: {word_count})"
                )
                return draft

            feedback = critic_feedback
            ui.print_system(
                f"Iteration {iteration} — not converged ({word_count} words)."
            )
            ui.print_agent_message("Critic Feedback", critic_feedback)

        ui.print_error(f"Max iterations ({max_iterations}) reached.")
        return None
