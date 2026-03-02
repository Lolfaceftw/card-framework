import json
import time
from typing import TypedDict

from agents.client import AgentTaskClient, get_default_agent_client
from agents.dtos import (
    CriticTaskRequest,
    CriticTaskResponse,
    IndexTaskRequest,
    SummarizerTaskRequest,
)
from agents.loop_context import SummarizerLoopMemory
from events import EventBus, get_event_bus
from orchestration.transcript import TranscriptLike, coerce_transcript


class IterationDiagnostic(TypedDict):
    """Per-iteration diagnostic details emitted by the orchestrator loop."""

    iteration: int
    summarizer_latency_seconds: float
    critic_latency_seconds: float
    critic_status: str
    critic_word_count: int
    critic_feedback: str
    unresolved_issue_count: int
    persisted_issue_signatures: list[str]
    stagnation_detected: bool


class LoopDiagnostics(TypedDict):
    """Structured metadata for a full summarizer-critic loop execution."""

    converged: bool
    iterations_run: int
    max_iterations: int
    draft: str
    final_status: str
    final_word_count: int
    final_feedback: str
    iteration_details: list[IterationDiagnostic]


class Orchestrator:
    """Drive the Summarizer and Critic loop."""

    _DEFAULT_TIMEOUTS = {
        "summarizer": 180,
        "critic": 180,
        "retrieval": 60,
        "index": 300,
    }
    _FULL_TRANSCRIPT_TIMEOUT_FLOORS = {
        "summarizer": 900.0,
        "critic": 300.0,
    }

    def __init__(
        self,
        retrieval_port: int,
        summarizer_port: int,
        critic_port: int,
        timeouts: dict | None = None,
        agent_client: AgentTaskClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize ports, timeout policy, and injected dependencies."""
        self.retrieval_port = retrieval_port
        self.summarizer_port = summarizer_port
        self.critic_port = critic_port
        self.timeouts = {**self._DEFAULT_TIMEOUTS, **(timeouts or {})}
        self.agent_client = (
            agent_client if agent_client is not None else get_default_agent_client()
        )
        self.event_bus = event_bus if event_bus is not None else get_event_bus()

    def _resolve_stage_timeout(self, stage: str, full_transcript_text: str) -> float:
        """
        Resolve per-stage timeout with full-transcript safety floors.

        Full-transcript mode bypasses retrieval and sends much larger prompts,
        which can legitimately take longer than the baseline timeout.
        """
        timeout = float(self.timeouts.get(stage, self._DEFAULT_TIMEOUTS[stage]))
        if full_transcript_text.strip():
            floor = self._FULL_TRANSCRIPT_TIMEOUT_FLOORS.get(stage)
            if floor is not None and timeout < floor:
                return floor
        return timeout

    async def index_transcript(self, transcript: TranscriptLike) -> int:
        """Index transcript segments in retrieval service and return indexed count."""
        transcript_dto = coerce_transcript(transcript)
        segments = transcript_dto.retrieval_segments()
        index_task = IndexTaskRequest(action="index", segments=segments)
        self.event_bus.publish(
            "system_message",
            message=f"Indexing {len(segments)} segments in Retrieval Agent...",
        )
        index_response_raw = await self.agent_client.send_task(
            self.retrieval_port,
            index_task,
            timeout=self.timeouts["index"],
            metadata={"component": "orchestrator", "stage": "index"},
        )
        try:
            index_result = json.loads(index_response_raw)
            count = index_result.get("count", 0)
            self.event_bus.publish("status_message", f"Indexed {count} segments")
            return count
        except json.JSONDecodeError:
            self.event_bus.publish(
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
        """
        Run the iterative summarizer-critic loop and return the converged draft.

        Args:
            min_words: Minimum accepted word count.
            max_words: Maximum accepted word count.
            max_iterations: Hard cap on iterations before failure.
            full_transcript_text: Full transcript text used when retrieval is disabled.

        Returns:
            The converged XML draft, or ``None`` if convergence was not reached.
        """
        diagnostics = await self.run_loop_with_diagnostics(
            min_words=min_words,
            max_words=max_words,
            max_iterations=max_iterations,
            full_transcript_text=full_transcript_text,
        )
        if diagnostics["converged"]:
            return diagnostics["draft"]
        return None

    async def run_summarizer_once(
        self,
        *,
        min_words: int,
        max_words: int,
        full_transcript_text: str = "",
    ) -> str:
        """
        Run a single summarizer pass without critic feedback iterations.

        Args:
            min_words: Minimum target word count sent to summarizer.
            max_words: Maximum target word count sent to summarizer.
            full_transcript_text: Full transcript text when retrieval is disabled.

        Returns:
            Draft summary generated by the summarizer agent.
        """
        self.event_bus.publish("system_message", "Running single-pass summarizer stage...")
        summarizer_timeout = self._resolve_stage_timeout(
            "summarizer", full_transcript_text
        )
        summarizer_task = SummarizerTaskRequest(
            min_words=min_words,
            max_words=max_words,
            retrieval_port=self.retrieval_port,
            feedback="",
            previous_draft="",
            loop_context="",
            full_transcript=full_transcript_text,
        )
        return await self.agent_client.send_task(
            self.summarizer_port,
            summarizer_task,
            timeout=summarizer_timeout,
            metadata={"component": "orchestrator", "stage": "summarizer_single_pass"},
        )

    async def run_critic_once(
        self,
        *,
        draft: str,
        min_words: int,
        max_words: int,
        full_transcript_text: str = "",
    ) -> CriticTaskResponse:
        """
        Evaluate one existing draft with the critic.

        Args:
            draft: Pre-existing draft summary to evaluate.
            min_words: Minimum accepted word count.
            max_words: Maximum accepted word count.
            full_transcript_text: Full transcript text when retrieval is disabled.

        Returns:
            Parsed critic verdict payload.

        Raises:
            RuntimeError: When critic output is invalid JSON.
        """
        self.event_bus.publish("system_message", "Running critic-only evaluation stage...")
        critic_timeout = self._resolve_stage_timeout("critic", full_transcript_text)
        critic_task = CriticTaskRequest(
            draft=draft,
            min_words=min_words,
            max_words=max_words,
            full_transcript=full_transcript_text,
        )
        critic_response_raw = await self.agent_client.send_task(
            self.critic_port,
            critic_task,
            timeout=critic_timeout,
            metadata={"component": "orchestrator", "stage": "critic_single_pass"},
        )
        try:
            return CriticTaskResponse.model_validate_json(critic_response_raw)
        except Exception as exc:
            raise RuntimeError("Critic returned invalid JSON response.") from exc

    async def run_loop_with_diagnostics(
        self,
        min_words: int,
        max_words: int,
        max_iterations: int,
        full_transcript_text: str = "",
    ) -> LoopDiagnostics:
        """
        Run the loop and return structured per-iteration diagnostics.

        Args:
            min_words: Minimum accepted word count.
            max_words: Maximum accepted word count.
            max_iterations: Hard cap on iterations before failure.
            full_transcript_text: Full transcript text used when retrieval is disabled.

        Returns:
            A diagnostics dictionary including convergence state and iteration details.
        """
        feedback = ""
        draft = ""
        iteration_details: list[IterationDiagnostic] = []
        loop_memory = SummarizerLoopMemory(min_words=min_words, max_words=max_words)
        summarizer_timeout = self._resolve_stage_timeout(
            "summarizer", full_transcript_text
        )
        critic_timeout = self._resolve_stage_timeout("critic", full_transcript_text)

        for iteration in range(1, max_iterations + 1):
            self.event_bus.publish(
                "system_message", f"--- Iteration {iteration}/{max_iterations} ---"
            )
            self.event_bus.publish(
                "orchestrator_iteration_started",
                iteration=iteration,
                max_iterations=max_iterations,
            )

            loop_context = loop_memory.to_compact_prompt_block()
            summarizer_task = SummarizerTaskRequest(
                min_words=min_words,
                max_words=max_words,
                retrieval_port=self.retrieval_port,
                feedback=feedback,
                previous_draft=draft if feedback else "",
                loop_context=loop_context,
                full_transcript=full_transcript_text,
            )
            summarizer_started = time.perf_counter()
            draft = await self.agent_client.send_task(
                self.summarizer_port,
                summarizer_task,
                timeout=summarizer_timeout,
                metadata={
                    "component": "orchestrator",
                    "stage": "summarizer",
                    "iteration": iteration,
                },
            )
            summarizer_latency = round(time.perf_counter() - summarizer_started, 3)

            critic_task = CriticTaskRequest(
                draft=draft,
                min_words=min_words,
                max_words=max_words,
                full_transcript=full_transcript_text,
            )
            critic_started = time.perf_counter()
            critic_response_raw = await self.agent_client.send_task(
                self.critic_port,
                critic_task,
                timeout=critic_timeout,
                metadata={
                    "component": "orchestrator",
                    "stage": "critic",
                    "iteration": iteration,
                },
            )
            critic_latency = round(time.perf_counter() - critic_started, 3)

            try:
                critic_verdict = CriticTaskResponse.model_validate_json(
                    critic_response_raw
                )
            except Exception as exc:
                self.event_bus.publish(
                    "error_message", f"Critic failed or returned invalid JSON: {exc}"
                )
                memory_update = loop_memory.update_from_critic(
                    iteration=iteration,
                    critic_status="invalid_json",
                    feedback=(
                        "critic invalid json response; repair summary quality and produce "
                        "a valid critic-evaluable draft"
                    ),
                    word_count=None,
                )
                self.event_bus.publish(
                    "orchestrator_loop_memory_updated",
                    iteration=iteration,
                    unresolved_issue_count=memory_update.unresolved_issue_count,
                    persisted_issue_signatures=memory_update.persisted_issue_signatures,
                    stagnation_detected=memory_update.stagnation_detected,
                    early_stop_recommended=memory_update.early_stop_recommended,
                )
                for signature in memory_update.persisted_issue_signatures:
                    self.event_bus.publish(
                        "orchestrator_loop_issue_persisted",
                        iteration=iteration,
                        signature=signature,
                    )
                iteration_details.append(
                    {
                        "iteration": iteration,
                        "summarizer_latency_seconds": summarizer_latency,
                        "critic_latency_seconds": critic_latency,
                        "critic_status": "invalid_json",
                        "critic_word_count": 0,
                        "critic_feedback": str(exc),
                        "unresolved_issue_count": memory_update.unresolved_issue_count,
                        "persisted_issue_signatures": memory_update.persisted_issue_signatures,
                        "stagnation_detected": memory_update.stagnation_detected,
                    }
                )
                self.event_bus.publish(
                    "orchestrator_iteration_completed",
                    iteration=iteration,
                    converged=False,
                    critic_status="invalid_json",
                    word_count=0,
                    summarizer_latency_seconds=summarizer_latency,
                    critic_latency_seconds=critic_latency,
                )
                feedback = (
                    "Previous attempt could not be evaluated due to a system error. "
                    "Please try again."
                )
                if memory_update.stagnation_detected and memory_update.strategy_shift_hint:
                    feedback = f"{feedback}\n\n{memory_update.strategy_shift_hint}"
                if memory_update.early_stop_recommended:
                    self.event_bus.publish(
                        "error_message",
                        (
                            "Detected persistent loop stagnation. Ending before "
                            f"max_iterations ({max_iterations})."
                        ),
                    )
                    return {
                        "converged": False,
                        "iterations_run": iteration,
                        "max_iterations": max_iterations,
                        "draft": draft,
                        "final_status": "non_convergent_stagnation",
                        "final_word_count": 0,
                        "final_feedback": feedback,
                        "iteration_details": iteration_details,
                    }
                continue

            status = critic_verdict.status
            word_count = critic_verdict.word_count
            critic_feedback = critic_verdict.feedback
            unresolved_issue_count = 0
            persisted_issue_signatures: list[str] = []
            stagnation_detected = False
            if status != "pass":
                memory_update = loop_memory.update_from_critic(
                    iteration=iteration,
                    critic_status=status,
                    feedback=critic_feedback,
                    word_count=word_count,
                )
                unresolved_issue_count = memory_update.unresolved_issue_count
                persisted_issue_signatures = memory_update.persisted_issue_signatures
                stagnation_detected = memory_update.stagnation_detected
                self.event_bus.publish(
                    "orchestrator_loop_memory_updated",
                    iteration=iteration,
                    unresolved_issue_count=unresolved_issue_count,
                    persisted_issue_signatures=persisted_issue_signatures,
                    stagnation_detected=stagnation_detected,
                    early_stop_recommended=memory_update.early_stop_recommended,
                )
                for signature in persisted_issue_signatures:
                    self.event_bus.publish(
                        "orchestrator_loop_issue_persisted",
                        iteration=iteration,
                        signature=signature,
                    )
            iteration_details.append(
                {
                    "iteration": iteration,
                    "summarizer_latency_seconds": summarizer_latency,
                    "critic_latency_seconds": critic_latency,
                    "critic_status": status,
                    "critic_word_count": word_count,
                    "critic_feedback": critic_feedback,
                    "unresolved_issue_count": unresolved_issue_count,
                    "persisted_issue_signatures": persisted_issue_signatures,
                    "stagnation_detected": stagnation_detected,
                }
            )

            if status == "pass":
                self.event_bus.publish(
                    "status_message",
                    message=(
                        "CONVERGENCE at iteration "
                        f"{iteration} (Word count: {word_count})"
                    ),
                )
                self.event_bus.publish(
                    "orchestrator_iteration_completed",
                    iteration=iteration,
                    converged=True,
                    critic_status=status,
                    word_count=word_count,
                    summarizer_latency_seconds=summarizer_latency,
                    critic_latency_seconds=critic_latency,
                )
                return {
                    "converged": True,
                    "iterations_run": iteration,
                    "max_iterations": max_iterations,
                    "draft": draft,
                    "final_status": status,
                    "final_word_count": word_count,
                    "final_feedback": critic_feedback,
                    "iteration_details": iteration_details,
                }

            feedback = critic_feedback
            if status != "pass":
                if memory_update.stagnation_detected and memory_update.strategy_shift_hint:
                    feedback = f"{feedback}\n\n{memory_update.strategy_shift_hint}"
                if memory_update.early_stop_recommended:
                    self.event_bus.publish(
                        "error_message",
                        (
                            "Detected persistent loop stagnation. Ending before "
                            f"max_iterations ({max_iterations})."
                        ),
                    )
                    return {
                        "converged": False,
                        "iterations_run": iteration,
                        "max_iterations": max_iterations,
                        "draft": draft,
                        "final_status": "non_convergent_stagnation",
                        "final_word_count": word_count,
                        "final_feedback": feedback,
                        "iteration_details": iteration_details,
                    }
            self.event_bus.publish(
                "system_message",
                f"Iteration {iteration} - not converged ({word_count} words).",
            )
            self.event_bus.publish("agent_message", "Critic Feedback", critic_feedback)
            self.event_bus.publish(
                "orchestrator_iteration_completed",
                iteration=iteration,
                converged=False,
                critic_status=status,
                word_count=word_count,
                summarizer_latency_seconds=summarizer_latency,
                critic_latency_seconds=critic_latency,
            )

        self.event_bus.publish("error_message", f"Max iterations ({max_iterations}) reached.")
        return {
            "converged": False,
            "iterations_run": max_iterations,
            "max_iterations": max_iterations,
            "draft": "",
            "final_status": "max_iterations_reached",
            "final_word_count": 0,
            "final_feedback": feedback,
            "iteration_details": iteration_details,
        }

