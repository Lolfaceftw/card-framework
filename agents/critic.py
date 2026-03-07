"""Critic executor for duration-aware summary evaluation."""

from __future__ import annotations

import json
import re

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from agents.client import AgentTaskClient, get_default_agent_client
from agents.dtos import RetrieveTaskRequest
from agents.utils import count_words
from audio_pipeline.calibration import VoiceCloneCalibration
from events import EventBus, get_event_bus
from llm_provider import LLMProvider
from prompt_manager import PromptManager
from summary_xml import DEFAULT_EMO_PRESET, parse_summary_xml


class CriticExecutor(BaseA2AExecutor):
    """Evaluate a draft summary using deterministic checks plus an LLM critic."""

    def __init__(
        self,
        llm: LLMProvider,
        calibration: VoiceCloneCalibration,
        max_tool_turns: int = 5,
        retrieval_port: int = 9012,
        is_embedding_enabled: bool = True,
        agent_client: AgentTaskClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize critic executor and injected collaborators."""
        super().__init__("Critic")
        self.llm = llm
        self.calibration = calibration
        self.max_tool_turns = max_tool_turns
        self.retrieval_port = retrieval_port
        self.is_embedding_enabled = is_embedding_enabled
        self.agent_client = (
            agent_client if agent_client is not None else get_default_agent_client()
        )
        self.event_bus = event_bus if event_bus is not None else get_event_bus()

    def _resolve_target_seconds(self, *, target_seconds: int | None, min_words: int, max_words: int) -> int:
        """Resolve the intended duration target for the critic pass."""
        if target_seconds is not None and int(target_seconds) > 0:
            return int(target_seconds)
        midpoint_words = max(1, int((min_words + max_words) / 2))
        neutral_wpm = self.calibration.wpm_for(
            speaker="",
            emo_preset=DEFAULT_EMO_PRESET,
        )
        return max(1, int(round((midpoint_words / neutral_wpm) * 60.0)))

    def _run_deterministic_checks(
        self,
        *,
        draft: str,
        target_seconds: int,
        duration_tolerance_ratio: float,
    ) -> dict[str, object]:
        """Run non-LLM checks for duration, truncation, and XML coherence."""
        actual_count = count_words(draft)
        failures: list[str] = []
        tolerance_seconds = target_seconds * duration_tolerance_ratio
        min_seconds = target_seconds - tolerance_seconds
        max_seconds = target_seconds + tolerance_seconds

        try:
            turns = parse_summary_xml(draft)
        except ValueError:
            turns = []

        estimated_seconds = self.calibration.estimate_turns_seconds(turns) if turns else 0.0

        if not (min_seconds <= estimated_seconds <= max_seconds):
            failures.append(
                "DURATION: "
                f"{estimated_seconds:.2f}s outside {min_seconds:.2f}-{max_seconds:.2f}s."
            )

        if turns:
            last_block_text = turns[-1].text.strip()
            if last_block_text and not re.search(r"[.!?]\s*$", last_block_text):
                failures.append(
                    "TRUNCATION: The last speaker block does not end with sentence-ending punctuation (. ! ?)."
                )
        else:
            failures.append("COHERENCE: No properly closed speaker-tagged XML blocks found.")

        open_tags = re.findall(r"<([A-Za-z0-9_.-]+)(?:\s[^>]*)?>", draft)
        close_tags = re.findall(r"</([A-Za-z0-9_.-]+)>", draft)
        if open_tags != close_tags:
            failures.append("COHERENCE: Mismatched speaker tags.")

        return {
            "actual_word_count": actual_count,
            "actual_estimated_seconds": round(estimated_seconds, 3),
            "failures": failures,
            "status": "pass" if not failures else "fail",
        }

    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        """Execute one critic task end to end."""
        del context
        from agents.dtos import CriticTaskRequest

        req = CriticTaskRequest.model_validate(task_data)
        draft = req.draft
        target_seconds = self._resolve_target_seconds(
            target_seconds=req.target_seconds,
            min_words=req.min_words,
            max_words=req.max_words,
        )
        duration_tolerance_ratio = float(req.duration_tolerance_ratio or 0.05)
        tolerance_seconds = target_seconds * duration_tolerance_ratio
        full_transcript = req.full_transcript

        self.event_bus.publish("system_message", "Evaluating draft using LLM Critic Loop...")

        system_prompt = PromptManager.get_prompt(
            "critic_system",
            target_seconds=target_seconds,
            target_minutes=round(target_seconds / 60.0, 2),
            duration_tolerance_ratio=duration_tolerance_ratio,
            duration_tolerance_percent=round(duration_tolerance_ratio * 100.0, 2),
            min_seconds=round(target_seconds - tolerance_seconds, 2),
            max_seconds=round(target_seconds + tolerance_seconds, 2),
            is_embedding_enabled=self.is_embedding_enabled,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_deterministic_checks",
                    "description": "Runs hard checks on duration, truncation, and coherence.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "draft_text": {"type": "string"},
                        },
                        "required": ["draft_text"],
                    },
                },
            },
        ]

        if self.is_embedding_enabled:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": "verify_against_transcript",
                        "description": (
                            "Retrieves original transcript segments matching a semantic "
                            "query for factual verification."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The semantic query for transcript verification.",
                                },
                            },
                            "required": ["query"],
                        },
                    },
                }
            )

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "submit_verdict",
                    "description": "Submits the final critique and verdict.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "status": {"type": "string", "enum": ["pass", "fail"]},
                            "actual_word_count": {"type": "integer"},
                            "estimated_seconds": {"type": "number"},
                            "feedback": {"type": "string"},
                        },
                        "required": [
                            "status",
                            "actual_word_count",
                            "estimated_seconds",
                            "feedback",
                        ],
                    },
                },
            }
        )

        user_content = PromptManager.get_prompt("critic_user", draft=draft)
        if not self.is_embedding_enabled:
            user_content += f"\n\n--- FULL TRANSCRIPT ---\n{full_transcript}"

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_content,
            },
        ]

        context_data = {
            "draft": draft,
            "target_seconds": target_seconds,
            "duration_tolerance_ratio": duration_tolerance_ratio,
        }
        final_verdict = await self.run_agent_loop(
            messages, tools, self.max_tool_turns, context_data
        )

        if not final_verdict:
            stats = self._run_deterministic_checks(
                draft=draft,
                target_seconds=target_seconds,
                duration_tolerance_ratio=duration_tolerance_ratio,
            )
            final_verdict = {
                "status": stats["status"],
                "word_count": stats["actual_word_count"],
                "estimated_seconds": stats["actual_estimated_seconds"],
                "feedback": "CRITIC ERROR: LLM failed to submit a structured verdict. "
                + " | ".join(stats["failures"]),
            }

        result_json = json.dumps(final_verdict)
        if final_verdict.get("status") == "pass":
            self.event_bus.publish("status_message", message=f"Final Verdict: {result_json}")
        else:
            self.event_bus.publish(
                "agent_message",
                agent_name=self.name,
                message=f"Final Verdict: {result_json}",
            )

        await self.send_response(result_json, event_queue)

    async def process_tool_calls(
        self, tool_calls: list[dict], messages: list, context_data: dict
    ) -> tuple[bool, dict | None]:
        """Execute critic tool calls."""
        draft = context_data["draft"]
        target_seconds = int(context_data["target_seconds"])
        duration_tolerance_ratio = float(context_data["duration_tolerance_ratio"])
        final_verdict = None

        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]

            if name == "run_deterministic_checks":
                llm_draft_text = str(args.get("draft_text", ""))
                if llm_draft_text.strip() and llm_draft_text.strip() != draft.strip():
                    self.event_bus.publish(
                        "system_message",
                        "Ignoring LLM-supplied draft_text and using finalized draft from task payload.",
                    )
                results = self._run_deterministic_checks(
                    draft=draft,
                    target_seconds=target_seconds,
                    duration_tolerance_ratio=duration_tolerance_ratio,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": name,
                        "content": json.dumps(results),
                    }
                )
                self.event_bus.publish(
                    "tool_result",
                    tool_name="run_deterministic_checks",
                    result=(
                        "Status: "
                        f"{results.get('status')} ({len(results.get('failures', []))} failures)"
                    ),
                )
            elif name == "verify_against_transcript":
                query_text = args.get("query", "")
                retrieve_task = RetrieveTaskRequest(
                    action="retrieve", query=query_text, top_k=10
                )
                raw_resp = await self.agent_client.send_task(
                    self.retrieval_port, retrieve_task
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": name,
                        "content": raw_resp,
                    }
                )
                self.event_bus.publish(
                    "tool_result",
                    tool_name="verify_against_transcript",
                    result="Retrieved segments for verification.",
                )
            elif name == "submit_verdict":
                final_verdict = {
                    "status": args["status"],
                    "word_count": args["actual_word_count"],
                    "estimated_seconds": args.get("estimated_seconds", 0.0),
                    "feedback": args["feedback"],
                }
                break

        if final_verdict:
            return True, final_verdict
        return False, None
