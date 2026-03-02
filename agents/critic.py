import json
import re

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from agents.client import AgentTaskClient, get_default_agent_client
from agents.dtos import RetrieveTaskRequest
from agents.utils import count_words
from events import EventBus, get_event_bus
from llm_provider import LLMProvider
from prompt_manager import PromptManager


class CriticExecutor(BaseA2AExecutor):
    """
    A2A executor that evaluates a draft summary using an LLM critic
    supported by deterministic verification tools.
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_tool_turns: int = 5,
        retrieval_port: int = 9012,
        is_embedding_enabled: bool = True,
        agent_client: AgentTaskClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize critic executor and injected collaborators."""
        super().__init__("Critic")
        self.llm = llm
        self.max_tool_turns = max_tool_turns
        self.retrieval_port = retrieval_port
        self.is_embedding_enabled = is_embedding_enabled
        self.agent_client = (
            agent_client if agent_client is not None else get_default_agent_client()
        )
        self.event_bus = event_bus if event_bus is not None else get_event_bus()

    def _run_deterministic_checks(
        self, draft: str, min_words: int, max_words: int
    ) -> dict:
        actual_count = count_words(draft)
        failures = []

        # -- Check 1: Word count --
        if not (min_words <= actual_count <= max_words):
            failures.append(
                f"WORD COUNT: {actual_count} words, outside {min_words}-{max_words}."
            )

        # -- Check 2: Truncation detection --
        speaker_blocks = re.findall(
            r"<SPEAKER_\d+>(.*?)</SPEAKER_\d+>", draft, re.DOTALL
        )
        if speaker_blocks:
            last_block_text = speaker_blocks[-1].strip()
            if last_block_text and not re.search(r"[.!?]\s*$", last_block_text):
                failures.append(
                    "TRUNCATION: The last speaker block does not end with sentence-ending punctuation (. ! ?)."
                )
        else:
            failures.append("TRUNCATION: No properly closed <SPEAKER_XX> blocks found.")

        # -- Check 3: Coherence --
        open_tags = re.findall(r"<(SPEAKER_\d+)>", draft)
        close_tags = re.findall(r"</(SPEAKER_\d+)>", draft)
        if open_tags != close_tags:
            failures.append("COHERENCE: Mismatched speaker tags.")

        return {
            "actual_word_count": actual_count,
            "failures": failures,
            "status": "pass" if not failures else "fail",
        }

    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        from agents.dtos import CriticTaskRequest

        req = CriticTaskRequest.model_validate(task_data)
        draft = req.draft
        min_words = req.min_words
        max_words = req.max_words
        full_transcript = req.full_transcript

        self.event_bus.publish("system_message", "Evaluating draft using LLM Critic Loop...")

        system_prompt = PromptManager.get_prompt(
            "critic_system",
            min_words=min_words,
            max_words=max_words,
            is_embedding_enabled=self.is_embedding_enabled,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "run_deterministic_checks",
                    "description": "Runs hard checks on word count, truncation, coherence, and naturalness.",
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
                        "description": "Retrieves original transcript segments matching a semantic query for factual verification.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The semantic query (e.g., 'details about server architecture').",
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
                            "feedback": {"type": "string"},
                        },
                        "required": ["status", "actual_word_count", "feedback"],
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
            "min_words": min_words,
            "max_words": max_words,
        }
        final_verdict = await self.run_agent_loop(
            messages, tools, self.max_tool_turns, context_data
        )

        if not final_verdict:
            # Fallback if LLM failed to call submit_verdict
            stats = self._run_deterministic_checks(draft, min_words, max_words)
            final_verdict = {
                "status": stats["status"],
                "word_count": stats["actual_word_count"],
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
        draft = context_data["draft"]
        min_words = context_data["min_words"]
        max_words = context_data["max_words"]
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
                results = self._run_deterministic_checks(draft, min_words, max_words)
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
                    result=f"Status: {results.get('status')} ({len(results.get('failures', []))} failures)",
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
                    "feedback": args["feedback"],
                }
                break

        if final_verdict:
            return True, final_verdict
        return False, None

