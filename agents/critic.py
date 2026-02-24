import json
import re

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from agents.client import a2a_send_task
from agents.utils import count_words
from llm_provider import LLMProvider
from prompt_manager import PromptManager
from ui import ui


class CriticExecutor(BaseA2AExecutor):
    """
    A2A executor that evaluates a draft summary using an LLM critic
    supported by deterministic verification tools.
    """

    def __init__(
        self, llm: LLMProvider, max_tool_turns: int = 5, retrieval_port: int = 9012
    ):
        super().__init__("Critic")
        self.llm = llm
        self.max_tool_turns = max_tool_turns
        self.retrieval_port = retrieval_port

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
        draft = task_data.get("draft", "")
        min_words = task_data.get("min_words", 50)
        max_words = task_data.get("max_words", 100)

        ui.print_system("Evaluating draft using LLM Critic Loop...")

        system_prompt = PromptManager.get_prompt(
            "critic_system", min_words=min_words, max_words=max_words
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
            },
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
            },
        ]

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": PromptManager.get_prompt("critic_user", draft=draft),
            },
        ]

        final_verdict = None

        # Loop for up to max_tool_turns to allow for check -> thought -> verdict
        for _ in range(self.max_tool_turns):
            response = self.llm.chat(messages=messages, tools=tools)

            # Convert to dict for logging and subsequent calls to avoid serialization errors
            if hasattr(response, "model_dump"):
                msg_dict = response.model_dump()
                msg_dict = {
                    k: v
                    for k, v in msg_dict.items()
                    if k in ("role", "content", "tool_calls")
                }
            else:
                msg_dict = {
                    "role": response.role,
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in (response.tool_calls or [])
                    ]
                    if hasattr(response, "tool_calls") and response.tool_calls
                    else None,
                }
            messages.append(msg_dict)

            # Log tool calls if any
            if msg_dict.get("tool_calls"):
                for tc in msg_dict["tool_calls"]:
                    ui.print_tool_invocation(
                        tc["function"]["name"], tc["function"]["arguments"]
                    )

            # Manual tool parsing for vLLM/DeepSeek consistency
            tool_calls = []

            if msg_dict.get("tool_calls"):
                for tc in msg_dict["tool_calls"]:
                    func = tc.get("function", {})
                    tool_calls.append(
                        {
                            "id": tc.get("id", "unknown"),
                            "name": func.get("name"),
                            "arguments": json.loads(func.get("arguments", "{}")),
                        }
                    )
            # Check for XML tool calls in content
            elif msg_dict.get("content"):
                matches = re.findall(
                    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
                    msg_dict["content"],
                    re.DOTALL,
                )
                for m in matches:
                    try:
                        parsed = json.loads(m)
                        tool_calls.append(
                            {
                                "id": "unknown",
                                "name": parsed["name"],
                                "arguments": parsed["arguments"]
                                if isinstance(parsed["arguments"], dict)
                                else json.loads(parsed["arguments"]),
                            }
                        )
                    except (json.JSONDecodeError, KeyError):
                        pass

            if not tool_calls:
                # If no tools called, we might be stuck or the LLM is just talking.
                # In a production agent, we'd nudge it to use tools.
                continue

            for tc in tool_calls:
                name = tc["name"]
                args = tc["arguments"]

                if name == "run_deterministic_checks":
                    results = self._run_deterministic_checks(
                        args.get("draft_text", draft), min_words, max_words
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": name,
                            "content": json.dumps(results),
                        }
                    )
                    ui.print_tool_result(
                        "run_deterministic_checks",
                        f"Status: {results.get('status')} ({len(results.get('failures', []))} failures)",
                    )
                elif name == "verify_against_transcript":
                    query_text = args.get("query", "")
                    retrieve_task = json.dumps(
                        {
                            "action": "retrieve",
                            "query": query_text,
                            "top_k": 10,
                        }
                    )
                    raw_resp = await a2a_send_task(self.retrieval_port, retrieve_task)
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "name": name,
                            "content": raw_resp,
                        }
                    )
                    ui.print_tool_result(
                        "verify_against_transcript",
                        "Retrieved segments for verification.",
                    )
                elif name == "submit_verdict":
                    final_verdict = {
                        "status": args["status"],
                        "word_count": args["actual_word_count"],
                        "feedback": args["feedback"],
                    }
                    break

            if final_verdict:
                break

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
            ui.print_status(f"Final Verdict: {result_json}")
        else:
            ui.print_agent_message(self.name, f"Final Verdict: {result_json}")

        await self.send_response(result_json, event_queue)
