import json
import re

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from agents.client import a2a_send_task
from agents.message_registry import MessageRegistry
from agents.tool_handlers import build_revise_tools, build_summarizer_tools
from llm_provider import LLMProvider
from prompt_manager import PromptManager
from ui import ui


class SummarizerExecutor(BaseA2AExecutor):
    """
    A2A executor that produces an abstractive summary of a transcript.
    Uses incremental tool-loop: the LLM adds one speaker message at a time,
    gets automatic word-count feedback, and can edit/remove messages to
    stay within budget.

    On retry iterations (revise mode), the previous draft is pre-loaded
    into the registry and the LLM can only edit/remove — not add new messages.
    """

    def __init__(
        self,
        llm: LLMProvider,
        retrieval_port: int,
        max_tool_turns: int,
    ) -> None:
        super().__init__("Summarizer")
        self.llm = llm
        self.retrieval_port = retrieval_port
        self.max_tool_turns = max_tool_turns

    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        min_words = task_data.get("min_words", 50)
        max_words = task_data.get("max_words", 100)
        feedback = task_data.get("feedback", "")
        retrieval_port = task_data.get("retrieval_port", self.retrieval_port)
        previous_draft = task_data.get("previous_draft", "")

        revise_mode = bool(previous_draft and feedback)

        # ── Build per-task registry ──
        registry = MessageRegistry()

        if revise_mode:
            # Pre-load the registry from the previous draft XML
            self._load_draft_into_registry(registry, previous_draft)
            tool_registry = build_revise_tools(
                registry, retrieval_port, min_words, max_words
            )
            ui.print_system(
                f"🔄 REVISE MODE — loaded {len(registry)} messages from previous draft"
            )
        else:
            tool_registry = build_summarizer_tools(
                registry, retrieval_port, min_words, max_words
            )

        # ── Dynamically generate query for the Info Retrieval Agent ──
        query_sys_prompt = PromptManager.get_prompt("query_generation_system")
        query_user_prompt = (
            f"Goal: Find main topics, key arguments, and important lessons for a {min_words}-{max_words} word summary.\n"
            f"Feedback from previous attempt: {feedback if feedback else 'None'}"
        )
        ui.print_system("Generating dynamic retrieval query via LLM...")
        retrieve_query = self.llm.generate(
            system_prompt=query_sys_prompt, user_prompt=query_user_prompt, max_tokens=64
        ).strip()
        ui.print_system(f"Generated query: {retrieve_query}")

        retrieve_task = json.dumps(
            {
                "action": "retrieve",
                "query": retrieve_query,
                "top_k": 20,
            }
        )

        ui.print_system(f"Querying Info Retrieval Agent (port {retrieval_port})...")

        retrieval_response_raw = await a2a_send_task(
            retrieval_port, retrieve_task, timeout=60.0
        )

        retrieval_data = json.loads(retrieval_response_raw)
        segments = retrieval_data.get("segments", [])
        total_words = retrieval_data.get("total_words", 0)
        ui.print_system(f"Retrieved {len(segments)} segments ({total_words} words)")

        # ── Format retrieved segments ──
        transcript_excerpt = ""
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            text = seg.get("text", "")
            transcript_excerpt += f"[{speaker}]: {text}\n"

        # ── Build system prompt ──
        if revise_mode:
            system_prompt = PromptManager.get_prompt(
                "summarizer_revise",
                min_words=min_words,
                max_words=max_words,
                previous_draft=previous_draft,
                draft_line_map=json.dumps(registry.snapshot(), indent=2),
                feedback=feedback,
            )
        else:
            system_prompt = PromptManager.get_prompt(
                "summarizer_system", min_words=min_words, max_words=max_words
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": PromptManager.get_prompt(
                    "summarizer_user",
                    num_segments=len(segments),
                    total_words=total_words,
                    transcript_excerpt=transcript_excerpt,
                    feedback_block=f"\n\n--- CRITIC FEEDBACK ---\n{feedback}"
                    if feedback
                    else "",
                ),
            },
        ]

        mode_label = "revise" if revise_mode else "incremental"
        ui.print_system(f"Generating summary using {mode_label} tool loop...")

        finalized = False

        for _ in range(self.max_tool_turns):
            response_message = self.llm.chat(
                messages=messages,
                tools=tool_registry.get_tool_schemas(),
            )

            # ── Normalise to dict ──
            msg_dict = self._to_dict(response_message)
            messages.append(msg_dict)

            # ── Collect tool calls (structured or XML-fallback) ──
            tool_calls = self._extract_tool_calls(msg_dict)

            if not tool_calls:
                continue

            # Log invocations
            for tc in tool_calls:
                ui.print_tool_invocation(tc["name"], json.dumps(tc["arguments"]))

            # ── Dispatch each tool call via the registry ──
            for tc in tool_calls:
                name = tc["name"]
                args = tc["arguments"]
                result = await tool_registry.dispatch(name, args)

                if result is None:
                    result = {"error": f"Unknown tool: {name}"}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", "unknown"),
                        "name": name,
                        "content": json.dumps(result),
                    }
                )
                ui.print_tool_result(name, json.dumps(result, indent=2))

                # ── Auto count_words after any registry-modifying tool ──
                _registry_tools = {
                    "add_speaker_message",
                    "edit_message",
                    "remove_message",
                }
                if name in _registry_tools and "error" not in result:
                    count_result = await tool_registry.dispatch("count_words", {})
                    auto_id = f"auto_count_{tc.get('id', 'unknown')}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": auto_id,
                            "name": "count_words",
                            "content": json.dumps(count_result),
                        }
                    )
                    total_wc = count_result["total_word_count"]
                    ui.print_tool_result(
                        "count_words (auto)", f"Total: {total_wc} words"
                    )

                    # ── Notify LLM when budget is met (but don't break) ──
                    if min_words <= total_wc <= max_words:
                        ui.print_status(
                            f"📋 Budget in range: {total_wc} words "
                            f"(target {min_words}-{max_words}) — "
                            f"waiting for LLM to call finalize_draft"
                        )

                        # Auto save_draft so the LLM can review
                        save_result = await tool_registry.dispatch("save_draft", {})
                        save_id = f"auto_save_{tc.get('id', 'unknown')}"
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": save_id,
                                "name": "save_draft",
                                "content": json.dumps(save_result),
                            }
                        )
                        ui.print_tool_result(
                            "save_draft (auto)",
                            f"Saved {save_result.get('total_messages', 0)} messages",
                        )

                # ── Break only when LLM explicitly finalizes ──
                if name == "finalize_draft":
                    finalized = True
                    ui.print_status(
                        "✅ LLM called finalize_draft — submitting to Critic"
                    )

            if finalized:
                break

        # ── Assemble final XML output from registry ──
        result_xml = ""
        for speaker_id, content in registry.get_all():
            result_xml += f"<{speaker_id}>{content}</{speaker_id}>\n"
        result_xml = result_xml.strip()

        ui.print_agent_message(
            self.name, f"Tool-generated XML output:\n```xml\n{result_xml}\n```"
        )
        await self.send_response(result_xml, event_queue)

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _load_draft_into_registry(registry: MessageRegistry, draft_xml: str) -> None:
        """Parse <SPEAKER_XX>content</SPEAKER_XX> XML back into the registry."""
        pattern = re.compile(r"<(SPEAKER_\d+)>(.*?)</\1>", re.DOTALL)
        for match in pattern.finditer(draft_xml):
            speaker_id = match.group(1)
            content = match.group(2).strip()
            registry.add(speaker_id, content)

    @staticmethod
    def _to_dict(response_message) -> dict:
        """Normalise an LLM response object to a plain dict."""
        if hasattr(response_message, "model_dump"):
            d = response_message.model_dump()
            return {
                k: v for k, v in d.items() if k in ("role", "content", "tool_calls")
            }
        return {
            "role": response_message.role,
            "content": response_message.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in (response_message.tool_calls or [])
            ]
            if hasattr(response_message, "tool_calls") and response_message.tool_calls
            else None,
        }

    @staticmethod
    def _extract_tool_calls(msg_dict: dict) -> list[dict]:
        """Extract tool calls from structured field or XML fallback in content."""
        calls: list[dict] = []

        # 1. Structured tool calls
        if msg_dict.get("tool_calls"):
            for tc in msg_dict["tool_calls"]:
                func = tc.get("function", {})
                args = func.get("arguments", "{}")
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append(
                    {
                        "id": tc.get("id", "unknown"),
                        "name": func.get("name"),
                        "arguments": args,
                    }
                )
            return calls

        # 2. XML <tool_call> fallback (vLLM / DeepSeek)
        content = msg_dict.get("content") or ""
        xml_matches = re.findall(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, re.DOTALL
        )
        for raw in xml_matches:
            try:
                parsed = json.loads(raw)
                args = parsed.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append(
                    {
                        "id": "xml_fallback",
                        "name": parsed["name"],
                        "arguments": args,
                    }
                )
            except (json.JSONDecodeError, KeyError):
                pass

        # 3. Plain-text function call fallback
        if not calls:
            fn_matches = re.findall(
                r"add_speaker_message\s*\(\s*(SPEAKER_\d+)\s*,\s*\"(.*?)\"\s*\)",
                content,
                re.DOTALL,
            )
            for speaker_id, msg_content in fn_matches:
                calls.append(
                    {
                        "id": "text_fallback",
                        "name": "add_speaker_message",
                        "arguments": {
                            "speaker_id": speaker_id,
                            "content": msg_content.replace('\\"', '"'),
                        },
                    }
                )

        return calls
