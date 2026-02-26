import json
import re

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from agents.client import agent_client
from agents.dtos import RetrieveTaskRequest
from agents.message_registry import MessageRegistry
from agents.tool_handlers import build_revise_tools, build_summarizer_tools
from events import event_bus
from llm_provider import LLMProvider
from prompt_manager import PromptManager


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
        is_embedding_enabled: bool = True,
    ) -> None:
        super().__init__("Summarizer")
        self.llm = llm
        self.retrieval_port = retrieval_port
        self.max_tool_turns = max_tool_turns
        self.is_embedding_enabled = is_embedding_enabled

    async def handle_task(
        self, task_data: dict, context: RequestContext, event_queue: EventQueue
    ) -> None:
        from agents.dtos import SummarizerTaskRequest

        req = SummarizerTaskRequest.model_validate(task_data)
        min_words = req.min_words
        max_words = req.max_words
        feedback = req.feedback
        retrieval_port = req.retrieval_port
        previous_draft = req.previous_draft
        full_transcript = req.full_transcript

        revise_mode = bool(previous_draft and feedback)

        # ── Build per-task registry ──
        registry = MessageRegistry()

        if revise_mode:
            # Pre-load the registry from the previous draft XML
            self._load_draft_into_registry(registry, previous_draft)
            tool_registry = build_revise_tools(
                registry,
                retrieval_port,
                min_words,
                max_words,
                self.is_embedding_enabled,
            )
            event_bus.publish(
                "system_message",
                f"🔄 REVISE MODE — loaded {len(registry)} messages from previous draft",
            )
        else:
            tool_registry = build_summarizer_tools(
                registry,
                retrieval_port,
                min_words,
                max_words,
                self.is_embedding_enabled,
            )

        transcript_excerpt = ""
        total_words = 0
        num_segments = 0

        if self.is_embedding_enabled:
            # ── Dynamically generate query for the Info Retrieval Agent ──
            query_sys_prompt = PromptManager.get_prompt("query_generation_system")
            query_user_prompt = (
                f"Goal: Find main topics, key arguments, and important lessons for a {min_words}-{max_words} word summary.\n"
                f"Feedback from previous attempt: {feedback if feedback else 'None'}"
            )
            event_bus.publish(
                "system_message", "Generating dynamic retrieval query via LLM..."
            )
            retrieve_query = self.llm.generate(
                system_prompt=query_sys_prompt,
                user_prompt=query_user_prompt,
                max_tokens=64,
            ).strip()
            event_bus.publish("system_message", f"Generated query: {retrieve_query}")

            retrieve_task = RetrieveTaskRequest(
                action="retrieve", query=retrieve_query, top_k=20
            )

            event_bus.publish(
                "system_message",
                message=f"Querying Info Retrieval Agent (port {retrieval_port})...",
            )

            retrieval_response_raw = await agent_client.send_task(
                retrieval_port, retrieve_task, timeout=60.0
            )

            retrieval_data = json.loads(retrieval_response_raw)
            segments = retrieval_data.get("segments", [])
            total_words = retrieval_data.get("total_words", 0)
            num_segments = len(segments)
            event_bus.publish(
                "system_message",
                f"Retrieved {num_segments} segments ({total_words} words)",
            )
            event_bus.publish(
                "retrieval_stats",
                source="summarizer_initial",
                num_segments=num_segments,
                total_words=total_words,
            )

            # ── Format retrieved segments ──
            for seg in segments:
                speaker = seg.get("speaker", "UNKNOWN")
                text = seg.get("text", "")
                transcript_excerpt += f"[{speaker}]: {text}\n"
        else:
            transcript_excerpt = full_transcript
            event_bus.publish(
                "system_message",
                "Using full transcript directly (embeddings disabled).",
            )
            # We don't have segment counts/word counts directly here easily without parsing again,
            # but we can just say "all" or pass some defaults if needed.
            total_words = len(full_transcript.split())
            num_segments = full_transcript.count("\n")
            event_bus.publish(
                "retrieval_stats",
                source="full_transcript_fallback",
                num_segments=num_segments,
                total_words=total_words,
            )

        # ── Build system prompt ──
        if revise_mode:
            system_prompt = PromptManager.get_prompt(
                "summarizer_revise",
                min_words=min_words,
                max_words=max_words,
                previous_draft=previous_draft,
                draft_line_map=json.dumps(registry.snapshot(), indent=2),
                feedback=feedback,
                is_embedding_enabled=self.is_embedding_enabled,
            )
        else:
            system_prompt = PromptManager.get_prompt(
                "summarizer_system",
                min_words=min_words,
                max_words=max_words,
                is_embedding_enabled=self.is_embedding_enabled,
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": PromptManager.get_prompt(
                    "summarizer_user",
                    num_segments=num_segments,
                    total_words=total_words,
                    transcript_excerpt=transcript_excerpt,
                    feedback_block=f"\n\n--- CRITIC FEEDBACK ---\n{feedback}"
                    if feedback
                    else "",
                ),
            },
        ]

        mode_label = "revise" if revise_mode else "incremental"
        event_bus.publish(
            "system_message", f"Generating summary using {mode_label} tool loop..."
        )

        context_data = {
            "tool_registry": tool_registry,
            "min_words": min_words,
            "max_words": max_words,
            "max_tool_calls_per_turn": 1,
            "signature_dedupe_window_turns": 1,
            "replay_dedupe_tools": {
                "add_speaker_message",
                "edit_message",
                "remove_message",
                "finalize_draft",
            },
        }
        await self.run_agent_loop(
            messages,
            tool_registry.get_tool_schemas(),
            self.max_tool_turns,
            context_data,
        )

        # ── Assemble final XML output from registry ──
        result_xml = ""
        for speaker_id, content in registry.get_all():
            result_xml += f"<{speaker_id}>{content}</{speaker_id}>\n"
        result_xml = result_xml.strip()

        event_bus.publish(
            "agent_message",
            self.name,
            f"Tool-generated XML output:\n```xml\n{result_xml}\n```",
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

    async def process_tool_calls(
        self, tool_calls: list[dict], messages: list, context_data: dict
    ) -> tuple[bool, dict | None]:
        tool_registry = context_data["tool_registry"]
        min_words = context_data["min_words"]
        max_words = context_data["max_words"]
        finalized = False
        mutating_tools = {"add_speaker_message", "edit_message", "remove_message"}
        mutating_executed_this_turn = False

        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]

            if name in mutating_tools and mutating_executed_this_turn:
                skip_result = {
                    "status": "skipped",
                    "reason": "single_mutating_call_per_turn",
                    "message": (
                        "Only one mutating tool call is executed per assistant turn. "
                        "Issue the next mutation in a new turn after reading count_words."
                    ),
                }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.get("id", "unknown"),
                        "name": name,
                        "content": json.dumps(skip_result),
                    }
                )
                event_bus.publish(
                    "tool_result",
                    tool_name=f"{name} (skipped)",
                    result=json.dumps(skip_result, indent=2),
                )
                continue

            if name in mutating_tools:
                mutating_executed_this_turn = True

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
            event_bus.publish(
                "tool_result", tool_name=name, result=json.dumps(result, indent=2)
            )

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
                event_bus.publish(
                    "tool_result",
                    tool_name="count_words (auto)",
                    result=f"Total: {total_wc} words",
                )

                # ── Notify LLM when budget is met (but don't break) ──
                if min_words <= total_wc <= max_words:
                    event_bus.publish(
                        "status_message",
                        message=f"📋 Budget in range: {total_wc} words "
                        f"(target {min_words}-{max_words}) — "
                        f"waiting for LLM to call finalize_draft",
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
                    event_bus.publish(
                        "tool_result",
                        tool_name="save_draft (auto)",
                        result=f"Saved {save_result.get('total_messages', 0)} messages",
                    )

            # ── Break only when LLM explicitly finalizes ──
            if name == "finalize_draft":
                finalized = True
                draft_xml = str(result.get("draft_xml", "")).strip()
                if draft_xml:
                    event_bus.publish(
                        "agent_message",
                        self.name,
                        f"Finalized Draft:\n```xml\n{draft_xml}\n```",
                    )
                event_bus.publish(
                    "status_message",
                    message="✅ LLM called finalize_draft — submitting to Critic",
                )

        if finalized:
            return True, None
        return False, None

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
