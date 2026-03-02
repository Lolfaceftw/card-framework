import json
import re
from typing import Any

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from agents.client import AgentTaskClient, get_default_agent_client
from agents.dtos import RetrieveTaskRequest
from agents.loop_context import build_loop_context_prompt_block
from agents.message_registry import MessageRegistry
from agents.summarizer_loop_controller import SummarizerLoopController
from agents.summarizer_tool_dispatcher import (
    SummarizerToolDispatcher,
    coerce_positive_int,
)
from agents.tool_handlers import build_revise_tools, build_summarizer_tools
from events import EventBus, get_event_bus
from llm_provider import LLMProvider
from prompt_manager import PromptManager


class SummarizerExecutor(BaseA2AExecutor):
    """
    A2A executor that produces an abstractive summary of a transcript.

    It uses an incremental tool loop where the LLM mutates a line registry
    via add/edit/remove tools and explicitly calls finalize when done.
    """

    _DEFAULT_LOOP_GUARDRAILS: dict[str, Any] = {
        "enabled": False,
        "enable_stall_guidance": False,
        "enable_noop_edit_detection": False,
        "enable_extended_text_tool_parser": False,
        "enable_staged_discovery": False,
        "required_discovery_queries": 0,
        "max_discovery_queries": 0,
        "require_unique_discovery_queries": False,
        "stall_guidance_threshold_turns": 3,
        "stall_guidance_cooldown_turns": 2,
        "target_models": [],
        "target_providers": [],
    }
    def __init__(
        self,
        llm: LLMProvider,
        retrieval_port: int,
        max_tool_turns: int,
        is_embedding_enabled: bool = True,
        loop_guardrails: dict[str, Any] | None = None,
        agent_client: AgentTaskClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize summarizer executor and injected collaborators."""
        super().__init__("Summarizer")
        self.llm = llm
        self.retrieval_port = retrieval_port
        self.max_tool_turns = max_tool_turns
        self.is_embedding_enabled = is_embedding_enabled
        self.loop_guardrails = dict(loop_guardrails or {})
        self.agent_client = (
            agent_client if agent_client is not None else get_default_agent_client()
        )
        self.event_bus = event_bus if event_bus is not None else get_event_bus()
        self._tool_dispatcher = SummarizerToolDispatcher(
            agent_name=self.name,
            event_bus=self.event_bus,
        )
        self._loop_controller = SummarizerLoopController(
            run_agent_loop=self.run_agent_loop,
            max_tool_turns=max_tool_turns,
            is_embedding_enabled=is_embedding_enabled,
            event_bus=self.event_bus,
        )

    @staticmethod
    def _unwrap_provider(provider: LLMProvider) -> LLMProvider:
        """Unwrap logging/decorator wrappers and return the underlying provider."""
        current = provider
        seen: set[int] = set()
        while hasattr(current, "inner_provider") and id(current) not in seen:
            seen.add(id(current))
            inner = getattr(current, "inner_provider")
            if isinstance(inner, LLMProvider):
                current = inner
                continue
            break
        return current

    @staticmethod
    def _coerce_positive_int(raw_value: Any, *, default: int, minimum: int = 0) -> int:
        """Parse an integer with clamping fallback for runtime config values."""
        return coerce_positive_int(raw_value, default=default, minimum=minimum)

    @staticmethod
    def _normalise_matchers(raw_values: Any) -> list[str]:
        """Normalize string matcher lists to lowercase non-empty entries."""
        if not isinstance(raw_values, list):
            return []
        return [str(item).strip().lower() for item in raw_values if str(item).strip()]

    @staticmethod
    def _build_loop_context_prompt_block(raw_loop_context: Any) -> str:
        """Build a bounded loop-context payload for prompt injection."""
        return build_loop_context_prompt_block(raw_loop_context, char_cap=1024)

    def _resolve_provider_metadata(self) -> tuple[str, str]:
        """Return provider class name and best-effort model identifier."""
        provider = self._unwrap_provider(self.llm)
        provider_name = type(provider).__name__

        model_id = ""
        for attr in ("model_id", "model_name", "model"):
            value = getattr(provider, attr, None)
            if isinstance(value, str) and value.strip():
                model_id = value.strip()
                break

        if not model_id:
            model_id = provider_name
        return provider_name, model_id

    def _resolve_loop_guardrails(self) -> dict[str, Any]:
        """
        Resolve model-gated loop guardrail policy for this summarizer execution.

        Guardrails are considered active only when:
        - `enabled` is true
        - target provider/model matchers (if configured) match this provider
        """
        resolved = dict(self._DEFAULT_LOOP_GUARDRAILS)
        resolved.update(self.loop_guardrails)

        configured_enabled = bool(resolved.get("enabled", False))
        provider_name, model_id = self._resolve_provider_metadata()
        provider_name_lc = provider_name.lower()
        model_id_lc = model_id.lower()

        provider_matchers = self._normalise_matchers(resolved.get("target_providers"))
        model_matchers = self._normalise_matchers(resolved.get("target_models"))

        provider_match = (
            True
            if not provider_matchers
            else any(matcher in provider_name_lc for matcher in provider_matchers)
        )
        model_match = (
            True
            if not model_matchers
            else any(matcher in model_id_lc for matcher in model_matchers)
        )
        is_active = configured_enabled and provider_match and model_match
        required_discovery_queries = self._coerce_positive_int(
            resolved.get("required_discovery_queries"),
            default=0,
            minimum=0,
        )
        max_discovery_queries = self._coerce_positive_int(
            resolved.get("max_discovery_queries"),
            default=0,
            minimum=0,
        )
        if (
            max_discovery_queries > 0
            and max_discovery_queries < required_discovery_queries
        ):
            max_discovery_queries = required_discovery_queries

        return {
            "configured_enabled": configured_enabled,
            "active": is_active,
            "provider_name": provider_name,
            "model_id": model_id,
            "enable_stall_guidance": is_active
            and bool(resolved.get("enable_stall_guidance", False)),
            "enable_noop_edit_detection": is_active
            and bool(resolved.get("enable_noop_edit_detection", False)),
            "enable_extended_text_tool_parser": is_active
            and bool(resolved.get("enable_extended_text_tool_parser", False)),
            "enable_staged_discovery": is_active
            and bool(resolved.get("enable_staged_discovery", False)),
            "required_discovery_queries": required_discovery_queries,
            "max_discovery_queries": max_discovery_queries,
            "require_unique_discovery_queries": is_active
            and bool(resolved.get("require_unique_discovery_queries", False)),
            "stall_guidance_threshold_turns": self._coerce_positive_int(
                resolved.get("stall_guidance_threshold_turns"),
                default=3,
                minimum=1,
            ),
            "stall_guidance_cooldown_turns": self._coerce_positive_int(
                resolved.get("stall_guidance_cooldown_turns"),
                default=2,
                minimum=0,
            ),
            "provider_matchers": provider_matchers,
            "model_matchers": model_matchers,
            "provider_match": provider_match,
            "model_match": model_match,
        }

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
        loop_context = getattr(req, "loop_context", task_data.get("loop_context", ""))
        loop_context_block = self._build_loop_context_prompt_block(loop_context)

        revise_mode = bool(previous_draft and feedback)

        loop_guardrails = self._resolve_loop_guardrails()
        if loop_guardrails["active"]:
            self.event_bus.publish(
                "system_message",
                (
                    "Loop guardrails enabled for Summarizer "
                    f"(provider={loop_guardrails['provider_name']}, "
                    f"model={loop_guardrails['model_id']})."
                ),
            )
        elif loop_guardrails["configured_enabled"]:
            self.event_bus.publish(
                "system_message",
                (
                    "Loop guardrails configured but inactive for this model "
                    f"(provider={loop_guardrails['provider_name']}, "
                    f"model={loop_guardrails['model_id']})."
                ),
            )

        registry = MessageRegistry()

        if revise_mode:
            self._load_draft_into_registry(registry, previous_draft)
            tool_registry = build_revise_tools(
                registry,
                retrieval_port,
                min_words,
                max_words,
                self.is_embedding_enabled,
                agent_client=self.agent_client,
            )
            self.event_bus.publish(
                "system_message",
                f"REVISE MODE - loaded {len(registry)} messages from previous draft",
            )
        else:
            tool_registry = build_summarizer_tools(
                registry,
                retrieval_port,
                min_words,
                max_words,
                self.is_embedding_enabled,
                agent_client=self.agent_client,
            )

        transcript_excerpt = ""
        total_words = 0
        num_segments = 0

        if self.is_embedding_enabled:
            query_sys_prompt = PromptManager.get_prompt("query_generation_system")
            query_user_prompt = (
                "Goal: Find main topics, key arguments, and important lessons for a "
                f"{min_words}-{max_words} word summary.\n"
                f"Feedback from previous attempt: {feedback if feedback else 'None'}"
            )
            self.event_bus.publish(
                "system_message", "Generating dynamic retrieval query via LLM..."
            )
            retrieve_query = self.llm.generate(
                system_prompt=query_sys_prompt,
                user_prompt=query_user_prompt,
                max_tokens=64,
            ).strip()
            self.event_bus.publish("system_message", f"Generated query: {retrieve_query}")

            retrieve_task = RetrieveTaskRequest(
                action="retrieve", query=retrieve_query, top_k=20
            )

            self.event_bus.publish(
                "system_message",
                message=f"Querying Info Retrieval Agent (port {retrieval_port})...",
            )

            retrieval_response_raw = await self.agent_client.send_task(
                retrieval_port, retrieve_task, timeout=60.0
            )

            retrieval_data = json.loads(retrieval_response_raw)
            segments = retrieval_data.get("segments", [])
            total_words = retrieval_data.get("total_words", 0)
            num_segments = len(segments)
            self.event_bus.publish(
                "system_message",
                f"Retrieved {num_segments} segments ({total_words} words)",
            )
            self.event_bus.publish(
                "retrieval_stats",
                source="summarizer_initial",
                num_segments=num_segments,
                total_words=total_words,
            )

            for seg in segments:
                speaker = seg.get("speaker", "UNKNOWN")
                text = seg.get("text", "")
                transcript_excerpt += f"[{speaker}]: {text}\n"
        else:
            transcript_excerpt = full_transcript
            self.event_bus.publish(
                "system_message",
                "Using full transcript directly (embeddings disabled).",
            )
            total_words = len(full_transcript.split())
            num_segments = full_transcript.count("\n")
            self.event_bus.publish(
                "retrieval_stats",
                source="full_transcript_fallback",
                num_segments=num_segments,
                total_words=total_words,
            )

        if revise_mode:
            system_prompt = PromptManager.get_prompt(
                "summarizer_revise",
                min_words=min_words,
                max_words=max_words,
                previous_draft=previous_draft,
                draft_line_map=json.dumps(registry.snapshot(), indent=2),
                feedback=feedback,
                loop_context_block=loop_context_block,
                is_embedding_enabled=self.is_embedding_enabled,
                staged_discovery_enabled=self.is_embedding_enabled
                and loop_guardrails["enable_staged_discovery"],
                required_discovery_queries=loop_guardrails["required_discovery_queries"],
                max_discovery_queries=loop_guardrails["max_discovery_queries"],
                require_unique_discovery_queries=loop_guardrails[
                    "require_unique_discovery_queries"
                ],
            )
        else:
            system_prompt = PromptManager.get_prompt(
                "summarizer_system",
                min_words=min_words,
                max_words=max_words,
                is_embedding_enabled=self.is_embedding_enabled,
                staged_discovery_enabled=self.is_embedding_enabled
                and loop_guardrails["enable_staged_discovery"],
                required_discovery_queries=loop_guardrails["required_discovery_queries"],
                max_discovery_queries=loop_guardrails["max_discovery_queries"],
                require_unique_discovery_queries=loop_guardrails[
                    "require_unique_discovery_queries"
                ],
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
                    loop_context_block=loop_context_block,
                    feedback_block=f"\n\n--- CRITIC FEEDBACK ---\n{feedback}"
                    if feedback
                    else "",
                ),
            },
        ]

        mode_label = "revise" if revise_mode else "incremental"
        self.event_bus.publish(
            "system_message", f"Generating summary using {mode_label} tool loop..."
        )

        await self._loop_controller.run(
            messages=messages,
            tool_registry=tool_registry,
            min_words=min_words,
            max_words=max_words,
            loop_guardrails=loop_guardrails,
        )

        result_xml = ""
        for speaker_id, content in registry.get_all():
            result_xml += f"<{speaker_id}>{content}</{speaker_id}>\n"
        result_xml = result_xml.strip()

        self.event_bus.publish(
            "agent_message",
            self.name,
            f"Tool-generated XML output:\n```xml\n{result_xml}\n```",
        )
        await self.send_response(result_xml, event_queue)

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
        return await self._tool_dispatcher.process_tool_calls(
            tool_calls=tool_calls,
            messages=messages,
            context_data=context_data,
        )

