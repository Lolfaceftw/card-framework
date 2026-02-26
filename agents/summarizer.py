import hashlib
import json
import re
from typing import Any

from a2a.server.agent_execution import RequestContext
from a2a.server.events import EventQueue

from agents.base import BaseA2AExecutor
from agents.client import agent_client
from agents.dtos import RetrieveTaskRequest
from agents.message_registry import MessageRegistry
from agents.tool_call_utils import build_tool_signature
from agents.tool_handlers import build_revise_tools, build_summarizer_tools
from events import event_bus
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
    ) -> None:
        super().__init__("Summarizer")
        self.llm = llm
        self.retrieval_port = retrieval_port
        self.max_tool_turns = max_tool_turns
        self.is_embedding_enabled = is_embedding_enabled
        self.loop_guardrails = dict(loop_guardrails or {})

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
        if isinstance(raw_value, bool):
            return default
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            return default
        return parsed if parsed >= minimum else default

    @staticmethod
    def _normalise_matchers(raw_values: Any) -> list[str]:
        """Normalize string matcher lists to lowercase non-empty entries."""
        if not isinstance(raw_values, list):
            return []
        return [str(item).strip().lower() for item in raw_values if str(item).strip()]

    @staticmethod
    def _normalize_query_text(raw_query: Any) -> str:
        """Normalize retrieval queries for dedupe and stage-budget accounting."""
        if raw_query is None:
            return ""
        return " ".join(str(raw_query).strip().lower().split())

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

    @staticmethod
    def _build_mutation_fingerprint(name: str, arguments: dict[str, Any]) -> str:
        """Build a lightweight fingerprint used for oscillation detection."""
        if name == "edit_message":
            line = arguments.get("line", "unknown")
            content = str(arguments.get("new_content", ""))
            digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
            return f"edit:{line}:{digest}"
        if name == "remove_message":
            line = arguments.get("line", "unknown")
            return f"remove:{line}"
        if name == "add_speaker_message":
            speaker_id = str(arguments.get("speaker_id", "UNKNOWN"))
            content = str(arguments.get("content", ""))
            digest = hashlib.sha1(content.encode("utf-8")).hexdigest()[:12]
            return f"add:{speaker_id}:{digest}"
        return name

    @staticmethod
    def _is_oscillating(fingerprints: list[str]) -> bool:
        """
        Detect short ABAB mutation oscillation patterns.

        Example: edit line 6 to A, edit line 5 to B, edit line 6 to A, edit line 5 to B.
        """
        if len(fingerprints) < 4:
            return False
        a, b, c, d = fingerprints[-4:]
        return a == c and b == d and a != b

    @staticmethod
    def _build_stall_guidance(
        *,
        min_words: int,
        max_words: int,
        total_words: int | None,
        stagnation_turns: int,
    ) -> str:
        """Build a targeted guidance message after repeated no-progress turns."""
        base = (
            "You are repeating edits without meaningful progress "
            f"for {stagnation_turns} turns. "
        )
        if total_words is None:
            return (
                base
                + "Make one materially different edit and wait for count_words before the next tool call."
            )

        if total_words > max_words:
            return (
                base
                + f"Current total is {total_words}, above the {max_words} max. "
                + "Remove a line with remove_message(line) or materially shorten one line with "
                + "edit_message(line, shorter_content). Do not re-issue unchanged edits."
            )
        if total_words < min_words:
            return (
                base
                + f"Current total is {total_words}, below the {min_words} min. "
                + "Add a new line with add_speaker_message(...) or materially expand one line with "
                + "edit_message(line, expanded_content). Do not re-issue unchanged edits."
            )
        return (
            base
            + f"Current total is {total_words}, within {min_words}-{max_words}. "
            + "If no substantive improvements remain, call finalize_draft(). "
            + "Otherwise make one materially different edit only."
        )

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

        loop_guardrails = self._resolve_loop_guardrails()
        if loop_guardrails["active"]:
            event_bus.publish(
                "system_message",
                (
                    "Loop guardrails enabled for Summarizer "
                    f"(provider={loop_guardrails['provider_name']}, "
                    f"model={loop_guardrails['model_id']})."
                ),
            )
        elif loop_guardrails["configured_enabled"]:
            event_bus.publish(
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
            )
            event_bus.publish(
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
            total_words = len(full_transcript.split())
            num_segments = full_transcript.count("\n")
            event_bus.publish(
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

        context_data: dict[str, Any] = {
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
            "enable_extended_text_tool_parser": loop_guardrails[
                "enable_extended_text_tool_parser"
            ],
            "enable_stall_guidance": loop_guardrails["enable_stall_guidance"],
            "enable_noop_edit_detection": loop_guardrails[
                "enable_noop_edit_detection"
            ],
            "stall_guidance_threshold_turns": loop_guardrails[
                "stall_guidance_threshold_turns"
            ],
            "stall_guidance_cooldown_turns": loop_guardrails[
                "stall_guidance_cooldown_turns"
            ],
            "stagnation_turns": 0,
            "last_stall_guidance_turn": -10_000,
            "loop_turn_index": 0,
            "last_mutation_signature": None,
            "last_total_word_count": None,
            "recent_line_edit_fingerprints": [],
            "loop_guardrail_provider": loop_guardrails["provider_name"],
            "loop_guardrail_model": loop_guardrails["model_id"],
            "enable_staged_discovery": self.is_embedding_enabled
            and loop_guardrails["enable_staged_discovery"],
            "required_discovery_queries": loop_guardrails["required_discovery_queries"],
            "max_discovery_queries": loop_guardrails["max_discovery_queries"],
            "require_unique_discovery_queries": loop_guardrails[
                "require_unique_discovery_queries"
            ],
            "discovery_queries_completed": 0,
            "discovery_query_history": [],
        }
        if context_data["enable_staged_discovery"]:
            max_discovery_queries = int(context_data["max_discovery_queries"])
            max_info = (
                f", max {max_discovery_queries} total query_transcript calls"
                if max_discovery_queries > 0
                else ""
            )
            event_bus.publish(
                "system_message",
                (
                    "Staged discovery enabled: require "
                    f"{context_data['required_discovery_queries']} successful "
                    f"query_transcript calls before mutation/finalize{max_info}."
                ),
            )
        await self.run_agent_loop(
            messages,
            tool_registry.get_tool_schemas(),
            self.max_tool_turns,
            context_data,
        )

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
        min_words = int(context_data["min_words"])
        max_words = int(context_data["max_words"])
        finalized = False
        mutating_tools = {"add_speaker_message", "edit_message", "remove_message"}
        mutating_executed_this_turn = False

        context_data["loop_turn_index"] = int(context_data.get("loop_turn_index", 0)) + 1
        loop_turn_index = int(context_data["loop_turn_index"])

        enable_stall_guidance = bool(context_data.get("enable_stall_guidance", False))
        enable_noop_edit_detection = bool(
            context_data.get("enable_noop_edit_detection", False)
        )
        stall_threshold = self._coerce_positive_int(
            context_data.get("stall_guidance_threshold_turns"), default=3, minimum=1
        )
        guidance_cooldown_turns = self._coerce_positive_int(
            context_data.get("stall_guidance_cooldown_turns"), default=2, minimum=0
        )

        recent_fingerprints_raw = context_data.get("recent_line_edit_fingerprints", [])
        recent_fingerprints = (
            list(recent_fingerprints_raw)
            if isinstance(recent_fingerprints_raw, list)
            else []
        )

        turn_mutation_attempted = False
        turn_progressed = False
        stagnation_reason: str | None = None
        turn_total_wc: int | None = None
        turn_signature: str | None = None
        staged_discovery_enabled = bool(
            context_data.get("enable_staged_discovery", False)
        )
        required_discovery_queries = self._coerce_positive_int(
            context_data.get("required_discovery_queries"),
            default=0,
            minimum=0,
        )
        max_discovery_queries = self._coerce_positive_int(
            context_data.get("max_discovery_queries"),
            default=0,
            minimum=0,
        )
        require_unique_discovery_queries = bool(
            context_data.get("require_unique_discovery_queries", False)
        )
        discovery_queries_completed = self._coerce_positive_int(
            context_data.get("discovery_queries_completed"),
            default=0,
            minimum=0,
        )
        raw_discovery_history = context_data.get("discovery_query_history", [])
        discovery_query_history = (
            list(raw_discovery_history) if isinstance(raw_discovery_history, list) else []
        )
        discovery_query_set = {
            entry for entry in discovery_query_history if isinstance(entry, str) and entry
        }
        staged_blocked_tools = mutating_tools | {"finalize_draft"}

        for tc in tool_calls:
            name = tc["name"]
            args = tc["arguments"]

            if (
                staged_discovery_enabled
                and name in staged_blocked_tools
                and discovery_queries_completed < required_discovery_queries
            ):
                remaining_queries = required_discovery_queries - discovery_queries_completed
                skip_result = {
                    "status": "skipped",
                    "reason": "discovery_phase_incomplete",
                    "required_discovery_queries": required_discovery_queries,
                    "completed_discovery_queries": discovery_queries_completed,
                    "remaining_discovery_queries": remaining_queries,
                    "message": (
                        "Complete staged discovery first. Call query_transcript with "
                        f"{remaining_queries} more focused query"
                        f"{'' if remaining_queries == 1 else 'ies'} before mutating "
                        "the draft or finalizing."
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

            normalized_query = ""
            if staged_discovery_enabled and name == "query_transcript":
                normalized_query = self._normalize_query_text(args.get("query", ""))
                if not normalized_query:
                    skip_result = {
                        "status": "skipped",
                        "reason": "empty_query",
                        "message": (
                            "query_transcript requires a non-empty query string during "
                            "staged discovery."
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

                if (
                    max_discovery_queries > 0
                    and discovery_queries_completed >= max_discovery_queries
                ):
                    skip_result = {
                        "status": "skipped",
                        "reason": "discovery_query_budget_exhausted",
                        "max_discovery_queries": max_discovery_queries,
                        "completed_discovery_queries": discovery_queries_completed,
                        "message": (
                            "Discovery query budget has been exhausted. Continue with "
                            "drafting or refinement."
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

                if (
                    require_unique_discovery_queries
                    and normalized_query in discovery_query_set
                ):
                    skip_result = {
                        "status": "skipped",
                        "reason": "duplicate_discovery_query",
                        "message": (
                            "Use a different query_transcript string that covers a "
                            "new facet of the transcript."
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
                turn_mutation_attempted = True
                turn_signature = build_tool_signature(name, args)

            result = await tool_registry.dispatch(name, args)

            if result is None:
                result = {"error": f"Unknown tool: {name}", "error_code": "unknown_tool"}

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
            if staged_discovery_enabled and name == "query_transcript" and "error" not in result:
                if normalized_query:
                    if normalized_query not in discovery_query_set:
                        discovery_query_history.append(normalized_query)
                        discovery_query_set.add(normalized_query)
                    discovery_queries_completed += 1
                    context_data["discovery_queries_completed"] = (
                        discovery_queries_completed
                    )
                    context_data["discovery_query_history"] = discovery_query_history
                    if required_discovery_queries > 0:
                        remaining_queries = max(
                            required_discovery_queries - discovery_queries_completed, 0
                        )
                        if remaining_queries == 0:
                            event_bus.publish(
                                "status_message",
                                message=(
                                    "Staged discovery complete. "
                                    "Draft mutations are now allowed."
                                ),
                            )
                        else:
                            event_bus.publish(
                                "status_message",
                                message=(
                                    "Staged discovery progress: "
                                    f"{discovery_queries_completed}/"
                                    f"{required_discovery_queries} required "
                                    "query_transcript calls completed."
                                ),
                            )

            registry_mutation_tools = {
                "add_speaker_message",
                "edit_message",
                "remove_message",
            }
            if name in registry_mutation_tools and "error" not in result:
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
                total_wc = int(count_result["total_word_count"])
                turn_total_wc = total_wc
                context_data["last_total_word_count"] = total_wc
                event_bus.publish(
                    "tool_result",
                    tool_name="count_words (auto)",
                    result=f"Total: {total_wc} words",
                )

                if min_words <= total_wc <= max_words:
                    event_bus.publish(
                        "status_message",
                        message=(
                            f"Budget in range: {total_wc} words "
                            f"(target {min_words}-{max_words}) - "
                            "waiting for LLM to call finalize_draft"
                        ),
                    )

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

            if name in registry_mutation_tools:
                if "error" in result:
                    stagnation_reason = str(
                        result.get("error_code") or result.get("error") or "tool_error"
                    )
                else:
                    turn_progressed = True

                if (
                    name == "edit_message"
                    and enable_noop_edit_detection
                    and result.get("changed") is False
                ):
                    turn_progressed = False
                    stagnation_reason = "noop_edit"

                if turn_signature:
                    last_signature = context_data.get("last_mutation_signature")
                    if isinstance(last_signature, str) and last_signature == turn_signature:
                        if not turn_progressed:
                            stagnation_reason = stagnation_reason or "repeated_signature"

                fingerprint = self._build_mutation_fingerprint(name, args)
                recent_fingerprints.append(fingerprint)
                if len(recent_fingerprints) > 6:
                    recent_fingerprints = recent_fingerprints[-6:]
                if self._is_oscillating(recent_fingerprints):
                    turn_progressed = False
                    stagnation_reason = "oscillation"

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
                    message="LLM called finalize_draft - submitting to Critic",
                )

        context_data["recent_line_edit_fingerprints"] = recent_fingerprints
        context_data["discovery_queries_completed"] = discovery_queries_completed
        context_data["discovery_query_history"] = discovery_query_history
        if turn_signature:
            context_data["last_mutation_signature"] = turn_signature

        if turn_mutation_attempted:
            stagnation_turns = int(context_data.get("stagnation_turns", 0))
            if turn_progressed:
                stagnation_turns = 0
            else:
                stagnation_turns += 1
            context_data["stagnation_turns"] = stagnation_turns

            if enable_stall_guidance and stagnation_turns >= stall_threshold:
                last_guidance_turn = int(context_data.get("last_stall_guidance_turn", -10_000))
                if (loop_turn_index - last_guidance_turn) >= (guidance_cooldown_turns + 1):
                    if turn_total_wc is None:
                        last_total = context_data.get("last_total_word_count")
                        turn_total_wc = (
                            int(last_total)
                            if isinstance(last_total, int)
                            else None
                        )
                    guidance = self._build_stall_guidance(
                        min_words=min_words,
                        max_words=max_words,
                        total_words=turn_total_wc,
                        stagnation_turns=stagnation_turns,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": f"[STALL_GUIDANCE]\n{guidance}",
                        }
                    )
                    context_data["last_stall_guidance_turn"] = loop_turn_index
                    event_bus.publish(
                        "status_message",
                        message=(
                            "Stall guardrail triggered: repetitive/no-op edits detected. "
                            "Injected corrective guidance."
                        ),
                        model_id=context_data.get("loop_guardrail_model"),
                        provider=context_data.get("loop_guardrail_provider"),
                        stagnation_turns=stagnation_turns,
                        stall_reason=stagnation_reason or "unknown",
                    )

        if finalized:
            return True, None
        return False, None
