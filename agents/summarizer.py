"""Summarizer executor with duration-aware incremental tool use."""

from __future__ import annotations

import json
from pathlib import Path
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
from audio_pipeline.calibration import VoiceCloneCalibration
from audio_pipeline.live_draft_voice_clone import LiveDraftVoiceCloneSession
from audio_pipeline.voice_clone_orchestrator import VoiceCloneOrchestrator
from events import EventBus, get_event_bus
from llm_provider import LLMProvider
from prompt_manager import PromptManager
from summary_xml import DEFAULT_EMO_PRESET, parse_summary_xml, serialize_summary_turns


class SummarizerExecutor(BaseA2AExecutor):
    """Produce an abstractive, duration-targeted summary of a transcript."""

    _DEFAULT_NEUTRAL_WPM = 150.0

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
        calibration: VoiceCloneCalibration | None,
        is_embedding_enabled: bool = True,
        loop_guardrails: dict[str, Any] | None = None,
        voice_clone_orchestrator: VoiceCloneOrchestrator | None = None,
        live_draft_audio_enabled: bool = False,
        emo_preset_catalog: dict[str, str] | None = None,
        agent_client: AgentTaskClient | None = None,
        event_bus: EventBus | None = None,
    ) -> None:
        """Initialize summarizer executor and injected collaborators."""
        super().__init__("Summarizer")
        self.llm = llm
        self.retrieval_port = retrieval_port
        self.max_tool_turns = max_tool_turns
        self.calibration = calibration
        self.is_embedding_enabled = is_embedding_enabled
        self.loop_guardrails = dict(loop_guardrails or {})
        self.voice_clone_orchestrator = voice_clone_orchestrator
        self.live_draft_audio_enabled = bool(
            live_draft_audio_enabled and voice_clone_orchestrator is not None
        )
        if emo_preset_catalog:
            self.emo_preset_catalog = dict(emo_preset_catalog)
        elif calibration is not None:
            self.emo_preset_catalog = dict(calibration.preset_emo_texts)
        else:
            self.emo_preset_catalog = {
                DEFAULT_EMO_PRESET: "calm, neutral, steady conversational tone."
            }
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
        """Resolve model-gated loop guardrail policy for this summarizer execution."""
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

    def _resolve_target_seconds(self, *, task_data: dict[str, Any], req: Any) -> int:
        """Resolve the requested summary duration in seconds."""
        explicit_target_seconds = getattr(req, "target_seconds", None)
        if explicit_target_seconds is not None and int(explicit_target_seconds) > 0:
            return int(explicit_target_seconds)
        mid_words = max(1, int((int(req.min_words) + int(req.max_words)) / 2))
        if self.calibration is not None:
            neutral_wpm = self.calibration.wpm_for(
                speaker="",
                emo_preset=DEFAULT_EMO_PRESET,
            )
        else:
            neutral_wpm = self._DEFAULT_NEUTRAL_WPM
        derived_seconds = int(round((mid_words / neutral_wpm) * 60.0))
        self.event_bus.publish(
            "system_message",
            (
                "Legacy word-budget task detected; derived duration target of "
                f"{derived_seconds}s from midpoint word count."
            ),
        )
        return max(1, derived_seconds)

    def _build_emo_preset_guide(self) -> str:
        """Render the preset catalog as prompt-ready lines."""
        return "\n".join(
            f"- `{name}`: {emo_text}"
            for name, emo_text in self.emo_preset_catalog.items()
        )

    async def handle_task(
        self, task_data: dict[str, Any], context: RequestContext, event_queue: EventQueue
    ) -> None:
        from agents.dtos import SummarizerTaskRequest

        del context
        req = SummarizerTaskRequest.model_validate(task_data)
        feedback = req.feedback
        retrieval_port = req.retrieval_port
        previous_draft = req.previous_draft
        full_transcript = req.full_transcript
        speaker_samples_manifest_path = str(
            getattr(req, "speaker_samples_manifest_path", "")
        ).strip()
        draft_audio_state_path = str(
            getattr(req, "draft_audio_state_path", "")
        ).strip()
        loop_context = getattr(req, "loop_context", task_data.get("loop_context", ""))
        loop_context_block = self._build_loop_context_prompt_block(loop_context)
        target_seconds = self._resolve_target_seconds(task_data=task_data, req=req)
        duration_tolerance_ratio = float(
            getattr(req, "duration_tolerance_ratio", 0.05) or 0.05
        )
        target_minutes = round(target_seconds / 60.0, 2)
        tolerance_seconds = round(target_seconds * duration_tolerance_ratio, 2)
        min_seconds = round(target_seconds - tolerance_seconds, 2)
        max_seconds = round(target_seconds + tolerance_seconds, 2)
        emo_preset_guide = self._build_emo_preset_guide()

        revise_mode = bool(previous_draft and feedback)
        live_draft_session: LiveDraftVoiceCloneSession | None = None
        if self.live_draft_audio_enabled:
            if not speaker_samples_manifest_path:
                raise RuntimeError(
                    "Live drafting requires speaker_samples_manifest_path."
                )
            if not draft_audio_state_path:
                raise RuntimeError("Live drafting requires draft_audio_state_path.")
            if self.voice_clone_orchestrator is None:
                raise RuntimeError(
                    "Live drafting is enabled but voice_clone_orchestrator is missing."
                )
            live_draft_session = LiveDraftVoiceCloneSession.from_orchestrator(
                orchestrator=self.voice_clone_orchestrator,
                state_path=Path(draft_audio_state_path),
                speaker_samples_manifest_path=Path(speaker_samples_manifest_path),
            )

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
            restored_live_snapshot = None
            if live_draft_session is not None:
                restored_live_snapshot = live_draft_session.restore_snapshot_for_draft(
                    previous_draft
                )
            if restored_live_snapshot is not None:
                registry.load_from_snapshot(restored_live_snapshot)
                self.event_bus.publish(
                    "system_message",
                    (
                        "REVISE MODE - restored live draft audio state for "
                        f"{len(registry)} messages"
                    ),
                )
            else:
                self._load_draft_into_registry(registry, previous_draft)
                if live_draft_session is not None and len(registry) > 0:
                    live_draft_session.bootstrap_from_snapshot(registry.snapshot())
                    self.event_bus.publish(
                        "system_message",
                        (
                            "REVISE MODE - regenerated live draft audio state for "
                            f"{len(registry)} messages"
                        ),
                    )
            tool_registry = build_revise_tools(
                registry,
                retrieval_port,
                self.calibration,
                target_seconds,
                duration_tolerance_ratio,
                self.is_embedding_enabled,
                agent_client=self.agent_client,
                live_draft_session=live_draft_session,
            )
            if restored_live_snapshot is None:
                self.event_bus.publish(
                    "system_message",
                    f"REVISE MODE - loaded {len(registry)} messages from previous draft",
                )
        else:
            if live_draft_session is not None:
                live_draft_session.clear()
            tool_registry = build_summarizer_tools(
                registry,
                retrieval_port,
                self.calibration,
                target_seconds,
                duration_tolerance_ratio,
                self.is_embedding_enabled,
                agent_client=self.agent_client,
                live_draft_session=live_draft_session,
            )

        transcript_excerpt = ""
        total_words = 0
        num_segments = 0

        if self.is_embedding_enabled:
            query_sys_prompt = PromptManager.get_prompt("query_generation_system")
            query_user_prompt = (
                "Goal: Find main topics, key arguments, and important lessons for a "
                f"roughly {target_minutes} minute summary.\n"
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

        prompt_kwargs = {
            "target_seconds": target_seconds,
            "target_minutes": target_minutes,
            "duration_tolerance_ratio": duration_tolerance_ratio,
            "duration_tolerance_percent": round(duration_tolerance_ratio * 100.0, 2),
            "min_seconds": min_seconds,
            "max_seconds": max_seconds,
            "emo_preset_guide": emo_preset_guide,
            "is_embedding_enabled": self.is_embedding_enabled,
            "staged_discovery_enabled": self.is_embedding_enabled
            and loop_guardrails["enable_staged_discovery"],
            "required_discovery_queries": loop_guardrails["required_discovery_queries"],
            "max_discovery_queries": loop_guardrails["max_discovery_queries"],
            "require_unique_discovery_queries": loop_guardrails[
                "require_unique_discovery_queries"
            ],
        }
        if revise_mode:
            system_prompt = PromptManager.get_prompt(
                "summarizer_revise",
                previous_draft=previous_draft,
                draft_line_map=json.dumps(registry.snapshot(), indent=2),
                feedback=feedback,
                loop_context_block=loop_context_block,
                **prompt_kwargs,
            )
        else:
            system_prompt = PromptManager.get_prompt(
                "summarizer_system",
                **prompt_kwargs,
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
            target_seconds=target_seconds,
            duration_tolerance_ratio=duration_tolerance_ratio,
            loop_guardrails=loop_guardrails,
        )

        result_xml = serialize_summary_turns(registry.get_all()).strip()

        self.event_bus.publish(
            "agent_message",
            self.name,
            f"Tool-generated XML output:\n```xml\n{result_xml}\n```",
        )
        await self.send_response(result_xml, event_queue)

    @staticmethod
    def _load_draft_into_registry(registry: MessageRegistry, draft_xml: str) -> None:
        """Parse existing summary XML back into the registry."""
        for turn in parse_summary_xml(draft_xml):
            registry.add(turn.speaker, turn.text, turn.emo_preset)

    async def process_tool_calls(
        self, tool_calls: list[dict[str, Any]], messages: list, context_data: dict
    ) -> tuple[bool, dict | None]:
        """Delegate tool execution to the shared dispatcher."""
        return await self._tool_dispatcher.process_tool_calls(
            tool_calls=tool_calls,
            messages=messages,
            context_data=context_data,
        )
