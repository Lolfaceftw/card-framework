"""Prompt-budget policies and typed compaction contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import re
from typing import Any, Protocol

from card_framework.shared.llm_provider import (
    LLMProvider,
    MessageInput,
    normalize_messages,
)

_CONTEXT_WINDOW_ERROR_PATTERNS: tuple[str, ...] = (
    "maximum context length",
    "context length is",
    "context window",
    "context_length_exceeded",
    "please reduce the length of the messages",
    "requested ",
)
_COMPACTED_CONTEXT_HEADER = "Compacted prior summarizer context."
_MAX_COMPACTION_DRAFT_CHARS = 12_000
_MAX_COMPACTION_FEEDBACK_CHARS = 4_000
_MAX_COMPACTION_LOOP_CONTEXT_CHARS = 8_000
_MAX_COMPACTION_TRANSCRIPT_CHARS = 48_000
_MAX_COMPACTION_HISTORY_CHARS = 16_000


@dataclass(slots=True, frozen=True)
class PromptCompactionConfig:
    """Configurable controls for same-provider prompt compaction."""

    enabled: bool = True
    trigger_ratio: float = 0.8
    retry_on_overflow: bool = True
    max_overflow_retries: int = 1
    compress_with_same_provider: bool = True
    keep_latest_iterations: int = 1

    def __post_init__(self) -> None:
        if not 0.0 < self.trigger_ratio <= 1.0:
            raise ValueError("trigger_ratio must be within (0.0, 1.0].")
        if self.max_overflow_retries < 0:
            raise ValueError("max_overflow_retries must be >= 0.")
        if self.keep_latest_iterations < 0:
            raise ValueError("keep_latest_iterations must be >= 0.")


@dataclass(slots=True, frozen=True)
class ContextWindowProfile:
    """Provider-profile context-window metadata used by the budget manager."""

    profile_name: str
    context_window_tokens: int
    compaction_target_ratio: float = 0.65

    def __post_init__(self) -> None:
        if not str(self.profile_name).strip():
            raise ValueError("profile_name must be non-empty.")
        if self.context_window_tokens <= 0:
            raise ValueError("context_window_tokens must be > 0.")
        if not 0.0 < self.compaction_target_ratio <= 1.0:
            raise ValueError("compaction_target_ratio must be within (0.0, 1.0].")


@dataclass(slots=True, frozen=True)
class ContextBudgetSnapshot:
    """Estimated prompt-budget state before one LLM call."""

    profile_name: str
    estimated_tokens: int
    context_window_tokens: int
    usage_ratio: float
    trigger_ratio: float
    should_compact: bool


@dataclass(slots=True, frozen=True)
class PromptCompactionRequest:
    """Structured payload supplied to a prompt-compaction strategy."""

    profile_name: str
    current_draft: str
    latest_feedback: str
    loop_context: str
    transcript_context: str
    target_token_budget: int = 0
    source_iteration_span: tuple[int, int] | None = None
    messages: tuple[dict[str, object], ...] = ()


@dataclass(slots=True, frozen=True)
class PromptCompactionResult:
    """Structured result emitted by a prompt-compaction strategy."""

    compressed_transcript_context: str
    compressed_history_context: str
    compression_reason: str
    source_iteration_span: tuple[int, int] | None
    estimated_tokens_before: int
    estimated_tokens_after: int


class PromptCompactionStrategy(Protocol):
    """Protocol for strategies that compact prompt history/context."""

    def compact(
        self,
        *,
        request: PromptCompactionRequest,
    ) -> PromptCompactionResult:
        """Compact the supplied prompt payload into a smaller structured form."""


@dataclass(slots=True, frozen=True)
class NoOpPromptCompactionStrategy:
    """Placeholder strategy used before live compaction is wired into the loop."""

    token_estimator: Callable[[Sequence[str]], int]

    def compact(
        self,
        *,
        request: PromptCompactionRequest,
    ) -> PromptCompactionResult:
        """Return the original fields unchanged while preserving structure."""
        history_context = _messages_to_text(request.messages)
        text_blocks = (
            request.current_draft,
            request.latest_feedback,
            request.loop_context,
            request.transcript_context,
            history_context,
        )
        estimated_tokens = self.token_estimator(text_blocks)
        return PromptCompactionResult(
            compressed_transcript_context=request.transcript_context,
            compressed_history_context=history_context,
            compression_reason="noop",
            source_iteration_span=request.source_iteration_span,
            estimated_tokens_before=estimated_tokens,
            estimated_tokens_after=estimated_tokens,
        )


@dataclass(slots=True, frozen=True)
class LLMDrivenPromptCompactionStrategy:
    """Use the active LLM provider to compact transcript and history context."""

    llm: LLMProvider
    token_estimator: Callable[[Sequence[str]], int]
    max_tokens: int = 1_200

    def compact(
        self,
        *,
        request: PromptCompactionRequest,
    ) -> PromptCompactionResult:
        """Return an LLM-compressed structured context payload."""
        history_context = _messages_to_text(request.messages)
        estimated_tokens_before = self.token_estimator(
            (
                request.current_draft,
                request.latest_feedback,
                request.loop_context,
                request.transcript_context,
                history_context,
            )
        )
        bounded_payload = {
            "profile_name": request.profile_name,
            "target_token_budget": request.target_token_budget,
            "source_iteration_span": list(request.source_iteration_span)
            if request.source_iteration_span is not None
            else None,
            "current_draft": _truncate_text(
                request.current_draft,
                max_chars=_MAX_COMPACTION_DRAFT_CHARS,
            ),
            "latest_feedback": _truncate_text(
                request.latest_feedback,
                max_chars=_MAX_COMPACTION_FEEDBACK_CHARS,
            ),
            "loop_context": _truncate_text(
                request.loop_context,
                max_chars=_MAX_COMPACTION_LOOP_CONTEXT_CHARS,
            ),
            "transcript_context": _truncate_text(
                request.transcript_context,
                max_chars=_MAX_COMPACTION_TRANSCRIPT_CHARS,
            ),
            "message_history": _truncate_text(
                history_context,
                max_chars=_MAX_COMPACTION_HISTORY_CHARS,
            ),
        }
        system_prompt = (
            "You compress summarizer prompt context so it fits within a model "
            "context window without losing the current draft intent. Return strict "
            'JSON with keys "compressed_transcript_context" and '
            '"compressed_history_context". Preserve facts, chronology, named entities, '
            "speaker constraints, unresolved critic guidance, and duration-sensitive "
            "requirements. Do not include markdown fences. When target_token_budget "
            "is present, aim to keep the compacted transcript plus compacted history "
            "within that target."
        )
        user_prompt = json.dumps(bounded_payload, ensure_ascii=False)

        compression_reason = "llm_same_provider"
        try:
            raw_response = self.llm.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens,
            )
            parsed = _parse_compaction_response(raw_response)
        except Exception:
            parsed = None

        if parsed is None:
            compression_reason = "heuristic_fallback"
            compressed_transcript_context = _truncate_text(
                request.transcript_context,
                max_chars=_MAX_COMPACTION_TRANSCRIPT_CHARS // 2,
            )
            compressed_history_context = _truncate_text(
                history_context,
                max_chars=_MAX_COMPACTION_HISTORY_CHARS // 2,
            )
        else:
            compressed_transcript_context = str(
                parsed.get("compressed_transcript_context", "")
            ).strip()
            compressed_history_context = str(
                parsed.get("compressed_history_context", "")
            ).strip()
            if not compressed_transcript_context:
                compressed_transcript_context = _truncate_text(
                    request.transcript_context,
                    max_chars=_MAX_COMPACTION_TRANSCRIPT_CHARS // 2,
                )
            if not compressed_history_context:
                compressed_history_context = _truncate_text(
                    history_context,
                    max_chars=_MAX_COMPACTION_HISTORY_CHARS // 2,
                )

        estimated_tokens_after = self.token_estimator(
            (compressed_transcript_context, compressed_history_context)
        )
        return PromptCompactionResult(
            compressed_transcript_context=compressed_transcript_context,
            compressed_history_context=compressed_history_context,
            compression_reason=compression_reason,
            source_iteration_span=request.source_iteration_span,
            estimated_tokens_before=estimated_tokens_before,
            estimated_tokens_after=estimated_tokens_after,
        )


@dataclass(slots=True, frozen=True)
class ContextBudgetManager:
    """Estimate prompt usage and decide when prompt compaction should trigger."""

    config: PromptCompactionConfig
    profiles: Mapping[str, ContextWindowProfile]
    token_estimator: Callable[[Sequence[str]], int]

    def snapshot_for_text_blocks(
        self,
        *,
        profile_name: str,
        text_blocks: Sequence[str],
    ) -> ContextBudgetSnapshot:
        """Estimate one budget snapshot from prompt text blocks."""
        profile = self.profiles.get(profile_name)
        if profile is None:
            return ContextBudgetSnapshot(
                profile_name=profile_name,
                estimated_tokens=0,
                context_window_tokens=0,
                usage_ratio=0.0,
                trigger_ratio=self.config.trigger_ratio,
                should_compact=False,
            )
        estimated_tokens = self.token_estimator(text_blocks)
        usage_ratio = estimated_tokens / float(profile.context_window_tokens)
        return ContextBudgetSnapshot(
            profile_name=profile_name,
            estimated_tokens=estimated_tokens,
            context_window_tokens=profile.context_window_tokens,
            usage_ratio=usage_ratio,
            trigger_ratio=self.config.trigger_ratio,
            should_compact=(
                self.config.enabled and usage_ratio >= self.config.trigger_ratio
            ),
        )

    def snapshot_for_messages(
        self,
        *,
        profile_name: str,
        messages: Sequence[MessageInput],
        extra_text_blocks: Sequence[str] = (),
    ) -> ContextBudgetSnapshot:
        """Estimate one budget snapshot from normalized messages plus extra text."""
        text_blocks = list(extra_text_blocks)
        text_blocks.append(_messages_to_text(normalize_messages(messages)))
        return self.snapshot_for_text_blocks(
            profile_name=profile_name,
            text_blocks=text_blocks,
        )

    def target_token_budget(self, *, profile_name: str) -> int:
        """Return the configured post-compaction target token budget for one profile."""
        profile = self.profiles.get(profile_name)
        if profile is None:
            return 0
        return int(profile.context_window_tokens * profile.compaction_target_ratio)


def build_context_budget_manager(
    *,
    orchestrator_cfg: Mapping[str, Any],
    llm_profiles_cfg: Mapping[str, Any],
) -> ContextBudgetManager:
    """Build the prompt-budget manager from runtime config mappings."""
    prompt_compaction_cfg = _as_mapping(orchestrator_cfg.get("prompt_compaction", {}))
    config = PromptCompactionConfig(
        enabled=bool(prompt_compaction_cfg.get("enabled", True)),
        trigger_ratio=float(prompt_compaction_cfg.get("trigger_ratio", 0.8)),
        retry_on_overflow=bool(prompt_compaction_cfg.get("retry_on_overflow", True)),
        max_overflow_retries=int(
            prompt_compaction_cfg.get("max_overflow_retries", 1)
        ),
        compress_with_same_provider=bool(
            prompt_compaction_cfg.get("compress_with_same_provider", True)
        ),
        keep_latest_iterations=int(
            prompt_compaction_cfg.get("keep_latest_iterations", 1)
        ),
    )
    profiles: dict[str, ContextWindowProfile] = {}
    for profile_name, raw_payload in llm_profiles_cfg.items():
        if not isinstance(raw_payload, Mapping):
            continue
        context_window_tokens = int(raw_payload.get("context_window_tokens", 0) or 0)
        if context_window_tokens <= 0:
            continue
        compaction_target_ratio = float(
            raw_payload.get("compaction_target_ratio", 0.65)
        )
        profiles[str(profile_name)] = ContextWindowProfile(
            profile_name=str(profile_name),
            context_window_tokens=context_window_tokens,
            compaction_target_ratio=compaction_target_ratio,
        )
    return ContextBudgetManager(
        config=config,
        profiles=profiles,
        token_estimator=estimate_prompt_tokens_from_text_blocks,
    )


def estimate_prompt_tokens_from_text_blocks(text_blocks: Sequence[str]) -> int:
    """Estimate prompt tokens conservatively from plain-text blocks."""
    normalized = "\n".join(str(block) for block in text_blocks if str(block))
    if not normalized.strip():
        return 0
    return int(math.ceil(len(normalized) / 4.0))


def is_context_window_overflow_error(error: BaseException) -> bool:
    """Return whether one provider error looks like a context-window overflow."""
    message = str(error).strip().lower()
    if not message:
        return False
    return any(pattern in message for pattern in _CONTEXT_WINDOW_ERROR_PATTERNS)


def apply_compaction_to_messages(
    *,
    messages: Sequence[MessageInput],
    compaction: PromptCompactionResult,
    keep_latest_iterations: int,
) -> list[dict[str, object]]:
    """Replace older prompt history with one structured compacted-context message."""
    normalized = normalize_messages(messages)
    if not normalized:
        return []
    normalized = [
        dict(message)
        for message in normalized
        if not _is_synthetic_compaction_message(message)
    ]
    if not normalized:
        return []

    first_system: dict[str, object] | None = None
    remaining_messages = normalized
    if normalized[0].get("role") == "system":
        first_system = normalized[0]
        remaining_messages = normalized[1:]

    tail_message_count = max(2, max(0, keep_latest_iterations) * 4)
    retained_start_index = max(0, len(remaining_messages) - tail_message_count)
    while (
        retained_start_index > 0
        and remaining_messages[retained_start_index].get("role") == "tool"
    ):
        retained_start_index -= 1
    retained_tail = [dict(message) for message in remaining_messages[retained_start_index:]]

    compacted_messages: list[dict[str, object]] = []
    if first_system is not None:
        compacted_messages.append(first_system)
    compacted_messages.append(
        {
            "role": "system",
            "content": _render_compacted_context_message(compaction),
        }
    )
    compacted_messages.extend(retained_tail)
    return compacted_messages


def _messages_to_text(messages: Sequence[MessageInput]) -> str:
    """Render one message history into plain text for token estimation."""
    if not messages:
        return ""
    normalized = normalize_messages(messages)
    lines: list[str] = []
    for message in normalized:
        role = str(message.get("role", "user")).strip() or "user"
        content = str(message.get("content", ""))
        tool_calls = message.get("tool_calls")
        rendered_parts: list[str] = []
        if content:
            rendered_parts.append(content)
        if isinstance(tool_calls, Sequence) and not isinstance(tool_calls, (str, bytes)):
            for tool_call in tool_calls:
                if not isinstance(tool_call, Mapping):
                    continue
                function_payload = tool_call.get("function", {})
                function_mapping = (
                    function_payload if isinstance(function_payload, Mapping) else {}
                )
                tool_name = str(function_mapping.get("name", "")).strip() or "unknown_tool"
                tool_arguments = str(function_mapping.get("arguments", "")).strip()
                rendered_parts.append(
                    f"[tool_call:{tool_name}] {tool_arguments}".strip()
                )
        tool_name = str(message.get("name", "")).strip()
        tool_call_id = str(message.get("tool_call_id", "")).strip()
        if role == "tool" and tool_name:
            rendered_parts.append(f"[tool_name:{tool_name}]")
        if role == "tool" and tool_call_id:
            rendered_parts.append(f"[tool_call_id:{tool_call_id}]")
        if not rendered_parts:
            continue
        lines.append(f"[{role}] {' '.join(rendered_parts)}".strip())
    return "\n".join(lines)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return one mapping-like config payload as a plain mapping."""
    return value if isinstance(value, Mapping) else {}


def _parse_compaction_response(raw_response: str) -> dict[str, object] | None:
    """Extract one JSON object from raw LLM compaction output."""
    raw_text = str(raw_response).strip()
    if not raw_text:
        return None
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match is None:
            return None
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _render_compacted_context_message(compaction: PromptCompactionResult) -> str:
    """Render one structured compacted context note for message history replay."""
    sections = [
        f"{_COMPACTED_CONTEXT_HEADER} Treat this as the authoritative "
        "summary of older transcript/history that was removed to fit the context window.",
        "Compacted history context:",
        compaction.compressed_history_context or "(none)",
        "",
        "Compacted transcript grounding:",
        compaction.compressed_transcript_context or "(none)",
    ]
    return "\n".join(sections).strip()


def _truncate_text(text: str, *, max_chars: int) -> str:
    """Keep head and tail context while bounding one text block by character count."""
    normalized = str(text).strip()
    if len(normalized) <= max_chars:
        return normalized
    if max_chars <= 32:
        return normalized[:max_chars]
    head_chars = max_chars // 2
    tail_chars = max_chars - head_chars - 17
    return (
        f"{normalized[:head_chars]}\n[... trimmed ...]\n"
        f"{normalized[-max(0, tail_chars):]}"
    )


def _is_synthetic_compaction_message(message: Mapping[str, object]) -> bool:
    """Return whether one history message is a synthetic compaction note."""
    if str(message.get("role", "")).strip() != "system":
        return False
    content = str(message.get("content", "")).strip()
    return content.startswith(_COMPACTED_CONTEXT_HEADER)
