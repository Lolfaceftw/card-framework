from __future__ import annotations

import json

import pytest

from card_framework.agents.prompt_compaction import (
    ContextBudgetManager,
    ContextWindowProfile,
    LLMDrivenPromptCompactionStrategy,
    PromptCompactionConfig,
    PromptCompactionRequest,
    apply_compaction_to_messages,
    build_context_budget_manager,
    estimate_prompt_tokens_from_text_blocks,
    is_context_window_overflow_error,
)


class _FakeCompactionLLM:
    def __init__(self, response_payload: dict[str, str]) -> None:
        self.calls: list[dict[str, object]] = []
        self.response_payload = response_payload

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
            }
        )
        return json.dumps(self.response_payload)


def test_context_budget_manager_triggers_compaction_at_configured_ratio() -> None:
    manager = ContextBudgetManager(
        config=PromptCompactionConfig(enabled=True, trigger_ratio=0.8),
        profiles={
            "deepseek_chat": ContextWindowProfile(
                profile_name="deepseek_chat",
                context_window_tokens=100,
                compaction_target_ratio=0.65,
            )
        },
        token_estimator=lambda blocks: 80,
    )

    snapshot = manager.snapshot_for_text_blocks(
        profile_name="deepseek_chat",
        text_blocks=["ignored"],
    )

    assert snapshot.estimated_tokens == 80
    assert snapshot.usage_ratio == pytest.approx(0.8)
    assert snapshot.should_compact is True


def test_build_context_budget_manager_reads_profile_metadata() -> None:
    manager = build_context_budget_manager(
        orchestrator_cfg={
            "prompt_compaction": {
                "enabled": True,
                "trigger_ratio": 0.8,
                "retry_on_overflow": True,
                "max_overflow_retries": 1,
                "compress_with_same_provider": True,
                "keep_latest_iterations": 1,
            }
        },
        llm_profiles_cfg={
            "deepseek_chat": {
                "context_window_tokens": 131072,
                "compaction_target_ratio": 0.65,
            }
        },
    )

    profile = manager.profiles["deepseek_chat"]
    assert manager.config.trigger_ratio == pytest.approx(0.8)
    assert profile.context_window_tokens == 131072
    assert profile.compaction_target_ratio == pytest.approx(0.65)
    assert manager.target_token_budget(profile_name="deepseek_chat") == 85196


def test_estimate_prompt_tokens_from_text_blocks_uses_conservative_fallback() -> None:
    estimated = estimate_prompt_tokens_from_text_blocks(["abcdefgh"])
    assert estimated == 2


def test_build_context_budget_manager_registers_default_vllm_profile() -> None:
    manager = build_context_budget_manager(
        orchestrator_cfg={"prompt_compaction": {"enabled": True, "trigger_ratio": 0.8}},
        llm_profiles_cfg={
            "vllm_local": {
                "context_window_tokens": 32768,
                "compaction_target_ratio": 0.65,
            }
        },
    )

    assert manager.profiles["vllm_local"].context_window_tokens == 32768


def test_snapshot_for_messages_counts_tool_call_payloads() -> None:
    manager = ContextBudgetManager(
        config=PromptCompactionConfig(enabled=True, trigger_ratio=0.5),
        profiles={
            "deepseek_chat": ContextWindowProfile(
                profile_name="deepseek_chat",
                context_window_tokens=20,
            )
        },
        token_estimator=estimate_prompt_tokens_from_text_blocks,
    )

    snapshot = manager.snapshot_for_messages(
        profile_name="deepseek_chat",
        messages=[
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "add_speaker_message",
                            "arguments": '{"speaker_id":"SPEAKER_00","content":"hello"}',
                        },
                    }
                ],
            }
        ],
    )

    assert snapshot.estimated_tokens > 0


def test_llm_driven_prompt_compaction_strategy_uses_provider_output() -> None:
    llm = _FakeCompactionLLM(
        {
            "compressed_transcript_context": "short transcript grounding",
            "compressed_history_context": "short history summary",
        }
    )
    strategy = LLMDrivenPromptCompactionStrategy(
        llm=llm,
        token_estimator=estimate_prompt_tokens_from_text_blocks,
    )

    result = strategy.compact(
        request=PromptCompactionRequest(
            profile_name="deepseek_chat",
            current_draft="<SPEAKER_00>Hello</SPEAKER_00>",
            latest_feedback="Too long",
            loop_context="Fix chronology",
            transcript_context="Very long transcript context",
            target_token_budget=65,
            source_iteration_span=(1, 3),
            messages=(
                {"role": "system", "content": "Summarizer"},
                {"role": "user", "content": "Transcript here"},
            ),
        )
    )

    assert llm.calls
    assert '"target_token_budget": 65' in str(llm.calls[0]["user_prompt"])
    assert result.compression_reason == "llm_same_provider"
    assert result.compressed_transcript_context == "short transcript grounding"
    assert result.compressed_history_context == "short history summary"


def test_apply_compaction_to_messages_preserves_system_and_recent_tail() -> None:
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old transcript"},
        {"role": "assistant", "content": "old reply"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "query_transcript", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "query_transcript", "content": "{}"},
    ]

    compacted = apply_compaction_to_messages(
        messages=messages,
        compaction=LLMDrivenPromptCompactionStrategy(
            llm=_FakeCompactionLLM({}),
            token_estimator=estimate_prompt_tokens_from_text_blocks,
        ).compact(
            request=PromptCompactionRequest(
                profile_name="deepseek_chat",
                current_draft="draft",
                latest_feedback="feedback",
                loop_context="loop",
                transcript_context="transcript",
                messages=tuple(messages),
            )
        ),
        keep_latest_iterations=1,
    )

    assert compacted[0]["role"] == "system"
    assert "Compacted prior summarizer context" in str(compacted[1]["content"])
    assert compacted[-1]["role"] == "tool"


def test_apply_compaction_to_messages_replaces_prior_synthetic_summary() -> None:
    base_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old transcript"},
    ]
    compaction_result = LLMDrivenPromptCompactionStrategy(
        llm=_FakeCompactionLLM(
            {
                "compressed_transcript_context": "short transcript grounding",
                "compressed_history_context": "short history summary",
            }
        ),
        token_estimator=estimate_prompt_tokens_from_text_blocks,
    ).compact(
        request=PromptCompactionRequest(
            profile_name="deepseek_chat",
            current_draft="draft",
            latest_feedback="feedback",
            loop_context="loop",
            transcript_context="transcript",
            messages=tuple(base_messages),
        )
    )

    first = apply_compaction_to_messages(
        messages=base_messages,
        compaction=compaction_result,
        keep_latest_iterations=1,
    )
    second = apply_compaction_to_messages(
        messages=first,
        compaction=compaction_result,
        keep_latest_iterations=1,
    )

    synthetic_messages = [
        message
        for message in second
        if "Compacted prior summarizer context" in str(message.get("content", ""))
    ]
    assert len(synthetic_messages) == 1


def test_is_context_window_overflow_error_matches_deepseek_failure_text() -> None:
    error = RuntimeError(
        "Error code: 400 - {'error': {'message': \"This model's maximum context "
        "length is 131072 tokens. However, you requested 133783 tokens.\"}}"
    )

    assert is_context_window_overflow_error(error) is True
