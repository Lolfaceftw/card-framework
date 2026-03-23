from card_framework.agents.dtos import AssistantMessage, Function, ToolCall
from card_framework.audio_pipeline.interjector import (
    LLMInterjectionPlanner,
    _build_eligible_turns,
    _build_heuristic_interjection_decisions,
)
from card_framework.audio_pipeline.voice_clone_contracts import VoiceCloneTurn
from card_framework.providers.logging_provider import LoggingLLMProvider
from card_framework.shared.llm_provider import LLMProvider


class _ThinkingLLMProvider(LLMProvider):
    """Minimal fake provider that exposes an ``enable_thinking`` flag."""

    def __init__(self) -> None:
        self.enable_thinking = True
        self.chat_calls = 0
        self.generate_calls = 0

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        del system_prompt, user_prompt, max_tokens
        self.generate_calls += 1
        assert self.enable_thinking is False
        return '{"decisions":[{"host_turn_index":1,"should_interject":false}]}'

    def chat(
        self,
        messages,
        tools=None,
        tool_choice=None,
        max_tokens: int | None = None,
    ):
        del messages, tools, tool_choice, max_tokens
        self.chat_calls += 1
        assert self.enable_thinking is False
        return AssistantMessage(
            tool_calls=[
                ToolCall(
                    id="call_1",
                    function=Function(
                        name="submit_interjection_decision",
                        arguments='{"host_turn_index":1,"should_interject":false}',
                    ),
                )
            ]
        )


class _FallbackThinkingLLMProvider(_ThinkingLLMProvider):
    """Fake provider that forces the planner onto the text fallback path."""

    def chat(
        self,
        messages,
        tools=None,
        tool_choice=None,
        max_tokens: int | None = None,
    ):
        del messages, tools, tool_choice, max_tokens
        self.chat_calls += 1
        assert self.enable_thinking is False
        raise RuntimeError("tool calling unavailable")


def test_interjection_planner_uses_tool_call_and_restores_thinking() -> None:
    provider = _ThinkingLLMProvider()
    planner = LLMInterjectionPlanner(llm=LoggingLLMProvider(inner_provider=provider))

    decisions = planner.plan(
        [
            VoiceCloneTurn(
                speaker="SPEAKER_00",
                text="one two three four five six seven eight nine ten eleven twelve",
            ),
            VoiceCloneTurn(
                speaker="SPEAKER_01",
                text="yeah, that tracks for me too",
            ),
        ]
    )

    assert provider.chat_calls == 1
    assert provider.generate_calls == 0
    assert provider.enable_thinking is True
    assert len(decisions) == 1
    assert decisions[0].host_turn_index == 1
    assert decisions[0].should_interject is False


def test_interjection_planner_can_skip_tool_call_path() -> None:
    """Support the low-request planner mode that uses one text response only."""
    provider = _ThinkingLLMProvider()
    planner = LLMInterjectionPlanner(
        llm=LoggingLLMProvider(inner_provider=provider),
        prefer_tool_call_planning=False,
    )

    decisions = planner.plan(
        [
            VoiceCloneTurn(
                speaker="SPEAKER_00",
                text="one two three four five six seven eight nine ten eleven twelve",
            ),
            VoiceCloneTurn(
                speaker="SPEAKER_01",
                text="yeah, that tracks for me too",
            ),
        ]
    )

    assert provider.chat_calls == 0
    assert provider.generate_calls == 1
    assert provider.enable_thinking is True
    assert len(decisions) == 1
    assert decisions[0].host_turn_index == 1
    assert decisions[0].should_interject is False


def test_interjection_planner_falls_back_to_generate_when_tool_call_missing() -> None:
    provider = _FallbackThinkingLLMProvider()
    planner = LLMInterjectionPlanner(llm=LoggingLLMProvider(inner_provider=provider))

    decisions = planner.plan(
        [
            VoiceCloneTurn(
                speaker="SPEAKER_00",
                text="one two three four five six seven eight nine ten eleven twelve",
            ),
            VoiceCloneTurn(
                speaker="SPEAKER_01",
                text="yeah, that tracks for me too",
            ),
        ]
    )

    assert provider.chat_calls == 1
    assert provider.generate_calls == 1
    assert provider.enable_thinking is True
    assert len(decisions) == 1
    assert decisions[0].host_turn_index == 1
    assert decisions[0].should_interject is False


def test_heuristic_interjection_fallback_builds_grounded_backchannel() -> None:
    eligible_turns = _build_eligible_turns(
        [
            VoiceCloneTurn(
                speaker="SPEAKER_00",
                text="one two three four five six seven eight nine ten eleven twelve",
            ),
            VoiceCloneTurn(
                speaker="SPEAKER_01",
                text="Totally, that tracks for me too.",
            ),
            VoiceCloneTurn(
                speaker="SPEAKER_00",
                text="alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu",
            ),
            VoiceCloneTurn(
                speaker="SPEAKER_01",
                text="Exactly, that is the key point.",
            ),
        ]
    )

    decisions = _build_heuristic_interjection_decisions(
        eligible_turns,
        max_interjection_words=5,
        min_host_progress_ratio=0.35,
        max_host_progress_ratio=0.75,
        max_interjections=1,
    )

    assert len(decisions) == 3
    assert decisions[0].should_interject is True
    assert decisions[0].interjection_speaker == "SPEAKER_01"
    assert decisions[0].interjection_text == "Totally."
    assert decisions[0].anchor_start_token_index == decisions[0].anchor_end_token_index
    assert decisions[1].should_interject is False
    assert decisions[2].should_interject is False
