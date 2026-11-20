"""Tests for agent-name inference used by streamed provider callbacks."""

from card_framework.shared.llm_provider import (
    infer_agent_name,
    infer_agent_name_from_text,
)


def test_infer_agent_name_from_text_prefers_earliest_role_match() -> None:
    text = (
        "You are an elite Editorial Critic and Compliance Enforcer. "
        "Strictly provide the checklist for the summarizer agent to use."
    )

    assert infer_agent_name_from_text(text) == "Critic"


def test_infer_agent_name_from_text_keeps_summarizer_revise_labeled_correctly() -> None:
    text = (
        "You are an elite Editorial Summarizer operating in REVISE MODE. "
        "A previous draft was submitted but the Critic found issues."
    )

    assert infer_agent_name_from_text(text) == "Summarizer"


def test_infer_agent_name_uses_latest_system_message() -> None:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an elite Editorial Critic and Compliance Enforcer. "
                "Strictly provide the checklist for the summarizer agent to use."
            ),
        },
        {"role": "user", "content": "Evaluate the draft."},
    ]

    assert infer_agent_name(messages) == "Critic"
