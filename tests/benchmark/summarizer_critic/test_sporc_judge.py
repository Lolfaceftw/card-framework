"""Judge coverage for the SPoRC benchmark."""

from __future__ import annotations

import json

from card_framework.benchmark.summarizer_critic.sporc.judge import SPoRCJudge
from card_framework.shared.prompt_manager import PromptManager


class _FakeJudgeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.system_prompt = ""
        self.user_prompt = ""
        self.max_tokens: int | None = -1

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.max_tokens = max_tokens
        return self.response


def test_sporc_judge_parses_three_required_dimensions() -> None:
    llm = _FakeJudgeLLM(
        '{"Factualness": 0.9, "Naturalness": 0.8, '
        '"Speaker Grammar Similarity": 0.7, "confidence": 0.6, "rationale": "ok"}'
    )
    judge = SPoRCJudge(judge_llm=llm)

    result = judge.evaluate(
        source_text="[SPEAKER_00]: hello",
        summary_xml='<SPEAKER_00 emo_preset="neutral">hello</SPEAKER_00>',
        transcript_per_speaker_messages={"SPEAKER_00": ["hello"]},
        summary_per_speaker_messages={"SPEAKER_00": ["hello"]},
        speaker_alias_map={"Alex Host": "SPEAKER_00"},
        speaker_notes=["hostSpeakerLabels: {'Alex Host': ['SPEAKER_00']}"],
    )

    assert result.status == "ok"
    assert result.scores == {
        "Factualness": 0.9,
        "Naturalness": 0.8,
        "Speaker Grammar Similarity": 0.7,
    }
    assert "Transcript per-speaker message lists" in llm.user_prompt
    assert "Speaker alias mapping" in llm.user_prompt
    assert "Guide questions" in llm.user_prompt
    assert "5-point Likert scale" in llm.user_prompt
    assert "F1." in llm.user_prompt
    assert "N1." in llm.user_prompt
    assert "S1." in llm.user_prompt
    assert "1 -> 0.00" in llm.user_prompt
    assert "average the three normalized item scores" in llm.user_prompt
    assert '"item_scores"' in llm.user_prompt
    assert llm.max_tokens is None


def test_sporc_judge_parses_optional_item_scores() -> None:
    llm = _FakeJudgeLLM(
        json.dumps(
            {
                "item_scores": {
                    "F1": 1.0,
                    "F2": 0.75,
                    "F3": 0.5,
                    "N1": 1.0,
                    "N2": 0.75,
                    "N3": 0.75,
                    "S1": 0.75,
                    "S2": 0.5,
                    "S3": 1.0,
                },
                "Factualness": 0.75,
                "Naturalness": 0.8333,
                "Speaker Grammar Similarity": 0.75,
                "confidence": 0.6,
                "rationale": "ok",
            }
        )
    )
    judge = SPoRCJudge(judge_llm=llm)

    result = judge.evaluate(
        source_text="[SPEAKER_00]: hello",
        summary_xml='<SPEAKER_00 emo_preset="neutral">hello</SPEAKER_00>',
        transcript_per_speaker_messages={"SPEAKER_00": ["hello"]},
        summary_per_speaker_messages={"SPEAKER_00": ["hello"]},
        speaker_alias_map={},
        speaker_notes=[],
    )

    assert result.status == "ok"
    assert result.item_scores is not None
    assert result.item_scores["F1"] == 1.0
    assert result.item_scores["S2"] == 0.5
    assert result.confidence == 0.6
    assert result.rationale == "ok"


def test_sporc_judge_derives_dimension_scores_from_items_when_present() -> None:
    llm = _FakeJudgeLLM(
        json.dumps(
            {
                "item_scores": {
                    "F1": 1.0,
                    "F2": 0.75,
                    "F3": 0.5,
                    "N1": 1.0,
                    "N2": 0.75,
                    "N3": 0.75,
                    "S1": 0.75,
                    "S2": 0.5,
                    "S3": 1.0,
                },
                "Factualness": 0.0,
                "Naturalness": 0.0,
                "Speaker Grammar Similarity": 0.0,
            }
        )
    )
    judge = SPoRCJudge(judge_llm=llm)

    result = judge.evaluate(
        source_text="[SPEAKER_00]: hello",
        summary_xml='<SPEAKER_00 emo_preset="neutral">hello</SPEAKER_00>',
        transcript_per_speaker_messages={"SPEAKER_00": ["hello"]},
        summary_per_speaker_messages={"SPEAKER_00": ["hello"]},
        speaker_alias_map={},
        speaker_notes=[],
    )

    assert result.status == "ok"
    assert result.scores == {
        "Factualness": 0.75,
        "Naturalness": 0.8333,
        "Speaker Grammar Similarity": 0.75,
    }


def test_sporc_judge_rejects_out_of_range_scores_instead_of_clamping() -> None:
    llm = _FakeJudgeLLM(
        json.dumps(
            {
                "item_scores": {
                    "F1": 5,
                    "F2": 0.75,
                    "F3": 0.5,
                    "N1": 1.0,
                    "N2": 0.75,
                    "N3": 0.75,
                    "S1": 0.75,
                    "S2": 0.5,
                    "S3": 1.0,
                },
                "Factualness": 5,
                "Naturalness": 0.75,
                "Speaker Grammar Similarity": 0.75,
            }
        )
    )
    judge = SPoRCJudge(judge_llm=llm)

    result = judge.evaluate(
        source_text="[SPEAKER_00]: hello",
        summary_xml='<SPEAKER_00 emo_preset="neutral">hello</SPEAKER_00>',
        transcript_per_speaker_messages={"SPEAKER_00": ["hello"]},
        summary_per_speaker_messages={"SPEAKER_00": ["hello"]},
        speaker_alias_map={},
        speaker_notes=[],
    )

    assert result.status == "error"
    assert result.scores is None


def test_sporc_judge_includes_full_transcript_in_prompt_by_default() -> None:
    llm = _FakeJudgeLLM(
        '{"Factualness": 1.0, "Naturalness": 1.0, '
        '"Speaker Grammar Similarity": 1.0, "confidence": 1.0, "rationale": "ok"}'
    )
    judge = SPoRCJudge(judge_llm=llm)
    source_text = f"{'A' * 30000}TAIL_MARKER"

    result = judge.evaluate(
        source_text=source_text,
        summary_xml='<SPEAKER_00 emo_preset="neutral">hello</SPEAKER_00>',
        transcript_per_speaker_messages={"SPEAKER_00": ["hello"]},
        summary_per_speaker_messages={"SPEAKER_00": ["hello"]},
        speaker_alias_map={"Alex Host": "SPEAKER_00"},
        speaker_notes=["hostSpeakerLabels: {'Alex Host': ['SPEAKER_00']}"],
    )

    assert result.status == "ok"
    assert "TAIL_MARKER" in llm.user_prompt


def test_sporc_judge_uses_shared_conversation_summary_templates() -> None:
    llm = _FakeJudgeLLM(
        '{"Factualness": 0.9, "Naturalness": 0.8, '
        '"Speaker Grammar Similarity": 0.7, "confidence": 0.6, "rationale": "ok"}'
    )
    judge = SPoRCJudge(judge_llm=llm)
    original_get_prompt = PromptManager.__dict__["get_prompt"]
    prompt_calls: list[tuple[str, dict[str, object]]] = []

    def _fake_get_prompt(cls, key: str, **kwargs: object) -> str:
        prompt_calls.append((key, kwargs))
        return f"{key} prompt"

    PromptManager.get_prompt = classmethod(_fake_get_prompt)
    try:
        result = judge.evaluate(
            source_text="[SPEAKER_00]: hello",
            summary_xml='<SPEAKER_00 emo_preset="neutral">hello</SPEAKER_00>',
            transcript_per_speaker_messages={"SPEAKER_00": ["hello"]},
            summary_per_speaker_messages={"SPEAKER_00": ["hello"]},
            speaker_alias_map={"Alex Host": "SPEAKER_00"},
            speaker_notes=["hostSpeakerLabels: {'Alex Host': ['SPEAKER_00']}"],
        )
    finally:
        PromptManager.get_prompt = original_get_prompt

    assert result.status == "ok"
    assert [call[0] for call in prompt_calls] == [
        "conversation_summary_judge_system",
        "conversation_summary_judge_user",
    ]
    assert prompt_calls[0][1]["dimension_names"] == [
        "Factualness",
        "Naturalness",
        "Speaker Grammar Similarity",
    ]
    assert prompt_calls[1][1]["speaker_alias_map_json"] == json.dumps(
        {"Alex Host": "SPEAKER_00"},
        indent=2,
        sort_keys=True,
    )
    assert prompt_calls[1][1]["speaker_notes_json"] == json.dumps(
        ["hostSpeakerLabels: {'Alex Host': ['SPEAKER_00']}"],
        indent=2,
    )
    assert llm.system_prompt == "conversation_summary_judge_system prompt"
    assert llm.user_prompt == "conversation_summary_judge_user prompt"


def test_sporc_judge_system_prompt_mentions_human_compatible_likert_flow() -> None:
    llm = _FakeJudgeLLM(
        '{"Factualness": 0.9, "Naturalness": 0.8, '
        '"Speaker Grammar Similarity": 0.7, "confidence": 0.6, "rationale": "ok"}'
    )
    judge = SPoRCJudge(judge_llm=llm)

    result = judge.evaluate(
        source_text="[SPEAKER_00]: hello",
        summary_xml='<SPEAKER_00 emo_preset="neutral">hello</SPEAKER_00>',
        transcript_per_speaker_messages={"SPEAKER_00": ["hello"]},
        summary_per_speaker_messages={"SPEAKER_00": ["hello"]},
        speaker_alias_map={"Alex Host": "SPEAKER_00"},
        speaker_notes=["hostSpeakerLabels: {'Alex Host': ['SPEAKER_00']}"],
    )

    assert result.status == "ok"
    assert "human survey participant" in llm.system_prompt
    assert "average within the dimension" in llm.system_prompt
