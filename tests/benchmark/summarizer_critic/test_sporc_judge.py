"""Judge coverage for the SPoRC benchmark."""

from __future__ import annotations

from card_framework.benchmark.summarizer_critic.sporc.judge import SPoRCJudge


class _FakeJudgeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.user_prompt = ""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        del system_prompt, max_tokens
        self.user_prompt = user_prompt
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
    assert "filler words" in llm.user_prompt
