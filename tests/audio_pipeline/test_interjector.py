import json
from pathlib import Path
import subprocess

from card_framework.agents.dtos import AssistantMessage, Function, ToolCall
import card_framework.audio_pipeline.interjector as interjector_module
from card_framework.audio_pipeline.interjector import (
    AlignedToken,
    InterjectionDecision,
    InterjectorOrchestrator,
    LLMInterjectionPlanner,
    TurnAlignmentResult,
    _align_turn_tokens,
    _build_eligible_turns,
    _build_heuristic_interjection_decisions,
    _decode_audio_waveform,
    _find_best_matching_phrase_span,
    _refine_anchor_window_alignment,
)
from card_framework.audio_pipeline.contracts import WordTimestamp
from card_framework.audio_pipeline.voice_clone_contracts import (
    VoiceCloneProvider,
    VoiceCloneTurn,
    VoiceSampleReference,
)
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


class _RepairThinkingLLMProvider(_ThinkingLLMProvider):
    """Fake provider that returns a repair-only payload for timing revisions."""

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        self.generate_calls += 1
        assert self.enable_thinking is False
        if "audio timing budget" in system_prompt:
            assert "Current interjection text: Absolutely." in user_prompt
            return '{"interjection_text":"Yeah."}'
        return super().generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
        )


class _FixedPlanner:
    """Planner stub that returns one decision and repairs its text when needed."""

    def __init__(
        self,
        *,
        style: str = "backchannel",
        text: str = "Absolutely.",
        anchor_start_token_index: int = 2,
        anchor_end_token_index: int = 2,
        anchor_text: str = "gamma",
    ) -> None:
        self.repair_calls = 0
        self.style = style
        self.text = text
        self.anchor_start_token_index = anchor_start_token_index
        self.anchor_end_token_index = anchor_end_token_index
        self.anchor_text = anchor_text

    def plan(self, summary_turns) -> list[InterjectionDecision]:
        del summary_turns
        return [
            InterjectionDecision(
                host_turn_index=1,
                should_interject=True,
                interjection_style=self.style,
                interjection_speaker="SPEAKER_01",
                interjection_text=self.text,
                anchor_start_token_index=self.anchor_start_token_index,
                anchor_end_token_index=self.anchor_end_token_index,
                anchor_text=self.anchor_text,
            )
        ]

    def repair_interjection_text(self, **kwargs) -> str:
        del kwargs
        self.repair_calls += 1
        return "Yeah."


class _DurationAwareProvider(VoiceCloneProvider):
    """Voice-clone stub that records synthesized durations by output path."""

    def __init__(self) -> None:
        self.duration_by_path: dict[Path, int] = {}
        self.calls: list[str] = []

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None = None,
        emotion_vector=None,
        progress_callback=None,
    ) -> Path:
        del reference_audio_path, emo_text, emotion_vector, progress_callback
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        output_audio_path.write_bytes(b"wav")
        self.calls.append(text)
        if text == "Absolutely.":
            duration_ms = 650
        elif text == "Mm.":
            duration_ms = 40
        else:
            duration_ms = 180
        self.duration_by_path[output_audio_path.resolve()] = duration_ms
        return output_audio_path.resolve()

    def close(self) -> None:
        return None


class _ImpossibleTimingAligner:
    """Return aligned timestamps that exceed the actual audio duration."""

    def align_words(self, **kwargs) -> list[WordTimestamp]:
        del kwargs
        return [
            WordTimestamp(word="alpha", start_time_ms=0, end_time_ms=400),
            WordTimestamp(word="beta", start_time_ms=400, end_time_ms=1600),
        ]


class _StubWordTimingBackend:
    """Return deterministic word timings keyed by absolute audio path."""

    def __init__(self, timings_by_path: dict[Path, list[WordTimestamp]]) -> None:
        self.timings_by_path = {path.resolve(): words for path, words in timings_by_path.items()}
        self.closed = False

    def align_words(self, *, audio_path: Path, language: str) -> list[WordTimestamp]:
        del language
        return list(self.timings_by_path.get(audio_path.resolve(), []))

    def close(self) -> None:
        self.closed = True


def test_prepare_interjection_attempt_audio_records_leading_silence_trim(
    monkeypatch,
    tmp_path: Path,
) -> None:
    rendered_path = tmp_path / "overlay.wav"
    rendered_path.write_bytes(b"wav")

    monkeypatch.setattr(
        interjector_module,
        "_trim_leading_silence_in_place",
        lambda **kwargs: 520,
    )

    prepared = interjector_module._prepare_interjection_attempt_audio(
        rendered_path=rendered_path,
        original_duration_ms=650,
        trim_leading_silence_enabled=True,
        trim_leading_silence_threshold_db=-50.0,
        trim_leading_silence_min_duration_ms=0,
        trim_leading_silence_preserve_ms=5,
        audio_codec="pcm_s24le",
        timeout_seconds=30,
    )

    assert prepared.path == rendered_path
    assert prepared.duration_ms == 520
    assert prepared.leading_silence_trim_ms == 130


def test_find_best_matching_phrase_span_handles_near_match() -> None:
    matched = _find_best_matching_phrase_span(
        aligned_words=[
            WordTimestamp(word="It", start_time_ms=0, end_time_ms=180),
            WordTimestamp(word="was", start_time_ms=180, end_time_ms=360),
            WordTimestamp(word="damn", start_time_ms=960, end_time_ms=1260),
            WordTimestamp(word="broke", start_time_ms=1260, end_time_ms=1660),
        ],
        target_text="dam broke",
        min_similarity=0.75,
        max_window_delta=1,
    )

    assert matched is not None
    assert matched.start_time_ms == 960
    assert matched.end_time_ms == 1660
    assert matched.similarity >= 0.75


def test_find_best_matching_phrase_span_prefers_anchor_neighborhood() -> None:
    matched = _find_best_matching_phrase_span(
        aligned_words=[
            WordTimestamp(word="damn", start_time_ms=200, end_time_ms=500),
            WordTimestamp(word="broke", start_time_ms=500, end_time_ms=900),
            WordTimestamp(word="then", start_time_ms=900, end_time_ms=1200),
            WordTimestamp(word="damn", start_time_ms=1800, end_time_ms=2100),
            WordTimestamp(word="broke", start_time_ms=2100, end_time_ms=2500),
        ],
        target_text="dam broke",
        min_similarity=0.75,
        max_window_delta=1,
        preferred_center_ms=2200,
    )

    assert matched is not None
    assert matched.start_time_ms == 1800
    assert matched.end_time_ms == 2500


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


def test_interjection_planner_repairs_text_with_json_payload() -> None:
    provider = _RepairThinkingLLMProvider()
    planner = LLMInterjectionPlanner(llm=LoggingLLMProvider(inner_provider=provider))

    repaired = planner.repair_interjection_text(
        host_turn_text="alpha beta gamma delta",
        next_turn_text="yeah that tracks",
        interjection_style="backchannel",
        anchor_text="gamma",
        current_text="Absolutely.",
        available_overlap_ms=220,
        measured_duration_ms=640,
        attempt_index=1,
    )

    assert repaired == "Yeah."
    assert provider.generate_calls == 1
    assert provider.enable_thinking is True


def test_interjector_repairs_long_candidate_until_it_fits(monkeypatch, tmp_path: Path) -> None:
    summary_xml = (
        '<SPEAKER_00 emo_preset="neutral">alpha beta gamma delta</SPEAKER_00>\n'
        '<SPEAKER_01 emo_preset="neutral">yeah that tracks for me</SPEAKER_01>'
    )
    host_turn_audio = tmp_path / "turn_001.wav"
    next_turn_audio = tmp_path / "turn_002.wav"
    merged_audio = tmp_path / "voice_cloned.wav"
    speaker_manifest = tmp_path / "speaker_manifest.json"
    reference_audio = tmp_path / "speaker_01.wav"
    voice_manifest = tmp_path / "voice_manifest.json"
    for path in (host_turn_audio, next_turn_audio, merged_audio, reference_audio):
        path.write_bytes(b"wav")
    speaker_manifest.write_text("{}", encoding="utf-8")
    voice_manifest.write_text(
        json.dumps(
            {
                "speaker_samples_manifest_path": str(speaker_manifest),
                "merged_output_audio_path": str(merged_audio),
                "artifacts": [
                    {
                        "turn_index": 1,
                        "output_audio_path": str(host_turn_audio),
                    },
                    {
                        "turn_index": 2,
                        "output_audio_path": str(next_turn_audio),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    provider = _DurationAwareProvider()
    planner = _FixedPlanner()
    orchestrator = InterjectorOrchestrator(
        planner=planner,
        provider=provider,
        output_dir=tmp_path / "interjector",
        merged_output_normalization=None,
        aligner=None,
        repair_enabled=True,
        repair_max_attempts=4,
    )

    duration_by_path = {
        host_turn_audio.resolve(): 2000,
        next_turn_audio.resolve(): 1500,
        merged_audio.resolve(): 3500,
        reference_audio.resolve(): 3000,
    }

    def _fake_probe(audio_path: Path) -> int | None:
        resolved = audio_path.resolve()
        if resolved in provider.duration_by_path:
            return provider.duration_by_path[resolved]
        return duration_by_path.get(resolved)

    monkeypatch.setattr(interjector_module, "probe_audio_duration_ms", _fake_probe)
    monkeypatch.setattr(
        interjector_module,
        "load_speaker_sample_references",
        lambda _path: {
            "SPEAKER_01": VoiceSampleReference(
                speaker="SPEAKER_01",
                path=reference_audio.resolve(),
                duration_ms=3000,
            )
        },
    )
    monkeypatch.setattr(
        interjector_module,
        "_align_turn_tokens",
        lambda **kwargs: TurnAlignmentResult(
            tokens=(
                AlignedToken(token="alpha", start_time_ms=0, end_time_ms=450),
                AlignedToken(token="beta", start_time_ms=450, end_time_ms=950),
                AlignedToken(token="gamma", start_time_ms=950, end_time_ms=1450),
                AlignedToken(token="delta", start_time_ms=1450, end_time_ms=2000),
            ),
            source="uniform_fallback",
        ),
    )
    monkeypatch.setattr(
        interjector_module,
        "_refine_anchor_window_alignment",
        lambda **kwargs: (
            AlignedToken(token="gamma", start_time_ms=1100, end_time_ms=1400),
        ),
    )

    def _fake_mix_audio_with_overlays(
        *,
        base_audio_path: Path,
        overlays,
        output_path: Path,
        audio_codec: str,
        timeout_seconds: int,
    ) -> None:
        del base_audio_path, overlays, audio_codec, timeout_seconds
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"mixed")

    monkeypatch.setattr(
        interjector_module,
        "_mix_audio_with_overlays",
        _fake_mix_audio_with_overlays,
    )

    result = orchestrator.run(
        summary_xml=summary_xml,
        voice_clone_manifest_path=voice_manifest,
    )

    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.text == "Yeah."
    assert artifact.timing_source == "anchor_window_refinement"
    assert artifact.repair_attempt_count == 1
    assert artifact.end_time_ms <= artifact.legal_overlap_end_time_ms
    assert artifact.next_turn_start_time_ms == 2000
    assert provider.calls == ["Absolutely.", "Yeah."]
    assert planner.repair_calls == 1


def test_interjector_uses_trimmed_overlay_duration_for_window_fit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary_xml = (
        '<SPEAKER_00 emo_preset="neutral">alpha beta gamma delta</SPEAKER_00>\n'
        '<SPEAKER_01 emo_preset="neutral">yeah that tracks for me</SPEAKER_01>'
    )
    host_turn_audio = tmp_path / "turn_001.wav"
    next_turn_audio = tmp_path / "turn_002.wav"
    merged_audio = tmp_path / "voice_cloned.wav"
    speaker_manifest = tmp_path / "speaker_manifest.json"
    reference_audio = tmp_path / "speaker_01.wav"
    voice_manifest = tmp_path / "voice_manifest.json"
    for path in (host_turn_audio, next_turn_audio, merged_audio, reference_audio):
        path.write_bytes(b"wav")
    speaker_manifest.write_text("{}", encoding="utf-8")
    voice_manifest.write_text(
        json.dumps(
            {
                "speaker_samples_manifest_path": str(speaker_manifest),
                "merged_output_audio_path": str(merged_audio),
                "artifacts": [
                    {"turn_index": 1, "output_audio_path": str(host_turn_audio)},
                    {"turn_index": 2, "output_audio_path": str(next_turn_audio)},
                ],
            }
        ),
        encoding="utf-8",
    )

    provider = _DurationAwareProvider()
    planner = _FixedPlanner(
        style="backchannel",
        text="Absolutely.",
        anchor_start_token_index=2,
        anchor_end_token_index=2,
        anchor_text="gamma",
    )
    orchestrator = InterjectorOrchestrator(
        planner=planner,
        provider=provider,
        output_dir=tmp_path / "interjector",
        merged_output_normalization=None,
        aligner=None,
        repair_enabled=True,
        repair_max_attempts=4,
    )

    duration_by_path = {
        host_turn_audio.resolve(): 2000,
        next_turn_audio.resolve(): 1500,
        merged_audio.resolve(): 3500,
        reference_audio.resolve(): 3000,
    }

    def _fake_probe(audio_path: Path) -> int | None:
        resolved = audio_path.resolve()
        if resolved in provider.duration_by_path:
            return provider.duration_by_path[resolved]
        return duration_by_path.get(resolved)

    monkeypatch.setattr(interjector_module, "probe_audio_duration_ms", _fake_probe)
    monkeypatch.setattr(
        interjector_module,
        "_trim_leading_silence_in_place",
        lambda **kwargs: 520,
    )
    monkeypatch.setattr(
        interjector_module,
        "load_speaker_sample_references",
        lambda _path: {
            "SPEAKER_01": VoiceSampleReference(
                speaker="SPEAKER_01",
                path=reference_audio.resolve(),
                duration_ms=3000,
            )
        },
    )
    monkeypatch.setattr(
        interjector_module,
        "_align_turn_tokens",
        lambda **kwargs: TurnAlignmentResult(
            tokens=(
                AlignedToken(token="alpha", start_time_ms=0, end_time_ms=400),
                AlignedToken(token="beta", start_time_ms=400, end_time_ms=800),
                AlignedToken(token="gamma", start_time_ms=800, end_time_ms=1100),
                AlignedToken(token="delta", start_time_ms=1100, end_time_ms=2000),
            ),
            source="forced_alignment",
        ),
    )
    monkeypatch.setattr(
        interjector_module,
        "_mix_audio_with_overlays",
        lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"mixed"),
    )

    result = orchestrator.run(
        summary_xml=summary_xml,
        voice_clone_manifest_path=voice_manifest,
    )

    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.text == "Absolutely."
    assert artifact.duration_ms == 520
    assert artifact.leading_silence_trim_ms == 130
    assert artifact.repair_attempt_count == 0
    assert artifact.end_time_ms == artifact.start_time_ms + 520
    assert provider.calls == ["Absolutely."]


def test_interjector_uses_echo_phrase_match_and_pacing_fit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary_xml = (
        '<SPEAKER_00 emo_preset="neutral">it was like a dam broke all at once</SPEAKER_00>\n'
        '<SPEAKER_01 emo_preset="neutral">yeah that tracks for me</SPEAKER_01>'
    )
    host_turn_audio = tmp_path / "turn_001.wav"
    next_turn_audio = tmp_path / "turn_002.wav"
    merged_audio = tmp_path / "voice_cloned.wav"
    speaker_manifest = tmp_path / "speaker_manifest.json"
    reference_audio = tmp_path / "speaker_01.wav"
    voice_manifest = tmp_path / "voice_manifest.json"
    for path in (host_turn_audio, next_turn_audio, merged_audio, reference_audio):
        path.write_bytes(b"wav")
    speaker_manifest.write_text("{}", encoding="utf-8")
    voice_manifest.write_text(
        json.dumps(
            {
                "speaker_samples_manifest_path": str(speaker_manifest),
                "merged_output_audio_path": str(merged_audio),
                "artifacts": [
                    {"turn_index": 1, "output_audio_path": str(host_turn_audio)},
                    {"turn_index": 2, "output_audio_path": str(next_turn_audio)},
                ],
            }
        ),
        encoding="utf-8",
    )

    class _EchoDurationProvider(VoiceCloneProvider):
        def __init__(self) -> None:
            self.duration_by_path: dict[Path, int] = {}
            self.calls: list[str] = []

        def synthesize(
            self,
            *,
            reference_audio_path: Path,
            text: str,
            output_audio_path: Path,
            emo_text: str | None = None,
            emotion_vector=None,
            progress_callback=None,
        ) -> Path:
            del reference_audio_path, emo_text, emotion_vector, progress_callback
            output_audio_path.parent.mkdir(parents=True, exist_ok=True)
            output_audio_path.write_bytes(b"wav")
            self.calls.append(text)
            self.duration_by_path[output_audio_path.resolve()] = 1400
            return output_audio_path.resolve()

        def close(self) -> None:
            return None

    provider = _EchoDurationProvider()
    planner = _FixedPlanner(
        style="echo_agreement",
        text="dam broke",
        anchor_start_token_index=5,
        anchor_end_token_index=6,
        anchor_text="dam broke",
    )
    orchestrator = InterjectorOrchestrator(
        planner=planner,
        provider=provider,
        output_dir=tmp_path / "interjector",
        merged_output_normalization=None,
        aligner=None,
        repair_enabled=False,
        echo_alignment_offset_ms=0,
    )
    attempt_output = (
        tmp_path
        / "interjector"
        / "001_echo_agreement_SPEAKER_01.attempt_001.wav"
    ).resolve()
    backend = _StubWordTimingBackend(
        {
            host_turn_audio: [
                WordTimestamp(word="it", start_time_ms=0, end_time_ms=180),
                WordTimestamp(word="was", start_time_ms=180, end_time_ms=360),
                WordTimestamp(word="like", start_time_ms=360, end_time_ms=560),
                WordTimestamp(word="a", start_time_ms=560, end_time_ms=700),
                WordTimestamp(word="damn", start_time_ms=1260, end_time_ms=1560),
                WordTimestamp(word="broke", start_time_ms=1560, end_time_ms=1960),
                WordTimestamp(word="all", start_time_ms=1960, end_time_ms=2200),
            ],
            attempt_output: [
                WordTimestamp(word="Damn", start_time_ms=0, end_time_ms=520),
                WordTimestamp(word="broke", start_time_ms=760, end_time_ms=1160),
            ],
        }
    )
    orchestrator._echo_word_timing_backend = backend

    duration_by_path = {
        host_turn_audio.resolve(): 3000,
        next_turn_audio.resolve(): 1500,
        merged_audio.resolve(): 4500,
        reference_audio.resolve(): 3000,
    }

    def _fake_probe(audio_path: Path) -> int | None:
        resolved = audio_path.resolve()
        if resolved in provider.duration_by_path:
            return provider.duration_by_path[resolved]
        return duration_by_path.get(resolved)

    captured: dict[str, float] = {}

    monkeypatch.setattr(interjector_module, "probe_audio_duration_ms", _fake_probe)
    monkeypatch.setattr(
        interjector_module,
        "load_speaker_sample_references",
        lambda _path: {
            "SPEAKER_01": VoiceSampleReference(
                speaker="SPEAKER_01",
                path=reference_audio.resolve(),
                duration_ms=3000,
            )
        },
    )
    monkeypatch.setattr(
        interjector_module,
        "_align_turn_tokens",
        lambda **kwargs: TurnAlignmentResult(
            tokens=(
                AlignedToken(token="it", start_time_ms=0, end_time_ms=500),
                AlignedToken(token="was", start_time_ms=500, end_time_ms=1000),
                AlignedToken(token="like", start_time_ms=1000, end_time_ms=1500),
                AlignedToken(token="a", start_time_ms=1500, end_time_ms=2000),
                AlignedToken(token="dam", start_time_ms=2000, end_time_ms=2400),
                AlignedToken(token="broke", start_time_ms=2400, end_time_ms=2800),
                AlignedToken(token="all", start_time_ms=2800, end_time_ms=3000),
            ),
            source="uniform_fallback",
        ),
    )
    monkeypatch.setattr(
        interjector_module,
        "_mix_audio_with_overlays",
        lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"mixed"),
    )
    monkeypatch.setattr(
        interjector_module,
        "_apply_audio_tempo_in_place",
        lambda **kwargs: (
            captured.update({"tempo_ratio": kwargs["tempo_ratio"]}) or 820
        ),
    )

    result = orchestrator.run(
        summary_xml=summary_xml,
        voice_clone_manifest_path=voice_manifest,
    )

    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.timing_source == "echo_phrase_match"
    assert artifact.start_time_ms == 1260
    assert artifact.tempo_adjustment_ratio > 1.70
    assert artifact.duration_ms == 820
    assert captured["tempo_ratio"] == artifact.tempo_adjustment_ratio
    assert backend.closed is True
    assert provider.calls == ["dam broke"]


def test_interjector_repairs_even_when_overlap_window_is_below_preferred_minimum(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary_xml = (
        '<SPEAKER_00 emo_preset="neutral">alpha beta gamma delta</SPEAKER_00>\n'
        '<SPEAKER_01 emo_preset="neutral">yeah that tracks for me</SPEAKER_01>'
    )
    host_turn_audio = tmp_path / "turn_001.wav"
    next_turn_audio = tmp_path / "turn_002.wav"
    merged_audio = tmp_path / "voice_cloned.wav"
    speaker_manifest = tmp_path / "speaker_manifest.json"
    reference_audio = tmp_path / "speaker_01.wav"
    voice_manifest = tmp_path / "voice_manifest.json"
    for path in (host_turn_audio, next_turn_audio, merged_audio, reference_audio):
        path.write_bytes(b"wav")
    speaker_manifest.write_text("{}", encoding="utf-8")
    voice_manifest.write_text(
        json.dumps(
            {
                "speaker_samples_manifest_path": str(speaker_manifest),
                "merged_output_audio_path": str(merged_audio),
                "artifacts": [
                    {"turn_index": 1, "output_audio_path": str(host_turn_audio)},
                    {"turn_index": 2, "output_audio_path": str(next_turn_audio)},
                ],
            }
        ),
        encoding="utf-8",
    )

    provider = _DurationAwareProvider()
    planner = _FixedPlanner(
        style="echo_agreement",
        text="Absolutely.",
        anchor_start_token_index=3,
        anchor_end_token_index=3,
        anchor_text="delta",
    )
    orchestrator = InterjectorOrchestrator(
        planner=planner,
        provider=provider,
        output_dir=tmp_path / "interjector",
        merged_output_normalization=None,
        aligner=None,
        repair_enabled=True,
        repair_max_attempts=4,
    )

    duration_by_path = {
        host_turn_audio.resolve(): 2000,
        next_turn_audio.resolve(): 1500,
        merged_audio.resolve(): 3500,
        reference_audio.resolve(): 3000,
    }

    def _fake_probe(audio_path: Path) -> int | None:
        resolved = audio_path.resolve()
        if resolved in provider.duration_by_path:
            return provider.duration_by_path[resolved]
        return duration_by_path.get(resolved)

    monkeypatch.setattr(interjector_module, "probe_audio_duration_ms", _fake_probe)
    monkeypatch.setattr(
        interjector_module,
        "load_speaker_sample_references",
        lambda _path: {
            "SPEAKER_01": VoiceSampleReference(
                speaker="SPEAKER_01",
                path=reference_audio.resolve(),
                duration_ms=3000,
            )
        },
    )
    monkeypatch.setattr(
        interjector_module,
        "_align_turn_tokens",
        lambda **kwargs: TurnAlignmentResult(
            tokens=(
                AlignedToken(token="alpha", start_time_ms=0, end_time_ms=450),
                AlignedToken(token="beta", start_time_ms=450, end_time_ms=950),
                AlignedToken(token="gamma", start_time_ms=950, end_time_ms=1450),
                AlignedToken(token="delta", start_time_ms=1450, end_time_ms=2000),
            ),
            source="uniform_fallback",
        ),
    )
    monkeypatch.setattr(
        interjector_module,
        "_refine_anchor_window_alignment",
        lambda **kwargs: (
            AlignedToken(token="delta", start_time_ms=1750, end_time_ms=1800),
        ),
    )
    monkeypatch.setattr(
        interjector_module,
        "_mix_audio_with_overlays",
        lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"mixed"),
    )

    result = orchestrator.run(
        summary_xml=summary_xml,
        voice_clone_manifest_path=voice_manifest,
    )

    assert len(result.artifacts) == 1
    artifact = result.artifacts[0]
    assert artifact.text == "Mm."
    assert artifact.end_time_ms <= artifact.legal_overlap_end_time_ms


def test_interjector_records_rejection_when_no_repair_state_can_fit(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary_xml = (
        '<SPEAKER_00 emo_preset="neutral">alpha beta gamma delta</SPEAKER_00>\n'
        '<SPEAKER_01 emo_preset="neutral">yeah that tracks for me</SPEAKER_01>'
    )
    host_turn_audio = tmp_path / "turn_001.wav"
    next_turn_audio = tmp_path / "turn_002.wav"
    merged_audio = tmp_path / "voice_cloned.wav"
    speaker_manifest = tmp_path / "speaker_manifest.json"
    reference_audio = tmp_path / "speaker_01.wav"
    voice_manifest = tmp_path / "voice_manifest.json"
    for path in (host_turn_audio, next_turn_audio, merged_audio, reference_audio):
        path.write_bytes(b"wav")
    speaker_manifest.write_text("{}", encoding="utf-8")
    voice_manifest.write_text(
        json.dumps(
            {
                "speaker_samples_manifest_path": str(speaker_manifest),
                "merged_output_audio_path": str(merged_audio),
                "artifacts": [
                    {"turn_index": 1, "output_audio_path": str(host_turn_audio)},
                    {"turn_index": 2, "output_audio_path": str(next_turn_audio)},
                ],
            }
        ),
        encoding="utf-8",
    )

    provider = _DurationAwareProvider()
    planner = _FixedPlanner(
        style="echo_agreement",
        text="Absolutely.",
        anchor_start_token_index=3,
        anchor_end_token_index=3,
        anchor_text="delta",
    )
    orchestrator = InterjectorOrchestrator(
        planner=planner,
        provider=provider,
        output_dir=tmp_path / "interjector",
        merged_output_normalization=None,
        aligner=None,
        repair_enabled=True,
        repair_max_attempts=None,
    )

    duration_by_path = {
        host_turn_audio.resolve(): 2000,
        next_turn_audio.resolve(): 1500,
        merged_audio.resolve(): 3500,
        reference_audio.resolve(): 3000,
    }

    def _fake_probe(audio_path: Path) -> int | None:
        resolved = audio_path.resolve()
        if resolved in provider.duration_by_path:
            return provider.duration_by_path[resolved]
        return duration_by_path.get(resolved)

    monkeypatch.setattr(interjector_module, "probe_audio_duration_ms", _fake_probe)
    monkeypatch.setattr(
        interjector_module,
        "load_speaker_sample_references",
        lambda _path: {
            "SPEAKER_01": VoiceSampleReference(
                speaker="SPEAKER_01",
                path=reference_audio.resolve(),
                duration_ms=3000,
            )
        },
    )
    monkeypatch.setattr(
        interjector_module,
        "_align_turn_tokens",
        lambda **kwargs: TurnAlignmentResult(
            tokens=(
                AlignedToken(token="alpha", start_time_ms=0, end_time_ms=450),
                AlignedToken(token="beta", start_time_ms=450, end_time_ms=950),
                AlignedToken(token="gamma", start_time_ms=950, end_time_ms=1450),
                AlignedToken(token="delta", start_time_ms=1450, end_time_ms=2000),
            ),
            source="uniform_fallback",
        ),
    )
    monkeypatch.setattr(
        interjector_module,
        "_refine_anchor_window_alignment",
        lambda **kwargs: (
            AlignedToken(token="delta", start_time_ms=1790, end_time_ms=1810),
        ),
    )
    monkeypatch.setattr(
        interjector_module,
        "_mix_audio_with_overlays",
        lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"mixed"),
    )

    result = orchestrator.run(
        summary_xml=summary_xml,
        voice_clone_manifest_path=voice_manifest,
    )

    assert result.artifacts == ()
    assert len(result.rejected_artifacts) >= 1
    assert result.rejected_artifacts[0].failure_reason == "repair_state_exhausted"


def test_interjector_honors_repair_cap_when_no_legal_window_exists(
    monkeypatch,
    tmp_path: Path,
) -> None:
    summary_xml = (
        '<SPEAKER_00 emo_preset="neutral">alpha beta gamma delta</SPEAKER_00>\n'
        '<SPEAKER_01 emo_preset="neutral">yeah that tracks for me</SPEAKER_01>'
    )
    host_turn_audio = tmp_path / "turn_001.wav"
    next_turn_audio = tmp_path / "turn_002.wav"
    merged_audio = tmp_path / "voice_cloned.wav"
    speaker_manifest = tmp_path / "speaker_manifest.json"
    reference_audio = tmp_path / "speaker_01.wav"
    voice_manifest = tmp_path / "voice_manifest.json"
    for path in (host_turn_audio, next_turn_audio, merged_audio, reference_audio):
        path.write_bytes(b"wav")
    speaker_manifest.write_text("{}", encoding="utf-8")
    voice_manifest.write_text(
        json.dumps(
            {
                "speaker_samples_manifest_path": str(speaker_manifest),
                "merged_output_audio_path": str(merged_audio),
                "artifacts": [
                    {"turn_index": 1, "output_audio_path": str(host_turn_audio)},
                    {"turn_index": 2, "output_audio_path": str(next_turn_audio)},
                ],
            }
        ),
        encoding="utf-8",
    )

    provider = _DurationAwareProvider()
    planner = _FixedPlanner(
        style="backchannel",
        text="Absolutely.",
        anchor_start_token_index=3,
        anchor_end_token_index=3,
        anchor_text="delta",
    )
    orchestrator = InterjectorOrchestrator(
        planner=planner,
        provider=provider,
        output_dir=tmp_path / "interjector",
        merged_output_normalization=None,
        aligner=None,
        repair_enabled=True,
        repair_max_attempts=2,
    )

    duration_by_path = {
        host_turn_audio.resolve(): 2000,
        next_turn_audio.resolve(): 1500,
        merged_audio.resolve(): 3500,
        reference_audio.resolve(): 3000,
    }

    def _fake_probe(audio_path: Path) -> int | None:
        resolved = audio_path.resolve()
        if resolved in provider.duration_by_path:
            return provider.duration_by_path[resolved]
        return duration_by_path.get(resolved)

    monkeypatch.setattr(interjector_module, "probe_audio_duration_ms", _fake_probe)
    monkeypatch.setattr(
        interjector_module,
        "load_speaker_sample_references",
        lambda _path: {
            "SPEAKER_01": VoiceSampleReference(
                speaker="SPEAKER_01",
                path=reference_audio.resolve(),
                duration_ms=3000,
            )
        },
    )
    monkeypatch.setattr(
        interjector_module,
        "_align_turn_tokens",
        lambda **kwargs: TurnAlignmentResult(
            tokens=(
                AlignedToken(token="alpha", start_time_ms=0, end_time_ms=450),
                AlignedToken(token="beta", start_time_ms=450, end_time_ms=950),
                AlignedToken(token="gamma", start_time_ms=950, end_time_ms=1450),
                AlignedToken(token="delta", start_time_ms=1450, end_time_ms=2000),
            ),
            source="uniform_fallback",
        ),
    )
    monkeypatch.setattr(
        interjector_module,
        "_refine_anchor_window_alignment",
        lambda **kwargs: (
            AlignedToken(token="delta", start_time_ms=1790, end_time_ms=1810),
        ),
    )
    monkeypatch.setattr(
        interjector_module,
        "_mix_audio_with_overlays",
        lambda **kwargs: Path(kwargs["output_path"]).write_bytes(b"mixed"),
    )

    result = orchestrator.run(
        summary_xml=summary_xml,
        voice_clone_manifest_path=voice_manifest,
    )

    assert result.artifacts == ()
    assert len(result.rejected_artifacts) >= 1
    assert result.rejected_artifacts[0].failure_reason == "repair_attempt_limit_reached"


def test_decode_audio_waveform_resamples_to_forced_alignment_rate(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    audio_path = tmp_path / "turn.wav"
    audio_path.write_bytes(b"wav")

    def _fake_subprocess_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout=b"\x00\x00\x00\x00",
            stderr=b"",
        )

    monkeypatch.setattr(interjector_module, "ensure_command_available", lambda _cmd: None)
    monkeypatch.setattr(interjector_module.subprocess, "run", _fake_subprocess_run)

    waveform = _decode_audio_waveform(audio_path)

    assert waveform.size == 1
    assert captured["command"] == [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(interjector_module._FORCED_ALIGNMENT_SAMPLE_RATE_HZ),
        "-",
    ]


def test_align_turn_tokens_falls_back_when_aligned_timestamps_exceed_audio_duration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "turn.wav"
    audio_path.write_bytes(b"wav")
    monkeypatch.setattr(interjector_module, "probe_audio_duration_ms", lambda _path: 1000)
    monkeypatch.setattr(interjector_module, "_decode_audio_waveform", lambda _path: object())

    result = _align_turn_tokens(
        audio_path=audio_path,
        text="alpha beta",
        language="en",
        aligner=_ImpossibleTimingAligner(),
        batch_size=4,
    )

    assert result.source == "uniform_fallback"
    assert result.tokens[-1].end_time_ms == 1000


def test_refine_anchor_window_alignment_rejects_impossible_timestamps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "turn.wav"
    audio_path.write_bytes(b"wav")
    monkeypatch.setattr(interjector_module, "probe_audio_duration_ms", lambda _path: 1000)
    monkeypatch.setattr(interjector_module, "_decode_audio_waveform", lambda _path: object())

    result = _refine_anchor_window_alignment(
        audio_path=audio_path,
        full_text="alpha beta gamma delta",
        anchor_start_token_index=1,
        anchor_end_token_index=2,
        language="en",
        aligner=_ImpossibleTimingAligner(),
        batch_size=4,
    )

    assert result == ()
