from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

import audio_pipeline.interjector as interjector_module
from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.interjector import (
    AlignedToken,
    InterjectionDecision,
    InterjectorOrchestrator,
    _parse_plan_payload,
    _SynthesizedOverlay,
    _align_turn_tokens,
    _build_eligible_turns,
    _load_voice_clone_manifest_bundle,
    _render_eligible_turns_block,
    _validate_llm_decisions,
)
from audio_pipeline.voice_clone_contracts import VoiceCloneTurn


class _StaticPlanner:
    """Return a fixed set of planner decisions for test orchestration runs."""

    def __init__(self, decisions: list[InterjectionDecision]) -> None:
        self.decisions = decisions

    def plan(self, summary_turns: list[VoiceCloneTurn]) -> list[InterjectionDecision]:
        del summary_turns
        return list(self.decisions)


class _RecordingProvider:
    """Capture synthesis calls and emit dummy WAV files."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Path | str]] = []

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        progress_callback=None,
    ) -> Path:
        del progress_callback
        self.calls.append(
            {
                "reference_audio_path": reference_audio_path,
                "text": text,
                "output_audio_path": output_audio_path,
            }
        )
        output_audio_path.parent.mkdir(parents=True, exist_ok=True)
        output_audio_path.write_bytes(b"overlay")
        return output_audio_path


def _write_stage_three_fixture(
    tmp_path: Path,
    *,
    summary_turns: list[VoiceCloneTurn],
) -> tuple[Path, Path, Path, dict[str, int]]:
    """Write minimal stage-3 artifacts and duration mappings for tests."""
    work_dir = tmp_path / "voice_clone"
    work_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = tmp_path / "speaker_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    durations: dict[str, int] = {}
    artifacts: list[dict[str, object]] = []
    for index, turn in enumerate(summary_turns, start=1):
        artifact_path = work_dir / f"{index:03d}.wav"
        artifact_path.write_bytes(f"turn-{index}".encode("utf-8"))
        durations[str(artifact_path.resolve())] = 1600 if index == 1 else 700
        artifacts.append(
            {
                "turn_index": index,
                "speaker": turn.speaker,
                "text": turn.text,
                "reference_audio_path": str(sample_dir / f"{turn.speaker}.wav"),
                "output_audio_path": str(artifact_path),
            }
        )

    merged_output_path = work_dir / "voice_cloned.wav"
    merged_output_path.write_bytes(b"base-audio")
    durations[str(merged_output_path.resolve())] = 2300

    samples: list[dict[str, object]] = []
    for speaker in {turn.speaker for turn in summary_turns}:
        sample_path = sample_dir / f"{speaker}.wav"
        sample_path.write_bytes(f"sample-{speaker}".encode("utf-8"))
        samples.append(
            {
                "speaker": speaker,
                "path": str(sample_path),
                "duration_ms": 3000,
            }
        )
    speaker_manifest_path = sample_dir / "manifest.json"
    speaker_manifest_path.write_text(
        json.dumps({"samples": samples}, indent=2),
        encoding="utf-8",
    )

    voice_clone_manifest_path = work_dir / "manifest.json"
    voice_clone_manifest_path.write_text(
        json.dumps(
            {
                "speaker_samples_manifest_path": str(speaker_manifest_path),
                "merged_output_audio_path": str(merged_output_path),
                "artifacts": artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return voice_clone_manifest_path, speaker_manifest_path, merged_output_path, durations


def test_validate_llm_decisions_rejects_wrong_next_speaker() -> None:
    summary_turns = [
        VoiceCloneTurn(speaker="A", text="fees matter a lot"),
        VoiceCloneTurn(speaker="B", text="right"),
    ]
    eligible_turns = _build_eligible_turns(summary_turns)

    decisions = _validate_llm_decisions(
        [
            InterjectionDecision(
                host_turn_index=1,
                should_interject=True,
                interjection_style="backchannel",
                interjection_speaker="A",
                interjection_text="right",
                anchor_start_token_index=1,
                anchor_end_token_index=2,
            )
        ],
        eligible_turns=eligible_turns,
        max_interjection_words=5,
    )

    assert decisions == [
        InterjectionDecision(host_turn_index=1, should_interject=False)
    ]


def test_validate_llm_decisions_coerces_echo_to_anchor_text() -> None:
    summary_turns = [
        VoiceCloneTurn(speaker="A", text="fees matter a lot"),
        VoiceCloneTurn(speaker="B", text="exactly"),
    ]
    eligible_turns = _build_eligible_turns(summary_turns)

    decisions = _validate_llm_decisions(
        [
            InterjectionDecision(
                host_turn_index=1,
                should_interject=True,
                interjection_style="echo_agreement",
                interjection_speaker="B",
                interjection_text="absolutely",
                anchor_start_token_index=1,
                anchor_end_token_index=3,
            )
        ],
        eligible_turns=eligible_turns,
        max_interjection_words=5,
    )

    assert decisions[0].should_interject is True
    assert decisions[0].interjection_text == "matter a lot"
    assert decisions[0].anchor_text == "matter a lot"


def test_validate_llm_decisions_rejects_anchor_before_progress_window() -> None:
    summary_turns = [
        VoiceCloneTurn(
            speaker="A",
            text="one two three four five six seven eight nine ten",
        ),
        VoiceCloneTurn(speaker="B", text="right"),
    ]
    eligible_turns = _build_eligible_turns(summary_turns)

    decisions = _validate_llm_decisions(
        [
            InterjectionDecision(
                host_turn_index=1,
                should_interject=True,
                interjection_style="backchannel",
                interjection_speaker="B",
                interjection_text="right",
                anchor_start_token_index=0,
                anchor_end_token_index=1,
            )
        ],
        eligible_turns=eligible_turns,
        max_interjection_words=5,
        min_host_progress_ratio=0.35,
        max_host_progress_ratio=0.90,
    )

    assert decisions == [
        InterjectionDecision(host_turn_index=1, should_interject=False)
    ]


def test_render_eligible_turns_block_includes_preferred_anchor_window() -> None:
    summary_turns = [
        VoiceCloneTurn(
            speaker="A",
            text="one two three four five six seven eight nine ten",
        ),
        VoiceCloneTurn(speaker="B", text="right"),
    ]
    eligible_turns = _build_eligible_turns(summary_turns)

    block = _render_eligible_turns_block(
        eligible_turns,
        min_host_progress_ratio=0.35,
        max_host_progress_ratio=0.90,
    )

    assert "Preferred anchor token window" in block
    assert "3..8" in block
    assert "anchor_end_token_index" in block
    assert "anchor_start_token_index" in block


def test_parse_plan_payload_recovers_complete_decisions_from_truncated_json() -> None:
    raw_response = """{
  "decisions": [
    {
      "host_turn_index": 2,
      "should_interject": true,
      "interjection_style": "backchannel",
      "interjection_speaker": "SPEAKER_02",
      "interjection_text": "Right.",
      "anchor_start_token_index": 13,
      "anchor_end_token_index": 14,
      "anchor_text": "systematic errors"
    },
    {
      "host_turn_index": 4,
      "should_interject": true,
      "interjection_style": "echo_agreement",
      "interjection_speaker": "SPEAKER_00",
      "interjection_text": "Exactly.",
      "anchor_start_token_index": 19,
"""

    payload = _parse_plan_payload(raw_response)

    assert payload is not None
    assert len(payload.decisions) == 1
    assert payload.decisions[0].host_turn_index == 2
    assert payload.decisions[0].interjection_text == "Right."


def test_validate_llm_decisions_defaults_missing_turns_to_false() -> None:
    summary_turns = [
        VoiceCloneTurn(speaker="A", text="fees matter a lot"),
        VoiceCloneTurn(speaker="B", text="right"),
        VoiceCloneTurn(speaker="A", text="costs compound over time"),
        VoiceCloneTurn(speaker="B", text="exactly"),
    ]
    eligible_turns = _build_eligible_turns(summary_turns)

    decisions = _validate_llm_decisions(
        [
            InterjectionDecision(
                host_turn_index=3,
                should_interject=True,
                interjection_style="echo_agreement",
                interjection_speaker="B",
                interjection_text="compound over time",
                anchor_start_token_index=1,
                anchor_end_token_index=3,
            )
        ],
        eligible_turns=eligible_turns,
        max_interjection_words=5,
        min_host_progress_ratio=0.25,
        max_host_progress_ratio=0.90,
    )

    assert decisions == [
        InterjectionDecision(host_turn_index=1, should_interject=False),
        InterjectionDecision(host_turn_index=2, should_interject=False),
        InterjectionDecision(
            host_turn_index=3,
            should_interject=True,
            interjection_style="echo_agreement",
            interjection_speaker="B",
            interjection_text="compound over time",
            anchor_start_token_index=1,
            anchor_end_token_index=3,
            anchor_text="compound over time",
        ),
    ]


def test_align_turn_tokens_falls_back_without_aligner(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "turn.wav"
    audio_path.write_bytes(b"wav")
    monkeypatch.setattr(interjector_module, "probe_audio_duration_ms", lambda path: 400)

    aligned_tokens = _align_turn_tokens(
        audio_path=audio_path,
        text="fees matter a lot",
        language="en",
        aligner=None,
        batch_size=8,
    )

    assert [token.token for token in aligned_tokens] == ["fees", "matter", "a", "lot"]
    assert aligned_tokens[0].start_time_ms == 0
    assert aligned_tokens[-1].end_time_ms == 400


def test_load_voice_clone_manifest_bundle_rejects_turn_count_mismatch(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    speaker_manifest_path = tmp_path / "speaker_samples.json"
    speaker_manifest_path.write_text('{"samples": []}', encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "speaker_samples_manifest_path": str(speaker_manifest_path),
                "artifacts": [{"output_audio_path": "001.wav"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        NonRetryableAudioStageError,
        match="artifact count does not match summary turn count",
    ):
        _load_voice_clone_manifest_bundle(
            summary_turns=[
                VoiceCloneTurn(speaker="A", text="one"),
                VoiceCloneTurn(speaker="B", text="two"),
            ],
            manifest_path=manifest_path,
        )


def test_mix_audio_with_overlays_builds_expected_ffmpeg_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    base_audio_path = tmp_path / "base.wav"
    overlay_path = tmp_path / "overlay.wav"
    output_path = tmp_path / "mixed.wav"
    base_audio_path.write_bytes(b"base")
    overlay_path.write_bytes(b"overlay")
    captured_command: list[str] = []

    def _fake_run(*args, **kwargs):
        del kwargs
        command = list(args[0])
        captured_command.extend(command)
        Path(command[-1]).write_bytes(b"mixed")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(interjector_module, "ensure_command_available", lambda name: None)
    monkeypatch.setattr(interjector_module.subprocess, "run", _fake_run)

    interjector_module._mix_audio_with_overlays(
        base_audio_path=base_audio_path,
        overlays=[_SynthesizedOverlay(path=overlay_path, start_time_ms=250)],
        output_path=output_path,
        audio_codec="pcm_s24le",
        timeout_seconds=30,
    )

    assert output_path.exists()
    assert any("adelay=250|250" in part for part in captured_command)
    assert any("amix=inputs=2" in part for part in captured_command)


def test_interjector_orchestrator_copies_base_audio_when_no_overlays(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary_turns = [
        VoiceCloneTurn(speaker="A", text="fees matter a lot"),
        VoiceCloneTurn(speaker="B", text="right"),
    ]
    summary_xml = "<A>fees matter a lot</A><B>right</B>"
    manifest_path, _speaker_manifest_path, merged_output_path, durations = (
        _write_stage_three_fixture(tmp_path, summary_turns=summary_turns)
    )
    provider = _RecordingProvider()
    monkeypatch.setattr(
        interjector_module,
        "probe_audio_duration_ms",
        lambda path: durations.get(str(path.resolve()), 250),
    )

    orchestrator = InterjectorOrchestrator(
        planner=_StaticPlanner(
            [InterjectionDecision(host_turn_index=1, should_interject=False)]
        ),
        provider=provider,
        output_dir=tmp_path / "interjector",
    )
    result = orchestrator.run(
        summary_xml=summary_xml,
        voice_clone_manifest_path=manifest_path,
    )
    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert provider.calls == []
    assert result.merged_output_audio_path.read_bytes() == merged_output_path.read_bytes()
    assert manifest_payload["artifact_count"] == 0


def test_interjector_orchestrator_uses_next_speaker_sample_for_overlap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    summary_turns = [
        VoiceCloneTurn(speaker="A", text="fees matter a lot"),
        VoiceCloneTurn(speaker="B", text="right"),
    ]
    summary_xml = "<A>fees matter a lot</A><B>right</B>"
    manifest_path, speaker_manifest_path, _merged_output_path, durations = (
        _write_stage_three_fixture(tmp_path, summary_turns=summary_turns)
    )
    provider = _RecordingProvider()
    mixed_outputs: list[dict[str, object]] = []

    monkeypatch.setattr(
        interjector_module,
        "probe_audio_duration_ms",
        lambda path: durations.get(str(path.resolve()), 250),
    )
    monkeypatch.setattr(
        interjector_module,
        "_align_turn_tokens",
        lambda **kwargs: [
            AlignedToken(token="fees", start_time_ms=0, end_time_ms=200),
            AlignedToken(token="matter", start_time_ms=200, end_time_ms=400),
            AlignedToken(token="a", start_time_ms=400, end_time_ms=600),
            AlignedToken(token="lot", start_time_ms=600, end_time_ms=800),
        ],
    )

    def _fake_mix_audio_with_overlays(**kwargs) -> None:
        mixed_outputs.append(dict(kwargs))
        output_path = kwargs["output_path"]
        assert isinstance(output_path, Path)
        output_path.write_bytes(b"interjected")

    monkeypatch.setattr(
        interjector_module,
        "_mix_audio_with_overlays",
        _fake_mix_audio_with_overlays,
    )

    orchestrator = InterjectorOrchestrator(
        planner=_StaticPlanner(
            [
                InterjectionDecision(
                    host_turn_index=1,
                    should_interject=True,
                    interjection_style="backchannel",
                    interjection_speaker="B",
                    interjection_text="right",
                    anchor_start_token_index=1,
                    anchor_end_token_index=2,
                    anchor_text="matter a",
                )
            ]
        ),
        provider=provider,
        output_dir=tmp_path / "interjector",
    )
    result = orchestrator.run(
        summary_xml=summary_xml,
        voice_clone_manifest_path=manifest_path,
    )
    manifest_payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    speaker_samples = json.loads(speaker_manifest_path.read_text(encoding="utf-8"))["samples"]
    speaker_b_path = next(
        Path(sample["path"]).resolve()
        for sample in speaker_samples
        if sample["speaker"] == "B"
    )

    assert provider.calls[0]["reference_audio_path"] == speaker_b_path
    assert provider.calls[0]["text"] == "right"
    assert mixed_outputs[0]["overlays"][0].start_time_ms == 720
    assert manifest_payload["artifact_count"] == 1
    assert manifest_payload["artifacts"][0]["speaker"] == "B"
