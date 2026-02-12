"""Integration-style tests for Stage 3 with mocked heavy dependencies."""

from __future__ import annotations

import json
import wave
from pathlib import Path

from audio2script_and_summarizer.stage3_voice import (
    InterjectionCandidate,
    InterjectionPlannerProtocol,
    InterjectionRequest,
    TTSEngineProtocol,
    run_stage3_pipeline,
)


def _write_silent_wav(path: Path, duration_ms: int, sample_rate: int = 16000) -> None:
    """Create a silent mono WAV for deterministic tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = int(sample_rate * (duration_ms / 1000.0))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"\x00\x00" * sample_count)


class FakeTTSEngine(TTSEngineProtocol):
    """Minimal fake TTS backend that writes silent WAV output."""

    def infer(
        self,
        *,
        spk_audio_prompt: str,
        text: str,
        output_path: str,
        emo_alpha: float,
        use_emo_text: bool,
        emo_text: str,
        use_random: bool,
        verbose: bool,
    ) -> object:
        del spk_audio_prompt, emo_alpha, use_emo_text, emo_text, use_random, verbose
        duration_ms = max(900, min(3000, len(text.split()) * 180))
        _write_silent_wav(Path(output_path), duration_ms=duration_ms)
        return output_path


class FakePlanner(InterjectionPlannerProtocol):
    """Planner stub that injects one interjection for segment index 1."""

    def __init__(self, *, available: bool = True) -> None:
        self._available = available

    def ensure_available(self) -> bool:
        return self._available

    def propose(self, request: InterjectionRequest) -> InterjectionCandidate | None:
        if request.segment_index != 1:
            return None
        return InterjectionCandidate(
            segment_index=request.segment_index,
            interjector_index=-1,
            interjection_text="uh-huh",
            anchor_phrase="about",
            style="agreement",
            confidence=0.95,
        )


def _write_summary_and_voices(tmp_path: Path) -> Path:
    """Create a test summary JSON and voice prompt WAVs."""
    voice_a = tmp_path / "SPEAKER_00.wav"
    voice_b = tmp_path / "SPEAKER_01.wav"
    _write_silent_wav(voice_a, 500)
    _write_silent_wav(voice_b, 500)
    summary_path = tmp_path / "audio_summary.json"
    payload = [
        {
            "speaker": "SPEAKER_00",
            "voice_sample": "SPEAKER_00.wav",
            "text": "Welcome back to the show.",
            "use_emo_text": True,
            "emo_text": "Warm",
            "emo_alpha": 0.6,
        },
        {
            "speaker": "SPEAKER_01",
            "voice_sample": "SPEAKER_01.wav",
            "text": "Today we are talking about practical system design.",
            "use_emo_text": True,
            "emo_text": "Thoughtful",
            "emo_alpha": 0.7,
        },
    ]
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    return summary_path


def test_run_stage3_pipeline_with_mocked_engines(tmp_path: Path) -> None:
    """Produce output WAV and log when fake planner returns interjections."""
    summary_path = _write_summary_and_voices(tmp_path)
    output_path = tmp_path / "final.wav"

    result = run_stage3_pipeline(
        summary_json_path=summary_path,
        output_wav_path=output_path,
        indextts_cfg_path=tmp_path / "cfg.yaml",
        indextts_model_dir=tmp_path,
        device="cpu",
        interjection_max_ratio=1.0,
        tts_engine=FakeTTSEngine(),
        planner=FakePlanner(available=True),
    )

    assert result.output_wav_path == output_path.resolve()
    assert result.output_wav_path.exists()
    assert result.interjection_log_path.exists()
    log_payload = json.loads(result.interjection_log_path.read_text(encoding="utf-8"))
    assert len(log_payload) == 1


def test_run_stage3_pipeline_degrades_when_planner_unavailable(tmp_path: Path) -> None:
    """Continue synthesis without interjections when planner is unavailable."""
    summary_path = _write_summary_and_voices(tmp_path)
    output_path = tmp_path / "final_no_interjections.wav"

    result = run_stage3_pipeline(
        summary_json_path=summary_path,
        output_wav_path=output_path,
        indextts_cfg_path=tmp_path / "cfg.yaml",
        indextts_model_dir=tmp_path,
        device="cpu",
        interjection_max_ratio=0.5,
        tts_engine=FakeTTSEngine(),
        planner=FakePlanner(available=False),
    )

    assert result.output_wav_path.exists()
    assert result.interjection_count == 0
    log_payload = json.loads(result.interjection_log_path.read_text(encoding="utf-8"))
    assert log_payload == []
