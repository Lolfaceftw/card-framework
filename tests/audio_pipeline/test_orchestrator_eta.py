from pathlib import Path

from audio_pipeline.contracts import DiarizationTurn, TimedTextSegment
from audio_pipeline.eta import LinearStageEtaStrategy, StageSpeedProfile
from audio_pipeline.orchestrator import AudioToScriptOrchestrator


class _StubSeparator:
    def separate_vocals(self, input_audio_path: Path, output_dir: Path, *, device: str) -> Path:
        del output_dir, device
        return input_audio_path


class _StubTranscriber:
    def transcribe(self, audio_path: Path, *, device: str) -> list[TimedTextSegment]:
        del audio_path, device
        return [TimedTextSegment(start_time_ms=0, end_time_ms=800, text="hello world")]


class _StubDiarizer:
    def diarize(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        device: str,
    ) -> list[DiarizationTurn]:
        del audio_path, output_dir, device
        return [DiarizationTurn(speaker="SPEAKER_00", start_time_ms=0, end_time_ms=1000)]


def test_orchestrator_announces_eta_for_each_stage(monkeypatch, tmp_path) -> None:
    messages: list[tuple[str, str]] = []

    def _capture(event_type: str, message: str, **kwargs) -> None:
        del kwargs
        messages.append((event_type, message))

    monkeypatch.setattr("audio_pipeline.orchestrator.event_bus.publish", _capture)
    monkeypatch.setattr("audio_pipeline.orchestrator.probe_audio_duration_ms", lambda _: 2000)

    orchestrator = AudioToScriptOrchestrator(
        separator=_StubSeparator(),
        transcriber=_StubTranscriber(),
        diarizer=_StubDiarizer(),
        eta_strategy=LinearStageEtaStrategy(
            separation=StageSpeedProfile(cpu=2.0, cuda=1.0),
            transcription=StageSpeedProfile(cpu=2.0, cuda=1.0),
            diarization=StageSpeedProfile(cpu=2.0, cuda=1.0),
        ),
        eta_update_interval_seconds=0.0,
    )

    input_audio_path = tmp_path / "audio.wav"
    input_audio_path.write_bytes(b"fake")
    output_transcript_path = tmp_path / "out.json"

    orchestrator.run(
        input_audio_path=input_audio_path,
        output_transcript_path=output_transcript_path,
        work_dir=tmp_path / "work",
        device="cpu",
    )

    eta_start_messages = [
        msg for _, msg in messages if "estimated time left" in msg
    ]
    assert len(eta_start_messages) >= 3
    assert any("separating sources" in msg for msg in eta_start_messages)
    assert any("transcribing vocals" in msg for msg in eta_start_messages)
    assert any("running NeMo diarization" in msg for msg in eta_start_messages)
