import json
from pathlib import Path
import time
from typing import cast

import pytest

from audio_pipeline.contracts import (
    DiarizationTurn,
    TimedTextSegment,
    TranscriptionResult,
    WordTimestamp,
)
import threading

from audio_pipeline.eta import DynamicEtaTracker, LinearStageEtaStrategy, StageSpeedProfile
from audio_pipeline.orchestrator import AudioToScriptOrchestrator


class _StubSeparator:
    def separate_vocals(
        self,
        input_audio_path: Path,
        output_dir: Path,
        *,
        device: str,
        progress_callback=None,
    ) -> Path:
        del output_dir, device, progress_callback
        return input_audio_path


class _StubTranscriber:
    def transcribe(
        self,
        audio_path: Path,
        *,
        device: str,
        progress_callback=None,
    ) -> TranscriptionResult:
        del audio_path, device, progress_callback
        return TranscriptionResult(
            segments=[TimedTextSegment(start_time_ms=0, end_time_ms=800, text="hello world")],
            word_timestamps=[
                WordTimestamp(word="hello", start_time_ms=0, end_time_ms=400),
                WordTimestamp(word="world", start_time_ms=400, end_time_ms=800),
            ],
            language="en",
        )


class _StubDiarizer:
    def diarize(
        self,
        audio_path: Path,
        output_dir: Path,
        *,
        device: str,
        progress_callback=None,
    ) -> list[DiarizationTurn]:
        del audio_path, output_dir, device, progress_callback
        return [DiarizationTurn(speaker="SPEAKER_00", start_time_ms=0, end_time_ms=1000)]


def test_orchestrator_hides_eta_without_history(monkeypatch, tmp_path) -> None:
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

    assert not any("estimated time left" in message for _, message in messages)


def test_orchestrator_announces_eta_for_each_stage(monkeypatch, tmp_path) -> None:
    messages: list[tuple[str, str]] = []

    def _capture(event_type: str, message: str, **kwargs) -> None:
        del kwargs
        messages.append((event_type, message))

    monkeypatch.setattr("audio_pipeline.orchestrator.event_bus.publish", _capture)
    monkeypatch.setattr("audio_pipeline.orchestrator.probe_audio_duration_ms", lambda _: 2000)
    monkeypatch.setattr(
        "audio_pipeline.orchestrator.AudioToScriptOrchestrator._load_eta_profile",
        lambda self, *, profile_path, context: None,
    )
    strategy = LinearStageEtaStrategy(
        separation=StageSpeedProfile(cpu=2.0, cuda=1.0),
        transcription=StageSpeedProfile(cpu=2.0, cuda=1.0),
        diarization=StageSpeedProfile(cpu=2.0, cuda=1.0),
    )
    for stage_name in ("separation", "transcription", "diarization"):
        strategy.observe_stage_duration(
            stage=stage_name,
            audio_duration_ms=2000,
            elapsed_seconds=2.0,
            device="cpu",
        )

    orchestrator = AudioToScriptOrchestrator(
        separator=_StubSeparator(),
        transcriber=_StubTranscriber(),
        diarizer=_StubDiarizer(),
        eta_strategy=strategy,
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


def test_orchestrator_persists_completed_stage_eta_before_next_stage(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr("audio_pipeline.orchestrator.probe_audio_duration_ms", lambda _: 2000)

    profile_path = tmp_path / "work" / "eta_profile.json"
    profile_exists_when_transcription_starts: list[bool] = []

    class _CheckingTranscriber:
        def transcribe(
            self,
            audio_path: Path,
            *,
            device: str,
            progress_callback=None,
        ) -> TranscriptionResult:
            del audio_path, device, progress_callback
            profile_exists_when_transcription_starts.append(profile_path.exists())
            raise RuntimeError("stop after separation")

    orchestrator = AudioToScriptOrchestrator(
        separator=_StubSeparator(),
        transcriber=_CheckingTranscriber(),
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

    with pytest.raises(RuntimeError, match="stop after separation"):
        orchestrator.run(
            input_audio_path=input_audio_path,
            output_transcript_path=tmp_path / "out.json",
            work_dir=tmp_path / "work",
            device="cpu",
        )

    assert profile_exists_when_transcription_starts == [True]
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    assert payload["audio_stages"]["separation"]["cpu"]["samples"] == 1


def test_orchestrator_eta_reports_overrun_until_stage_stops(monkeypatch) -> None:
    messages: list[tuple[str, str]] = []

    def _capture(event_type: str, message: str, **kwargs) -> None:
        del kwargs
        messages.append((event_type, message))

    monkeypatch.setattr("audio_pipeline.orchestrator.event_bus.publish", _capture)

    orchestrator = AudioToScriptOrchestrator(
        separator=_StubSeparator(),
        transcriber=_StubTranscriber(),
        diarizer=_StubDiarizer(),
        eta_strategy=LinearStageEtaStrategy(
            separation=StageSpeedProfile(cpu=2.0, cuda=1.0),
            transcription=StageSpeedProfile(cpu=2.0, cuda=1.0),
            diarization=StageSpeedProfile(cpu=2.0, cuda=1.0),
        ),
        eta_update_interval_seconds=0.01,
    )

    class _DeterministicStopEvent:
        def __init__(self) -> None:
            self.calls = 0

        def wait(self, _timeout: float) -> bool:
            self.calls += 1
            return self.calls > 1

    orchestrator._publish_eta_updates(
        stage="separation",
        eta_tracker=DynamicEtaTracker(initial_total_seconds=0.01),
        tracker_lock=threading.Lock(),
        started_at=time.monotonic() - 0.2,
        stop_event=cast(threading.Event, _DeterministicStopEvent()),
    )

    assert any(
        "running longer than estimate by" in message for _, message in messages
    )
