"""Tests for the NeMo diarizer gateway."""

from __future__ import annotations

from pathlib import Path

from card_framework.audio_pipeline.contracts import DiarizationTurn
from card_framework.audio_pipeline.gateways.nemo_diarizer_gateway import (
    NemoSpeakerDiarizer,
)
from card_framework.shared.events import event_bus


def test_nemo_diarizer_reports_loading_messages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Emit shared loading-status messages around NeMo diarization startup."""
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    output_dir = tmp_path / "diarization"
    diarizer = NemoSpeakerDiarizer()

    monkeypatch.setattr(
        diarizer,
        "_run_nemo_msdd",
        lambda *, audio_path, output_dir, device: [
            DiarizationTurn(
                speaker="SPEAKER_00",
                start_time_ms=0,
                end_time_ms=1000,
            ),
            DiarizationTurn(
                speaker="SPEAKER_01",
                start_time_ms=1000,
                end_time_ms=2000,
            ),
        ],
    )

    messages: list[str] = []

    def _capture(message: str) -> None:
        messages.append(message)

    event_bus.subscribe("system_message", _capture)
    try:
        turns = diarizer.diarize(
            audio_path=audio_path,
            output_dir=output_dir,
            device="cpu",
        )
    finally:
        event_bus.unsubscribe("system_message", _capture)

    assert len(turns) == 2
    assert messages[0] == "Loading NeMo diarization model 'diar_msdd_telephonic'..."
    assert messages[1].startswith(
        "Loaded NeMo diarization model 'diar_msdd_telephonic' in "
    )
