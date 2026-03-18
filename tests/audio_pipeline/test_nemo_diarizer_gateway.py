"""Tests for the NeMo diarizer gateway."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_nemo_diarizer_builds_msdd_config_with_vad_postprocessing_overrides(
    tmp_path: Path,
) -> None:
    omegaconf = pytest.importorskip("omegaconf")
    manifest_path = tmp_path / "diarization_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    output_dir = tmp_path / "diarization"
    diarizer = NemoSpeakerDiarizer(
        min_speakers=2,
        max_speakers=5,
        vad_onset=0.75,
        vad_offset=0.55,
        vad_pad_onset=0.05,
        vad_pad_offset=0.0,
        vad_min_duration_on=0.3,
        vad_min_duration_off=0.7,
    )

    cfg = diarizer._build_msdd_config(
        manifest_path=manifest_path,
        output_dir=output_dir,
        omega_conf=omegaconf.OmegaConf,
        device="cpu",
    )

    vad_parameters = cfg.diarizer.vad.parameters
    assert vad_parameters.onset == pytest.approx(0.75)
    assert vad_parameters.offset == pytest.approx(0.55)
    assert vad_parameters.pad_onset == pytest.approx(0.05)
    assert vad_parameters.pad_offset == pytest.approx(0.0)
    assert vad_parameters.min_duration_on == pytest.approx(0.3)
    assert vad_parameters.min_duration_off == pytest.approx(0.7)
    clustering_parameters = cfg.diarizer.clustering.parameters
    assert clustering_parameters.max_num_speakers == 5
    if "min_num_speakers" in clustering_parameters:
        assert clustering_parameters.min_num_speakers == 2
