from __future__ import annotations

from pathlib import Path

from audio_pipeline.gateways.sortformer_diarizer_gateway import SortformerSpeakerDiarizer


class _FakeStreamingModules:
    def __init__(self) -> None:
        self.chunk_len = 0
        self.chunk_right_context = 0
        self.fifo_len = 0
        self.spkcache_update_period = 0
        self.spkcache_len = 0
        self.checked = False

    def _check_streaming_parameters(self) -> None:
        self.checked = True


class _FakeStreamingModel:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.sortformer_modules = _FakeStreamingModules()

    def diarize(self, *, audio, batch_size: int):
        del audio, batch_size
        return self.payload


def test_configure_streaming_updates_model_modules() -> None:
    diarizer = SortformerSpeakerDiarizer(
        model_name="nvidia/diar_streaming_sortformer_4spk-v2",
        streaming_mode=True,
        chunk_len=124,
        chunk_right_context=1,
        fifo_len=124,
        spkcache_update_period=124,
        spkcache_len=188,
    )
    model = _FakeStreamingModel(payload=[])

    diarizer._configure_streaming(model)

    modules = model.sortformer_modules
    assert modules.chunk_len == 124
    assert modules.chunk_right_context == 1
    assert modules.fifo_len == 124
    assert modules.spkcache_update_period == 124
    assert modules.spkcache_len == 188
    assert modules.checked is True


def test_diarize_parses_segments_and_normalizes_labels(
    monkeypatch,
    tmp_path: Path,
) -> None:
    diarizer = SortformerSpeakerDiarizer(
        model_name="nvidia/diar_sortformer_4spk-v1",
        streaming_mode=False,
    )
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"wav")
    diarizer._model_cache["cpu"] = _FakeStreamingModel(
        payload=[[(0.0, 1.0, 1), (1.0, 2.0, 0)]]
    )

    monkeypatch.setattr(
        "audio_pipeline.gateways.sortformer_diarizer_gateway.prepare_diarization_audio",
        lambda *, audio_path, output_dir: audio_path,
    )

    turns = diarizer.diarize(
        audio_path=input_audio,
        output_dir=tmp_path / "out",
        device="cpu",
    )

    assert [turn.speaker for turn in turns] == ["SPEAKER_00", "SPEAKER_01"]
    assert turns[0].start_time_ms == 0
    assert turns[1].end_time_ms == 2000
