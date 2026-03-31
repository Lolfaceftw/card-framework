from pathlib import Path

from card_framework.audio_pipeline.eta import StageProgressUpdate
from card_framework.audio_pipeline.gateways import demucs_gateway
from card_framework.audio_pipeline.gateways.demucs_gateway import DemucsSourceSeparator
from card_framework.shared.events import event_bus


def test_demucs_separator_plans_outer_windows() -> None:
    separator = DemucsSourceSeparator(
        window_length_seconds=600,
        window_context_seconds=5,
    )

    windows = separator._plan_windows(
        total_duration_ms=1_500_000,
        device="cpu",
    )

    assert [
        (
            window.base_start_ms,
            window.base_end_ms,
            window.context_start_ms,
            window.context_end_ms,
        )
        for window in windows
    ] == [
        (0, 600_000, 0, 605_000),
        (600_000, 1_200_000, 595_000, 1_205_000),
        (1_200_000, 1_500_000, 1_195_000, 1_500_000),
    ]


def test_demucs_separator_uses_smaller_windows_on_cuda() -> None:
    separator = DemucsSourceSeparator(
        window_length_seconds=600,
        window_context_seconds=5,
        cuda_window_length_seconds=180,
    )

    windows = separator._plan_windows(
        total_duration_ms=500_000,
        device="cuda",
    )

    assert [
        (
            window.base_start_ms,
            window.base_end_ms,
            window.context_start_ms,
            window.context_end_ms,
        )
        for window in windows
    ] == [
        (0, 180_000, 0, 185_000),
        (180_000, 360_000, 175_000, 365_000),
        (360_000, 500_000, 355_000, 500_000),
    ]


def test_demucs_separator_uses_single_pass_for_short_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    separator = DemucsSourceSeparator()
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"audio")
    output_dir = tmp_path / "separation"
    progress_updates: list[StageProgressUpdate] = []
    system_messages: list[str] = []
    single_pass_calls: list[tuple[Path, Path, str]] = []

    def _fake_probe(audio_path: Path) -> int:
        del audio_path
        return 30_000

    def _fake_single_pass(
        self,
        *,
        input_audio_path: Path,
        output_dir: Path,
        device: str,
    ) -> Path:
        del self
        single_pass_calls.append((input_audio_path, output_dir, device))
        vocals_path = output_dir / "htdemucs" / input_audio_path.stem / "vocals.wav"
        vocals_path.parent.mkdir(parents=True, exist_ok=True)
        vocals_path.write_bytes(b"vocals")
        return vocals_path

    def _unexpected_windowed(**kwargs) -> Path:
        raise AssertionError(f"Windowed separation should not run: {kwargs}")

    monkeypatch.setattr(demucs_gateway, "probe_audio_duration_ms", _fake_probe)
    monkeypatch.setattr(
        DemucsSourceSeparator,
        "_separate_single_pass_vocals",
        _fake_single_pass,
    )
    monkeypatch.setattr(
        DemucsSourceSeparator,
        "_separate_windowed_vocals",
        _unexpected_windowed,
    )
    monkeypatch.setattr(demucs_gateway, "ensure_module_available", lambda module: None)
    monkeypatch.setattr(demucs_gateway, "ensure_command_available", lambda command: None)

    def _capture_system(message: str) -> None:
        system_messages.append(message)

    event_bus.subscribe("system_message", _capture_system)
    try:
        result = separator.separate_vocals(
            input_audio_path=input_audio,
            output_dir=output_dir,
            device="cpu",
            progress_callback=progress_updates.append,
        )
    finally:
        event_bus.unsubscribe("system_message", _capture_system)

    assert result == output_dir / "htdemucs" / "input" / "vocals.wav"
    assert single_pass_calls == [(input_audio, output_dir, "cpu")]
    assert [update.note for update in progress_updates] == ["chunk 1/1"]
    assert [update.processed_audio_ms for update in progress_updates] == [30_000]
    assert system_messages[0] == "Loading Demucs source separation 'htdemucs'..."
    assert system_messages[1].startswith(
        "Loaded Demucs source separation 'htdemucs' in "
    )


def test_demucs_separator_uses_windowed_mode_for_long_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    separator = DemucsSourceSeparator(
        window_length_seconds=600,
        window_context_seconds=5,
    )
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"audio")
    output_dir = tmp_path / "separation"
    progress_updates: list[StageProgressUpdate] = []
    system_messages: list[str] = []
    extracted_windows: list[tuple[int, int, Path]] = []
    trimmed_windows: list[tuple[int, int, Path]] = []
    concat_inputs: list[Path] = []

    def _fake_probe(audio_path: Path) -> int:
        del audio_path
        return 1_500_000

    def _fake_extract(
        self,
        *,
        source_audio_path: Path,
        start_ms: int,
        end_ms: int,
        output_path: Path,
    ) -> None:
        del self, source_audio_path
        extracted_windows.append((start_ms, end_ms, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"chunk")

    def _fake_single_pass(
        self,
        *,
        input_audio_path: Path,
        output_dir: Path,
        device: str,
    ) -> Path:
        del self, device
        vocals_path = output_dir / "htdemucs" / input_audio_path.stem / "vocals.wav"
        vocals_path.parent.mkdir(parents=True, exist_ok=True)
        vocals_path.write_bytes(b"vocals")
        return vocals_path

    def _fake_trim(
        self,
        *,
        source_audio_path: Path,
        start_ms: int,
        end_ms: int,
        output_path: Path,
    ) -> None:
        del self, source_audio_path
        trimmed_windows.append((start_ms, end_ms, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"trimmed")

    def _fake_concat(
        self,
        *,
        input_paths: list[Path],
        output_path: Path,
    ) -> None:
        del self
        concat_inputs.extend(input_paths)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"vocals")

    monkeypatch.setattr(demucs_gateway, "probe_audio_duration_ms", _fake_probe)
    monkeypatch.setattr(
        DemucsSourceSeparator,
        "_extract_audio_window",
        _fake_extract,
    )
    monkeypatch.setattr(
        DemucsSourceSeparator,
        "_separate_single_pass_vocals",
        _fake_single_pass,
    )
    monkeypatch.setattr(
        DemucsSourceSeparator,
        "_trim_audio_window",
        _fake_trim,
    )
    monkeypatch.setattr(
        DemucsSourceSeparator,
        "_concat_audio_files",
        _fake_concat,
    )
    monkeypatch.setattr(demucs_gateway, "ensure_module_available", lambda module: None)
    monkeypatch.setattr(demucs_gateway, "ensure_command_available", lambda command: None)

    def _capture_system(message: str) -> None:
        system_messages.append(message)

    event_bus.subscribe("system_message", _capture_system)
    try:
        result = separator.separate_vocals(
            input_audio_path=input_audio,
            output_dir=output_dir,
            device="cpu",
            progress_callback=progress_updates.append,
        )
    finally:
        event_bus.unsubscribe("system_message", _capture_system)

    assert result == output_dir / "htdemucs" / "input" / "vocals.wav"
    assert extracted_windows == [
        (
            0,
            605_000,
            output_dir / "_windowed" / "input" / "inputs" / "input_chunk_0001.wav",
        ),
        (
            595_000,
            1_205_000,
            output_dir / "_windowed" / "input" / "inputs" / "input_chunk_0002.wav",
        ),
        (
            1_195_000,
            1_500_000,
            output_dir / "_windowed" / "input" / "inputs" / "input_chunk_0003.wav",
        ),
    ]
    assert [(start_ms, end_ms) for start_ms, end_ms, _ in trimmed_windows] == [
        (0, 600_000),
        (5_000, 605_000),
        (5_000, 305_000),
    ]
    assert [path.name for path in concat_inputs] == ["0001.wav", "0002.wav", "0003.wav"]
    assert [update.note for update in progress_updates] == [
        "chunk 1/3",
        "chunk 2/3",
        "chunk 3/3",
    ]
    assert [update.processed_audio_ms for update in progress_updates] == [
        600_000,
        1_200_000,
        1_500_000,
    ]
    assert system_messages[0] == "Loading Demucs source separation 'htdemucs'..."
    assert system_messages[1].startswith(
        "Loaded Demucs source separation 'htdemucs' in "
    )
    assert not (output_dir / "_windowed" / "input").exists()


def test_demucs_separator_reports_cuda_windowing_mode_for_long_audio(
    tmp_path: Path,
    monkeypatch,
) -> None:
    separator = DemucsSourceSeparator(
        window_length_seconds=600,
        window_context_seconds=5,
        cuda_window_length_seconds=180,
    )
    input_audio = tmp_path / "input.wav"
    input_audio.write_bytes(b"audio")
    output_dir = tmp_path / "separation"
    progress_updates: list[StageProgressUpdate] = []
    system_messages: list[str] = []
    extracted_windows: list[tuple[int, int, Path]] = []

    def _fake_probe(audio_path: Path) -> int:
        del audio_path
        return 500_000

    def _fake_extract(
        self,
        *,
        source_audio_path: Path,
        start_ms: int,
        end_ms: int,
        output_path: Path,
    ) -> None:
        del self, source_audio_path
        extracted_windows.append((start_ms, end_ms, output_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"chunk")

    def _fake_single_pass(
        self,
        *,
        input_audio_path: Path,
        output_dir: Path,
        device: str,
    ) -> Path:
        del self
        assert device == "cuda"
        vocals_path = output_dir / "htdemucs" / input_audio_path.stem / "vocals.wav"
        vocals_path.parent.mkdir(parents=True, exist_ok=True)
        vocals_path.write_bytes(b"vocals")
        return vocals_path

    def _fake_trim(
        self,
        *,
        source_audio_path: Path,
        start_ms: int,
        end_ms: int,
        output_path: Path,
    ) -> None:
        del self, source_audio_path, start_ms, end_ms
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"trimmed")

    def _fake_concat(
        self,
        *,
        input_paths: list[Path],
        output_path: Path,
    ) -> None:
        del self, input_paths
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"vocals")

    monkeypatch.setattr(demucs_gateway, "probe_audio_duration_ms", _fake_probe)
    monkeypatch.setattr(
        DemucsSourceSeparator,
        "_extract_audio_window",
        _fake_extract,
    )
    monkeypatch.setattr(
        DemucsSourceSeparator,
        "_separate_single_pass_vocals",
        _fake_single_pass,
    )
    monkeypatch.setattr(
        DemucsSourceSeparator,
        "_trim_audio_window",
        _fake_trim,
    )
    monkeypatch.setattr(
        DemucsSourceSeparator,
        "_concat_audio_files",
        _fake_concat,
    )
    monkeypatch.setattr(demucs_gateway, "ensure_module_available", lambda module: None)
    monkeypatch.setattr(demucs_gateway, "ensure_command_available", lambda command: None)

    def _capture_system(message: str) -> None:
        system_messages.append(message)

    event_bus.subscribe("system_message", _capture_system)
    try:
        separator.separate_vocals(
            input_audio_path=input_audio,
            output_dir=output_dir,
            device="cuda",
            progress_callback=progress_updates.append,
        )
    finally:
        event_bus.unsubscribe("system_message", _capture_system)

    assert extracted_windows == [
        (
            0,
            185_000,
            output_dir / "_windowed" / "input" / "inputs" / "input_chunk_0001.wav",
        ),
        (
            175_000,
            365_000,
            output_dir / "_windowed" / "input" / "inputs" / "input_chunk_0002.wav",
        ),
        (
            355_000,
            500_000,
            output_dir / "_windowed" / "input" / "inputs" / "input_chunk_0003.wav",
        ),
    ]
    assert [update.note for update in progress_updates] == [
        "chunk 1/3",
        "chunk 2/3",
        "chunk 3/3",
    ]
    assert system_messages[0] == "Loading Demucs source separation 'htdemucs'..."
    assert (
        system_messages[1]
        == "Demucs CUDA windowing active: using 180s outer windows instead of the default 600s."
    )
    assert system_messages[2].startswith(
        "Loaded Demucs source separation 'htdemucs' in "
    )
