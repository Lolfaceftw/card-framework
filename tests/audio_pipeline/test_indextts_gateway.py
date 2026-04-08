"""Tests for IndexTTS gateway subprocess command construction."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
import sys

import pytest

from card_framework.audio_pipeline.emotion.emotion_vectors import EmotionVector
from card_framework.audio_pipeline.errors import NonRetryableAudioStageError
from card_framework.audio_pipeline.gateways import (
    indextts_voice_clone_gateway as gateway_module,
)
from card_framework.audio_pipeline.gateways.indextts_voice_clone_gateway import (
    IndexTTSVoiceCloneGateway,
    _CudaMemorySnapshot,
    _PersistentIndexTTSSubprocessWorker,
    _build_persistent_worker_command,
    _build_persistent_worker_env,
)


def test_persistent_worker_command_pins_active_python() -> None:
    """Pin the nested uv worker to the active interpreter."""
    runner_project_dir = Path("runner-project")
    runner_script = Path("runner.py")
    cfg_path = Path("config.yaml")
    model_dir = Path("models")

    command = _build_persistent_worker_command(
        uv_executable="uv",
        runner_project_dir=runner_project_dir,
        runner_script=runner_script,
        cfg_path=cfg_path,
        model_dir=model_dir,
        device="cuda",
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
    )

    assert command[:6] == [
        "uv",
        "run",
        "--project",
        str(runner_project_dir),
        "--python",
        sys.executable,
    ]
    assert command[6:8] == ["python", str(runner_script)]


def test_persistent_worker_env_prepends_repo_source_root() -> None:
    """Forward the repo source tree into nested worker imports."""
    source_root = Path("repo") / "src"

    environment = _build_persistent_worker_env(
        source_root=source_root,
        existing_env={
            "PYTHONPATH": "existing-path",
            "OTHER": "value",
            "VIRTUAL_ENV": ".venv",
        },
    )

    pythonpath_entries = environment["PYTHONPATH"].split(os.pathsep)
    assert environment["OTHER"] == "value"
    assert "VIRTUAL_ENV" not in environment
    assert pythonpath_entries[0] == str(source_root.resolve())
    assert "existing-path" in pythonpath_entries


def test_persistent_worker_env_moves_existing_source_root_to_front() -> None:
    source_root = Path("repo") / "src"
    source_root_text = str(source_root.resolve())

    environment = _build_persistent_worker_env(
        source_root=source_root,
        existing_env={
            "PYTHONPATH": os.pathsep.join(
                ["shadow-path", source_root_text, "other-path"]
            ),
        },
    )

    assert environment["PYTHONPATH"].split(os.pathsep) == [
        source_root_text,
        "shadow-path",
        "other-path",
    ]


def test_worker_failure_detail_keeps_recent_logs_when_streaming() -> None:
    worker = _PersistentIndexTTSSubprocessWorker.__new__(
        _PersistentIndexTTSSubprocessWorker
    )
    worker._stream_output = True
    worker._log_tail = deque(
        [
            "Traceback (most recent call last):",
            "RuntimeError: simulated worker boom",
        ],
        maxlen=80,
    )

    detail = worker._format_failure_detail(
        "IndexTTS subprocess worker exited unexpectedly"
    )

    assert "IndexTTS subprocess worker exited unexpectedly" in detail
    assert "Recent subprocess output:" in detail
    assert "RuntimeError: simulated worker boom" in detail
    assert "See streamed subprocess output above for full context." in detail


def test_worker_failure_detail_keeps_recent_logs_when_error_text_is_duplicated() -> (
    None
):
    worker = _PersistentIndexTTSSubprocessWorker.__new__(
        _PersistentIndexTTSSubprocessWorker
    )
    worker._stream_output = True
    worker._log_tail = deque(
        [
            "IndexTTS subprocess worker exited unexpectedly",
            "RuntimeError: duplicated failure text",
        ],
        maxlen=80,
    )

    detail = worker._format_failure_detail(
        "IndexTTS subprocess worker exited unexpectedly"
    )

    assert "Recent subprocess output:" in detail
    assert "RuntimeError: duplicated failure text" in detail


def test_worker_init_closes_process_when_ready_wait_raises(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("model: test\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    runner_project_dir = tmp_path / "runner"
    runner_project_dir.mkdir()

    close_called: list[bool] = []

    class _FakePipe:
        def __init__(self) -> None:
            self.closed = False

        def write(self, _text: str) -> None:
            return None

        def flush(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdin = _FakePipe()
            self.stdout = _FakePipe()

        def poll(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    class _FakeThread:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            del timeout
            return None

    fake_process = _FakeProcess()
    monkeypatch.setattr(
        gateway_module.subprocess, "Popen", lambda *args, **kwargs: fake_process
    )
    monkeypatch.setattr(gateway_module.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        gateway_module._PersistentIndexTTSSubprocessWorker,
        "_wait_for_protocol_message",
        lambda self, **kwargs: (_ for _ in ()).throw(
            NonRetryableAudioStageError("startup wait failed")
        ),
    )

    original_close = gateway_module._PersistentIndexTTSSubprocessWorker.close

    def _recording_close(self) -> None:
        close_called.append(True)
        original_close(self)

    monkeypatch.setattr(
        gateway_module._PersistentIndexTTSSubprocessWorker,
        "close",
        _recording_close,
    )

    try:
        gateway_module._PersistentIndexTTSSubprocessWorker(
            cfg_path=cfg_path,
            model_dir=model_dir,
            device="cuda",
            use_fp16=False,
            use_cuda_kernel=False,
            use_deepspeed=False,
            use_accel=False,
            use_torch_compile=False,
            runner_project_dir=runner_project_dir,
            uv_executable="uv",
            stream_output=False,
        )
    except NonRetryableAudioStageError as exc:
        assert str(exc) == "startup wait failed"
    else:
        raise AssertionError("expected worker startup failure")

    assert close_called == [True]
    assert fake_process.stdin.closed is True
    assert fake_process.stdout.closed is True


def test_gateway_normalizes_direct_emotion_vector_for_provider() -> None:
    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=Path("config.yaml"),
        model_dir=Path("models"),
        normalize_emotion_vector=True,
        emotion_vector_apply_bias=True,
        emotion_vector_max_sum=0.8,
    )

    normalized = gateway._normalize_provider_emotion_vector(
        EmotionVector((0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.3, 0.5))
    )

    assert normalized is not None
    assert sum(normalized.values) <= 0.800001


def test_gateway_prefers_direct_emotion_vector_over_emo_text() -> None:
    """Use explicit vectors without invoking the emo-text analysis path."""

    class _FailingQwenEmotion:
        def inference(self, emo_text: str):
            raise AssertionError(f"qwen_emo should not be called for {emo_text!r}")

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=Path("config.yaml"),
        model_dir=Path("models"),
        normalize_emotion_vector=False,
    )

    resolved = gateway._resolve_direct_or_cached_emo_vector(
        tts_model=type("FakeTTS", (), {"qwen_emo": _FailingQwenEmotion()})(),
        emotion_vector=EmotionVector((0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24)),
        emo_text="warmly",
    )

    assert resolved == [0.14, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.24]


def test_gateway_retries_unexpected_subprocess_startup_crash_on_cuda(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("model: test\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    runner_project_dir = tmp_path / "runner"
    runner_project_dir.mkdir()

    calls: list[tuple[str, bool, bool, bool, bool, bool]] = []
    notices: list[tuple[str, str]] = []
    cleanup_calls: list[str] = []

    class _FakeWorker:
        startup_attempts = 0

        def __init__(
            self,
            *,
            device: str,
            use_fp16: bool,
            use_cuda_kernel: bool,
            use_deepspeed: bool,
            use_accel: bool,
            use_torch_compile: bool,
            **_: object,
        ) -> None:
            calls.append(
                (
                    device,
                    use_fp16,
                    use_cuda_kernel,
                    use_deepspeed,
                    use_accel,
                    use_torch_compile,
                )
            )
            if device == "cuda" and _FakeWorker.startup_attempts == 0:
                _FakeWorker.startup_attempts += 1
                raise NonRetryableAudioStageError(
                    "IndexTTS subprocess worker exited unexpectedly: boom"
                )
            self._device = device

        @property
        def device(self) -> str:
            return self._device

        def is_alive(self) -> bool:
            return True

        def close(self) -> None:
            return None

    monkeypatch.setattr(
        gateway_module, "_PersistentIndexTTSSubprocessWorker", _FakeWorker
    )
    monkeypatch.setattr(
        gateway_module.event_bus,
        "publish",
        lambda channel, message, *args, **kwargs: notices.append(
            (channel, str(message))
        ),
    )
    monkeypatch.setattr(
        gateway_module,
        "release_torch_device_memory_with_reporting",
        lambda **kwargs: cleanup_calls.append(str(kwargs.get("device", ""))),
    )
    monkeypatch.setattr(
        gateway_module,
        "_query_cuda_memory_snapshot",
        lambda: _CudaMemorySnapshot(free_mb=10788, total_mb=12288, used_mb=1500),
    )

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        device="cuda",
        use_fp16=True,
        use_cuda_kernel=True,
        use_deepspeed=True,
        use_accel=True,
        use_torch_compile=True,
        runner_project_dir=runner_project_dir,
    )

    worker = gateway._get_or_init_subprocess_worker()

    assert worker.device == "cuda"
    assert calls == [
        ("cuda", True, True, True, True, True),
        ("cuda", True, True, True, True, True),
    ]
    assert cleanup_calls == ["cuda"]
    assert any("retrying once on CUDA" in message for _channel, message in notices)


def test_gateway_retries_synthesis_with_cuda_when_cached_worker_dies(
    monkeypatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("model: test\n", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    runner_project_dir = tmp_path / "runner"
    runner_project_dir.mkdir()
    reference_audio_path = tmp_path / "reference.wav"
    reference_audio_path.write_bytes(b"wav")
    output_audio_path = tmp_path / "out.wav"

    events: list[str] = []
    worker_events: list[str] = []
    cleanup_calls: list[str] = []

    class _FakeWorker:
        synth_failures = 0

        def __init__(self, *, device: str, **_: object) -> None:
            self._device = device
            worker_events.append(f"init:{device}")

        @property
        def device(self) -> str:
            return self._device

        def is_alive(self) -> bool:
            return True

        def close(self) -> None:
            worker_events.append(f"close:{self._device}")

        def synthesize(self, **kwargs: object) -> Path:
            worker_events.append(f"synth:{self._device}")
            if self._device == "cuda" and _FakeWorker.synth_failures == 0:
                _FakeWorker.synth_failures += 1
                raise NonRetryableAudioStageError(
                    "IndexTTS subprocess worker exited unexpectedly before synthesis: boom"
                )
            rendered_path = Path(str(kwargs["output_audio_path"]))
            rendered_path.write_bytes(b"wav")
            return rendered_path

    monkeypatch.setattr(
        gateway_module, "_PersistentIndexTTSSubprocessWorker", _FakeWorker
    )
    monkeypatch.setattr(
        gateway_module.event_bus,
        "publish",
        lambda channel, message, *args, **kwargs: events.append(f"{channel}:{message}"),
    )
    monkeypatch.setattr(
        gateway_module,
        "release_torch_device_memory_with_reporting",
        lambda **kwargs: cleanup_calls.append(str(kwargs.get("device", ""))),
    )
    monkeypatch.setattr(
        gateway_module,
        "_query_cuda_memory_snapshot",
        lambda: _CudaMemorySnapshot(free_mb=10788, total_mb=12288, used_mb=1500),
    )

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        device="cuda",
        runner_project_dir=runner_project_dir,
    )

    rendered_path = gateway._synthesize_subprocess(
        reference_audio_path=reference_audio_path,
        text="hello there",
        output_audio_path=output_audio_path,
        emo_text=None,
        emotion_vector=None,
    )
    second_output_audio_path = tmp_path / "out-second.wav"
    second_rendered_path = gateway._synthesize_subprocess(
        reference_audio_path=reference_audio_path,
        text="hello again",
        output_audio_path=second_output_audio_path,
        emo_text=None,
        emotion_vector=None,
    )

    assert rendered_path == output_audio_path
    assert output_audio_path.exists()
    assert second_rendered_path == second_output_audio_path
    assert second_output_audio_path.exists()
    assert worker_events == [
        "init:cuda",
        "synth:cuda",
        "close:cuda",
        "init:cuda",
        "synth:cuda",
        "synth:cuda",
    ]
    assert cleanup_calls == ["cuda"]
    assert any("retrying once on CUDA" in event for event in events)


def test_gateway_waits_for_cuda_headroom_before_startup(monkeypatch) -> None:
    events: list[str] = []
    sleep_calls: list[float] = []
    cleanup_calls: list[str] = []
    snapshots = iter(
        [
            _CudaMemorySnapshot(free_mb=5000, total_mb=12288, used_mb=7288),
            _CudaMemorySnapshot(free_mb=5500, total_mb=12288, used_mb=6788),
            _CudaMemorySnapshot(free_mb=10788, total_mb=12288, used_mb=1500),
        ]
    )

    monkeypatch.setattr(
        gateway_module, "_query_cuda_memory_snapshot", lambda: next(snapshots)
    )
    monkeypatch.setattr(
        gateway_module.time, "sleep", lambda seconds: sleep_calls.append(seconds)
    )
    monkeypatch.setattr(
        gateway_module,
        "release_torch_device_memory_with_reporting",
        lambda **kwargs: cleanup_calls.append(str(kwargs.get("device", ""))),
    )
    monkeypatch.setattr(
        gateway_module.event_bus,
        "publish",
        lambda channel, message, *args, **kwargs: events.append(f"{channel}:{message}"),
    )

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=Path("config.yaml"),
        model_dir=Path("models"),
        device="cuda",
        cuda_startup_min_free_memory_mb=6000,
        cuda_startup_wait_seconds=30.0,
        cuda_startup_poll_seconds=2.0,
    )

    gateway._wait_for_cuda_worker_headroom(device="cuda")

    assert cleanup_calls == ["cuda"]
    assert sleep_calls == [2.0]
    assert any(
        "Waiting for IndexTTS CUDA VRAM budget/headroom" in event for event in events
    )
    assert any(
        "Detected enough IndexTTS CUDA VRAM budget/headroom" in event
        for event in events
    )


def test_gateway_requires_cuda_memory_telemetry_for_headroom_guard(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gateway_module, "_query_cuda_memory_snapshot", lambda: None)

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=Path("config.yaml"),
        model_dir=Path("models"),
        device="cuda",
        cuda_startup_min_free_memory_mb=6000,
    )

    with pytest.raises(
        NonRetryableAudioStageError, match="Unable to query NVIDIA GPU memory"
    ):
        gateway._wait_for_cuda_worker_headroom(device="cuda")


def test_gateway_still_enforces_telemetry_when_free_headroom_threshold_is_zero(
    monkeypatch,
) -> None:
    monkeypatch.setattr(gateway_module, "_query_cuda_memory_snapshot", lambda: None)

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=Path("config.yaml"),
        model_dir=Path("models"),
        device="cuda",
        cuda_startup_min_free_memory_mb=0,
        cuda_vram_budget_mb=8192,
    )

    with pytest.raises(
        NonRetryableAudioStageError, match="Unable to query NVIDIA GPU memory"
    ):
        gateway._wait_for_cuda_worker_headroom(device="cuda")


def test_gateway_raises_when_cuda_headroom_never_recovers(monkeypatch) -> None:
    events: list[str] = []
    cleanup_calls: list[str] = []
    monkeypatch.setattr(
        gateway_module,
        "_query_cuda_memory_snapshot",
        lambda: _CudaMemorySnapshot(free_mb=3500, total_mb=12288, used_mb=8788),
    )
    monkeypatch.setattr(gateway_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        gateway_module,
        "release_torch_device_memory_with_reporting",
        lambda **kwargs: cleanup_calls.append(str(kwargs.get("device", ""))),
    )
    monkeypatch.setattr(
        gateway_module.event_bus,
        "publish",
        lambda channel, message, *args, **kwargs: events.append(f"{channel}:{message}"),
    )

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=Path("config.yaml"),
        model_dir=Path("models"),
        device="cuda",
        cuda_startup_min_free_memory_mb=6000,
        cuda_startup_wait_seconds=0.0,
        cuda_startup_poll_seconds=1.0,
    )

    with pytest.raises(
        NonRetryableAudioStageError, match="Insufficient CUDA VRAM budget/headroom"
    ):
        gateway._wait_for_cuda_worker_headroom(device="cuda")

    assert cleanup_calls == ["cuda"]
    assert any(
        "Waiting for IndexTTS CUDA VRAM budget/headroom" in event for event in events
    )
