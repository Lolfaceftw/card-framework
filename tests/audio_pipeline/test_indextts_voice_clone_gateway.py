from __future__ import annotations

import builtins
from pathlib import Path
import subprocess
import sys
import types

import pytest

from audio_pipeline.errors import DependencyMissingError
from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.gateways.indextts_voice_clone_gateway import (
    IndexTTSVoiceCloneGateway,
)


class _StubIndexTTS2:
    """Capture constructor and inference calls for gateway contract tests."""

    init_calls: list[dict[str, object]] = []
    infer_calls: list[dict[str, object]] = []

    def __init__(
        self,
        *,
        cfg_path: str,
        model_dir: str,
        use_fp16: bool,
        device: str,
        use_cuda_kernel: bool,
        use_deepspeed: bool,
        use_accel: bool,
        use_torch_compile: bool,
    ) -> None:
        self.init_calls.append(
            {
                "cfg_path": cfg_path,
                "model_dir": model_dir,
                "use_fp16": use_fp16,
                "device": device,
                "use_cuda_kernel": use_cuda_kernel,
                "use_deepspeed": use_deepspeed,
                "use_accel": use_accel,
                "use_torch_compile": use_torch_compile,
            }
        )

    def infer(
        self,
        *,
        spk_audio_prompt: str,
        text: str,
        output_path: str,
        verbose: bool,
        max_text_tokens_per_segment: int,
        use_emo_text: bool,
        emo_text: str | None,
    ) -> str:
        self.infer_calls.append(
            {
                "spk_audio_prompt": spk_audio_prompt,
                "text": text,
                "output_path": output_path,
                "verbose": verbose,
                "max_text_tokens_per_segment": max_text_tokens_per_segment,
                "use_emo_text": use_emo_text,
                "emo_text": emo_text,
            }
        )
        Path(output_path).write_bytes(b"wav")
        return output_path


def _install_stub_indextts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install an in-memory ``indextts.infer_v2`` module for tests."""
    package_module = types.ModuleType("indextts")
    infer_module = types.ModuleType("indextts.infer_v2")
    infer_module.IndexTTS2 = _StubIndexTTS2
    package_module.infer_v2 = infer_module
    monkeypatch.setitem(sys.modules, "indextts", package_module)
    monkeypatch.setitem(sys.modules, "indextts.infer_v2", infer_module)


def test_indextts_gateway_invokes_model_with_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _StubIndexTTS2.init_calls.clear()
    _StubIndexTTS2.infer_calls.clear()
    _install_stub_indextts(monkeypatch)

    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("version: test", encoding="utf-8")
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")
    output_a = tmp_path / "output_a.wav"
    output_b = tmp_path / "output_b.wav"

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        device="cpu",
        use_fp16=False,
        use_cuda_kernel=False,
        use_deepspeed=False,
        use_accel=False,
        use_torch_compile=False,
        verbose=True,
        max_text_tokens_per_segment=99,
        execution_backend="inprocess",
    )

    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="First segment",
        output_audio_path=output_a,
    )
    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="Second segment",
        output_audio_path=output_b,
    )

    assert len(_StubIndexTTS2.init_calls) == 1
    assert _StubIndexTTS2.init_calls[0] == {
        "cfg_path": str(cfg_path),
        "model_dir": str(model_dir),
        "use_fp16": False,
        "device": "cpu",
        "use_cuda_kernel": False,
        "use_deepspeed": False,
        "use_accel": False,
        "use_torch_compile": False,
    }
    assert len(_StubIndexTTS2.infer_calls) == 2
    assert _StubIndexTTS2.infer_calls[0] == {
        "spk_audio_prompt": str(reference_audio),
        "text": "First segment",
        "output_path": str(output_a),
        "verbose": True,
        "max_text_tokens_per_segment": 99,
        "use_emo_text": False,
        "emo_text": None,
    }
    assert _StubIndexTTS2.infer_calls[1] == {
        "spk_audio_prompt": str(reference_audio),
        "text": "Second segment",
        "output_path": str(output_b),
        "verbose": True,
        "max_text_tokens_per_segment": 99,
        "use_emo_text": False,
        "emo_text": None,
    }
    assert output_a.exists()
    assert output_b.exists()


def test_indextts_gateway_raises_dependency_missing_error_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    original_import = builtins.__import__

    def _import_hook(name: str, *args: object, **kwargs: object) -> object:
        if name == "indextts.infer_v2":
            raise ImportError("indextts missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_hook)

    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = model_dir / "config.yaml"
    cfg_path.write_text("version: test", encoding="utf-8")
    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        execution_backend="inprocess",
    )

    with pytest.raises(DependencyMissingError, match="IndexTTS2 is not installed"):
        gateway.synthesize(
            reference_audio_path=reference_audio,
            text="Hello",
            output_audio_path=tmp_path / "output.wav",
        )


def test_indextts_gateway_validates_model_artifact_paths(tmp_path: Path) -> None:
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")
    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=tmp_path / "missing" / "config.yaml",
        model_dir=tmp_path / "missing",
        execution_backend="inprocess",
    )

    with pytest.raises(
        NonRetryableAudioStageError,
        match="IndexTTS config file does not exist",
    ):
        gateway.synthesize(
            reference_audio_path=reference_audio,
            text="Hello",
            output_audio_path=tmp_path / "output.wav",
        )


def test_indextts_gateway_subprocess_backend_executes_runner_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("version: test", encoding="utf-8")
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")
    output_audio = tmp_path / "output.wav"
    runner_project_dir = tmp_path / "third_party" / "index_tts"
    runner_project_dir.mkdir(parents=True, exist_ok=True)
    (runner_project_dir / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    captured_commands: list[list[str]] = []
    captured_run_kwargs: list[dict[str, object]] = []

    def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_commands.append(command)
        captured_run_kwargs.append(dict(kwargs))
        output_audio.write_bytes(b"wav")
        return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        device="cuda",
        use_fp16=True,
        use_cuda_kernel=True,
        use_deepspeed=True,
        use_accel=True,
        use_torch_compile=True,
        verbose=True,
        max_text_tokens_per_segment=88,
        execution_backend="subprocess",
        runner_project_dir=runner_project_dir,
        uv_executable="uv",
    )

    rendered_path = gateway.synthesize(
        reference_audio_path=reference_audio,
        text="Text",
        output_audio_path=output_audio,
    )

    assert rendered_path == output_audio
    assert len(captured_commands) == 1
    command = captured_commands[0]
    run_kwargs = captured_run_kwargs[0]
    assert command[0] == "uv"
    assert "--project" in command
    assert str(runner_project_dir.resolve()) in command
    assert "--cfg-path" in command
    assert str(cfg_path) in command
    assert "--model-dir" in command
    assert str(model_dir) in command
    assert "--reference-audio-path" in command
    assert str(reference_audio) in command
    assert "--output-audio-path" in command
    assert str(output_audio) in command
    assert "--device" in command
    assert "cuda" in command
    assert "--use-fp16" in command
    assert "--use-cuda-kernel" in command
    assert "--use-deepspeed" in command
    assert "--use-accel" in command
    assert "--use-torch-compile" in command
    assert "--verbose" in command
    assert run_kwargs["check"] is False
    assert run_kwargs["text"] is True
    assert "capture_output" not in run_kwargs


def test_indextts_gateway_subprocess_backend_can_capture_output_when_streaming_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("version: test", encoding="utf-8")
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")
    output_audio = tmp_path / "output.wav"
    runner_project_dir = tmp_path / "third_party" / "index_tts"
    runner_project_dir.mkdir(parents=True, exist_ok=True)
    (runner_project_dir / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    captured_run_kwargs: list[dict[str, object]] = []

    def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del command
        captured_run_kwargs.append(dict(kwargs))
        output_audio.write_bytes(b"wav")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        execution_backend="subprocess",
        runner_project_dir=runner_project_dir,
        uv_executable="uv",
        stream_subprocess_output=False,
    )

    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="Text",
        output_audio_path=output_audio,
    )

    assert len(captured_run_kwargs) == 1
    run_kwargs = captured_run_kwargs[0]
    assert run_kwargs["check"] is False
    assert run_kwargs["text"] is True
    assert run_kwargs["capture_output"] is True


def test_indextts_gateway_subprocess_backend_streaming_failure_reports_stream_hint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("version: test", encoding="utf-8")
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")
    output_audio = tmp_path / "output.wav"
    runner_project_dir = tmp_path / "third_party" / "index_tts"
    runner_project_dir.mkdir(parents=True, exist_ok=True)
    (runner_project_dir / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")

    def _fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del command, kwargs
        return subprocess.CompletedProcess(args=[], returncode=2, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _fake_run)

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        execution_backend="subprocess",
        runner_project_dir=runner_project_dir,
        uv_executable="uv",
        stream_subprocess_output=True,
    )

    with pytest.raises(
        NonRetryableAudioStageError,
        match="See streamed subprocess output above.",
    ):
        gateway.synthesize(
            reference_audio_path=reference_audio,
            text="Text",
            output_audio_path=output_audio,
        )
