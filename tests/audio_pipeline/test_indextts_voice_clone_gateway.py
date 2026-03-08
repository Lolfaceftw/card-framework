from __future__ import annotations

from collections import deque
import builtins
from pathlib import Path
import sys
import types

import pytest

import audio_pipeline.gateways.indextts_voice_clone_gateway as gateway_module
from audio_pipeline.errors import DependencyMissingError
from audio_pipeline.errors import NonRetryableAudioStageError
from audio_pipeline.gateways.indextts_voice_clone_gateway import (
    IndexTTSVoiceCloneGateway,
)


@pytest.fixture(autouse=True)
def _reset_shared_gateway_state() -> None:
    """Reset shared IndexTTS gateway caches between tests."""
    IndexTTSVoiceCloneGateway.release_all_cached_resources()
    _StubIndexTTS2.init_calls.clear()
    _StubIndexTTS2.infer_calls.clear()
    _StubQwenEmotion.inference_calls.clear()
    _FakePersistentWorker.init_calls.clear()
    _FakePersistentWorker.synthesize_calls.clear()
    _FakePersistentWorker.close_calls = 0
    yield
    IndexTTSVoiceCloneGateway.release_all_cached_resources()


class _StubQwenEmotion:
    """Capture emotion-text analysis calls for cache tests."""

    inference_calls: list[str] = []

    def inference(self, text: str) -> dict[str, float]:
        type(self).inference_calls.append(text)
        return {
            "happy": 0.0,
            "angry": 0.0,
            "sad": 0.0,
            "afraid": 0.0,
            "disgusted": 0.0,
            "melancholic": 0.0,
            "surprised": 0.0,
            "calm": 1.0,
        }


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
        self.qwen_emo = _StubQwenEmotion()
        type(self).init_calls.append(
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
        emo_vector: list[float] | None = None,
        do_sample: bool = True,
        top_p: float = 0.8,
        top_k: int = 30,
        temperature: float = 0.8,
        length_penalty: float = 0.0,
        num_beams: int = 1,
        repetition_penalty: float = 10.0,
        max_mel_tokens: int = 1500,
    ) -> str:
        type(self).infer_calls.append(
            {
                "spk_audio_prompt": spk_audio_prompt,
                "text": text,
                "output_path": output_path,
                "verbose": verbose,
                "max_text_tokens_per_segment": max_text_tokens_per_segment,
                "use_emo_text": use_emo_text,
                "emo_text": emo_text,
                "emo_vector": emo_vector,
                "do_sample": do_sample,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                "length_penalty": length_penalty,
                "num_beams": num_beams,
                "repetition_penalty": repetition_penalty,
                "max_mel_tokens": max_mel_tokens,
            }
        )
        Path(output_path).write_bytes(b"wav")
        return output_path


class _FakePersistentWorker:
    """Capture shared persistent-worker lifecycle for subprocess backend tests."""

    init_calls: list[dict[str, object]] = []
    synthesize_calls: list[dict[str, object]] = []
    close_calls = 0

    def __init__(self, **kwargs: object) -> None:
        type(self).init_calls.append(dict(kwargs))
        self._alive = True

    def synthesize(
        self,
        *,
        reference_audio_path: Path,
        text: str,
        output_audio_path: Path,
        emo_text: str | None,
        verbose: bool,
        max_text_tokens_per_segment: int,
        generation_kwargs: dict[str, object],
    ) -> Path:
        type(self).synthesize_calls.append(
            {
                "reference_audio_path": reference_audio_path,
                "text": text,
                "output_audio_path": output_audio_path,
                "emo_text": emo_text,
                "verbose": verbose,
                "max_text_tokens_per_segment": max_text_tokens_per_segment,
                "generation_kwargs": generation_kwargs,
            }
        )
        output_audio_path.write_bytes(b"wav")
        return output_audio_path

    def close(self) -> None:
        type(self).close_calls += 1
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive


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
        "emo_vector": None,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 0.8,
        "length_penalty": 0.0,
        "num_beams": 1,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 1500,
    }
    assert _StubIndexTTS2.infer_calls[1] == {
        "spk_audio_prompt": str(reference_audio),
        "text": "Second segment",
        "output_path": str(output_b),
        "verbose": True,
        "max_text_tokens_per_segment": 99,
        "use_emo_text": False,
        "emo_text": None,
        "emo_vector": None,
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 0.8,
        "length_penalty": 0.0,
        "num_beams": 1,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 1500,
    }
    assert output_a.exists()
    assert output_b.exists()


def test_indextts_gateway_caches_emo_text_vectors_across_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_stub_indextts(monkeypatch)

    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("version: test", encoding="utf-8")
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        execution_backend="inprocess",
    )

    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="First segment",
        output_audio_path=tmp_path / "output_a.wav",
        emo_text="warmly, gently, and supportively, like an empathetic co-host.",
    )
    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="Second segment",
        output_audio_path=tmp_path / "output_b.wav",
        emo_text="warmly, gently, and supportively, like an empathetic co-host.",
    )

    assert _StubQwenEmotion.inference_calls == [
        "warmly, gently, and supportively, like an empathetic co-host."
    ]
    assert _StubIndexTTS2.infer_calls[0]["use_emo_text"] is False
    assert _StubIndexTTS2.infer_calls[0]["emo_text"] is None
    assert _StubIndexTTS2.infer_calls[0]["emo_vector"] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert _StubIndexTTS2.infer_calls[1]["emo_vector"] == [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_indextts_gateway_shares_inprocess_model_across_matching_gateways(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_stub_indextts(monkeypatch)

    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("version: test", encoding="utf-8")
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")

    gateway_a = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        execution_backend="inprocess",
    )
    gateway_b = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        execution_backend="inprocess",
    )

    gateway_a.synthesize(
        reference_audio_path=reference_audio,
        text="First",
        output_audio_path=tmp_path / "output_a.wav",
    )
    gateway_b.synthesize(
        reference_audio_path=reference_audio,
        text="Second",
        output_audio_path=tmp_path / "output_b.wav",
    )

    assert len(_StubIndexTTS2.init_calls) == 1
    assert len(_StubIndexTTS2.infer_calls) == 2


def test_indextts_gateway_close_releases_shared_inprocess_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_stub_indextts(monkeypatch)

    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("version: test", encoding="utf-8")
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        execution_backend="inprocess",
    )
    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="First",
        output_audio_path=tmp_path / "output_a.wav",
    )
    gateway.close()
    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="Second",
        output_audio_path=tmp_path / "output_b.wav",
    )

    assert len(_StubIndexTTS2.init_calls) == 2


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


def test_build_persistent_worker_command_includes_expected_runtime_args(
    tmp_path: Path,
) -> None:
    runner_project_dir = tmp_path / "third_party" / "index_tts"
    runner_script = tmp_path / "audio_pipeline" / "runners" / "indextts_persistent_runner.py"
    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"

    command = gateway_module._build_persistent_worker_command(
        uv_executable="uv",
        runner_project_dir=runner_project_dir,
        runner_script=runner_script,
        cfg_path=cfg_path,
        model_dir=model_dir,
        device="cuda",
        use_fp16=True,
        use_cuda_kernel=True,
        use_deepspeed=True,
        use_accel=True,
        use_torch_compile=True,
    )

    assert command == [
        "uv",
        "run",
        "--project",
        str(runner_project_dir),
        "python",
        str(runner_script),
        "--cfg-path",
        str(cfg_path),
        "--model-dir",
        str(model_dir),
        "--device",
        "cuda",
        "--use-fp16",
        "--use-cuda-kernel",
        "--use-deepspeed",
        "--use-accel",
        "--use-torch-compile",
    ]


def test_indextts_gateway_generation_kwargs_skip_length_penalty_without_beam_search(
    tmp_path: Path,
) -> None:
    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=tmp_path / "checkpoints" / "config.yaml",
        model_dir=tmp_path / "checkpoints",
        num_beams=1,
        length_penalty=0.4,
    )

    assert gateway._generation_kwargs() == {
        "do_sample": True,
        "top_p": 0.8,
        "top_k": 30,
        "temperature": 0.8,
        "num_beams": 1,
        "repetition_penalty": 10.0,
        "max_mel_tokens": 1500,
    }


def test_persistent_worker_record_log_line_sanitizes_stream_output_for_cp1252(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = object.__new__(gateway_module._PersistentIndexTTSSubprocessWorker)
    worker._stream_output = True
    worker._log_tail = deque(maxlen=5)

    printed: list[str] = []

    class _StdoutStub:
        encoding = "cp1252"

    def _fake_print(*args: object, **kwargs: object) -> None:
        del kwargs
        printed.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr(gateway_module.sys, "stdout", _StdoutStub())
    monkeypatch.setattr(builtins, "print", _fake_print)

    worker._record_log_line("Summarizer \u2192 ready \U0001f449")

    assert list(worker._log_tail) == ["Summarizer \u2192 ready \U0001f449"]
    assert printed == ["Summarizer -> ready ?"]


def test_indextts_gateway_subprocess_backend_reuses_shared_worker_across_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        gateway_module,
        "_PersistentIndexTTSSubprocessWorker",
        _FakePersistentWorker,
    )

    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("version: test", encoding="utf-8")
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")
    runner_project_dir = tmp_path / "third_party" / "index_tts"
    runner_project_dir.mkdir(parents=True, exist_ok=True)

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        device="cuda",
        use_fp16=True,
        execution_backend="subprocess",
        runner_project_dir=runner_project_dir,
        uv_executable="uv",
        verbose=True,
        max_text_tokens_per_segment=88,
        num_beams=2,
        top_p=0.9,
    )

    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="Text A",
        output_audio_path=tmp_path / "output_a.wav",
    )
    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="Text B",
        output_audio_path=tmp_path / "output_b.wav",
    )

    assert len(_FakePersistentWorker.init_calls) == 1
    assert len(_FakePersistentWorker.synthesize_calls) == 2
    assert _FakePersistentWorker.synthesize_calls[0] == {
        "reference_audio_path": reference_audio,
        "text": "Text A",
        "output_audio_path": tmp_path / "output_a.wav",
        "emo_text": None,
        "verbose": True,
        "max_text_tokens_per_segment": 88,
        "generation_kwargs": {
            "do_sample": True,
            "top_p": 0.9,
            "top_k": 30,
            "temperature": 0.8,
            "length_penalty": 0.0,
            "num_beams": 2,
            "repetition_penalty": 10.0,
            "max_mel_tokens": 1500,
        },
    }
    assert _FakePersistentWorker.synthesize_calls[1]["text"] == "Text B"


def test_indextts_gateway_subprocess_backend_shares_worker_across_matching_gateways(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        gateway_module,
        "_PersistentIndexTTSSubprocessWorker",
        _FakePersistentWorker,
    )

    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("version: test", encoding="utf-8")
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")
    runner_project_dir = tmp_path / "third_party" / "index_tts"
    runner_project_dir.mkdir(parents=True, exist_ok=True)

    gateway_a = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        execution_backend="subprocess",
        runner_project_dir=runner_project_dir,
    )
    gateway_b = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        execution_backend="subprocess",
        runner_project_dir=runner_project_dir,
    )

    gateway_a.synthesize(
        reference_audio_path=reference_audio,
        text="Text A",
        output_audio_path=tmp_path / "output_a.wav",
    )
    gateway_b.synthesize(
        reference_audio_path=reference_audio,
        text="Text B",
        output_audio_path=tmp_path / "output_b.wav",
    )

    assert len(_FakePersistentWorker.init_calls) == 1
    assert len(_FakePersistentWorker.synthesize_calls) == 2


def test_indextts_gateway_close_releases_shared_subprocess_worker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        gateway_module,
        "_PersistentIndexTTSSubprocessWorker",
        _FakePersistentWorker,
    )

    cfg_path = tmp_path / "checkpoints" / "config.yaml"
    model_dir = tmp_path / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text("version: test", encoding="utf-8")
    reference_audio = tmp_path / "speaker.wav"
    reference_audio.write_bytes(b"ref")
    runner_project_dir = tmp_path / "third_party" / "index_tts"
    runner_project_dir.mkdir(parents=True, exist_ok=True)

    gateway = IndexTTSVoiceCloneGateway(
        cfg_path=cfg_path,
        model_dir=model_dir,
        execution_backend="subprocess",
        runner_project_dir=runner_project_dir,
    )
    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="Text A",
        output_audio_path=tmp_path / "output_a.wav",
    )
    gateway.close()
    gateway.synthesize(
        reference_audio_path=reference_audio,
        text="Text B",
        output_audio_path=tmp_path / "output_b.wav",
    )

    assert len(_FakePersistentWorker.init_calls) == 2
    assert _FakePersistentWorker.close_calls == 1
