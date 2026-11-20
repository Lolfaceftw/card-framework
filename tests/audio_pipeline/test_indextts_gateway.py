"""Tests for IndexTTS gateway subprocess command construction."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from card_framework.audio_pipeline.emotion_vectors import EmotionVector
from card_framework.audio_pipeline.gateways.indextts_voice_clone_gateway import (
    IndexTTSVoiceCloneGateway,
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
        existing_env={"PYTHONPATH": "existing-path", "OTHER": "value"},
    )

    pythonpath_entries = environment["PYTHONPATH"].split(os.pathsep)
    assert environment["OTHER"] == "value"
    assert pythonpath_entries[0] == str(source_root.resolve())
    assert "existing-path" in pythonpath_entries


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
