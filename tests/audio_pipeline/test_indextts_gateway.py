"""Tests for IndexTTS gateway subprocess command construction."""

from __future__ import annotations

from pathlib import Path
import sys

from card_framework.audio_pipeline.gateways.indextts_voice_clone_gateway import (
    _build_persistent_worker_command,
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
