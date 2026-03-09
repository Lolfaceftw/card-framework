"""Real smoke test for the packaged `card_framework.infer` export."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import venv

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.integration
def test_built_wheel_exports_infer_symbol(tmp_path: Path) -> None:
    """Build the wheel, install it into a clean venv, and import `infer`."""
    build_result = subprocess.run(
        ["uv", "build", "--wheel"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert build_result.returncode == 0, build_result.stderr

    wheel_paths = sorted((REPO_ROOT / "dist").glob("card_framework-*.whl"))
    assert wheel_paths, "No built wheel was found in dist/."
    wheel_path = wheel_paths[-1]

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python_path = (
        venv_dir / "Scripts" / "python.exe"
        if sys.platform == "win32"
        else venv_dir / "bin" / "python"
    )

    install_result = subprocess.run(
        [str(python_path), "-m", "pip", "install", "--no-deps", str(wheel_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=180,
    )
    assert install_result.returncode == 0, install_result.stderr

    import_result = subprocess.run(
        [
            str(python_path),
            "-c",
            (
                "import inspect; "
                "from card_framework import InferenceResult, infer; "
                "print(callable(infer)); "
                "print(InferenceResult.__name__); "
                "print(tuple(inspect.signature(infer).parameters))"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert import_result.returncode == 0, import_result.stderr
    assert "True" in import_result.stdout
    assert "InferenceResult" in import_result.stdout
    assert "target_duration_seconds" in import_result.stdout
    assert "device" in import_result.stdout
    assert "vllm_url" in import_result.stdout
