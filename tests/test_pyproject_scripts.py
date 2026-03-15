"""Regression coverage for pyproject script declarations."""

from __future__ import annotations

from pathlib import Path
import tomllib


def _load_pyproject_payload() -> dict[str, object]:
    """Load the repository pyproject payload for metadata assertions."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"

    with pyproject_path.open("rb") as handle:
        return tomllib.load(handle)


def test_pyproject_scripts_are_parseable_and_include_benchmark_supervisors() -> None:
    """Keep benchmark supervisor scripts valid for uv-driven workflows."""
    payload = _load_pyproject_payload()

    scripts = payload["project"]["scripts"]
    assert scripts["card-framework-benchmark-watch"] == (
        "card_framework.benchmark.watchdog:main"
    )
    assert scripts["card-framework-qa"] == "card_framework.benchmark.qa:main"
    assert scripts["card-framework-qa-supervisor"] == (
        "card_framework.benchmark.qa_supervisor:main"
    )


def test_pyproject_declares_cross_platform_packaged_cpu_metadata() -> None:
    """Keep the published metadata aligned with the supported CPU platform set."""
    payload = _load_pyproject_payload()

    project = payload["project"]
    classifiers = project["classifiers"]
    dependencies = project["dependencies"]
    uv_sources = payload["tool"]["uv"]["sources"]

    assert "Operating System :: Microsoft :: Windows" in classifiers
    assert "Operating System :: POSIX :: Linux" in classifiers
    assert "Operating System :: MacOS :: MacOS X" in classifiers
    assert "nemo-toolkit[asr]>=2.4.0; sys_platform == 'linux'" in dependencies
    assert uv_sources["torch"] == [
        {
            "index": "pytorch-cu126",
            "marker": "sys_platform == 'linux' or sys_platform == 'win32'",
        }
    ]
    assert uv_sources["torchaudio"] == [
        {
            "index": "pytorch-cu126",
            "marker": "sys_platform == 'linux' or sys_platform == 'win32'",
        }
    ]
