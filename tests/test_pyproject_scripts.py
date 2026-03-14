"""Regression coverage for pyproject script declarations."""

from __future__ import annotations

from pathlib import Path
import tomllib


def test_pyproject_scripts_are_parseable_and_include_benchmark_supervisors() -> None:
    """Keep benchmark supervisor scripts valid for uv-driven workflows."""
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"

    with pyproject_path.open("rb") as handle:
        payload = tomllib.load(handle)

    scripts = payload["project"]["scripts"]
    assert scripts["card-framework-benchmark-watch"] == (
        "card_framework.benchmark.watchdog:main"
    )
    assert scripts["card-framework-qa"] == "card_framework.benchmark.qa:main"
    assert scripts["card-framework-qa-supervisor"] == (
        "card_framework.benchmark.qa_supervisor:main"
    )
