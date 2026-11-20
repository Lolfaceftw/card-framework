"""Shared pytest configuration for the repository."""

from __future__ import annotations

import importlib


def pytest_configure() -> None:
    """Preload required dependencies before test modules install fallback stubs."""
    for module_name in (
        "a2a.server.agent_execution",
        "a2a.server.events",
        "a2a.utils",
        "numpy",
        "openai",
    ):
        importlib.import_module(module_name)
