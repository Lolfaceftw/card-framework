"""Type shim for the runtime summarizer_deepseek compatibility module."""

from typing import Any


def __getattr__(name: str) -> Any: ...


def main(*args: Any, **kwargs: Any) -> int: ...
