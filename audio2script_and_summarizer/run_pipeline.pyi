"""Type shim for the runtime run_pipeline compatibility facade."""

from typing import Any


def __getattr__(name: str) -> Any: ...


def main(*args: Any, **kwargs: Any) -> int: ...
