"""Compatibility wrapper for the benchmark package.

This module preserves the old entrypoint while delegating to the new benchmark
runner. The default behavior now executes the ``smoke`` preset.
"""

from __future__ import annotations

import sys

from card_framework.benchmark.run import main as benchmark_main


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark CLI through the legacy eval entrypoint.

    Args:
        argv: Optional raw CLI arguments.

    Returns:
        Process exit code from the delegated benchmark runner.
    """
    effective_argv = list(sys.argv[1:] if argv is None else argv)
    if effective_argv:
        return benchmark_main(effective_argv)
    return benchmark_main(["execute", "--preset", "smoke"])


if __name__ == "__main__":
    raise SystemExit(main())
