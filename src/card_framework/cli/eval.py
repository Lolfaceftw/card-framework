"""Compatibility wrapper for the benchmark package.

This module preserves the old entrypoint while delegating to the new benchmark
runner. The default behavior now executes the ``smoke`` preset.
"""

from __future__ import annotations

from card_framework.benchmark.run import main as benchmark_main


def main() -> int:
    """Run the smoke benchmark preset through the new benchmark CLI."""
    return benchmark_main(["execute", "--preset", "smoke"])


if __name__ == "__main__":
    raise SystemExit(main())

