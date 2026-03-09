"""Lightweight CLI for the event-driven terminal UI module."""

from __future__ import annotations

import argparse


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the lightweight CLI parser for the terminal UI module."""
    return argparse.ArgumentParser(
        description=(
            "Inspect the shared terminal UI event subscribers. Importing "
            "`card_framework.cli.ui` wires the runtime event bus; this CLI "
            "prints help and exits."
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Run the lightweight UI CLI."""
    parser = build_arg_parser()
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
