"""The ``card-framework`` command: shorten a recording, or check this machine."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import IO, TYPE_CHECKING

from card_framework.shared.durations import duration_label, parse_duration

if TYPE_CHECKING:
    from card_framework.runtime.preflight import Check

PROG = "card-framework"
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_NOT_READY = 3
EXIT_INTERRUPTED = 130

_DESCRIPTION = "Shorten long conversations to a target length, in the speakers' own voices."
_EXAMPLES = """\
examples:
  card-framework doctor
  card-framework shorten episode.wav --to 5m
  card-framework shorten episode.wav --to 90s --llm deepseek_chat
  card-framework shorten episode.wav --to 5m --llm-url http://gpu-box:8000/v1

The LLM defaults to a vLLM server at http://localhost:8000/v1. Set
CARD_FRAMEWORK_VLLM_API_KEY if that server needs a key. API profiles read their
key from the environment or a .env file in the folder you run from, for example
DEEPSEEK_API_KEY for --llm deepseek_chat.
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the ``card-framework`` command."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=_DESCRIPTION,
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {_version()}")
    commands = parser.add_subparsers(
        title="commands", dest="command", metavar="COMMAND", required=True
    )

    shorten = commands.add_parser(
        "shorten",
        help="shorten one recording to a target length",
        description="Shorten one recording to a target length, in the speakers' own voices.",
        epilog=_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    shorten.add_argument(
        "audio",
        type=Path,
        help="recording to shorten (WAV, MP3, or anything FFmpeg reads)",
    )
    shorten.add_argument(
        "-t",
        "--to",
        dest="target",
        required=True,
        type=_duration_arg,
        metavar="LENGTH",
        help="target length, such as 30s, 5m, 1m30s, or 1:30",
    )
    shorten.add_argument(
        "-o",
        "--out",
        type=Path,
        metavar="DIR",
        help="folder for the results (default: ./<recording>-<length>)",
    )
    shorten.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="where models run (default: auto, which uses a CUDA GPU when available)",
    )
    _add_llm_arguments(shorten)
    shorten.add_argument(
        "--json",
        action="store_true",
        help="print the result paths as JSON instead of the final audio path",
    )
    shorten.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show the pipeline's full output instead of the stage list",
    )
    shorten.set_defaults(handler=_run_shorten)

    doctor = commands.add_parser(
        "doctor",
        help="check that this machine is ready to run CARD",
        description="Check tools, GPU, LLM, and storage without downloading anything.",
    )
    _add_llm_arguments(doctor)
    doctor.add_argument("--json", action="store_true", help="print the checks as JSON")
    doctor.set_defaults(handler=_run_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``card-framework`` command and return its exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        _error("stopped")
        return EXIT_INTERRUPTED


def _add_llm_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--llm",
        metavar="PROFILE",
        help="built-in LLM profile, such as deepseek_chat or google_gemini",
    )
    group.add_argument(
        "--llm-url",
        metavar="URL",
        help="OpenAI-compatible server to use for every LLM step, such as a vLLM server",
    )


def _run_shorten(args: argparse.Namespace) -> int:
    audio_path: Path = args.audio
    if not audio_path.is_file():
        _error(f"{audio_path}: no such file")
        return EXIT_USAGE
    output_dir = args.out or Path.cwd() / f"{audio_path.stem}-{duration_label(args.target)}"

    from card_framework.api import NotReadyError, RuntimeBootstrapError, infer

    try:
        result = infer(
            audio_path,
            output_dir,
            args.target,
            device=args.device,
            vllm_url=args.llm_url,
            llm=args.llm,
            verbose=args.verbose,
        )
    except NotReadyError as exc:
        _print_checks(exc.checks, stream=sys.stderr)
        return EXIT_NOT_READY
    except RuntimeBootstrapError as exc:
        _error(str(exc))
        return EXIT_NOT_READY
    except (FileNotFoundError, ValueError) as exc:
        _error(str(exc))
        return EXIT_USAGE
    except RuntimeError as exc:
        _error(str(exc))
        return EXIT_FAILED

    if args.json:
        payload = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(result).items()
        }
        print(json.dumps(payload, indent=2))
    else:
        print(result.final_audio_path or result.summary_xml_path)
    return EXIT_OK


def _run_doctor(args: argparse.Namespace) -> int:
    from card_framework.api import doctor_checks

    checks = doctor_checks(llm=args.llm, vllm_url=args.llm_url)
    if args.json:
        print(json.dumps([asdict(check) for check in checks], indent=2))
    else:
        _print_checks(checks, stream=sys.stdout, title="doctor")
    return EXIT_NOT_READY if any(check.status == "fail" for check in checks) else EXIT_OK


def _print_checks(checks: Sequence[Check], *, stream: IO[str], title: str = "") -> None:
    from rich.padding import Padding
    from rich.table import Table
    from rich.text import Text

    from card_framework.shared.stage_view import header, make_console

    console = make_console(stream)
    fancy = console.is_terminal and not console.is_dumb_terminal
    failures = sum(1 for check in checks if check.status == "fail")

    if not fancy:
        for check in checks:
            print(f"[{check.status}] {check.name}: {check.detail}", file=stream)
            if check.fix and check.status != "ok":
                print(f"       fix: {check.fix}", file=stream)
        print("ready" if not failures else f"not ready: {failures} problem(s) to fix", file=stream)
        return

    if title:
        console.print(header(title))
        console.print()
    glyphs = {"ok": ("✓", "card.done"), "warn": ("!", "bold"), "fail": ("✗", "card.fail")}
    table = Table.grid(padding=(0, 2))
    table.add_column(width=2)
    table.add_column(no_wrap=True)
    table.add_column(overflow="fold")
    for check in checks:
        glyph, style = glyphs[check.status]
        detail = Text(check.detail, style="" if check.status == "fail" else "card.muted")
        if check.fix and check.status != "ok":
            detail.append(f"\n{check.fix}", style="card.faint")
        table.add_row(Text(glyph, style=style), check.name, detail)
    console.print(Padding(table, (0, 0, 0, 3)))
    console.print()
    if failures:
        noun = "problem" if failures == 1 else "problems"
        console.print(Text(f"   Not ready: {failures} {noun} to fix.", style="card.fail"))
    else:
        console.print(Text("   Ready to shorten.", style="card.done"))


def _duration_arg(text: str) -> int:
    try:
        return parse_duration(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _version() -> str:
    try:
        return importlib.metadata.version("card-framework")
    except importlib.metadata.PackageNotFoundError:
        return "unknown (not installed)"


def _error(message: str) -> None:
    print(f"{PROG}: {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
