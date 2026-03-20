"""Container-only CLI for diarization-first speaker-sample bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from card_framework.audio_pipeline.calibration import (
    _bootstrap_speaker_samples_from_diarization_local,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Docker speaker-sample bootstrap."""
    parser = argparse.ArgumentParser(
        description=(
            "Run diarization-only speaker-sample bootstrap inside the Docker "
            "runtime."
        )
    )
    parser.add_argument(
        "--audio-cfg-json",
        required=True,
        help="Path to a JSON file containing the resolved audio config mapping.",
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Container-visible source audio path.",
    )
    parser.add_argument(
        "--bootstrap-transcript-path",
        required=True,
        help="Container-visible output path for the lightweight bootstrap transcript.",
    )
    parser.add_argument(
        "--bootstrap-audio-work-dir",
        required=True,
        help="Container-visible work directory for separation and diarization artifacts.",
    )
    parser.add_argument(
        "--speaker-samples-output-dir",
        required=True,
        help="Container-visible output directory for speaker sample artifacts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the local diarization-only bootstrap inside the container."""
    args = build_parser().parse_args(argv)
    audio_cfg_payload = json.loads(
        Path(args.audio_cfg_json).read_text(encoding="utf-8")
    )
    if not isinstance(audio_cfg_payload, dict):
        raise ValueError("--audio-cfg-json must point to a JSON object.")
    _bootstrap_speaker_samples_from_diarization_local(
        project_root=Path.cwd().resolve(),
        audio_cfg=dict(audio_cfg_payload),
        audio_path=Path(args.audio_path).resolve(),
        bootstrap_transcript_path=Path(args.bootstrap_transcript_path).resolve(),
        bootstrap_audio_work_dir=Path(args.bootstrap_audio_work_dir).resolve(),
        speaker_samples_output_dir=Path(args.speaker_samples_output_dir).resolve(),
        status_reporter=print,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
