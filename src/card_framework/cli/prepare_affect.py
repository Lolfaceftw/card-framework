"""Precompute transcript affect features for controlled stage-2 studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, cast

from omegaconf import OmegaConf

from card_framework.audio_pipeline.contracts import TranscriptPayload
from card_framework.audio_pipeline.factory import build_transcript_affect_analyzer
from card_framework.audio_pipeline.io import write_transcript_atomic
from card_framework.orchestration.transcript import Transcript
from card_framework.shared.paths import DEFAULT_CONFIG_PATH


def build_parser() -> argparse.ArgumentParser:
    """Build the non-interactive affect-preparation parser.

    Returns:
        Parser requiring the transcript and isolated preprocessing work path.
    """
    parser = argparse.ArgumentParser(
        description="Analyze one transcript once and persist its affect metadata."
    )
    parser.add_argument("--transcript-path", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config-file", default=str(DEFAULT_CONFIG_PATH))
    return parser


def _load_audio_config(config_path: Path, *, work_dir: Path) -> dict[str, Any]:
    """Load the configured audio section with an isolated work directory.

    Args:
        config_path: Repository runtime configuration file.
        work_dir: Study-owned directory for the affect manifest and profiles.

    Returns:
        Resolved audio configuration accepted by the analyzer factory.

    Raises:
        ValueError: The selected config does not contain an audio mapping or has
            affect analysis disabled.
    """
    config = OmegaConf.load(config_path)
    audio = OmegaConf.to_container(config.get("audio"), resolve=True)
    if not isinstance(audio, dict):
        raise ValueError(f"Config does not define an audio mapping: {config_path}")
    affect = audio.get("affect_analysis")
    if not isinstance(affect, dict) or not bool(affect.get("enabled", False)):
        raise ValueError("audio.affect_analysis.enabled must be true.")
    audio["work_dir"] = str(work_dir.resolve())
    return audio


def _load_transcript(transcript_path: Path) -> Transcript:
    """Load one existing stage-2 transcript from JSON.

    Args:
        transcript_path: Prepared transcript JSON path.

    Returns:
        Transcript domain object containing source-audio metadata.

    Raises:
        ValueError: The JSON document is not an object.
    """
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Transcript must be a JSON object: {transcript_path}")
    return Transcript.from_mapping(payload)


def main(argv: list[str] | None = None) -> int:
    """Analyze the transcript and replace it with the enriched frozen input.

    Args:
        argv: Optional command-line arguments. ``None`` reads the process argv.

    Returns:
        Zero after the analyzer writes its manifest and enriched transcript.

    Raises:
        RuntimeError: The configured providers cannot build an analyzer.
    """
    args = build_parser().parse_args(argv)
    transcript_path = Path(args.transcript_path).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    config_path = Path(args.config_file).expanduser().resolve()
    audio_config = _load_audio_config(config_path, work_dir=work_dir)
    analyzer = build_transcript_affect_analyzer(
        audio_config,
        project_root=project_root,
    )
    if analyzer is None:
        raise RuntimeError("The configured affect analyzer was not created.")
    result = analyzer.analyze_transcript(
        transcript=_load_transcript(transcript_path),
        base_dir=project_root,
    )
    write_transcript_atomic(
        cast(TranscriptPayload, result.transcript.to_payload()),
        transcript_path,
    )
    print(
        json.dumps(
            {
                "transcript_path": str(transcript_path),
                "manifest_path": str(result.manifest_path),
                "profile_path": (
                    str(result.profile_path) if result.profile_path is not None else None
                ),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
