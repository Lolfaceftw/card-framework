"""Create or load the project voice-clone calibration artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from omegaconf import OmegaConf

from card_framework.audio_pipeline.calibration import (
    VoiceCloneCalibration,
    ensure_voice_clone_calibration,
)
from card_framework.shared.paths import DEFAULT_CONFIG_PATH, REPO_ROOT


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse calibration CLI arguments.

    Args:
        argv: Optional raw CLI arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate per-speaker voice-clone speaking rates from the configured "
            "emotion preset catalog."
        )
    )
    parser.add_argument(
        "--speaker-samples-manifest",
        help="Optional path to an existing speaker-sample manifest JSON.",
    )
    parser.add_argument(
        "--transcript-path",
        help="Optional path to a transcript JSON that can be used to generate samples.",
    )
    parser.add_argument(
        "--audio-path",
        help=(
            "Optional audio path used when calibration must bootstrap transcript "
            "and speaker samples from raw audio."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild calibration even when the artifact already exists.",
    )
    return parser.parse_args(argv)


def load_audio_config(project_root: Path) -> dict[str, Any]:
    """Load the repository audio config as a plain dictionary.

    Args:
        project_root: Repository root path.

    Returns:
        Resolved ``audio`` config mapping.

    Raises:
        FileNotFoundError: If the repository config file is missing.
        ValueError: If the config file does not define an ``audio`` mapping.
    """
    config_path = DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    config = OmegaConf.load(config_path)
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict):
        raise ValueError(f"{config_path} did not resolve to a mapping.")
    audio_cfg = resolved.get("audio", {})
    if not isinstance(audio_cfg, dict):
        raise ValueError(f"{config_path} must define an audio mapping.")
    return audio_cfg


def resolve_optional_path(project_root: Path, raw_value: str | None) -> Path | None:
    """Resolve one optional CLI path relative to the repository root.

    Args:
        project_root: Repository root path.
        raw_value: Optional path string from CLI.

    Returns:
        Resolved absolute path or ``None`` when unset.
    """
    if raw_value is None:
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    return candidate.resolve()


def build_calibration_report(
    calibration: VoiceCloneCalibration,
) -> dict[str, Any]:
    """Build a compact JSON-serializable calibration report.

    Args:
        calibration: Loaded or generated calibration artifact.

    Returns:
        Structured report for CLI output.
    """
    presets = {
        preset_name: {
            "emo_text": emo_text,
            "default_wpm": calibration.preset_default_wpm.get(preset_name),
        }
        for preset_name, emo_text in calibration.preset_emo_texts.items()
    }
    speakers = {
        speaker: {
            preset_name: {
                "emo_text": calibration.preset_emo_texts.get(preset_name, ""),
                "wpm": wpm,
            }
            for preset_name, wpm in preset_map.items()
        }
        for speaker, preset_map in calibration.speaker_preset_wpm.items()
    }
    return {
        "artifact_path": str(calibration.artifact_path),
        "generated_at_utc": calibration.generated_at_utc,
        "speaker_samples_manifest_path": str(
            calibration.speaker_samples_manifest_path
        ),
        "calibration_phrases": list(calibration.calibration_phrases),
        "presets": presets,
        "speakers": speakers,
    }


def main(argv: list[str] | None = None) -> int:
    """Run calibration or load the existing artifact.

    Args:
        argv: Optional raw CLI arguments.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    project_root = REPO_ROOT
    audio_cfg = load_audio_config(project_root)
    calibration = ensure_voice_clone_calibration(
        project_root=project_root,
        audio_cfg=audio_cfg,
        speaker_samples_manifest_path=resolve_optional_path(
            project_root,
            args.speaker_samples_manifest,
        ),
        transcript_path=resolve_optional_path(project_root, args.transcript_path),
        audio_path=resolve_optional_path(project_root, args.audio_path),
        force=bool(args.force),
    )
    print(
        json.dumps(
            build_calibration_report(calibration),
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

