"""Run only the stage-1 audio pipeline and write one transcript artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf

from card_framework.audio_pipeline.factory import build_audio_to_script_orchestrator
from card_framework.audio_pipeline.runtime import resolve_device, resolve_path
from card_framework.cli.setup_and_run import build_pipeline_session_artifacts
from card_framework.shared.config_loading import load_runtime_config, normalize_runtime_config
from card_framework.shared.env import load_repo_dotenv
from card_framework.shared.events import event_bus
from card_framework.shared.paths import DEFAULT_CONFIG_PATH, REPO_ROOT


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the stage-1 audio pipeline (separation, ASR, diarization, "
            "alignment, transcript write) using the repo config plus optional "
            "Hydra-style overrides."
        )
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Source audio file path, relative to the repository root when needed.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Runtime YAML config path. Defaults to the repo's active config.",
    )
    parser.add_argument(
        "--device",
        help="Optional explicit runtime device: auto, cpu, or cuda.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional explicit run id. Defaults to a fresh UUID4.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional OmegaConf dotlist override. Repeat for multiple values.",
    )
    return parser.parse_args(argv)


def _has_override_key(overrides: list[str], key: str) -> bool:
    prefix = f"{key}="
    return any(item.strip().startswith(prefix) for item in overrides)


def _print_event(message: str) -> None:
    print(message, flush=True)


def _build_audio_metadata(audio_cfg: dict[str, Any]) -> dict[str, str]:
    separation_cfg = audio_cfg.get("separation", {}) if isinstance(audio_cfg, dict) else {}
    asr_cfg = audio_cfg.get("asr", {}) if isinstance(audio_cfg, dict) else {}
    diarization_cfg = audio_cfg.get("diarization", {}) if isinstance(audio_cfg, dict) else {}
    return {
        "separator_model": str(separation_cfg.get("model", "htdemucs")),
        "transcriber_model": str(
            asr_cfg.get("model", "ibm-granite/granite-4.0-1b-speech")
        ),
        "diarizer_backend": str(diarization_cfg.get("provider", "nemo")),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = REPO_ROOT.resolve()
    load_repo_dotenv(repo_root=repo_root, search_cwd=True)

    run_id = str(args.run_id).strip() if args.run_id else str(uuid4())
    session_artifacts = build_pipeline_session_artifacts(run_id=run_id)
    overrides = [item for item in args.override if str(item).strip()]

    cfg = load_runtime_config(Path(args.config_path).resolve())
    merged_cfg = cfg
    if overrides:
        merged_cfg = normalize_runtime_config(
            OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        )
    if not _has_override_key(overrides, "audio.audio_path"):
        OmegaConf.update(
            merged_cfg,
            "audio.audio_path",
            str(resolve_path(str(args.audio_path), base_dir=repo_root)),
            merge=False,
        )
    if args.device and not _has_override_key(overrides, "audio.device"):
        OmegaConf.update(
            merged_cfg,
            "audio.device",
            str(args.device).strip(),
            merge=False,
        )
    if not _has_override_key(overrides, "audio.work_dir"):
        OmegaConf.update(
            merged_cfg,
            "audio.work_dir",
            session_artifacts["work_dir"],
            merge=False,
        )
    if not _has_override_key(overrides, "audio.output_transcript_path"):
        OmegaConf.update(
            merged_cfg,
            "audio.output_transcript_path",
            session_artifacts["transcript_path"],
            merge=False,
        )
    merged_cfg = normalize_runtime_config(merged_cfg)
    resolved_cfg = OmegaConf.to_container(merged_cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        raise ValueError("Resolved runtime config must be a mapping.")

    audio_cfg = resolved_cfg.get("audio", {})
    if not isinstance(audio_cfg, dict):
        raise ValueError("Resolved audio config must be a mapping.")

    runtime_device = resolve_device(str(audio_cfg.get("device", "auto")))
    audio_orchestrator = build_audio_to_script_orchestrator(audio_cfg)
    eta_cfg = audio_cfg.get("eta", {})
    eta_store = "artifacts/audio_stage/eta_history.sqlite3"
    if isinstance(eta_cfg, dict) and str(eta_cfg.get("store_path", "")).strip():
        eta_store = str(eta_cfg["store_path"])
    audio_orchestrator.eta_profile_path = resolve_path(eta_store, base_dir=repo_root)

    input_audio_path = resolve_path(str(audio_cfg.get("audio_path", "")), base_dir=repo_root)
    output_transcript_path = resolve_path(
        str(audio_cfg.get("output_transcript_path", session_artifacts["transcript_path"])),
        base_dir=repo_root,
    )
    work_dir = resolve_path(
        str(audio_cfg.get("work_dir", session_artifacts["work_dir"])),
        base_dir=repo_root,
    )

    print(f"run_id: {run_id}", flush=True)
    print(f"audio_path: {input_audio_path}", flush=True)
    print(f"device: {runtime_device}", flush=True)
    print(f"work_dir: {work_dir}", flush=True)
    print(f"transcript_path: {output_transcript_path}", flush=True)

    subscriptions = ("system_message", "status_message")
    for event_type in subscriptions:
        event_bus.subscribe(event_type, _print_event)
    try:
        result = audio_orchestrator.run(
            input_audio_path=input_audio_path,
            output_transcript_path=output_transcript_path,
            work_dir=work_dir,
            device=runtime_device,
            metadata_overrides=_build_audio_metadata(audio_cfg),
        )
    finally:
        for event_type in subscriptions:
            event_bus.unsubscribe(event_type, _print_event)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "transcript_path": str(result.transcript_path),
                "segment_count": result.segment_count,
                "warnings": result.warnings,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
