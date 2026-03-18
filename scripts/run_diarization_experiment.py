"""Run only speaker diarization on a fixed audio artifact and summarize it."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf

from card_framework.audio_pipeline.alignment import align_segments_with_speakers
from card_framework.audio_pipeline.contracts import DiarizationTurn, TimedTextSegment
from card_framework.audio_pipeline.factory import build_speaker_diarizer
from card_framework.audio_pipeline.io import build_transcript_payload, write_transcript_atomic
from card_framework.audio_pipeline.runtime import resolve_device, resolve_path, utc_now_iso
from card_framework.shared.config_loading import load_runtime_config, normalize_runtime_config
from card_framework.shared.env import load_repo_dotenv
from card_framework.shared.events import event_bus
from card_framework.shared.paths import DEFAULT_CONFIG_PATH, REPO_ROOT


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the configured diarizer on an existing audio artifact and "
            "optionally project the result back onto an existing timed transcript "
            "for a quick speaker-labeled preview."
        )
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Audio file to diarize, relative to the repository root when needed.",
    )
    parser.add_argument(
        "--base-transcript-path",
        help=(
            "Optional existing transcript JSON whose segment timings will be reused "
            "to build a diarization-preview transcript."
        ),
    )
    parser.add_argument(
        "--output-transcript-path",
        help=(
            "Optional explicit preview transcript output path. Defaults to "
            "'preview_transcript.json' inside the experiment output directory."
        ),
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
        "--output-dir",
        help=(
            "Optional experiment output directory. Defaults to "
            "'artifacts/diarization_experiments/runs/<run-id>'."
        ),
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


def _load_timed_segments(transcript_path: Path) -> list[TimedTextSegment]:
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    raw_segments = payload.get("segments", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_segments, list):
        raise ValueError("Transcript payload must contain a top-level segments list.")

    timed_segments: list[TimedTextSegment] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        timed_segments.append(
            TimedTextSegment(
                start_time_ms=int(item["start_time"]),
                end_time_ms=int(item["end_time"]),
                text=text,
            )
        )
    if not timed_segments:
        raise ValueError(f"Transcript '{transcript_path}' did not contain any timed text.")
    return timed_segments


def _summarize_turns(turns: list[DiarizationTurn]) -> dict[str, Any]:
    turn_counts = Counter(turn.speaker for turn in turns)
    speaker_durations_ms = Counter()
    short_turn_counts = {
        "lte_250ms": 0,
        "lte_500ms": 0,
        "lte_1000ms": 0,
    }
    for turn in turns:
        duration_ms = max(0, turn.end_time_ms - turn.start_time_ms)
        speaker_durations_ms[turn.speaker] += duration_ms
        if duration_ms <= 250:
            short_turn_counts["lte_250ms"] += 1
        if duration_ms <= 500:
            short_turn_counts["lte_500ms"] += 1
        if duration_ms <= 1000:
            short_turn_counts["lte_1000ms"] += 1

    return {
        "turn_count": len(turns),
        "unique_speaker_count": len(turn_counts),
        "speaker_turn_counts": dict(sorted(turn_counts.items())),
        "speaker_durations_seconds": {
            speaker: round(duration_ms / 1000, 3)
            for speaker, duration_ms in sorted(speaker_durations_ms.items())
        },
        "short_turn_counts": short_turn_counts,
    }


def _build_preview_metadata(
    *,
    audio_path: Path,
    runtime_device: str,
    diarizer_backend: str,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "source_audio_path": str(audio_path),
        "device": runtime_device,
        "generated_at_utc": utc_now_iso(),
        "warnings": warnings,
        "diarizer_backend": diarizer_backend,
        "alignment_backend": "segment_overlap_preview",
        "forced_alignment_enabled": False,
        "forced_alignment_satisfied": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    repo_root = REPO_ROOT.resolve()
    load_repo_dotenv(repo_root=repo_root, search_cwd=True)

    run_id = str(args.run_id).strip() if args.run_id else str(uuid4())
    overrides = [item for item in args.override if str(item).strip()]

    cfg = load_runtime_config(Path(args.config_path).resolve())
    merged_cfg = cfg
    if overrides:
        merged_cfg = normalize_runtime_config(
            OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
        )
    if args.device and not _has_override_key(overrides, "audio.device"):
        OmegaConf.update(
            merged_cfg,
            "audio.device",
            str(args.device).strip(),
            merge=False,
        )
    merged_cfg = normalize_runtime_config(merged_cfg)
    resolved_cfg = OmegaConf.to_container(merged_cfg, resolve=True)
    if not isinstance(resolved_cfg, dict):
        raise ValueError("Resolved runtime config must be a mapping.")

    audio_cfg = resolved_cfg.get("audio", {})
    if not isinstance(audio_cfg, dict):
        raise ValueError("Resolved audio config must be a mapping.")

    diarization_cfg = audio_cfg.get("diarization", {})
    if not isinstance(diarization_cfg, dict):
        diarization_cfg = {}

    runtime_device = resolve_device(str(audio_cfg.get("device", "auto")))
    diarizer = build_speaker_diarizer(audio_cfg)

    audio_path = resolve_path(str(args.audio_path), base_dir=repo_root)
    output_dir = resolve_path(
        str(args.output_dir or f"artifacts/diarization_experiments/runs/{run_id}"),
        base_dir=repo_root,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_transcript_path: Path | None = None
    if args.base_transcript_path:
        base_transcript_path = resolve_path(str(args.base_transcript_path), base_dir=repo_root)
    preview_transcript_path: Path | None = None
    if base_transcript_path is not None:
        preview_transcript_path = resolve_path(
            str(args.output_transcript_path or (output_dir / "preview_transcript.json")),
            base_dir=repo_root,
        )

    print(f"run_id: {run_id}", flush=True)
    print(f"audio_path: {audio_path}", flush=True)
    print(f"device: {runtime_device}", flush=True)
    print(f"output_dir: {output_dir}", flush=True)
    if base_transcript_path is not None:
        print(f"base_transcript_path: {base_transcript_path}", flush=True)
    if preview_transcript_path is not None:
        print(f"preview_transcript_path: {preview_transcript_path}", flush=True)

    subscriptions = ("system_message", "status_message")
    for event_type in subscriptions:
        event_bus.subscribe(event_type, _print_event)
    try:
        turns = diarizer.diarize(
            audio_path=audio_path,
            output_dir=output_dir,
            device=runtime_device,
        )
    finally:
        for event_type in subscriptions:
            event_bus.unsubscribe(event_type, _print_event)

    warnings: list[str] = []
    if len({turn.speaker for turn in turns}) <= 1:
        warnings.append(
            "Diarization produced only one speaker label for this experiment audio."
        )

    preview_segment_count = 0
    preview_speaker_count = 0
    if base_transcript_path is not None and preview_transcript_path is not None:
        base_segments = _load_timed_segments(base_transcript_path)
        preview_segments = align_segments_with_speakers(
            asr_segments=base_segments,
            diarization_turns=turns,
            default_speaker=str(audio_cfg.get("default_speaker", "SPEAKER_00")),
            merge_gap_ms=int(audio_cfg.get("merge_gap_ms", 800)),
        )
        preview_segment_count = len(preview_segments)
        preview_speaker_count = len({segment.speaker for segment in preview_segments})
        payload = build_transcript_payload(
            segments=preview_segments,
            metadata=_build_preview_metadata(
                audio_path=audio_path,
                runtime_device=runtime_device,
                diarizer_backend=str(diarization_cfg.get("provider", "nemo")),
                warnings=warnings,
            ),
        )
        write_transcript_atomic(payload, preview_transcript_path)

    print(
        json.dumps(
            {
                "run_id": run_id,
                "audio_path": str(audio_path),
                "output_dir": str(output_dir),
                "preview_transcript_path": (
                    str(preview_transcript_path) if preview_transcript_path is not None else None
                ),
                "warnings": warnings,
                "summary": _summarize_turns(turns),
                "preview_segment_count": preview_segment_count,
                "preview_speaker_count": preview_speaker_count,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
