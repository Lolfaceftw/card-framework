"""Rebuild stage-3 artifacts from persisted live-draft state for SPoRC runs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from card_framework.audio_pipeline.live_draft_voice_clone import (
    LiveDraftVoiceCloneState,
    load_live_draft_voice_clone_state,
)
from card_framework.audio_pipeline.voice_clone_orchestrator import (
    merge_audio_artifacts_to_wav,
)
from card_framework.shared.summary_xml import SummaryTurn, serialize_summary_turns


@dataclass(slots=True, frozen=True)
class RecoveryResult:
    """Describe one recovered run root."""

    run_root: Path
    state_path: Path
    summary_path: Path
    manifest_path: Path
    merged_audio_path: Path
    turn_count: int


def _utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Recover summary.xml and stage-3 voice-clone artifacts from a saved "
            "live-draft state file."
        )
    )
    parser.add_argument(
        "--run-root",
        action="append",
        required=True,
        help=(
            "Absolute run-root path to recover. Repeat this flag to recover "
            "multiple runs in one invocation."
        ),
    )
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--merge-audio-codec",
        default="pcm_s24le",
        help="FFmpeg codec used for merged stage-3 WAV output.",
    )
    parser.add_argument(
        "--merge-timeout-seconds",
        type=int,
        default=300,
        help="Timeout used for merged stage-3 WAV output.",
    )
    return parser.parse_args()


def _find_state_path(run_root: Path) -> Path:
    """Return the newest live-draft state file under one run root."""
    voice_clone_dir = (run_root / "audio_stage" / "voice_clone").resolve()
    candidates = sorted(
        voice_clone_dir.glob("live_draft_*.state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No live-draft state file found under {voice_clone_dir}."
        )
    return candidates[0]


def _build_summary_turns(state: LiveDraftVoiceCloneState) -> tuple[list[SummaryTurn], list[dict[str, object]]]:
    """Convert persisted live-draft state into summary turns and manifest rows."""
    turns: list[SummaryTurn] = []
    manifest_rows: list[dict[str, object]] = []
    for index, item in enumerate(state.current_snapshot, start=1):
        turn_id = str(item.get("turn_id", "")).strip()
        if not turn_id:
            raise ValueError(f"Snapshot item {index} is missing turn_id.")
        record = state.turn_audio.get(turn_id)
        if record is None:
            raise ValueError(
                f"Snapshot item {index} references missing turn audio for turn_id={turn_id}."
            )
        if not record.output_audio_path.exists():
            raise FileNotFoundError(
                f"Recovered turn audio is missing on disk: {record.output_audio_path}"
            )
        turns.append(
            SummaryTurn(
                speaker=record.speaker,
                text=record.text,
                emo_preset=record.emo_preset,
            )
        )
        manifest_rows.append(
            {
                "turn_index": index,
                "turn_id": record.turn_id,
                "speaker": record.speaker,
                "text": record.text,
                "emo_preset": record.emo_preset,
                "reference_audio_path": None,
                "output_audio_path": str(record.output_audio_path),
                "duration_ms": record.duration_ms,
                "word_count": record.word_count,
                "actual_wpm": round(record.actual_wpm, 6),
            }
        )
    if not turns:
        raise ValueError("Live-draft state snapshot is empty.")
    return turns, manifest_rows


def _speaker_sample_path_by_speaker(manifest_path: Path) -> dict[str, str]:
    """Return the longest available reference sample path for each speaker."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    speaker_to_best_sample: dict[str, tuple[float, str]] = {}
    for item in payload.get("artifacts", []):
        if not isinstance(item, dict):
            continue
        speaker = str(item.get("speaker", "")).strip()
        output_audio_path = str(item.get("output_audio_path", "")).strip()
        if not speaker or not output_audio_path:
            continue
        duration_ms = float(item.get("duration_ms") or 0.0)
        existing = speaker_to_best_sample.get(speaker)
        if existing is None or duration_ms >= existing[0]:
            speaker_to_best_sample[speaker] = (duration_ms, output_audio_path)
    return {
        speaker: sample_path
        for speaker, (_, sample_path) in speaker_to_best_sample.items()
    }


def recover_run_root(
    run_root: Path,
    *,
    merge_audio_codec: str,
    merge_timeout_seconds: int,
) -> RecoveryResult:
    """Recover one run root from its persisted live-draft state."""
    resolved_run_root = run_root.resolve()
    state_path = _find_state_path(resolved_run_root)
    state = load_live_draft_voice_clone_state(state_path)
    turns, manifest_rows = _build_summary_turns(state)
    sample_paths = _speaker_sample_path_by_speaker(state.speaker_samples_manifest_path)

    summary_xml = serialize_summary_turns(turns)
    summary_path = (resolved_run_root / "summary.xml").resolve()
    summary_path.write_text(summary_xml + "\n", encoding="utf-8")

    voice_clone_dir = (resolved_run_root / "audio_stage" / "voice_clone").resolve()
    voice_clone_dir.mkdir(parents=True, exist_ok=True)
    merged_audio_path = (voice_clone_dir / "voice_cloned.wav").resolve()
    merge_audio_artifacts_to_wav(
        artifact_paths=[
            Path(str(row["output_audio_path"])).resolve() for row in manifest_rows
        ],
        output_path=merged_audio_path,
        audio_codec=merge_audio_codec,
        timeout_seconds=merge_timeout_seconds,
    )

    for row in manifest_rows:
        speaker = str(row["speaker"])
        row["reference_audio_path"] = sample_paths.get(speaker)

    manifest_path = (voice_clone_dir / "manifest.json").resolve()
    manifest_payload = {
        "generated_at_utc": _utc_now_iso(),
        "speaker_samples_manifest_path": str(state.speaker_samples_manifest_path),
        "artifact_count": len(manifest_rows),
        "merge_segments": True,
        "merged_output_audio_path": str(merged_audio_path),
        "artifacts": manifest_rows,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    return RecoveryResult(
        run_root=resolved_run_root,
        state_path=state_path.resolve(),
        summary_path=summary_path,
        manifest_path=manifest_path,
        merged_audio_path=merged_audio_path,
        turn_count=len(manifest_rows),
    )


def main() -> int:
    """Recover one or more run roots and print a JSON summary."""
    args = _parse_args()
    results = [
        recover_run_root(
            Path(value),
            merge_audio_codec=args.merge_audio_codec,
            merge_timeout_seconds=args.merge_timeout_seconds,
        )
        for value in args.run_root
    ]
    payload = {
        "generated_at_utc": _utc_now_iso(),
        "results": [
            {
                "run_root": str(item.run_root),
                "state_path": str(item.state_path),
                "summary_path": str(item.summary_path),
                "manifest_path": str(item.manifest_path),
                "merged_audio_path": str(item.merged_audio_path),
                "turn_count": item.turn_count,
            }
            for item in results
        ],
    }
    if args.json_output:
        Path(args.json_output).resolve().write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
