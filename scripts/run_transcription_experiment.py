"""Run only the configured ASR transcriber on an existing audio artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any
from uuid import uuid4

from omegaconf import OmegaConf

from card_framework.audio_pipeline.contracts import TimedTextSegment, TranscriptionResult
from card_framework.audio_pipeline.factory import build_audio_to_script_orchestrator
from card_framework.audio_pipeline.runtime import resolve_device, resolve_path
from card_framework.shared.config_loading import load_runtime_config, normalize_runtime_config
from card_framework.shared.env import load_repo_dotenv
from card_framework.shared.events import event_bus
from card_framework.shared.paths import DEFAULT_CONFIG_PATH, REPO_ROOT

_NON_WORD_EDGE_PATTERN = re.compile(r"^\W+|\W+$")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the configured ASR transcriber on an existing audio artifact "
            "and write raw timed text plus seam-oriented summary metrics."
        )
    )
    parser.add_argument(
        "--audio-path",
        required=True,
        help="Audio file to transcribe, relative to the repository root when needed.",
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
        "--output-path",
        help=(
            "Optional explicit output JSON path. Defaults to "
            "'artifacts/granite_experiments/runs/<run-id>/transcription.json'."
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


def _normalize_token(token: str) -> str:
    normalized = _NON_WORD_EDGE_PATTERN.sub("", token.strip().lower())
    return normalized or token.strip().lower()


def _normalized_tokens(text: str) -> list[str]:
    return [_normalize_token(token) for token in text.split() if _normalize_token(token)]


def _adjacent_repeat_run(text: str) -> tuple[str, int]:
    tokens = _normalized_tokens(text)
    best_token = ""
    best_run = 0
    current_token = ""
    current_run = 0
    for token in tokens:
        if token == current_token:
            current_run += 1
        else:
            current_token = token
            current_run = 1
        if current_run > best_run:
            best_token = current_token
            best_run = current_run
    return best_token, best_run


def _boundary_seam_candidates(
    segments: list[TimedTextSegment],
    *,
    max_window_tokens: int = 20,
    min_match_tokens: int = 2,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for index in range(1, len(segments)):
        previous_tokens = _normalized_tokens(segments[index - 1].text)
        current_tokens = _normalized_tokens(segments[index].text)
        if not previous_tokens or not current_tokens:
            continue

        search_start = max(0, len(previous_tokens) - max_window_tokens)
        best_match = 0
        best_phrase: list[str] = []
        for start in range(search_start, len(previous_tokens)):
            match_size = 0
            while (
                start + match_size < len(previous_tokens)
                and match_size < len(current_tokens)
                and previous_tokens[start + match_size] == current_tokens[match_size]
            ):
                match_size += 1
            if match_size > best_match:
                best_match = match_size
                best_phrase = current_tokens[:match_size]

        if best_match < min_match_tokens:
            continue
        candidates.append(
            {
                "previous_index": index - 1,
                "current_index": index,
                "match_tokens": best_match,
                "matched_phrase": " ".join(best_phrase),
                "previous_tail": " ".join(previous_tokens[-max_window_tokens:]),
                "current_head": " ".join(current_tokens[:max_window_tokens]),
            }
        )
    return candidates


def _build_summary(result: TranscriptionResult) -> dict[str, Any]:
    segments = result.segments
    repeat_rows: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        token, run = _adjacent_repeat_run(segment.text)
        if run >= 2:
            repeat_rows.append(
                {
                    "segment_index": index,
                    "token": token,
                    "run_length": run,
                    "text": segment.text,
                }
            )
    seam_candidates = _boundary_seam_candidates(segments)
    return {
        "segment_count": len(segments),
        "word_timestamp_count": len(result.word_timestamps),
        "segments_with_adjacent_repeat_ge2": len(repeat_rows),
        "top_adjacent_repeats": repeat_rows[:20],
        "boundary_seam_candidate_count": len(seam_candidates),
        "top_boundary_seam_candidates": seam_candidates[:20],
    }


def _serialize_segments(segments: list[TimedTextSegment]) -> list[dict[str, Any]]:
    return [
        {
            "start_time_ms": segment.start_time_ms,
            "end_time_ms": segment.end_time_ms,
            "text": segment.text,
        }
        for segment in segments
    ]


def _serialize_words(words: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "word": str(word.word),
            "start_time_ms": int(word.start_time_ms),
            "end_time_ms": int(word.end_time_ms),
        }
        for word in words
    ]


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

    runtime_device = resolve_device(str(audio_cfg.get("device", "auto")))
    orchestrator = build_audio_to_script_orchestrator(audio_cfg)
    transcriber = orchestrator.transcriber

    audio_path = resolve_path(str(args.audio_path), base_dir=repo_root)
    output_path = resolve_path(
        str(
            args.output_path
            or f"artifacts/granite_experiments/runs/{run_id}/transcription.json"
        ),
        base_dir=repo_root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"run_id: {run_id}", flush=True)
    print(f"audio_path: {audio_path}", flush=True)
    print(f"device: {runtime_device}", flush=True)
    print(f"output_path: {output_path}", flush=True)

    subscriptions = ("system_message", "status_message")
    for event_type in subscriptions:
        event_bus.subscribe(event_type, _print_event)
    try:
        result = transcriber.transcribe(
            audio_path=audio_path,
            device=runtime_device,
        )
    finally:
        for event_type in subscriptions:
            event_bus.unsubscribe(event_type, _print_event)
        release_cached_resources = getattr(transcriber, "release_cached_resources", None)
        if callable(release_cached_resources):
            release_cached_resources(device=runtime_device)

    payload = {
        "metadata": {
            "audio_path": str(audio_path),
            "device": runtime_device,
            "language": result.language,
        },
        "summary": _build_summary(result),
        "segments": _serialize_segments(result.segments),
        "word_timestamps": _serialize_words(result.word_timestamps),
        "full_text": " ".join(segment.text for segment in result.segments).strip(),
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
