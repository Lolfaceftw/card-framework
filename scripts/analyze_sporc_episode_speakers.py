"""Analyze one prepared SPoRC episode for overlap-heavy speaker labeling.

This helper is intentionally memory-safe for the large long-form SPoRC slice:

- it reads the compact manifest header once,
- streams sample rows from the manifest sidecar until it finds the requested row,
- loads only that sample's transcript JSON.

The goal is to quantify how much of one episode is ambiguous before we decide
whether to force a single speaker label, show multiple speakers, or suppress
speaker labels for overlap-heavy turns.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from statistics import median
from typing import Any

from card_framework.benchmark.summarizer_critic.sporc.config import (
    DEFAULT_PREPARED_FULL_ROOT,
    PreparedBenchmarkSample,
)
from card_framework.benchmark.summarizer_critic.sporc.dataset import (
    iter_prepared_dataset_samples,
)

DEFAULT_MANIFEST_PATH = (
    DEFAULT_PREPARED_FULL_ROOT / "benchmark_longform_15k_manifest.compact.json"
)
_WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


@dataclass(slots=True, frozen=True)
class EpisodeSegment:
    """One prepared transcript segment with overlap-aware speaker metadata."""

    index: int
    speaker: str
    speaker_candidates: tuple[str, ...]
    start_time: int
    end_time: int
    text: str

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_time - self.start_time)

    @property
    def word_count(self) -> int:
        return len(_WORD_PATTERN.findall(self.text))

    @property
    def is_uncertain(self) -> bool:
        return len(self.speaker_candidates) > 1


def _dedupe_preserve_order(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = str(value).strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return tuple(deduped)


def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def _segment_candidates(raw_segment: dict[str, Any]) -> tuple[str, ...]:
    primary = str(raw_segment.get("speaker", "UNKNOWN")).strip() or "UNKNOWN"
    overlap = _coerce_string_list(raw_segment.get("overlap_speakers"))
    return _dedupe_preserve_order([primary] + overlap)


def _load_sample(
    *,
    manifest_path: Path,
    selection_number: int,
    sample_id: str,
    episode_substring: str,
) -> tuple[int, PreparedBenchmarkSample]:
    target_selection = max(1, int(selection_number))
    sample_id_casefold = sample_id.casefold().strip()
    episode_casefold = episode_substring.casefold().strip()

    for position, sample in enumerate(
        iter_prepared_dataset_samples(manifest_path),
        start=1,
    ):
        if sample_id_casefold and sample.sample_id.casefold() == sample_id_casefold:
            return position, sample
        if episode_casefold:
            haystacks = (
                sample.episode_key,
                str(sample.metadata.get("episode_title", "")),
                sample.sample_id,
            )
            if any(episode_casefold in value.casefold() for value in haystacks):
                return position, sample
        if not sample_id_casefold and not episode_casefold and position == target_selection:
            return position, sample

    requested = []
    if sample_id_casefold:
        requested.append(f"sample_id='{sample_id}'")
    if episode_casefold:
        requested.append(f"episode_substring='{episode_substring}'")
    if not requested:
        requested.append(f"selection_number={target_selection}")
    raise RuntimeError(
        "Unable to find a prepared sample matching "
        + ", ".join(requested)
        + f" in '{manifest_path.resolve()}'."
    )


def _load_segments(transcript_path: Path) -> list[EpisodeSegment]:
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    raw_segments = payload.get("segments", [])
    if not isinstance(raw_segments, list):
        raise RuntimeError(f"Transcript '{transcript_path}' does not contain a segments list.")

    segments: list[EpisodeSegment] = []
    for index, raw_segment in enumerate(raw_segments, start=1):
        if not isinstance(raw_segment, dict):
            continue
        text = str(raw_segment.get("text", "")).strip()
        if not text:
            continue
        segments.append(
            EpisodeSegment(
                index=index,
                speaker=str(raw_segment.get("speaker", "UNKNOWN")).strip() or "UNKNOWN",
                speaker_candidates=_segment_candidates(raw_segment),
                start_time=int(raw_segment.get("start_time", 0)),
                end_time=int(raw_segment.get("end_time", raw_segment.get("start_time", 0))),
                text=text,
            )
        )
    if not segments:
        raise RuntimeError(f"Transcript '{transcript_path}' did not contain usable segments.")
    return segments


def _share(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return round((float(numerator) / float(denominator)) * 100.0, 2)


def _top_pairs(segments: list[EpisodeSegment], *, limit: int) -> list[dict[str, Any]]:
    pair_counts = Counter(
        ",".join(segment.speaker_candidates)
        for segment in segments
        if segment.is_uncertain
    )
    return [
        {"speaker_pair": pair, "count": count}
        for pair, count in pair_counts.most_common(limit)
    ]


def _build_examples(
    segments: list[EpisodeSegment],
    *,
    limit: int,
    text_limit: int,
) -> list[dict[str, Any]]:
    uncertain_segments = [segment for segment in segments if segment.is_uncertain]
    examples: list[dict[str, Any]] = []
    for segment in uncertain_segments[:limit]:
        previous_speaker = None
        next_speaker = None
        if segment.index > 1 and segment.index - 2 < len(segments):
            previous_speaker = segments[segment.index - 2].speaker
        if segment.index < len(segments):
            next_speaker = segments[segment.index].speaker
        text = segment.text
        if len(text) > text_limit:
            text = text[:text_limit].rstrip() + "..."
        examples.append(
            {
                "segment_index": segment.index,
                "speaker": segment.speaker,
                "speaker_candidates": list(segment.speaker_candidates),
                "word_count": segment.word_count,
                "duration_ms": segment.duration_ms,
                "previous_speaker": previous_speaker,
                "next_speaker": next_speaker,
                "text": text,
            }
        )
    return examples


def _classify_recommendation(segments: list[EpisodeSegment]) -> dict[str, Any]:
    uncertain = [segment for segment in segments if segment.is_uncertain]
    if not uncertain:
        return {
            "speaker_assignment_policy": "single_speaker_ok",
            "display_policy": "single_label",
            "prompt_policy": "current_flattened_prompt_is_safe",
            "logic_changes": [],
        }

    total_words = sum(segment.word_count for segment in segments)
    uncertain_words = sum(segment.word_count for segment in uncertain)
    short_uncertain = [
        segment
        for segment in uncertain
        if segment.duration_ms <= 500 and segment.word_count <= 3
    ]
    medium_uncertain = [
        segment
        for segment in uncertain
        if segment.duration_ms <= 1500 and segment.word_count <= 8
    ]
    long_uncertain = [
        segment
        for segment in uncertain
        if segment not in medium_uncertain
    ]

    if _share(uncertain_words, total_words) >= 20.0 or _share(len(long_uncertain), len(uncertain)) >= 50.0:
        return {
            "speaker_assignment_policy": "avoid_forced_single_speaker_reassignment",
            "display_policy": "join_overlap_candidates_with_comma_or_hide_label",
            "prompt_policy": "render_ambiguous_turns_with_all_candidates_instead_of_primary_only",
            "logic_changes": [
                "Do not flatten ambiguous turns to one speaker in transcript prompts.",
                "When overlap_speakers exists, display all speaker candidates such as 'SPEAKER_00, SPEAKER_02'.",
                "Only attempt transcript-side reassignment for very short ambiguous spans after an explicit safety gate.",
            ],
        }

    if short_uncertain:
        return {
            "speaker_assignment_policy": "allow_small_heuristic_cleanup_only",
            "display_policy": "prefer_multi_label_fallback",
            "prompt_policy": "preserve_candidates_for_remaining_ambiguous_turns",
            "logic_changes": [
                "Keep multi-speaker labels by default.",
                "Optionally relabel only tiny ambiguous spans at or below 500ms and 3 words when neighboring context agrees.",
            ],
        }

    return {
        "speaker_assignment_policy": "keep_overlap_visible",
        "display_policy": "join_overlap_candidates_with_comma",
        "prompt_policy": "preserve_candidates_in_prompt",
        "logic_changes": [
            "Keep ambiguous turns as multi-speaker instead of guessing.",
        ],
    }


def build_report(
    *,
    manifest_path: Path,
    selection_number: int,
    sample_id: str,
    episode_substring: str,
    example_count: int,
    text_limit: int,
) -> dict[str, Any]:
    position, sample = _load_sample(
        manifest_path=manifest_path,
        selection_number=selection_number,
        sample_id=sample_id,
        episode_substring=episode_substring,
    )
    transcript_path = Path(sample.transcript_path).resolve()
    segments = _load_segments(transcript_path)

    uncertain = [segment for segment in segments if segment.is_uncertain]
    certain = [segment for segment in segments if not segment.is_uncertain]
    total_duration_ms = sum(segment.duration_ms for segment in segments)
    uncertain_duration_ms = sum(segment.duration_ms for segment in uncertain)
    total_words = sum(segment.word_count for segment in segments)
    uncertain_words = sum(segment.word_count for segment in uncertain)

    uncertain_word_counts = [segment.word_count for segment in uncertain]
    uncertain_durations = [segment.duration_ms for segment in uncertain]

    report = {
        "manifest_path": str(manifest_path.resolve()),
        "selection_number": position,
        "sample": {
            "sample_id": sample.sample_id,
            "episode_title": str(sample.metadata.get("episode_title", "")),
            "episode_key": sample.episode_key,
            "target_seconds": sample.target_seconds,
            "transcript_path": str(transcript_path),
        },
        "transcript_stats": {
            "segment_count": len(segments),
            "certain_segment_count": len(certain),
            "uncertain_segment_count": len(uncertain),
            "uncertain_segment_pct": _share(len(uncertain), len(segments)),
            "word_count": total_words,
            "uncertain_word_count": uncertain_words,
            "uncertain_word_pct": _share(uncertain_words, total_words),
            "duration_ms": total_duration_ms,
            "uncertain_duration_ms": uncertain_duration_ms,
            "uncertain_duration_pct": _share(uncertain_duration_ms, total_duration_ms),
        },
        "uncertain_segment_profile": {
            "median_words": median(uncertain_word_counts) if uncertain_word_counts else 0,
            "median_duration_ms": median(uncertain_durations) if uncertain_durations else 0,
            "short_500ms_3words_or_less": sum(
                1
                for segment in uncertain
                if segment.duration_ms <= 500 and segment.word_count <= 3
            ),
            "medium_1500ms_8words_or_less": sum(
                1
                for segment in uncertain
                if segment.duration_ms <= 1500 and segment.word_count <= 8
            ),
            "long_or_dense_uncertain_segments": sum(
                1
                for segment in uncertain
                if segment.duration_ms > 1500 or segment.word_count > 8
            ),
            "top_uncertain_speaker_pairs": _top_pairs(segments, limit=5),
        },
        "recommendation": _classify_recommendation(segments),
        "examples": _build_examples(
            segments,
            limit=max(1, example_count),
            text_limit=max(40, text_limit),
        ),
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one prepared SPoRC transcript row from the compact 15k-word "
            "manifest without loading the full dataset into memory."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Prepared SPoRC manifest header to stream from.",
    )
    parser.add_argument(
        "--selection-number",
        type=int,
        default=1,
        help="1-based row number inside the manifest sidecar when no matcher is provided.",
    )
    parser.add_argument(
        "--sample-id",
        default="",
        help="Exact sample_id override.",
    )
    parser.add_argument(
        "--episode-substring",
        default="",
        help="Case-insensitive substring match against title, episode_key, or sample_id.",
    )
    parser.add_argument(
        "--example-count",
        type=int,
        default=5,
        help="How many uncertain example segments to print.",
    )
    parser.add_argument(
        "--text-limit",
        type=int,
        default=220,
        help="Maximum number of characters to print per example.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON.",
    )
    return parser


def _print_human_report(report: dict[str, Any]) -> None:
    sample = report["sample"]
    stats = report["transcript_stats"]
    uncertain_profile = report["uncertain_segment_profile"]
    recommendation = report["recommendation"]

    print("SPoRC episode speaker-overlap analysis")
    print(f"selection_number: {report['selection_number']}")
    print(f"sample_id: {sample['sample_id']}")
    print(f"episode_title: {sample['episode_title']}")
    print(f"episode_key: {sample['episode_key']}")
    print(f"transcript_path: {sample['transcript_path']}")
    print()
    print("transcript_stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()
    print("uncertain_segment_profile:")
    for key, value in uncertain_profile.items():
        print(f"  {key}: {value}")
    print()
    print("recommendation:")
    for key, value in recommendation.items():
        print(f"  {key}: {value}")
    print()
    print("examples:")
    for example in report["examples"]:
        print(
            "  - "
            f"segment_index={example['segment_index']} "
            f"speaker={example['speaker']} "
            f"candidates={example['speaker_candidates']} "
            f"words={example['word_count']} "
            f"duration_ms={example['duration_ms']} "
            f"prev={example['previous_speaker']} "
            f"next={example['next_speaker']}"
        )
        print(f"    text={example['text']}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    report = build_report(
        manifest_path=args.manifest_path,
        selection_number=args.selection_number,
        sample_id=args.sample_id,
        episode_substring=args.episode_substring,
        example_count=args.example_count,
        text_limit=args.text_limit,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_human_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
