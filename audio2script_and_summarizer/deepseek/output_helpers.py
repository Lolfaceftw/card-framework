"""DeepSeek output shaping and summary-report persistence helpers."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence

from ..speaker_validation import TranscriptSegment, ValidatedDialogueLine
from .constants import (
    DISFLUENCY_PATTERN,
    SUMMARY_LINE_COUNT_MAX,
    SUMMARY_LINE_COUNT_MIN,
    SUMMARY_LINE_WORD_TARGET,
)
from .models import FinalScriptLine, SummaryReport
from .tooling import _count_words_in_text

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def _target_line_bounds(
    segment_count: int,
    word_budget: int | None,
) -> tuple[int, int]:
    """Estimate summary line-count bounds from segment count and word budget."""
    if word_budget is not None and word_budget > 0:
        estimated = max(1, int(round(word_budget / SUMMARY_LINE_WORD_TARGET)))
        lower_bound = max(SUMMARY_LINE_COUNT_MIN, estimated - 4)
        upper_bound = min(SUMMARY_LINE_COUNT_MAX, estimated + 5)
    else:
        estimated = max(1, segment_count // 2)
        lower_bound = max(SUMMARY_LINE_COUNT_MIN, estimated - 8)
        upper_bound = min(SUMMARY_LINE_COUNT_MAX, estimated + 8)
    if lower_bound > upper_bound:
        lower_bound = upper_bound
    return lower_bound, upper_bound

def post_process_script(
    lines: list[ValidatedDialogueLine], voice_sample_dir: str
) -> list[FinalScriptLine]:
    """Inject voice paths and validation metadata into final output schema.

    Args:
        lines: Validated and repaired dialogue lines.
        voice_sample_dir: Directory that stores speaker reference wav files.

    Returns:
        Final JSON-ready script lines.
    """
    final_output: list[FinalScriptLine] = []

    for line in lines:
        voice_path = os.path.join(voice_sample_dir, f"{line.speaker}.wav")
        normalized_voice_path = voice_path.replace("\\", "/")
        if not os.path.exists(voice_path):
            logger.warning(
                "Voice sample not found for speaker '%s': %s",
                line.speaker,
                voice_path,
            )

        final_output.append(
            {
                "speaker": line.speaker,
                "voice_sample": normalized_voice_path,
                "use_emo_text": True,
                "emo_text": line.emo_text,
                "emo_alpha": line.emo_alpha,
                "text": line.text,
                "source_segment_ids": line.source_segment_ids,
                "validation_status": line.validation_status,
                "repair_reason": line.repair_reason,
            }
        )

    return final_output

def _count_words(lines: list[ValidatedDialogueLine]) -> int:
    """Count dialogue words only from line text fields.

    This intentionally excludes JSON structure, keys, and metadata.
    """
    return sum(len(line.text.split()) for line in lines)

def _naturalness_metrics_from_validated_lines(
    lines: list[ValidatedDialogueLine],
) -> dict[str, float | int]:
    """Compute summary-level naturalness metrics for diagnostics reporting."""
    texts = [line.text for line in lines]
    if not texts:
        return {
            "avg_words_per_line": 0.0,
            "short_question_ratio": 0.0,
            "disfluency_count": 0,
            "line_count": 0,
        }
    word_counts = [_count_words_in_text(text) for text in texts]
    short_question_lines = sum(
        1
        for text, words in zip(texts, word_counts)
        if text.strip().endswith("?") and words <= 5
    )
    disfluency_count = sum(len(DISFLUENCY_PATTERN.findall(text)) for text in texts)
    return {
        "avg_words_per_line": float(sum(word_counts)) / float(len(word_counts)),
        "short_question_ratio": float(short_question_lines) / float(len(word_counts)),
        "disfluency_count": disfluency_count,
        "line_count": len(texts),
    }

def _count_words_from_segments(
    segments: Sequence[dict[str, Any] | TranscriptSegment],
) -> int:
    """Count words across transcript segments."""
    total = 0
    for seg in segments:
        if hasattr(seg, "text"):
            text = getattr(seg, "text", "")
        elif isinstance(seg, dict):
            text = seg.get("text", "")
        else:
            text = str(seg)
        total += len(str(text).split())
    return total

def _resolve_summary_report_path(output_path: str, report_path: str | None) -> str:
    """Resolve the diagnostics report path for summarizer output."""
    if report_path and report_path.strip():
        return report_path
    output = Path(output_path)
    return str(output.with_name(f"{output.name}.report.json"))

def _write_summary_report(report_path: str, report: SummaryReport) -> None:
    """Write deterministic summary diagnostics report to disk."""
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)
