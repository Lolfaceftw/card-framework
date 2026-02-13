"""Subjective MOS kit writer for voice cloning benchmarks."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np

from benchmarks.voice_clone.constants import (
    CMOS_SCORE_MAX,
    CMOS_SCORE_MIN,
    MOS_SCORE_MAX,
    MOS_SCORE_MIN,
)
from benchmarks.voice_clone.types import ManifestItem, PairScoreRow


def build_mos_instruction_markdown() -> str:
    """Return rating instructions for SMOS/CMOS annotation."""
    return f"""# Voice Cloning Similarity Rating Guide

## Objective
Rate whether generated speech preserves target speaker identity versus reference audio.

## Context
Each pair has one holdout reference recording and one generated clip. Judge
speaker identity similarity only.

## Inputs
- `mos_pairs.csv` metadata with pair ids.
- Audio files under `audio/` for each pair.

## Output contract
- Fill `ratings_template.csv` with one row per pair id.
- Scales:
  - `smos_a`: integer [{MOS_SCORE_MIN}, {MOS_SCORE_MAX}]
  - `smos_b`: integer [{MOS_SCORE_MIN}, {MOS_SCORE_MAX}]
  - `cmos_ab`: integer [{CMOS_SCORE_MIN}, {CMOS_SCORE_MAX}] (positive means A more similar)
  - `more_similar_to_reference`: `A`, `B`, or `TIE`

## Rules
- Use headphones and a quiet room.
- Focus on speaker identity cues (timbre, accent, prosodic identity).
- Ignore waveform displays and metadata labels during judging.

## Examples
- If A is clearly more similar: `smos_a=5`, `smos_b=2`, `cmos_ab=3`, `A`.
- If tied: `smos_a=4`, `smos_b=4`, `cmos_ab=0`, `TIE`.

## Evaluation
- All pair ids rated once.
- No missing required scores.
- Scores stay within allowed ranges.
"""


def write_mos_kit(
    *,
    rows: list[PairScoreRow],
    item_by_id: dict[int, ManifestItem],
    output_dir: Path,
    rng: np.random.Generator,
) -> Path:
    """Write randomized A/B audio packs and rating templates.

    Args:
        rows: Per-item score rows.
        item_by_id: Manifest lookup by item id.
        output_dir: Benchmark run output directory.
        rng: Random generator for deterministic randomization.

    Returns:
        Path to created MOS package directory.
    """
    mos_dir = output_dir / "mos_pack"
    audio_dir = mos_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    randomized_rows = [rows[idx] for idx in rng.permutation(len(rows))]
    pairs_csv_path = mos_dir / "mos_pairs.csv"
    key_csv_path = mos_dir / "mos_key_internal.csv"
    template_csv_path = mos_dir / "ratings_template.csv"
    instructions_path = mos_dir / "mos_instructions.md"
    pair_assignments: list[dict[str, object]] = []

    for pair_index, row in enumerate(randomized_rows):
        pair_id = f"pair_{pair_index:04d}"
        generated_side = "A" if bool(rng.integers(0, 2)) else "B"
        file_a = audio_dir / f"{pair_id}_A.wav"
        file_b = audio_dir / f"{pair_id}_B.wav"
        if generated_side == "A":
            shutil.copy2(row.generated_wav, file_a)
            shutil.copy2(row.reference_wav, file_b)
            a_label = "generated"
            b_label = "reference"
        else:
            shutil.copy2(row.reference_wav, file_a)
            shutil.copy2(row.generated_wav, file_b)
            a_label = "reference"
            b_label = "generated"
        pair_assignments.append(
            {
                "pair_id": pair_id,
                "item_id": row.item_id,
                "speaker_id": row.speaker_id,
                "text": item_by_id[row.item_id].text,
                "audio_a_path": str(file_a.relative_to(mos_dir)),
                "audio_b_path": str(file_b.relative_to(mos_dir)),
                "generated_side": generated_side,
                "a_label": a_label,
                "b_label": b_label,
            }
        )

    with pairs_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "item_id",
                "speaker_id",
                "text",
                "audio_a_path",
                "audio_b_path",
            ],
        )
        writer.writeheader()
        for assignment in pair_assignments:
            writer.writerow(
                {
                    "pair_id": assignment["pair_id"],
                    "item_id": assignment["item_id"],
                    "speaker_id": assignment["speaker_id"],
                    "text": assignment["text"],
                    "audio_a_path": assignment["audio_a_path"],
                    "audio_b_path": assignment["audio_b_path"],
                }
            )

    with key_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "generated_side",
                "a_label",
                "b_label",
            ],
        )
        writer.writeheader()
        for assignment in pair_assignments:
            writer.writerow(
                {
                    "pair_id": assignment["pair_id"],
                    "generated_side": assignment["generated_side"],
                    "a_label": assignment["a_label"],
                    "b_label": assignment["b_label"],
                }
            )

    with template_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rater_id",
                "pair_id",
                "smos_a",
                "smos_b",
                "cmos_ab",
                "more_similar_to_reference",
                "notes",
            ],
        )
        writer.writeheader()
        for pair_index in range(len(randomized_rows)):
            writer.writerow(
                {
                    "rater_id": "",
                    "pair_id": f"pair_{pair_index:04d}",
                    "smos_a": "",
                    "smos_b": "",
                    "cmos_ab": "",
                    "more_similar_to_reference": "",
                    "notes": "",
                }
            )

    instructions_path.write_text(build_mos_instruction_markdown(), encoding="utf-8")
    return mos_dir
