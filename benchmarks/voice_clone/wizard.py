"""Interactive setup wizard for IndexTTS2 voice-clone benchmarking."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from audio2script_and_summarizer.logging_utils import LOG_FILE_ENV_VAR, configure_logging
from audio2script_and_summarizer.stage3_voice import (
    discover_summary_json_files,
    load_summary_entries,
    resolve_voice_sample_path,
)
from benchmarks.voice_clone.constants import DEFAULT_ASR_MODEL_SIZE
from benchmarks.voice_clone.holdout_wizard import HoldoutSplitResult, run_holdout_wizard
from benchmarks.voice_clone.runner import choose_benchmark_device, run_benchmark
from benchmarks.voice_clone.utils import now_utc_compact

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_WIZARD_SEARCH_ROOT: Final[Path] = Path(".")
DEFAULT_MANIFEST_DIR: Final[Path] = Path("benchmarks") / "manifests"


@dataclass(slots=True, frozen=True)
class ManifestBuildResult:
    """Represent generated manifest details."""

    manifest_path: Path
    item_count: int
    used_prompt_dir: Path | None
    used_holdout_dir: Path | None
    fallback_prompt_count: int
    fallback_reference_count: int


def _prompt_text(prompt: str, *, default: str | None = None) -> str:
    """Prompt user for text input with optional default value."""
    suffix = f" [{default}]" if default else ""
    raw = input(f"{prompt}{suffix}: ").strip()
    if raw:
        return raw
    return default or ""


def _prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    """Prompt user for a yes/no answer."""
    default_token = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{default_token}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please answer 'y' or 'n'.")


def _prompt_int(
    prompt: str,
    *,
    default: int | None = None,
    allow_empty: bool = False,
) -> int | None:
    """Prompt user for integer input."""
    default_text = "" if default is None else str(default)
    while True:
        raw = _prompt_text(prompt, default=default_text if default is not None else None)
        if not raw and allow_empty:
            return None
        try:
            return int(raw)
        except ValueError:
            print("Please enter a valid integer.")


def _pick_summary_json(search_root: Path) -> Path:
    """Select a summary JSON path from discovered files or manual input."""
    candidates = discover_summary_json_files(search_root)
    if not candidates:
        manual = _prompt_text(
            "No summary JSON found. Enter summary JSON path manually",
            default="",
        )
        path = Path(manual).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Summary JSON not found: {path}")
        return path

    print("\nDiscovered summary JSON files (newest first):")
    for index, candidate in enumerate(candidates[:10], start=1):
        print(f"  {index}. {candidate}")
    selected_index = _prompt_int(
        "Select summary number",
        default=1,
        allow_empty=False,
    )
    assert selected_index is not None
    bounded_index = max(1, min(selected_index, len(candidates)))
    return candidates[bounded_index - 1]


def build_manifest_from_summary(
    *,
    summary_json_path: Path,
    manifest_path: Path,
    prompt_dir: Path | None = None,
    holdout_dir: Path | None = None,
    max_items: int | None = None,
) -> ManifestBuildResult:
    """Build a benchmark manifest from Stage 2 summary JSON.

    Args:
        summary_json_path: Stage 2 summary JSON path.
        manifest_path: Destination manifest path.
        prompt_dir: Optional directory containing generated prompt WAVs.
        holdout_dir: Optional directory containing holdout reference WAVs.
        max_items: Optional cap on number of rows.

    Returns:
        Generated manifest metadata.
    """
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be >= 1 when provided.")

    entries = load_summary_entries(summary_json_path.resolve())
    summary_dir = summary_json_path.resolve().parent
    selected_entries = entries if max_items is None else entries[: max(0, max_items)]
    rows: list[dict[str, object]] = []
    fallback_prompt_count = 0
    fallback_reference_count = 0

    for entry in selected_entries:
        resolved_source_prompt_path = resolve_voice_sample_path(entry.voice_sample, summary_dir)
        prompt_path = resolved_source_prompt_path
        if prompt_dir is not None:
            candidate_prompt = (prompt_dir / resolved_source_prompt_path.name).resolve()
            if candidate_prompt.exists():
                prompt_path = candidate_prompt
            else:
                fallback_prompt_count += 1

        reference_path = prompt_path
        if holdout_dir is not None:
            candidate_reference = (holdout_dir / resolved_source_prompt_path.name).resolve()
            if candidate_reference.exists():
                reference_path = candidate_reference
            else:
                fallback_reference_count += 1

        rows.append(
            {
                "speaker_id": entry.speaker,
                "prompt_wav": str(prompt_path),
                "reference_wav": str(reference_path),
                "text": entry.text,
                "use_emo_text": entry.use_emo_text,
                "emo_text": entry.emo_text,
                "emo_alpha": entry.emo_alpha,
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return ManifestBuildResult(
        manifest_path=manifest_path.resolve(),
        item_count=len(rows),
        used_prompt_dir=prompt_dir.resolve() if prompt_dir is not None else None,
        used_holdout_dir=holdout_dir.resolve() if holdout_dir is not None else None,
        fallback_prompt_count=fallback_prompt_count,
        fallback_reference_count=fallback_reference_count,
    )


def run_wizard() -> int:
    """Run interactive wizard to create and optionally execute benchmark.

    Returns:
        Process-style return code.
    """
    print("IndexTTS2 Benchmark Setup Wizard")
    print("- This creates a manifest from your summary JSON.")
    print("- prompt_wav: cloning prompt audio.")
    print("- reference_wav: scoring target audio (holdout if available).")
    print("")

    search_root_raw = _prompt_text(
        "Search root for *_summary.json files",
        default=str(DEFAULT_WIZARD_SEARCH_ROOT),
    )
    search_root = Path(search_root_raw).expanduser().resolve()
    summary_json_path = _pick_summary_json(search_root)
    print(f"\nSelected summary: {summary_json_path}")

    split_result: HoldoutSplitResult | None = None
    if _prompt_yes_no(
        "Run holdout split wizard to auto-generate prompt and holdout WAVs?",
        default=False,
    ):
        split_result = run_holdout_wizard(summary_json_path=summary_json_path)

    prompt_dir: Path | None = None
    holdout_dir: Path | None = None
    if split_result is not None:
        prompt_dir = split_result.prompt_dir
        holdout_dir = split_result.holdout_dir
        print(f"Using generated prompt dir: {prompt_dir}")
        print(f"Using generated holdout dir: {holdout_dir}")
    else:
        use_holdout = _prompt_yes_no(
            "Use a separate holdout directory for reference_wav?",
            default=False,
        )
        if use_holdout:
            holdout_raw = _prompt_text("Holdout WAV directory path", default="")
            holdout_dir = Path(holdout_raw).expanduser().resolve()
            if not holdout_dir.exists():
                raise FileNotFoundError(f"Holdout directory not found: {holdout_dir}")

    manifest_name_default = f"voice_clone_manifest_{now_utc_compact()}.json"
    manifest_raw = _prompt_text(
        "Manifest output path",
        default=str((DEFAULT_MANIFEST_DIR / manifest_name_default)),
    )
    manifest_path = Path(manifest_raw).expanduser().resolve()
    max_items = _prompt_int(
        "Limit number of rows (blank = all)",
        default=None,
        allow_empty=True,
    )
    result = build_manifest_from_summary(
        summary_json_path=summary_json_path,
        manifest_path=manifest_path,
        prompt_dir=prompt_dir,
        holdout_dir=holdout_dir,
        max_items=max_items,
    )
    print(f"\nManifest created: {result.manifest_path}")
    print(f"Rows: {result.item_count}")
    if result.used_prompt_dir is not None:
        print(f"Prompt dir: {result.used_prompt_dir}")
        if result.fallback_prompt_count > 0:
            print(
                "Warning: "
                f"{result.fallback_prompt_count} row(s) fell back to summary voice_sample "
                "because matching prompt files were not found."
            )
    if result.used_holdout_dir is not None:
        print(f"Holdout dir: {result.used_holdout_dir}")
        if result.fallback_reference_count > 0:
            print(
                "Warning: "
                f"{result.fallback_reference_count} row(s) fell back to prompt_wav "
                "because matching holdout files were not found."
            )

    if not _prompt_yes_no("Run benchmark now?", default=True):
        print("\nRun later with:")
        print(
            "uv run python benchmarks/benchmark_indextts2_voice_clone.py "
            f"--manifest \"{result.manifest_path}\""
        )
        return 0

    cfg_path = Path(
        _prompt_text(
            "IndexTTS2 config path",
            default="voice-cloner-and-interjector/checkpoints/config.yaml",
        )
    ).expanduser().resolve()
    model_dir = Path(
        _prompt_text(
            "IndexTTS2 model directory",
            default="voice-cloner-and-interjector/checkpoints",
        )
    ).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"IndexTTS2 config not found: {cfg_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"IndexTTS2 model directory not found: {model_dir}")
    device = choose_benchmark_device(
        _prompt_text("Device (cpu/cuda/cuda:0, blank=auto)", default="")
    )
    output_dir_default = str(Path("benchmarks") / "runs" / now_utc_compact())
    output_dir = Path(_prompt_text("Benchmark output dir", default=output_dir_default)).resolve()
    run_max_items = _prompt_int("Benchmark max-items (blank = no cap)", allow_empty=True)
    prepare_mos_kit = _prompt_yes_no("Generate MOS rating kit?", default=True)
    allow_overlap = _prompt_yes_no(
        "Allow prompt_wav == reference_wav overlap (exploratory only)?",
        default=False,
    )
    asr_model_raw = _prompt_text(
        "ASR model for WER/CER (blank disables text metrics)",
        default=DEFAULT_ASR_MODEL_SIZE,
    )
    require_text_metrics = _prompt_yes_no(
        "Require WER/CER metrics (fail run if unavailable)?",
        default=False,
    )
    seed_value = _prompt_int("Random seed", default=1337, allow_empty=False)
    assert seed_value is not None
    log_level = _prompt_text("Log level", default="INFO").upper()
    os.environ["LOG_LEVEL"] = log_level

    output_dir.mkdir(parents=True, exist_ok=True)
    wizard_log_path = output_dir / "wizard.log"
    os.environ[LOG_FILE_ENV_VAR] = str(wizard_log_path)
    configure_logging(
        level=log_level,
        logs_dir=output_dir,
        component="benchmarks.voice_clone_wizard",
        enable_console=True,
    )
    logger.info(
        "wizard_run_benchmark manifest=%s output_dir=%s",
        result.manifest_path,
        output_dir,
    )

    artifacts = run_benchmark(
        manifest_path=result.manifest_path,
        cfg_path=cfg_path,
        model_dir=model_dir,
        device=device,
        output_dir=output_dir,
        max_items=run_max_items,
        seed=seed_value,
        prepare_mos_kit=prepare_mos_kit,
        allow_prompt_reference_overlap=allow_overlap,
        asr_model=asr_model_raw.strip() or None,
        require_text_metrics=require_text_metrics,
    )
    print("\nBenchmark completed.")
    print(f"Metrics: {artifacts.metrics_json}")
    print(f"Pair scores: {artifacts.pair_scores_csv}")
    if artifacts.mos_dir is not None:
        print(f"MOS kit: {artifacts.mos_dir}")
    return 0
