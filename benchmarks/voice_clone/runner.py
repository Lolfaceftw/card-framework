"""CLI runner for IndexTTS2 voice cloning benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
from pathlib import Path

import numpy as np

from audio2script_and_summarizer.logging_utils import LOG_FILE_ENV_VAR, configure_logging
from audio2script_and_summarizer.stage3_voice import IndexTTS2Engine
from benchmarks.voice_clone.constants import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    RESEARCH_CUTOFF_DATE,
)
from benchmarks.voice_clone.embedder import WavLMSpeakerEmbedder
from benchmarks.voice_clone.manifest import load_manifest
from benchmarks.voice_clone.metrics import (
    bootstrap_ci95_eer,
    bootstrap_ci95_mean,
    collect_asv_pairs,
    compute_eer,
    make_pair_rows,
    write_pair_scores_csv,
)
from benchmarks.voice_clone.mos import write_mos_kit
from benchmarks.voice_clone.types import (
    BenchmarkArtifacts,
    ManifestItem,
    SpeakerEmbedderProtocol,
    TTSEngineProtocol,
)
from benchmarks.voice_clone.utils import now_utc_compact, slugify, utc_iso8601_now

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def choose_benchmark_device(requested_device: str | None) -> str:
    """Resolve device string from user input and runtime availability."""
    if requested_device:
        return requested_device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def synthesize_generated_audio(
    *,
    items: list[ManifestItem],
    generated_dir: Path,
    tts_engine: TTSEngineProtocol,
) -> dict[int, Path]:
    """Run TTS generation for all manifest rows."""
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated_paths: dict[int, Path] = {}
    for item in items:
        generated_path = generated_dir / f"{item.item_id:05d}_{slugify(item.speaker_id)}.wav"
        logger.info(
            "benchmark_tts_generate item_id=%d speaker=%s prompt=%s",
            item.item_id,
            item.speaker_id,
            item.prompt_wav,
        )
        tts_engine.infer(
            spk_audio_prompt=str(item.prompt_wav),
            text=item.text,
            output_path=str(generated_path),
            emo_alpha=item.emo_alpha,
            use_emo_text=item.use_emo_text,
            emo_text=item.emo_text,
            use_random=False,
            verbose=False,
        )
        generated_paths[item.item_id] = generated_path.resolve()
    return generated_paths


def run_benchmark(
    *,
    manifest_path: Path,
    cfg_path: Path,
    model_dir: Path,
    device: str,
    output_dir: Path,
    max_items: int | None,
    seed: int,
    prepare_mos_kit: bool,
    tts_engine: TTSEngineProtocol | None = None,
    embedder: SpeakerEmbedderProtocol | None = None,
) -> BenchmarkArtifacts:
    """Run end-to-end IndexTTS2 voice-cloning benchmark.

    Args:
        manifest_path: Benchmark manifest JSON path.
        cfg_path: IndexTTS2 config path.
        model_dir: IndexTTS2 checkpoint directory.
        device: Runtime device string.
        output_dir: Output directory for artifacts.
        max_items: Optional row cap for smoke runs.
        seed: Random seed for deterministic operations.
        prepare_mos_kit: Whether to emit subjective MOS package.
        tts_engine: Optional injected TTS engine for tests.
        embedder: Optional injected speaker embedder for tests.

    Returns:
        Paths to generated benchmark artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_log = output_dir / "run.log"
    os.environ[LOG_FILE_ENV_VAR] = str(run_log)
    configure_logging(
        level=os.getenv("LOG_LEVEL", "INFO"),
        logs_dir=output_dir,
        component="benchmarks.voice_clone",
        enable_console=True,
    )

    logger.info(
        "benchmark_start manifest=%s cfg=%s model_dir=%s output_dir=%s",
        manifest_path,
        cfg_path,
        model_dir,
        output_dir,
    )
    items = load_manifest(manifest_path, max_items=max_items)
    unique_speakers = sorted({item.speaker_id for item in items})
    logger.info(
        "benchmark_manifest_loaded items=%d speaker_count=%d",
        len(items),
        len(unique_speakers),
    )

    runtime_tts_engine: TTSEngineProtocol = tts_engine or IndexTTS2Engine(
        cfg_path=cfg_path,
        model_dir=model_dir,
        device=device,
    )
    runtime_embedder: SpeakerEmbedderProtocol = embedder or WavLMSpeakerEmbedder(
        device=device
    )
    generated_dir = output_dir / "generated"
    generated_paths = synthesize_generated_audio(
        items=items,
        generated_dir=generated_dir,
        tts_engine=runtime_tts_engine,
    )

    unique_reference_paths = sorted({item.reference_wav for item in items})
    reference_embeddings = {
        path: runtime_embedder.embed(path) for path in unique_reference_paths
    }
    generated_embeddings = {
        item.item_id: runtime_embedder.embed(generated_paths[item.item_id])
        for item in items
    }

    pair_rows = make_pair_rows(
        items=items,
        generated_paths=generated_paths,
        generated_embeddings=generated_embeddings,
        reference_embeddings=reference_embeddings,
    )
    pair_scores_csv = output_dir / "pair_scores.csv"
    write_pair_scores_csv(pair_rows, pair_scores_csv)

    cosine_values = np.array([row.same_item_cosine for row in pair_rows], dtype=np.float64)
    top1_values = np.array([float(row.top1_correct) for row in pair_rows], dtype=np.float64)
    asv_scores, asv_labels, asv_positive_pairs, asv_negative_pairs = collect_asv_pairs(
        items=items,
        generated_embeddings=generated_embeddings,
        reference_embeddings=reference_embeddings,
    )
    asv_eer = compute_eer(asv_scores, asv_labels)

    bootstrap_rng = np.random.default_rng(seed)
    cosine_ci95 = bootstrap_ci95_mean(
        cosine_values, rng=bootstrap_rng, sample_count=DEFAULT_BOOTSTRAP_SAMPLES
    )
    top1_ci95 = bootstrap_ci95_mean(
        top1_values, rng=bootstrap_rng, sample_count=DEFAULT_BOOTSTRAP_SAMPLES
    )
    asv_eer_ci95 = bootstrap_ci95_eer(
        asv_scores,
        asv_labels,
        rng=bootstrap_rng,
        sample_count=DEFAULT_BOOTSTRAP_SAMPLES,
    )

    item_by_id = {item.item_id: item for item in items}
    mos_dir: Path | None = None
    if prepare_mos_kit:
        mos_rng = np.random.default_rng(seed)
        mos_dir = write_mos_kit(
            rows=pair_rows,
            item_by_id=item_by_id,
            output_dir=output_dir,
            rng=mos_rng,
        )

    metrics_json = output_dir / "metrics_summary.json"
    summary_payload: dict[str, object] = {
        "run_timestamp_utc": utc_iso8601_now(),
        "research_cutoff_date": RESEARCH_CUTOFF_DATE,
        "manifest_path": str(manifest_path),
        "config": {
            "cfg_path": str(cfg_path),
            "model_dir": str(model_dir),
            "device": device,
            "max_items": max_items,
            "seed": seed,
            "prepare_mos_kit": prepare_mos_kit,
        },
        "dataset": {
            "item_count": len(items),
            "speaker_count": len(unique_speakers),
            "speakers": unique_speakers,
        },
        "metrics": {
            "ss_cosine_mean": float(np.mean(cosine_values)),
            "ss_cosine_std": float(statistics.pstdev(cosine_values.tolist()))
            if cosine_values.size > 1
            else 0.0,
            "ss_cosine_ci95": list(cosine_ci95) if cosine_ci95 is not None else None,
            "top1_speaker_acc": float(np.mean(top1_values)),
            "top1_speaker_ci95": list(top1_ci95) if top1_ci95 is not None else None,
            "asv_eer": asv_eer,
            "asv_eer_ci95": list(asv_eer_ci95) if asv_eer_ci95 is not None else None,
            "asv_positive_pairs": asv_positive_pairs,
            "asv_negative_pairs": asv_negative_pairs,
        },
        "artifacts": {
            "pair_scores_csv": str(pair_scores_csv),
            "generated_dir": str(generated_dir),
            "mos_dir": str(mos_dir) if mos_dir is not None else None,
            "run_log": str(run_log),
        },
    }
    metrics_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    logger.info(
        "benchmark_complete items=%d ss_mean=%.4f top1=%.4f asv_eer=%s",
        len(items),
        summary_payload["metrics"]["ss_cosine_mean"],
        summary_payload["metrics"]["top1_speaker_acc"],
        "null" if asv_eer is None else f"{asv_eer:.4f}",
    )

    return BenchmarkArtifacts(
        output_dir=output_dir,
        generated_dir=generated_dir,
        pair_scores_csv=pair_scores_csv,
        metrics_json=metrics_json,
        run_log=run_log,
        mos_dir=mos_dir,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for benchmark command."""
    parser = argparse.ArgumentParser(
        description="Benchmark IndexTTS2 voice cloning similarity for CARD pipeline."
    )
    parser.add_argument("--manifest", required=True, help="Path to benchmark manifest JSON.")
    parser.add_argument(
        "--cfg-path",
        default="checkpoints/config.yaml",
        help="IndexTTS2 config path.",
    )
    parser.add_argument(
        "--model-dir",
        default="checkpoints",
        help="IndexTTS2 checkpoint directory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Runtime device (for example cpu, cuda, cuda:0). Defaults to auto.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to benchmarks/runs/<timestamp>.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional cap on number of manifest rows.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Deterministic random seed.",
    )
    parser.add_argument(
        "--prepare-mos-kit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate MOS subjective rating package.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Console log level.",
    )
    return parser.parse_args()


def main() -> int:
    """Run benchmark CLI command."""
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    cfg_path = Path(args.cfg_path).resolve()
    model_dir = Path(args.model_dir).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file does not exist: {manifest_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"IndexTTS2 config does not exist: {cfg_path}")
    if not model_dir.exists():
        raise FileNotFoundError(f"IndexTTS2 model directory does not exist: {model_dir}")

    os.environ["LOG_LEVEL"] = str(args.log_level)
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else (Path("benchmarks") / "runs" / now_utc_compact()).resolve()
    )
    run_benchmark(
        manifest_path=manifest_path,
        cfg_path=cfg_path,
        model_dir=model_dir,
        device=choose_benchmark_device(args.device),
        output_dir=output_dir,
        max_items=args.max_items,
        seed=int(args.seed),
        prepare_mos_kit=bool(args.prepare_mos_kit),
    )
    return 0
