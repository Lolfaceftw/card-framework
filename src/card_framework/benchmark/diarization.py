"""CLI entrypoint for speaker diarization benchmark runs."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from omegaconf import DictConfig, OmegaConf

from card_framework.audio_pipeline.factory import build_speaker_diarizer
from card_framework.audio_pipeline.gateways.diarization_common import parse_rttm_file, write_rttm_file
from card_framework.audio_pipeline.runtime import probe_audio_duration_ms, resolve_device
from card_framework.benchmark.artifacts import git_info, utc_now_iso, write_json_with_hash
from card_framework.benchmark.diarization_datasets import (
    DiarizationDatasetPreparationError,
    prepare_ami_manifest,
)
from card_framework.benchmark.diarization_types import (
    DiarizationBenchmarkReport,
    DiarizationBenchmarkSample,
    DiarizationProviderAggregate,
    DiarizationSampleRunResult,
)
from card_framework.shared.paths import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_DIARIZATION_MANIFEST_PATH,
    REPO_ROOT,
)

DEFAULT_DIARIZATION_MANIFEST = str(DEFAULT_DIARIZATION_MANIFEST_PATH)
DEFAULT_AMI_DATA_ROOT = "artifacts/diarization_datasets/ami"


class DiarizationBenchmarkRuntimeError(RuntimeError):
    """Raised when diarization benchmark execution cannot proceed."""


def _commands_executed() -> list[str]:
    """Capture command provenance for report artifacts."""
    return [" ".join(["python", *sys.argv])]


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for diarization benchmark commands."""
    parser = argparse.ArgumentParser(description="Speaker diarization benchmark runner")
    subparsers = parser.add_subparsers(dest="command")

    execute = subparsers.add_parser(
        "execute",
        help="Execute diarization provider comparisons",
    )
    execute.add_argument(
        "--manifest",
        default=DEFAULT_DIARIZATION_MANIFEST,
        help="Path to diarization benchmark manifest JSON",
    )
    execute.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Base application config YAML",
    )
    execute.add_argument(
        "--output-dir",
        default="artifacts/diarization_benchmark",
        help="Output directory root",
    )
    execute.add_argument(
        "--providers",
        default="",
        help=(
            "Comma-separated diarization provider IDs. Defaults to the configured "
            "baseline plus pyannote and NeMo Sortformer alternatives."
        ),
    )
    execute.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Runtime device override for benchmark inference",
    )
    execute.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Optional sample cap override (0 means all manifest samples)",
    )
    execute.add_argument(
        "--collar",
        type=float,
        default=0.0,
        help="DER/JER collar in seconds. Defaults to strict no-collar scoring.",
    )
    execute.add_argument(
        "--skip-overlap",
        action="store_true",
        help="Skip overlapping speech when scoring DER/JER.",
    )

    prepare = subparsers.add_parser(
        "prepare-manifest",
        help="Download public AMI assets and build a diarization manifest",
    )
    prepare.add_argument(
        "--output",
        default=DEFAULT_DIARIZATION_MANIFEST,
        help="Target manifest path",
    )
    prepare.add_argument(
        "--data-root",
        default=DEFAULT_AMI_DATA_ROOT,
        help="Directory used to cache downloaded AMI audio and references",
    )
    prepare.add_argument(
        "--subset",
        default="test",
        choices=["train", "dev", "test"],
        help="AMI subset from BUTSpeechFIT/AMI-diarization-setup",
    )
    prepare.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Optional sample cap for the prepared manifest (0 means all subset meetings)",
    )
    prepare.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload AMI assets even when cached copies already exist",
    )
    return parser


def _load_base_config(config_path: Path) -> DictConfig:
    """Load base application config for provider and logging settings."""
    if not config_path.exists():
        raise DiarizationBenchmarkRuntimeError(
            f"Config path does not exist: {config_path}"
        )
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, DictConfig):
        raise DiarizationBenchmarkRuntimeError(
            "Failed to load diarization benchmark base config."
        )
    return cfg


def _default_provider_ids(active_provider: str) -> list[str]:
    """Return default provider comparison set with stable ordering."""
    provider_ids: list[str] = []
    for provider_id in (
        active_provider,
        "pyannote_community1",
        "nemo_sortformer_streaming",
        "nemo_sortformer_offline",
    ):
        if provider_id and provider_id not in provider_ids:
            provider_ids.append(provider_id)
    return provider_ids


def _resolve_provider_ids(raw: str, *, active_provider: str) -> list[str]:
    """Resolve provider IDs from CLI or default benchmark set."""
    parsed = [item.strip() for item in raw.split(",") if item.strip()]
    return parsed or _default_provider_ids(active_provider)


def _resolve_manifest_path(path_value: str, *, repo_root: Path) -> Path:
    """Resolve one possibly-relative manifest path against repo root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _load_manifest(
    manifest_path: Path,
    *,
    repo_root: Path,
) -> list[DiarizationBenchmarkSample]:
    """Load diarization benchmark manifest samples."""
    if not manifest_path.exists():
        raise DiarizationBenchmarkRuntimeError(
            f"Manifest path does not exist: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DiarizationBenchmarkRuntimeError(
            f"Failed to parse manifest JSON at {manifest_path}: {exc}"
        ) from exc

    samples_payload = payload.get("samples")
    if not isinstance(samples_payload, list) or not samples_payload:
        raise DiarizationBenchmarkRuntimeError(
            "Diarization manifest must contain a non-empty 'samples' list."
        )

    samples: list[DiarizationBenchmarkSample] = []
    for index, entry in enumerate(samples_payload, start=1):
        if not isinstance(entry, dict):
            raise DiarizationBenchmarkRuntimeError(
                f"Manifest sample #{index} must be an object."
            )

        audio_value = entry.get("audio_path") or entry.get("audio_filepath")
        rttm_value = (
            entry.get("reference_rttm_path")
            or entry.get("rttm_path")
            or entry.get("rttm_filepath")
        )
        if not audio_value or not rttm_value:
            raise DiarizationBenchmarkRuntimeError(
                f"Manifest sample #{index} must include audio and RTTM paths."
            )

        audio_path = _resolve_manifest_path(str(audio_value), repo_root=repo_root)
        reference_rttm_path = _resolve_manifest_path(
            str(rttm_value),
            repo_root=repo_root,
        )
        uem_value = entry.get("uem_path") or entry.get("uem_filepath")
        uem_path = (
            str(_resolve_manifest_path(str(uem_value), repo_root=repo_root))
            if uem_value
            else None
        )

        if not audio_path.exists():
            raise DiarizationBenchmarkRuntimeError(
                f"Manifest audio path does not exist: {audio_path}"
            )
        if not reference_rttm_path.exists():
            raise DiarizationBenchmarkRuntimeError(
                f"Manifest RTTM path does not exist: {reference_rttm_path}"
            )

        sample_id = str(entry.get("sample_id", "")).strip() or audio_path.stem
        dataset = str(entry.get("dataset", "unknown")).strip() or "unknown"
        subset = str(entry.get("subset", "")).strip()
        num_speakers_raw = entry.get("num_speakers")
        num_speakers = int(num_speakers_raw) if num_speakers_raw is not None else None
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        samples.append(
            DiarizationBenchmarkSample(
                sample_id=sample_id,
                dataset=dataset,
                subset=subset,
                audio_path=str(audio_path),
                reference_rttm_path=str(reference_rttm_path),
                uem_path=uem_path,
                num_speakers=num_speakers,
                metadata=metadata,
            )
        )
    return samples


def _normalize_argv(argv: list[str] | None) -> list[str]:
    """Preserve legacy execute-style invocations after adding subcommands."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        return ["execute"]
    if raw_argv[0] in {"execute", "prepare-manifest", "-h", "--help"}:
        return raw_argv
    return ["execute", *raw_argv]


def _annotation_from_turns(
    turns: list[Any],
    *,
    uri: str,
) -> Any:
    """Convert diarization turns into a pyannote ``Annotation``."""
    from pyannote.core import Annotation, Segment

    annotation = Annotation(uri=uri)
    for index, turn in enumerate(sorted(turns, key=lambda item: item.start_time_ms)):
        start_seconds = turn.start_time_ms / 1000.0
        end_seconds = turn.end_time_ms / 1000.0
        if end_seconds <= start_seconds:
            continue
        annotation[Segment(start_seconds, end_seconds), f"turn_{index:04d}"] = (
            turn.speaker
        )
    return annotation


def _load_uem_timeline(path: Path, *, uri: str) -> Any | None:
    """Load a pyannote ``Timeline`` from a per-sample UEM file."""
    from pyannote.core import Segment, Timeline

    segments: list[Any] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].startswith("#"):
            continue
        start_seconds = float(parts[2])
        end_seconds = float(parts[3])
        if end_seconds <= start_seconds:
            continue
        segments.append(Segment(start_seconds, end_seconds))

    if not segments:
        return None
    return Timeline(segments=segments, uri=uri)


def _score_turns(
    *,
    reference_turns: list[Any],
    predicted_turns: list[Any],
    sample_id: str,
    uem_path: Path | None,
    collar: float,
    skip_overlap: bool,
) -> tuple[float, float | None]:
    """Score predicted turns against reference RTTM using DER and optional JER."""
    from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

    reference_annotation = _annotation_from_turns(reference_turns, uri=sample_id)
    predicted_annotation = _annotation_from_turns(predicted_turns, uri=sample_id)
    uem_timeline = (
        _load_uem_timeline(uem_path, uri=sample_id) if uem_path is not None else None
    )

    der_metric = DiarizationErrorRate(
        collar=collar,
        skip_overlap=skip_overlap,
    )
    der = float(der_metric(reference_annotation, predicted_annotation, uem=uem_timeline))

    if uem_timeline is None:
        return der, None

    jer_metric = JaccardErrorRate(
        collar=collar,
        skip_overlap=skip_overlap,
    )
    jer = float(jer_metric(reference_annotation, predicted_annotation, uem=uem_timeline))
    return der, jer


def _reset_peak_gpu_memory_if_possible(device: str) -> None:
    """Reset torch CUDA peak-memory accounting when available."""
    if device != "cuda":
        return
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        return


def _peak_gpu_memory_mb(device: str) -> float | None:
    """Read peak torch CUDA memory in MiB when available."""
    if device != "cuda":
        return None
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        torch.cuda.synchronize()
        return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
    except Exception:
        return None


def _mean_or_none(values: list[float | None]) -> float | None:
    """Return rounded mean for non-null values."""
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(statistics.fmean(filtered), 6)


def _aggregate_results(
    *,
    provider_ids: list[str],
    results: list[DiarizationSampleRunResult],
) -> list[DiarizationProviderAggregate]:
    """Aggregate per-provider diarization benchmark metrics."""
    aggregates: list[DiarizationProviderAggregate] = []
    for provider_id in provider_ids:
        provider_results = [
            result for result in results if result.provider_id == provider_id
        ]
        passed_results = [
            result for result in provider_results if result.status == "pass"
        ]
        peak_values = [
            result.peak_gpu_memory_mb
            for result in provider_results
            if result.peak_gpu_memory_mb is not None
        ]
        aggregates.append(
            DiarizationProviderAggregate(
                provider_id=provider_id,
                total_samples=len(provider_results),
                passed_samples=len(passed_results),
                failed_samples=len(provider_results) - len(passed_results),
                mean_der=_mean_or_none([result.der for result in passed_results]),
                mean_jer=_mean_or_none([result.jer for result in passed_results]),
                mean_duration_seconds=_mean_or_none(
                    [result.duration_seconds for result in passed_results]
                ),
                mean_real_time_factor=_mean_or_none(
                    [result.real_time_factor for result in passed_results]
                ),
                max_peak_gpu_memory_mb=(
                    round(max(peak_values), 3) if peak_values else None
                ),
            )
        )
    return aggregates


def _provider_audio_cfg(
    *,
    base_audio_cfg: dict[str, Any],
    provider_id: str,
) -> dict[str, Any]:
    """Build one provider-specific audio config mapping."""
    audio_cfg = copy.deepcopy(base_audio_cfg)
    diarization_cfg = audio_cfg.setdefault("diarization", {})
    diarization_cfg["provider"] = provider_id
    return audio_cfg


def _run_provider_on_sample(
    *,
    diarizer: Any,
    provider_id: str,
    sample: DiarizationBenchmarkSample,
    output_root: Path,
    device: str,
    collar: float,
    skip_overlap: bool,
) -> DiarizationSampleRunResult:
    """Run one diarizer provider on one benchmark sample."""
    sample_started = time.perf_counter()
    work_dir = output_root / "work" / provider_id / sample.sample_id
    predicted_rttm_path = (
        output_root / "predictions" / provider_id / f"{sample.sample_id}.rttm"
    )
    audio_duration_ms = probe_audio_duration_ms(Path(sample.audio_path))

    try:
        _reset_peak_gpu_memory_if_possible(device)
        predicted_turns = diarizer.diarize(
            audio_path=Path(sample.audio_path),
            output_dir=work_dir,
            device=device,
        )
        write_rttm_file(
            predicted_turns,
            predicted_rttm_path,
            uri=sample.sample_id,
        )
        der, jer = _score_turns(
            reference_turns=parse_rttm_file(Path(sample.reference_rttm_path)),
            predicted_turns=predicted_turns,
            sample_id=sample.sample_id,
            uem_path=Path(sample.uem_path) if sample.uem_path else None,
            collar=collar,
            skip_overlap=skip_overlap,
        )
        duration_seconds = round(time.perf_counter() - sample_started, 6)
        audio_duration_seconds = (
            round(audio_duration_ms / 1000.0, 6)
            if audio_duration_ms is not None
            else None
        )
        real_time_factor = (
            round(duration_seconds / audio_duration_seconds, 6)
            if audio_duration_seconds and audio_duration_seconds > 0
            else None
        )
        return DiarizationSampleRunResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            subset=sample.subset,
            provider_id=provider_id,
            status="pass",
            duration_seconds=duration_seconds,
            audio_duration_seconds=audio_duration_seconds,
            real_time_factor=real_time_factor,
            predicted_turn_count=len(predicted_turns),
            predicted_rttm_path=str(predicted_rttm_path),
            der=round(der, 6),
            jer=round(jer, 6) if jer is not None else None,
            peak_gpu_memory_mb=_peak_gpu_memory_mb(device),
        )
    except Exception as exc:
        duration_seconds = round(time.perf_counter() - sample_started, 6)
        audio_duration_seconds = (
            round(audio_duration_ms / 1000.0, 6)
            if audio_duration_ms is not None
            else None
        )
        return DiarizationSampleRunResult(
            sample_id=sample.sample_id,
            dataset=sample.dataset,
            subset=sample.subset,
            provider_id=provider_id,
            status="fail",
            duration_seconds=duration_seconds,
            audio_duration_seconds=audio_duration_seconds,
            real_time_factor=None,
            predicted_turn_count=0,
            predicted_rttm_path=str(predicted_rttm_path),
            peak_gpu_memory_mb=_peak_gpu_memory_mb(device),
            error_message=str(exc),
        )


def execute_command(args: argparse.Namespace) -> int:
    """Execute diarization benchmark command and write report artifacts."""
    repo_root = REPO_ROOT
    config_path = (repo_root / args.config).resolve()
    manifest_path = (repo_root / args.manifest).resolve()
    if not manifest_path.exists():
        raise DiarizationBenchmarkRuntimeError(
            f"Manifest path does not exist: {manifest_path}. "
            "Run `uv run python -m benchmark.diarization prepare-manifest` "
            "to download AMI and create the default manifest."
        )
    base_cfg = _load_base_config(config_path)
    samples = _load_manifest(manifest_path, repo_root=repo_root)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise DiarizationBenchmarkRuntimeError(
            "No diarization benchmark samples selected."
        )

    base_audio_cfg = OmegaConf.to_container(base_cfg.get("audio", {}), resolve=True)
    if not isinstance(base_audio_cfg, dict):
        raise DiarizationBenchmarkRuntimeError(
            "Base config is missing an 'audio' mapping."
        )

    active_provider = str(
        base_audio_cfg.get("diarization", {}).get("provider", "nemo")
    ).strip() or "nemo"
    provider_ids = _resolve_provider_ids(
        str(args.providers),
        active_provider=active_provider,
    )
    resolved_device = resolve_device(str(args.device))
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())

    output_root = (repo_root / args.output_dir / run_id).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[DiarizationSampleRunResult] = []
    failures: list[dict[str, Any]] = []

    for provider_id in provider_ids:
        try:
            diarizer = build_speaker_diarizer(
                _provider_audio_cfg(
                    base_audio_cfg=base_audio_cfg,
                    provider_id=provider_id,
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "provider_id": provider_id,
                    "failure_category": "provider_build_error",
                    "error": str(exc),
                }
            )
            for sample in samples:
                results.append(
                    DiarizationSampleRunResult(
                        sample_id=sample.sample_id,
                        dataset=sample.dataset,
                        subset=sample.subset,
                        provider_id=provider_id,
                        status="fail",
                        duration_seconds=0.0,
                        audio_duration_seconds=None,
                        real_time_factor=None,
                        predicted_turn_count=0,
                        predicted_rttm_path=str(
                            output_root
                            / "predictions"
                            / provider_id
                            / f"{sample.sample_id}.rttm"
                        ),
                        error_message=str(exc),
                    )
                )
            continue

        for sample in samples:
            result = _run_provider_on_sample(
                diarizer=diarizer,
                provider_id=provider_id,
                sample=sample,
                output_root=output_root,
                device=resolved_device,
                collar=float(args.collar),
                skip_overlap=bool(args.skip_overlap),
            )
            if result.status != "pass":
                failures.append(
                    {
                        "provider_id": provider_id,
                        "sample_id": sample.sample_id,
                        "failure_category": "sample_run_error",
                        "error": result.error_message or "Unknown sample failure",
                    }
                )
            results.append(result)

    aggregates = _aggregate_results(provider_ids=provider_ids, results=results)
    git_commit, git_branch = git_info(repo_root)
    report = DiarizationBenchmarkReport(
        run_id=run_id,
        status="completed",
        generated_at_utc=utc_now_iso(),
        git_commit=git_commit,
        git_branch=git_branch,
        manifest_path=str(manifest_path),
        config_path=str(config_path),
        device=resolved_device,
        providers=provider_ids,
        commands_executed=_commands_executed(),
        results=results,
        aggregates=aggregates,
        failures=failures,
    )

    report_path = output_root / "diarization_report.json"
    report_sha256 = write_json_with_hash(report_path, asdict(report))
    verification_path = output_root / "verification.json"
    write_json_with_hash(
        verification_path,
        {
            "status": "not_verified",
            "evidence": {
                "report_json": {
                    "path": str(report_path),
                    "sha256": report_sha256,
                    "producer_command": "uv run python -m benchmark.diarization ...",
                },
                "provenance": {
                    "run_id": run_id,
                    "git_commit": git_commit,
                    "git_branch": git_branch,
                    "generated_at_utc": utc_now_iso(),
                },
                "commands_executed": _commands_executed(),
            },
        },
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "report_path": str(report_path),
                "verification_path": str(verification_path),
                "device": resolved_device,
                "providers": provider_ids,
                "aggregates": [asdict(aggregate) for aggregate in aggregates],
            },
            indent=2,
        )
    )
    return 0


def prepare_manifest_command(args: argparse.Namespace) -> int:
    """Download AMI assets and build a diarization manifest."""
    repo_root = REPO_ROOT
    output_path = (repo_root / args.output).resolve()
    data_root = (repo_root / args.data_root).resolve()
    manifest = prepare_ami_manifest(
        output_path=output_path,
        data_root=data_root,
        subset=str(args.subset),
        num_samples=int(args.num_samples),
        force_download=bool(args.force_download),
    )
    print(
        json.dumps(
            {
                "manifest_path": str(output_path),
                "data_root": str(data_root),
                "subset": str(args.subset),
                "samples": len(manifest["samples"]),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Program entrypoint."""
    parser = _build_parser()
    args = parser.parse_args(_normalize_argv(argv))
    try:
        if args.command == "execute":
            return execute_command(args)
        if args.command == "prepare-manifest":
            return prepare_manifest_command(args)
        parser.print_help()
        return 2
    except (
        DiarizationBenchmarkRuntimeError,
        DiarizationDatasetPreparationError,
    ) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

