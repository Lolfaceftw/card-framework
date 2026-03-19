"""Isolated embedding-model memory profiler and autotuner."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
from typing import Any

from omegaconf import OmegaConf

from card_framework.shared.paths import REPO_ROOT

_DEFAULT_RESULTS_ROOT = (
    REPO_ROOT / "artifacts" / "embedding_memory_profile"
).resolve()
_DEFAULT_CONFIG_PATH = (REPO_ROOT / "conf" / "config.yaml").resolve()
_DEFAULT_EMBEDDING_PROFILE = "qwen3_4b"
_POWERSHELL_PREFIX: tuple[str, ...] = (
    "powershell",
    "-NoProfile",
    "-Command",
)
_POWERSHELL_GPU_PROCESS_MEMORY_SCRIPT = (
    "$dedicated = Get-Counter '\\GPU Process Memory(*)\\Dedicated Usage' | "
    "Select-Object -ExpandProperty CounterSamples; "
    "$shared = Get-Counter '\\GPU Process Memory(*)\\Shared Usage' | "
    "Select-Object -ExpandProperty CounterSamples; "
    "$rows = @{}; "
    "foreach ($sample in $dedicated) { "
    "if ($sample.InstanceName -match 'pid_(\\d+)_') { "
    "$rows[$sample.InstanceName] = [pscustomobject]@{ "
    "instance_name = $sample.InstanceName; "
    "pid = [int]$matches[1]; "
    "dedicated_bytes = [int64]$sample.CookedValue; "
    "shared_bytes = [int64]0 "
    "} "
    "} "
    "}; "
    "foreach ($sample in $shared) { "
    "if ($sample.InstanceName -match 'pid_(\\d+)_') { "
    "if (-not $rows.ContainsKey($sample.InstanceName)) { "
    "$rows[$sample.InstanceName] = [pscustomobject]@{ "
    "instance_name = $sample.InstanceName; "
    "pid = [int]$matches[1]; "
    "dedicated_bytes = [int64]0; "
    "shared_bytes = [int64]$sample.CookedValue "
    "} "
    "} else { "
    "$rows[$sample.InstanceName].shared_bytes = [int64]$sample.CookedValue "
    "} "
    "} "
    "}; "
    "@($rows.Values) | ConvertTo-Json -Compress"
)
_POWERSHELL_GPU_ADAPTER_MEMORY_SCRIPT = (
    "$dedicated = Get-Counter '\\GPU Adapter Memory(*)\\Dedicated Usage' | "
    "Select-Object -ExpandProperty CounterSamples; "
    "$shared = Get-Counter '\\GPU Adapter Memory(*)\\Shared Usage' | "
    "Select-Object -ExpandProperty CounterSamples; "
    "$rows = @{}; "
    "foreach ($sample in $dedicated) { "
    "$rows[$sample.InstanceName] = [pscustomobject]@{ "
    "instance_name = $sample.InstanceName; "
    "dedicated_bytes = [int64]$sample.CookedValue; "
    "shared_bytes = [int64]0 "
    "} "
    "}; "
    "foreach ($sample in $shared) { "
    "if (-not $rows.ContainsKey($sample.InstanceName)) { "
    "$rows[$sample.InstanceName] = [pscustomobject]@{ "
    "instance_name = $sample.InstanceName; "
    "dedicated_bytes = [int64]0; "
    "shared_bytes = [int64]$sample.CookedValue "
    "} "
    "} else { "
    "$rows[$sample.InstanceName].shared_bytes = [int64]$sample.CookedValue "
    "} "
    "}; "
    "@($rows.Values) | ConvertTo-Json -Compress"
)
_MAX_SEQ_LENGTH_CANDIDATES = (None, 4096, 2048, 1024, 768, 512, 384, 256)
_ATTENTION_IMPLEMENTATION_CANDIDATES = (None, "sdpa", "eager")


@dataclass(slots=True, frozen=True)
class EmbeddingProfileConfig:
    """Resolved embedding profile used as the autotune baseline."""

    profile_id: str
    model_name: str
    device: str
    torch_dtype: str
    batch_size: int


@dataclass(slots=True, frozen=True)
class ExperimentConfig:
    """One isolated embedding-memory profiling configuration."""

    experiment_id: str
    model_name: str
    device: str
    torch_dtype: str
    batch_size: int
    texts_per_batch: int
    words_per_text: int
    encode_passes: int
    hold_seconds: float
    max_seq_length: int | None = None
    attention_implementation: str | None = None


@dataclass(slots=True, frozen=True)
class ProcessMemorySample:
    """One GPU process-memory counter row."""

    instance_name: str
    pid: int
    dedicated_bytes: int
    shared_bytes: int


@dataclass(slots=True, frozen=True)
class AggregatedProcessMemory:
    """Aggregated GPU process memory across all GPU instances for one PID."""

    pid: int
    dedicated_bytes: int
    shared_bytes: int

    @property
    def dedicated_mebibytes(self) -> float:
        """Return dedicated usage in MiB."""
        return self.dedicated_bytes / float(1024 * 1024)

    @property
    def shared_mebibytes(self) -> float:
        """Return shared usage in MiB."""
        return self.shared_bytes / float(1024 * 1024)


@dataclass(slots=True, frozen=True)
class AdapterMemorySample:
    """One GPU adapter-memory counter row."""

    instance_name: str
    dedicated_bytes: int
    shared_bytes: int


@dataclass(slots=True, frozen=True)
class AggregatedAdapterMemory:
    """Aggregated GPU adapter memory across all visible adapters."""

    dedicated_bytes: int
    shared_bytes: int

    @property
    def dedicated_mebibytes(self) -> float:
        """Return dedicated usage in MiB."""
        return self.dedicated_bytes / float(1024 * 1024)

    @property
    def shared_mebibytes(self) -> float:
        """Return shared usage in MiB."""
        return self.shared_bytes / float(1024 * 1024)


@dataclass(slots=True, frozen=True)
class WorkerSummary:
    """Structured payload emitted by the worker child."""

    load_seconds: float
    encode_seconds: float
    max_seq_length: int | None
    actual_token_count: int
    torch_peak_allocated_mb: float | None
    torch_peak_reserved_mb: float | None
    status: str
    error: str = ""


@dataclass(slots=True, frozen=True)
class ExperimentResult:
    """One completed profiling result with external and internal telemetry."""

    experiment: ExperimentConfig
    exit_code: int
    load_seconds: float
    encode_seconds: float
    actual_token_count: int
    baseline_dedicated_mb: float
    baseline_shared_mb: float
    peak_dedicated_mb: float
    peak_shared_mb: float
    peak_total_gpu_used_mb: float
    peak_raw_shared_mb: float
    torch_peak_allocated_mb: float | None
    torch_peak_reserved_mb: float | None
    status: str
    stdout_path: str
    stderr_path: str
    telemetry_path: str
    error: str = ""

    @property
    def succeeded(self) -> bool:
        """Return whether the experiment satisfied all profiler constraints."""
        return self.status == "passed"


def _utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _run_subprocess_command(command: tuple[str, ...], timeout_seconds: float) -> str:
    """Run one command and return stdout with safe UTF-8 decoding."""
    completed = subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def _parse_json_rows(output: str) -> tuple[dict[str, object], ...]:
    """Parse PowerShell JSON output into a normalized tuple of rows."""
    text = output.strip()
    if not text:
        return ()
    payload = json.loads(text)
    if isinstance(payload, dict):
        return (payload,)
    if isinstance(payload, list):
        return tuple(row for row in payload if isinstance(row, dict))
    raise ValueError("Expected JSON object or array payload.")


def parse_process_memory_rows(output: str) -> tuple[ProcessMemorySample, ...]:
    """Parse PowerShell GPU process-memory JSON rows."""
    rows = _parse_json_rows(output)
    samples: list[ProcessMemorySample] = []
    for row in rows:
        try:
            instance_name = str(row.get("instance_name", ""))
            pid = int(row.get("pid", 0))
            dedicated_bytes = int(row.get("dedicated_bytes", 0))
            shared_bytes = int(row.get("shared_bytes", 0))
        except (TypeError, ValueError):
            continue
        if pid <= 0:
            continue
        samples.append(
            ProcessMemorySample(
                instance_name=instance_name,
                pid=pid,
                dedicated_bytes=max(0, dedicated_bytes),
                shared_bytes=max(0, shared_bytes),
            )
        )
    return tuple(samples)


def aggregate_process_memory_for_pid(
    samples: tuple[ProcessMemorySample, ...],
    pid: int,
) -> AggregatedProcessMemory:
    """Aggregate dedicated/shared GPU memory for one PID."""
    dedicated_bytes = 0
    shared_bytes = 0
    for sample in samples:
        if sample.pid != pid:
            continue
        dedicated_bytes += sample.dedicated_bytes
        shared_bytes += sample.shared_bytes
    return AggregatedProcessMemory(
        pid=pid,
        dedicated_bytes=dedicated_bytes,
        shared_bytes=shared_bytes,
    )


def _collect_process_gpu_memory(timeout_seconds: float) -> tuple[ProcessMemorySample, ...]:
    """Collect current GPU process-memory counters from Windows perf counters."""
    output = _run_subprocess_command(
        (*_POWERSHELL_PREFIX, _POWERSHELL_GPU_PROCESS_MEMORY_SCRIPT),
        timeout_seconds,
    )
    return parse_process_memory_rows(output)


def parse_adapter_memory_rows(output: str) -> tuple[AdapterMemorySample, ...]:
    """Parse PowerShell GPU adapter-memory JSON rows."""
    rows = _parse_json_rows(output)
    samples: list[AdapterMemorySample] = []
    for row in rows:
        try:
            instance_name = str(row.get("instance_name", ""))
            dedicated_bytes = int(row.get("dedicated_bytes", 0))
            shared_bytes = int(row.get("shared_bytes", 0))
        except (TypeError, ValueError):
            continue
        samples.append(
            AdapterMemorySample(
                instance_name=instance_name,
                dedicated_bytes=max(0, dedicated_bytes),
                shared_bytes=max(0, shared_bytes),
            )
        )
    return tuple(samples)


def aggregate_adapter_memory(
    samples: tuple[AdapterMemorySample, ...],
) -> AggregatedAdapterMemory:
    """Aggregate dedicated/shared GPU memory across all adapter rows."""
    return AggregatedAdapterMemory(
        dedicated_bytes=sum(sample.dedicated_bytes for sample in samples),
        shared_bytes=sum(sample.shared_bytes for sample in samples),
    )


def _collect_adapter_gpu_memory(timeout_seconds: float) -> tuple[AdapterMemorySample, ...]:
    """Collect current GPU adapter-memory counters from Windows perf counters."""
    output = _run_subprocess_command(
        (*_POWERSHELL_PREFIX, _POWERSHELL_GPU_ADAPTER_MEMORY_SCRIPT),
        timeout_seconds,
    )
    return parse_adapter_memory_rows(output)


def _resolve_results_root(raw_results_root: str) -> Path:
    """Resolve a timestamped results root when the caller omits one."""
    if raw_results_root.strip():
        return (REPO_ROOT / raw_results_root).resolve()
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return (_DEFAULT_RESULTS_ROOT / session_id).resolve()


def _resolve_embedding_profile(
    *,
    config_path: Path,
    profile_id: str | None,
) -> EmbeddingProfileConfig:
    """Resolve one embedding profile from config."""
    cfg = OmegaConf.load(config_path)
    resolved_profile_id = profile_id or str(cfg.selection.embedding or "").strip()
    if not resolved_profile_id:
        resolved_profile_id = _DEFAULT_EMBEDDING_PROFILE
    profile_node = cfg.profiles.embedding.get(resolved_profile_id)
    if profile_node is None:
        raise KeyError(f"Unknown embedding profile: {resolved_profile_id}")
    profile_payload = OmegaConf.to_container(profile_node, resolve=True)
    if not isinstance(profile_payload, dict):
        raise TypeError("Embedding profile payload must resolve to a mapping.")
    return EmbeddingProfileConfig(
        profile_id=resolved_profile_id,
        model_name=str(profile_payload.get("model_name", "")),
        device=str(profile_payload.get("device", "cuda")),
        torch_dtype=str(profile_payload.get("torch_dtype", "float16")),
        batch_size=int(profile_payload.get("batch_size", 1)),
    )


def build_autotune_candidates(
    *,
    base_profile: EmbeddingProfileConfig,
    texts_per_batch: int,
    words_per_text: int,
    encode_passes: int,
    hold_seconds: float,
) -> tuple[ExperimentConfig, ...]:
    """Build a deterministic candidate ladder for embedding-memory tuning."""
    candidates: list[ExperimentConfig] = []
    seen: set[tuple[object, ...]] = set()
    batch_sizes = tuple(
        dict.fromkeys(
            [
                max(1, int(base_profile.batch_size)),
                2,
                1,
            ]
        )
    )
    for batch_size in batch_sizes:
        for max_seq_length in _MAX_SEQ_LENGTH_CANDIDATES:
            for attention_implementation in _ATTENTION_IMPLEMENTATION_CANDIDATES:
                key = (
                    base_profile.torch_dtype,
                    batch_size,
                    max_seq_length,
                    attention_implementation,
                )
                if key in seen:
                    continue
                seen.add(key)
                experiment_id = (
                    f"bs{batch_size}"
                    f"-seq{max_seq_length if max_seq_length is not None else 'default'}"
                    f"-attn{attention_implementation or 'default'}"
                )
                candidates.append(
                    ExperimentConfig(
                        experiment_id=experiment_id,
                        model_name=base_profile.model_name,
                        device=base_profile.device,
                        torch_dtype=base_profile.torch_dtype,
                        batch_size=batch_size,
                        texts_per_batch=max(batch_size, texts_per_batch),
                        words_per_text=words_per_text,
                        encode_passes=encode_passes,
                        hold_seconds=hold_seconds,
                        max_seq_length=max_seq_length,
                        attention_implementation=attention_implementation,
                    )
                )
    return tuple(candidates)


def render_notes_markdown(
    *,
    generated_at_utc: str,
    config_path: Path,
    base_profile: EmbeddingProfileConfig,
    dedicated_limit_mb: float,
    allow_shared_mb: float,
    results: tuple[ExperimentResult, ...],
    selected: ExperimentResult | None,
) -> str:
    """Render one markdown notes file from all completed experiments."""
    lines = [
        "# Embedding Memory Profile",
        "",
        f"- Generated at: `{generated_at_utc}`",
        f"- Config path: `{config_path}`",
        f"- Embedding profile: `{base_profile.profile_id}`",
        f"- Model: `{base_profile.model_name}`",
        f"- Device: `{base_profile.device}`",
        f"- Dedicated limit: `{dedicated_limit_mb:.1f} MiB`",
        f"- Allowed shared GPU memory: `{allow_shared_mb:.1f} MiB`",
        "",
    ]
    if selected is not None:
        lines.extend(
            [
                "## Selected Configuration",
                "",
                f"- Experiment: `{selected.experiment.experiment_id}`",
                f"- Batch size: `{selected.experiment.batch_size}`",
                "- Max sequence length: "
                + (
                    f"`{selected.experiment.max_seq_length}`"
                    if selected.experiment.max_seq_length is not None
                    else "`default`"
                ),
                "- Attention implementation: "
                + (
                    f"`{selected.experiment.attention_implementation}`"
                    if selected.experiment.attention_implementation is not None
                    else "`default`"
                ),
                (
                    "- Baseline dedicated GPU memory: "
                    f"`{selected.baseline_dedicated_mb:.1f} MiB`"
                ),
                (
                    "- Baseline shared GPU memory: "
                    f"`{selected.baseline_shared_mb:.1f} MiB`"
                ),
                (
                    "- Peak added dedicated GPU memory: "
                    f"`{selected.peak_dedicated_mb:.1f} MiB`"
                ),
                (
                    "- Peak added shared GPU memory: "
                    f"`{selected.peak_shared_mb:.1f} MiB`"
                ),
                (
                    "- Peak raw dedicated GPU memory: "
                    f"`{selected.peak_total_gpu_used_mb:.1f} MiB`"
                ),
                (
                    "- Peak raw shared GPU memory: "
                    f"`{selected.peak_raw_shared_mb:.1f} MiB`"
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## Experiments",
            "",
            "| Experiment | Batch | Max Seq | Attn | Tokens | Added Dedicated MiB | Added Shared MiB | Raw Dedicated MiB | Raw Shared MiB | Torch Peak Reserved MiB | Status |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        lines.append(
            "| "
            + " | ".join(
                [
                    result.experiment.experiment_id,
                    str(result.experiment.batch_size),
                    (
                        str(result.experiment.max_seq_length)
                        if result.experiment.max_seq_length is not None
                        else "default"
                    ),
                    result.experiment.attention_implementation or "default",
                    str(result.actual_token_count),
                    f"{result.peak_dedicated_mb:.1f}",
                    f"{result.peak_shared_mb:.1f}",
                    f"{result.peak_total_gpu_used_mb:.1f}",
                    f"{result.peak_raw_shared_mb:.1f}",
                    (
                        f"{result.torch_peak_reserved_mb:.1f}"
                        if result.torch_peak_reserved_mb is not None
                        else "-"
                    ),
                    result.status,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Run Notes", ""])
    for result in results:
        lines.extend(
            [
                f"### {result.experiment.experiment_id}",
                "",
                f"- Exit code: `{result.exit_code}`",
                f"- Load seconds: `{result.load_seconds:.3f}`",
                f"- Encode seconds: `{result.encode_seconds:.3f}`",
                f"- Baseline dedicated GPU memory: `{result.baseline_dedicated_mb:.1f} MiB`",
                f"- Baseline shared GPU memory: `{result.baseline_shared_mb:.1f} MiB`",
                f"- Peak added dedicated GPU memory: `{result.peak_dedicated_mb:.1f} MiB`",
                f"- Peak added shared GPU memory: `{result.peak_shared_mb:.1f} MiB`",
                f"- Peak raw dedicated GPU memory: `{result.peak_total_gpu_used_mb:.1f} MiB`",
                f"- Peak raw shared GPU memory: `{result.peak_raw_shared_mb:.1f} MiB`",
                f"- Stdout: `{result.stdout_path}`",
                f"- Stderr: `{result.stderr_path}`",
                f"- Telemetry: `{result.telemetry_path}`",
                f"- Error: `{result.error or 'none'}`",
                "",
            ]
        )
    if selected is None:
        lines.extend(
            [
                "## Outcome",
                "",
                textwrap.dedent(
                    """\
                    No tested configuration satisfied the dedicated-memory and shared-memory
                    constraints. Inspect the run logs and telemetry artifacts before trying
                    more aggressive changes outside this sweep.
                    """
                ).strip(),
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _build_synthetic_text(words_per_text: int, index: int) -> str:
    """Build one deterministic transcript-like synthetic text payload."""
    words = [
        f"segment{index}",
        "podcast",
        "summary",
        "embedding",
        "retrieval",
        "speaker",
        "context",
        "episode",
    ]
    repeated_words = []
    while len(repeated_words) < words_per_text:
        repeated_words.extend(words)
    return " ".join(repeated_words[:words_per_text])


def _dtype_value(dtype_name: str) -> object:
    """Resolve a torch dtype from config-like text."""
    import torch

    normalized = dtype_name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get(normalized, torch.float16)


def _worker_main(args: argparse.Namespace) -> int:
    """Load the embedding model in isolation and run controlled encode passes."""
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for embedding-memory profiling.")

    torch.cuda.init()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    model_kwargs: dict[str, Any] = {
        "device_map": args.device,
        "torch_dtype": _dtype_value(str(args.torch_dtype)),
    }
    if args.attention_implementation:
        model_kwargs["attn_implementation"] = str(args.attention_implementation)

    start_time = time.perf_counter()
    model = SentenceTransformer(
        str(args.model_name),
        model_kwargs=model_kwargs,
        tokenizer_kwargs={"padding_side": "left"},
    )
    load_seconds = time.perf_counter() - start_time

    if int(args.max_seq_length or 0) > 0:
        model.max_seq_length = int(args.max_seq_length)

    texts = [
        _build_synthetic_text(int(args.words_per_text), index)
        for index in range(int(args.texts_per_batch))
    ]
    token_count = len(
        model.tokenizer(
            texts[0],
            truncation=True,
            max_length=int(model.max_seq_length),
        )["input_ids"]
    )

    time.sleep(float(args.hold_seconds))

    encode_started = time.perf_counter()
    for _ in range(int(args.encode_passes)):
        model.encode(
            texts,
            batch_size=int(args.batch_size),
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - encode_started

    time.sleep(float(args.hold_seconds))
    summary = WorkerSummary(
        load_seconds=load_seconds,
        encode_seconds=encode_seconds,
        max_seq_length=(
            int(model.max_seq_length) if getattr(model, "max_seq_length", None) else None
        ),
        actual_token_count=token_count,
        torch_peak_allocated_mb=(
            torch.cuda.max_memory_allocated() / float(1024 * 1024)
        ),
        torch_peak_reserved_mb=(
            torch.cuda.max_memory_reserved() / float(1024 * 1024)
        ),
        status="ok",
    )
    print(json.dumps(asdict(summary)), flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


def _child_command(experiment: ExperimentConfig) -> list[str]:
    """Build one worker subprocess command for a single experiment."""
    return [
        sys.executable,
        "-m",
        "card_framework.benchmark.embedding_memory_profiler",
        "--worker",
        "--model-name",
        experiment.model_name,
        "--device",
        experiment.device,
        "--torch-dtype",
        experiment.torch_dtype,
        "--batch-size",
        str(experiment.batch_size),
        "--texts-per-batch",
        str(experiment.texts_per_batch),
        "--words-per-text",
        str(experiment.words_per_text),
        "--encode-passes",
        str(experiment.encode_passes),
        "--hold-seconds",
        str(experiment.hold_seconds),
        "--max-seq-length",
        str(experiment.max_seq_length or 0),
        "--attention-implementation",
        str(experiment.attention_implementation or ""),
    ]


def _collect_worker_summary(stdout_path: Path) -> WorkerSummary:
    """Parse the worker's final JSON summary from stdout."""
    lines = [
        line.strip()
        for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]
    for line in reversed(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        return WorkerSummary(
            load_seconds=float(payload.get("load_seconds", 0.0) or 0.0),
            encode_seconds=float(payload.get("encode_seconds", 0.0) or 0.0),
            max_seq_length=(
                int(payload["max_seq_length"])
                if payload.get("max_seq_length") is not None
                else None
            ),
            actual_token_count=int(payload.get("actual_token_count", 0) or 0),
            torch_peak_allocated_mb=(
                float(payload["torch_peak_allocated_mb"])
                if payload.get("torch_peak_allocated_mb") is not None
                else None
            ),
            torch_peak_reserved_mb=(
                float(payload["torch_peak_reserved_mb"])
                if payload.get("torch_peak_reserved_mb") is not None
                else None
            ),
            status=str(payload.get("status", "error")),
            error=str(payload.get("error", "")),
        )
    return WorkerSummary(
        load_seconds=0.0,
        encode_seconds=0.0,
        max_seq_length=None,
        actual_token_count=0,
        torch_peak_allocated_mb=None,
        torch_peak_reserved_mb=None,
        status="error",
        error="worker did not emit a JSON summary",
    )


def _experiment_status(
    *,
    exit_code: int,
    peak_dedicated_mb: float,
    peak_shared_mb: float,
    peak_raw_dedicated_mb: float,
    dedicated_limit_mb: float,
    allow_shared_mb: float,
    worker_summary: WorkerSummary,
) -> tuple[str, str]:
    """Classify an experiment outcome against the requested constraints."""
    if exit_code != 0:
        return "failed", worker_summary.error or f"worker exited with code {exit_code}"
    if peak_shared_mb > allow_shared_mb:
        return (
            "failed",
            f"shared GPU memory reached {peak_shared_mb:.1f} MiB "
            f"(limit {allow_shared_mb:.1f} MiB)",
        )
    if peak_raw_dedicated_mb > dedicated_limit_mb:
        return (
            "failed",
            f"dedicated GPU memory reached {peak_raw_dedicated_mb:.1f} MiB "
            f"(limit {dedicated_limit_mb:.1f} MiB)",
        )
    return "passed", ""


def _run_experiment(
    *,
    experiment: ExperimentConfig,
    results_root: Path,
    sample_interval_seconds: float,
    command_timeout_seconds: float,
    dedicated_limit_mb: float,
    allow_shared_mb: float,
) -> ExperimentResult:
    """Run one worker experiment and monitor its GPU memory from the parent."""
    run_root = (results_root / experiment.experiment_id).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    stdout_path = (run_root / "stdout.txt").resolve()
    stderr_path = (run_root / "stderr.txt").resolve()
    telemetry_path = (run_root / "telemetry.jsonl").resolve()

    peak_dedicated_mb = 0.0
    peak_shared_mb = 0.0
    baseline_adapter = aggregate_adapter_memory(
        _collect_adapter_gpu_memory(command_timeout_seconds)
    )
    baseline_dedicated_mb = baseline_adapter.dedicated_mebibytes
    baseline_shared_mb = baseline_adapter.shared_mebibytes
    peak_total_gpu_used_mb = baseline_dedicated_mb
    peak_raw_shared_mb = baseline_shared_mb

    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle, telemetry_path.open("w", encoding="utf-8") as telemetry_handle:
        process = subprocess.Popen(
            _child_command(experiment),
            cwd=str(REPO_ROOT),
            env=os.environ.copy(),
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            while True:
                try:
                    aggregated = aggregate_adapter_memory(
                        _collect_adapter_gpu_memory(command_timeout_seconds)
                    )
                except Exception as exc:
                    telemetry_handle.write(
                        json.dumps(
                            {
                                "timestamp_utc": _utc_now_iso(),
                                "pid": process.pid,
                                "telemetry_error": str(exc),
                            }
                        )
                        + "\n"
                    )
                else:
                    raw_dedicated_mb = aggregated.dedicated_mebibytes
                    raw_shared_mb = aggregated.shared_mebibytes
                    peak_dedicated_mb = max(
                        peak_dedicated_mb,
                        max(0.0, raw_dedicated_mb - baseline_dedicated_mb),
                    )
                    peak_shared_mb = max(
                        peak_shared_mb,
                        max(0.0, raw_shared_mb - baseline_shared_mb),
                    )
                    peak_total_gpu_used_mb = max(
                        peak_total_gpu_used_mb,
                        raw_dedicated_mb,
                    )
                    peak_raw_shared_mb = max(
                        peak_raw_shared_mb,
                        raw_shared_mb,
                    )
                    telemetry_handle.write(
                        json.dumps(
                            {
                                "timestamp_utc": _utc_now_iso(),
                                "pid": process.pid,
                                "baseline_dedicated_mb": baseline_dedicated_mb,
                                "baseline_shared_mb": baseline_shared_mb,
                                "raw_dedicated_mb": raw_dedicated_mb,
                                "raw_shared_mb": raw_shared_mb,
                                "added_dedicated_mb": max(
                                    0.0,
                                    raw_dedicated_mb - baseline_dedicated_mb,
                                ),
                                "added_shared_mb": max(
                                    0.0,
                                    raw_shared_mb - baseline_shared_mb,
                                ),
                            }
                        )
                        + "\n"
                    )
                    telemetry_handle.flush()
                exit_code = process.poll()
                if exit_code is not None:
                    break
                time.sleep(sample_interval_seconds)
            process.wait(timeout=10.0)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)

    worker_summary = _collect_worker_summary(stdout_path)
    status, error = _experiment_status(
        exit_code=int(process.returncode or 0),
        peak_dedicated_mb=peak_dedicated_mb,
        peak_shared_mb=peak_shared_mb,
        peak_raw_dedicated_mb=peak_total_gpu_used_mb,
        dedicated_limit_mb=dedicated_limit_mb,
        allow_shared_mb=allow_shared_mb,
        worker_summary=worker_summary,
    )
    return ExperimentResult(
        experiment=experiment,
        exit_code=int(process.returncode or 0),
        load_seconds=worker_summary.load_seconds,
        encode_seconds=worker_summary.encode_seconds,
        actual_token_count=worker_summary.actual_token_count,
        baseline_dedicated_mb=baseline_dedicated_mb,
        baseline_shared_mb=baseline_shared_mb,
        peak_dedicated_mb=peak_dedicated_mb,
        peak_shared_mb=peak_shared_mb,
        peak_total_gpu_used_mb=peak_total_gpu_used_mb,
        peak_raw_shared_mb=peak_raw_shared_mb,
        torch_peak_allocated_mb=worker_summary.torch_peak_allocated_mb,
        torch_peak_reserved_mb=worker_summary.torch_peak_reserved_mb,
        status=status,
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
        telemetry_path=str(telemetry_path),
        error=error or worker_summary.error,
    )


def _write_session_artifacts(
    *,
    results_root: Path,
    config_path: Path,
    base_profile: EmbeddingProfileConfig,
    results: tuple[ExperimentResult, ...],
    selected: ExperimentResult | None,
    dedicated_limit_mb: float,
    allow_shared_mb: float,
) -> None:
    """Persist JSONL and markdown session artifacts."""
    jsonl_path = (results_root / "results.jsonl").resolve()
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    {
                        "experiment": asdict(result.experiment),
                        "exit_code": result.exit_code,
                        "load_seconds": result.load_seconds,
                        "encode_seconds": result.encode_seconds,
                        "actual_token_count": result.actual_token_count,
                        "baseline_dedicated_mb": result.baseline_dedicated_mb,
                        "baseline_shared_mb": result.baseline_shared_mb,
                        "peak_dedicated_mb": result.peak_dedicated_mb,
                        "peak_shared_mb": result.peak_shared_mb,
                        "peak_total_gpu_used_mb": result.peak_total_gpu_used_mb,
                        "peak_raw_shared_mb": result.peak_raw_shared_mb,
                        "torch_peak_allocated_mb": result.torch_peak_allocated_mb,
                        "torch_peak_reserved_mb": result.torch_peak_reserved_mb,
                        "status": result.status,
                        "stdout_path": result.stdout_path,
                        "stderr_path": result.stderr_path,
                        "telemetry_path": result.telemetry_path,
                        "error": result.error,
                    }
                )
                + "\n"
            )
    notes_path = (results_root / "notes.md").resolve()
    notes_path.write_text(
        render_notes_markdown(
            generated_at_utc=_utc_now_iso(),
            config_path=config_path,
            base_profile=base_profile,
            dedicated_limit_mb=dedicated_limit_mb,
            allow_shared_mb=allow_shared_mb,
            results=results,
            selected=selected,
        ),
        encoding="utf-8",
    )


def _parent_main(args: argparse.Namespace) -> int:
    """Run the parent autotune loop."""
    config_path = Path(args.config).resolve()
    results_root = _resolve_results_root(str(args.results_root))
    results_root.mkdir(parents=True, exist_ok=True)
    base_profile = _resolve_embedding_profile(
        config_path=config_path,
        profile_id=(str(args.embedding_profile).strip() or None),
    )
    candidates = build_autotune_candidates(
        base_profile=base_profile,
        texts_per_batch=int(args.texts_per_batch),
        words_per_text=int(args.words_per_text),
        encode_passes=int(args.encode_passes),
        hold_seconds=float(args.hold_seconds),
    )
    results: list[ExperimentResult] = []
    selected: ExperimentResult | None = None
    for experiment in candidates:
        result = _run_experiment(
            experiment=experiment,
            results_root=results_root,
            sample_interval_seconds=float(args.sample_interval_seconds),
            command_timeout_seconds=float(args.command_timeout_seconds),
            dedicated_limit_mb=float(args.dedicated_limit_mb),
            allow_shared_mb=float(args.allow_shared_mb),
        )
        results.append(result)
        _write_session_artifacts(
            results_root=results_root,
            config_path=config_path,
            base_profile=base_profile,
            results=tuple(results),
            selected=selected,
            dedicated_limit_mb=float(args.dedicated_limit_mb),
            allow_shared_mb=float(args.allow_shared_mb),
        )
        if result.succeeded:
            selected = result
            break
    _write_session_artifacts(
        results_root=results_root,
        config_path=config_path,
        base_profile=base_profile,
        results=tuple(results),
        selected=selected,
        dedicated_limit_mb=float(args.dedicated_limit_mb),
        allow_shared_mb=float(args.allow_shared_mb),
    )
    summary = {
        "status": "ok" if selected is not None else "no_fit",
        "results_root": str(results_root),
        "notes_path": str((results_root / "notes.md").resolve()),
        "results_path": str((results_root / "results.jsonl").resolve()),
        "selected_experiment": selected.experiment.experiment_id if selected else "",
    }
    print(json.dumps(summary), flush=True)
    return 0 if selected is not None else 1


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the embedding-memory profiler."""
    parser = argparse.ArgumentParser(
        description=(
            "Profile the isolated embedding model against dedicated/shared "
            "GPU memory constraints."
        ),
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", default=str(_DEFAULT_CONFIG_PATH))
    parser.add_argument("--embedding-profile", default="")
    parser.add_argument("--results-root", default="")
    parser.add_argument("--dedicated-limit-mb", type=float, default=12288.0)
    parser.add_argument("--allow-shared-mb", type=float, default=0.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.2)
    parser.add_argument("--command-timeout-seconds", type=float, default=5.0)
    parser.add_argument("--texts-per-batch", type=int, default=4)
    parser.add_argument("--words-per-text", type=int, default=768)
    parser.add_argument("--encode-passes", type=int, default=2)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--model-name", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="float16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=0)
    parser.add_argument("--attention-implementation", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the embedding-memory profiler CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.worker:
        return _worker_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
