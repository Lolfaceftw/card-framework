"""Tests for the isolated embedding-memory profiler helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from card_framework.benchmark.embedding_memory_profiler import (
    EmbeddingProfileConfig,
    ExperimentConfig,
    ExperimentResult,
    _POWERSHELL_GPU_ADAPTER_MEMORY_SCRIPT,
    _POWERSHELL_GPU_PROCESS_MEMORY_SCRIPT,
    aggregate_adapter_memory,
    aggregate_process_memory_for_pid,
    build_autotune_candidates,
    parse_adapter_memory_rows,
    parse_process_memory_rows,
    render_notes_markdown,
)


def test_parse_process_memory_rows_and_aggregate_by_pid() -> None:
    output = """
    [
      {"instance_name":"pid_101_luid_0x0_phys_0","pid":101,"dedicated_bytes":1048576,"shared_bytes":0},
      {"instance_name":"pid_101_luid_0x0_phys_1","pid":101,"dedicated_bytes":2097152,"shared_bytes":524288},
      {"instance_name":"pid_202_luid_0x0_phys_0","pid":202,"dedicated_bytes":4194304,"shared_bytes":1048576}
    ]
    """

    samples = parse_process_memory_rows(output)
    aggregated = aggregate_process_memory_for_pid(samples, 101)

    assert len(samples) == 3
    assert aggregated.pid == 101
    assert aggregated.dedicated_bytes == 3 * 1024 * 1024
    assert aggregated.shared_bytes == 512 * 1024


def test_build_autotune_candidates_includes_default_and_reduced_batch_sizes() -> None:
    profile = EmbeddingProfileConfig(
        profile_id="qwen3_4b",
        model_name="Qwen/Qwen3-Embedding-4B",
        device="cuda",
        torch_dtype="float16",
        batch_size=4,
    )

    candidates = build_autotune_candidates(
        base_profile=profile,
        texts_per_batch=4,
        words_per_text=768,
        encode_passes=2,
        hold_seconds=1.5,
    )

    batch_sizes = {candidate.batch_size for candidate in candidates}
    experiment_ids = {candidate.experiment_id for candidate in candidates}

    assert {1, 2, 4}.issubset(batch_sizes)
    assert "bs4-seqdefault-attndefault" in experiment_ids
    assert len(candidates) == len(experiment_ids)


def test_worker_summary_slots_still_serialize_with_asdict() -> None:
    profile = EmbeddingProfileConfig(
        profile_id="qwen3_4b",
        model_name="Qwen/Qwen3-Embedding-4B",
        device="cuda",
        torch_dtype="float16",
        batch_size=4,
    )

    assert asdict(profile)["model_name"] == "Qwen/Qwen3-Embedding-4B"


def test_parse_adapter_memory_rows_and_aggregate_totals() -> None:
    output = """
    [
      {"instance_name":"luid_0x0_phys_0","dedicated_bytes":1048576,"shared_bytes":0},
      {"instance_name":"luid_0x1_phys_0","dedicated_bytes":2097152,"shared_bytes":524288}
    ]
    """

    samples = parse_adapter_memory_rows(output)
    aggregated = aggregate_adapter_memory(samples)

    assert len(samples) == 2
    assert aggregated.dedicated_bytes == 3 * 1024 * 1024
    assert aggregated.shared_bytes == 512 * 1024


def test_powershell_gpu_counter_scripts_target_expected_counters() -> None:
    assert "\\GPU Process Memory(*)\\Dedicated Usage" in _POWERSHELL_GPU_PROCESS_MEMORY_SCRIPT
    assert "\\GPU Adapter Memory(*)\\Dedicated Usage" in _POWERSHELL_GPU_ADAPTER_MEMORY_SCRIPT


def test_render_notes_markdown_includes_selected_configuration() -> None:
    profile = EmbeddingProfileConfig(
        profile_id="qwen3_4b",
        model_name="Qwen/Qwen3-Embedding-4B",
        device="cuda",
        torch_dtype="float16",
        batch_size=4,
    )
    experiment = ExperimentConfig(
        experiment_id="bs1-seq512-attndefault",
        model_name=profile.model_name,
        device=profile.device,
        torch_dtype=profile.torch_dtype,
        batch_size=1,
        texts_per_batch=4,
        words_per_text=768,
        encode_passes=2,
        hold_seconds=1.5,
        max_seq_length=512,
    )
    result = ExperimentResult(
        experiment=experiment,
        exit_code=0,
        load_seconds=12.5,
        encode_seconds=4.0,
        actual_token_count=512,
        baseline_dedicated_mb=700.0,
        baseline_shared_mb=24.0,
        peak_dedicated_mb=8192.0,
        peak_shared_mb=0.0,
        peak_total_gpu_used_mb=8892.0,
        peak_raw_shared_mb=24.0,
        torch_peak_allocated_mb=7800.0,
        torch_peak_reserved_mb=8000.0,
        status="passed",
        stdout_path="stdout.txt",
        stderr_path="stderr.txt",
        telemetry_path="telemetry.jsonl",
    )

    notes = render_notes_markdown(
        generated_at_utc="2026-03-19T00:00:00+00:00",
        config_path=Path("conf/config.yaml"),
        base_profile=profile,
        dedicated_limit_mb=12288.0,
        allow_shared_mb=0.0,
        results=(result,),
        selected=result,
    )

    assert "# Embedding Memory Profile" in notes
    assert "## Selected Configuration" in notes
    assert "bs1-seq512-attndefault" in notes
    assert "Peak added shared GPU memory: `0.0 MiB`" in notes
    assert "Peak raw dedicated GPU memory: `8892.0 MiB`" in notes
