"""Tests for affect-analysis resource profiling helpers."""

from __future__ import annotations

from card_framework.audio_pipeline.emotion.affect_profiling import (
    aggregate_adapter_memory,
    aggregate_process_memory_for_pid,
    parse_adapter_memory_rows,
    parse_process_memory_rows,
)


def test_parse_and_aggregate_process_gpu_rows() -> None:
    output = (
        '[{"instance_name":"pid_10_luid_0x0_phys_0","pid":10,'
        '"dedicated_bytes":1048576,"shared_bytes":2097152},'
        '{"instance_name":"pid_10_luid_0x0_phys_1","pid":10,'
        '"dedicated_bytes":3145728,"shared_bytes":4194304}]'
    )

    rows = parse_process_memory_rows(output)
    aggregated = aggregate_process_memory_for_pid(rows, pid=10)

    assert len(rows) == 2
    assert aggregated.dedicated_bytes == 4 * 1024 * 1024
    assert aggregated.shared_bytes == 6 * 1024 * 1024


def test_parse_and_aggregate_adapter_gpu_rows() -> None:
    output = (
        '[{"instance_name":"gpu_0","dedicated_bytes":5242880,"shared_bytes":1048576},'
        '{"instance_name":"gpu_1","dedicated_bytes":2097152,"shared_bytes":0}]'
    )

    rows = parse_adapter_memory_rows(output)
    aggregated = aggregate_adapter_memory(rows)

    assert len(rows) == 2
    assert aggregated.dedicated_bytes == 7 * 1024 * 1024
    assert aggregated.shared_bytes == 1 * 1024 * 1024

