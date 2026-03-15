"""Runner progress helper coverage for the summarizer-critic benchmark."""

from __future__ import annotations

import argparse

from card_framework.benchmark.summarizer_critic.qmsum.runner import (
    _select_samples,
    _format_eta,
    _progress_snapshot,
    _resolve_benchmark_ports,
    BenchmarkExecutionError,
    build_execute_parser,
)
from card_framework.benchmark.summarizer_critic.config import PreparedBenchmarkSample


def _sample(sample_id: str) -> PreparedBenchmarkSample:
    return PreparedBenchmarkSample(
        sample_id=sample_id,
        dataset="qmsum_general_test",
        meeting_id=sample_id,
        transcript_path=f"{sample_id}.json",
        reference_summary="reference",
        query_text="query",
        target_seconds=60,
        metadata={},
    )


def test_format_eta_compacts_hours_minutes_and_seconds() -> None:
    assert _format_eta(None) == "n/a"
    assert _format_eta(0.0) == "0s"
    assert _format_eta(95.0) == "1m35s"
    assert _format_eta(3720.0) == "1h02m"


def test_progress_snapshot_reports_remaining_rate_and_eta() -> None:
    remaining, average_text, eta_text = _progress_snapshot(
        completed_samples=3,
        total_samples=10,
        elapsed_seconds=600.0,
    )

    assert remaining == 7
    assert average_text == "0.30 ex/min"
    assert eta_text == "23m20s"


def test_progress_snapshot_reports_na_until_first_completion() -> None:
    remaining, average_text, eta_text = _progress_snapshot(
        completed_samples=0,
        total_samples=10,
        elapsed_seconds=45.0,
    )

    assert remaining == 10
    assert average_text == "n/a"
    assert eta_text == "n/a"


def test_build_execute_parser_accepts_explicit_service_ports() -> None:
    parser = argparse.ArgumentParser()

    build_execute_parser(parser)
    args = parser.parse_args(
        [
            "--retrieval-port",
            "31000",
            "--summarizer-port",
            "31001",
            "--critic-port",
            "31002",
        ]
    )

    assert args.retrieval_port == 31000
    assert args.summarizer_port == 31001
    assert args.critic_port == 31002


def test_build_execute_parser_accepts_sample_seed() -> None:
    parser = argparse.ArgumentParser()

    build_execute_parser(parser)
    args = parser.parse_args(["--sample-seed", "17"])

    assert args.sample_seed == 17


def test_select_samples_preserves_manifest_order_without_seed() -> None:
    samples = (_sample("a"), _sample("b"), _sample("c"))

    assert [
        sample.sample_id
        for sample in _select_samples(
            samples,
            max_samples=2,
            sample_seed=None,
        )
    ] == ["a", "b"]


def test_select_samples_uses_seed_independent_of_manifest_order() -> None:
    forward = (_sample("a"), _sample("b"), _sample("c"), _sample("d"))
    reverse = tuple(reversed(forward))

    forward_ids = [
        sample.sample_id
        for sample in _select_samples(
            forward,
            max_samples=3,
            sample_seed=11,
        )
    ]
    reverse_ids = [
        sample.sample_id
        for sample in _select_samples(
            reverse,
            max_samples=3,
            sample_seed=11,
        )
    ]

    assert forward_ids == reverse_ids


def test_resolve_benchmark_ports_rejects_duplicates() -> None:
    args = argparse.Namespace(
        retrieval_port=32000,
        summarizer_port=32000,
        critic_port=32002,
    )

    try:
        _resolve_benchmark_ports(args)
    except BenchmarkExecutionError as exc:
        assert "must be distinct" in str(exc)
    else:
        raise AssertionError(
            "Expected duplicate ports to raise BenchmarkExecutionError"
        )
