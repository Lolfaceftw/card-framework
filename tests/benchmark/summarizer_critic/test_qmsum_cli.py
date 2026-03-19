"""CLI coverage for QMSum summarizer-critic helpers."""

from __future__ import annotations

import argparse

from card_framework.benchmark.summarizer_critic.qmsum.cli import (
    _build_supervised_child_command,
)


def test_build_supervised_child_command_forwards_sample_seed() -> None:
    args = argparse.Namespace(
        dataset_manifest="manifest.json",
        qmsum_repo_dir="QMSum",
        reference_wpm=150.0,
        output_root="runs",
        provider_profiles="profiles.yaml",
        config="config.yaml",
        summarizer_provider="vllm_default",
        critic_provider="vllm_default",
        judge_provider="vllm_default",
        embedding_mode="disabled",
        judge_rubric="judge.yaml",
        judge_repeats=1,
        alignscore_model="sentence-transformers/all-MiniLM-L6-v2",
        alignscore_source_chunk_words=256,
        max_samples=25,
        sample_seed=17,
        max_iterations=3,
        summarizer_timeout_seconds=30.0,
        critic_timeout_seconds=40.0,
        retrieval_timeout_seconds=50.0,
        index_timeout_seconds=60.0,
        prepare_if_missing=False,
        force_download_dataset=False,
        disable_order_swap=False,
    )

    command = _build_supervised_child_command(args)

    assert "--sample-seed" in command
    assert "17" in command
