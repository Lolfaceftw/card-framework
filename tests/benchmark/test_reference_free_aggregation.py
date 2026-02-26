from __future__ import annotations

from benchmark.metrics import aggregate_results
from benchmark.types import CellRunResult, SampleRunResult


def test_aggregate_results_reference_free_metrics() -> None:
    results = [
        CellRunResult(
            cell_id="c1",
            provider_id="p1",
            embedding_id="disabled",
            repeat_index=1,
            status="pass",
            sample_results=[
                SampleRunResult(
                    sample_id="s1",
                    dataset="d",
                    status="pass",
                    duration_seconds=2.0,
                    converged=True,
                    iterations_run=1,
                    final_status="pass",
                    final_word_count=520,
                    word_budget_in_range=True,
                    fallback_verdict_used=False,
                    retrieval_events=1,
                    tool_invocations=2,
                    alignscore=0.75,
                    alignscore_backend="sentence_transformer_proxy",
                    judge_scores={
                        "factuality": 0.8,
                        "relevance": 0.7,
                        "coherence": 0.9,
                        "overall": 0.82,
                    },
                    judge_pairwise_winner="candidate",
                    judge_order_consistent=True,
                    judge_repeat_delta=0.03,
                    reference_free_status="ok",
                ),
                SampleRunResult(
                    sample_id="s2",
                    dataset="d",
                    status="fail",
                    duration_seconds=4.0,
                    converged=False,
                    iterations_run=3,
                    final_status="max_iterations_reached",
                    final_word_count=0,
                    word_budget_in_range=False,
                    fallback_verdict_used=True,
                    retrieval_events=1,
                    tool_invocations=3,
                    alignscore=None,
                    judge_scores=None,
                    reference_free_status="error",
                ),
            ],
        )
    ]

    aggregate = aggregate_results(results)

    assert aggregate.total_samples == 2
    assert aggregate.critic_pass_rate == 0.5
    assert aggregate.alignscore_mean == 0.75
    assert aggregate.judge_overall_mean == 0.82
    assert aggregate.judge_factuality_mean == 0.8
    assert aggregate.judge_order_consistency_rate == 1.0
    assert aggregate.judge_repeat_stability_mean == 0.03
    assert aggregate.reference_free_coverage_rate == 0.5
