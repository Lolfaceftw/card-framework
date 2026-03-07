"""Unit tests for QA benchmark timeout config parsing."""

from __future__ import annotations

from omegaconf import OmegaConf
import pytest

from benchmark.qa_settings import (
    CorrectorRuntimeConfig,
    EvaluatorRuntimeConfig,
    QAConfigError,
    QuoteRelevanceConfig,
    resolve_corrector_runtime_config,
    resolve_evaluator_runtime_config,
    resolve_input_guard_config,
    resolve_workflow_timeouts,
)


def test_resolve_workflow_timeouts_uses_defaults() -> None:
    """Use built-in defaults when timeout keys are absent."""
    cfg = OmegaConf.create({"qa": {}})
    creator, evaluator, stop = resolve_workflow_timeouts(cfg)
    assert creator == 240.0
    assert evaluator == 900.0
    assert stop == 10.0


def test_resolve_workflow_timeouts_reads_config_values() -> None:
    """Read timeout overrides from benchmark config structure."""
    cfg = OmegaConf.create(
        {
            "qa": {
                "timeouts": {
                    "creator_request_seconds": 300,
                    "evaluator_request_seconds": 1200,
                    "server_stop_seconds": 15,
                }
            }
        }
    )
    creator, evaluator, stop = resolve_workflow_timeouts(cfg)
    assert creator == 300.0
    assert evaluator == 1200.0
    assert stop == 15.0


def test_resolve_workflow_timeouts_rejects_non_positive_values() -> None:
    """Reject invalid non-positive timeout values."""
    cfg = OmegaConf.create({"qa": {"timeouts": {"evaluator_request_seconds": 0}}})
    with pytest.raises(QAConfigError):
        resolve_workflow_timeouts(cfg)


def test_resolve_input_guard_config_uses_defaults() -> None:
    """Use built-in defaults for input guard config when unspecified."""
    cfg = OmegaConf.create({"qa": {}})
    (
        overlap_ratio,
        shared_tokens,
        shared_distinctive_tokens,
        shared_name_phrases,
        mode,
    ) = resolve_input_guard_config(cfg)
    assert overlap_ratio == 0.03
    assert shared_tokens == 12
    assert shared_distinctive_tokens == 3
    assert shared_name_phrases == 2
    assert mode == "error"


def test_resolve_input_guard_config_rejects_invalid_mode() -> None:
    """Reject unsupported input guard mode values."""
    cfg = OmegaConf.create({"qa": {"input_guard": {"mode": "invalid"}}})
    with pytest.raises(QAConfigError):
        resolve_input_guard_config(cfg)


def test_resolve_evaluator_runtime_config_uses_defaults() -> None:
    """Use built-in evaluator runtime defaults when config section is absent."""
    cfg = OmegaConf.create({"qa": {}})
    resolved = resolve_evaluator_runtime_config(cfg)
    assert resolved == EvaluatorRuntimeConfig(
        max_tool_turns=320,
        max_attempts_per_question=3,
        chat_max_tokens=160,
        no_tool_call_patience=3,
        max_tool_calls_per_turn=1,
        per_question_concurrency=1,
        quote_relevance=QuoteRelevanceConfig(
            mode="hybrid",
            min_shared_tokens=1,
            min_distinctive_shared_tokens=1,
            min_token_length=3,
            semantic_threshold=None,
            auto_repair=True,
            repair_min_score=0.25,
            min_candidate_chars=6,
        ),
    )


def test_resolve_evaluator_runtime_config_reads_values() -> None:
    """Read explicit evaluator runtime settings from QA config."""
    cfg = OmegaConf.create(
        {
            "qa": {
                "evaluator": {
                    "max_tool_turns": 77,
                    "max_attempts_per_question": 4,
                    "chat_max_tokens": 222,
                    "no_tool_call_patience": 5,
                    "max_tool_calls_per_turn": 2,
                    "per_question_concurrency": 5,
                    "quote_relevance": {
                        "mode": "semantic_similarity",
                        "min_shared_tokens": 2,
                        "min_distinctive_shared_tokens": 0,
                        "min_token_length": 4,
                        "semantic_threshold": 0.31,
                        "auto_repair": False,
                        "repair_min_score": 0.61,
                        "min_candidate_chars": 8,
                    },
                }
            }
        }
    )
    resolved = resolve_evaluator_runtime_config(cfg)
    assert resolved.max_tool_turns == 77
    assert resolved.max_attempts_per_question == 4
    assert resolved.chat_max_tokens == 222
    assert resolved.no_tool_call_patience == 5
    assert resolved.max_tool_calls_per_turn == 2
    assert resolved.per_question_concurrency == 5
    assert resolved.quote_relevance.mode == "semantic_similarity"
    assert resolved.quote_relevance.semantic_threshold == 0.31
    assert resolved.quote_relevance.auto_repair is False
    assert resolved.quote_relevance.repair_min_score == 0.61
    assert resolved.quote_relevance.min_candidate_chars == 8


def test_resolve_evaluator_runtime_config_rejects_invalid_mode() -> None:
    """Reject unsupported evaluator quote relevance mode values."""
    cfg = OmegaConf.create(
        {"qa": {"evaluator": {"quote_relevance": {"mode": "unknown"}}}}
    )
    with pytest.raises(QAConfigError):
        resolve_evaluator_runtime_config(cfg)


def test_resolve_evaluator_runtime_config_accepts_hybrid_mode() -> None:
    """Accept hybrid quote relevance mode for lexical-plus-semantic fallback."""
    cfg = OmegaConf.create(
        {"qa": {"evaluator": {"quote_relevance": {"mode": "hybrid"}}}}
    )
    resolved = resolve_evaluator_runtime_config(cfg)
    assert resolved.quote_relevance.mode == "hybrid"


def test_resolve_evaluator_runtime_config_parses_string_bool_false() -> None:
    """Parse string boolean values for auto_repair deterministically."""
    cfg = OmegaConf.create(
        {"qa": {"evaluator": {"quote_relevance": {"auto_repair": "false"}}}}
    )
    resolved = resolve_evaluator_runtime_config(cfg)
    assert resolved.quote_relevance.auto_repair is False


def test_resolve_evaluator_runtime_config_rejects_invalid_bool() -> None:
    """Reject non-boolean-like values for auto_repair."""
    cfg = OmegaConf.create(
        {"qa": {"evaluator": {"quote_relevance": {"auto_repair": "not-bool"}}}}
    )
    with pytest.raises(QAConfigError):
        resolve_evaluator_runtime_config(cfg)


def test_resolve_evaluator_runtime_config_rejects_non_positive_concurrency() -> None:
    """Reject non-positive per-question evaluator concurrency values."""
    cfg = OmegaConf.create({"qa": {"evaluator": {"per_question_concurrency": 0}}})
    with pytest.raises(QAConfigError):
        resolve_evaluator_runtime_config(cfg)


def test_resolve_corrector_runtime_config_uses_defaults() -> None:
    """Use built-in Corrector defaults when config section is absent."""
    cfg = OmegaConf.create({"qa": {}})
    assert resolve_corrector_runtime_config(cfg) == CorrectorRuntimeConfig(
        enabled=False,
        max_tokens=700,
        max_examples=2,
    )


def test_resolve_corrector_runtime_config_reads_values() -> None:
    """Read explicit Corrector runtime settings from QA config."""
    cfg = OmegaConf.create(
        {"qa": {"corrector": {"enabled": False, "max_tokens": 1200, "max_examples": 4}}}
    )
    resolved = resolve_corrector_runtime_config(cfg)
    assert resolved.enabled is False
    assert resolved.max_tokens == 1200
    assert resolved.max_examples == 4
