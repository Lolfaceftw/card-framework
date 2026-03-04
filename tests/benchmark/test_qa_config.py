"""Unit tests for QA benchmark timeout config parsing."""

from __future__ import annotations

from omegaconf import OmegaConf
import pytest

from benchmark.qa_settings import (
    QAConfigError,
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
