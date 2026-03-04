"""Configuration parsing helpers for QA benchmark execution."""

from __future__ import annotations

from typing import Any, Literal, cast

from omegaconf import DictConfig


class QAConfigError(RuntimeError):
    """Raised when QA benchmark configuration values are invalid."""


InputGuardMode = Literal["error", "warn", "off"]


def as_positive_float(
    *,
    raw_value: Any,
    field_name: str,
    default_value: float,
) -> float:
    """Convert one config value into a positive float with strict validation."""
    if raw_value is None:
        return default_value
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise QAConfigError(
            f"{field_name} must be a positive number; got {raw_value!r}"
        ) from exc
    if parsed <= 0:
        raise QAConfigError(f"{field_name} must be > 0; got {raw_value!r}")
    return parsed


def resolve_workflow_timeouts(qa_cfg: DictConfig) -> tuple[float, float, float]:
    """Resolve creator/evaluator request and server-stop timeouts from QA config."""
    timeout_cfg = qa_cfg.get("qa", {}).get("timeouts", {})
    creator_request_timeout = as_positive_float(
        raw_value=timeout_cfg.get("creator_request_seconds"),
        field_name="qa.timeouts.creator_request_seconds",
        default_value=240.0,
    )
    evaluator_request_timeout = as_positive_float(
        raw_value=timeout_cfg.get("evaluator_request_seconds"),
        field_name="qa.timeouts.evaluator_request_seconds",
        default_value=900.0,
    )
    server_stop_timeout = as_positive_float(
        raw_value=timeout_cfg.get("server_stop_seconds"),
        field_name="qa.timeouts.server_stop_seconds",
        default_value=10.0,
    )
    return creator_request_timeout, evaluator_request_timeout, server_stop_timeout


def as_non_negative_int(
    *,
    raw_value: Any,
    field_name: str,
    default_value: int,
) -> int:
    """Convert one config value into a non-negative integer."""
    if raw_value is None:
        return default_value
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise QAConfigError(
            f"{field_name} must be an integer >= 0; got {raw_value!r}"
        ) from exc
    if parsed < 0:
        raise QAConfigError(f"{field_name} must be >= 0; got {raw_value!r}")
    return parsed


def as_ratio(
    *,
    raw_value: Any,
    field_name: str,
    default_value: float,
) -> float:
    """Convert one config value into a ratio in ``[0.0, 1.0]``."""
    if raw_value is None:
        return default_value
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError) as exc:
        raise QAConfigError(
            f"{field_name} must be a ratio in [0.0, 1.0]; got {raw_value!r}"
        ) from exc
    if parsed < 0.0 or parsed > 1.0:
        raise QAConfigError(f"{field_name} must be in [0.0, 1.0]; got {raw_value!r}")
    return parsed


def resolve_input_guard_config(
    qa_cfg: DictConfig,
) -> tuple[float, int, int, int, InputGuardMode]:
    """Resolve preflight source-summary input-guard settings."""
    guard_cfg = qa_cfg.get("qa", {}).get("input_guard", {})
    min_overlap_ratio = as_ratio(
        raw_value=guard_cfg.get("min_overlap_ratio"),
        field_name="qa.input_guard.min_overlap_ratio",
        default_value=0.03,
    )
    min_shared_tokens = as_non_negative_int(
        raw_value=guard_cfg.get("min_shared_tokens"),
        field_name="qa.input_guard.min_shared_tokens",
        default_value=12,
    )
    min_shared_distinctive_tokens = as_non_negative_int(
        raw_value=guard_cfg.get("min_shared_distinctive_tokens"),
        field_name="qa.input_guard.min_shared_distinctive_tokens",
        default_value=3,
    )
    min_shared_name_phrases = as_non_negative_int(
        raw_value=guard_cfg.get("min_shared_name_phrases"),
        field_name="qa.input_guard.min_shared_name_phrases",
        default_value=2,
    )
    raw_mode = str(guard_cfg.get("mode", "error")).strip().lower()
    if raw_mode not in {"error", "warn", "off"}:
        raise QAConfigError("qa.input_guard.mode must be one of: error, warn, off")
    return (
        min_overlap_ratio,
        min_shared_tokens,
        min_shared_distinctive_tokens,
        min_shared_name_phrases,
        cast(InputGuardMode, raw_mode),
    )
