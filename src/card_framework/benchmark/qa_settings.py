"""Configuration parsing helpers for QA benchmark execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from omegaconf import DictConfig


class QAConfigError(RuntimeError):
    """Raised when QA benchmark configuration values are invalid."""


InputGuardMode = Literal["error", "warn", "off"]
QuoteRelevanceMode = Literal["lexical_overlap", "semantic_similarity", "hybrid", "off"]


@dataclass(slots=True, frozen=True)
class QuoteRelevanceConfig:
    """Typed quote-relevance validation settings for evaluator runtime."""

    mode: QuoteRelevanceMode
    min_shared_tokens: int
    min_distinctive_shared_tokens: int
    min_token_length: int
    semantic_threshold: float | None
    auto_repair: bool
    repair_min_score: float
    min_candidate_chars: int


@dataclass(slots=True, frozen=True)
class EvaluatorRuntimeConfig:
    """Typed evaluator runtime configuration resolved from qa config."""

    max_tool_turns: int
    max_attempts_per_question: int
    chat_max_tokens: int
    no_tool_call_patience: int
    max_tool_calls_per_turn: int
    per_question_concurrency: int
    quote_relevance: QuoteRelevanceConfig


@dataclass(slots=True, frozen=True)
class CorrectorRuntimeConfig:
    """Typed Corrector runtime configuration resolved from QA config."""

    enabled: bool
    max_tokens: int
    max_examples: int


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


def as_positive_int(
    *,
    raw_value: Any,
    field_name: str,
    default_value: int,
) -> int:
    """Convert one config value into a positive integer with strict validation."""
    if raw_value is None:
        return default_value
    try:
        parsed = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise QAConfigError(
            f"{field_name} must be an integer > 0; got {raw_value!r}"
        ) from exc
    if parsed <= 0:
        raise QAConfigError(f"{field_name} must be > 0; got {raw_value!r}")
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


def as_optional_ratio(
    *,
    raw_value: Any,
    field_name: str,
) -> float | None:
    """Convert optional ratio value into ``None`` or value in ``[0.0, 1.0]``."""
    if raw_value is None:
        return None
    return as_ratio(raw_value=raw_value, field_name=field_name, default_value=0.0)


def as_bool(
    *,
    raw_value: Any,
    field_name: str,
    default_value: bool,
) -> bool:
    """Convert one config value into a strict boolean."""
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        normalized = raw_value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise QAConfigError(f"{field_name} must be a boolean; got {raw_value!r}")


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


def resolve_evaluator_runtime_config(qa_cfg: DictConfig) -> EvaluatorRuntimeConfig:
    """Resolve strict typed evaluator runtime settings from QA config."""
    evaluator_cfg = qa_cfg.get("qa", {}).get("evaluator", {})
    quote_cfg = evaluator_cfg.get("quote_relevance", {})

    max_tool_turns = as_positive_int(
        raw_value=evaluator_cfg.get("max_tool_turns"),
        field_name="qa.evaluator.max_tool_turns",
        default_value=320,
    )
    max_attempts_per_question = as_positive_int(
        raw_value=evaluator_cfg.get("max_attempts_per_question"),
        field_name="qa.evaluator.max_attempts_per_question",
        default_value=3,
    )
    chat_max_tokens = as_positive_int(
        raw_value=evaluator_cfg.get("chat_max_tokens"),
        field_name="qa.evaluator.chat_max_tokens",
        default_value=160,
    )
    no_tool_call_patience = as_positive_int(
        raw_value=evaluator_cfg.get("no_tool_call_patience"),
        field_name="qa.evaluator.no_tool_call_patience",
        default_value=3,
    )
    max_tool_calls_per_turn = as_positive_int(
        raw_value=evaluator_cfg.get("max_tool_calls_per_turn"),
        field_name="qa.evaluator.max_tool_calls_per_turn",
        default_value=1,
    )
    per_question_concurrency = as_positive_int(
        raw_value=evaluator_cfg.get("per_question_concurrency"),
        field_name="qa.evaluator.per_question_concurrency",
        default_value=1,
    )

    raw_mode = str(quote_cfg.get("mode", "hybrid")).strip().lower()
    if raw_mode not in {"lexical_overlap", "semantic_similarity", "hybrid", "off"}:
        raise QAConfigError(
            "qa.evaluator.quote_relevance.mode must be one of: "
            "lexical_overlap, semantic_similarity, hybrid, off"
        )

    quote_relevance = QuoteRelevanceConfig(
        mode=cast(QuoteRelevanceMode, raw_mode),
        min_shared_tokens=as_positive_int(
            raw_value=quote_cfg.get("min_shared_tokens"),
            field_name="qa.evaluator.quote_relevance.min_shared_tokens",
            default_value=1,
        ),
        min_distinctive_shared_tokens=as_non_negative_int(
            raw_value=quote_cfg.get("min_distinctive_shared_tokens"),
            field_name=("qa.evaluator.quote_relevance.min_distinctive_shared_tokens"),
            default_value=1,
        ),
        min_token_length=as_positive_int(
            raw_value=quote_cfg.get("min_token_length"),
            field_name="qa.evaluator.quote_relevance.min_token_length",
            default_value=3,
        ),
        semantic_threshold=as_optional_ratio(
            raw_value=quote_cfg.get("semantic_threshold"),
            field_name="qa.evaluator.quote_relevance.semantic_threshold",
        ),
        auto_repair=as_bool(
            raw_value=quote_cfg.get("auto_repair"),
            field_name="qa.evaluator.quote_relevance.auto_repair",
            default_value=True,
        ),
        repair_min_score=as_ratio(
            raw_value=quote_cfg.get("repair_min_score"),
            field_name="qa.evaluator.quote_relevance.repair_min_score",
            default_value=0.25,
        ),
        min_candidate_chars=as_positive_int(
            raw_value=quote_cfg.get("min_candidate_chars"),
            field_name="qa.evaluator.quote_relevance.min_candidate_chars",
            default_value=6,
        ),
    )

    return EvaluatorRuntimeConfig(
        max_tool_turns=max_tool_turns,
        max_attempts_per_question=max_attempts_per_question,
        chat_max_tokens=chat_max_tokens,
        no_tool_call_patience=no_tool_call_patience,
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        per_question_concurrency=per_question_concurrency,
        quote_relevance=quote_relevance,
    )


def resolve_corrector_runtime_config(qa_cfg: DictConfig) -> CorrectorRuntimeConfig:
    """Resolve strict typed Corrector settings from QA config."""
    corrector_cfg = qa_cfg.get("qa", {}).get("corrector", {})
    enabled = as_bool(
        raw_value=corrector_cfg.get("enabled"),
        field_name="qa.corrector.enabled",
        default_value=False,
    )
    max_tokens = as_positive_int(
        raw_value=corrector_cfg.get("max_tokens"),
        field_name="qa.corrector.max_tokens",
        default_value=700,
    )
    max_examples = as_positive_int(
        raw_value=corrector_cfg.get("max_examples"),
        field_name="qa.corrector.max_examples",
        default_value=2,
    )
    return CorrectorRuntimeConfig(
        enabled=enabled,
        max_tokens=max_tokens,
        max_examples=max_examples,
    )
