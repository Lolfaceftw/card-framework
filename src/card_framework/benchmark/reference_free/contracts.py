"""Contracts and schema helpers for reference-free benchmark metrics."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class ReferenceFreeContractError(RuntimeError):
    """Raised when reference-free rubric contracts are invalid."""


@dataclass(slots=True, frozen=True)
class JudgeRubricDimension:
    """One scoring dimension used by the LLM judge rubric."""

    name: str
    description: str
    min_score: float
    max_score: float


@dataclass(slots=True, frozen=True)
class JudgeRubric:
    """Structured rubric contract for reference-free judge scoring."""

    rubric_id: str
    version: str
    dimensions: list[JudgeRubricDimension]
    pairwise_dimension: str


@dataclass(slots=True)
class AlignScoreEvaluation:
    """Outcome of alignment-style reference-free scoring."""

    status: str
    score: float | None
    backend: str
    error_message: str | None = None


@dataclass(slots=True)
class JudgeEvaluation:
    """Outcome of LLM judge reference-free scoring."""

    status: str
    scores: dict[str, float] | None
    pairwise_winner: str | None
    order_consistent: bool | None
    repeat_delta: float | None
    error_message: str | None = None


@dataclass(slots=True)
class ReferenceFreeEvaluation:
    """Combined reference-free evaluation output for one sample."""

    status: str
    alignscore: float | None
    alignscore_backend: str | None
    judge_scores: dict[str, float] | None
    judge_pairwise_winner: str | None
    judge_order_consistent: bool | None
    judge_repeat_delta: float | None
    error_message: str | None = None


def load_judge_rubric(path: Path) -> JudgeRubric:
    """Load and validate the LLM judge rubric from JSON file.

    Args:
        path: Absolute or relative path to rubric JSON.

    Returns:
        Parsed and validated :class:`JudgeRubric`.

    Raises:
        ReferenceFreeContractError: If file is missing or contract is invalid.
    """
    if not path.exists():
        raise ReferenceFreeContractError(f"Judge rubric does not exist: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - defensive I/O branch
        raise ReferenceFreeContractError(f"Failed to parse rubric JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise ReferenceFreeContractError("Rubric payload must be an object")

    rubric_id = str(payload.get("rubric_id", "")).strip()
    version = str(payload.get("version", "")).strip()
    pairwise_dimension = str(payload.get("pairwise_dimension", "overall")).strip()

    raw_dimensions = payload.get("dimensions")
    if not isinstance(raw_dimensions, list) or not raw_dimensions:
        raise ReferenceFreeContractError("Rubric must contain a non-empty dimensions list")

    dimensions: list[JudgeRubricDimension] = []
    for raw_dimension in raw_dimensions:
        if not isinstance(raw_dimension, dict):
            raise ReferenceFreeContractError("Each rubric dimension must be an object")

        name = str(raw_dimension.get("name", "")).strip()
        description = str(raw_dimension.get("description", "")).strip()
        min_score = raw_dimension.get("min_score", 0.0)
        max_score = raw_dimension.get("max_score", 1.0)

        if not name or not description:
            raise ReferenceFreeContractError(
                "Each rubric dimension requires non-empty name and description"
            )

        if not isinstance(min_score, (int, float)) or not isinstance(max_score, (int, float)):
            raise ReferenceFreeContractError(
                f"Rubric dimension '{name}' score bounds must be numeric"
            )

        if float(max_score) <= float(min_score):
            raise ReferenceFreeContractError(
                f"Rubric dimension '{name}' max_score must be greater than min_score"
            )

        dimensions.append(
            JudgeRubricDimension(
                name=name,
                description=description,
                min_score=float(min_score),
                max_score=float(max_score),
            )
        )

    if not rubric_id or not version:
        raise ReferenceFreeContractError("Rubric requires non-empty rubric_id and version")

    if pairwise_dimension not in {dimension.name for dimension in dimensions}:
        raise ReferenceFreeContractError(
            f"pairwise_dimension '{pairwise_dimension}' is not defined in dimensions"
        )

    return JudgeRubric(
        rubric_id=rubric_id,
        version=version,
        dimensions=dimensions,
        pairwise_dimension=pairwise_dimension,
    )


def clamp_score(value: float, *, min_score: float = 0.0, max_score: float = 1.0) -> float:
    """Clamp a floating score into inclusive range and round for report stability.

    Args:
        value: Raw score.
        min_score: Inclusive lower bound.
        max_score: Inclusive upper bound.

    Returns:
        Clamped score rounded to 4 decimals.
    """
    return round(max(min_score, min(max_score, value)), 4)


def normalize_numeric(value: Any) -> float | None:
    """Convert numeric-like values into float, returning ``None`` when invalid.

    Args:
        value: Input to normalize.

    Returns:
        Parsed float or ``None``.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None
