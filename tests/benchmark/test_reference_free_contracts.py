from __future__ import annotations

from pathlib import Path

import pytest

from card_framework.benchmark.reference_free.contracts import (
    ReferenceFreeContractError,
    load_judge_rubric,
)
from card_framework.shared.paths import DEFAULT_JUDGE_RUBRIC_PATH


def test_load_judge_rubric_valid_contract() -> None:
    rubric = load_judge_rubric(DEFAULT_JUDGE_RUBRIC_PATH)

    assert rubric.rubric_id == "summarization_reference_free_v1"
    assert rubric.pairwise_dimension == "overall"
    assert {dimension.name for dimension in rubric.dimensions} == {
        "factuality",
        "relevance",
        "coherence",
        "overall",
    }


def test_load_judge_rubric_invalid_contract(tmp_path: Path) -> None:
    bad_path = tmp_path / "bad_rubric.json"
    bad_path.write_text('{"rubric_id": "x", "version": "1", "dimensions": []}', encoding="utf-8")

    with pytest.raises(ReferenceFreeContractError):
        load_judge_rubric(bad_path)

